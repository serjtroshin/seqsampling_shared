from __future__ import annotations

import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class VLLMRuntimeConfig:
    """Runtime launch configuration for vLLM server startup."""

    runtime_config: str = "configs/vllm_config/local_python.yaml"
    runner_script: str = "configs/vllm_config/vllm_server_runner.py"
    runner_python: str = ".venv/bin/python"


@dataclass
class VLLMGenerationContext:
    """Inputs required to render vLLM startup + generation shell lines."""

    root: Path
    run_dir: Path
    pipeline_log: Path
    server_host: str
    server_port: int
    slurm_port_from_jobid: bool
    vllm_runtime_config_path: Path
    vllm_runner_script_path: Path
    vllm_runner_python: str
    vllm_model_config_path: Path
    vllm_generated_command_path: Path
    vllm_log_file: Path
    vllm_port_meta_file: Path
    generation_cmd: str


def write_vllm_runtime_model_config(
    *, path: Path, model: str, vllm_server_args: list[str]
) -> Path:
    """Write a temporary model config consumed by vLLM server runner."""
    payload = {
        "model": model,
        "vllm_server_args": [str(arg) for arg in vllm_server_args],
    }
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))
    return path


def render_vllm_generation_lines(ctx: VLLMGenerationContext) -> list[str]:
    """Render shell lines that lock, launch, monitor, and use a vLLM server."""

    runner_python = ctx.vllm_runner_python
    if "/" in runner_python and not Path(runner_python).is_absolute():
        runner_python = str((ctx.root / runner_python).resolve())
    runner_python_quoted = shlex.quote(runner_python)

    if ctx.slurm_port_from_jobid:
        port_default_setup = textwrap.dedent(
            """
            if [ -z "${VLLM_PORT:-}" ]; then
              if [ -n "${SLURM_JOB_ID:-}" ]; then
                export VLLM_PORT=$((8000 + SLURM_JOB_ID % 500))
              else
                export VLLM_PORT=8000
              fi
            fi
            """
        ).strip()
    else:
        port_default_setup = f'export VLLM_PORT="${{VLLM_PORT:-{ctx.server_port}}}"'

    port_select_under_lock = textwrap.dedent(
        f"""
        export VLLM_PORT=$({runner_python_quoted} - <<'PY'
import os, socket
start = int(os.getenv("VLLM_PORT", "8000"))
scan = int(os.getenv("VLLM_PORT_SCAN", "100"))
chosen = None
for p in range(start, start + scan):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", p))
        except OSError:
            continue
        chosen = p
        break
if chosen is None:
    raise SystemExit("No free port found in range [%d, %d)" % (start, start + scan))
print(chosen)
PY
)
        """
    ).strip()

    detect_actual_port = textwrap.dedent(
        f"""
        ACTUAL_PORT=$({runner_python_quoted} - <<'PY'
import os, pathlib, re, sys, time
log = pathlib.Path({repr(str(ctx.vllm_log_file))})
vllm_pid = int(os.getenv('VLLM_PID', '0') or '0')
detect_secs = int(os.getenv('VLLM_PORT_DETECT_SECS', '3600'))
patterns = [
    re.compile(r'Starting vLLM API server .*:(\\d+)'),
    re.compile(r'Uvicorn running on .*:(\\d+)'),
]
deadline = time.time() + detect_secs
while time.time() < deadline:
    if log.exists():
        text = log.read_text(errors='ignore')
        matches = []
        for pat in patterns:
            matches.extend(m.group(1) for m in pat.finditer(text))
        if matches:
            print(matches[-1])
            sys.exit(0)
    if vllm_pid > 0:
        try:
            os.kill(vllm_pid, 0)
        except ProcessLookupError:
            print('vLLM process exited before startup port was detected', file=sys.stderr)
            sys.exit(2)
        except PermissionError:
            pass
    time.sleep(1)
print(f'Could not detect vLLM port within {{detect_secs}}s', file=sys.stderr)
sys.exit(1)
PY
)
        """
    ).strip()

    wait_server = textwrap.dedent(
        f"""
        export VLLM_WAIT_SECS=${{VLLM_WAIT_SECS:-1200}}
        export VLLM_PID
        {runner_python_quoted} - <<'PY'
import json, os, sys, time, urllib.error, urllib.request
host = "{ctx.server_host}"
port = int(os.getenv("VLLM_PORT", "{ctx.server_port}"))
wait_secs = int(os.getenv("VLLM_WAIT_SECS", "1200"))
vllm_pid = int(os.getenv("VLLM_PID", "0") or "0")
deadline = time.time() + wait_secs
models_url = f"http://{{host}}:{{port}}/v1/models"
last_status = ""
while time.time() < deadline:
    if vllm_pid > 0:
        try:
            os.kill(vllm_pid, 0)
        except ProcessLookupError:
            print("vLLM process exited before readiness check passed", file=sys.stderr)
            sys.exit(2)
        except PermissionError:
            pass
    try:
        with urllib.request.urlopen(models_url, timeout=2.0) as resp:
            payload = json.load(resp)
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            sys.exit(0)
        last_status = f"unexpected payload: {{payload}}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        last_status = str(exc)
    time.sleep(1)
msg = f"vLLM not ready after {{wait_secs}}s on {{host}}:{{port}}"
if last_status:
    msg += f"; last check: {{last_status}}"
print(msg, file=sys.stderr)
sys.exit(1)
PY
        """
    ).strip()

    # Keep existing behavior (runner emits a shell command string).
    vllm_cmd_build = (
        f"VLLM_CMD=$({runner_python_quoted}"
        f" {shlex.quote(str(ctx.vllm_runner_script_path))}"
        f" --model-config {shlex.quote(str(ctx.vllm_model_config_path))}"
        f" --vllm-config {shlex.quote(str(ctx.vllm_runtime_config_path))}"
        f" --host {shlex.quote(ctx.server_host)}"
        ' --port "$VLLM_PORT")'
    )

    return [
        "LOCK_NODE=\"${VLLM_LOCK_NODE:-${SLURMD_NODENAME:-$(hostname -s 2>/dev/null || hostname)}}\"",
        "LOCK_NODE=\"${LOCK_NODE//[^A-Za-z0-9._-]/_}\"",
        "LOCK_FILE=\"${VLLM_PORT_LOCK_FILE:-/tmp/vllm_port_lock.${LOCK_NODE}}\"",
        "LOCK_WAIT_SECS=\"${VLLM_LOCK_WAIT_SECS:-4000}\"",
        "LOCK_STALE_SECS=\"${VLLM_LOCK_STALE_SECS:-21600}\"",
        "LOCK_HELD=0",
        "release_startup_lock() {",
        "  if [ \"${LOCK_HELD:-0}\" -eq 1 ]; then",
        "    flock -u 9 || true",
        "    exec 9>&-",
        "    LOCK_HELD=0",
        "  fi",
        "}",
        "lock_mtime_epoch() {",
        "  local f=\"$1\"",
        "  local mtime=\"\"",
        "  mtime=$(stat -c %Y \"$f\" 2>/dev/null || true)",
        "  if [ -z \"$mtime\" ]; then",
        "    mtime=$(stat -f %m \"$f\" 2>/dev/null || true)",
        "  fi",
        "  echo \"$mtime\"",
        "}",
        "if [ -f \"$LOCK_FILE\" ]; then",
        "  now_epoch=$(date +%s)",
        "  lock_epoch=$(lock_mtime_epoch \"$LOCK_FILE\")",
        "  if [ -n \"$lock_epoch\" ]; then",
        "    lock_age=$((now_epoch - lock_epoch))",
        "    if [ \"$lock_age\" -gt \"$LOCK_STALE_SECS\" ]; then",
        "      exec 8>\"$LOCK_FILE\"",
        "      if flock -n 8; then",
        "        glog \"stale lock file detected (age=${lock_age}s > ${LOCK_STALE_SECS}s); recreating $LOCK_FILE\"",
        "        rm -f \"$LOCK_FILE\"",
        "      else",
        "        glog \"lock file age=${lock_age}s but lock is currently held; keeping $LOCK_FILE\"",
        "      fi",
        "      exec 8>&-",
        "    fi",
        "  fi",
        "fi",
        "if ! command -v flock >/dev/null 2>&1; then",
        "  glog \"ERROR: flock is required for synchronized vLLM startup\"",
        "  exit 1",
        "fi",
        "glog \"acquiring vLLM startup lock: $LOCK_FILE (timeout=${LOCK_WAIT_SECS}s)\"",
        "exec 9>\"$LOCK_FILE\"",
        "if ! flock -w \"$LOCK_WAIT_SECS\" 9; then",
        "  glog \"ERROR: failed to acquire startup lock: $LOCK_FILE\"",
        "  exit 1",
        "fi",
        "LOCK_HELD=1",
        "glog \"lock acquired\"",
        *port_default_setup.splitlines(),
        "glog \"selecting vLLM port from base=$VLLM_PORT (scan=${VLLM_PORT_SCAN:-100})\"",
        *port_select_under_lock.splitlines(),
        "glog \"selected requested VLLM_PORT=$VLLM_PORT\"",
        f"cat > {shlex.quote(str(ctx.vllm_port_meta_file))} <<EOF",
        "requested_port=$VLLM_PORT",
        "lock_file=$LOCK_FILE",
        "lock_acquired_at=$(date)",
        "EOF",
        'HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"',
        'if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then',
        '  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"',
        "  glog \"HF_TOKEN loaded from ~/.cache/huggingface/token\"",
        "fi",
        'if [ -z "${HF_TOKEN:-}" ]; then',
        "  glog \"WARNING: HF_TOKEN is not set; gated Hugging Face models may fail\"",
        "fi",
        vllm_cmd_build,
        f"printf '%s\\n' \"$VLLM_CMD\" > {shlex.quote(str(ctx.vllm_generated_command_path))}",
        "glog \"generated vLLM command: $VLLM_CMD\"",
        f"bash -lc \"$VLLM_CMD\" > {shlex.quote(str(ctx.vllm_log_file))} 2>&1 &",
        "VLLM_PID=$!",
        f"glog \"vLLM pid=$VLLM_PID (log: {ctx.vllm_log_file})\"",
        "trap 'release_startup_lock; "
        "if [ -n \"${VLLM_PID:-}\" ]; then kill \"$VLLM_PID\" || true; fi' EXIT",
        "glog \"detecting vLLM startup port from log\"",
        "export VLLM_PID",
        "set +e",
        *detect_actual_port.splitlines(),
        "detect_status=$?",
        "set -e",
        "if [ \"$detect_status\" -ne 0 ]; then",
        "  glog \"ERROR: failed to detect vLLM startup port (status=$detect_status)\"",
        f"  if [ -f {shlex.quote(str(ctx.vllm_log_file))} ]; then",
        "    glog \"tail of vllm_server.log:\"",
        f"    tail -n 80 {shlex.quote(str(ctx.vllm_log_file))} | tee -a \"$PIPELINE_LOG\"",
        "  fi",
        "  exit \"$detect_status\"",
        "fi",
        "if [ \"$ACTUAL_PORT\" != \"$VLLM_PORT\" ]; then",
        "  glog \"WARNING: requested port $VLLM_PORT but vLLM reported $ACTUAL_PORT\"",
        "fi",
        "export VLLM_PORT=$ACTUAL_PORT",
        f"cat >> {shlex.quote(str(ctx.vllm_port_meta_file))} <<EOF",
        "actual_port=$VLLM_PORT",
        "lock_released_at=$(date)",
        "EOF",
        "release_startup_lock",
        "glog \"lock released; using VLLM_PORT=$VLLM_PORT\"",
        "glog \"waiting for vLLM readiness\"",
        "set +e",
        *wait_server.splitlines(),
        "wait_status=$?",
        "set -e",
        "if [ \"$wait_status\" -ne 0 ]; then",
        "  glog \"ERROR: vLLM readiness check failed (status=$wait_status)\"",
        f"  if [ -f {shlex.quote(str(ctx.vllm_log_file))} ]; then",
        "    glog \"tail of vllm_server.log:\"",
        f"    tail -n 80 {shlex.quote(str(ctx.vllm_log_file))} | tee -a \"$PIPELINE_LOG\"",
        "  fi",
        "  exit \"$wait_status\"",
        "fi",
        "glog \"vLLM ready; running generation\"",
        ctx.generation_cmd,
        "glog \"generation done\"",
    ]
