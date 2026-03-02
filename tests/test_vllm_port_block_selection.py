from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

from active_mt_distill.pipeline.vllm_runtime import (
    VLLMGenerationContext as ActiveVLLMGenerationContext,
)
from active_mt_distill.pipeline.vllm_runtime import render_vllm_generation_lines as render_active_lines


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline_vllm = _load_module("test_pipeline_vllm_module", "pipelines/pipeline_vllm.py")


def _find_free_block(block: int = 6) -> int:
    for base in range(25000, 40000):
        sockets: list[socket.socket] = []
        ok = True
        try:
            for offset in range(block):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", base + offset))
                sockets.append(sock)
        except OSError:
            ok = False
        finally:
            for sock in sockets:
                sock.close()
        if ok:
            return base
    raise RuntimeError("Could not find a free contiguous port block for test.")


def _extract_port_selection_python(lines: list[str]) -> str:
    start_idx = next(i for i, line in enumerate(lines) if line.startswith("export VLLM_PORT=$("))
    body: list[str] = []
    for line in lines[start_idx + 1 :]:
        if line == "PY":
            break
        body.append(line)
    return "\n".join(body)


def _run_port_selector(script: str, *, start: int, scan: int, block: int) -> int:
    env = os.environ.copy()
    env["VLLM_PORT"] = str(start)
    env["VLLM_PORT_SCAN"] = str(scan)
    env["VLLM_PORT_BLOCK_SIZE"] = str(block)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return int(result.stdout.strip())


def test_pipeline_vllm_port_selector_skips_ports_with_busy_neighbors(tmp_path: Path) -> None:
    ctx = pipeline_vllm.VLLMGenerationContext(
        root=tmp_path,
        run_dir=tmp_path / "run",
        pipeline_log=tmp_path / "pipeline.log",
        server_host="127.0.0.1",
        server_port=8000,
        slurm_port_from_jobid=True,
        vllm_runtime_config_path=tmp_path / "runtime.yaml",
        vllm_runner_script_path=tmp_path / "runner.py",
        vllm_runner_python=sys.executable,
        vllm_model_config_path=tmp_path / "model.yaml",
        vllm_generated_command_path=tmp_path / "vllm_generated_command.sh",
        vllm_log_file=tmp_path / "vllm_server.log",
        vllm_port_meta_file=tmp_path / "vllm_port.txt",
        generation_cmd="echo generate",
    )
    script = _extract_port_selection_python(pipeline_vllm.render_vllm_generation_lines(ctx))

    base = _find_free_block()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("0.0.0.0", base + 1))
    blocker.listen(1)
    try:
        assert _run_port_selector(script, start=base, scan=6, block=3) == base + 2
    finally:
        blocker.close()


def test_active_mt_vllm_port_selector_skips_ports_with_busy_neighbors(tmp_path: Path) -> None:
    ctx = ActiveVLLMGenerationContext(
        root=tmp_path,
        run_dir=tmp_path / "run",
        pipeline_log=tmp_path / "pipeline.log",
        server_host="127.0.0.1",
        server_port=8000,
        slurm_port_from_jobid=True,
        vllm_runtime_config_path=tmp_path / "runtime.yaml",
        vllm_runner_script_path=tmp_path / "runner.py",
        vllm_runner_python=sys.executable,
        vllm_model_config_path=tmp_path / "model.yaml",
        vllm_generated_command_path=tmp_path / "vllm_generated_command.sh",
        vllm_log_file=tmp_path / "vllm_server.log",
        vllm_port_meta_file=tmp_path / "vllm_port.txt",
        generation_cmd="echo generate",
    )
    script = _extract_port_selection_python(render_active_lines(ctx))

    base = _find_free_block()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("0.0.0.0", base + 1))
    blocker.listen(1)
    try:
        assert _run_port_selector(script, start=base, scan=6, block=3) == base + 2
    finally:
        blocker.close()
