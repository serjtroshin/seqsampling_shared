from __future__ import annotations

import argparse
import logging
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


FATAL_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\btraceback\b",
        r"\bruntimeerror\b",
        r"\bvalueerror\b",
        r"\bassertionerror\b",
        r"\bcuda out of memory\b",
        r"\bout of memory\b",
        r"\bslurmstepd: error\b",
        r"\berror:\b",
    ]
]


@dataclass
class RunnerSlurmConfig:
    partition: str = "cpu"
    time: str = "2-00:00:00"
    cpus_per_task: int = 2
    mem: str = "16G"
    exclude: str | None = None
    additional_args: list[str] = field(default_factory=list)


@dataclass
class RunnerConfig:
    max_running: int = 2
    poll_seconds: int = 30
    python_bin: str = ".venv/bin/python"
    log_root: str = "outputs/mt/sweeps"
    slurm: RunnerSlurmConfig = field(default_factory=RunnerSlurmConfig)


@dataclass
class SweepConfig:
    name: str = "mt_sweep"
    base_config: str = "pipelines/configs/wmt24p.yaml"
    experiment_dir: str = "outputs/mt/pipeline_runs"
    global_overrides: list[str] = field(default_factory=list)
    sweeps: list[list[str]] = field(default_factory=list)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


@dataclass
class SweepTask:
    task_id: str
    run_name: str
    overrides: list[str]
    run_dir: Path
    command: list[str] = field(default_factory=list)
    gen_job_id: str | None = None
    eval_job_id: str | None = None
    state: str = "PENDING"  # PENDING | ACTIVE | DONE | FAILED
    detail: str = ""


@dataclass
class SlurmJobInfo:
    state: str
    reason: str


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_sweep_config(path: Path) -> SweepConfig:
    base = OmegaConf.structured(SweepConfig)
    user = OmegaConf.load(path)
    merged = OmegaConf.merge(base, user)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _setup_logger(work_dir: Path) -> logging.Logger:
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "sweep.log"

    logger = logging.getLogger("standalone_mt_sweep_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _extract_override(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for item in reversed(overrides):
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _set_override_if_missing(overrides: list[str], key: str, value: str) -> list[str]:
    if _extract_override(overrides, key) is None:
        return [*overrides, f"{key}={value}"]
    return overrides


def _resolve_python_bin(root: Path, configured: str) -> str:
    candidate = Path(configured)
    if not candidate.is_absolute():
        # Keep the venv symlink path (do not dereference) so Python keeps
        # the virtualenv context and installed packages.
        candidate = (root / candidate).absolute()
    if candidate.exists():
        return str(candidate)
    return configured


def _infer_language_folder(overrides: list[str]) -> str:
    tgt = _extract_override(overrides, "tgt")
    if tgt:
        return f"en-{tgt}"

    prompts_path = _extract_override(overrides, "prompts_path")
    if prompts_path:
        stem = Path(prompts_path).stem
        match = re.search(r"(en-[A-Za-z]{2,3}(?:_[A-Za-z]{2,3})?)", stem)
        if match:
            return match.group(1)
    return "unknown-lp"


def _infer_scenario_folder(overrides: list[str]) -> str:
    scenario = _extract_override(overrides, "scenario")
    if not scenario:
        return "unknown-scenario"
    return Path(scenario).stem or "unknown-scenario"


def _infer_model_folder(overrides: list[str]) -> str:
    model_config = _extract_override(overrides, "model_config")
    if model_config:
        return Path(model_config).stem or "unknown-model"

    model = _extract_override(overrides, "model")
    if model:
        return _slugify(model.replace("/", "-")) or "unknown-model"
    return "unknown-model"


def _default_output_root(root: Path, experiment_dir: Path, overrides: list[str]) -> str:
    run_base = (
        experiment_dir
        / _infer_language_folder(overrides)
        / _infer_scenario_folder(overrides)
        / _infer_model_folder(overrides)
    )
    try:
        return str(run_base.relative_to(root))
    except ValueError:
        return str(run_base)


def _require_model_override(task_id: str, overrides: list[str]) -> None:
    model_config = (_extract_override(overrides, "model_config") or "").strip()
    model = (_extract_override(overrides, "model") or "").strip()
    if model_config or model:
        return
    raise ValueError(
        f"Sweep task {task_id} is missing model selection. "
        "Provide either 'model_config=...' or 'model=...' in global_overrides, "
        "sweeps, or CLI --override."
    )


def _build_tasks(
    cfg: SweepConfig, root: Path, experiment_dir: Path, cli_overrides: list[str]
) -> list[SweepTask]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_slug = _slugify(cfg.name) or "sweep"
    tasks: list[SweepTask] = []

    for idx, sweep_overrides in enumerate(cfg.sweeps):
        task_id = f"t{idx:03d}"
        overrides = [*cfg.global_overrides, *sweep_overrides, *cli_overrides]
        _require_model_override(task_id, overrides)
        overrides = _set_override_if_missing(overrides, "submit_jobs", "true")
        overrides = _set_override_if_missing(
            overrides,
            "output_root",
            _default_output_root(root, experiment_dir, overrides),
        )

        run_name = _extract_override(overrides, "run_name")
        if not run_name:
            run_name = f"{stamp}_{sweep_slug}_{idx:03d}"
            overrides.append(f"run_name={run_name}")

        output_root_override = _extract_override(overrides, "output_root")
        run_base = _resolve_path(root, output_root_override) if output_root_override else experiment_dir

        tasks.append(
            SweepTask(
                task_id=task_id,
                run_name=run_name,
                overrides=overrides,
                run_dir=(run_base / run_name).resolve(),
            )
        )
    return tasks


def _write_run_dirs_index(work_dir: Path, tasks: list[SweepTask]) -> Path:
    """Write one experiment run directory path per line for quick navigation."""
    index_path = work_dir / "experiment_run_dirs.txt"
    lines = [str(task.run_dir) for task in tasks]
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    index_path.write_text(payload, encoding="utf-8")
    return index_path


def _extract_job_id(text: str, label: str) -> str | None:
    match = re.search(rf"Submitted {label} job:\s*(\d+)", text)
    return match.group(1) if match else None


def _submit_task(
    task: SweepTask,
    *,
    root: Path,
    python_bin: str,
    base_config: Path,
    logger: logging.Logger,
) -> tuple[bool, str]:
    cmd = [python_bin, "pipelines/pipeline.py", "--config", str(base_config), *task.overrides]
    task.command = cmd

    logger.info("[submit] %s run=%s", task.task_id, task.run_name)
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode != 0:
        return False, output or "pipeline submit failed"

    task.gen_job_id = _extract_job_id(output, "generation")
    task.eval_job_id = _extract_job_id(output, "evaluation")
    task.state = "ACTIVE"
    detail = f"gen_job={task.gen_job_id or '?'} eval_job={task.eval_job_id or '?'}"
    return True, detail


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _contains_fatal_error(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = _read_text(path)
    return any(pattern.search(text) for pattern in FATAL_PATTERNS)


def _eval_done(run_dir: Path) -> bool:
    out_path = run_dir / "slurm_eval.out"
    if not out_path.exists():
        return False
    return "[eval] done" in _read_text(out_path).lower()


def _query_slurm_jobs(
    root: Path, job_ids: set[str], logger: logging.Logger
) -> dict[str, SlurmJobInfo] | None:
    if not job_ids:
        return {}
    cmd = ["squeue", "-h", "-o", "%i|%t|%R", "-j", ",".join(sorted(job_ids))]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("squeue failed; will keep tasks active until done/error files appear.")
        return None
    jobs: dict[str, SlurmJobInfo] = {}
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id, state, reason = parts
        jobs[job_id.strip()] = SlurmJobInfo(state=state.strip(), reason=reason.strip())
    return jobs


def _cancel_jobs(root: Path, job_ids: set[str], logger: logging.Logger) -> None:
    if not job_ids:
        return
    cmd = ["scancel", *sorted(job_ids)]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "scancel failed").strip()
        logger.warning("[cancel-failed] jobs=%s detail=%s", ",".join(sorted(job_ids)), detail)
        return
    logger.info("[cancel] jobs=%s", ",".join(sorted(job_ids)))


def _is_dependency_never_satisfied(job: SlurmJobInfo | None) -> bool:
    if job is None:
        return False
    if job.state != "PD":
        return False
    reason = job.reason.lower().strip()
    reason = reason.strip("()")
    return "dependencyneversatisfied" in reason


def _task_status(
    task: SweepTask, job_map: dict[str, SlurmJobInfo] | None
) -> tuple[str, str, set[str]]:
    known_job_ids = {job_id for job_id in [task.gen_job_id, task.eval_job_id] if job_id}

    if _eval_done(task.run_dir):
        return "DONE", "evaluation finished", set()

    gen_err = task.run_dir / "slurm_gen.err"
    eval_err = task.run_dir / "slurm_eval.err"
    if _contains_fatal_error(gen_err) or _contains_fatal_error(eval_err):
        cancel_ids = set()
        if job_map is not None:
            cancel_ids = {job_id for job_id in known_job_ids if job_id in job_map}
        return "FAILED", "fatal error found in slurm stderr", cancel_ids

    if job_map is not None and _is_dependency_never_satisfied(
        job_map.get(task.eval_job_id or "")
    ):
        cancel_ids = {job_id for job_id in known_job_ids if job_id in job_map}
        return "FAILED", "evaluation dependency never satisfied", cancel_ids

    if known_job_ids:
        if job_map is None:
            return "ACTIVE", "squeue unavailable; waiting for done/error logs", set()
        if any(job_id in job_map for job_id in known_job_ids):
            return "ACTIVE", "slurm job still active", set()
        return "FAILED", "slurm jobs no longer active and eval did not finish", set()

    return "ACTIVE", "waiting for logs", set()


def _build_reentry_args(
    *,
    config_path: Path,
    work_dir: Path,
    max_running: int | None,
    poll_seconds: int | None,
    overrides: list[str],
) -> list[str]:
    args = ["--config", str(config_path), "--work-dir", str(work_dir)]
    if max_running is not None:
        args.extend(["--max-running", str(max_running)])
    if poll_seconds is not None:
        args.extend(["--poll-seconds", str(poll_seconds)])
    for override in overrides:
        args.extend(["--override", override])
    return args


def _render_master_job_script(
    *,
    root: Path,
    python_bin: str,
    slurm: RunnerSlurmConfig,
    work_dir: Path,
    job_name: str,
    module_args: list[str],
) -> str:
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --time={slurm.time}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --output={work_dir}/slurm_runner.out",
        f"#SBATCH --error={work_dir}/slurm_runner.err",
    ]
    if slurm.exclude:
        header.append(f"#SBATCH --exclude={slurm.exclude}")
    if slurm.additional_args:
        header.extend(slurm.additional_args)

    cmd = [python_bin, str(Path(__file__).resolve()), *module_args]
    cmd_line = " \\\n  ".join(shlex.quote(item) for item in cmd)
    body = [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(root))}",
        'HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"',
        'if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then',
        '  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"',
        "fi",
        'if [ -z "${HF_TOKEN:-}" ]; then',
        '  echo "[runner] WARNING: HF_TOKEN is not set; gated Hugging Face models may fail." >&2',
        "fi",
        cmd_line,
        "",
    ]
    return "\n".join(header + body)


def _submit_master_job(
    *,
    root: Path,
    cfg: SweepConfig,
    python_bin: str,
    work_dir: Path,
    module_args: list[str],
    logger: logging.Logger,
) -> None:
    script_text = _render_master_job_script(
        root=root,
        python_bin=python_bin,
        slurm=cfg.runner.slurm,
        work_dir=work_dir,
        job_name=f"mt-sweep-{_slugify(cfg.name) or 'runner'}",
        module_args=module_args,
    )
    script_path = work_dir / "runner_job.sh"
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(0o755)

    result = subprocess.run(["sbatch", str(script_path)], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "sbatch failed").strip()
        raise RuntimeError(detail)
    logger.info("[submit] master runner job submitted: %s", (result.stdout or "").strip())
    logger.info("[submit] runner script: %s", script_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal FIFO sweep runner for standalone MT pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Sweep config YAML path (relative to standalone root or absolute).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional work directory for sweep logs.",
    )
    parser.add_argument(
        "--max-running",
        type=int,
        default=None,
        help="Override max concurrent active tasks.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=None,
        help="Override polling interval.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit this sweep runner itself as a CPU SLURM master job.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Extra pipeline override(s) applied to every task (repeatable), "
            "for example --override model_config=configs/model_configs/qwen3-32b-instruct.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = _resolve_path(root, args.config)
    cfg = _load_sweep_config(config_path)

    max_running = args.max_running if args.max_running is not None else cfg.runner.max_running
    poll_seconds = args.poll_seconds if args.poll_seconds is not None else cfg.runner.poll_seconds
    python_bin = _resolve_python_bin(root, cfg.runner.python_bin)
    base_config = _resolve_path(root, cfg.base_config)
    experiment_dir = _resolve_path(root, cfg.experiment_dir)

    if args.work_dir is not None:
        work_dir = _resolve_path(root, args.work_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        work_dir = _resolve_path(root, Path(cfg.runner.log_root) / f"{_slugify(cfg.name)}_{ts}")
    logger = _setup_logger(work_dir)

    if args.submit:
        module_args = _build_reentry_args(
            config_path=config_path,
            work_dir=work_dir,
            max_running=args.max_running,
            poll_seconds=args.poll_seconds,
            overrides=list(args.override or []),
        )
        _submit_master_job(
            root=root,
            cfg=cfg,
            python_bin=python_bin,
            work_dir=work_dir,
            module_args=module_args,
            logger=logger,
        )
        return

    tasks = _build_tasks(cfg, root, experiment_dir, list(args.override or []))
    run_dirs_index = _write_run_dirs_index(work_dir, tasks)
    logger.info("Experiment run dirs index: %s", run_dirs_index)
    if not tasks:
        logger.info("No tasks configured; exiting.")
        return

    pending: deque[SweepTask] = deque(tasks)
    active: list[SweepTask] = []
    done: list[SweepTask] = []
    failed: list[SweepTask] = []
    loop_idx = 0

    logger.info(
        "Sweep started | name=%s tasks=%d max_running=%d poll=%ss",
        cfg.name,
        len(tasks),
        max_running,
        poll_seconds,
    )

    while pending or active:
        job_ids = {job_id for t in active for job_id in [t.gen_job_id, t.eval_job_id] if job_id}
        job_map = _query_slurm_jobs(root, job_ids, logger=logger)

        still_active: list[SweepTask] = []
        for task in active:
            status, detail, cancel_ids = _task_status(task, job_map)
            task.detail = detail
            if status == "DONE":
                task.state = "DONE"
                done.append(task)
                logger.info("[done] %s run=%s", task.task_id, task.run_name)
            elif status == "FAILED":
                task.state = "FAILED"
                failed.append(task)
                _cancel_jobs(root, cancel_ids, logger=logger)
                logger.error("[failed] %s run=%s detail=%s", task.task_id, task.run_name, detail)
            else:
                still_active.append(task)
        active = still_active

        while pending and len(active) < max_running:
            task = pending.popleft()
            ok, detail = _submit_task(
                task,
                root=root,
                python_bin=python_bin,
                base_config=base_config,
                logger=logger,
            )
            task.detail = detail
            if ok:
                logger.info("[queued] %s %s", task.task_id, detail)
                active.append(task)
            else:
                task.state = "FAILED"
                failed.append(task)
                logger.error("[submit-failed] %s %s", task.task_id, detail)

        logger.info(
            "[loop %d] pending=%d active=%d done=%d failed=%d",
            loop_idx,
            len(pending),
            len(active),
            len(done),
            len(failed),
        )

        if pending or active:
            time.sleep(poll_seconds)
        loop_idx += 1

    logger.info("Sweep finished | done=%d failed=%d total=%d", len(done), len(failed), len(tasks))


if __name__ == "__main__":
    main()
