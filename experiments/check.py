from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueueJob:
    job_id: str
    state: str
    name: str
    elapsed: str
    reason: str


@dataclass
class SweepWorkdirStatus:
    work_dir: Path
    job_id: str | None
    loop_summary: str
    finished: bool


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _run_squeue() -> list[QueueJob]:
    cmd = ["squeue", "-h", "-o", "%i|%t|%j|%M|%R"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "squeue failed").strip()
        raise RuntimeError(detail)

    jobs: list[QueueJob] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        jobs.append(
            QueueJob(
                job_id=parts[0].strip(),
                state=parts[1].strip(),
                name=parts[2].strip(),
                elapsed=parts[3].strip(),
                reason=parts[4].strip(),
            )
        )
    return jobs


def _job_sort_key(job: QueueJob) -> tuple[int, str]:
    try:
        return (0, f"{int(job.job_id):020d}")
    except ValueError:
        return (1, job.job_id)


def _find_sweep_workdirs(root: Path) -> dict[str, SweepWorkdirStatus]:
    sweep_root = root / "outputs" / "mt" / "sweeps"
    statuses: dict[str, SweepWorkdirStatus] = {}
    if not sweep_root.exists():
        return statuses

    for log_path in sweep_root.glob("*/*/sweep.log"):
        work_dir = log_path.parent
        text = _read_text(log_path)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue

        job_id: str | None = None
        for line in lines:
            match = re.search(r"Submitted batch job (\d+)", line)
            if match:
                job_id = match.group(1)
                break

        loop_summary = "no loop summary"
        finished = False
        for line in reversed(lines):
            if "Sweep finished" in line:
                loop_summary = line
                finished = True
                break
            if "[loop " in line:
                loop_summary = line
                break

        if job_id is not None:
            statuses[job_id] = SweepWorkdirStatus(
                work_dir=work_dir,
                job_id=job_id,
                loop_summary=loop_summary,
                finished=finished,
            )
    return statuses


def _infer_run_status(run_dir: Path) -> tuple[str, str]:
    eval_out = run_dir / "slurm_eval.out"
    eval_err = run_dir / "slurm_eval.err"
    gen_out = run_dir / "slurm_gen.out"
    gen_err = run_dir / "slurm_gen.err"

    if eval_out.exists() and "[eval] done" in _read_text(eval_out).lower():
        return "DONE", "evaluation finished"

    for path in [gen_err, eval_err, gen_out]:
        if not path.exists():
            continue
        text = _read_text(path)
        if any(pattern.search(text) for pattern in FATAL_PATTERNS):
            return "FAILED", f"fatal pattern in {path.name}"

    if any(path.exists() for path in [gen_out, gen_err, eval_out, eval_err]):
        return "ACTIVE", "logs exist, no completion marker yet"

    return "PENDING", "no logs yet"


def _print_queue_jobs(label: str, jobs: list[QueueJob]) -> None:
    print(label)
    if not jobs:
        print("  (none)")
        return
    for job in sorted(jobs, key=_job_sort_key):
        print(
            f"  {job.job_id}  {job.state:>2}  {job.name}  "
            f"elapsed={job.elapsed} reason={job.reason}"
        )


def _run_name_from_job(job_name: str) -> str:
    for prefix in ["parseq-gen-", "parseq-eval-"]:
        if job_name.startswith(prefix):
            return job_name[len(prefix) :]
    return job_name


def _print_experiment_job_summary(experiment_jobs: list[QueueJob]) -> None:
    print("Experiment Jobs By Run")
    if not experiment_jobs:
        print("  (none)")
        return

    grouped: dict[str, dict[str, QueueJob | None]] = {}
    for job in experiment_jobs:
        run_name = _run_name_from_job(job.name)
        entry = grouped.setdefault(run_name, {"gen": None, "eval": None})
        if job.name.startswith("parseq-gen-"):
            entry["gen"] = job
        elif job.name.startswith("parseq-eval-"):
            entry["eval"] = job

    for run_name in sorted(grouped):
        entry = grouped[run_name]
        gen = entry["gen"]
        evl = entry["eval"]

        gen_state = gen.state if gen else "-"
        eval_state = evl.state if evl else "-"
        status = "ACTIVE"
        if evl and evl.state == "PD" and "DependencyNeverSatisfied" in evl.reason:
            status = "BLOCKED"
        elif gen and gen.state == "PD":
            status = "QUEUED"
        elif gen and gen.state in {"CG"}:
            status = "FINISHING"

        detail = ""
        if evl and "DependencyNeverSatisfied" in evl.reason:
            detail = "eval dependency never satisfied"
        elif evl:
            detail = f"eval_reason={evl.reason}"
        elif gen:
            detail = f"gen_reason={gen.reason}"

        print(f"  {status:8} run={run_name} gen={gen_state} eval={eval_state} {detail}".rstrip())


def _print_sweep_workdir_summary(
    active_sweeps: list[QueueJob], sweep_workdirs: dict[str, SweepWorkdirStatus]
) -> None:
    print("Active Sweep Workdirs")
    if not active_sweeps:
        print("  (none)")
        return

    for sweep in sorted(active_sweeps, key=_job_sort_key):
        info = sweep_workdirs.get(sweep.job_id)
        if info is None:
            print(f"  {sweep.job_id}  {sweep.name}  work_dir=<unknown>")
            continue
        print(f"  {sweep.job_id}  {sweep.name}  work_dir={info.work_dir}")
        print(f"    {info.loop_summary}")

        index_path = info.work_dir / "experiment_run_dirs.txt"
        if not index_path.exists():
            print("    experiment_run_dirs.txt missing")
            continue

        run_dirs = [
            Path(line.strip())
            for line in _read_text(index_path).splitlines()
            if line.strip()
        ]
        counts = {"DONE": 0, "ACTIVE": 0, "FAILED": 0, "PENDING": 0}
        for run_dir in run_dirs:
            status, _ = _infer_run_status(run_dir)
            counts[status] = counts.get(status, 0) + 1
        print(
            "    runs: "
            f"DONE={counts.get('DONE', 0)} "
            f"ACTIVE={counts.get('ACTIVE', 0)} "
            f"FAILED={counts.get('FAILED', 0)} "
            f"PENDING={counts.get('PENDING', 0)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show status of sweep and experiment jobs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Standalone project root (default: parent of experiments folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    try:
        jobs = _run_squeue()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    sweep_jobs = [job for job in jobs if job.name.startswith("mt-sweep-")]
    experiment_jobs = [
        job
        for job in jobs
        if job.name.startswith("parseq-gen-") or job.name.startswith("parseq-eval-")
    ]
    sweep_workdirs = _find_sweep_workdirs(root)

    print(f"Root: {root}")
    print()
    _print_queue_jobs("Sweep Jobs", sweep_jobs)
    print()
    _print_queue_jobs("Experiment Jobs", experiment_jobs)
    print()
    _print_experiment_job_summary(experiment_jobs)
    print()
    _print_sweep_workdir_summary(sweep_jobs, sweep_workdirs)


if __name__ == "__main__":
    main()
