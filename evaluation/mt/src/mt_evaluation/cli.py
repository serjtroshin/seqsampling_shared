from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import default_metrics, load_pipeline_config, preset_metric
from .pipeline import evaluate_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MT outputs and write reports into an evaluation folder.")
    parser.add_argument("--file", required=True, help="Path to samples JSONL produced by sampling runs.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for reports. Defaults to <samples_dir>/evaluation.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file describing metrics and hyperparameters.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric preset to run (repeatable). Presets: comet_qe (reference-based), comet_kiwi_qe (reference-free), comet_ref, xcomet_xxl, metricx24_ref, metricx24_qe. "
            "Ignored when --config is provided."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Default batch size for preset metrics (ignored when per-metric config sets one).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for preset metrics.",
    )
    parser.add_argument(
        "--generation-key",
        action="append",
        default=None,
        help="Generation field(s) to evaluate. Repeat to pass multiple keys.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of response records scored.",
    )
    parser.add_argument(
        "--no-final-answers",
        action="store_true",
        help="Skip evaluating 'final_answers' even if present.",
    )
    return parser


def _run_isolated_metrics(args: argparse.Namespace) -> bool:
    if not args.config:
        return False

    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    metrics_raw = data.get("metrics")
    if not isinstance(metrics_raw, list) or len(metrics_raw) <= 1:
        return False

    with tempfile.TemporaryDirectory(prefix="mt_eval_metrics_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, metric in enumerate(metrics_raw):
            if not isinstance(metric, dict):
                raise ValueError(f"Metric entry at index {idx} must be an object.")

            metric_name = str(metric.get("name") or metric.get("type") or f"metric_{idx}")
            metric_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", metric_name).strip("_") or f"metric_{idx}"
            single_cfg = dict(data)
            single_cfg["metrics"] = [metric]

            single_cfg_path = tmp_root / f"{idx:02d}_{metric_slug}.json"
            single_cfg_path.write_text(json.dumps(single_cfg), encoding="utf-8")

            cmd: list[str] = [
                sys.executable,
                "-m",
                "mt_evaluation.cli",
                "--file",
                args.file,
                "--config",
                str(single_cfg_path),
            ]
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])
            if args.generation_key:
                for key in args.generation_key:
                    cmd.extend(["--generation-key", key])
            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.no_final_answers:
                cmd.append("--no-final-answers")

            print(f"[mt-eval] running metric in isolated process: {metric_name}")
            subprocess.run(cmd, check=True)

    return True


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if _run_isolated_metrics(args):
        return

    input_file = Path(args.file)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.config:
        config = load_pipeline_config(Path(args.config))
        metrics = config.metrics
        generation_keys = tuple(args.generation_key) if args.generation_key else config.generation_keys
        include_final_answers = config.include_final_answers and (not args.no_final_answers)
    else:
        metric_names = args.metric or ["comet_kiwi_qe"]
        metrics = [preset_metric(name, batch_size=args.batch_size, model_name=args.model) for name in metric_names]
        generation_keys = tuple(args.generation_key) if args.generation_key else ("generations", "solutions")
        include_final_answers = not args.no_final_answers
        if not metrics:
            metrics = default_metrics(default_batch_size=args.batch_size, model_name=args.model)

    outputs = evaluate_file(
        input_file=input_file,
        metrics=metrics,
        output_dir=output_dir,
        generation_keys=generation_keys,
        include_final_answers=include_final_answers,
        max_samples=args.max_samples,
    )
    for metric_name, out_path in outputs.items():
        print(f"{metric_name}: {out_path}")


if __name__ == "__main__":
    main()
