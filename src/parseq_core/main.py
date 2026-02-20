from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .enumeration import run_enumeration_scenario
from .scenario import Scenario
from .iterative import run_iterative_scenario
from .multi_turn import run_multi_turn_scenario
from .parallel import run_parallel_scenario


def run_scenario(scenario_path: Path, overrides: list[str] | None = None, draft_prompt: bool = False) -> Path:
    scenario = Scenario.load(scenario_path, overrides=overrides)
    _maybe_dump_resolved_config(scenario)
    if scenario.is_multi_turn():
        return run_multi_turn_scenario(scenario_path, overrides=overrides, draft_prompt=draft_prompt)
    if scenario.is_iterative() or scenario.is_iterkpar():
        return run_iterative_scenario(scenario_path, overrides=overrides, draft_prompt=draft_prompt)
    if scenario.is_enumeration():
        return run_enumeration_scenario(scenario_path, overrides=overrides, draft_prompt=draft_prompt)

    # Parallel path
    if scenario.reuse_from:
        print(f"[reuse] trying reuse_from={scenario.reuse_from}", flush=True)
        reused = _maybe_reuse_parallel(scenario)
        if reused:
            print(f"[reuse] reused parallel outputs from {scenario.reuse_from}", flush=True)
            return reused

    if scenario.request_logprobs:
        print("[warn] request_logprobs is not supported in unified sampler path yet; ignoring.", flush=True)

    return run_parallel_scenario(scenario, draft_prompt=draft_prompt)


def _maybe_dump_resolved_config(scenario: Scenario) -> None:
    """Write the resolved scenario config to dump_path if provided."""
    if not scenario.dump_path:
        return
    cfg = getattr(scenario, "_resolved_cfg", None)
    scenario.dump_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        scenario.dump_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    else:  # fallback
        import json
        scenario.dump_path.write_text(json.dumps(scenario.model_dump(), indent=2, default=str), encoding="utf-8")


def _maybe_reuse_parallel(scenario: Scenario) -> Path | None:
    """If reuse_from is set and configs match, symlink outputs and skip sampling."""
    reuse_dir = scenario.reuse_from
    if reuse_dir is None:
        return None
    reuse_dir = Path(reuse_dir).expanduser().resolve() / "parallel"
    src_samples = reuse_dir / "samples.jsonl"
    src_cfg = reuse_dir / "scenario_resolved.yaml"
    if not src_samples.exists() or not src_cfg.exists():
        print(f"[reuse] missing reuse files: samples={src_samples.exists()} cfg={src_cfg.exists()}", flush=True)
        print(f"[reuse] cannot reuse from {reuse_dir}", flush=True)
        return None

    try:
        from omegaconf import OmegaConf

        cur_cfg = getattr(scenario, "_resolved_cfg", None)
        if cur_cfg is None:
            return None
        src_cfg_data = OmegaConf.load(src_cfg)

        def strip(cfg):
            # remove fields that are allowed to differ
            cfg = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, dict):
                cfg.pop("output_dir", None)
                cfg.pop("dump_path", None)
                cfg.pop("reuse_from", None)
                cfg.pop("vllm_base_url", None)
                cfg.pop("vllm_port", None)
            return cfg

        cur_flat = strip(cur_cfg)
        src_flat = strip(src_cfg_data)
        if cur_flat != src_flat:
            diffs = []
            keys = set(cur_flat.keys()) | set(src_flat.keys())
            for k in sorted(keys):
                if cur_flat.get(k) != src_flat.get(k):
                    diffs.append(f"{k}: want={cur_flat.get(k)} have={src_flat.get(k)}")
            print(f"[reuse] config mismatch; skip reuse. Diffs: {'; '.join(diffs)}", flush=True)
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"[reuse] failed to load/compare configs: {exc}", flush=True)
        return None

    dst_samples = scenario.output_path()
    dst_cfg = scenario.dump_path or dst_samples.parent / "scenario_resolved.yaml"
    dst_samples.parent.mkdir(parents=True, exist_ok=True)
    dst_cfg.parent.mkdir(parents=True, exist_ok=True)
    for dst, src in [(dst_samples, src_samples), (dst_cfg, src_cfg)]:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    print(f"[reuse] reused parallel outputs from {src_samples}", flush=True)
    return dst_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a sampling scenario and save generations to disk."
    )
    parser.add_argument("--scenario", type=Path, required=True, help="Path to scenario YAML/JSON file.")
    parser.add_argument(
        "--draft-prompt",
        action="store_true",
        help="Print the first request (system+user) and exit without running generation.",
    )
    args, hydra_overrides = parser.parse_known_args()

    output = run_scenario(args.scenario, overrides=hydra_overrides, draft_prompt=args.draft_prompt)
    print(f"Saved generations to {output}")


if __name__ == "__main__":
    main()
