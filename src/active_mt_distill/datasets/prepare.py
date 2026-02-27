from __future__ import annotations

import argparse
import os
from pathlib import Path

from active_mt_distill.config import (
    ensure_dir_symlink,
    find_repo_root,
    get_data_config,
    get_language_pairs,
    load_merged_config,
    resolve_path,
)
from active_mt_distill.datasets.flores import build_flores_devtest
from active_mt_distill.datasets.opus100 import build_opus_pools


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare frozen FLORES devtest + OPUS-100 train/candidate pools."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("synth_data/configs/exp_base.yaml"),
        help="Path to active distillation config YAML.",
    )
    parser.add_argument(
        "--user-config",
        type=Path,
        default=None,
        help="Optional user override config. Defaults to synth_data/configs/users/$USER.yaml if present.",
    )
    parser.add_argument(
        "--limit-flores-per-pair",
        type=int,
        default=None,
        help="Optional devtest limit per language pair for smoke runs.",
    )
    parser.add_argument("--skip-flores", action="store_true", help="Skip FLORES devtest generation.")
    parser.add_argument("--skip-opus", action="store_true", help="Skip OPUS-100 pool generation.")
    return parser.parse_args()


def _setup_storage_links(
    *,
    cfg: dict,
    data_cfg: dict,
    repo_root: Path,
) -> None:
    base_data_folder = str(cfg.get("base_data_folder", "")).strip()
    cache_folder = str(cfg.get("cache_folder", "")).strip()
    model_store = str(cfg.get("model_store_folder", "")).strip()
    local_link_root = resolve_path(
        repo_root,
        str(cfg.get("local_output_symlink_root", "synth_data/outputs")),
    )
    local_link_root.mkdir(parents=True, exist_ok=True)

    if base_data_folder:
        base_dir = resolve_path(repo_root, base_data_folder)
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = None

    processed_dir = resolve_path(repo_root, str(Path(str(data_cfg["flores_devtest_path"])).parent))
    iterations_dir = resolve_path(repo_root, str(data_cfg.get("iterations_folder", processed_dir.parent / "iterations")))
    logs_dir = resolve_path(repo_root, str(data_cfg.get("log_folder", processed_dir.parent / "logs")))
    processed_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ensure_dir_symlink(local_link_root / "processed", processed_dir)
    ensure_dir_symlink(local_link_root / "iterations", iterations_dir)
    ensure_dir_symlink(local_link_root / "logs", logs_dir)

    if cache_folder:
        cache_dir = resolve_path(repo_root, cache_folder)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ensure_dir_symlink(local_link_root / "cache", cache_dir)
    if model_store:
        model_dir = resolve_path(repo_root, model_store)
        model_dir.mkdir(parents=True, exist_ok=True)
        ensure_dir_symlink(local_link_root / "models", model_dir)
    if base_dir is not None:
        ensure_dir_symlink(local_link_root / "base_data_folder", base_dir)


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    cfg, user_cfg_path = load_merged_config(config_path, user_config_path=args.user_config)
    repo_root = find_repo_root(config_path.parent)

    pairs = get_language_pairs(cfg)
    data_cfg = get_data_config(cfg)
    seed = int(cfg.get("seed", 13))

    _setup_storage_links(cfg=cfg, data_cfg=data_cfg, repo_root=repo_root)

    hf_cache_dir = data_cfg.get("hf_cache_dir", None)
    if hf_cache_dir:
        hf_cache_root = resolve_path(repo_root, str(hf_cache_dir))
        hf_hub_cache = hf_cache_root / "hub"
        hf_datasets_cache = hf_cache_root / "datasets"
        hf_cache_root.mkdir(parents=True, exist_ok=True)
        hf_hub_cache.mkdir(parents=True, exist_ok=True)
        hf_datasets_cache.mkdir(parents=True, exist_ok=True)

        # Ensure both datasets and huggingface_hub use writable caches.
        os.environ.setdefault("HF_HOME", str(hf_cache_root))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
        hf_cache_dir = str(hf_datasets_cache)

    print(f"[info] config={config_path}")
    if user_cfg_path is not None:
        print(f"[info] user_config={user_cfg_path}")
    print(f"[info] repo_root={repo_root}")
    print(f"[info] pairs={pairs}")

    if not args.skip_flores:
        flores_out = resolve_path(repo_root, str(data_cfg["flores_devtest_path"]))
        n_flores = build_flores_devtest(
            pairs=pairs,
            out_path=flores_out,
            hf_cache_dir=hf_cache_dir,
            limit_per_pair=args.limit_flores_per_pair,
        )
        print(f"[ok] FLORES devtest -> {flores_out} ({n_flores} rows)")

    if not args.skip_opus:
        train_out = resolve_path(repo_root, str(data_cfg["opus_pool_path"]))
        candidate_out = resolve_path(repo_root, str(data_cfg["candidate_pool_path"]))
        n_train, n_candidate = build_opus_pools(
            pairs=pairs,
            train_out_path=train_out,
            candidate_out_path=candidate_out,
            train_size_per_pair=int(data_cfg["opus_train_size_per_pair"]),
            candidate_size_per_pair=int(data_cfg["candidate_pool_size_per_pair"]),
            seed=seed,
            hf_cache_dir=hf_cache_dir,
        )
        print(f"[ok] OPUS train pool -> {train_out} ({n_train} rows)")
        print(f"[ok] OPUS candidate pool -> {candidate_out} ({n_candidate} rows)")

    if args.skip_flores and args.skip_opus:
        print("[warn] nothing to do (both --skip-flores and --skip-opus set)")


if __name__ == "__main__":
    main()
