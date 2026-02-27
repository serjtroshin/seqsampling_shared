from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def find_repo_root(start: Path) -> Path:
    """Find repo root by walking up until .git is found."""
    resolved = start.resolve()
    for candidate in [resolved, *resolved.parents]:
        if (candidate / ".git").exists():
            return candidate
    return resolved


def resolve_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}.")
    return data


def discover_user_config(
    base_config_path: str | Path,
    *,
    explicit_user_config: str | Path | None = None,
) -> Path | None:
    if explicit_user_config is not None:
        path = Path(explicit_user_config).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"User config not found: {path}")
        return path

    config_path = Path(base_config_path).expanduser().resolve()
    username = os.environ.get("USER", "").strip()
    if not username:
        return None
    candidate = config_path.parent / "users" / f"{username}.yaml"
    if candidate.exists():
        return candidate.resolve()
    return None


def load_merged_config(
    config_path: str | Path,
    *,
    user_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    base_path = Path(config_path).expanduser().resolve()
    base_cfg = OmegaConf.load(base_path)
    user_cfg_path = discover_user_config(base_path, explicit_user_config=user_config_path)
    if user_cfg_path is not None:
        merged = OmegaConf.merge(base_cfg, OmegaConf.load(user_cfg_path))
    else:
        merged = base_cfg
    data = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {base_path}, got {type(data).__name__}.")
    return data, user_cfg_path


def is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def ensure_dir_symlink(link_path: Path, target_dir: Path) -> None:
    target = target_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    link = Path(os.path.abspath(str(link_path.expanduser())))
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise ValueError(
                f"Cannot replace non-empty path with symlink: {link}. "
                "Move or remove it first."
            )

    link.symlink_to(target, target_is_directory=True)


def get_language_pairs(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    raw_pairs = cfg.get("languages", {}).get("pairs", [])
    pairs: list[tuple[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise ValueError(f"Language pair entries must be mappings, got {item!r}.")
        src = str(item.get("src", "")).strip()
        tgt = str(item.get("tgt", "")).strip()
        if not src or not tgt:
            raise ValueError(f"Invalid language pair entry: {item!r}")
        pairs.append((src, tgt))
    if not pairs:
        raise ValueError("No language pairs configured under languages.pairs.")
    return pairs


def get_data_config(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}))
    if not data_cfg:
        raise ValueError("Missing required config section: data")

    base_data_folder = str(cfg.get("base_data_folder", "")).strip()
    if base_data_folder:
        data_cfg.setdefault("processed_folder", f"{base_data_folder}/processed")
        data_cfg.setdefault("iterations_folder", f"{base_data_folder}/iterations")
        data_cfg.setdefault("log_folder", f"{base_data_folder}/logs")

    if "processed_folder" in data_cfg:
        processed_folder = str(data_cfg["processed_folder"])
        data_cfg.setdefault("flores_devtest_path", f"{processed_folder}/flores_devtest.jsonl")
        data_cfg.setdefault("opus_pool_path", f"{processed_folder}/opus_train_pool.jsonl")
        data_cfg.setdefault("candidate_pool_path", f"{processed_folder}/candidate_pool.jsonl")

    if "flores_devtest_path" not in data_cfg:
        raise ValueError("Missing data.flores_devtest_path")
    if "opus_pool_path" not in data_cfg:
        raise ValueError("Missing data.opus_pool_path")

    if "candidate_pool_path" not in data_cfg:
        opus_path = Path(str(data_cfg["opus_pool_path"]))
        data_cfg["candidate_pool_path"] = str(opus_path.with_name("candidate_pool.jsonl"))
    data_cfg.setdefault("opus_train_size_per_pair", 20000)
    data_cfg.setdefault("candidate_pool_size_per_pair", 2000)
    return data_cfg


def get_sft_config(cfg: dict[str, Any]) -> dict[str, Any]:
    sft_cfg = dict(cfg.get("sft", {}))
    if not sft_cfg:
        raise ValueError("Missing required config section: sft")

    cache_folder = str(cfg.get("cache_folder", "")).strip()
    model_store = str(cfg.get("model_store_folder", "")).strip()
    if cache_folder:
        sft_cfg.setdefault("cache_base", f"{cache_folder}/llamafactory")
    if model_store:
        sft_cfg.setdefault("artifact_output_root", f"{model_store}/sft_runs")

    sft_cfg.setdefault("engine", "llamafactory_apptainer")
    sft_cfg.setdefault("apptainer_bin", "apptainer")
    sft_cfg.setdefault("use_gpu", True)
    sft_cfg.setdefault("cleanenv", True)
    sft_cfg.setdefault("dataset_name", "active_mt_train")
    sft_cfg.setdefault("finetuning_type", "lora")
    sft_cfg.setdefault("lora_target", "all")
    sft_cfg.setdefault("cutoff_len", 1024)
    sft_cfg.setdefault("per_device_train_batch_size", 2)
    sft_cfg.setdefault("gradient_accumulation_steps", 8)
    sft_cfg.setdefault("learning_rate", 2e-5)
    sft_cfg.setdefault("num_train_epochs", 1.0)
    sft_cfg.setdefault("logging_steps", 10)
    sft_cfg.setdefault("save_steps", 200)
    sft_cfg.setdefault("bf16", True)
    sft_cfg.setdefault("fp16", False)
    sft_cfg.setdefault("valid_ratio", 0.0)
    sft_cfg.setdefault("output_dir_name", "checkpoints")
    return sft_cfg
