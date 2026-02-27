from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from active_mt_distill.config import (
    ensure_dir_symlink,
    find_repo_root,
    get_sft_config,
    is_subpath,
    load_merged_config,
    resolve_path,
)
from active_mt_distill.sft.dataset_export import export_for_llamafactory


def _expand_path(path_like: str | Path) -> Path:
    return Path(os.path.expandvars(str(path_like))).expanduser().resolve()


def _render_sft_config(
    *,
    template_path: Path | None,
    dynamic_config: dict[str, Any],
    out_path: Path,
) -> None:
    base_cfg = OmegaConf.create({})
    if template_path is not None:
        base_cfg = OmegaConf.load(template_path)
    merged = OmegaConf.merge(base_cfg, OmegaConf.create(dynamic_config))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=merged, f=str(out_path))


def _build_apptainer_train_command(
    *,
    image_sif: Path,
    run_config_path: Path,
    bind_paths: list[Path],
    sft_cfg: dict[str, Any],
) -> str:
    apptainer_bin = str(sft_cfg.get("apptainer_bin", "apptainer"))
    use_gpu = bool(sft_cfg.get("use_gpu", True))
    cleanenv = bool(sft_cfg.get("cleanenv", True))

    user = os.environ.get("USER", "user")
    cache_base_default = f"/tmp/{user}/active_mt_distill_lf_cache"
    cache_base = _expand_path(str(sft_cfg.get("cache_base", cache_base_default)))
    hf_home = cache_base / "huggingface"
    xdg_cache_home = cache_base / "xdg_cache"
    tmp_dir = cache_base / "tmp"

    mkdir_targets = [hf_home, xdg_cache_home, tmp_dir]
    mkdir_cmd = "mkdir -p " + " ".join(shlex.quote(str(path)) for path in mkdir_targets)

    command: list[str] = [apptainer_bin, "exec"]
    if use_gpu:
        command.append("--nv")
    if cleanenv:
        command.append("--cleanenv")

    all_binds = {path.resolve() for path in bind_paths}
    all_binds.update({hf_home.resolve(), xdg_cache_home.resolve(), tmp_dir.resolve()})
    for host_path in sorted(all_binds):
        command.extend(["-B", f"{host_path}:{host_path}"])

    env_pairs = {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "TMPDIR": str(tmp_dir),
    }
    for key, value in env_pairs.items():
        command.extend(["--env", f"{key}={value}"])

    command.append(str(image_sif))
    command.extend(["llamafactory-cli", "train", str(run_config_path)])

    # Auto-forward HF credentials into container without requiring manual export.
    # Priority:
    # 1) explicit sft.hf_token
    # 2) already-exported APPTAINERENV_HF_TOKEN / HF_TOKEN
    # 3) ~/.cache/huggingface/token
    hf_token = str(sft_cfg.get("hf_token", "")).strip()
    auth_setup_parts: list[str] = []
    if hf_token:
        quoted = shlex.quote(hf_token)
        auth_setup_parts.append(
            f"export APPTAINERENV_HF_TOKEN={quoted} "
            f"APPTAINERENV_HUGGING_FACE_HUB_TOKEN={quoted}"
        )
    else:
        auth_setup_parts.append(
            'if [ -z "${APPTAINERENV_HF_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then '
            'export APPTAINERENV_HF_TOKEN="${HF_TOKEN}" '
            'APPTAINERENV_HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; '
            "fi"
        )
        auth_setup_parts.append(
            'if [ -z "${APPTAINERENV_HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then '
            'export APPTAINERENV_HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")" '
            'APPTAINERENV_HUGGING_FACE_HUB_TOKEN="${APPTAINERENV_HF_TOKEN}"; '
            "fi"
        )

    auth_setup_cmd = " && ".join(auth_setup_parts)
    return f"{mkdir_cmd} && {auth_setup_cmd} && {shlex.join(command)}"


def _resolve_artifact_out_dir(
    *,
    out_dir: str | Path,
    cfg: dict[str, Any],
    repo_root: Path,
) -> tuple[Path, Path | None]:
    requested = Path(str(out_dir)).expanduser()
    local_path = requested if requested.is_absolute() else (repo_root / requested)
    local_path = local_path.resolve()

    base_data_folder = str(cfg.get("base_data_folder", "")).strip()
    cache_folder = str(cfg.get("cache_folder", "")).strip()
    model_store = str(cfg.get("model_store_folder", "")).strip()
    allowed_roots: list[Path] = []
    for raw in (base_data_folder, cache_folder, model_store):
        if raw:
            allowed_roots.append(_expand_path(raw))

    if allowed_roots and any(is_subpath(local_path, root) for root in allowed_roots):
        local_path.mkdir(parents=True, exist_ok=True)
        return local_path, None

    artifact_output_root_raw = str(cfg.get("artifact_output_root", "")).strip()
    if not artifact_output_root_raw:
        local_path.mkdir(parents=True, exist_ok=True)
        return local_path, None

    artifact_output_root = _expand_path(artifact_output_root_raw)
    artifact_output_root.mkdir(parents=True, exist_ok=True)
    real_dir = artifact_output_root / local_path.name
    real_dir.mkdir(parents=True, exist_ok=True)

    ensure_dir_symlink(local_path, real_dir)
    return real_dir, local_path


def _setup_local_output_links(cfg: dict[str, Any], repo_root: Path) -> None:
    local_link_root = resolve_path(
        repo_root,
        str(cfg.get("local_output_symlink_root", "synth_data/outputs")),
    )
    local_link_root.mkdir(parents=True, exist_ok=True)

    for key, link_name in [
        ("base_data_folder", "base_data_folder"),
        ("cache_folder", "cache"),
        ("model_store_folder", "models"),
        ("artifact_output_root", "sft_runs"),
    ]:
        raw = str(cfg.get(key, "")).strip()
        if not raw:
            continue
        target = _expand_path(raw)
        target.mkdir(parents=True, exist_ok=True)
        ensure_dir_symlink(local_link_root / link_name, target)


def run_sft(train_jsonl: str, base_model: str, out_dir: str, cfg: dict[str, Any]) -> str:
    """
    Run LlamaFactory SFT in an apptainer container.

    Returns the output checkpoint directory path.
    """

    repo_root = find_repo_root(Path.cwd())
    _setup_local_output_links(cfg, repo_root)
    out_root, local_symlink_path = _resolve_artifact_out_dir(out_dir=out_dir, cfg=cfg, repo_root=repo_root)

    dataset_name = str(cfg.get("dataset_name", "active_mt_train"))
    valid_ratio = float(cfg.get("valid_ratio", 0.0))
    seed = int(cfg.get("seed", 13))
    export_result = export_for_llamafactory(
        train_jsonl=train_jsonl,
        output_dir=out_root / "lf_dataset",
        dataset_name=dataset_name,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    output_dir_name = str(cfg.get("output_dir_name", "checkpoints"))
    checkpoint_dir = out_root / output_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    template_path_value = str(cfg.get("template_path", "")).strip()
    template_path = _expand_path(template_path_value) if template_path_value else None

    dynamic_train_cfg: dict[str, Any] = {
        "stage": "sft",
        "do_train": True,
        "overwrite_output_dir": True,
        "model_name_or_path": base_model,
        "dataset_dir": export_result["export_dir"],
        "dataset": export_result["train_dataset_name"],
        "output_dir": str(checkpoint_dir),
        "finetuning_type": str(cfg.get("finetuning_type", "lora")),
        "lora_target": str(cfg.get("lora_target", "all")),
        "cutoff_len": int(cfg.get("cutoff_len", 1024)),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(cfg.get("learning_rate", 2e-5)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 1.0)),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "save_steps": int(cfg.get("save_steps", 200)),
        "bf16": bool(cfg.get("bf16", True)),
        "fp16": bool(cfg.get("fp16", False)),
    }
    if export_result["valid_dataset_name"]:
        dynamic_train_cfg["eval_dataset"] = export_result["valid_dataset_name"]
        dynamic_train_cfg["do_eval"] = True

    run_config_path = out_root / "llamafactory_sft.yaml"
    _render_sft_config(
        template_path=template_path,
        dynamic_config=dynamic_train_cfg,
        out_path=run_config_path,
    )

    image_raw = str(cfg.get("apptainer_image", "")).strip()
    if not image_raw:
        raise ValueError("Missing sft.apptainer_image.")
    image_sif = _expand_path(image_raw)
    bind_dirs = [
        Path(train_jsonl).expanduser().resolve().parent,
        Path(export_result["export_dir"]).resolve(),
        checkpoint_dir.resolve(),
        run_config_path.resolve().parent,
    ]
    for bind_dir in bind_dirs:
        bind_dir.mkdir(parents=True, exist_ok=True)

    command = _build_apptainer_train_command(
        image_sif=image_sif,
        run_config_path=run_config_path,
        bind_paths=bind_dirs,
        sft_cfg=cfg,
    )

    command_path = out_root / "run_llamafactory_sft.sh"
    command_path.write_text(command + "\n", encoding="utf-8")
    command_path.chmod(0o755)

    dry_run = bool(cfg.get("dry_run", False))
    if dry_run:
        if local_symlink_path is not None:
            print(f"[dry-run] local output symlink: {local_symlink_path} -> {out_root}")
        print("[dry-run] generated LlamaFactory command:")
        print(command)
        return str(checkpoint_dir)

    subprocess.run(command, shell=True, check=True)
    return str(checkpoint_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LlamaFactory SFT with apptainer.")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Canonical train JSONL input.")
    parser.add_argument("--base-model", type=str, required=True, help="Base model path or HF model id.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output root for this SFT run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("synth_data/configs/exp_base.yaml"),
        help="Config YAML with sft section.",
    )
    parser.add_argument(
        "--user-config",
        type=Path,
        default=None,
        help="Optional user override config. Defaults to synth_data/configs/users/$USER.yaml if present.",
    )
    parser.add_argument("--apptainer-image", type=Path, default=None, help="Optional override for image path.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional dataset name override.")
    parser.add_argument("--dry-run", action="store_true", help="Only write config and print command.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    cfg_all, user_cfg_path = load_merged_config(config_path, user_config_path=args.user_config)
    cfg = get_sft_config(cfg_all)
    repo_root = find_repo_root(config_path.parent)
    cfg["seed"] = int(cfg_all.get("seed", 13))
    for key in ("base_data_folder", "cache_folder", "model_store_folder", "local_output_symlink_root"):
        if key in cfg_all:
            cfg[key] = cfg_all[key]

    if args.apptainer_image is not None:
        cfg["apptainer_image"] = str(args.apptainer_image)
    else:
        cfg["apptainer_image"] = str(resolve_path(repo_root, str(cfg["apptainer_image"])))

    if "template_path" in cfg and str(cfg["template_path"]).strip():
        cfg["template_path"] = str(resolve_path(repo_root, str(cfg["template_path"])))
    if args.dataset_name:
        cfg["dataset_name"] = args.dataset_name
    cfg["dry_run"] = args.dry_run

    checkpoint_dir = run_sft(
        train_jsonl=str(args.train_jsonl),
        base_model=args.base_model,
        out_dir=str(args.out_dir),
        cfg=cfg,
    )
    payload: dict[str, Any] = {"checkpoint_dir": checkpoint_dir}
    if user_cfg_path is not None:
        payload["user_config"] = str(user_cfg_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
