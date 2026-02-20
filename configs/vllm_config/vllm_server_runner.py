#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


SUPPORTED_RUNTIMES = {"apptainer", "local_python"}
SUPPORTED_ENTRYPOINTS = {"serve", "api_server"}


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(data).__name__}.")
    return data


def _as_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field '{field_name}' must be a list, got {type(value).__name__}.")
    return [str(item) for item in value]


def _get_str(cfg: dict[str, Any], key: str, *, required: bool = False, default: str = "") -> str:
    value = cfg.get(key, None)
    if value is None:
        if required:
            raise ValueError(f"Missing required config key: '{key}'.")
        return default
    return str(value)


def _get_bool(cfg: dict[str, Any], key: str, *, default: bool) -> bool:
    value = cfg.get(key, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Field '{key}' must be a bool, got {value!r}.")


def _as_str_dict(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Field '{field_name}' must be a mapping, got {type(value).__name__}.")
    return {str(k): str(v) for k, v in value.items()}


def _default_local_cache_base() -> Path:
    user = os.environ.get("USER", "user")
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        return Path(slurm_tmpdir).expanduser() / "parseq_vllm_cache"
    return Path("/tmp") / user / "parseq_vllm_cache"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _local_cache_env_defaults(vllm_cfg: dict[str, Any]) -> tuple[dict[str, str], list[Path]]:
    cache_base_raw = _get_str(vllm_cfg, "cache_base", default=str(_default_local_cache_base()))
    cache_base = Path(cache_base_raw).expanduser()

    xdg_cache_home = Path(_get_str(vllm_cfg, "xdg_cache_home", default=str(cache_base / "xdg_cache"))).expanduser()
    hf_home = Path(_get_str(vllm_cfg, "hf_home", default=str(cache_base / "huggingface"))).expanduser()
    hf_hub_cache = Path(_get_str(vllm_cfg, "hf_hub_cache", default=str(hf_home / "hub"))).expanduser()
    vllm_cache_dir = Path(_get_str(vllm_cfg, "vllm_cache_dir", default=str(cache_base / "vllm"))).expanduser()
    torch_extensions_dir = Path(
        _get_str(vllm_cfg, "torch_extensions_dir", default=str(cache_base / "torch_extensions"))
    ).expanduser()
    triton_cache_dir = Path(_get_str(vllm_cfg, "triton_cache_dir", default=str(cache_base / "triton"))).expanduser()

    flashinfer_workspace_base = Path(
        _get_str(vllm_cfg, "flashinfer_workspace_base", default=str(cache_base / "flashinfer"))
    ).expanduser()
    flashinfer_cache_dir = Path(
        _get_str(
            vllm_cfg,
            "flashinfer_cache_dir",
            default=str(flashinfer_workspace_base / ".cache" / "flashinfer"),
        )
    ).expanduser()
    flashinfer_workspace_dir = Path(
        _get_str(
            vllm_cfg,
            "flashinfer_workspace_dir",
            default=str(flashinfer_workspace_base / "workspace"),
        )
    ).expanduser()

    env_defaults = {
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
        "HF_HUB_CACHE": str(hf_hub_cache),
        "VLLM_CACHE_DIR": str(vllm_cache_dir),
        "TORCH_EXTENSIONS_DIR": str(torch_extensions_dir),
        "TRITON_CACHE_DIR": str(triton_cache_dir),
        "FLASHINFER_WORKSPACE_BASE": str(flashinfer_workspace_base),
        "FLASHINFER_CACHE_DIR": str(flashinfer_cache_dir),
        "FLASHINFER_WORKSPACE_DIR": str(flashinfer_workspace_dir),
    }
    mkdir_targets = _dedupe_paths(
        [
            xdg_cache_home,
            hf_home,
            hf_hub_cache,
            vllm_cache_dir,
            torch_extensions_dir,
            triton_cache_dir,
            flashinfer_cache_dir,
            flashinfer_workspace_dir,
        ]
    )
    return env_defaults, mkdir_targets


def _server_tokens(
    *,
    model_name: str,
    host: str,
    port: int,
    entrypoint: str,
    python_bin: str,
    vllm_bin: str,
) -> list[str]:
    if entrypoint not in SUPPORTED_ENTRYPOINTS:
        allowed = ", ".join(sorted(SUPPORTED_ENTRYPOINTS))
        raise ValueError(f"Unsupported entrypoint '{entrypoint}'. Allowed: {allowed}.")

    if entrypoint == "serve":
        return [vllm_bin, "serve", model_name, "--host", host, "--port", str(port)]

    return [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
    ]


def _build_local_python_command(
    model_cfg: dict[str, Any],
    vllm_cfg: dict[str, Any],
    *,
    host_override: str | None = None,
    port_override: int | None = None,
) -> str:
    model_name = _get_str(model_cfg, "model", required=True)
    model_server_args = _as_str_list(model_cfg.get("vllm_server_args"), "model.vllm_server_args")

    host = host_override if host_override is not None else _get_str(vllm_cfg, "host", default="127.0.0.1")
    port = port_override if port_override is not None else int(_get_str(vllm_cfg, "port", default="8000"))
    entrypoint = _get_str(vllm_cfg, "entrypoint", default="api_server")
    python_bin = _get_str(vllm_cfg, "python_bin", default=sys.executable)
    vllm_bin = _get_str(vllm_cfg, "vllm_bin", default="vllm")
    extra_server_args = _as_str_list(vllm_cfg.get("extra_server_args"), "vllm.extra_server_args")
    use_default_cache_env = _get_bool(vllm_cfg, "use_default_cache_env", default=True)
    env_overrides = _as_str_dict(vllm_cfg.get("env"), "vllm.env")

    env: dict[str, str] = {}
    mkdir_targets: list[Path] = []
    if use_default_cache_env:
        default_env, mkdir_targets = _local_cache_env_defaults(vllm_cfg)
        env.update(default_env)
    env.update(env_overrides)

    hf_token = _get_str(vllm_cfg, "hf_token", default="")
    if hf_token and "HF_TOKEN" not in env:
        env["HF_TOKEN"] = hf_token

    command_tokens = _server_tokens(
        model_name=model_name,
        host=host,
        port=port,
        entrypoint=entrypoint,
        python_bin=python_bin,
        vllm_bin=vllm_bin,
    )
    command_tokens.extend(model_server_args)
    command_tokens.extend(extra_server_args)
    command = shlex.join(command_tokens)

    if not env:
        if not mkdir_targets:
            return command
        mkdir_cmd = "mkdir -p " + " ".join(shlex.quote(str(path)) for path in mkdir_targets)
        return f"{mkdir_cmd} && {command}"

    env_prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())
    if not mkdir_targets:
        return f"{env_prefix} {command}"
    mkdir_cmd = "mkdir -p " + " ".join(shlex.quote(str(path)) for path in mkdir_targets)
    return f"{mkdir_cmd} && {env_prefix} {command}"


def _build_apptainer_command(
    model_cfg: dict[str, Any],
    vllm_cfg: dict[str, Any],
    *,
    host_override: str | None = None,
    port_override: int | None = None,
) -> str:
    model_name = _get_str(model_cfg, "model", required=True)
    model_server_args = _as_str_list(model_cfg.get("vllm_server_args"), "model.vllm_server_args")

    image_sif = _get_str(vllm_cfg, "image_sif", required=True)
    cache_base = Path(_get_str(vllm_cfg, "cache_base", required=True)).expanduser()
    host = host_override if host_override is not None else _get_str(vllm_cfg, "host", default="127.0.0.1")
    port = port_override if port_override is not None else int(_get_str(vllm_cfg, "port", default="8000"))
    entrypoint = _get_str(vllm_cfg, "entrypoint", default="serve")
    python_bin = _get_str(vllm_cfg, "python_bin", default="python")
    vllm_bin = _get_str(vllm_cfg, "vllm_bin", default="vllm")
    apptainer_bin = _get_str(vllm_cfg, "apptainer_bin", default="apptainer")
    container_cache_root = _get_str(
        vllm_cfg,
        "container_cache_root",
        default=f"{cache_base.as_posix().rstrip('/')}/.cache",
    ).rstrip("/")
    use_gpu = bool(vllm_cfg.get("use_gpu", True))
    cleanenv = bool(vllm_cfg.get("cleanenv", True))
    extra_server_args = _as_str_list(vllm_cfg.get("extra_server_args"), "vllm.extra_server_args")

    user = os.environ.get("USER", "user")
    flashinfer_workspace_base = _get_str(vllm_cfg, "flashinfer_workspace_base", default=f"/tmp/{user}")
    flashinfer_cache_dir = _get_str(
        vllm_cfg,
        "flashinfer_cache_dir",
        default=f"{flashinfer_workspace_base.rstrip('/')}/.cache/flashinfer",
    )
    flashinfer_workspace_dir = _get_str(
        vllm_cfg,
        "flashinfer_workspace_dir",
        default=f"{flashinfer_workspace_base.rstrip('/')}/flashinfer",
    )

    hf_cache_dir = cache_base / "hf"
    vllm_cache_dir = cache_base / "vllm"
    torch_ext_dir = cache_base / "torch_extensions"
    tmp_dir = cache_base / "tmp"
    tmp_user_dir = tmp_dir / user
    flashinfer_host_cache_dir = tmp_user_dir / ".cache" / "flashinfer"
    flashinfer_host_workspace_dir = tmp_user_dir / "flashinfer"

    mkdir_targets = [
        hf_cache_dir,
        vllm_cache_dir,
        torch_ext_dir,
        tmp_dir,
        flashinfer_host_cache_dir,
        flashinfer_host_workspace_dir,
    ]
    mkdir_cmd = "mkdir -p " + " ".join(shlex.quote(str(path)) for path in mkdir_targets)

    container_hf_home = f"{container_cache_root}/huggingface"
    container_vllm_cache = f"{container_cache_root}/vllm"
    container_torch_ext_dir = f"{container_cache_root}/torch_extensions"

    command_tokens: list[str] = [apptainer_bin, "exec"]
    if use_gpu:
        command_tokens.append("--nv")
    if cleanenv:
        command_tokens.append("--cleanenv")

    bind_pairs = [
        (hf_cache_dir, container_hf_home),
        (vllm_cache_dir, container_vllm_cache),
        (torch_ext_dir, container_torch_ext_dir),
        (tmp_dir, "/tmp"),
    ]
    for host_path, container_path in bind_pairs:
        command_tokens.extend(["-B", f"{host_path}:{container_path}"])

    env_pairs = [
        ("XDG_CACHE_HOME", f"{container_cache_root}/xdg_cache"),
        ("HF_HOME", container_hf_home),
        ("TRANSFORMERS_CACHE", container_hf_home),
        ("HUGGINGFACE_HUB_CACHE", f"{container_hf_home}/hub"),
        ("VLLM_CACHE_DIR", container_vllm_cache),
        ("TORCH_EXTENSIONS_DIR", container_torch_ext_dir),
        ("FLASHINFER_WORKSPACE_BASE", flashinfer_workspace_base),
        ("FLASHINFER_CACHE_DIR", flashinfer_cache_dir),
        ("FLASHINFER_WORKSPACE_DIR", flashinfer_workspace_dir),
    ]
    for key, value in env_pairs:
        command_tokens.extend(["--env", f"{key}={value}"])

    hf_token = vllm_cfg.get("hf_token", None)
    if hf_token:
        command_tokens.extend(["--env", f"HF_TOKEN={hf_token}"])

    command_tokens.append(image_sif)
    command_tokens.extend(
        _server_tokens(
            model_name=model_name,
            host=host,
            port=port,
            entrypoint=entrypoint,
            python_bin=python_bin,
            vllm_bin=vllm_bin,
        )
    )
    command_tokens.extend(model_server_args)
    command_tokens.extend(extra_server_args)

    return f"{mkdir_cmd} && {shlex.join(command_tokens)}"


def build_command_from_dicts(
    model_cfg: dict[str, Any],
    vllm_cfg: dict[str, Any],
    *,
    host_override: str | None = None,
    port_override: int | None = None,
) -> str:
    runtime = _get_str(vllm_cfg, "runtime", required=True)

    if runtime not in SUPPORTED_RUNTIMES:
        allowed = ", ".join(sorted(SUPPORTED_RUNTIMES))
        raise ValueError(f"Unsupported runtime '{runtime}'. Allowed: {allowed}.")

    if runtime == "apptainer":
        return _build_apptainer_command(
            model_cfg,
            vllm_cfg,
            host_override=host_override,
            port_override=port_override,
        )
    return _build_local_python_command(
        model_cfg,
        vllm_cfg,
        host_override=host_override,
        port_override=port_override,
    )


def build_command(
    model_config_path: Path,
    vllm_config_path: Path,
    *,
    host_override: str | None = None,
    port_override: int | None = None,
) -> str:
    model_cfg = _load_yaml_dict(model_config_path)
    vllm_cfg = _load_yaml_dict(vllm_config_path)
    return build_command_from_dicts(
        model_cfg,
        vllm_cfg,
        host_override=host_override,
        port_override=port_override,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a vLLM server shell command from a model config "
            "(configs/model_configs/*.yaml) and a runtime config "
            "(configs/vllm_config/*.yaml)."
        )
    )
    parser.add_argument("--model-config", required=True, type=Path, help="Path to model config YAML.")
    parser.add_argument("--vllm-config", required=True, type=Path, help="Path to vLLM runtime config YAML.")
    parser.add_argument("--host", type=str, default=None, help="Optional runtime host override.")
    parser.add_argument("--port", type=int, default=None, help="Optional runtime port override.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = build_command(
        args.model_config,
        args.vllm_config,
        host_override=args.host,
        port_override=args.port,
    )
    print(command)


if __name__ == "__main__":
    main()
