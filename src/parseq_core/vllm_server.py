from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .scenario import Scenario


def _load_runner_builder(root: Path) -> Callable[..., str]:
    runner_path = root / "configs" / "vllm_config" / "vllm_server_runner.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"vLLM runner script not found: {runner_path}")

    spec = importlib.util.spec_from_file_location("parseq_vllm_server_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner module spec: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_command_from_dicts", None)
    if not callable(builder):
        raise RuntimeError(f"Runner module missing callable build_command_from_dicts: {runner_path}")
    return builder


def _build_server_command(root: Path, scenario: Scenario) -> str:
    model_server_args = list(scenario.vllm_server_args)
    if scenario.trust_remote_code and "--trust-remote-code" not in model_server_args:
        model_server_args = ["--trust-remote-code", *model_server_args]

    model_cfg: dict[str, Any] = {
        "model": scenario.model,
        "vllm_server_args": model_server_args,
    }
    vllm_cfg: dict[str, Any] = {
        "runtime": "local_python",
        "entrypoint": "api_server",
        "python_bin": sys.executable,
        "host": scenario.vllm_host,
        "port": scenario.vllm_port,
    }

    build_command_from_dicts = _load_runner_builder(root)
    cmd = build_command_from_dicts(
        model_cfg,
        vllm_cfg,
        host_override=scenario.vllm_host,
        port_override=scenario.vllm_port,
    )
    if not isinstance(cmd, str) or not cmd.strip():
        raise RuntimeError("vLLM runner returned an empty launch command")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a vLLM OpenAI-compatible server from a scenario."
    )
    parser.add_argument("--scenario", type=Path, required=True)
    args = parser.parse_args()

    scenario = Scenario.load(args.scenario)
    if scenario.backend != "vllm":
        raise ValueError("Scenario backend must be 'vllm' to run the vLLM server.")

    root = Path(__file__).resolve().parents[2]
    cmd = _build_server_command(root, scenario)
    subprocess.run(["bash", "-lc", cmd], check=True)


if __name__ == "__main__":
    main()
