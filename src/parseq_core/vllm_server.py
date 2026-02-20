from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .scenario import Scenario


def _build_server_command(scenario: Scenario) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        scenario.model,
        "--host",
        scenario.vllm_host,
        "--port",
        str(scenario.vllm_port),
    ]

    if scenario.trust_remote_code:
        cmd.append("--trust-remote-code")

    if scenario.vllm_server_args:
        cmd.extend(scenario.vllm_server_args)

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

    cmd = _build_server_command(scenario)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
