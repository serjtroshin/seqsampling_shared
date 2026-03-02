from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "pipelines"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline_module = _load_module("test_pipeline_module", "pipelines/pipeline.py")
sweep_runner = _load_module("test_sweep_runner_module", "experiments/mt/sweep_runner.py")

PipelineConfig = pipeline_module.PipelineConfig
SingleScenarioPipeline = pipeline_module.SingleScenarioPipeline


def test_model_config_sampling_defaults_do_not_override_explicit_sweep_overrides(
    tmp_path: Path,
) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"source":"hello"}\n', encoding="utf-8")

    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: base-model",
                f"prompts_path: {prompts_path}",
                "output_dir: outputs",
                "temperature: 0.9",
                "top_p: 0.95",
                "top_k: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model_cfg_path = tmp_path / "model.yaml"
    model_cfg_path.write_text(
        "\n".join(
            [
                "model: override-model",
                "scenario_override:",
                "  temperature: 0.7",
                "  top_p: 0.8",
                "  top_k: 20",
                "  max_tokens: 8192",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        scenario=str(scenario_path),
        model_config=str(model_cfg_path),
        output_root=str(tmp_path / "runs"),
        run_name="precedence_test",
        submit_jobs=False,
        scenario_overrides=["temperature=0.0", "top_p=1.0", "top_k=1"],
    )

    pipeline = SingleScenarioPipeline(cfg)

    assert pipeline.resolved_scenario.temperature == 0.0
    assert pipeline.resolved_scenario.top_p == 1.0
    assert pipeline.resolved_scenario.top_k == 1
    assert pipeline.resolved_scenario.max_tokens == 8192


def test_sweep_runner_sampling_folder_uses_effective_sampling_overrides(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: base-model",
                f"prompts_path: {tmp_path / 'prompts.jsonl'}",
                "output_dir: outputs",
                "temperature: 0.9",
                "top_p: 0.95",
                "top_k: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "prompts.jsonl").write_text('{"source":"hello"}\n', encoding="utf-8")

    model_cfg_path = tmp_path / "model.yaml"
    model_cfg_path.write_text(
        "\n".join(
            [
                "model: override-model",
                "scenario_override:",
                "  temperature: 0.7",
                "  top_p: 0.8",
                "  top_k: 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    explicit_overrides = [
        f"scenario={scenario_path}",
        f"model_config={model_cfg_path}",
        "scenario_overrides=[temperature=0.0,top_p=1.0,top_k=1]",
    ]
    default_only_overrides = [
        f"scenario={scenario_path}",
        f"model_config={model_cfg_path}",
    ]

    assert sweep_runner._infer_sampling_folder(tmp_path, explicit_overrides) == "temp_0.0.p1.0.k1"
    assert sweep_runner._infer_sampling_folder(tmp_path, default_only_overrides) == "temp_0.7.p0.8.k20"
