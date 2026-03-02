from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from active_mt_distill.pipeline import (
    EvalConfig,
    EvalOnlyPipeline,
    FinetuneConfig,
    GenerationConfig,
    PipelineConfig,
)


def test_eval_only_pipeline_writes_script_and_resolves_prompt_overrides(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "en-ru.jsonl").write_text('{"source":"hello","target":"privet"}\n', encoding="utf-8")
    (prompts_dir / "default.txt").write_text("fallback prompt\n", encoding="utf-8")

    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text('{"generations":["privet"]}\n', encoding="utf-8")

    eval_tool_dir = tmp_path / "eval_tool"
    (eval_tool_dir / "src").mkdir(parents=True)
    fake_python = eval_tool_dir / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_parallel",
                "backend: vllm",
                "sampling_mode: parallel",
                "model: test-model",
                "prompts_path: prompts/default.txt",
                "output_dir: outputs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        scenario=str(scenario_path),
        prompts_path=str(prompts_dir),
        prompt_key="source",
        extra_data_fields=["target"],
        limit_prompts=3,
        prompt_template="Translate: {prompt}",
        system_prompt="You are a translator.",
        tgt="ru",
        samples_path=str(samples_path),
        output_root=str(tmp_path / "runs"),
        run_name="unit_test_run",
        submit_jobs=False,
        eval=EvalConfig(
            tool_dir=str(eval_tool_dir),
            tool_python=str(fake_python),
            config_path=None,
            metrics=["metric_a"],
            batch_size=2,
        ),
    )

    pipeline = EvalOnlyPipeline(cfg)
    pipeline.run()

    resolved_scenario = pipeline.resolved_scenario
    assert resolved_scenario.prompts_path == (prompts_dir / "en-ru.jsonl").resolve()
    assert resolved_scenario.prompt_key == "source"
    assert resolved_scenario.extra_data_fields == ["target"]
    assert resolved_scenario.limit_prompts == 3
    assert resolved_scenario.prompt_template == "Translate: {prompt}"
    assert resolved_scenario.system_prompt == "You are a translator."
    assert resolved_scenario.tgt == "ru"
    assert resolved_scenario.tgt_lang_name == "Russian"

    script_text = pipeline.eval_job_script.read_text(encoding="utf-8")
    assert str(samples_path) in script_text
    assert "--metric metric_a" in script_text
    assert "--batch-size 2" in script_text
    assert "mt_evaluation.cli" in script_text

    resolved_cfg = OmegaConf.load(pipeline.scenario_dump_path)
    assert str(resolved_cfg.prompts_path) == str((prompts_dir / "en-ru.jsonl").resolve())
    assert resolved_cfg.prompt_key == "source"
    assert list(resolved_cfg.extra_data_fields) == ["target"]


def test_chained_pipeline_writes_finetune_generation_and_eval_scripts(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "en-ru.jsonl").write_text('{"source":"hello"}\n', encoding="utf-8")

    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        '{"id":"1","src_lang":"eng_Latn","tgt_lang":"rus_Cyrl","src_text":"hello","tgt_text":"privet"}\n',
        encoding="utf-8",
    )

    eval_tool_dir = tmp_path / "eval_tool"
    (eval_tool_dir / "src").mkdir(parents=True)
    fake_python = eval_tool_dir / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_parallel",
                "backend: vllm",
                "sampling_mode: parallel",
                "model: base-scenario-model",
                "prompts_path: prompts/en-ru.jsonl",
                "output_dir: outputs",
                "vllm_host: 127.0.0.1",
                "vllm_port: 8000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sft_config_path = tmp_path / "exp_base.yaml"
    sft_config_path.write_text(
        "\n".join(
            [
                "seed: 13",
                "sft:",
                "  apptainer_image: /tmp/fake_llamafactory.sif",
                "  template_path: synth_data/configs/llama_factory/sft_template.yaml",
                "  output_dir_name: checkpoints",
                "  finetuning_type: lora",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        scenario=str(scenario_path),
        prompts_path=str(prompts_dir),
        tgt="ru",
        output_root=str(tmp_path / "runs"),
        run_name="chain_test_run",
        submit_jobs=False,
        enable_finetune=True,
        enable_generation=True,
        finetune=FinetuneConfig(
            enabled=True,
            config_path=str(sft_config_path),
            train_jsonl=str(train_jsonl),
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapter_name="ru_adapter",
        ),
        generation=GenerationConfig(
            enabled=True,
            model=None,
            core_python=".venv/bin/python",
        ),
        eval=EvalConfig(
            tool_dir=str(eval_tool_dir),
            tool_python=str(fake_python),
            config_path=None,
            metrics=["metric_a"],
            batch_size=2,
        ),
    )

    pipeline = EvalOnlyPipeline(cfg)
    pipeline.run()

    assert pipeline.generation_scenario is not None
    assert pipeline.generation_model_name == "ru_adapter"
    assert pipeline.served_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert pipeline.samples_path == pipeline.generation_scenario.output_path()
    assert pipeline.generation_scenario.model == "ru_adapter"

    finetune_script = pipeline.finetune_job_script.read_text(encoding="utf-8")
    assert "active_mt_distill.sft.llamafactory_runner" in finetune_script
    assert str(train_jsonl) in finetune_script
    assert "meta-llama/Llama-3.1-8B-Instruct" in finetune_script

    gen_script = pipeline.gen_job_script.read_text(encoding="utf-8")
    assert "generated vLLM command" in gen_script
    assert "vllm_server_runner.py" in gen_script
    assert "parseq_core.main" in gen_script
    assert "ru_adapter" in gen_script

    vllm_model_cfg = OmegaConf.load(pipeline.vllm_runtime_model_config)
    assert vllm_model_cfg.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert "--enable-lora" in list(vllm_model_cfg.vllm_server_args)
    assert "--lora-modules" in list(vllm_model_cfg.vllm_server_args)
    assert any(
        str(item).startswith("ru_adapter=") and str(pipeline.finetune_checkpoint_dir) in str(item)
        for item in vllm_model_cfg.vllm_server_args
    )

    eval_script = pipeline.eval_job_script.read_text(encoding="utf-8")
    assert str(pipeline.samples_path) in eval_script
    assert "mt_evaluation.cli" in eval_script
