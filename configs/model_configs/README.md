# Model-specific vLLM configs

Place YAML files here keyed by model slug (any filename). Each file can set:

```yaml
model: google/gemma-3-12b-it
ngpus: 1
vllm_runtime_config: configs/vllm_config/local_python.yaml
vllm_server_args:
  - "--max-model-len"
  - "8192"
  - "--gpu-memory-utilization"
  - "0.95"
  - "--max-num-batched-tokens"
  - "4096"
scenario_override:
  temperature: 0.7
  top_p: 0.8
```

The pipeline will load the file whose `model` matches the requested model string, apply `vllm_server_args` to the scenario, select vLLM runtime config via `vllm_runtime_config`, and pass `scenario_override` as CLI overrides when running `parseq`. If no matching file is found, the scenario's own `vllm_server_args` are used unchanged.

Notes:
- You can select a model config in two ways:
  - pass full model id (matches YAML `model:`), e.g. `model=Qwen/Qwen3-32b`
  - pass config key (matches filename stem), e.g. `model=qwen3-32b-thinking` for `qwen3-32b-thinking.yaml`
- When selected by config key, runtime still uses the YAML `model:` value for vLLM/parseq requests.
- `vllm_runtime_config` points to a file under `configs/vllm_config/` and controls whether that model launches via local Python or Apptainer.
- In this monorepo, `configs/vllm_config/` remains available as a compatibility symlink to the standalone location.
- `ngpus` (optional, default pipeline value `1`) controls GPU count used by the generation SLURM job (`#SBATCH --gres=...`) and default vLLM `--tensor-parallel-size` injection when `ngpus > 1`.
- If pipeline-level `ngpus` is set to a non-default value, it takes precedence over model config `ngpus`.
- `scenario_override` only applies to keys that are not already set explicitly by the pipeline (e.g., `model`, `num_generations`, `output_dir`). For those, pipeline values win.
- Unknown `scenario_override` keys are ignored with a warning in the pipeline log.
