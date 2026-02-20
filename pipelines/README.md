# Single-Scenario MT Pipeline (SLURM)

This pipeline submits exactly two SLURM jobs:
1. Generation job
2. Evaluation job (with dependency on generation success)

The dependency is configurable (`eval_dependency`, default `afterok`).

## What It Does

- Runs one scenario only (single `scenario` in config)
- Writes `gen_job.sh` and `eval_job.sh` into a run folder
- Submits generation with `sbatch`
- Submits evaluation with `sbatch --dependency=<eval_dependency>:<gen_job_id>`
- For `backend: vllm`, generation starts a run-local vLLM server with node-safe port locking and writes `vllm_server.log`

## Prerequisites

1. Install both environments:

```bash
./scripts/install_envs.sh
```

2. Activate the core environment:

```bash
source .venv/bin/activate
```

3. Ensure `sbatch` is available.

4. If your scenario uses `backend: vllm`, make sure the model backend endpoint is reachable by the generation job (for example, shared vLLM service or your cluster-specific setup).

## Run

From `standalone/parallel-sequential-core`:

```bash
source .venv/bin/activate
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml
```

With overrides:

```bash
python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  --model-config configs/model_configs/qwen-3-4b-instruct-2507.yaml \
  --tgt zh \
  run_name=pta_qwen4b \
  slurm_gen.time=4:00:00 \
  slurm_eval.time=6:00:00
```

Qwen 32B Instruct (4 GPUs):

```bash
python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  --model-config configs/model_configs/qwen3-32b-instruct.yaml \
  --tgt ru
```

`--tgt` automatically sets `src=en`, `src_lang_name=English`, and `tgt_lang_name` using `pipelines/langauges.py` (for example `ru -> Russian`, `zh -> Chinese`).

Real-data flow (for folders under `data/wmt24p` and `data/wmttest2024_plus_doc`):

```bash
# 1) Prepare processed JSONL inputs
#    This processes ALL wmt24p language pairs when --lp is omitted.
python scripts/process_real_data.py --dataset wmt24p

#    Optional: process one specific pair instead:
# python scripts/process_real_data.py --dataset wmt24p --lp en-de_DE

#    For the doc dataset, process one specific pair:
python scripts/process_real_data.py --dataset wmttest2024_plus_doc --lp en-de

# 2) Submit pipeline on processed wmt24p file
#    Config default is tgt=de and prompts_path directory data/processed/wmt24p
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml

#    Example override target language (same config, different pair file)
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml --tgt ru

#    If multiple matches exist for one target (e.g. zh_CN + zh_TW), set explicit prompts_path:
# python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml --tgt zh \
#   prompts_path=data/processed/wmt24p/en-zh_CN.jsonl

# 3) Submit pipeline on processed wmttest doc file
python pipelines/pipeline.py --config pipelines/configs/wmt24p_doc.yaml
```

Dry-run mode (write scripts only, do not submit):

```bash
python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  submit_jobs=false
```

## Config

Base fields:
- `scenario`: single scenario YAML to run
- `tgt`: target language code (default `de`; source is always English)
- `model_config`: optional path to `configs/model_configs/*.yaml`; pipeline reads `model:` from that file
- `model`: optional model override
- `prompts_path`: optional prompt input (file path or directory; if directory, pipeline resolves `en-<tgt>*.jsonl`; if multiple matches exist, pass an explicit file path)
- `prompt_key`: prompt key when prompts are JSONL
- `extra_data_fields`: copied from prompts JSONL to outputs
- `num_generations`, `max_tokens`, `limit_prompts`: generation overrides
- `output_root`: root for run folders
- `run_name`: fixed run folder name (optional). If omitted, the pipeline auto-names runs as `<timestamp>_<scenario>_<dataset>_<language-pair>` (for example `20260220-101500_mt_multi_turn_please_translate_again_wmttest2024_plus_doc_en-de`).
- `scenario_overrides`: extra scenario dotlist overrides
- `core_python`: Python used for generation command (default `.venv/bin/python`)
- `submit_jobs`: if `false`, only writes job scripts
- `eval_dependency`: dependency type for eval submission (`afterok`, etc.)

Evaluation block (`eval`):
- `tool_dir`, `tool_python`: evaluation environment location
- `config_path`: optional evaluation config JSON
- `metrics`, `batch_size`: used when `config_path` is null

SLURM blocks:
- `slurm_gen`: resources for generation job
- `slurm_eval`: resources for evaluation job

Each SLURM block supports:
- `partition`
- `time`
- `gres`
- `cpus_per_task`
- `mem`
- `exclude`
- `additional_args` (extra raw `#SBATCH` lines)

Use quotes for `time` values in YAML (for example `"2:00:00"`), so they are not parsed as integers.

## Output Layout

Each run writes:
- `<output_root>/<run_name>/gen_job.sh`
- `<output_root>/<run_name>/eval_job.sh`
- `<output_root>/<run_name>/slurm_gen.out`
- `<output_root>/<run_name>/slurm_gen.err`
- `<output_root>/<run_name>/slurm_eval.out`
- `<output_root>/<run_name>/slurm_eval.err`
- `<output_root>/<run_name>/pipeline.log`
- `<output_root>/<run_name>/vllm_server.log` (when scenario backend is `vllm`)
- `<output_root>/<run_name>/vllm_port.txt` (selected/used port metadata)
- `<output_root>/<run_name>/vllm_generated_command.sh` (rendered vLLM launch command)
- generation artifacts under `<run>/generation/<scenario_name>/`
- evaluation artifacts under `<run>/evaluation/`
