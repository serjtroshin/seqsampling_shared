# parallel-sequential-core

Standalone core package for running parallel, iterative, enumeration, and multi-turn sampling scenarios.

## Installation (uv)

```bash
./scripts/install_envs.sh
```

Then activate the core environment:

```bash
source .venv/bin/activate
```

## Hugging Face auth (required for gated models)

Use the standard Hugging Face login flow once:

```bash
huggingface-cli login
# or: hf auth login
```

This saves your token at `~/.cache/huggingface/token`.

Sweep launchers and SLURM runner scripts in this repo auto-load:

```bash
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
```

if `HF_TOKEN` is not already set.

## Quick start (minimal example with vLLM server + client)
0. Make sure you activated the environment (`source .venv/bin/activate`).

1. Start vLLM (on a node):

```bash
parseq-core-vllm-server --scenario configs/example_vllm.yaml &
```

2. Run sampling:

```bash
parseq-core --scenario configs/example_parallel.yaml
```

3. Outputs are written under `outputs/`.

Note: to run multiple experiments it is nice to submit the job automatically via SLURM. TBD example.

## MT generation + evaluation example scripts (example)

Run generation (small demo override):

```bash
# Start vLLM server (if not already running):
parseq-core-vllm-server --scenario configs/example_vllm.yaml &
```

Then:

```bash
./scripts/01_generate_mt_example.sh
```

Run evaluation on generated outputs (using a free GPU):

```bash
./scripts/02_evaluate_mt_example.sh
```

These use:
- MT scenario: `configs/mt/mt_multi_turn_please_translate_again.yaml`
- MT demo prompts: `data/mt_demo_en_de.jsonl` (contains `source`/`target`)
- generated samples: `outputs/mt/en-de/multi_turn_please_translate_again_demo/samples_history1.jsonl`
- evaluation metric: `comet_kiwi_qe` via `evaluation/mt`
- model-specific runtime configs: `configs/model_configs/`
- vLLM runtime launch configs: `configs/vllm_config/`

## Model configs

Define model YAML files in `configs/model_configs/`, for example:

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
```

Pass it to the pipeline with `--model-config`:

```bash
python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  --model-config configs/model_configs/qwen-3-4b-instruct-2507.yaml
```

Qwen 32B Instruct (requires 4 GPUs):

```bash
python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  --model-config configs/model_configs/qwen3-32b-instruct.yaml \
  --tgt ru
```

`qwen3-32b-instruct.yaml` sets `ngpus: 4` and `--tensor-parallel-size 4`, so generation is submitted with 4 GPUs.


# Running a pipeline via SLURM jobs (generation + evaluation) -- most convenient for sweeps

Minimal SLURM flow (submits generation, then dependent evaluation):

```bash
./scripts/install_envs.sh
source .venv/bin/activate

python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  --model-config configs/model_configs/qwen-3-4b-instruct-2507.yaml
```

`--tgt` uses `pipelines/langauges.py` (for example `ru`, `zh`, `de`). Source language is always English, and default `tgt` is `de`.

## Process real data folders and run pipeline

### Process the data:

Prepare pipeline-ready JSONL files from both real data folders:

```bash
source .venv/bin/activate

# Folder 1: wmt24p (sentence-level JSONL already present; this normalizes fields)
python scripts/process_real_data.py --dataset wmt24p  # processes all language pairs

# Folder 2: wmttest2024_plus_doc (paired text files -> JSONL with source/target)
python scripts/process_real_data.py --dataset wmttest2024_plus_doc
```

### Run the pipeline for the sentence-level wmt24p data:

Then run the pipeline on the processed JSONL files. For `wmt24p`, the default target language is German (`de`), but you can switch it with `--tgt` (for example `ru` or `zh`).
Submit pipeline for processed `wmt24p` data (sentence-level translation):

```bash
source .venv/bin/activate
# Uses default tgt=de from config and auto-resolves:
# data/processed/wmt24p/en-de*.jsonl
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml

# Example: switch target language without changing prompts_path
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml --tgt ru

# If multiple files match one target (e.g. zh_CN and zh_TW), set explicit prompts_path:
# python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml --tgt zh \
#   prompts_path=data/processed/wmt24p/en-zh_CN.jsonl
```

### Run the pipeline for the doc-level wmt24p data:

Submit pipeline for processed `wmttest2024_plus_doc` data (document-level translation):
```bash
source .venv/bin/activate
python pipelines/pipeline.py --config pipelines/configs/wmt24p_doc.yaml
```

### Tips how to override the parameters:

Choose scenario (`multi_turn` vs `mt_parallel`):

```bash
source .venv/bin/activate
# Multi-turn scenario (default in pipeline configs)
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml \
  scenario=configs/mt/mt_multi_turn_please_translate_again.yaml

# Parallel MT scenario
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml \
  scenario=configs/mt/mt_parallel.yaml
```

Override sampling parameters (for example `temperature`, `top_p`):

```bash
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml \
  submit_jobs=false \
  run_name=dryrun_sampling \
  scenario_overrides='["temperature=0.3","top_p=0.9"]'
```

Override number of prompts for a quick run:

```bash
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml \
  limit_prompts=8
```

Override turns / fan-out:

```bash
# Multi-turn scenario:
# - num_generations = number of turns (k)
# - n_parallel = generations per turn (should be 1 by default)
# - history_k = keep last k turns in conversation history
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml \
  scenario=configs/mt/mt_multi_turn_please_translate_again.yaml \
  scenario_overrides='["num_generations=5","n_parallel=1","history_k=5"]'

# Parallel scenario:
# - num_generations = fan-out per prompt
# - n_parallel and history_k are ignored in this mode
python pipelines/pipeline.py --config pipelines/configs/wmt24p.yaml \
  scenario=configs/mt/mt_parallel.yaml \
  scenario_overrides='["num_generations=12"]'
```

Note: `history_k` and `n_parallel` can be passed to `mt_parallel`, but that scenario is parallel-only and uses only `num_generations`.



# Running a pipeline with a scenario on a pre-allocated GPU node -- useful for debugging or if you have a running vLLM service

Use this if you already have an interactive GPU allocation and want to run jobs directly on that node:

```bash
./scripts/install_envs.sh
source .venv/bin/activate

# 1) Start vLLM in terminal A
VLLM_CMD=$(python configs/vllm_config/vllm_server_runner.py \
  --model-config configs/model_configs/qwen-3-4b-instruct-2507.yaml \
  --vllm-config configs/vllm_config/local_python.yaml)
bash -lc "$VLLM_CMD"

# 2) In terminal B: write pipeline scripts only
python pipelines/pipeline.py \
  --config pipelines/configs/wmt24p.yaml \
  --model-config configs/model_configs/qwen-3-4b-instruct-2507.yaml \
  run_name=please_translate_again_qwen4b \
  submit_jobs=false

# 3) Run scripts directly on the node (no sbatch)
bash outputs/mt/pipeline_runs/please_translate_again_qwen4b/gen_job.sh
bash outputs/mt/pipeline_runs/please_translate_again_qwen4b/eval_job.sh
```



## Reuse from another project

In a consumer project `pyproject.toml`:

```toml
dependencies = [
  "parallel-sequential-core",
]
```

Then install from GitHub:

```bash
uv pip install "parallel-sequential-core @ git+https://github.com/<you>/parallel-sequential-core.git"
```
