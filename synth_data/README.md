# synth_data setup

This folder contains the first implementation step from `synth_data/plan.md`:

- dataset setup (frozen FLORES devtest + OPUS-100 train/candidate pools)
- LlamaFactory dataset export + apptainer SFT runner scaffolding

## Prerequisites

- active environment: `source .venv/bin/activate`
- optional dependency for dataset downloads:
  - `.venv/bin/uv pip install datasets`
- apptainer image path set in `synth_data/configs/exp_base.yaml`:
  - `sft.apptainer_image`
- user-specific storage overrides:
  - `synth_data/configs/users/stroshi.yaml`

## Storage layout

All real artifacts are written under:

- `base_data_folder`
- `cache_folder`
- `model_store_folder`

For `stroshi`, these are configured to `/fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/...`.

Local convenience paths are symlink directories under:

- `synth_data/outputs/processed`
- `synth_data/outputs/iterations`
- `synth_data/outputs/logs`
- `synth_data/outputs/cache`
- `synth_data/outputs/models`

## 1) Prepare datasets

```bash
bash synth_data/scripts/prepare_data.sh
```

This writes:

- `synth_data/outputs/processed/flores_devtest.jsonl`
- `synth_data/outputs/processed/opus_train_pool.jsonl`
- `synth_data/outputs/processed/candidate_pool.jsonl`

Smoke-test mode (small FLORES subset):

```bash
bash synth_data/scripts/prepare_data.sh synth_data/configs/exp_base.yaml --limit-flores-per-pair 32
```

## 2) Export to LlamaFactory format

```bash
bash synth_data/scripts/export_lf_dataset.sh \
  synth_data/outputs/processed/opus_train_pool.jsonl \
  synth_data/outputs/iterations/iter_00/lf_dataset \
  active_mt_train
```

## 3) Build or run LlamaFactory SFT command (apptainer)

Dry-run by default:

```bash
bash synth_data/scripts/run_llamafactory_sft.sh \
  synth_data/configs/exp_base.yaml \
  synth_data/outputs/processed/opus_train_pool.jsonl \
  meta-llama/Llama-3.1-8B-Instruct \
  synth_data/outputs/models/sft_runs/iter_00
```

Execute training:

```bash
bash synth_data/scripts/run_llamafactory_sft.sh \
  synth_data/configs/exp_base.yaml \
  synth_data/outputs/processed/opus_train_pool.jsonl \
  meta-llama/Llama-3.1-8B-Instruct \
  synth_data/outputs/models/sft_runs/iter_00 \
  run
```
