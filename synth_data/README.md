# synth_data

Minimal MT data + training utilities for this project:

- prepare frozen eval + training pools
- run LlamaFactory SFT in Apptainer

## Installation

```bash
bash scripts/install_envs.sh
source .venv/bin/activate
```

If you train/evaluate gated Hugging Face models:

```bash
huggingface-cli login
```

## Data model

After data prep you get three files:

- `flores_devtest.jsonl`: fixed evaluation set (do not train on this)
- `opus_train_pool.jsonl`: base SFT training pool
- `candidate_pool.jsonl`: pool for future active selection steps

## Storage model

Paths are driven by:

- base config: `synth_data/configs/exp_base.yaml`
- user override: `synth_data/configs/users/<user>.yaml`

For `stroshi`, artifacts are stored under:

- `/fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data`

Local project paths under `synth_data/outputs/` and `synth_data/data/` are symlinks to that external storage.

## Quickstart

1) Prepare datasets:

```bash
bash synth_data/scripts/prepare_data.sh
```

2) Pull LlamaFactory image into Apptainer SIF:

```bash
bash synth_data/scripts/pull_llamafactory_apptainer.sh
```

3) Run SFT command dry-run (writes config/export files, prints train command):

```bash
bash synth_data/scripts/run_llamafactory_sft.sh
```

4) Run real training:

```bash
bash synth_data/scripts/run_llamafactory_sft.sh \
  synth_data/configs/exp_base.yaml \
  synth_data/data/processed/opus_train_pool.jsonl \
  meta-llama/Llama-3.1-8B-Instruct \
  synth_data/data/iterations/iter_00 \
  run
```

## GPU smoke test

On a GPU machine:

```bash
bash synth_data/scripts/test_llamafactory_gpu.sh
```

Tiny real train smoke test:

```bash
bash synth_data/scripts/test_llamafactory_gpu.sh --run-train
```


### Data storage:
Currently we use symlinks for the outputs folder:
```bash
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 55 Feb 27 15:06 base_data_folder -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 62 Feb 27 15:06 cache -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/.cache
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 66 Feb 27 15:06 iterations -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/iterations
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 60 Feb 27 15:06 logs -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/logs
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 62 Feb 27 15:06 models -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/models
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 65 Feb 27 15:06 processed -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/processed
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 71 Feb 27 15:09 sft_runs -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/models/sft_runs
```

And for the data:
```bash
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 66 Feb 27 15:10 iterations -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/iterations
lrwxrwxrwx 1 stroshi ivi-ilps-unix-fnwi 65 Feb 27 15:10 processed -> /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/processed
```