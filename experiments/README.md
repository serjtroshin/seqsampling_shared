# Experiments

This folder contains runnable experiment helpers for the standalone project.

## Structure

- `parallel_vs_sequential/`:
  sweep launcher scripts.
  - `run_mt_parallel_sweep.sh`
  - `run_mt_multi_turn_sweep.sh`
- `mt/sweep_runner.py`:
  minimal FIFO sweep queue for MT pipeline runs.
- `mt/sweep_configs/example.yaml`:
  example sweep config for the MT sweep runner.

## MT Sweep Runner

`mt/sweep_runner.py` submits multiple `pipelines/pipeline.py` runs with a simple queue:

- FIFO order (no priorities).
- bounded concurrency (`runner.max_running`).
- status polling until each run is `DONE` or `FAILED`.

### Config shape

```yaml
name: wmt24p_example
base_config: pipelines/configs/wmt24p.yaml
experiment_dir: outputs/mt/pipeline_runs

runner:
  max_running: 2
  poll_seconds: 30
  python_bin: .venv/bin/python
  log_root: outputs/mt/sweeps
  slurm:
    partition: cpu
    time: 2-00:00:00
    cpus_per_task: 2
    mem: 16G
    exclude: null
    additional_args: []

global_overrides: []

sweeps:
  - ["model_config=configs/model_configs/qwen-3-4b-instruct-2507.yaml", "tgt=ru"]
  - ["model_config=configs/model_configs/qwen-3-4b-instruct-2507.yaml", "tgt=zh"]
```

### Run locally

From `standalone/parallel-sequential-core`:

```bash
source .venv/bin/activate
python experiments/mt/sweep_runner.py \
  --config experiments/mt/sweep_configs/example.yaml
```

### Submit CPU master runner job

Use `--submit` to submit one CPU SLURM job that runs the queue controller:

```bash
source .venv/bin/activate
python experiments/mt/sweep_runner.py \
  --config experiments/mt/sweep_configs/example.yaml \
  --work-dir outputs/mt/sweeps/example_submit_$(date +%Y%m%d-%H%M%S) \
  --submit
```

The master job script is written to `<work-dir>/runner_job.sh`.

### Scenario-specific launchers

From `standalone/parallel-sequential-core`:

```bash
./experiments/parallel_vs_sequential/run_mt_parallel_sweep.sh configs/model_configs/<model>.yaml
./experiments/parallel_vs_sequential/run_mt_multi_turn_sweep.sh configs/model_configs/<model>.yaml
```

Output folders are separated:

- `mt_parallel` sweep runs: `outputs/mt/pipeline_runs/<lang>/mt_parallel/<model>/...`
- multi-turn sweep runs: `outputs/mt/pipeline_runs/<lang>/mt_multi_turn_please_translate_again/<model>/...`
