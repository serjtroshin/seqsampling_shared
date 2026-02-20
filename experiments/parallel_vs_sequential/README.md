# Parallel vs Sequential Sweeps

Run from repo root and pass a model config path (this is the model you want to use):

```bash
bash experiments/parallel_vs_sequential/run_mt_multi_turn_sweep.sh configs/model_configs/<model>.yaml
bash experiments/parallel_vs_sequential/run_mt_multi_turn_plus_doc_sweep.sh configs/model_configs/<model>.yaml
bash experiments/parallel_vs_sequential/run_mt_parallel_sweep.sh configs/model_configs/<model>.yaml
```

Example:

```bash
bash experiments/parallel_vs_sequential/run_mt_multi_turn_sweep.sh configs/model_configs/gemma-3-4b-it.yaml
bash experiments/parallel_vs_sequential/run_mt_multi_turn_plus_doc_sweep.sh configs/model_configs/gemma-3-4b-it.yaml
```
