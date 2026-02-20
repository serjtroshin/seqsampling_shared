# MT Evaluation Sub-Environment

Minimal standalone evaluator for MT outputs in this repo.

It reuses the existing report shape (`n`, `average_score`, `scores`) and writes results into a dedicated
`evaluation/` subfolder next to each `samples.jsonl` file:

- `.../parallel/evaluation/comet_qe.json`
- `.../parallel/evaluation/comet_kiwi_qe.json`
- `.../parallel/evaluation/xcomet_xxl.json`
- optional final-answer reports: `.../parallel/evaluation/comet_kiwi_qe.FA.json`

## 1) Create separate environment

```bash
cd evaluation/mt
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## 2) Quick run (preset metric)

```bash
mt-eval \
  --file /path/to/samples.jsonl \
  --metric comet_qe \
  --batch-size 8
```

By default this evaluates `generations`/`solutions` and writes into `<samples_dir>/evaluation`.

## 3) Run multiple metrics

```bash
mt-eval \
  --file /path/to/samples.jsonl \
  --metric comet_qe \
  --metric comet_kiwi_qe \
  --metric xcomet_xxl \
  --batch-size 8
```

## 4) Per-metric hyperparameters (JSON config)

Use `metrics.example.json` as a template:

```bash
mt-eval --file /path/to/samples.jsonl --config metrics.example.json
```

## 5) MetricX smoke test

```bash
./evaluation/mt/run_metricx_smoke_test.sh
```

This runs one MetricX metric on a tiny 2-row sample file and checks that a report
is written with numeric scores.

Config supports:
- abstract metric type (`comet_qe`, `comet_ref`)
- output name (`name`) -> `<output_dir>/<name>.json`
- model checkpoint (`model`)
- per-metric `batch_size`
- `include_final_answers`
- input `generation_keys`
- multi-metric configs are run in isolated sequential subprocesses by default (use `--no-isolate-metrics` to disable).
- `metricx24` type with optional fields: `tokenizer`, `max_input_length`, `qe`, `metricx_python`, `metricx_repo_dir`.

## Notes

- `comet_qe` uses `Unbabel/wmt22-comet-da` (reference-based).
- `comet_kiwi_qe` uses `Unbabel/wmt23-cometkiwi-da-xl` (reference-free).
- `xcomet_xxl` uses `Unbabel/XCOMET-XXL` (reference-based).
- `xcomet_xxl` reports `system_score` and `error_spans` in addition to segment scores.
- `xcomet_xxl` is memory-heavy; start with `batch_size: 1` in metric config.
- `metricx24_ref` / `metricx24_qe` run MetricX-24 through `evaluation/mt/metricx/.venv/bin/python`.
- MetricX scores are error scores (`direction: lower_is_better`).
- Reference-based metrics require references in input rows (`ref`, `reference`, `target`, or `gold`).
