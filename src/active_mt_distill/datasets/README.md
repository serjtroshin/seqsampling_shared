# active_mt_distill.datasets

This package prepares the dataset artifacts used by the synthetic MT pipeline.

## Main logic

1. Load config (`exp_base.yaml` + optional `configs/users/$USER.yaml`).
2. Resolve storage paths (external artifact roots + local symlink views).
3. Set writable HF cache env vars.
4. Build frozen FLORES devtest set.
5. Build OPUS-100 train/candidate pools (disjoint by construction).
6. Write canonical JSONL rows with stable ids + provenance metadata.

## Files

- `prepare.py`: CLI orchestrator (`python -m active_mt_distill.datasets.prepare`)
- `flores.py`: creates frozen eval JSONL from `facebook/flores` devtest
- `opus100.py`: creates train/candidate pools from `Helsinki-NLP/opus-100`
- `schema.py`: canonical row builder (`make_record`)
- `language_codes.py`: FLORES code conversion + language display names

## Output schema

Each row has:

- `id`
- `src_lang`, `tgt_lang`
- `src_text`, `tgt_text`
- `split`
- `provenance`: `source`, `iter`, `strategy`, `parent_id`

## CLI usage

Prepare both FLORES + OPUS:

```bash
PYTHONPATH=src .venv/bin/python -m active_mt_distill.datasets.prepare \
  --config synth_data/configs/exp_base.yaml
```