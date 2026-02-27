from __future__ import annotations

"""FLORES devtest loader that writes canonical frozen eval JSONL."""

from pathlib import Path

from datasets import load_dataset  # type: ignore[import-untyped]

from active_mt_distill.datasets.schema import make_record
from active_mt_distill.io import write_jsonl


def build_flores_devtest(
    *,
    pairs: list[tuple[str, str]],
    out_path: Path,
    hf_cache_dir: str | None = None,
    limit_per_pair: int | None = None,
) -> int:
    # FLORES repository relies on remote dataset code, so this loader must opt in.
    dataset = load_dataset(
        "facebook/flores",
        "all",
        split="devtest",
        cache_dir=hf_cache_dir,
        trust_remote_code=True,
    )

    rows: list[dict] = []
    for src_lang, tgt_lang in pairs:
        src_field = f"sentence_{src_lang}"
        tgt_field = f"sentence_{tgt_lang}"
        pair_count = 0

        # Build a frozen per-pair eval set with deterministic ids.
        for row in dataset:
            src_text = str(row.get(src_field, "")).strip()
            tgt_text = str(row.get(tgt_field, "")).strip()
            if not src_text or not tgt_text:
                continue

            record = make_record(
                record_id=f"flores:devtest:{src_lang}-{tgt_lang}:{pair_count:06d}",
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=src_text,
                tgt_text=tgt_text,
                split="devtest",
                source="flores",
                iteration=0,
                strategy="frozen_eval",
            )
            rows.append(record)
            pair_count += 1
            if limit_per_pair is not None and pair_count >= limit_per_pair:
                break

        if pair_count == 0:
            raise ValueError(
                f"No rows found for FLORES pair {src_lang}-{tgt_lang}. "
                f"Expected fields '{src_field}' and '{tgt_field}'."
            )

    return write_jsonl(out_path, rows)
