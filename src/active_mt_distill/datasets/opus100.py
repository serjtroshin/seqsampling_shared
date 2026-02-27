from __future__ import annotations

from pathlib import Path
from typing import Any

from active_mt_distill.datasets.language_codes import flores_to_iso2
from active_mt_distill.datasets.schema import make_record
from active_mt_distill.io import write_jsonl


def _load_dataset_fn():
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Missing optional dependency 'datasets'. "
            "Install it first, for example: .venv/bin/uv pip install datasets"
        ) from exc
    return load_dataset


def _try_load_opus_split(
    load_dataset,
    *,
    src_iso2: str,
    tgt_iso2: str,
    hf_cache_dir: str | None,
):
    attempts = [f"{src_iso2}-{tgt_iso2}", f"{tgt_iso2}-{src_iso2}"]
    errors: list[str] = []

    for config_name in attempts:
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", config_name, split="train", cache_dir=hf_cache_dir)
            return dataset, config_name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{config_name}: {exc}")

    details = "\n".join(errors)
    raise RuntimeError(
        "Could not load OPUS-100 split for "
        f"{src_iso2}-{tgt_iso2}. Tried:\n{details}"
    )


def _extract_parallel_text(row: dict[str, Any], src_iso2: str, tgt_iso2: str) -> tuple[str, str]:
    translation = row.get("translation", {})
    if not isinstance(translation, dict):
        return "", ""
    src_text = str(translation.get(src_iso2, "")).strip()
    tgt_text = str(translation.get(tgt_iso2, "")).strip()
    return src_text, tgt_text


def build_opus_pools(
    *,
    pairs: list[tuple[str, str]],
    train_out_path: Path,
    candidate_out_path: Path,
    train_size_per_pair: int,
    candidate_size_per_pair: int,
    seed: int,
    hf_cache_dir: str | None = None,
) -> tuple[int, int]:
    load_dataset = _load_dataset_fn()

    train_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for pair_idx, (src_lang, tgt_lang) in enumerate(pairs):
        src_iso2 = flores_to_iso2(src_lang)
        tgt_iso2 = flores_to_iso2(tgt_lang)

        dataset, config_name = _try_load_opus_split(
            load_dataset,
            src_iso2=src_iso2,
            tgt_iso2=tgt_iso2,
            hf_cache_dir=hf_cache_dir,
        )

        pair_seed = seed + pair_idx * 10_003
        shuffled = dataset.shuffle(seed=pair_seed)

        n_train = 0
        n_candidate = 0
        needed_train = train_size_per_pair
        needed_candidate = candidate_size_per_pair

        for row in shuffled:
            src_text, tgt_text = _extract_parallel_text(row, src_iso2, tgt_iso2)
            if not src_text or not tgt_text:
                continue

            if n_train < needed_train:
                record = make_record(
                    record_id=f"opus100:train:{src_lang}-{tgt_lang}:{n_train:06d}",
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_text=src_text,
                    tgt_text=tgt_text,
                    split="train",
                    source="opus100",
                    iteration=0,
                    strategy=f"base_pool:{config_name}",
                )
                train_rows.append(record)
                n_train += 1
                continue

            if n_candidate < needed_candidate:
                record = make_record(
                    record_id=f"opus100:candidate:{src_lang}-{tgt_lang}:{n_candidate:06d}",
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_text=src_text,
                    tgt_text=tgt_text,
                    split="candidate",
                    source="opus100",
                    iteration=0,
                    strategy=f"candidate_pool:{config_name}",
                )
                candidate_rows.append(record)
                n_candidate += 1

            if n_train >= needed_train and n_candidate >= needed_candidate:
                break

        if n_train < needed_train or n_candidate < needed_candidate:
            raise ValueError(
                f"Insufficient OPUS rows for {src_lang}-{tgt_lang} "
                f"(loaded config {config_name}): got train={n_train}/{needed_train}, "
                f"candidate={n_candidate}/{needed_candidate}"
            )

    train_count = write_jsonl(train_out_path, train_rows)
    candidate_count = write_jsonl(candidate_out_path, candidate_rows)
    return train_count, candidate_count

