from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from active_mt_distill.datasets.language_codes import language_name
from active_mt_distill.io import hash_file_sha256, read_jsonl


def _to_alpaca_row(row: dict[str, Any]) -> dict[str, str]:
    src_lang = str(row.get("src_lang", "")).strip()
    tgt_lang = str(row.get("tgt_lang", "")).strip()
    src_text = str(row.get("src_text", "")).strip()
    tgt_text = str(row.get("tgt_text", "")).strip()
    row_id = str(row.get("id", "")).strip()
    if not src_lang or not tgt_lang or not src_text or not tgt_text:
        raise ValueError(f"Row missing required fields for export: {row}")

    src_name = language_name(src_lang)
    tgt_name = language_name(tgt_lang)
    return {
        "id": row_id,
        "instruction": f"Translate from {src_name} to {tgt_name}.",
        "input": src_text,
        "output": tgt_text,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def export_for_llamafactory(
    *,
    train_jsonl: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    valid_ratio: float = 0.0,
    seed: int = 13,
) -> dict[str, Any]:
    in_path = Path(train_jsonl).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_jsonl(in_path)]
    if not rows:
        raise ValueError(f"No rows found in {in_path}.")

    alpaca_rows = [_to_alpaca_row(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(alpaca_rows)

    if not (0.0 <= valid_ratio < 1.0):
        raise ValueError("valid_ratio must be in [0.0, 1.0).")
    n_valid = int(len(alpaca_rows) * valid_ratio)
    valid_rows = alpaca_rows[:n_valid]
    train_rows = alpaca_rows[n_valid:]

    train_file = out_dir / "lf_train.json"
    valid_file = out_dir / "lf_valid.json"
    _write_json(train_file, train_rows)
    if valid_rows:
        _write_json(valid_file, valid_rows)

    train_dataset_name = dataset_name
    valid_dataset_name = f"{dataset_name}_valid"
    dataset_info: dict[str, Any] = {
        train_dataset_name: {
            "file_name": train_file.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    if valid_rows:
        dataset_info[valid_dataset_name] = {
            "file_name": valid_file.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }

    dataset_info_path = out_dir / "dataset_info.json"
    _write_json(dataset_info_path, dataset_info)

    pair_counts: Counter[str] = Counter()
    for row in rows:
        pair_counts[f"{row['src_lang']}-{row['tgt_lang']}"] += 1

    manifest = {
        "source_train_jsonl": str(in_path),
        "source_sha256": hash_file_sha256(in_path),
        "dataset_name": dataset_name,
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "pair_counts": dict(sorted(pair_counts.items())),
        "files": {
            "train": train_file.name,
            "valid": valid_file.name if valid_rows else None,
            "dataset_info": dataset_info_path.name,
        },
    }
    manifest_path = out_dir / "export_manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "export_dir": str(out_dir),
        "train_dataset_name": train_dataset_name,
        "valid_dataset_name": valid_dataset_name if valid_rows else None,
        "train_file": str(train_file),
        "valid_file": str(valid_file) if valid_rows else None,
        "dataset_info_file": str(dataset_info_path),
        "manifest_file": str(manifest_path),
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonical JSONL into LlamaFactory dataset files.")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Canonical training JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for LlamaFactory dataset files.")
    parser.add_argument("--dataset-name", type=str, default="active_mt_train", help="Dataset name in dataset_info.")
    parser.add_argument("--valid-ratio", type=float, default=0.0, help="Validation split ratio [0.0, 1.0).")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for split shuffle.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = export_for_llamafactory(
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

