#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _available_wmt24p_pairs(data_root: Path) -> list[str]:
    return sorted(p.stem for p in (data_root / "wmt24p").glob("*.jsonl"))


def _available_doc_pairs(data_root: Path) -> list[str]:
    doc_root = data_root / "wmttest2024_plus_doc"
    return sorted(p.name for p in doc_root.iterdir() if p.is_dir() and p.name.startswith("en-"))


def _resolve_wmt24p_file(data_root: Path, lp: str) -> Path:
    wmt_root = data_root / "wmt24p"
    exact = wmt_root / f"{lp}.jsonl"
    if exact.exists():
        return exact

    # Allow short form like en-de -> en-de_DE.jsonl (if unique).
    candidates = sorted(wmt_root.glob(f"{lp}_*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous wmt24p pair '{lp}'. Matches: {[c.stem for c in candidates]}. "
            "Use full pair name (e.g. en-de_DE)."
        )
    raise FileNotFoundError(f"Could not find wmt24p file for pair '{lp}'.")


def _process_wmt24p_pair(data_root: Path, out_root: Path, lp: str, limit: int | None) -> tuple[Path, int]:
    in_path = _resolve_wmt24p_file(data_root, lp)
    out_path = out_root / "wmt24p" / in_path.name

    def rows():
        for idx, row in enumerate(_read_jsonl(in_path)):
            if limit is not None and idx >= limit:
                break
            source = str(row.get("source", "")).strip()
            target = str(row.get("target", "")).strip()
            if not source or not target:
                continue
            out = {
                "id": str(row.get("id", idx)),
                "lp": str(row.get("lp", in_path.stem)),
                "source": source,
                "target": target,
            }
            for key in ("domain", "document_id", "segment_id", "is_bad_source", "original_target"):
                if key in row:
                    out[key] = row[key]
            yield out

    return out_path, _write_jsonl(out_path, rows())


def _process_doc_pair(data_root: Path, out_root: Path, lp: str, limit: int | None) -> tuple[Path, int]:
    pair_dir = data_root / "wmttest2024_plus_doc" / lp
    if not pair_dir.exists():
        raise FileNotFoundError(f"Doc dataset pair directory not found: {pair_dir}")

    src_lang, tgt_lang = lp.split("-", maxsplit=1)
    src_path = pair_dir / f"test.{lp}.{src_lang}"
    tgt_path = pair_dir / f"test.{lp}.{tgt_lang}"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(f"Missing expected files for pair '{lp}': {src_path}, {tgt_path}")

    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
    n = min(len(src_lines), len(tgt_lines))
    if limit is not None:
        n = min(n, limit)

    out_path = out_root / "wmttest2024_plus_doc" / f"{lp}.jsonl"

    def rows():
        for idx in range(n):
            source = src_lines[idx].strip()
            target = tgt_lines[idx].strip()
            if not source or not target:
                continue
            yield {
                "id": str(idx),
                "lp": lp,
                "source": source,
                "target": target,
                "domain": "doc",
                "document_id": f"{lp}_doc",
                "segment_id": idx,
                "is_bad_source": False,
            }

    return out_path, _write_jsonl(out_path, rows())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare standalone real-data folders into pipeline-ready JSONL files "
            "with source/target fields."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["wmt24p", "wmttest2024_plus_doc", "all"],
        default="all",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--lp",
        action="append",
        default=None,
        help=(
            "Language pair to process (repeatable). "
            "Examples: en-de_DE for wmt24p, en-de for wmttest2024_plus_doc. "
            "If omitted, process all available pairs in selected dataset(s)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per processed pair.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root containing wmt24p/ and wmttest2024_plus_doc/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output root for processed JSONL files.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root if args.data_root.is_absolute() else (root / args.data_root)
    out_root = args.out_dir if args.out_dir.is_absolute() else (root / args.out_dir)

    requested = set(args.lp or [])

    to_run: list[tuple[str, str]] = []
    if args.dataset in {"wmt24p", "all"}:
        pairs = _available_wmt24p_pairs(data_root)
        if requested:
            # allow short names; we resolve later
            selected = sorted(requested)
        else:
            selected = pairs
            print(
                f"[info] dataset=wmt24p and no --lp provided; "
                f"processing all {len(selected)} language pairs."
            )
        to_run.extend(("wmt24p", lp) for lp in selected)
    if args.dataset in {"wmttest2024_plus_doc", "all"}:
        pairs = _available_doc_pairs(data_root)
        if requested:
            selected = sorted(lp for lp in requested if lp in pairs)
            # keep explicit values that may fail with useful error
            selected = sorted(set(selected) | {lp for lp in requested if lp not in pairs})
        else:
            selected = pairs
            print(
                f"[info] dataset=wmttest2024_plus_doc and no --lp provided; "
                f"processing all {len(selected)} language pairs."
            )
        to_run.extend(("wmttest2024_plus_doc", lp) for lp in selected)

    if not to_run:
        raise ValueError("No dataset pairs selected for processing.")

    print(f"[info] data_root={data_root}")
    print(f"[info] out_root={out_root}")
    print(f"[info] row_limit_per_pair={'all' if args.limit is None else args.limit}")

    processed = 0
    for dataset, lp in to_run:
        if dataset == "wmt24p":
            out_path, n_rows = _process_wmt24p_pair(data_root, out_root, lp, args.limit)
        else:
            out_path, n_rows = _process_doc_pair(data_root, out_root, lp, args.limit)
        processed += 1
        print(f"[ok] {dataset}:{lp} -> {out_path} ({n_rows} rows)")

    print(f"[done] processed {processed} pair(s).")


if __name__ == "__main__":
    main()
