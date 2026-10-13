#!/usr/bin/env python3
"""Confirm numeric coverage across split text files.

By default, validates `train.txt`, `val.txt`, and `test.txt` in this directory.
The script reports:
- total entries and unique indices
- duplicate indices (within/across files)
- missing indices in the full [min, max] numeric range
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_index(name: str) -> int:
    return int(Path(name).stem)


def load_indices(path: Path) -> list[int]:
    values: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        values.append(parse_index(line))
    return values


def format_preview(values: Iterable[int], max_items: int = 20) -> str:
    seq = sorted(values)
    if not seq:
        return "[]"
    if len(seq) <= max_items:
        return str(seq)
    head = ", ".join(map(str, seq[: max_items // 2]))
    tail = ", ".join(map(str, seq[-(max_items // 2) :]))
    return f"[{head}, ..., {tail}]"


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = ArgumentParser(description="Validate numeric coverage across split files")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train.txt", "val.txt", "test.txt"],
        help="Split files to validate (default: train.txt val.txt test.txt)",
    )
    args = parser.parse_args()

    paths = [Path(p) if Path(p).is_absolute() else base_dir / p for p in args.files]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    all_values: list[int] = []
    per_file_counts: dict[Path, Counter[int]] = {}

    for path in paths:
        indices = load_indices(path)
        per_file_counts[path] = Counter(indices)
        all_values.extend(indices)
        print(f"{path.name}: {len(indices)} entries, {len(set(indices))} unique")

    if not all_values:
        print("No entries found across provided files.")
        return

    combined_counts = Counter(all_values)
    unique_values = set(all_values)
    min_idx, max_idx = min(unique_values), max(unique_values)

    full_range = set(range(min_idx, max_idx + 1))
    missing = sorted(full_range - unique_values)
    duplicates = sorted(idx for idx, count in combined_counts.items() if count > 1)

    print("\nCombined summary")
    print(f"Total entries: {len(all_values)}")
    print(f"Unique indices: {len(unique_values)}")
    print(f"Index range: {min_idx}..{max_idx}")

    if missing:
        print(f"Missing indices: {len(missing)}")
        print(f"Missing preview: {format_preview(missing)}")
    else:
        print("Missing indices: 0 (full range is covered)")

    if duplicates:
        print(f"Duplicate indices: {len(duplicates)}")
        print(f"Duplicate preview: {format_preview(duplicates)}")
    else:
        print("Duplicate indices: 0")

    overlap_pairs: list[str] = []
    for idx in duplicates:
        owners = [p.name for p, cnt in per_file_counts.items() if cnt.get(idx, 0) > 0]
        overlap_pairs.append(f"{idx}: {', '.join(owners)}")

    if overlap_pairs:
        print("\nOverlaps by file:")
        for row in overlap_pairs[:30]:
            print(row)
        if len(overlap_pairs) > 30:
            print(f"... and {len(overlap_pairs) - 30} more")


if __name__ == "__main__":
    main()
