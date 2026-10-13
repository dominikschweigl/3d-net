#!/usr/bin/env python3
"""Create train.txt and val.txt from trainval.txt.

For each video in trainval, the last ~val_fraction of its frames form a
contiguous validation segment; the earlier frames go to train.  This keeps
temporal coherence (train = earlier, val = later) and distributes val samples
across every condition/video present in trainval.

Usage:
    python create_train_val_split.py
    python create_train_val_split.py --val-fraction 0.10 --imagesets-dir imagesets/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Split trainval.txt into train.txt and val.txt.")
    parser.add_argument(
        "--imagesets-dir",
        type=Path,
        default=script_dir / "imagesets",
        help="Directory containing trainval.txt and where train.txt/val.txt will be written.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of trainval frames to put in val (default: 0.10).",
    )
    return parser.parse_args()


def parse_imageset(path: Path) -> dict[str, list[str]]:
    """Return {video_id: [sorted raw line strings]} preserving original format."""
    lines_by_video: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        last_sep = line.rfind("_")
        if last_sep == -1:
            continue
        video_id = line[:last_sep]
        lines_by_video[video_id].append(line)
    # Sort by frame number (last segment) within each video
    return {
        vid: sorted(lines, key=lambda l: int(l.rsplit("_", 1)[1]))
        for vid, lines in lines_by_video.items()
    }


def split_video(lines: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Return (train_lines, val_lines) by taking the last val_fraction as val."""
    n_val = max(1, round(len(lines) * val_fraction))
    n_val = min(n_val, len(lines))
    return lines[:-n_val], lines[-n_val:]


def main() -> None:
    args = parse_args()

    trainval_path = args.imagesets_dir / "trainval.txt"
    if not trainval_path.exists():
        raise FileNotFoundError(f"trainval.txt not found: {trainval_path}")

    lines_by_video = parse_imageset(trainval_path)
    total = sum(len(v) for v in lines_by_video.values())

    train_lines: list[str] = []
    val_lines: list[str] = []

    print(f"{'Video':<14}  {'Total':>6}  {'Train':>6}  {'Val':>6}  {'Val%':>6}  Val frame range")
    print("-" * 72)

    for video_id in sorted(lines_by_video):
        lines = lines_by_video[video_id]
        train, val = split_video(lines, args.val_fraction)
        train_lines.extend(train)
        val_lines.extend(val)

        val_frames = [int(l.rsplit("_", 1)[1]) for l in val]
        val_range = f"{val_frames[0]}–{val_frames[-1]}" if val_frames else "—"
        pct = 100.0 * len(val) / len(lines)
        print(
            f"{video_id:<14}  {len(lines):>6}  {len(train):>6}  {len(val):>6}  {pct:>5.1f}%  {val_range}"
        )

    print("-" * 72)
    pct_val = 100.0 * len(val_lines) / total if total else 0.0
    pct_train = 100.0 * len(train_lines) / total if total else 0.0
    print(f"{'TOTAL':<14}  {total:>6}  {len(train_lines):>6}  {len(val_lines):>6}  {pct_val:>5.1f}%")
    print(f"\nSplit: train={len(train_lines)} ({pct_train:.1f}%)  val={len(val_lines)} ({pct_val:.1f}%)")

    train_path = args.imagesets_dir / "train.txt"
    val_path = args.imagesets_dir / "val.txt"
    train_path.write_text("\n".join(train_lines) + "\n")
    val_path.write_text("\n".join(val_lines) + "\n")
    print(f"\nWrote {train_path}")
    print(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
