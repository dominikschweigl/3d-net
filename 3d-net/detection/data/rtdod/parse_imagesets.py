#!/usr/bin/env python3
"""Parse RTDOD imagesets and report which video sequences and frame ranges appear in each split.

Output example:
  trainval  (12223 frames)
    video 0000 : frames 0–1499  (1500 frames)
    video 0001 : frames 0–899   ( 900 frames)
    ...
  test  (3967 frames)
    ...
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Summarise RTDOD imageset splits by video sequence.")
    parser.add_argument(
        "--imagesets-dir",
        type=Path,
        default=script_dir / "imagesets",
        help="Directory containing the .txt imageset files.",
    )
    return parser.parse_args()


def parse_imageset(path: Path) -> dict[str, list[int]]:
    """Return {video_id: sorted list of frame numbers} for every entry in the file.

    Filename format: <CONDITION>_<SEQNUM>_<FRAMENUM>
    where CONDITION can be multi-character (e.g. NR for Night Rain).
    video_id is everything except the last underscore-segment, e.g. "NR_0042".
    """
    frames_by_video: dict[str, list[int]] = defaultdict(list)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Split off the frame number (last segment); the rest is the video id.
        last_sep = line.rfind("_")
        if last_sep == -1:
            print(f"  [warn] unexpected line format: {line!r}")
            continue
        video_id = line[:last_sep]
        try:
            frame_num = int(line[last_sep + 1:])
        except ValueError:
            print(f"  [warn] non-integer frame number: {line!r}")
            continue
        frames_by_video[video_id].append(frame_num)
    return {vid: sorted(frames) for vid, frames in frames_by_video.items()}


def find_gaps(frames: list[int]) -> list[tuple[int, int]]:
    """Return list of (gap_start, gap_end) ranges missing from a sorted frame list."""
    gaps = []
    for i in range(len(frames) - 1):
        if frames[i + 1] != frames[i] + 1:
            gaps.append((frames[i] + 1, frames[i + 1] - 1))
    return gaps


def summarise(frames_by_video: dict[str, list[int]]) -> None:
    total = sum(len(f) for f in frames_by_video.values())
    print(f"  {len(frames_by_video)} video(s), {total} frames total")
    for video_id in sorted(frames_by_video):
        frames = frames_by_video[video_id]
        lo, hi = frames[0], frames[-1]
        pct = 100.0 * len(frames) / total if total else 0.0
        gaps = find_gaps(frames)
        gap_note = ""
        if gaps:
            gap_ranges = ", ".join(f"{a}–{b}" for a, b in gaps)
            gap_note = f"  *** {len(gaps)} gap(s): [{gap_ranges}] ***"
        print(f"    {video_id} : frames {lo:6d} – {hi:6d}  ({len(frames):5d} frames, {pct:5.1f}%){gap_note}")


def main() -> None:
    args = parse_args()
    txt_files = sorted(args.imagesets_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {args.imagesets_dir}")

    splits: dict[str, dict[str, list[int]]] = {}
    for txt_path in txt_files:
        splits[txt_path.stem] = parse_imageset(txt_path)

    grand_total = sum(
        len(frames)
        for frames_by_video in splits.values()
        for frames in frames_by_video.values()
    )

    print(f"\nDataset split  ({grand_total} frames total)")
    for split_name, frames_by_video in splits.items():
        split_total = sum(len(f) for f in frames_by_video.values())
        pct = 100.0 * split_total / grand_total if grand_total else 0.0
        print(f"  {split_name:<12} {split_total:6d} frames  ({pct:.1f}%)")

    for split_name, frames_by_video in splits.items():
        split_total = sum(len(f) for f in frames_by_video.values())
        pct = 100.0 * split_total / grand_total if grand_total else 0.0
        print(f"\n{split_name}  ({split_total} frames, {pct:.1f}% of dataset)")
        summarise(frames_by_video)


if __name__ == "__main__":
    main()
