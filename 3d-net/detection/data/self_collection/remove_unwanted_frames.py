#!/usr/bin/env python3
"""Delete unwanted/corrupted extracted frame pairs (RGB + IR) by aufnahme and frame-index ranges.

Filenames are expected to match: <anything>_<aufnahme>_<frame:06d>.<ext>
e.g.  hausern_05_05_26_aufnahme1_005763.jpg

Ranges are inclusive on both ends.  Already-deleted files are silently skipped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Ranges to delete — inclusive [start, end] frame indices per aufnahme
# ---------------------------------------------------------------------------
REMOVE_RANGES: dict[str, list[tuple[int, int]]] = {
    "aufnahme1": [
        (0, 5760),
        (9060, 9465),
        (9678, 9696),
        (17865, 19323),
    ],
    "aufnahme2": [
        (0, 645),
        (5145, 5262),
        (7032, 7434),
        (7536, 7629),
        (7683, 7698),
        (7935, 8013),
        (8103, 8124),
        (8202, 8220),
        (8247, 8295),
        (8409, 8427),
        (9003, 9090),
        (9159, 9225),
        (9405, 9429),
        (9543, 9588),
        (9687, 10416),
    ],
}

_FILENAME_RE = re.compile(r"^.+_(aufnahme\d+)_(\d+)\.[^.]+$")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def should_delete(name: str) -> bool:
    m = _FILENAME_RE.match(name)
    if m is None:
        return False
    aufnahme, frame_str = m.group(1), m.group(2)
    ranges = REMOVE_RANGES.get(aufnahme)
    if ranges is None:
        return False
    frame = int(frame_str)
    return any(lo <= frame <= hi for lo, hi in ranges)


def collect_deletions(rgb_dir: Path, ir_dir: Path) -> list[tuple[Path | None, Path | None]]:
    """Return paired (rgb_path, ir_path) — either may be None if already deleted."""
    stems: dict[str, dict[str, Path]] = {}  # stem -> {"rgb": path, "ir": path}

    for folder, key in ((rgb_dir, "rgb"), (ir_dir, "ir")):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if should_delete(p.name):
                entry = stems.setdefault(p.stem, {})
                entry[key] = p

    pairs: list[tuple[Path | None, Path | None]] = []
    for stem, paths in stems.items():
        pairs.append((paths.get("rgb"), paths.get("ir")))
    return sorted(pairs, key=lambda t: str(t[0] or t[1]))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Remove unwanted frame pairs by aufnahme/frame-index ranges.")
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=script_dir / "img",
        help="Root img dir containing rgb/ and ir/ subfolders (default: ./img).",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Explicit RGB directory (overrides --img-dir).",
    )
    parser.add_argument(
        "--ir-dir",
        type=Path,
        default=None,
        help="Explicit IR directory (overrides --img-dir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rgb_dir = args.rgb_dir if args.rgb_dir is not None else args.img_dir / "rgb"
    ir_dir  = args.ir_dir  if args.ir_dir  is not None else args.img_dir / "ir"

    print(f"Scanning RGB: {rgb_dir}")
    print(f"Scanning IR:  {ir_dir}")

    pairs = collect_deletions(rgb_dir, ir_dir)

    if not pairs:
        print("No matching files found.")
        return

    mode = "[DRY RUN] Would delete" if args.dry_run else "Deleting"
    deleted = 0
    for rgb_path, ir_path in pairs:
        for p in (rgb_path, ir_path):
            if p is None:
                continue
            print(f"  {mode}: {p}")
            if not args.dry_run:
                p.unlink()
                deleted += 1

    if args.dry_run:
        total = sum(
            (1 if rgb_path else 0) + (1 if ir_path else 0)
            for rgb_path, ir_path in pairs
        )
        print(f"\nDry run complete — {total} file(s) would be deleted across {len(pairs)} pairs.")
    else:
        print(f"\nDeleted {deleted} file(s) across {len(pairs)} pairs.")


if __name__ == "__main__":
    main()
