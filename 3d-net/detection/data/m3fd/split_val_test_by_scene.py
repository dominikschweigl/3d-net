#!/usr/bin/env python3
"""Split an image list into scene-wise validation/test halves.

Scenes are inferred from discontinuities in sequential image indices, e.g.
00010.png, 00011.png, 00012.png, 00030.png starts a new scene at 00030.png.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable


def parse_frame_index(filename: str) -> int:
    """Parse numeric stem from a filename like 00123.png."""
    return int(Path(filename).stem)


def split_into_scenes(image_names: Iterable[str]) -> list[list[str]]:
    """Group image names into scenes using non-consecutive indices as boundaries."""
    scenes: list[list[str]] = []
    current_scene: list[str] = []
    previous_idx: int | None = None

    for name in image_names:
        idx = parse_frame_index(name)

        if previous_idx is None or idx != previous_idx + 1:
            if current_scene:
                scenes.append(current_scene)
            current_scene = [name]
        else:
            current_scene.append(name)

        previous_idx = idx

    if current_scene:
        scenes.append(current_scene)

    return scenes


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Split a source list into scene-wise halves: first half -> val, "
            "second half -> test."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("test.txt"),
        help="Source list file to split (default: detection/data/m3fd/test.txt)",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path(__file__).with_name("val.txt"),
        help="Output path for validation list",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path(__file__).with_name("test.txt"),
        help="Output path for test list",
    )
    args = parser.parse_args()

    source_lines = [line.strip() for line in args.source.read_text().splitlines() if line.strip()]
    scenes = split_into_scenes(source_lines)

    val_lines: list[str] = []
    test_lines: list[str] = []
    for scene in scenes:
        split_idx = len(scene) // 2
        val_lines.extend(scene[:split_idx])
        test_lines.extend(scene[split_idx:])

    args.val_out.write_text("\n".join(val_lines) + "\n")
    args.test_out.write_text("\n".join(test_lines) + "\n")

    print(f"Scenes: {len(scenes)}")
    print(f"Validation images: {len(val_lines)} -> {args.val_out}")
    print(f"Test images: {len(test_lines)} -> {args.test_out}")


if __name__ == "__main__":
    main()
