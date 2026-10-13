"""
Investigate all images in img/ir subfolder and check that the grayscale
property holds (all channels are identical).
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def check_grayscale(ir_dir: Path) -> None:
    images = sorted(ir_dir.glob("*.jpg")) + sorted(ir_dir.glob("*.png"))

    if not images:
        print(f"No images found in {ir_dir}")
        return

    print(f"Checking {len(images)} images in {ir_dir}\n")

    failed = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[error] Could not read: {img_path.name}")
            continue

        if img.ndim == 2:
            # Already single-channel — fine
            continue

        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if not (np.array_equal(b, g) and np.array_equal(b, r)):
            max_bg = int(np.abs(b.astype(int) - g.astype(int)).max())
            max_br = int(np.abs(b.astype(int) - r.astype(int)).max())
            failed.append((img_path.name, max_bg, max_br))

    if not failed:
        print("All images pass the grayscale check.")
    else:
        print(f"{len(failed)} / {len(images)} images FAIL the grayscale check:\n")
        print(f"{'Filename':<50}  {'max |B-G|':>10}  {'max |B-R|':>10}")
        print("-" * 74)
        for name, max_bg, max_br in failed:
            print(f"{name:<50}  {max_bg:>10}  {max_br:>10}")


def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Check that all IR images in img/ir are truly grayscale"
    )
    parser.add_argument(
        "--ir-dir",
        type=Path,
        default=here / "img" / "ir",
        help="Path to the ir image directory (default: img/ir relative to this script)",
    )
    args = parser.parse_args()

    if not args.ir_dir.exists():
        print(f"Directory not found: {args.ir_dir}")
        return

    check_grayscale(args.ir_dir)


if __name__ == "__main__":
    main()
