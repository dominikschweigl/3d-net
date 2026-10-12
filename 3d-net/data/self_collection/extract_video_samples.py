"""
Extract paired RGB/IR frame samples from visible+thermal video pairs for annotation.

Videos are expected in a source directory with two subfolders:
  <source>/rgb/  — visible videos
  <source>/ir/   — thermal videos

Videos are paired by sorted order (1st rgb ↔ 1st ir, 2nd ↔ 2nd, …).
The rgb filename stem is used for output naming.

Every Nth frame is extracted and written to:
  <output_dir>/rgb/<stem>_<frame_idx:06d>.jpg
  <output_dir>/ir/<stem>_<frame_idx:06d>.jpg
"""

import argparse
import cv2
from pathlib import Path


def extract_pairs(source_dir: Path, output_dir: Path, frame_interval: int, offset: int = 3) -> None:
    rgb_videos = sorted((source_dir / "rgb").glob("*.mp4"))
    ir_videos = sorted((source_dir / "ir").glob("*.mp4"))

    if not rgb_videos:
        print(f"No .mp4 files found in {source_dir / 'rgb'}")
        return
    if not ir_videos:
        print(f"No .mp4 files found in {source_dir / 'ir'}")
        return
    if len(rgb_videos) != len(ir_videos):
        print(f"[warn] Unequal number of videos: {len(rgb_videos)} rgb vs {len(ir_videos)} ir — pairing by order up to the shorter list")

    out_rgb_dir = output_dir / "rgb"
    out_ir_dir = output_dir / "ir"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_ir_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(rgb_videos)} rgb and {len(ir_videos)} ir videos")

    total_saved = 0
    for vis_path, thermal_path in zip(rgb_videos, ir_videos):
        if vis_path.stem != thermal_path.stem:
            print(f"[skip] Stem mismatch: rgb '{vis_path.stem}' vs ir '{thermal_path.stem}' — skipping pair")
            continue
        stem = vis_path.stem
        saved = _extract_pair(vis_path, thermal_path, stem, out_rgb_dir, out_ir_dir, frame_interval, offset)
        total_saved += saved

    print(f"\nTotal frame pairs saved: {total_saved}")


def _extract_pair(
    vis_path: Path,
    thermal_path: Path,
    stem: str,
    rgb_dir: Path,
    ir_dir: Path,
    frame_interval: int,
    offset: int = 3,
) -> int:
    cap_vis = cv2.VideoCapture(str(vis_path))
    cap_ir = cv2.VideoCapture(str(thermal_path))

    total_vis = int(cap_vis.get(cv2.CAP_PROP_FRAME_COUNT))
    total_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vis = cap_vis.get(cv2.CAP_PROP_FPS)
    fps_ir = cap_ir.get(cv2.CAP_PROP_FPS)
    total = min(total_vis, total_ir - abs(offset))
    expected = total // frame_interval

    print(f"\n{'='*50}")
    print(f"Pair     : {stem}")
    print(f"Visible  : {total_vis} frames @ {fps_vis:.1f} fps  ({total_vis/fps_vis:.1f}s)  [{vis_path.name}]")
    print(f"Thermal  : {total_ir} frames @ {fps_ir:.1f} fps  ({total_ir/fps_ir:.1f}s)  [{thermal_path.name}]")
    print(f"Using    : {total} frames  (offset={offset:+d})")
    print(f"Interval : every {frame_interval} frames  →  ~{expected} pairs expected")

    if total == 0:
        print("[skip] Could not read frame count")
        cap_vis.release()
        cap_ir.release()
        return 0

    indices = range(0, total, frame_interval)
    saved = 0
    skipped = 0

    for frame_idx in indices:
        cap_vis.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_ir.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + offset)

        ret_vis, frame_vis = cap_vis.read()
        ret_ir, frame_ir = cap_ir.read()

        if not ret_vis or not ret_ir:
            skipped += 1
            continue

        name = f"{stem}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(rgb_dir / name), frame_vis)
        frame_ir_gray = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(ir_dir / name), frame_ir_gray)
        saved += 1

    cap_vis.release()
    cap_ir.release()

    print(f"Saved    : {saved} pairs  (skipped {skipped} unreadable frames)")
    return saved


def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Extract paired RGB/IR frame samples from video pairs")
    parser.add_argument(
        "--source",
        type=Path,
        default=here / "video" / "hausern_05_05_26",
        help="Directory containing rgb/ and ir/ subfolders with paired .mp4 videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "img",
        help="Output directory (will contain rgb/ and ir/ subdirs)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Save every Nth frame (default: 3)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Frame offset applied to IR video relative to RGB (raw frames, default: 0)",
    )
    args = parser.parse_args()

    print(f"Source   : {args.source}")
    print(f"Output   : {args.output}")
    print(f"Interval : every {args.interval} frames")
    print(f"Offset   : {args.offset:+d} frames (IR relative to RGB)")

    extract_pairs(args.source, args.output, args.interval, args.offset)
    print("\nDone.")


if __name__ == "__main__":
    main()
