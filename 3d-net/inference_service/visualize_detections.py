#!/usr/bin/env python3
"""Visualize inference detections on M3FD or 3D-Net dataset images and save to disk.

Reads the ``inference_detections.csv`` produced by PerfMonitor, maps each
frame_id back to the corresponding visible/IR image pair, draws the detection
bounding boxes, and saves a composite PNG per frame.

Output layout per image:
- Top:          VIS + IR overlay with detections
- Bottom-left:  VIS with detections
- Bottom-right: IR  with detections

Usage:
    # M3FD dataset (default)
    python -m detection.inference_service.m3fd_visualize_detections --dataset m3fd
    python -m detection.inference_service.m3fd_visualize_detections \
        --dataset m3fd \
        --detections-csv logs/inference_detections.csv \
        --dataset-root detection/data/m3fd \
        --save-dir logs/detection_frames \
        --splits train

    # 3D-Net dataset
    python -m detection.inference_service.m3fd_visualize_detections --dataset 3d_net
    python -m detection.inference_service.m3fd_visualize_detections \
        --dataset 3d_net \
        --dataset-root detection/data/3d_net \
        --videos hausern_20_01_26_aufnahme5
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Colour palette (BGR) ─────────────────────────────────────────────────────

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person":     (80, 220, 80),
    "car":        (40, 180, 255),
    "bus":        (0, 128, 255),
    "motorcycle": (220, 80, 220),
    "bicycle":    (255, 200, 50),
    "truck":      (80, 80, 255),
}

_DEFAULT_COLOR = (255, 255, 255)

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"

# Suffix pattern shared with ThreeDNetFrameProvider: '_NNNNNN'
_FRAME_SUFFIX_LEN = 7


# ── Data loading ─────────────────────────────────────────────────────────────

def load_m3fd_filenames(dataset_root: Path, splits: List[str]) -> List[str]:
    """Read M3FD split .txt files and return a deduplicated ordered filename list."""
    seen: set = set()
    filenames: List[str] = []
    for split in splits:
        txt = dataset_root / f"{split}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Split file not found: {txt}")
        for line in txt.read_text().splitlines():
            name = line.strip()
            if name and name not in seen:
                seen.add(name)
                filenames.append(name)
    return filenames


def load_3d_net_filenames(
    dataset_root: Path,
    videos: Optional[List[str]] = None,
) -> List[str]:
    """Scan the 3D-Net rgb/ directory and return a sorted list of filenames.

    Mirrors ThreeDNetFrameProvider: groups frames by video name (the stem
    prefix before the trailing '_NNNNNN' frame-number suffix), optionally
    filters by video name, and flattens in video-then-frame order.
    """
    rgb_dir = dataset_root / "rgb"
    ir_dir  = dataset_root / "ir"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR directory not found: {ir_dir}")

    video_frames: Dict[str, List[str]] = defaultdict(list)
    for path in sorted(rgb_dir.iterdir()):
        if not path.is_file():
            continue
        if not (ir_dir / path.name).exists():
            continue
        stem = path.stem
        if len(stem) <= _FRAME_SUFFIX_LEN:
            continue
        video_frames[stem[:-_FRAME_SUFFIX_LEN]].append(path.name)

    available = set(video_frames.keys())
    if videos is not None:
        unknown = set(videos) - available
        if unknown:
            raise ValueError(
                f"Unknown video(s): {sorted(unknown)}. Available: {sorted(available)}"
            )
        selected = {v: video_frames[v] for v in videos}
    else:
        selected = video_frames

    return [
        name
        for video in sorted(selected.keys())
        for name in selected[video]
    ]


def load_detections(csv_path: Path) -> Dict[str, List[dict]]:
    """Load detections CSV and group rows by frame_id."""
    frames: Dict[str, List[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames[row["frame_id"]].append(row)
    return frames


def frame_idx_from_id(frame_id: str) -> int:
    """Extract the integer frame index from a frame_id like '20260414_114357_0000042'."""
    return int(frame_id.rsplit("_", 1)[-1])


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_detections(image_bgr: np.ndarray, dets: List[dict]) -> np.ndarray:
    """Draw detection boxes with class name, confidence and track ID on a BGR image."""
    out = image_bgr.copy()
    for det in dets:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        name = det["class_name"]
        conf = float(det["confidence"])
        tid  = det.get("track_id", "")

        color = CLASS_COLORS_BGR.get(name, _DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"T{tid} {name} {conf:.2f}" if tid != "" else f"{name} {conf:.2f}"
        (_, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_y = y1 - 6 if y1 > th + 6 else y1 + th + 6
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def make_overlay(vis_bgr: np.ndarray, ir_gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend visible image with a colormapped IR image."""
    ir_color = cv2.applyColorMap(ir_gray, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(vis_bgr, 1.0 - alpha, ir_color, alpha, 0.0)


def make_composite(
    frame_id: str,
    filename: str,
    vis_bgr: np.ndarray,
    ir_gray: np.ndarray,
    dets: List[dict],
    alpha: float,
) -> np.ndarray:
    """
    Compose a single output image:
      top row    — full-width overlay with detections
      bottom row — VIS (left) and IR (right) at half width, both with detections
    """
    h, w = vis_bgr.shape[:2]
    half_w, half_h = w // 2, h // 2

    overlay_bgr = make_overlay(vis_bgr, ir_gray, alpha)

    overlay_drawn = draw_detections(overlay_bgr, dets)
    vis_drawn     = draw_detections(vis_bgr, dets)
    ir_drawn      = draw_detections(cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR), dets)

    top = overlay_drawn
    bottom_vis = cv2.resize(vis_drawn, (half_w, half_h))
    bottom_ir  = cv2.resize(ir_drawn,  (half_w, half_h))
    bottom = np.concatenate([bottom_vis, bottom_ir], axis=1)

    for text, img, x, y in [
        ("VIS + detections", bottom, 6, 18),
        ("IR + detections",  bottom, half_w + 6, 18),
    ]:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    header = f"{frame_id}  |  {filename}  |  {len(dets)} det"
    cv2.putText(top, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return np.concatenate([top, bottom], axis=0)


# ── Save loop ─────────────────────────────────────────────────────────────────

def save_frames(
    frame_ids: List[str],
    detections: Dict[str, List[dict]],
    filenames: List[str],
    vis_dir: Path,
    ir_dir: Path,
    save_dir: Path,
    alpha: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    n_images = len(filenames)
    n_frames = len(frame_ids)

    for i, frame_id in enumerate(frame_ids):
        frame_idx = frame_idx_from_id(frame_id)
        filename  = filenames[(frame_idx - 1) % n_images]
        dets      = detections.get(frame_id, [])

        vis_bgr = cv2.imread(str(vis_dir / filename), cv2.IMREAD_COLOR)
        ir_raw  = cv2.imread(str(ir_dir  / filename), cv2.IMREAD_ANYDEPTH)

        if vis_bgr is None or ir_raw is None:
            print(f"[WARN] could not load {filename} — skipping {frame_id}")
            continue

        ir_u8 = ir_raw if ir_raw.dtype == np.uint8 else cv2.normalize(
            ir_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        if ir_u8.ndim == 3:
            ir_u8 = cv2.cvtColor(ir_u8, cv2.COLOR_BGR2GRAY)

        composite = make_composite(frame_id, filename, vis_bgr, ir_u8, dets, alpha)

        out_path = save_dir / f"{frame_id}.png"
        cv2.imwrite(str(out_path), composite)
        print(f"[{i+1:>{len(str(n_frames))}}/{n_frames}] {out_path.name}  ({len(dets)} det)")

    print(f"\nSaved {n_frames} frames to {save_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save annotated detection frames to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["m3fd", "3d_net"],
        default="m3fd",
        help="Dataset to visualize. Default: m3fd.",
    )
    parser.add_argument(
        "--detections-csv",
        type=Path,
        default=None,
        help=(
            "Path to inference_detections.csv. "
            "Default: logs/inference_detections.csv next to this file."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Path to the dataset root directory. "
            "Default: detection/data/m3fd  (--dataset m3fd) or "
            "detection/data/3d_net  (--dataset 3d_net)."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save output PNGs. Default: logs/detection_frames/ next to this file.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="[m3fd] Split(s) used during inference (default: train).",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        metavar="VIDEO",
        help="[3d_net] Video name(s) to include (default: all).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="IR overlay blend alpha in [0, 1]. Default: 0.45.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Fill in path defaults that depend on the chosen dataset.
    if args.detections_csv is None:
        args.detections_csv = _HERE / "logs" / "inference_detections.csv"
    if args.save_dir is None:
        args.save_dir = _HERE / "logs" / "detection_frames"
    if args.dataset_root is None:
        args.dataset_root = _DATA / ("m3fd" if args.dataset == "m3fd" else "3d_net")

    if not args.detections_csv.exists():
        raise FileNotFoundError(f"Detections CSV not found: {args.detections_csv}")

    detections = load_detections(args.detections_csv)
    frame_ids  = sorted(detections.keys())

    if args.dataset == "m3fd":
        filenames = load_m3fd_filenames(args.dataset_root, args.splits)
        vis_dir = args.dataset_root / "Vis"
        ir_dir  = args.dataset_root / "Ir"
        dataset_desc = f"M3FD: {len(filenames)} images from split(s) {args.splits}"
    else:
        filenames = load_3d_net_filenames(args.dataset_root, args.videos)
        vis_dir = args.dataset_root / "rgb"
        ir_dir  = args.dataset_root / "ir"
        videos_label = str(args.videos) if args.videos else "all videos"
        dataset_desc = f"3D-Net: {len(filenames)} frames from {videos_label}"

    total_dets = sum(len(v) for v in detections.values())
    print(f"Loaded {total_dets} detections across {len(frame_ids)} frames from {args.detections_csv}")
    print(f"Dataset: {dataset_desc}")
    print(f"Saving to {args.save_dir}\n")

    save_frames(
        frame_ids=frame_ids,
        detections=detections,
        filenames=filenames,
        vis_dir=vis_dir,
        ir_dir=ir_dir,
        save_dir=args.save_dir,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
