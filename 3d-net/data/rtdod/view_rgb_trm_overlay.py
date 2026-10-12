#!/usr/bin/env python3
"""Simple RTDOD RGB/Thermal overlay viewer.

Pairs images by filename stem from `rgb/` and `trm/`, then displays:
- Top: RGB + Thermal overlay
- Bottom-left: RGB image
- Bottom-right: Thermal image

Controls:
- Space / Right Arrow: next image
- Backspace / Left Arrow: previous image
- Q / Esc: quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CLASS_COLORS_BGR = {
    "person":        (80, 220, 80),
    "car":           (40, 180, 255),
    "bus":           (0, 128, 255),
    "truck":         (80, 80, 255),
    "motorcycle":    (220, 80, 220),
    "bicycle":       (255, 200, 40),
    "boat":          (40, 220, 220),
    "dog":           (160, 100, 40),
    "sports ball":   (255, 80, 80),
    "baby carriage": (200, 255, 160),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="View RTDOD RGB/Thermal overlays paired by filename.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=script_dir,
        help="Path to RTDOD root containing rgb/ and trm/.",
    )
    parser.add_argument("--rgb-dir", type=Path, default=None, help="Optional explicit RGB directory.")
    parser.add_argument("--trm-dir", type=Path, default=None, help="Optional explicit thermal directory.")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Optional explicit annotation directory containing Pascal VOC XML files.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Thermal overlay alpha in [0, 1].")
    parser.add_argument("--start-index", type=int, default=0, help="Start pair index.")
    return parser.parse_args()


def list_images(folder: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    }


def list_xmls(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Annotation dir not found: {folder}")
    return {p.stem: p for p in sorted(folder.glob("*.xml")) if p.is_file()}


def build_samples(
    rgb_dir: Path,
    trm_dir: Path,
    annotation_dir: Path,
) -> list[tuple[Path, Path, Path | None]]:
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if not trm_dir.exists():
        raise FileNotFoundError(f"Thermal dir not found: {trm_dir}")

    rgb_images = list_images(rgb_dir)
    trm_images = list_images(trm_dir)
    xmls = list_xmls(annotation_dir)

    common_stems = sorted(set(rgb_images).intersection(trm_images))
    return [(rgb_images[s], trm_images[s], xmls.get(s)) for s in common_stems]


def parse_annotation(xml_path: Path) -> tuple[int, int, list[tuple[str, tuple[float, float, float, float]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid <size> in {xml_path}")

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="Unknown").strip() or "Unknown"
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            x_min = float(bnd.findtext("xmin", default="0"))
            y_min = float(bnd.findtext("ymin", default="0"))
            x_max = float(bnd.findtext("xmax", default="0"))
            y_max = float(bnd.findtext("ymax", default="0"))
        except ValueError:
            continue

        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append((label, (x_min, y_min, x_max, y_max)))

    return width, height, boxes


def draw_bboxes(
    image_bgr: np.ndarray,
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    src_width: int,
    src_height: int,
) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("draw_bboxes expects a BGR image")

    out = image_bgr.copy()
    dst_height, dst_width = out.shape[:2]
    sx = dst_width / max(src_width, 1)
    sy = dst_height / max(src_height, 1)

    for label, (x_min, y_min, x_max, y_max) in boxes:
        x1 = int(round(x_min * sx))
        y1 = int(round(y_min * sy))
        x2 = int(round(x_max * sx))
        y2 = int(round(y_max * sy))

        x1 = max(0, min(x1, dst_width - 1))
        y1 = max(0, min(y1, dst_height - 1))
        x2 = max(0, min(x2, dst_width - 1))
        y2 = max(0, min(y2, dst_height - 1))

        color = CLASS_COLORS_BGR.get(label, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 6 if y1 > 16 else y1 + 14
        cv2.putText(
            out,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_float = image.astype(np.float32)
    min_val = float(np.min(image_float))
    max_val = float(np.max(image_float))
    if max_val <= min_val:
        return np.zeros_like(image_float, dtype=np.uint8)
    image_norm = (image_float - min_val) / (max_val - min_val)
    return (image_norm * 255.0).astype(np.uint8)


def make_overlay(rgb_bgr: np.ndarray, trm_gray: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    trm_resized = cv2.resize(trm_gray, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    trm_color = cv2.applyColorMap(trm_resized, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, trm_color, alpha, 0.0)


def run_viewer(samples: list[tuple[Path, Path, Path | None]], alpha: float, start_index: int) -> None:
    if not samples:
        raise RuntimeError("No RGB/Thermal filename pairs found.")

    idx = max(0, min(start_index, len(samples) - 1))
    annotation_cache: dict[Path, tuple[int, int, list[tuple[str, tuple[float, float, float, float]]]]] = {}

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_overlay = fig.add_subplot(gs[0, :])
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_trm = fig.add_subplot(gs[1, 1])

    fig.canvas.manager.set_window_title("RTDOD RGB + Thermal Overlay")
    ax_overlay.axis("off")
    ax_rgb.axis("off")
    ax_trm.axis("off")

    overlay_artist = None
    rgb_artist = None
    trm_artist = None

    def render() -> None:
        nonlocal overlay_artist, rgb_artist, trm_artist

        rgb_path, trm_path, annotation_path = samples[idx]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        trm = cv2.imread(str(trm_path), cv2.IMREAD_ANYDEPTH)

        if rgb is None or trm is None:
            raise RuntimeError(f"Unreadable pair: {rgb_path} | {trm_path}")

        if trm.ndim == 3:
            trm = cv2.cvtColor(trm, cv2.COLOR_BGR2GRAY)

        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        src_width, src_height = rgb.shape[1], rgb.shape[0]
        if annotation_path is not None:
            if annotation_path not in annotation_cache:
                annotation_cache[annotation_path] = parse_annotation(annotation_path)
            src_width, src_height, boxes = annotation_cache[annotation_path]

        trm_u8 = normalize_to_uint8(trm)
        overlay_bgr = make_overlay(rgb, trm_u8, alpha)

        rgb_annotated_bgr = draw_bboxes(rgb, boxes, src_width=src_width, src_height=src_height)
        overlay_annotated_bgr = draw_bboxes(
            overlay_bgr,
            boxes,
            src_width=src_width,
            src_height=src_height,
        )
        trm_annotated_bgr = draw_bboxes(
            cv2.cvtColor(trm_u8, cv2.COLOR_GRAY2BGR),
            boxes,
            src_width=src_width,
            src_height=src_height,
        )

        overlay_rgb = cv2.cvtColor(overlay_annotated_bgr, cv2.COLOR_BGR2RGB)
        rgb_display = cv2.cvtColor(rgb_annotated_bgr, cv2.COLOR_BGR2RGB)
        trm_display = cv2.cvtColor(trm_annotated_bgr, cv2.COLOR_BGR2RGB)

        if overlay_artist is None:
            overlay_artist = ax_overlay.imshow(overlay_rgb)
            rgb_artist = ax_rgb.imshow(rgb_display)
            trm_artist = ax_trm.imshow(trm_display)
        else:
            overlay_artist.set_data(overlay_rgb)
            rgb_artist.set_data(rgb_display)
            trm_artist.set_data(trm_display)

        ax_overlay.set_title(
            f"{idx + 1}/{len(samples)} | alpha={alpha:.2f} | "
            "Space/Right=next, Backspace/Left=prev, Q/Esc=quit\n"
            f"RGB: {rgb_path.name} | Thermal: {trm_path.name} | boxes={len(boxes)}",
            fontsize=10,
        )
        ax_rgb.set_title("RGB + boxes", fontsize=10)
        ax_trm.set_title("Thermal + boxes", fontsize=10)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal idx
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(fig)
            return
        if key in (" ", "space", "right"):
            idx = (idx + 1) % len(samples)
            render()
            return
        if key in ("backspace", "left"):
            idx = (idx - 1) % len(samples)
            render()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    rgb_dir = args.rgb_dir if args.rgb_dir is not None else args.dataset_root / "rgb"
    trm_dir = args.trm_dir if args.trm_dir is not None else args.dataset_root / "trm"
    annotation_dir = (
        args.annotation_dir if args.annotation_dir is not None else args.dataset_root / "annotations"
    )

    samples = build_samples(rgb_dir=rgb_dir, trm_dir=trm_dir, annotation_dir=annotation_dir)
    with_xml = sum(1 for _, _, xml_path in samples if xml_path is not None)
    print(
        f"Loaded {len(samples)} RGB/Thermal pairs from {rgb_dir} and {trm_dir} "
        f"({with_xml} with annotations from {annotation_dir})"
    )
    run_viewer(samples=samples, alpha=args.alpha, start_index=args.start_index)


if __name__ == "__main__":
    main()
