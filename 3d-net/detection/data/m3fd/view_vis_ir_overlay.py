#!/usr/bin/env python3
"""Simple M3FD VIS/IR overlay viewer.

Pairs images by filename stem from `Vis/` and `Ir/`, then displays:
- Top: VIS + IR overlay
- Bottom-left: VIS image
- Bottom-right: IR image

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
    "People": (80, 220, 80),
    "Car": (40, 180, 255),
    "Bus": (0, 128, 255),
    "Lamp": (255, 255, 80),
    "Motorcycle": (220, 80, 220),
    "Truck": (80, 80, 255),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dataset_root_default = script_dir

    parser = argparse.ArgumentParser(description="View M3FD VIS/IR overlays paired by filename.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root_default,
        help="Path to M3FD root containing Vis/ and Ir/.",
    )
    parser.add_argument("--vis-dir", type=Path, default=None, help="Optional explicit VIS directory.")
    parser.add_argument("--ir-dir", type=Path, default=None, help="Optional explicit IR directory.")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Optional explicit annotation directory containing Pascal VOC XML files.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="IR overlay alpha in [0, 1].")
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
    vis_dir: Path,
    ir_dir: Path,
    annotation_dir: Path,
) -> list[tuple[Path, Path, Path | None]]:
    if not vis_dir.exists():
        raise FileNotFoundError(f"VIS dir not found: {vis_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR dir not found: {ir_dir}")

    vis_images = list_images(vis_dir)
    ir_images = list_images(ir_dir)
    xmls = list_xmls(annotation_dir)

    common_stems = sorted(set(vis_images).intersection(ir_images))
    return [(vis_images[s], ir_images[s], xmls.get(s)) for s in common_stems]


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


def make_overlay(vis_bgr: np.ndarray, ir_gray: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    ir_resized = cv2.resize(ir_gray, (vis_bgr.shape[1], vis_bgr.shape[0]))
    ir_color = cv2.applyColorMap(ir_resized, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(vis_bgr, 1.0 - alpha, ir_color, alpha, 0.0)


def run_viewer(samples: list[tuple[Path, Path, Path | None]], alpha: float, start_index: int) -> None:
    if not samples:
        raise RuntimeError("No VIS/IR filename pairs found.")

    idx = max(0, min(start_index, len(samples) - 1))
    annotation_cache: dict[Path, tuple[int, int, list[tuple[str, tuple[float, float, float, float]]]]] = {}

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_overlay = fig.add_subplot(gs[0, :])
    ax_vis = fig.add_subplot(gs[1, 0])
    ax_ir = fig.add_subplot(gs[1, 1])

    fig.canvas.manager.set_window_title("M3FD VIS + IR Overlay")
    ax_overlay.axis("off")
    ax_vis.axis("off")
    ax_ir.axis("off")

    overlay_artist = None
    vis_artist = None
    ir_artist = None

    def render() -> None:
        nonlocal overlay_artist, vis_artist, ir_artist

        vis_path, ir_path, annotation_path = samples[idx]
        vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_ANYDEPTH)

        if vis is None or ir is None:
            raise RuntimeError(f"Unreadable pair: {vis_path} | {ir_path}")

        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        src_width, src_height = vis.shape[1], vis.shape[0]
        if annotation_path is not None:
            if annotation_path not in annotation_cache:
                annotation_cache[annotation_path] = parse_annotation(annotation_path)
            src_width, src_height, boxes = annotation_cache[annotation_path]

        ir_u8 = normalize_to_uint8(ir)
        overlay_bgr = make_overlay(vis, ir_u8, alpha)

        vis_annotated_bgr = draw_bboxes(vis, boxes, src_width=src_width, src_height=src_height)
        overlay_annotated_bgr = draw_bboxes(
            overlay_bgr,
            boxes,
            src_width=src_width,
            src_height=src_height,
        )
        ir_annotated_bgr = draw_bboxes(
            cv2.cvtColor(ir_u8, cv2.COLOR_GRAY2BGR),
            boxes,
            src_width=src_width,
            src_height=src_height,
        )

        overlay_rgb = cv2.cvtColor(overlay_annotated_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = cv2.cvtColor(vis_annotated_bgr, cv2.COLOR_BGR2RGB)
        ir_rgb = cv2.cvtColor(ir_annotated_bgr, cv2.COLOR_BGR2RGB)

        if overlay_artist is None:
            overlay_artist = ax_overlay.imshow(overlay_rgb)
            vis_artist = ax_vis.imshow(vis_rgb)
            ir_artist = ax_ir.imshow(ir_rgb)
        else:
            overlay_artist.set_data(overlay_rgb)
            vis_artist.set_data(vis_rgb)
            ir_artist.set_data(ir_rgb)

        ax_overlay.set_title(
            f"{idx + 1}/{len(samples)} | alpha={alpha:.2f} | "
            "Space/Right=next, Backspace/Left=prev, Q/Esc=quit\n"
            f"VIS: {vis_path.name} | IR: {ir_path.name} | boxes={len(boxes)}",
            fontsize=10,
        )
        ax_vis.set_title("VIS + boxes", fontsize=10)
        ax_ir.set_title("Thermal (IR) + boxes", fontsize=10)
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

    vis_dir = args.vis_dir if args.vis_dir is not None else args.dataset_root / "Vis"
    ir_dir = args.ir_dir if args.ir_dir is not None else args.dataset_root / "Ir"
    annotation_dir = (
        args.annotation_dir if args.annotation_dir is not None else args.dataset_root / "Annotation"
    )

    samples = build_samples(vis_dir=vis_dir, ir_dir=ir_dir, annotation_dir=annotation_dir)
    with_xml = sum(1 for _, _, xml_path in samples if xml_path is not None)
    print(
        f"Loaded {len(samples)} VIS/IR pairs from {vis_dir} and {ir_dir} "
        f"({with_xml} with annotations from {annotation_dir})"
    )
    run_viewer(samples=samples, alpha=args.alpha, start_index=args.start_index)


if __name__ == "__main__":
    main()
