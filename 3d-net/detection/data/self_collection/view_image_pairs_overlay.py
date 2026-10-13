#!/usr/bin/env python3
"""Self-collected dataset RGB/IR overlay viewer.

Pairs images by filename stem from `img/rgb/` and `img/ir/`, then displays:
- Top:          RGB + Thermal overlay
- Bottom-left:  RGB image
- Bottom-right: Thermal image

Controls:
- Space / Right Arrow: next image
- Backspace / Left Arrow: previous image
- Q / Esc: quit
- "Go to frame" box: type a 1-based frame number and press Enter
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CLASS_COLORS_BGR = {
    "bus":        (80, 220, 80),
    "bicycle":             (40, 180, 255),
    "motorcycle":          (0, 128, 255),
    "truck":           (80, 80, 255),
    "person":             (220, 80, 220),
    "car":                (255, 200, 40),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="View self-collected RGB/IR overlays paired by filename.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=script_dir,
        help="Path to dataset root containing img/rgb/ and img/ir/.",
    )
    parser.add_argument("--rgb-dir", type=Path, default=None, help="Optional explicit RGB directory.")
    parser.add_argument("--ir-dir",  type=Path, default=None, help="Optional explicit IR/thermal directory.")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Optional annotation directory containing Pascal VOC XML files.",
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
        return {}
    return {p.stem: p for p in sorted(folder.glob("*.xml")) if p.is_file()}


def build_samples(
    rgb_dir: Path,
    ir_dir: Path,
    annotation_dir: Path | None,
) -> list[tuple[Path, Path, Path | None]]:
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR dir not found: {ir_dir}")

    rgb_images = list_images(rgb_dir)
    ir_images  = list_images(ir_dir)
    xmls       = list_xmls(annotation_dir) if annotation_dir is not None else {}

    common_stems = sorted(set(rgb_images).intersection(ir_images))
    return [(rgb_images[s], ir_images[s], xmls.get(s)) for s in common_stems]


def parse_annotation(xml_path: Path) -> tuple[int, int, list[tuple[str, tuple[float, float, float, float]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width  = int(size.findtext("width",  default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid <size> in {xml_path}")

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="Unknown").strip() or "Unknown"
        bnd   = obj.find("bndbox")
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
    out = image_bgr.copy()
    dst_height, dst_width = out.shape[:2]
    sx = dst_width  / max(src_width,  1)
    sy = dst_height / max(src_height, 1)

    for label, (x_min, y_min, x_max, y_max) in boxes:
        x1 = max(0, min(int(round(x_min * sx)), dst_width  - 1))
        y1 = max(0, min(int(round(y_min * sy)), dst_height - 1))
        x2 = max(0, min(int(round(x_max * sx)), dst_width  - 1))
        y2 = max(0, min(int(round(y_max * sy)), dst_height - 1))

        color = CLASS_COLORS_BGR.get(label, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 6 if y1 > 16 else y1 + 14
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img_f = image.astype(np.float32)
    lo, hi = float(np.min(img_f)), float(np.max(img_f))
    if hi <= lo:
        return np.zeros_like(img_f, dtype=np.uint8)
    return ((img_f - lo) / (hi - lo) * 255.0).astype(np.uint8)


def make_overlay(rgb_bgr: np.ndarray, ir_gray: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    ir_resized = cv2.resize(ir_gray, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    ir_color   = cv2.applyColorMap(ir_resized, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, ir_color, alpha, 0.0)


def run_viewer(samples: list[tuple[Path, Path, Path | None]], alpha: float, start_index: int) -> None:
    if not samples:
        raise RuntimeError("No RGB/IR filename pairs found.")

    idx = max(0, min(start_index, len(samples) - 1))
    annotation_cache: dict[Path, tuple[int, int, list]] = {}

    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(bottom=0.08)
    gs  = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_overlay = fig.add_subplot(gs[0, :])
    ax_rgb     = fig.add_subplot(gs[1, 0])
    ax_ir      = fig.add_subplot(gs[1, 1])
    ax_goto    = fig.add_axes([0.35, 0.01, 0.12, 0.04])
    goto_box   = mwidgets.TextBox(ax_goto, "Go to frame: ", initial="")

    fig.canvas.manager.set_window_title("Self-collected RGB + IR Overlay")
    ax_overlay.axis("off")
    ax_rgb.axis("off")
    ax_ir.axis("off")

    overlay_artist = rgb_artist = ir_artist = None

    def render() -> None:
        nonlocal overlay_artist, rgb_artist, ir_artist

        rgb_path, ir_path, annotation_path = samples[idx]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir  = cv2.imread(str(ir_path),  cv2.IMREAD_ANYDEPTH)

        if rgb is None or ir is None:
            raise RuntimeError(f"Unreadable pair: {rgb_path} | {ir_path}")

        if ir.ndim == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        src_width, src_height = rgb.shape[1], rgb.shape[0]
        if annotation_path is not None:
            if annotation_path not in annotation_cache:
                annotation_cache[annotation_path] = parse_annotation(annotation_path)
            src_width, src_height, boxes = annotation_cache[annotation_path]

        ir_u8        = normalize_to_uint8(ir)
        overlay_bgr  = make_overlay(rgb, ir_u8, alpha)

        rgb_ann     = draw_bboxes(rgb,                                boxes, src_width, src_height)
        overlay_ann = draw_bboxes(overlay_bgr,                        boxes, src_width, src_height)
        ir_ann      = draw_bboxes(cv2.cvtColor(ir_u8, cv2.COLOR_GRAY2BGR), boxes, src_width, src_height)

        overlay_disp = cv2.cvtColor(overlay_ann, cv2.COLOR_BGR2RGB)
        rgb_disp     = cv2.cvtColor(rgb_ann,     cv2.COLOR_BGR2RGB)
        ir_disp      = cv2.cvtColor(ir_ann,      cv2.COLOR_BGR2RGB)

        if overlay_artist is None:
            overlay_artist = ax_overlay.imshow(overlay_disp)
            rgb_artist     = ax_rgb.imshow(rgb_disp)
            ir_artist      = ax_ir.imshow(ir_disp)
        else:
            overlay_artist.set_data(overlay_disp)
            rgb_artist.set_data(rgb_disp)
            ir_artist.set_data(ir_disp)

        ax_overlay.set_title(
            f"{idx + 1}/{len(samples)} | alpha={alpha:.2f} | "
            "Space/Right=next, Backspace/Left=prev, Q/Esc=quit, Go-to-frame=type number+Enter\n"
            f"RGB: {rgb_path.name} | IR: {ir_path.name}"
            + (f" | boxes={len(boxes)}" if boxes else ""),
            fontsize=10,
        )
        ax_rgb.set_title("RGB" + (" + boxes" if boxes else ""), fontsize=10)
        ax_ir.set_title("Thermal" + (" + boxes" if boxes else ""), fontsize=10)
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

    def on_goto(text: str) -> None:
        nonlocal idx
        try:
            n = int(text.strip())
            idx = max(0, min(n - 1, len(samples) - 1))
            render()
        except ValueError:
            pass
        goto_box.set_val("")

    fig.canvas.mpl_connect("key_press_event", on_key)
    goto_box.on_submit(on_goto)
    render()
    plt.show()


def main() -> None:
    args = parse_args()

    rgb_dir = args.rgb_dir if args.rgb_dir is not None else args.dataset_root / "img" / "rgb"
    ir_dir  = args.ir_dir  if args.ir_dir  is not None else args.dataset_root / "img" / "ir"
    annotation_dir = args.annotation_dir  # None if not provided — annotations are optional

    samples = build_samples(rgb_dir=rgb_dir, ir_dir=ir_dir, annotation_dir=annotation_dir)
    with_xml = sum(1 for _, _, xml_path in samples if xml_path is not None)
    print(
        f"Loaded {len(samples)} RGB/IR pairs from:\n"
        f"  RGB: {rgb_dir}\n"
        f"  IR:  {ir_dir}"
        + (f"\n  Annotations: {annotation_dir} ({with_xml} found)" if annotation_dir is not None else "")
    )
    run_viewer(samples=samples, alpha=args.alpha, start_index=args.start_index)


if __name__ == "__main__":
    main()
