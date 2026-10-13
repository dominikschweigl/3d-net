#!/usr/bin/env python3
"""VisDrone VIS/IR overlay viewer.

Supports both annotation formats:
  - <bndbox> with xmin/ymin/xmax/ymax
  - <polygon> with x1/y1…x4/y4  (AABB is derived from point extents)

Layout:
- Top:          VIS + IR overlay (alpha blend, inferno thermal) with bounding boxes
- Bottom-left:  VIS image with bounding boxes
- Bottom-right: IR thermal image (inferno colormap) with bounding boxes

Controls:
- Space / Right Arrow   : next image
- Backspace / Left Arrow: previous image
- Q / Esc               : quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# BGR colours per class label (case-insensitive match)
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "car":         (255, 180,  40),
    "truck":       ( 80,  80, 255),
    "bus":         (  0, 128, 255),
    "people":      ( 80, 220,  80),
    "person":      ( 80, 220,  80),
    "motorcycle":  (220,  80, 220),
    "lamp":        (255, 255,  80),
    "feright car": (160, 160, 255),
}
DEFAULT_COLOR_BGR = (200, 200, 200)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="View VisDrone VIS/IR overlays.")
    parser.add_argument("--dataset-root", type=Path, default=script_dir)
    parser.add_argument("--split",        type=str,  default="train")
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="IR blend alpha in [0, 1].")
    parser.add_argument("--start-index",  type=int,  default=0)
    return parser.parse_args()


# ── file discovery ────────────────────────────────────────────────────────────

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


def build_samples(dataset_root: Path, split: str):
    vis_img_dir = dataset_root / split / "trainimg"
    ir_img_dir  = dataset_root / split / "trainimgr"
    vis_lbl_dir = dataset_root / split / "trainlabel"
    ir_lbl_dir  = dataset_root / split / "trainlabelr"

    for d in (vis_img_dir, ir_img_dir):
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")

    vis_images = list_images(vis_img_dir)
    ir_images  = list_images(ir_img_dir)
    vis_xmls   = list_xmls(vis_lbl_dir)
    ir_xmls    = list_xmls(ir_lbl_dir)

    common = sorted(set(vis_images) & set(ir_images))
    return [
        (vis_images[s], ir_images[s], vis_xmls.get(s), ir_xmls.get(s))
        for s in common
    ]


# ── annotation parsing ────────────────────────────────────────────────────────

def _parse_polygon(elem) -> tuple[float, float, float, float] | None:
    """Parse <polygon> with x1/y1…xN/yN and return AABB (x1,y1,x2,y2)."""
    xs, ys = [], []
    i = 1
    while True:
        xt = elem.findtext(f"x{i}")
        yt = elem.findtext(f"y{i}")
        if xt is None or yt is None:
            break
        try:
            xs.append(float(xt))
            ys.append(float(yt))
        except ValueError:
            pass
        i += 1
    if len(xs) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _parse_bndbox(elem) -> tuple[float, float, float, float] | None:
    """Parse <bndbox> with xmin/ymin/xmax/ymax."""
    try:
        x1 = float(elem.findtext("xmin", "0"))
        y1 = float(elem.findtext("ymin", "0"))
        x2 = float(elem.findtext("xmax", "0"))
        y2 = float(elem.findtext("ymax", "0"))
    except ValueError:
        return None
    return x1, y1, x2, y2


def parse_annotation(
    xml_path: Path | None,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return list of (label, (x1, y1, x2, y2)) from a Pascal-VOC-style XML.

    Handles both <bndbox> and <polygon> annotation formats.
    """
    if xml_path is None or not xml_path.exists():
        return []

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"[warn] XML parse error in {xml_path}: {exc}")
        return []

    root = tree.getroot()
    results = []

    for obj in root.findall("object"):
        label = (obj.findtext("name") or "unknown").strip()

        box: tuple[float, float, float, float] | None = None

        bnd = obj.find("bndbox")
        if bnd is not None:
            box = _parse_bndbox(bnd)

        if box is None:
            poly = obj.find("polygon")
            if poly is not None:
                box = _parse_polygon(poly)

        if box is None:
            continue

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        results.append((label, (x1, y1, x2, y2)))

    return results


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[tuple[str, tuple[float, float, float, float]]],
) -> np.ndarray:
    out = image_bgr.copy()
    for label, (x1, y1, x2, y2) in boxes:
        color = CLASS_COLORS_BGR.get(label.lower(), DEFAULT_COLOR_BGR)
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, pt1, pt2, color, 2)
        text_y = pt1[1] - 5 if pt1[1] > 15 else pt2[1] + 13
        cv2.putText(out, label, (pt1[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def apply_inferno(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def make_overlay(vis_bgr: np.ndarray, ir_inferno_bgr: np.ndarray, alpha: float) -> np.ndarray:
    ir_resized = cv2.resize(ir_inferno_bgr, (vis_bgr.shape[1], vis_bgr.shape[0]))
    return cv2.addWeighted(vis_bgr, 1.0 - alpha, ir_resized, alpha, 0.0)


# ── viewer ────────────────────────────────────────────────────────────────────

def run_viewer(
    samples: list[tuple[Path, Path, Path | None, Path | None]],
    alpha: float,
    start_index: int,
) -> None:
    if not samples:
        raise RuntimeError("No VIS/IR pairs found.")

    idx = max(0, min(start_index, len(samples) - 1))
    annotation_cache: dict[Path, list] = {}

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0e0e0e")
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.08, wspace=0.04)

    ax_overlay = fig.add_subplot(gs[0, :])
    ax_vis     = fig.add_subplot(gs[1, 0])
    ax_ir      = fig.add_subplot(gs[1, 1])

    for ax in (ax_overlay, ax_vis, ax_ir):
        ax.axis("off")
        ax.set_facecolor("#0e0e0e")

    try:
        fig.canvas.manager.set_window_title("VisDrone VIS/IR Viewer")
    except Exception:
        pass

    overlay_im = vis_im = ir_im = None

    def get_annotations(xml_path: Path | None) -> list:
        if xml_path is None:
            return []
        if xml_path not in annotation_cache:
            annotation_cache[xml_path] = parse_annotation(xml_path)
        return annotation_cache[xml_path]

    def render() -> None:
        nonlocal overlay_im, vis_im, ir_im

        vis_path, ir_path, vis_xml, ir_xml = samples[idx]

        vis_bgr = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        ir_bgr  = cv2.imread(str(ir_path),  cv2.IMREAD_COLOR)

        if vis_bgr is None or ir_bgr is None:
            print(f"[warn] unreadable pair: {vis_path} | {ir_path}")
            return

        vis_boxes = get_annotations(vis_xml)
        ir_boxes  = get_annotations(ir_xml)
        all_boxes = vis_boxes + [b for b in ir_boxes if b not in vis_boxes]

        ir_inferno = apply_inferno(ir_bgr)

        # Top: blended overlay with all boxes
        overlay_bgr = make_overlay(vis_bgr, ir_inferno, alpha)
        overlay_ann = draw_boxes(overlay_bgr, all_boxes)

        # Bottom-left: plain VIS + vis boxes
        vis_ann = draw_boxes(vis_bgr, vis_boxes)

        # Bottom-right: inferno IR + ir boxes
        ir_ann = draw_boxes(ir_inferno, ir_boxes)

        overlay_rgb = cv2.cvtColor(overlay_ann, cv2.COLOR_BGR2RGB)
        vis_rgb     = cv2.cvtColor(vis_ann,     cv2.COLOR_BGR2RGB)
        ir_rgb      = cv2.cvtColor(ir_ann,      cv2.COLOR_BGR2RGB)

        if overlay_im is None:
            overlay_im = ax_overlay.imshow(overlay_rgb)
            vis_im     = ax_vis.imshow(vis_rgb)
            ir_im      = ax_ir.imshow(ir_rgb)
        else:
            overlay_im.set_data(overlay_rgb)
            vis_im.set_data(vis_rgb)
            ir_im.set_data(ir_rgb)
            ax_overlay.set_xlim(0, overlay_rgb.shape[1])
            ax_overlay.set_ylim(overlay_rgb.shape[0], 0)
            ax_vis.set_xlim(0, vis_rgb.shape[1])
            ax_vis.set_ylim(vis_rgb.shape[0], 0)
            ax_ir.set_xlim(0, ir_rgb.shape[1])
            ax_ir.set_ylim(ir_rgb.shape[0], 0)

        title_kw = dict(fontsize=10, color="white", pad=6)
        ax_overlay.set_title(
            f"[{idx + 1}/{len(samples)}]  VIS + IR Overlay  |  α={alpha:.2f}  |  "
            f"Space/→ next  ·  ←/Backspace prev  ·  Q/Esc quit\n"
            f"{vis_path.name}  |  boxes={len(all_boxes)}",
            **title_kw,
        )
        ax_vis.set_title(f"VIS  ({len(vis_boxes)} boxes)", **title_kw)
        ax_ir.set_title(f"IR — inferno  ({len(ir_boxes)} boxes)", **title_kw)

        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal idx
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(fig)
            return
        if key in (" ", "right"):
            idx = (idx + 1) % len(samples)
            render()
        elif key in ("backspace", "left"):
            idx = (idx - 1) % len(samples)
            render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.tight_layout()
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    samples = build_samples(args.dataset_root, args.split)
    print(
        f"Found {len(samples)} VIS/IR pairs  "
        f"(split='{args.split}', root={args.dataset_root})"
    )
    run_viewer(samples=samples, alpha=args.alpha, start_index=args.start_index)


if __name__ == "__main__":
    main()