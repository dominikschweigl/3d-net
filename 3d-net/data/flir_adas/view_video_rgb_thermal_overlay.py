#!/usr/bin/env python3
"""Simple FLIR video-set RGB/thermal overlay viewer.

Uses the official 1:1 filename mapping in rgb_to_thermal_vid_map.json for
video_rgb_test <-> video_thermal_test.

Controls:
- Space / Right Arrow: next
- Backspace / Left Arrow: previous
- Q / Esc: quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dataset_root_default = script_dir / "FLIR_ADAS_v2"

    parser = argparse.ArgumentParser(description="View mapped FLIR video RGB/thermal overlays.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root_default,
        help="Path to FLIR_ADAS_v2 directory.",
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        default=script_dir / "rgb_to_thermal_vid_map.json",
        help="Path to rgb_to_thermal_vid_map.json.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Thermal overlay alpha in [0, 1].")
    parser.add_argument("--start-index", type=int, default=0, help="Start pair index.")
    return parser.parse_args()


def build_pairs(dataset_root: Path, map_file: Path) -> List[Tuple[Path, Path]]:
    rgb_dir = dataset_root / "video_rgb_test" / "data"
    thermal_dir = dataset_root / "video_thermal_test" / "data"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB test dir not found: {rgb_dir}")
    if not thermal_dir.exists():
        raise FileNotFoundError(f"Thermal test dir not found: {thermal_dir}")
    if not map_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {map_file}")

    with map_file.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    pairs: List[Tuple[Path, Path]] = []
    for rgb_name, thermal_name in sorted(mapping.items()):
        rgb_path = rgb_dir / rgb_name
        thermal_path = thermal_dir / thermal_name
        if rgb_path.exists() and thermal_path.exists():
            pairs.append((rgb_path, thermal_path))

    return pairs


def thermal_to_inferno(thermal_bgr: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    thermal_resized = cv2.resize(thermal_bgr, out_size)
    thermal_gray = cv2.cvtColor(thermal_resized, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)


def make_overlay(rgb_bgr: np.ndarray, thermal_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    thermal_color = thermal_to_inferno(thermal_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, thermal_color, alpha, 0.0)


def run_viewer(pairs: List[Tuple[Path, Path]], alpha: float, start_index: int) -> None:
    if not pairs:
        raise RuntimeError("No mapped RGB/thermal pairs found in video test set.")

    idx = max(0, min(start_index, len(pairs) - 1))

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_overlay = fig.add_subplot(gs[0, :])
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_thermal = fig.add_subplot(gs[1, 1])

    fig.canvas.manager.set_window_title("FLIR Video RGB+Thermal")
    ax_overlay.axis("off")
    ax_rgb.axis("off")
    ax_thermal.axis("off")

    overlay_artist = None
    rgb_artist = None
    thermal_artist = None

    def render() -> None:
        nonlocal idx, overlay_artist, rgb_artist, thermal_artist

        rgb_path, thermal_path = pairs[idx]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(thermal_path), cv2.IMREAD_COLOR)
        if rgb is None or thermal is None:
            raise RuntimeError(f"Unreadable pair: {rgb_path} | {thermal_path}")

        overlay_bgr = make_overlay(rgb, thermal, alpha)
        thermal_color_bgr = thermal_to_inferno(thermal, (rgb.shape[1], rgb.shape[0]))

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        thermal_rgb = cv2.cvtColor(thermal_color_bgr, cv2.COLOR_BGR2RGB)

        if overlay_artist is None:
            overlay_artist = ax_overlay.imshow(overlay_rgb)
            rgb_artist = ax_rgb.imshow(rgb_rgb)
            thermal_artist = ax_thermal.imshow(thermal_rgb)
        else:
            overlay_artist.set_data(overlay_rgb)
            rgb_artist.set_data(rgb_rgb)
            thermal_artist.set_data(thermal_rgb)

        ax_overlay.set_title(
            f"{idx + 1}/{len(pairs)} | alpha={alpha:.2f} | "
            "Space/Right=next, Backspace/Left=prev, Q/Esc=quit\n"
            f"RGB: {rgb_path.name}\nTHM: {thermal_path.name}",
            fontsize=10,
        )
        ax_rgb.set_title("RGB", fontsize=10)
        ax_thermal.set_title("Thermal (Inferno)", fontsize=10)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal idx
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(fig)
            return
        if key in (" ", "space", "right"):
            idx = (idx + 1) % len(pairs)
            render()
            return
        if key in ("backspace", "left"):
            idx = (idx - 1) % len(pairs)
            render()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    pairs = build_pairs(args.dataset_root, args.map_file)
    print(f"Loaded {len(pairs)} mapped video pairs from {args.map_file.name}")
    run_viewer(pairs=pairs, alpha=args.alpha, start_index=args.start_index)


if __name__ == "__main__":
    main()
