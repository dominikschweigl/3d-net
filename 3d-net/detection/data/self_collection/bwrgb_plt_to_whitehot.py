#!/usr/bin/env python3
"""
Convert a BWRGB-encoded thermal video to white-hot grayscale.

Palette loaded from BWRGB.plt (256 entries, index 0 = coldest, 255 = hottest).
A full (256,256,256) LUT is built via KDTree nearest-neighbour over the 256
palette entries so that compressed/blended pixels map smoothly to the closest
palette index rather than falling through to black.

Usage:
    python bwrgb_plt_to_whitehot.py input.mp4 output.mp4
    python bwrgb_plt_to_whitehot.py input.mp4 output.mp4 --clahe
"""

import av
import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial import KDTree

_PLT_PATH = Path(__file__).parent / "BWRGB.plt"


def _build_lut(plt_path):
    """
    Load BWRGB.plt and build a full (256,256,256) uint8 LUT via KDTree.

    The 256 palette RGB entries are used as reference points. Every possible
    RGB value is mapped to the index of its nearest palette entry, which is
    directly used as the white-hot grayscale value (0=coldest, 255=hottest).
    """
    palette = []
    for line in Path(plt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r, g, b = [float(v) for v in line.split(";") if v.strip()]
        palette.append([r, g, b])
    palette = np.array(palette, dtype=np.float32)
    assert palette.shape == (256, 3), f"Expected 256 palette entries, got {palette.shape[0]}"

    tree = KDTree(palette)

    idx8 = np.arange(256, dtype=np.float32)
    R, G, B = np.meshgrid(idx8, idx8, idx8, indexing='ij')
    queries = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1)

    print("  querying KDTree for all 16M RGB values (all CPU cores)...", flush=True)
    _, nearest_idx = tree.query(queries, workers=-1)

    return nearest_idx.reshape(256, 256, 256).astype(np.uint8)


print("Building RGB->temperature LUT from BWRGB.plt...", flush=True)
_LUT = _build_lut(_PLT_PATH)  # (256,256,256) uint8
print("LUT ready.\n", flush=True)


def frame_to_whitehot(frame_bgr, clahe=None):
    """
    Convert one BWRGB BGR frame to white-hot uint8 grayscale.

    Each pixel's RGB is looked up in the full LUT, which maps every possible
    RGB to the nearest palette index (0=coldest, 255=hottest) via KDTree
    nearest-neighbour, so compressed/blended pixels are handled gracefully.
    """
    r = frame_bgr[:, :, 2]
    g = frame_bgr[:, :, 1]
    b = frame_bgr[:, :, 0]

    gray = _LUT[r, g, b]

    if clahe is not None:
        gray = clahe.apply(gray)

    return gray


def process_video(input_path, output_path, use_clahe=True):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    with av.open(input_path) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = 'AUTO'

        fps   = float(in_stream.average_rate)
        total = in_stream.frames

        print(f"Input : {input_path}  ({in_stream.width}x{in_stream.height} @ {fps:.2f} fps, {total} frames)")
        print(f"Output: {output_path}  CLAHE={'on' if use_clahe else 'off'}\n")

        with av.open(output_path, 'w') as out_container:
            out_stream = out_container.add_stream('libx264', rate=in_stream.average_rate)
            out_stream.width  = in_stream.width
            out_stream.height = in_stream.height
            out_stream.pix_fmt = 'yuv420p'
            out_stream.codec_context.time_base = in_stream.time_base
            out_stream.codec_context.max_b_frames = 0

            frame_idx = 0
            last_pts  = -1
            for frame in in_container.decode(in_stream):
                if frame.pts is None or frame.pts <= last_pts:
                    continue
                last_pts = frame.pts

                bgr  = frame.to_ndarray(format='bgr24')
                gray = frame_to_whitehot(bgr, clahe)

                out_frame = av.VideoFrame.from_ndarray(gray, format='gray8').reformat(format='yuv420p')
                out_frame.pts       = frame.pts
                out_frame.time_base = in_stream.time_base

                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)

                frame_idx += 1
                if total and (frame_idx % 10 == 0 or frame_idx == total):
                    print(f"  processed {frame_idx}/{total} frames ({frame_idx/total*100:.1f}%)", end="\r", flush=True)

            for packet in out_stream.encode():
                out_container.mux(packet)

    print(f"\nDone. Wrote {frame_idx} frames to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BWRGB thermal video to white-hot grayscale using BWRGB.plt palette."
    )
    parser.add_argument("input",  help="Input BWRGB .mp4 video")
    parser.add_argument("output", help="Output grayscale video")
    parser.add_argument("--clahe", action="store_true",
                        help="Enable adaptive contrast stretching (CLAHE).")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file '{args.input}' not found.")

    process_video(args.input, args.output, use_clahe=args.clahe)


if __name__ == "__main__":
    main()
