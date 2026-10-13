"""Visualize DeepSort predictions vs ground truth for one session.

Draws ground-truth boxes (green) and DeepSort tracked boxes (red) on every
frame of the chosen session and writes them as a video.

Usage
-----
    python visualize_deepsort.py --session hausern/2026-01-20/aufnahme5
    python visualize_deepsort.py --list-sessions          # print available sessions
    python visualize_deepsort.py --session <key> --out /tmp/deepsort_viz.mp4
"""

from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np
import fiftyone as fo
from deep_sort_realtime.deepsort_tracker import DeepSort


DATASET_NAME = "3d_net_rgbt"
GT_FIELD     = "ground_truth"
OUT_PATH     = "/tmp/deepsort_viz.mp4"
FPS          = 5

GT_COLOR    = (0, 200, 0)    # green  — ground truth
DS_COLOR    = (0, 0, 220)    # red    — DeepSort Kalman output
RAW_COLOR   = (220, 150, 0)  # blue   — DeepSort raw detection box
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.5
THICKNESS   = 2


def _session_key(s: fo.Sample) -> tuple:
    return (s.get_field("location") or "", s.get_field("date") or "", s.get_field("session") or "")


def _frame_index(s: fo.Sample) -> int:
    return s.get_field("frame_index") or 0


def _session_str(key: tuple) -> str:
    return "/".join(filter(None, key))


def list_sessions(dataset: fo.Dataset) -> list[tuple]:
    seen: set[tuple] = set()
    for sample in dataset.iter_samples():
        seen.add(_session_key(sample))
    return sorted(seen)


def draw_box(img: np.ndarray, x1: float, y1: float, x2: float, y2: float,
             label: str, color: tuple) -> None:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    cv2.rectangle(img, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), color, -1)
    cv2.putText(img, label, (int(x1), int(y1) - 2), FONT, FONT_SCALE, (255, 255, 255), 1)


def run_visualization(
    samples: list[fo.Sample],
    gt_field: str,
    out_path: str,
    fps: int,
    max_age: int,
) -> None:
    tracker = DeepSort(max_age=max_age)
    writer: cv2.VideoWriter | None = None

    for sample in sorted(samples, key=_frame_index):
        img = cv2.imread(sample.local_path)
        if img is None:
            logging.warning("Could not read %s", sample.local_path)
            continue

        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        img_h, img_w = img.shape[:2]
        vis = img.copy()

        dets_field: fo.Detections | None = sample.get_field(gt_field)
        bbs = []

        if dets_field and dets_field.detections:
            for det in dets_field.detections:
                x, y, w, h = det.bounding_box
                x1, y1 = x * img_w, y * img_h
                x2, y2 = (x + w) * img_w, (y + h) * img_h
                bbs.append(([x1, y1, x2 - x1, y2 - y1], det.confidence or 1.0, det.label or ""))
                draw_box(vis, x1, y1, x2, y2, f"GT {det.index}", GT_COLOR)

        tracks = tracker.update_tracks(bbs, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            lx1, ly1, lx2, ly2 = track.to_ltrb()
            draw_box(vis, lx1, ly1, lx2, ly2, f"DS {track.track_id}", DS_COLOR)
            if track.original_ltwh is not None:
                rx, ry, rw, rh = track.original_ltwh
                draw_box(vis, rx, ry, rx + rw, ry + rh, f"raw {track.track_id}", RAW_COLOR)

        legend_y = 20
        for text, color in [
            ("GT (green)", GT_COLOR),
            ("DeepSort Kalman (red)", DS_COLOR),
            ("DeepSort raw det (blue)", RAW_COLOR),
        ]:
            cv2.putText(vis, text, (10, legend_y), FONT, FONT_SCALE, color, THICKNESS)
            legend_y += 20

        writer.write(vis)

    if writer:
        writer.release()
        logging.info("Saved video to %s", out_path)
    else:
        logging.warning("No frames written.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize DeepSort predictions vs ground truth.",
    )
    parser.add_argument("--name",     default=DATASET_NAME)
    parser.add_argument("--gt-field", default=GT_FIELD)
    parser.add_argument("--session",  default=None,
                        help="Session key as 'location/date/session'")
    parser.add_argument("--split",    default=None,
                        help="Filter by split before selecting session")
    parser.add_argument("--out",      default=OUT_PATH,
                        help="Output video path (default: %(default)s)")
    parser.add_argument("--fps",      type=int, default=FPS)
    parser.add_argument("--max-age",  type=int, default=5)
    parser.add_argument("--list-sessions", action="store_true",
                        help="Print all available sessions and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not fo.dataset_exists(args.name):
        raise SystemExit(f"Dataset '{args.name}' does not exist.")

    dataset = fo.load_dataset(args.name)
    collection: fo.Dataset | fo.DatasetView = dataset
    if args.split:
        collection = dataset.match(fo.ViewField("split") == args.split)

    if args.list_sessions:
        for key in list_sessions(collection):
            print(_session_str(key))
        return

    if not args.session:
        raise SystemExit("Provide --session <key> or use --list-sessions to see options.")

    parts = args.session.split("/")
    location = parts[0] if len(parts) > 0 else ""
    date     = parts[1] if len(parts) > 1 else ""
    session  = parts[2] if len(parts) > 2 else ""
    target_key = (location, date, session)

    samples = [
        s for s in collection.iter_samples(progress=True)
        if _session_key(s) == target_key
    ]

    if not samples:
        raise SystemExit(f"No samples found for session '{args.session}'.")

    logging.info("Found %d frames for session '%s'.", len(samples), args.session)
    run_visualization(samples, args.gt_field, args.out, args.fps, args.max_age)


if __name__ == "__main__":
    main()
