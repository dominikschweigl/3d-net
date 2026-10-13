"""Apply DeepSort tracking to detections in a FiftyOne image dataset.

Samples are grouped by (location, date, session) and sorted by frame_index.
Within each session the DeepSort tracker runs across the image sequence,
assigning consistent track IDs.  Results are stored in ``out_field`` on each
sample.
"""

from __future__ import annotations

import argparse
import itertools
import logging

import cv2
import fiftyone as fo
from deep_sort_realtime.deepsort_tracker import DeepSort


DATASET_NAME = "3d_net_rgbt"
IN_FIELD     = "ground_truth"
OUT_FIELD    = "ds_tracks"


# ──────────────────────────────────────────────────────────────────────────────
# Core tracking logic
# ──────────────────────────────────────────────────────────────────────────────

def _session_key(sample: fo.Sample) -> tuple:
    return (
        sample.get_field("location") or "",
        sample.get_field("date")     or "",
        sample.get_field("session")  or "",
    )


def _frame_index(sample: fo.Sample) -> int:
    return sample.get_field("frame_index") or 0


def track_session(
    samples: list[fo.Sample],
    in_field: str,
    out_field: str,
    max_age: int,
    keep_confidence: bool,
) -> int:
    """Run DeepSort over one ordered sequence of image samples."""
    tracker = DeepSort(max_age=max_age)
    written = 0

    for sample in sorted(samples, key=_frame_index):
        img = cv2.imread(sample.local_path)
        if img is None:
            logging.warning("Could not read image: %s", sample.local_path)
            continue

        img_h, img_w = img.shape[:2]
        detections_field: fo.Detections | None = sample.get_field(in_field)

        if detections_field is None or detections_field.detections is None:
            logging.warning("Sample '%s' is missing field '%s'", sample.id, in_field)
            continue

        bbs = []
        if detections_field and detections_field.detections:
            for det in detections_field.detections:
                x, y, w, h = det.bounding_box  # relative [0, 1]
                conf = det.confidence if det.confidence is not None else 1.0
                bbs.append(([x * img_w, y * img_h, w * img_w, h * img_h], conf, det.label))

        tracks = tracker.update_tracks(bbs, frame=img)

        tracked = []
        for track in tracks:
            if not track.is_confirmed() or track.original_ltwh is None:
                continue

            lx, ly, lw, lh = track.original_ltwh
            rel_x = max(0.0, min(lx / img_w, 1.0))
            rel_y = max(0.0, min(ly / img_h, 1.0))
            rel_w = min(lw / img_w, 1.0 - rel_x)
            rel_h = min(lh / img_h, 1.0 - rel_y)

            out_det = fo.Detection(
                label=track.get_det_class() or "",
                bounding_box=[rel_x, rel_y, rel_w, rel_h],
                index=track.track_id,
            )
            if keep_confidence:
                out_det.confidence = track.get_det_conf()

            tracked.append(out_det)

        sample[out_field] = fo.Detections(detections=tracked)
        sample.save()
        written += 1

    return written


def run_tracking(
    collection: fo.Dataset | fo.DatasetView,
    in_field: str,
    out_field: str,
    max_age: int,
    keep_confidence: bool,
) -> None:
    samples = list(collection.iter_samples(progress=True))
    samples.sort(key=lambda s: (*_session_key(s), _frame_index(s)))

    total_sessions = 0
    total_samples  = 0

    for key, group in itertools.groupby(samples, key=_session_key):
        session_samples = list(group)
        written = track_session(
            session_samples, in_field, out_field, max_age, keep_confidence,
        )
        total_sessions += 1
        total_samples  += written
        logging.info("Session %s → %d samples tracked.", key, written)

    logging.info("Finished: %d sessions, %d samples total.", total_sessions, total_samples)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply DeepSort tracking to a FiftyOne image dataset.",
    )
    parser.add_argument(
        "--name",
        default=DATASET_NAME,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--in-field",
        default=IN_FIELD,
        help="Sample field containing input Detections (default: %(default)s)",
    )
    parser.add_argument(
        "--out-field",
        default=OUT_FIELD,
        help="Sample field to store tracked Detections (default: %(default)s)",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max missed frames before a track is dropped (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-confidence",
        action="store_true",
        help="Store detection confidence on tracked objects",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Restrict to a split ('train', 'val', 'test'). "
             "Pass an empty string to run on the full dataset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not fo.dataset_exists(args.name):
        raise SystemExit(f"Dataset '{args.name}' does not exist.")

    dataset = fo.load_dataset(args.name)

    collection: fo.Dataset | fo.DatasetView = dataset
    if args.split:
        collection = dataset.match(fo.ViewField("split") == args.split)
        logging.info("Filtered to split='%s': %d samples.", args.split, len(collection))

    logging.info(
        "Tracking %d samples  in_field=%s  out_field=%s  max_age=%d",
        len(collection), args.in_field, args.out_field, args.max_age,
    )

    run_tracking(
        collection,
        in_field=args.in_field,
        out_field=args.out_field,
        max_age=args.max_age,
        keep_confidence=args.keep_confidence,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
