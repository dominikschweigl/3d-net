"""List all CVAT rectangle tracks stored in a FiftyOne dataset.

A track is uniquely identified by (annotation_run_key, index) because CVAT
resets track indices to 1 for every task.  The script builds a reverse mapping
from sample_id → run_key so each detection's index can be scoped correctly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import fiftyone as fo


DATASET_NAME = "3d_net_rgbt"
LABEL_FIELD  = "ground_truth"


def build_sample_to_run_map(dataset: fo.Dataset) -> dict[str, str]:
    """Map each sample ID to the annotation run key it belongs to."""
    sample_to_run: dict[str, str] = {}
    for key in dataset.list_annotation_runs():
        view = dataset.load_annotation_view(key)
        for sample_id in view.values("id"):
            sample_to_run[sample_id] = key
    return sample_to_run


def collect_tracks(
    collection: fo.Dataset | fo.DatasetView,
    label_field: str,
    sample_to_run: dict[str, str],
) -> dict[tuple, dict]:
    """Return a dict keyed by (run_key, track_index) with per-track metadata."""
    tracks: dict[tuple, dict] = defaultdict(lambda: {
        "label": None,
        "frame_count": 0,
        "session": None,
    })

    for sample in collection.iter_samples(progress=True):
        detections_field: fo.Detections | None = sample.get_field(label_field)
        if not detections_field or not detections_field.detections:
            continue

        run_key = sample_to_run.get(sample.id, "<unknown>")
        session = "/".join(filter(None, [
            sample.get_field("location") or "",
            sample.get_field("date")     or "",
            sample.get_field("session")  or "",
        ]))

        for det in detections_field.detections:
            if det.index is None:
                continue
            track = tracks[(run_key, det.index)]
            track["frame_count"] += 1
            if track["label"] is None:
                track["label"] = det.label
            if track["session"] is None:
                track["session"] = session

    return dict(tracks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List all CVAT tracks in a FiftyOne dataset.",
    )
    parser.add_argument(
        "--name",
        default=DATASET_NAME,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--field",
        default=LABEL_FIELD,
        help="Sample field containing Detections (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Restrict to a split ('train', 'val', 'test').",
    )
    args = parser.parse_args()

    if not fo.dataset_exists(args.name):
        raise SystemExit(f"Dataset '{args.name}' does not exist.")

    dataset = fo.load_dataset(args.name)

    collection: fo.Dataset | fo.DatasetView = dataset
    if args.split:
        collection = dataset.match(fo.ViewField("split") == args.split)

    sample_to_run = build_sample_to_run_map(dataset)

    tracks = collect_tracks(collection, args.field, sample_to_run)

    if not tracks:
        print("No tracks found.")
        return

    col_w = max(len(k) for k, _ in tracks) + 2
    print(f"\n{'Run key':<{col_w}}  {'ID':>5}  {'Label':<20}  {'Frames':>6}  Session")
    print("-" * (col_w + 5 + 20 + 6 + 30))
    for (run_key, idx), info in sorted(tracks.items()):
        print(
            f"{run_key:<{col_w}}  {idx:>5}  {info['label']:<20}"
            f"  {info['frame_count']:>6}  {info['session'] or ''}"
        )

    print(f"\nTotal tracks: {len(tracks)}")


if __name__ == "__main__":
    main()
