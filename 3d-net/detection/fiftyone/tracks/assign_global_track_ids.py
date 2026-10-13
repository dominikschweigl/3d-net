"""Assign globally unique track indices across all CVAT tasks.

CVAT resets track indices to 1 per task, so the same index can appear in
multiple tasks.  This script:
  1. Collects all unique (run_key, local_index) pairs from the label field.
  2. Assigns each pair a new globally unique integer starting from 1.
  3. Writes the global index back to every Detection in the dataset.

Run with --dry-run first to preview the mapping without writing anything.
"""

from __future__ import annotations

import argparse
import logging

import fiftyone as fo


DATASET_NAME = "3d_net_rgbt"
LABEL_FIELD  = "ground_truth"


def build_sample_to_run_map(dataset: fo.Dataset) -> dict[str, str]:
    sample_to_run: dict[str, str] = {}
    for key in dataset.list_annotation_runs():
        view = dataset.load_annotation_view(key)
        for sample_id in view.values("id"):
            sample_to_run[sample_id] = key
    return sample_to_run


def build_global_id_map(
    collection: fo.Dataset | fo.DatasetView,
    label_field: str,
    sample_to_run: dict[str, str],
) -> dict[tuple, int]:
    """Return a mapping (run_key, local_index) → global_id."""
    seen: dict[tuple, int] = {}
    counter = 1

    for sample in collection.iter_samples(progress=True):
        detections_field: fo.Detections | None = sample.get_field(label_field)
        if not detections_field or not detections_field.detections:
            continue

        run_key = sample_to_run.get(sample.id, "<unknown>")

        for det in detections_field.detections:
            if det.index is None:
                continue
            key = (run_key, det.index)
            if key not in seen:
                seen[key] = counter
                counter += 1

    return seen


def apply_global_ids(
    collection: fo.Dataset | fo.DatasetView,
    label_field: str,
    sample_to_run: dict[str, str],
    global_id_map: dict[tuple, int],
) -> int:
    updated = 0

    for sample in collection.iter_samples(progress=True):
        detections_field: fo.Detections | None = sample.get_field(label_field)
        if not detections_field or not detections_field.detections:
            continue

        run_key = sample_to_run.get(sample.id, "<unknown>")
        changed = False

        for det in detections_field.detections:
            if det.index is None:
                continue
            global_id = global_id_map.get((run_key, det.index))
            if global_id is not None and det.index != global_id:
                det.index = global_id
                changed = True

        if changed:
            sample[label_field] = detections_field
            sample.save()
            updated += 1

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign globally unique track indices across CVAT tasks.",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the mapping without writing anything to the dataset.",
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

    logging.info("Building sample → annotation run map...")
    sample_to_run = build_sample_to_run_map(dataset)

    logging.info("Collecting unique (run, local_index) pairs...")
    global_id_map = build_global_id_map(collection, args.field, sample_to_run)

    col_w = max((len(r) for r, _ in global_id_map), default=8) + 2
    print(f"\n{'Run key':<{col_w}}  {'Local ID':>8}  {'Global ID':>9}")
    print("-" * (col_w + 8 + 9 + 4))
    for (run_key, local_idx), global_id in sorted(global_id_map.items()):
        print(f"{run_key:<{col_w}}  {local_idx:>8}  {global_id:>9}")
    print(f"\nTotal unique tracks: {len(global_id_map)}")

    if args.dry_run:
        logging.info("Dry run — no changes written.")
        return

    logging.info("Writing global indices to dataset...")
    updated = apply_global_ids(collection, args.field, sample_to_run, global_id_map)
    logging.info("Done. %d samples updated.", updated)


if __name__ == "__main__":
    main()
