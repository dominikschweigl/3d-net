"""Merge global track indices that were determined to be the same physical object.

Some tracks span multiple annotation runs and got assigned different global
indices by assign_global_track_ids.py.  This script remaps those indices so
every pair of indices that belongs to the same object collapses to the lower
of the two values.

Edit MERGE_PAIRS below if the list changes, then run with --dry-run first.
"""

from __future__ import annotations

import argparse
import logging

import fiftyone as fo


DATASET_NAME = "3d_net_rgbt"
LABEL_FIELD  = "ground_truth"

# Each tuple (a, b) means index b should be renamed to a (a < b).
MERGE_PAIRS: list[tuple[int, int]] = [
    (1,  2),
    (6,  8),
    (7,  9),
    (18, 19),
    (33, 34),
    (39, 40),
    (41, 44),
    (42, 45),
    (43, 46),
    (51, 52),
    (54, 55),
    (64, 65),
    (70, 71),
    (74, 75),
    (87, 88),
    (92, 93),
]


def build_remap(pairs: list[tuple[int, int]]) -> dict[int, int]:
    """Return {old_index: new_index} for every index that must change."""
    remap: dict[int, int] = {}
    for keep, drop in pairs:
        if drop in remap:
            logging.warning(
                "Index %d already remapped to %d; skipping duplicate rule (%d→%d).",
                drop, remap[drop], drop, keep,
            )
            continue
        remap[drop] = keep
    return remap


def apply_remap(
    collection: fo.Dataset | fo.DatasetView,
    label_field: str,
    remap: dict[int, int],
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Rewrite indices in-place.  Returns (samples_changed, detections_changed)."""
    samples_changed = 0
    detections_changed = 0

    for sample in collection.iter_samples(progress=True):
        detections_field: fo.Detections | None = sample.get_field(label_field)
        if not detections_field or not detections_field.detections:
            continue

        changed = False
        for det in detections_field.detections:
            if det.index is None:
                continue
            new_idx = remap.get(det.index)
            if new_idx is not None:
                if not dry_run:
                    det.index = new_idx
                detections_changed += 1
                changed = True

        if changed and not dry_run:
            sample[label_field] = detections_field
            sample.save()
            samples_changed += 1
        elif changed:
            samples_changed += 1

    return samples_changed, detections_changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge global track indices that belong to the same object.",
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
        help="Print what would change without writing anything.",
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

    remap = build_remap(MERGE_PAIRS)

    print("\nIndex remapping to apply:")
    print(f"  {'Old':>6}  →  {'New':>6}")
    print(f"  {'-'*6}     {'-'*6}")
    for old, new in sorted(remap.items()):
        print(f"  {old:>6}  →  {new:>6}")

    if args.dry_run:
        logging.info("Dry run — scanning without writing...")
        samples_changed, dets_changed = apply_remap(
            collection, args.field, remap, dry_run=True
        )
        logging.info(
            "Would update %d detections across %d samples.",
            dets_changed, samples_changed,
        )
        return

    logging.info("Applying remapping...")
    samples_changed, dets_changed = apply_remap(
        collection, args.field, remap, dry_run=False
    )
    logging.info(
        "Done. %d detections updated across %d samples.",
        dets_changed, samples_changed,
    )


if __name__ == "__main__":
    main()
