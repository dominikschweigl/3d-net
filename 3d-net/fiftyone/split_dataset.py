"""Assign train/val/test splits to the 3d_net_rgbt FiftyOne dataset.

Each scene listed in SCENES is split independently 70 / 15 / 15 so that
temporal context is preserved per scene and no scene bleeds into another.
Use --scene-indices to process only a subset of SCENES (1-based).

Filtering:
  --dates  restricts processing to samples whose ``date`` field (YY_MM_DD)
           matches one of the given values, e.g. --dates 05_05_26.

The split is stored in the ``split`` field on each sample.  Running the script
again overwrites any previous split assignment.
"""

from __future__ import annotations

import argparse
import logging
from typing import NamedTuple

import fiftyone as fo


DATASET_NAME = "3d_net_rgbt"
SPLIT_FIELD  = "split"

# ---------------------------------------------------------------------------
# Scene definitions  (session, first_frame, last_frame)
# ---------------------------------------------------------------------------
class Scene(NamedTuple):
    session: str
    start:   int
    end:     int
    label:   str = ""   # optional human-readable note


SCENES: list[Scene] = [
    # --- aufnahme5 (20_01_26) ---
    Scene("aufnahme5",   801, 17946),
    # --- aufnahme2 (05_05_26) ---
    Scene("aufnahme2",   648,  2562),
    Scene("aufnahme2",  2565,  4140),
    Scene("aufnahme2",  4143,  4890),
    Scene("aufnahme2",  4893,  7029),
    Scene("aufnahme2",  7437,  9684),
    # --- aufnahme1 (05_05_26) ---
    Scene("aufnahme1",  5763,  6810),
    Scene("aufnahme1",  6813,  7113),
    Scene("aufnahme1",  7116,  8841),
    Scene("aufnahme1",  8844, 10266),
    Scene("aufnahme1", 10269, 10422),
    Scene("aufnahme1", 10425, 12189),
    Scene("aufnahme1", 12192, 14527),
    Scene("aufnahme1", 14530, 17103),
    Scene("aufnahme1", 17106, 17434),
    Scene("aufnahme1", 17437, 17862),
]

log = logging.getLogger(__name__)


def _split_samples(samples: list) -> dict[str, int]:
    """Apply 70/15/15 split in-place and return counts."""
    n       = len(samples)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)   # 70 % + 15 %
    counts  = {"train": 0, "val": 0, "test": 0}
    for i, sample in enumerate(samples):
        split = "train" if i < n_train else ("val" if i < n_val else "test")
        sample[SPLIT_FIELD] = split
        sample.save()
        counts[split] += 1
    return counts


def assign_splits_by_scenes(
    dataset: fo.Dataset,
    scenes: list[Scene],
    dates: list[str] | None = None,
) -> dict[str, int]:
    """Split each scene independently 70/15/15."""
    totals: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for scene in scenes:
        filter_expr = (
            (fo.ViewField("session") == scene.session)
            & (fo.ViewField("frame_index") >= scene.start)
            & (fo.ViewField("frame_index") <= scene.end)
        )
        if dates:
            filter_expr = fo.ViewField("date").is_in(dates) & filter_expr

        samples = sorted(
            dataset.match(filter_expr).iter_samples(progress=True),
            key=lambda s: s.get_field("frame_index") or 0,
        )

        note = f" ({scene.label})" if scene.label else ""
        if not samples:
            log.warning(
                "Scene %s:%d-%d%s matched 0 samples — skipping.",
                scene.session, scene.start, scene.end, note,
            )
            continue

        log.info(
            "Scene %s:%d-%d%s  →  %d samples",
            scene.session, scene.start, scene.end, note, len(samples),
        )
        counts = _split_samples(samples)
        for k, v in counts.items():
            totals[k] += v

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign train/val/test splits to the 3d_net_rgbt dataset.",
    )
    parser.add_argument(
        "--name",
        default=DATASET_NAME,
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        metavar="YY_MM_DD",
        help="Only process samples whose 'date' field matches one of these values, "
             "e.g. --dates 05_05_26",
    )
    parser.add_argument(
        "--scene-indices",
        nargs="+",
        type=int,
        metavar="N",
        help="1-based indices into SCENES to process. Omit to process all scenes.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not fo.dataset_exists(args.name):
        raise SystemExit(f"Dataset '{args.name}' does not exist.")

    dataset = fo.load_dataset(args.name)

    if SPLIT_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(SPLIT_FIELD, fo.StringField)
        dataset.save()
        log.info("Added '%s' field to dataset schema.", SPLIT_FIELD)

    scenes = SCENES
    if args.scene_indices:
        scenes = [SCENES[i - 1] for i in args.scene_indices]

    log.info(
        "Processing %d scene(s)%s",
        len(scenes),
        f", filtered to dates {args.dates}" if args.dates else "",
    )
    counts = assign_splits_by_scenes(dataset, scenes, dates=args.dates)

    total = sum(counts.values())
    if total == 0:
        log.warning("No samples were processed.")
        return

    log.info(
        "Done.  train=%d (%.1f %%)  val=%d (%.1f %%)  test=%d (%.1f %%)",
        counts["train"], 100 * counts["train"] / total,
        counts["val"],   100 * counts["val"]   / total,
        counts["test"],  100 * counts["test"]  / total,
    )


if __name__ == "__main__":
    main()
