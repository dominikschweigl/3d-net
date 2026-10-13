"""Merge source RGBT datasets into the 3d_net_data_pool_rgbt data pool.

Sets the pool's default_classes to the canonical six-class taxonomy and merges
all samples from the following source datasets:
  - rtdod_rgbt
  - 3d_net_rgbt
  - m3fd_rgbt
  - visdrone_dronevehicle_rgbt

Merge is idempotent: FiftyOne uses filepath as the key, so re-running will
upsert rather than duplicate existing samples.
"""

from __future__ import annotations

import logging

import fiftyone as fo

POOL_NAME = "3d_net_data_pool_rgbt"
POOL_CLASSES = ("person", "car", "motorcycle", "bicycle", "bus", "truck")
SOURCE_DATASETS = (
    "rtdod_rgbt",
    "3d_net_rgbt",
    "m3fd_rgbt",
    "visdrone_dronevehicle_rgbt",
)


def _prepare_pool() -> fo.Dataset:
    """Load the pool dataset (creating it if necessary) and set its classes."""
    if fo.dataset_exists(POOL_NAME):
        dataset = fo.load_dataset(POOL_NAME)
        logging.info("Loaded existing pool dataset '%s' (%d samples)", POOL_NAME, len(dataset))
    else:
        dataset = fo.Dataset(POOL_NAME)
        dataset.add_sample_field("thermal", fo.StringField)
        dataset.add_sample_field("split", fo.StringField)
        dataset.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
        dataset.add_sample_field("source_dataset", fo.StringField)
        if "thermal" not in dataset.app_config.media_fields:
            dataset.app_config.media_fields.append("thermal")
        logging.info("Created new pool dataset '%s'", POOL_NAME)

    dataset.default_classes = list(POOL_CLASSES)
    logging.info("Set default_classes to %s", POOL_CLASSES)
    return dataset


def _merge_source(pool: fo.Dataset, source_name: str) -> None:
    if not fo.dataset_exists(source_name):
        logging.warning("Source dataset '%s' does not exist — skipping", source_name)
        return

    source = fo.load_dataset(source_name)
    source_filepaths = source.values("filepath")
    before = len(pool)
    pool.merge_samples(source)
    added = len(pool) - before
    pool.select_by("filepath", source_filepaths).set_field("source_dataset", source_name).save()
    logging.info(
        "Merged '%s' (%d samples) into pool — %d new samples added",
        source_name,
        len(source),
        added,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("FiftyOne API URI: %s", fo.config.api_uri)

    pool = _prepare_pool()

    for source_name in SOURCE_DATASETS:
        _merge_source(pool, source_name)

    pool.persistent = True
    pool.save()
    logging.info(
        "Done. Pool dataset '%s' now contains %d samples.", POOL_NAME, len(pool)
    )


if __name__ == "__main__":
    main()
