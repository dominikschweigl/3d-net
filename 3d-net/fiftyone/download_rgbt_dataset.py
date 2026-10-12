"""Export the 3d_net_rgbt FiftyOne dataset to a local rgb/ + ir/ directory.

Uses the FiftyOne export() API so annotations are preserved alongside images.

Output structure:
    <output_dir>/rgb/       — RGB images (copied from FiftyOne filepath)
    <output_dir>/ir/        — IR images  (copied from FiftyOne thermal field)
    <output_dir>/labels.json — COCO annotations (ground_truth field)

RGB and IR files share the same filename stem so they can be matched by name.
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
from fiftyone import ViewField as F


# Maps CLI format name → (FiftyOne dataset type, labels output path relative to output_dir)
# COCO uses a single JSON file; VOC and YOLOv5 use a labels/ directory.
_FORMAT_CONFIGS: dict = {
    "coco":   (fot.COCODetectionDataset, "labels.json"),
    "voc":    (fot.VOCDetectionDataset,  "labels"),
    "yolov5": (fot.YOLOv5Dataset,        "labels"),
}


def export_dataset(
    *,
    dataset_name: str,
    output_dir: Path,
    label_field: str,
    fmt: str,
    split: str | None = None,
    index_only: bool = False,
) -> None:
    logging.info("Loading FiftyOne dataset '%s'...", dataset_name)
    dataset = fo.load_dataset(dataset_name)
    logging.info("Dataset has %d samples.", len(dataset))

    if split is not None:
        dataset = dataset.match(F("split") == split)
        logging.info("Filtered to split='%s': %d samples.", split, len(dataset))

    output_dir.mkdir(parents=True, exist_ok=True)

    if not index_only:
        dataset_type, labels_rel = _FORMAT_CONFIGS[fmt]
        rgb_dir     = str(output_dir / "rgb")
        ir_dir      = str(output_dir / "ir")
        labels_path = str(output_dir / labels_rel)

        # ── 1. Export RGB images + labels ─────────────────────────────────────
        logging.info("Exporting RGB images (%s format) to %s ...", fmt, rgb_dir)
        dataset.export(
            dataset_type=dataset_type,
            data_path=rgb_dir,
            labels_path=labels_path,
            label_field=label_field,
            export_media=True,
        )
        logging.info("RGB export done. Labels written to %s", labels_path)

        # ── 2. Export IR images ────────────────────────────────────────────────
        # set_field() creates a transient view where filepath = thermal without
        # persisting the change, so export() picks up the IR images.
        logging.info("Exporting IR images to %s ...", ir_dir)
        ir_view = dataset.set_field("filepath", F("thermal"))
        ir_view.export(
            dataset_type=fot.ImageDirectory,
            export_dir=ir_dir,
            export_media=True,
        )
        logging.info("IR export done.")

        # ── 3. Write split text files ──────────────────────────────────────────
        logging.info("Writing split files (train.txt, val.txt, test.txt) ...")
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for sample in dataset.iter_samples():
            split = getattr(sample, "split", None)
            if split in splits:
                # Keep extension so split entries match exported image filenames.
                splits[split].append(Path(sample.filepath).name)
        for split_name, filenames in splits.items():
            split_file = output_dir / f"{split_name}.txt"
            split_file.write_text("\n".join(filenames) + ("\n" if filenames else ""))
            logging.info("  %s: %d samples → %s", split_name, len(filenames), split_file)

    # ── 4. Stamp <index> (track ID) into VOC XML objects ──────────────────────
    if fmt == "voc" or index_only:
        logging.info("Stamping track <index> into VOC XML labels ...")
        _inject_index_into_xml(dataset, output_dir, label_field)

    logging.info("All done. Output at: %s", output_dir)


def _inject_index_into_xml(dataset, output_dir: Path, label_field: str) -> None:
    """Add <index> track-ID tags to every <object> in the exported VOC XML files.

    FiftyOne writes <object> elements in the same order as the detections list,
    so we zip them directly — no coordinate matching needed.
    """
    labels_dir = output_dir / "labels"
    skipped = 0
    for sample in dataset.iter_samples(progress=True):
        stem     = Path(sample.filepath).stem
        xml_path = labels_dir / f"{stem}.xml"
        if not xml_path.exists():
            continue

        dets = getattr(sample, label_field, None)
        if not dets or not dets.detections:
            continue

        indexed = [d for d in dets.detections if d.index is not None]
        if not indexed:
            continue

        tree    = ET.parse(xml_path)
        root    = tree.getroot()
        objects = root.findall("object")

        if len(objects) != len(indexed):
            logging.warning(
                "Object count mismatch in %s (%d XML vs %d indexed detections) — skipping",
                xml_path.name, len(objects), len(indexed),
            )
            skipped += 1
            continue

        for obj, det in zip(objects, indexed):
            for existing in obj.findall("index"):  # remove prior stamps to stay idempotent
                obj.remove(existing)
            idx_el      = ET.SubElement(obj, "index")
            idx_el.text = str(det.index)

        ET.indent(tree, space="    ")
        tree.write(str(xml_path), encoding="unicode", xml_declaration=False)

    if skipped:
        logging.warning("%d XML files skipped due to object count mismatches.", skipped)
    logging.info("Track index injection done.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 3d_net_rgbt FiftyOne dataset to local rgb/ and ir/ directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory (will contain rgb/, ir/, and label files)",
    )
    parser.add_argument(
        "--dataset-name",
        default="3d_net_rgbt",
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--label-field",
        default="ground_truth",
        help="Label field to export (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=list(_FORMAT_CONFIGS),
        default="voc",
        help="Annotation format: coco (labels.json), voc (labels/ XML), yolov5 (labels/ TXT) (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Export only samples from this split (default: all splits)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip image/label export; only stamp <index> into existing VOC XML files",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    export_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output,
        label_field=args.label_field,
        fmt=args.fmt,
        split=args.split,
        index_only=args.index_only,
    )


if __name__ == "__main__":
    main()
