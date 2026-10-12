"""Build a FiftyOne dataset for RTDOD RGB+thermal data.

Expected dataset layout under `detection/data/rtdod`:
- `rgb/`: visible-spectrum images
- `trm/`: thermal/infrared images
- `annotations/`: Pascal VOC-style XML annotations
- `imagesets/`: optional split files (train.txt, val.txt, test.txt, trainval.txt)

Pairing is done by filename stem across all three sources.

Only the following classes are included in the FiftyOne output:
  person, car, motorcycle, bus, truck, bicycle
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import boto3
import fiftyone as fo

try:
    from .rtdod_xml_parser import RTDODXmlParser
except ImportError:  # pragma: no cover
    from detection.fiftyone.fiftyone_builder.rtdod_xml_parser import RTDODXmlParser


DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "rtdod"
DEFAULT_S3_CACHE_DIR = Path(__file__).resolve().parent / "_s3_cache" / "rtdod"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
XML_PARSER = RTDODXmlParser(unknown_policy="skip")

# Classes kept in the FiftyOne output; all others are silently dropped.
OUTPUT_CLASSES: Set[str] = {"person", "car", "motorcycle", "bus", "truck", "bicycle"}

# Condition codes derived from the filename prefix (e.g. D_0000_000000 → "day").
# Longer codes must come first so "NR" is matched before "N".
_CONDITION_MAP: Dict[str, str] = {
    "NR": "night_rain",
    "D":  "day",
    "N":  "night",
    "F":  "fog",
    "E":  "evening",
    "R":  "rain",
}


def _condition_from_stem(stem: str) -> str:
    """Return a human-readable condition label from a filename stem like 'NR_0008_000050'."""
    prefix = stem.split("_")[0].upper()
    return _CONDITION_MAP.get(prefix, "unknown")


def _keep_label(label: str) -> Optional[str]:
    """Return the label unchanged if it should be kept, else None."""
    return label if label in OUTPUT_CLASSES else None


def _stems_with_extensions(directory: Path, extensions: Sequence[str]) -> Set[str]:
    stems: Set[str] = set()
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in extensions:
            stems.add(p.stem)
    return stems


def _xml_stems(directory: Path) -> Set[str]:
    return {p.stem for p in directory.glob("*.xml")}


def _warn_id_mismatches(*, rgb_ids: Set[str], trm_ids: Set[str], xml_ids: Set[str]) -> None:
    common = rgb_ids & trm_ids & xml_ids
    union = rgb_ids | trm_ids | xml_ids
    if common == union:
        return

    def _examples(values: Set[str], limit: int = 8) -> str:
        if not values:
            return ""
        shown = sorted(values)[:limit]
        more = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
        return ", ".join(shown) + more

    missing_rgb = union - rgb_ids
    missing_trm = union - trm_ids
    missing_xml = union - xml_ids

    logging.warning(
        "File/label mismatch detected. Common=%d, RGB=%d, TRM=%d, XML=%d. "
        "Missing: RGB=%d [%s]; TRM=%d [%s]; XML=%d [%s]",
        len(common),
        len(rgb_ids),
        len(trm_ids),
        len(xml_ids),
        len(missing_rgb),
        _examples(missing_rgb),
        len(missing_trm),
        _examples(missing_trm),
        len(missing_xml),
        _examples(missing_xml),
    )


def _resolve_image_by_stem(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _get_s3_client(endpoint_url: str, region_name: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )


def _list_s3_stem_to_key(
    *,
    s3_client,
    bucket: str,
    prefix: str,
    suffixes: Sequence[str],
) -> Dict[str, str]:
    suffixes_lc = tuple(s.lower() for s in suffixes)
    stem_to_key: Dict[str, str] = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    page_count = 0
    key_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        page_count += 1
        if page_count % 20 == 0:
            logging.info(
                "S3 listing progress for prefix '%s': %d pages, %d matched files",
                prefix,
                page_count,
                key_count,
            )
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if not key:
                continue
            if not key.lower().endswith(suffixes_lc):
                continue
            stem = Path(key).stem
            if stem not in stem_to_key:
                stem_to_key[stem] = key
                key_count += 1

    logging.info(
        "Finished S3 listing for prefix '%s': %d pages, %d matched files",
        prefix,
        page_count,
        key_count,
    )
    return stem_to_key


def _download_s3_file(*, s3_client, bucket: str, key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    s3_client.download_file(bucket, key, str(local_path))
    return local_path


def _try_download_s3_file(*, s3_client, bucket: str, key: str, local_path: Path) -> Optional[Path]:
    try:
        return _download_s3_file(s3_client=s3_client, bucket=bucket, key=key, local_path=local_path)
    except Exception:  # noqa: BLE001
        return None


def available_ids_s3(
    *,
    s3_client,
    bucket: str,
    rgb_prefix: str,
    trm_prefix: str,
    xml_prefix: str,
) -> tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    rgb_stem_to_key = _list_s3_stem_to_key(
        s3_client=s3_client, bucket=bucket, prefix=rgb_prefix, suffixes=IMAGE_EXTENSIONS,
    )
    trm_stem_to_key = _list_s3_stem_to_key(
        s3_client=s3_client, bucket=bucket, prefix=trm_prefix, suffixes=IMAGE_EXTENSIONS,
    )
    xml_stem_to_key = _list_s3_stem_to_key(
        s3_client=s3_client, bucket=bucket, prefix=xml_prefix, suffixes=(".xml",),
    )

    rgb_ids = set(rgb_stem_to_key)
    trm_ids = set(trm_stem_to_key)
    xml_ids = set(xml_stem_to_key)
    _warn_id_mismatches(rgb_ids=rgb_ids, trm_ids=trm_ids, xml_ids=xml_ids)
    ids = sorted(rgb_ids & trm_ids & xml_ids)
    return ids, rgb_stem_to_key, trm_stem_to_key, xml_stem_to_key


def available_ids(annotations_dir: Path, rgb_dir: Path, trm_dir: Path) -> List[str]:
    rgb_ids = _stems_with_extensions(rgb_dir, IMAGE_EXTENSIONS)
    trm_ids = _stems_with_extensions(trm_dir, IMAGE_EXTENSIONS)
    xml_ids = _xml_stems(annotations_dir)
    _warn_id_mismatches(rgb_ids=rgb_ids, trm_ids=trm_ids, xml_ids=xml_ids)
    return sorted(rgb_ids & trm_ids & xml_ids)


def _read_split_file(path: Path) -> Set[str]:
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            ids.add(Path(value).stem)
    return ids


def _build_split_map(
    *,
    ids: Iterable[str],
    train_ids_file: Optional[Path],
    val_ids_file: Optional[Path],
    test_ids_file: Optional[Path],
    default_split: str = "all",
) -> Dict[str, str]:
    split_map: Dict[str, str] = {stem: default_split for stem in ids}

    def apply_split(path: Optional[Path], split_name: str) -> None:
        if path is None:
            return
        if not path.exists():
            raise FileNotFoundError(f"Split file does not exist: {path}")
        for stem in _read_split_file(path):
            if stem in split_map:
                split_map[stem] = split_name

    apply_split(train_ids_file, "train")
    apply_split(val_ids_file, "val")
    apply_split(test_ids_file, "test")
    return split_map


def _default_split_file(imagesets_dir: Path, filename: str) -> Optional[Path]:
    candidate = imagesets_dir / filename
    return candidate if candidate.exists() else None


class RTDODFiftyOneBuilder:
    """Builds a FiftyOne dataset for paired RTDOD RGB/thermal images."""

    def __init__(
        self,
        *,
        data_root: Path = DATA_ROOT,
        parser: Optional[RTDODXmlParser] = None,
    ):
        self.data_root = Path(data_root)
        self.annotations_dir = self.data_root / "annotations"
        self.rgb_dir = self.data_root / "rgb"
        self.trm_dir = self.data_root / "trm"
        self.imagesets_dir = self.data_root / "imagesets"
        self.parser = parser or XML_PARSER

    def _validate_paths(self) -> None:
        for path in (self.annotations_dir, self.rgb_dir, self.trm_dir):
            if not path.exists():
                raise FileNotFoundError(f"Expected path missing: {path}")

    def prepare_dataset(self, name: str, overwrite: bool) -> fo.Dataset:
        if overwrite and fo.dataset_exists(name):
            fo.delete_dataset(name)
        elif fo.dataset_exists(name):
            logging.info("Dataset '%s' already exists; loading it", name)
            return fo.load_dataset(name)

        dataset = fo.load_dataset(name, create_if_necessary=True)
        dataset.add_sample_field("thermal", fo.StringField)
        dataset.add_sample_field("split", fo.StringField)
        dataset.add_sample_field("lighting_condition", fo.StringField)
        dataset.add_sample_field("ground_truth", fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
        dataset.default_classes = sorted(OUTPUT_CLASSES)

        if "thermal" not in dataset.app_config.media_fields:
            dataset.app_config.media_fields.append("thermal")
        
        return dataset

    def build(
        self,
        *,
        name: str,
        overwrite: bool = False,
        train_ids_file: Optional[Path] = None,
        val_ids_file: Optional[Path] = None,
        test_ids_file: Optional[Path] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "rtdod",
        s3_endpoint_url: str = "https://s3.de-west-1.psmanaged.com",
        s3_region: str = "de-west-1",
        s3_cache_dir: Path = DEFAULT_S3_CACHE_DIR,
    ) -> fo.Dataset:
        if not s3_bucket:
            self._validate_paths()
        dataset = self.prepare_dataset(name=name, overwrite=overwrite)

        if s3_bucket:
            logging.info(
                "Using S3 source: bucket=%s, prefix=%s, endpoint=%s",
                s3_bucket,
                s3_prefix,
                s3_endpoint_url,
            )
            s3_client = _get_s3_client(endpoint_url=s3_endpoint_url, region_name=s3_region)
            base_prefix = s3_prefix.rstrip("/")
            rgb_prefix = f"{base_prefix}/rgb/"
            trm_prefix = f"{base_prefix}/trm/"
            xml_prefix = f"{base_prefix}/annotations/"

            if train_ids_file is None:
                train_ids_file = _try_download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=f"{base_prefix}/imagesets/train.txt",
                    local_path=s3_cache_dir / "imagesets" / "train.txt",
                )
            if val_ids_file is None:
                val_ids_file = _try_download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=f"{base_prefix}/imagesets/val.txt",
                    local_path=s3_cache_dir / "imagesets" / "val.txt",
                )
            if test_ids_file is None:
                test_ids_file = _try_download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=f"{base_prefix}/imagesets/test.txt",
                    local_path=s3_cache_dir / "imagesets" / "test.txt",
                )

            ids, rgb_stem_to_key, trm_stem_to_key, xml_stem_to_key = available_ids_s3(
                s3_client=s3_client,
                bucket=s3_bucket,
                rgb_prefix=rgb_prefix,
                trm_prefix=trm_prefix,
                xml_prefix=xml_prefix,
            )
            logging.info("Paired %d IDs from S3", len(ids))
        else:
            logging.info("Using local source: %s", self.data_root)
            if train_ids_file is None:
                train_ids_file = _default_split_file(self.imagesets_dir, "train.txt")
            if val_ids_file is None:
                val_ids_file = _default_split_file(self.imagesets_dir, "val.txt")
            if test_ids_file is None:
                test_ids_file = _default_split_file(self.imagesets_dir, "test.txt")

            ids = available_ids(
                annotations_dir=self.annotations_dir,
                rgb_dir=self.rgb_dir,
                trm_dir=self.trm_dir,
            )
            logging.info("Paired %d IDs from local dataset", len(ids))

        logging.info(
            "Split files: train=%s val=%s test=%s",
            train_ids_file,
            val_ids_file,
            test_ids_file,
        )

        split_map = _build_split_map(
            ids=ids,
            train_ids_file=train_ids_file,
            val_ids_file=val_ids_file,
            test_ids_file=test_ids_file,
        )

        samples: List[fo.Sample] = []
        skipped = 0
        total_ids = len(ids)

        for index, stem in enumerate(ids, start=1):
            if index == 1 or index % 500 == 0:
                logging.info("Processing samples: %d/%d", index, total_ids)

            if s3_bucket:
                rgb_key = rgb_stem_to_key.get(stem)
                trm_key = trm_stem_to_key.get(stem)
                xml_key = xml_stem_to_key.get(stem)
                if not rgb_key or not trm_key or not xml_key:
                    skipped += 1
                    continue

                rgb_path = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket}/{rgb_key}"
                trm_path = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket}/{trm_key}"
                xml_path = _download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=xml_key,
                    local_path=s3_cache_dir / "annotations" / f"{stem}.xml",
                )
            else:
                rgb_path = _resolve_image_by_stem(self.rgb_dir, stem)
                trm_path = _resolve_image_by_stem(self.trm_dir, stem)
                xml_path = self.annotations_dir / f"{stem}.xml"

            if rgb_path is None or trm_path is None or not xml_path.exists():
                skipped += 1
                continue

            img_w, img_h, objects = self.parser.parse(xml_path)

            detections: List[fo.Detection] = []
            for obj in objects:
                output_label = _keep_label(obj.label)
                if output_label is None:
                    continue

                x_min, y_min, x_max, y_max = obj.bbox
                width = x_max - x_min
                height = y_max - y_min
                if width <= 0 or height <= 0:
                    continue

                bbox = [x_min / img_w, y_min / img_h, width / img_w, height / img_h]
                detections.append(
                    fo.Detection(
                        label=output_label,
                        bounding_box=bbox,
                        truncated=obj.truncated,
                        difficult=obj.difficult,
                    )
                )

            split = split_map.get(stem, "all")
            sample = fo.Sample(filepath=str(rgb_path))
            sample["thermal"] = str(trm_path)
            sample["split"] = split
            sample.tags = [f"target:{split}"]
            sample["lighting_condition"] = _condition_from_stem(stem)
            sample["ground_truth"] = fo.Detections(detections=detections)
            samples.append(sample)

        if samples:
            dataset.add_samples(samples)

        logging.info(
            "Built RTDOD dataset '%s' with %d samples (skipped %d)",
            name,
            len(samples),
            skipped,
        )
        return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FiftyOne dataset from RTDOD RGB+thermal data",
    )
    parser.add_argument(
        "--name",
        default="rtdod_rgbt",
        help="Name of the FiftyOne dataset to create (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root of the RTDOD dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing FiftyOne dataset with the same name",
    )
    parser.add_argument(
        "--train-ids-file",
        type=Path,
        default=None,
        help="Train IDs file (defaults to <data-root>/imagesets/train.txt if present)",
    )
    parser.add_argument(
        "--val-ids-file",
        type=Path,
        default=None,
        help="Val IDs file (defaults to <data-root>/imagesets/val.txt if present)",
    )
    parser.add_argument(
        "--test-ids-file",
        type=Path,
        default=None,
        help="Test IDs file (defaults to <data-root>/imagesets/test.txt if present)",
    )

    # Optional S3 backend
    parser.add_argument(
        "--s3-bucket",
        default="bernard-3d-net",
        help="If set, load data from this S3 bucket instead of --data-root",
    )
    parser.add_argument(
        "--s3-prefix",
        default="rtdod",
        help="Prefix within the S3 bucket that contains RTDOD folders",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        default="https://s3.de-west-1.psmanaged.com",
        help="S3 endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--s3-region",
        default="de-west-1",
        help="S3 region name (default: %(default)s)",
    )
    parser.add_argument(
        "--s3-cache-dir",
        type=Path,
        default=DEFAULT_S3_CACHE_DIR,
        help="Local cache dir for S3 downloads (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    print(f"FiftyOne API URI: {fo.config.api_uri}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()

    builder = RTDODFiftyOneBuilder(data_root=args.data_root)
    dataset = builder.build(
        name=args.name,
        overwrite=args.overwrite,
        train_ids_file=args.train_ids_file,
        val_ids_file=args.val_ids_file,
        test_ids_file=args.test_ids_file,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint_url=args.s3_endpoint_url,
        s3_region=args.s3_region,
        s3_cache_dir=args.s3_cache_dir,
    )
    dataset.persistent = True
    dataset.save()


if __name__ == "__main__":
    main()
