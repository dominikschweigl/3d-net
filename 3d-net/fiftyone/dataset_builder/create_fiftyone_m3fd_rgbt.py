"""Build a FiftyOne dataset for M3FD RGB+thermal data.

Expected dataset layout under `detection/data/m3fd`:
- `Vis/`: visible-spectrum images
- `Ir/`: thermal/infrared images
- `Annotation/`: Pascal VOC-style XML annotations

In addition to the above directories, split files (`train.txt`, `val.txt`, `test.txt`) 
may be present that list the filenames for each split. If not provided, 
all samples will be assigned to a default "all" split.

For this project the M3FD-zxSplit was used, which avoids data 
leakage by splitting scenes rather than individual images.
This split can be found in the paper "Rethinking Early-Fusion Strategies for 
Improved Multispectral Object Detection" (Zhang et al., ECCV 2024) and the 
corresponding code repository:
https://github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection

Pairing is done by filename stem across all three sources.
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
    from .m3fd_xml_parser import M3FDXmlParser
except ImportError:  # pragma: no cover
    # Allows running as a standalone script
    from detection.fiftyone.fiftyone_builder.m3fd_xml_parser import M3FDXmlParser


DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "m3fd"
DEFAULT_S3_CACHE_DIR = Path(__file__).resolve().parent / "_s3_cache" / "m3fd"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
XML_PARSER = M3FDXmlParser(unknown_policy="skip")


def _normalize_output_label(label: str) -> Optional[str]:
    """Normalizes class names for FiftyOne output.

    - convert to lowercase
    - rename "People" to "person"
    - exclude "Lamp" from output
    """
    normalized = label.strip().lower()
    if normalized == "people":
        normalized = "person"
    if normalized == "lamp":
        return None
    return normalized


def _stems_with_extensions(directory: Path, extensions: Sequence[str]) -> Set[str]:
    stems: Set[str] = set()
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in extensions:
            stems.add(p.stem)
    return stems


def _xml_stems(directory: Path) -> Set[str]:
    return {p.stem for p in directory.glob("*.xml")}


def _warn_id_mismatches(*, vis_ids: Set[str], ir_ids: Set[str], xml_ids: Set[str]) -> None:
    common = vis_ids & ir_ids & xml_ids
    union = vis_ids | ir_ids | xml_ids
    if common == union:
        return

    def _examples(values: Set[str], limit: int = 8) -> str:
        if not values:
            return ""
        shown = sorted(values)[:limit]
        more = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
        return ", ".join(shown) + more

    missing_vis = union - vis_ids
    missing_ir = union - ir_ids
    missing_xml = union - xml_ids

    logging.warning(
        "File/label mismatch detected. Common=%d, Vis=%d, Ir=%d, XML=%d. "
        "Missing: Vis=%d [%s]; Ir=%d [%s]; XML=%d [%s]",
        len(common),
        len(vis_ids),
        len(ir_ids),
        len(xml_ids),
        len(missing_vis),
        _examples(missing_vis),
        len(missing_ir),
        _examples(missing_ir),
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

            key_lc = key.lower()
            if not key_lc.endswith(suffixes_lc):
                continue

            stem = Path(key).stem
            # Keep first key for a stem to avoid non-deterministic overrides.
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
    vis_prefix: str,
    ir_prefix: str,
    xml_prefix: str,
) -> tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    vis_stem_to_key = _list_s3_stem_to_key(
        s3_client=s3_client,
        bucket=bucket,
        prefix=vis_prefix,
        suffixes=IMAGE_EXTENSIONS,
    )
    ir_stem_to_key = _list_s3_stem_to_key(
        s3_client=s3_client,
        bucket=bucket,
        prefix=ir_prefix,
        suffixes=IMAGE_EXTENSIONS,
    )
    xml_stem_to_key = _list_s3_stem_to_key(
        s3_client=s3_client,
        bucket=bucket,
        prefix=xml_prefix,
        suffixes=(".xml",),
    )

    vis_ids = set(vis_stem_to_key)
    ir_ids = set(ir_stem_to_key)
    xml_ids = set(xml_stem_to_key)
    _warn_id_mismatches(vis_ids=vis_ids, ir_ids=ir_ids, xml_ids=xml_ids)
    ids = sorted(vis_ids & ir_ids & xml_ids)
    return ids, vis_stem_to_key, ir_stem_to_key, xml_stem_to_key


def available_ids(annotations_dir: Path, vis_dir: Path, ir_dir: Path) -> List[str]:
    vis_ids = _stems_with_extensions(vis_dir, IMAGE_EXTENSIONS)
    ir_ids = _stems_with_extensions(ir_dir, IMAGE_EXTENSIONS)
    xml_ids = _xml_stems(annotations_dir)
    _warn_id_mismatches(vis_ids=vis_ids, ir_ids=ir_ids, xml_ids=xml_ids)
    return sorted(vis_ids & ir_ids & xml_ids)


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

        stem_ids = _read_split_file(path)
        for stem in stem_ids:
            if stem in split_map:
                split_map[stem] = split_name

    # Apply in order so later files can override earlier assignments.
    apply_split(train_ids_file, "train")
    apply_split(val_ids_file, "val")
    apply_split(test_ids_file, "test")
    return split_map


def _default_split_file(data_root: Path, filename: str) -> Optional[Path]:
    """Returns a default split file path when it exists."""
    candidate = data_root / filename
    return candidate if candidate.exists() else None


class M3FDFiftyOneBuilder:
    """Builds a FiftyOne dataset for paired M3FD VIS/IR images."""

    def __init__(
        self,
        *,
        data_root: Path = DATA_ROOT,
        parser: Optional[M3FDXmlParser] = None,
    ):
        self.data_root = Path(data_root)
        self.annotations_dir = self.data_root / "Annotation"
        self.vis_dir = self.data_root / "Vis"
        self.ir_dir = self.data_root / "Ir"
        self.parser = parser or XML_PARSER

    def _validate_paths(self) -> None:
        for path in (self.annotations_dir, self.vis_dir, self.ir_dir):
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
        dataset.add_sample_field("ground_truth", fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
        dataset.default_classes = [
            normalized
            for class_name in self.parser.class_names
            if (normalized := _normalize_output_label(class_name)) is not None
        ]

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
        s3_prefix: str = "m3fd",
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
            vis_prefix = f"{base_prefix}/Vis/"
            ir_prefix = f"{base_prefix}/Ir/"
            xml_prefix = f"{base_prefix}/Annotation/"

            if train_ids_file is None:
                train_ids_file = _try_download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=f"{base_prefix}/train.txt",
                    local_path=s3_cache_dir / "train.txt",
                )
            if val_ids_file is None:
                val_ids_file = _try_download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=f"{base_prefix}/val.txt",
                    local_path=s3_cache_dir / "val.txt",
                )
            if test_ids_file is None:
                test_ids_file = _try_download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=f"{base_prefix}/test.txt",
                    local_path=s3_cache_dir / "test.txt",
                )

            ids, vis_stem_to_key, ir_stem_to_key, xml_stem_to_key = available_ids_s3(
                s3_client=s3_client,
                bucket=s3_bucket,
                vis_prefix=vis_prefix,
                ir_prefix=ir_prefix,
                xml_prefix=xml_prefix,
            )
            logging.info("Paired %d IDs from S3", len(ids))
        else:
            logging.info("Using local source: %s", self.data_root)
            if train_ids_file is None:
                train_ids_file = _default_split_file(self.data_root, "train.txt")
            if val_ids_file is None:
                val_ids_file = _default_split_file(self.data_root, "val.txt")
            if test_ids_file is None:
                test_ids_file = _default_split_file(self.data_root, "test.txt")

            ids = available_ids(
                annotations_dir=self.annotations_dir,
                vis_dir=self.vis_dir,
                ir_dir=self.ir_dir,
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
                vis_key = vis_stem_to_key.get(stem)
                ir_key = ir_stem_to_key.get(stem)
                xml_key = xml_stem_to_key.get(stem)
                if not vis_key or not ir_key or not xml_key:
                    skipped += 1
                    continue

                vis_path = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket}/{vis_key}"
                ir_path = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket}/{ir_key}"
                xml_path = _download_s3_file(
                    s3_client=s3_client,
                    bucket=s3_bucket,
                    key=xml_key,
                    local_path=s3_cache_dir / "Annotation" / f"{stem}.xml",
                )
            else:
                vis_path = _resolve_image_by_stem(self.vis_dir, stem)
                ir_path = _resolve_image_by_stem(self.ir_dir, stem)
                xml_path = self.annotations_dir / f"{stem}.xml"

            if vis_path is None or ir_path is None or not xml_path.exists():
                skipped += 1
                continue

            img_w, img_h, objects = self.parser.parse(xml_path)

            detections: List[fo.Detection] = []
            for obj in objects:
                output_label = _normalize_output_label(obj.label)
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
            sample = fo.Sample(filepath=str(vis_path))
            sample["thermal"] = str(ir_path)
            sample["split"] = split
            sample.tags = [f"target:{split}"]
            sample["ground_truth"] = fo.Detections(detections=detections)
            samples.append(sample)

        if samples:
            dataset.add_samples(samples)

        logging.info(
            "Built M3FD dataset '%s' with %d samples (skipped %d)",
            name,
            len(samples),
            skipped,
        )
        return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FiftyOne dataset from M3FD RGB+thermal data",
    )
    parser.add_argument(
        "--name",
        default="m3fd_rgbt",
        help="Name of the FiftyOne dataset to create (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root of the M3FD dataset (default: %(default)s)",
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
        help="Train IDs file (defaults to <data-root>/train.txt if present)",
    )
    parser.add_argument(
        "--val-ids-file",
        type=Path,
        default=None,
        help="Val IDs file (defaults to <data-root>/val.txt if present)",
    )
    parser.add_argument(
        "--test-ids-file",
        type=Path,
        default=None,
        help="Test IDs file (defaults to <data-root>/test.txt if present)",
    )

    # Optional S3 backend
    parser.add_argument(
        "--s3-bucket",
        default="bernard-3d-net",
        help="If set, load data from this S3 bucket instead of --data-root",
    )
    parser.add_argument(
        "--s3-prefix",
        default="m3fd",
        help="Prefix within the S3 bucket that contains M3FD folders",
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

    builder = M3FDFiftyOneBuilder(data_root=args.data_root)
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
