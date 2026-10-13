"""Builds a FiftyOne dataset for the VisDrone DroneVehicle RGB+thermal data.

The raw dataset under ``data/visdrone_dronevehicle`` stores separate folders per
split:
- ``<split>/<split>img``: RGB imagery (color)
- ``<split>/<split>imgr``: thermal/IR imagery (grayscale stored as 3 channels)
- ``<split>/<split>label``: XML annotations for RGB
- ``<split>/<split>labelr``: XML annotations for thermal

Each XML uses quadrilateral polygons; we store both bounding boxes and the
polygons in the resulting FiftyOne samples. Bounding boxes are axis-aligned
(min/max of the polygon).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Set

import boto3
import fiftyone as fo
import fiftyone.core.storage as fos

try:
    from .visdrone_xml_parser import VisdroneXmlParser
except ImportError:  # pragma: no cover
    # Allows running as a standalone script
    from visdrone_xml_parser import VisdroneXmlParser

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "visdrone_dronevehicle"
DEFAULT_S3_CACHE_DIR = Path(__file__).resolve().parent / "_s3_cache" / "visdrone_dronevehicle"

XML_PARSER = VisdroneXmlParser(unknown_policy="skip")


def _stems(directory: Path, pattern: str) -> Set[str]:
    return {p.stem for p in directory.glob(pattern)}


def _warn_id_mismatches(
    *,
    rgb_ids: Set[str],
    thermal_ids: Set[str],
    rgb_xml_ids: Set[str],
    thermal_xml_ids: Set[str],
) -> None:
    common = rgb_ids & thermal_ids & rgb_xml_ids & thermal_xml_ids
    union = rgb_ids | thermal_ids | rgb_xml_ids | thermal_xml_ids
    if common == union:
        return

    def _examples(values: Set[str], limit: int = 8) -> str:
        if not values:
            return ""
        shown = sorted(values)[:limit]
        more = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
        return ", ".join(shown) + more

    missing_rgb = union - rgb_ids
    missing_thermal = union - thermal_ids
    missing_rgb_xml = union - rgb_xml_ids
    missing_thermal_xml = union - thermal_xml_ids

    logging.warning(
        "File/label mismatch detected. Common=%d, RGB=%d, thermal=%d, RGB-XML=%d, thermal-XML=%d. "
        "Missing: RGB=%d [%s]; thermal=%d [%s]; RGB-XML=%d [%s]; thermal-XML=%d [%s]",
        len(common),
        len(rgb_ids),
        len(thermal_ids),
        len(rgb_xml_ids),
        len(thermal_xml_ids),
        len(missing_rgb),
        _examples(missing_rgb),
        len(missing_thermal),
        _examples(missing_thermal),
        len(missing_rgb_xml),
        _examples(missing_rgb_xml),
        len(missing_thermal_xml),
        _examples(missing_thermal_xml),
    )


def available_ids_rgbt_raw(
    rgb_dir: Path,
    thermal_dir: Path,
    rgb_label_dir: Path,
    thermal_label_dir: Path,
) -> List[str]:
    """Returns stems present across RGB images, thermal images, and both XML label sets."""

    rgb_ids = _stems(rgb_dir, "*.jpg")
    thermal_ids = _stems(thermal_dir, "*.jpg")
    rgb_xml_ids = _stems(rgb_label_dir, "*.xml")
    thermal_xml_ids = _stems(thermal_label_dir, "*.xml")
    _warn_id_mismatches(
        rgb_ids=rgb_ids,
        thermal_ids=thermal_ids,
        rgb_xml_ids=rgb_xml_ids,
        thermal_xml_ids=thermal_xml_ids,
    )
    return sorted(rgb_ids & thermal_ids & rgb_xml_ids & thermal_xml_ids)


def _get_s3_client(endpoint_url: str, region_name: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )


def _list_s3_stems(*, s3_client, bucket: str, prefix: str, suffix: str) -> Set[str]:
    stems: Set[str] = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if not key or not key.endswith(suffix):
                continue
            stems.add(Path(key).stem)
    return stems


def available_ids_rgbt_s3(
    *,
    s3_client,
    bucket: str,
    rgb_prefix: str,
    thermal_prefix: str,
    rgb_label_prefix: str,
    thermal_label_prefix: str,
) -> List[str]:
    rgb_ids = _list_s3_stems(s3_client=s3_client, bucket=bucket, prefix=rgb_prefix, suffix=".jpg")
    thermal_ids = _list_s3_stems(
        s3_client=s3_client, bucket=bucket, prefix=thermal_prefix, suffix=".jpg"
    )
    rgb_xml_ids = _list_s3_stems(
        s3_client=s3_client, bucket=bucket, prefix=rgb_label_prefix, suffix=".xml"
    )
    thermal_xml_ids = _list_s3_stems(
        s3_client=s3_client, bucket=bucket, prefix=thermal_label_prefix, suffix=".xml"
    )
    _warn_id_mismatches(
        rgb_ids=rgb_ids,
        thermal_ids=thermal_ids,
        rgb_xml_ids=rgb_xml_ids,
        thermal_xml_ids=thermal_xml_ids,
    )
    return sorted(rgb_ids & thermal_ids & rgb_xml_ids & thermal_xml_ids)


def _download_s3_file(*, s3_client, bucket: str, key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    s3_client.download_file(bucket, key, str(local_path))
    return local_path


def prepare_dataset(name: str, overwrite: bool) -> fo.Dataset:
    if overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)
    elif fo.dataset_exists(name):
        logging.info("Dataset '%s' already exists; loading it", name)
        return fo.load_dataset(name)

    dataset = fo.load_dataset(name, create_if_necessary=True)
    dataset.add_sample_field("thermal", fo.StringField)
    dataset.add_sample_field("split", fo.StringField)
    dataset.add_sample_field(
        "gt_bbox_rgb", fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
    )
    dataset.add_sample_field(
        "gt_bbox_thermal", fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
    )
    dataset.add_sample_field(
        "gt_polygon_rgb", fo.EmbeddedDocumentField, embedded_doc_type=fo.Polylines
    )
    dataset.add_sample_field(
        "gt_polygon_thermal", fo.EmbeddedDocumentField, embedded_doc_type=fo.Polylines
    )
    dataset.default_classes = list(XML_PARSER.class_names)

    if "thermal" not in dataset.app_config.media_fields:
        dataset.app_config.media_fields.append("thermal")
    
    return dataset


def build_split(dataset: fo.Dataset, data_root: Path, split: str) -> int:
    rgb_dir = data_root / split / f"{split}img"
    thermal_dir = data_root / split / f"{split}imgr"
    label_rgb_dir = data_root / split / f"{split}label"
    label_thermal_dir = data_root / split / f"{split}labelr"

    for path in (rgb_dir, thermal_dir, label_rgb_dir, label_thermal_dir):
        if not path.exists():
            raise FileNotFoundError(f"Expected path missing: {path}")

    ids = available_ids_rgbt_raw(rgb_dir, thermal_dir, label_rgb_dir, label_thermal_dir)
    samples: List[fo.Sample] = []

    for stem in ids:
        rgb_path = rgb_dir / f"{stem}.jpg"
        thermal_path = thermal_dir / f"{stem}.jpg"
        rgb_xml = label_rgb_dir / f"{stem}.xml"
        thermal_xml = label_thermal_dir / f"{stem}.xml"

        rgb_w, rgb_h, rgb_objects = XML_PARSER.parse(rgb_xml)
        th_w, th_h, th_objects = XML_PARSER.parse(thermal_xml)

        # Prefer per-XML size; warn if sizes disagree
        if (rgb_w, rgb_h) != (th_w, th_h):
            logging.warning(
                "Size mismatch for %s (%dx%d) vs %s (%dx%d)",
                rgb_xml.name,
                rgb_w,
                rgb_h,
                thermal_xml.name,
                th_w,
                th_h,
            )

        def to_fiftyone(objects, w: int, h: int):
            dets: List[fo.Detection] = []
            polys: List[fo.Polyline] = []
            for o in objects:
                x_min, y_min, x_max, y_max = o.bbox
                bbox = [
                    x_min / w,
                    y_min / h,
                    (x_max - x_min) / w,
                    (y_max - y_min) / h,
                ]
                dets.append(fo.Detection(label=o.label, bounding_box=bbox))

                if o.polygon:
                    normalized_points = [[(x / w, y / h) for x, y in o.polygon]]
                    polys.append(
                        fo.Polyline(
                            label=o.label,
                            points=normalized_points,
                            closed=True,
                            filled=False,
                        )
                    )
            return dets, polys

        rgb_dets, rgb_polys = to_fiftyone(rgb_objects, rgb_w, rgb_h)
        thermal_dets, thermal_polys = to_fiftyone(th_objects, th_w, th_h)

        sample = fo.Sample(filepath=str(rgb_path))
        sample["split"] = split
        sample.tags = [f"target:{split}"]
        sample["thermal"] = str(thermal_path)
        sample["gt_bbox_rgb"] = fo.Detections(detections=rgb_dets)
        sample["gt_bbox_thermal"] = fo.Detections(detections=thermal_dets)
        sample["gt_polygon_rgb"] = fo.Polylines(polylines=rgb_polys)
        sample["gt_polygon_thermal"] = fo.Polylines(polylines=thermal_polys)
        samples.append(sample)

    dataset.add_samples(samples)
    return len(samples)


def build_split_s3(
    *,
    dataset: fo.Dataset,
    s3_client,
    bucket: str,
    prefix: str,
    cache_dir: Path,
    split: str,
) -> int:
    # Expected key layout under prefix mirrors the local dataset layout
    rgb_prefix = f"{prefix.rstrip('/')}/{split}/{split}img/"
    thermal_prefix = f"{prefix.rstrip('/')}/{split}/{split}imgr/"
    label_rgb_prefix = f"{prefix.rstrip('/')}/{split}/{split}label/"
    label_thermal_prefix = f"{prefix.rstrip('/')}/{split}/{split}labelr/"
    print(rgb_prefix)

    ids = available_ids_rgbt_s3(
        s3_client=s3_client,
        bucket=bucket,
        rgb_prefix=rgb_prefix,
        thermal_prefix=thermal_prefix,
        rgb_label_prefix=label_rgb_prefix,
        thermal_label_prefix=label_thermal_prefix,
    )

    samples: List[fo.Sample] = []
    for stem in ids:
        rgb_key = f"{rgb_prefix}{stem}.jpg"
        thermal_key = f"{thermal_prefix}{stem}.jpg"
        rgb_xml_key = f"{label_rgb_prefix}{stem}.xml"
        thermal_xml_key = f"{label_thermal_prefix}{stem}.xml"

        rgb_filepath = f"https://s3.de-west-1.psmanaged.com/{bucket}/{rgb_key}"
        thermal_filepath = f"https://s3.de-west-1.psmanaged.com/{bucket}/{thermal_key}"

        rgb_xml_path = _download_s3_file(
            s3_client=s3_client,
            bucket=bucket,
            key=rgb_xml_key,
            local_path=cache_dir / split / "rgb_xml" / f"{stem}.xml",
        )
        thermal_xml_path = _download_s3_file(
            s3_client=s3_client,
            bucket=bucket,
            key=thermal_xml_key,
            local_path=cache_dir / split / "thermal_xml" / f"{stem}.xml",
        )

        rgb_w, rgb_h, rgb_objects = XML_PARSER.parse(rgb_xml_path)
        th_w, th_h, th_objects = XML_PARSER.parse(thermal_xml_path)

        if (rgb_w, rgb_h) != (th_w, th_h):
            logging.warning(
                "Size mismatch for %s (%dx%d) vs %s (%dx%d)",
                rgb_xml_path.name,
                rgb_w,
                rgb_h,
                thermal_xml_path.name,
                th_w,
                th_h,
            )

        def to_fiftyone(objects, w: int, h: int):
            dets: List[fo.Detection] = []
            polys: List[fo.Polyline] = []
            for o in objects:
                x_min, y_min, x_max, y_max = o.bbox
                bbox = [
                    x_min / w,
                    y_min / h,
                    (x_max - x_min) / w,
                    (y_max - y_min) / h,
                ]
                dets.append(fo.Detection(label=o.label, bounding_box=bbox))

                if o.polygon:
                    normalized_points = [[(x / w, y / h) for x, y in o.polygon]]
                    polys.append(
                        fo.Polyline(
                            label=o.label,
                            points=normalized_points,
                            closed=True,
                            filled=False,
                        )
                    )
            return dets, polys

        rgb_dets, rgb_polys = to_fiftyone(rgb_objects, rgb_w, rgb_h)
        thermal_dets, thermal_polys = to_fiftyone(th_objects, th_w, th_h)

        sample = fo.Sample(filepath=rgb_filepath)
        sample["split"] = split
        sample.tags = [f"target:{split}"]
        sample["thermal"] = thermal_filepath
        sample["gt_bbox_rgb"] = fo.Detections(detections=rgb_dets)
        sample["gt_bbox_thermal"] = fo.Detections(detections=thermal_dets)
        sample["gt_polygon_rgb"] = fo.Polylines(polylines=rgb_polys)
        sample["gt_polygon_thermal"] = fo.Polylines(polylines=thermal_polys)
        samples.append(sample)

    dataset.add_samples(samples)
    return len(samples)


def build_dataset(
    name: str,
    data_root: Path = DATA_ROOT,
    splits: Iterable[str] = ("train", "val", "test"),
    overwrite: bool = False,
    *,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "visdrone_dronevehicle",
    s3_endpoint_url: str = "https://s3.de-west-1.psmanaged.com",
    s3_region: str = "de-west-1",
    s3_cache_dir: Path = DEFAULT_S3_CACHE_DIR,
) -> fo.Dataset:
    dataset = prepare_dataset(name, overwrite)
    counts = {}

    s3_client = None
    if s3_bucket:
        s3_client = _get_s3_client(endpoint_url=s3_endpoint_url, region_name=s3_region)

    for split in splits:
        if s3_bucket:
            count = build_split_s3(
                dataset=dataset,
                s3_client=s3_client,
                bucket=s3_bucket,
                prefix=s3_prefix,
                cache_dir=s3_cache_dir,
                split=split,
            )
        else:
            count = build_split(dataset, data_root, split)
        counts[split] = count
        logging.info("Added %d samples to split '%s'", count, split)

    logging.info("Dataset '%s' ready. Counts: %s", name, counts)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FiftyOne dataset from visdrone_dronevehicle RGB+thermal data",
    )
    parser.add_argument(
        "--name",
        default="visdrone_dronevehicle_rgbt",
        help="Name of the FiftyOne dataset to create (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root of the visdrone_dronevehicle dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to import (default: train val test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing FiftyOne dataset with the same name",
    )

    # Optional S3 backend
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="If set, load data from this S3 bucket instead of --data-root",
    )
    parser.add_argument(
        "--s3-prefix",
        default="visdrone_dronevehicle",
        help="Prefix within the S3 bucket that contains the visdrone_dronevehicle folders",
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
    dataset: fo.Dataset = build_dataset(
        name=args.name,
        data_root=args.data_root,
        splits=["test", "train"],
        overwrite=args.overwrite,
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
