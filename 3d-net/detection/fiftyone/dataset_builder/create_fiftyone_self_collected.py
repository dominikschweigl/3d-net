"""Build or incrementally update a FiftyOne dataset for self-collected RGB+IR data.

Images are sourced exclusively from S3. Pairing is done by filename stem across
the `rgb/` and `ir/` prefixes within the S3 bucket.

Filename convention (used to extract metadata):
    {location}_{YY}_{MM}_{DD}_{session}_{frame_index:06d}.jpg
    e.g. hausern_20_01_26_aufnahme5_000123.jpg

Running the script multiple times is safe: new pairs are added, existing samples
are left untouched (annotations from CVAT are preserved).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import fiftyone as fo


class SelfCollectedFiftyOneBuilder:
    """Builds and incrementally updates a FiftyOne dataset for self-collected RGB/IR pairs."""

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    OUTPUT_CLASSES   = {"truck", "bus", "motorcycle", "bicycle", "person", "car"}

    # Matches:  {location}_{YY}_{MM}_{DD}_{session}_{frame_index}
    # Groups:   (1=location, 2=date YY_MM_DD, 3=session, 4=frame_index digits)
    _STEM_RE = re.compile(r'^(.+?)_(\d{2}_\d{2}_\d{2})_(.+?)_(\d+)$')

    # ── S3 helpers ─────────────────────────────────────────────────────────────

    def _get_s3_client(self, endpoint_url: str, region_name: str):
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )

    def _list_s3_stem_to_key(
        self,
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
        key_count  = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            page_count += 1
            if page_count % 20 == 0:
                logging.info(
                    "S3 listing progress for prefix '%s': %d pages, %d matched files",
                    prefix, page_count, key_count,
                )
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key")
                if not key or not key.lower().endswith(suffixes_lc):
                    continue
                stem = Path(key).stem
                if stem not in stem_to_key:
                    stem_to_key[stem] = key
                    key_count += 1

        logging.info(
            "Finished S3 listing for prefix '%s': %d pages, %d matched files",
            prefix, page_count, key_count,
        )
        return stem_to_key

    def _paired_stems_s3(
        self,
        *,
        s3_client,
        bucket: str,
        rgb_prefix: str,
        ir_prefix: str,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
        rgb_stem_to_key = self._list_s3_stem_to_key(
            s3_client=s3_client, bucket=bucket, prefix=rgb_prefix, suffixes=self.IMAGE_EXTENSIONS,
        )
        ir_stem_to_key = self._list_s3_stem_to_key(
            s3_client=s3_client, bucket=bucket, prefix=ir_prefix, suffixes=self.IMAGE_EXTENSIONS,
        )

        rgb_ids  = set(rgb_stem_to_key)
        ir_ids   = set(ir_stem_to_key)
        only_rgb = rgb_ids - ir_ids
        only_ir  = ir_ids  - rgb_ids

        if only_rgb:
            logging.warning("%d S3 RGB keys have no IR pair (showing up to 8): %s",
                            len(only_rgb), ", ".join(sorted(only_rgb)[:8]))
        if only_ir:
            logging.warning("%d S3 IR keys have no RGB pair (showing up to 8): %s",
                            len(only_ir), ", ".join(sorted(only_ir)[:8]))

        ids = sorted(rgb_ids & ir_ids)
        logging.info("Paired %d IDs from S3 (rgb=%d, ir=%d)", len(ids), len(rgb_ids), len(ir_ids))
        return ids, rgb_stem_to_key, ir_stem_to_key

    # ── Sample construction ────────────────────────────────────────────────────

    def _parse_stem(self, stem: str) -> Dict[str, str | int]:
        m = self._STEM_RE.match(stem)
        if m:
            return {
                "location":    m.group(1),
                "date":        m.group(2),
                "session":     m.group(3),
                "frame_index": int(m.group(4)),
            }
        logging.debug("Stem '%s' does not match expected naming convention.", stem)
        return {"location": "unknown", "date": "unknown", "session": "unknown", "frame_index": -1}

    def _make_sample(self, stem: str, rgb_url: str, ir_url: str) -> fo.Sample:
        meta   = self._parse_stem(stem)
        sample = fo.Sample(filepath=rgb_url)
        sample["thermal"]      = ir_url
        sample["location"]     = meta["location"]
        sample["date"]         = meta["date"]
        sample["session"]      = meta["session"]
        sample["frame_index"]  = meta["frame_index"]
        sample["ground_truth"] = fo.Detections(detections=[])
        return sample

    # ── Dataset preparation ────────────────────────────────────────────────────

    def _prepare_dataset(self, name: str, overwrite: bool) -> fo.Dataset:
        if overwrite and fo.dataset_exists(name):
            fo.delete_dataset(name)
            logging.info("Deleted existing dataset '%s'", name)

        if fo.dataset_exists(name):
            logging.info("Loading existing dataset '%s'", name)
            return fo.load_dataset(name)

        dataset = fo.Dataset(name)
        dataset.add_sample_field("thermal",      fo.StringField)
        dataset.add_sample_field("location",     fo.StringField)
        dataset.add_sample_field("date",         fo.StringField)
        dataset.add_sample_field("session",      fo.StringField)
        dataset.add_sample_field("frame_index",  fo.IntField)
        dataset.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
        
        if "thermal" not in dataset.app_config.media_fields:
            dataset.app_config.media_fields.append("thermal")

        dataset.default_classes = sorted(self.OUTPUT_CLASSES)

        dataset.save()

        logging.info("Created new dataset '%s'", name)
        return dataset

    @staticmethod
    def _dims_from_header(header: bytes, ext: str) -> tuple[int, int] | None:
        """Parse image width/height from the first few KB of a JPEG or PNG file."""
        if ext == ".png":
            if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
                w = struct.unpack(">I", header[16:20])[0]
                h = struct.unpack(">I", header[20:24])[0]
                return w, h
        else:  # JPEG
            i = 2  # skip SOI marker FFD8
            while i + 4 <= len(header):
                if header[i] != 0xFF:
                    break
                marker = header[i + 1]
                # SOF markers that carry image dimensions
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                               0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    if i + 9 <= len(header):
                        h = struct.unpack(">H", header[i + 5 : i + 7])[0]
                        w = struct.unpack(">H", header[i + 7 : i + 9])[0]
                        return w, h
                if i + 4 > len(header):
                    break
                length = struct.unpack(">H", header[i + 2 : i + 4])[0]
                i += 2 + length
        return None

    def _fetch_dims(self, s3_client, bucket: str, key: str) -> tuple[int, int]:
        """Read width/height from a 4 KB range request; fall back to full download."""
        ext = Path(key).suffix.lower()
        resp = s3_client.get_object(Bucket=bucket, Key=key, Range="bytes=0-4095")
        header = resp["Body"].read()
        dims = self._dims_from_header(header, ext)
        if dims:
            return dims
        # Fallback: download the full image and decode
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        data = np.frombuffer(resp["Body"].read(), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        return w, h

    def _compute_metadata(
        self,
        samples: List[fo.Sample],
        s3_client,
        bucket: str,
        max_workers: int = 16,
    ) -> None:
        """Fetch image dimensions in parallel via S3 range requests."""
        logging.info("Computing image metadata for %d new samples...", len(samples))
        total = len(samples)
        done  = 0

        def process(sample: fo.Sample) -> fo.Sample:
            path = urlparse(sample.filepath).path.lstrip("/")
            key  = path.split("/", 1)[1]
            w, h = self._fetch_dims(s3_client, bucket, key)
            sample.metadata = fo.ImageMetadata(width=w, height=h)
            return sample

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process, s): s for s in samples}
            for future in as_completed(futures):
                future.result()
                done += 1
                print(f"\rMetadata: {done}/{total}", end="", flush=True)
        print()
        logging.info("Metadata computed.")

    def _existing_filepaths(self, dataset: fo.Dataset) -> Set[str]:
        if len(dataset) == 0:
            return set()
        return set(dataset.values("filepath"))

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        name: str,
        overwrite: bool = False,
        s3_bucket: str,
        s3_prefix: str  = "self_collected",
        s3_endpoint_url: str = "https://s3.de-west-1.psmanaged.com",
        s3_region: str  = "de-west-1",
    ) -> fo.Dataset:
        """
        Build or incrementally update the FiftyOne dataset from S3.

        - overwrite=True:  delete any existing dataset and rebuild from scratch.
        - overwrite=False: add new RGB/IR pairs not yet in the dataset;
                           existing samples (and their CVAT annotations) are untouched.
        """
        logging.info(
            "Using S3 source: bucket=%s, prefix=%s, endpoint=%s",
            s3_bucket, s3_prefix, s3_endpoint_url,
        )
        dataset        = self._prepare_dataset(name, overwrite)
        existing_paths = self._existing_filepaths(dataset)

        s3_client = self._get_s3_client(s3_endpoint_url, s3_region)
        base      = s3_prefix.rstrip("/")
        ids, rgb_map, ir_map = self._paired_stems_s3(
            s3_client=s3_client,
            bucket=s3_bucket,
            rgb_prefix=f"{base}/rgb/",
            ir_prefix=f"{base}/ir/",
        )

        endpoint    = s3_endpoint_url.rstrip("/")
        new_samples: List[fo.Sample] = []
        for stem in ids:
            rgb_url = f"{endpoint}/{s3_bucket}/{rgb_map[stem]}"
            if rgb_url in existing_paths:
                continue
            ir_url = f"{endpoint}/{s3_bucket}/{ir_map[stem]}"
            new_samples.append(self._make_sample(stem, rgb_url, ir_url))

        logging.info(
            "Found %d paired stems on S3, %d already in dataset, %d new to add",
            len(ids), len(existing_paths), len(new_samples),
        )

        if new_samples:
            self._compute_metadata(new_samples, s3_client, s3_bucket)
            dataset.add_samples(new_samples)

        logging.info("Dataset '%s' now contains %d samples total", name, len(dataset))
        return dataset


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a FiftyOne dataset from self-collected RGB+IR data on S3.",
    )
    parser.add_argument(
        "--name",
        default="3d_net_rgbt",
        help="FiftyOne dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing dataset with the same name and rebuild from scratch",
    )
    parser.add_argument(
        "--s3-bucket",
        default="bernard-3d-net",
        help="S3 bucket containing the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--s3-prefix",
        default="self-collected",
        help="Prefix within the S3 bucket (default: %(default)s)",
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    builder = SelfCollectedFiftyOneBuilder()
    dataset = builder.build(
        name=args.name,
        overwrite=args.overwrite,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint_url=args.s3_endpoint_url,
        s3_region=args.s3_region,
    )
    dataset.persistent = True
    dataset.save()
    logging.info("Done. Dataset '%s' saved.", args.name)


if __name__ == "__main__":
    main()
