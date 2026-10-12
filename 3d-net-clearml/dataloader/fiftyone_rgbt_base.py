from __future__ import annotations

import cv2
import os
import shutil
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING, Tuple
from urllib.parse import urlparse

from .base import RgbtYoloDataloader

if TYPE_CHECKING:
    import fiftyone as fo

# ---------------------------------------------------------------------------
# Export modes
#
# "tiff"    — current behaviour: 4-channel (BGR + thermal) TIFF stored under
#              images/{split}/, labels under labels/{split}/.
#              Dataset YAML has `channels: 4`.
#
# "vis_ir"  — wandahangFY RGBT mode: separate RGB JPG under visible/{split}/
#              and grayscale thermal JPG under infrared/{split}/ (same filename).
#              Labels sit alongside the visible images (visible/{split}/*.txt).
#              Dataset YAML points at visible/{split} — the fork replaces
#              "visible" → "infrared" at load time to fetch the thermal image.
#              No `channels:` key in the YAML; pass channels=4 as a train arg.
#
# "rgb"     — RGB-only: 3-channel BGR image saved under images/{split}/.
#              Dataset YAML has `channels: 3`.
#
# "thermal" — Thermal-only: single-channel grayscale image under images/{split}/.
#              Dataset YAML has `channels: 1`.
# ---------------------------------------------------------------------------

ExportMode = Literal["tiff", "vis_ir", "rgb", "thermal"]


class FiftyOneRgbtYoloDatasetDataloader(RgbtYoloDataloader):
    """Common FiftyOne -> YOLO exporter for 4-channel (RGB + thermal) datasets."""

    def __init__(
        self,
        *,
        fiftyone_dataset_name: str,
        out_root: Path,
        recreate: bool = True,
        image_ext: str = ".tiff",
        export_mode: ExportMode = "tiff",
        s3_endpoint_url: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        rgb_field: str = "filepath",
        thermal_field: str = "thermal",
        bbox_field: str = "gt_bbox_thermal",
        split_field: str = "split",
        class_names: Tuple[str, ...] = (),
    ) -> None:
        self.fiftyone_dataset_name = fiftyone_dataset_name
        self.out_root = out_root
        self.recreate = recreate
        self.image_ext = image_ext
        self.export_mode: ExportMode = export_mode
        self.rgb_field = rgb_field
        self.thermal_field = thermal_field
        self.bbox_field = bbox_field
        self.split_field = split_field
        self.class_names = class_names

        self.s3_endpoint_url = s3_endpoint_url or os.environ.get(
            "S3_ENDPOINT_URL", "https://s3.de-west-1.psmanaged.com"
        )
        self.s3_region_name = s3_region_name or os.environ.get("S3_REGION_NAME", "de-west-1")
        self.s3_access_key = os.environ.get("MINIO_ACCESS_KEY")
        self.s3_secret_key = os.environ.get("MINIO_SECRET_ACCESS_KEY")

    def prepare(self) -> Path:
        out_root = Path(self.out_root)
        dataset_yaml = out_root / "dataset.yaml"

        if not self.recreate and dataset_yaml.exists():
            print(f"Dataset already exists at {out_root}, skipping export.")
            return dataset_yaml

        if out_root.exists():
            shutil.rmtree(out_root)

        self._export_fiftyone_to_yolo_rgbt(out_root=out_root)

        if not dataset_yaml.exists():
            raise RuntimeError(f"Expected {dataset_yaml} to exist, but it was not created")

        return dataset_yaml

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_s3_from_url(self, url: str) -> Optional[Tuple[str, str]]:
        u = urlparse(url)
        parts = u.path.lstrip("/").split("/", 1)
        if len(parts) != 2:
            return None
        bucket, key = parts
        return (bucket, key) if bucket and key else None

    def _get_s3_client(self):
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            region_name=self.s3_region_name,
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
        )

    def _resolve_class_id(self, class_to_id: dict[str, int], cls_name: Optional[str]) -> int:
        cls = (cls_name or "object").strip()
        if cls in class_to_id:
            return class_to_id[cls]

        cls_lower = cls.lower()
        for name, idx in class_to_id.items():
            if name.lower() == cls_lower:
                return idx
        return 0

    def _get_sample_field(self, sample: "fo.Sample", field_name: str, default: Any = None) -> Any:
        """Safely read a FiftyOne sample field with a fallback default."""
        try:
            value = sample.get_field(field_name)
        except Exception:
            return default
        return default if value is None else value

    def _write_dataset_yaml(self, out_root: Path) -> None:
        """Write dataset.yaml with an absolute path so ultralytics resolves it correctly."""
        abs_root = out_root.resolve().as_posix()
        channels_map = {"tiff": 4, "vis_ir": 4, "rgb": 3, "thermal": 3}
        channels = channels_map[self.export_mode]

        if self.export_mode == "vis_ir":
            img_prefix = "visible"
        else:
            img_prefix = "images"

        content = (
            f"path: {abs_root}\n"
            f"train: {img_prefix}/train\n"
            f"val: {img_prefix}/val\n"
            f"test: {img_prefix}/test\n"
            f"nc: {len(self.class_names)}\n"
            f"names: {list(self.class_names)}\n"
            f"channels: {channels}\n"
        )
        (out_root / "dataset.yaml").write_text(content, encoding="utf-8")

    def _export_fiftyone_to_yolo_rgbt(self, *, out_root: Path, max_workers: int = 16) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import fiftyone as fo
        import numpy as np

        try:
            dataset = fo.load_dataset(self.fiftyone_dataset_name)
            print(f"Dataset stats: {dataset.stats()}")
            print(f"Field Schema: {dataset.get_field_schema()}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not load FiftyOne dataset '{self.fiftyone_dataset_name}'. "
                f"Make sure it exists on this machine/agent. Original error: {exc}"
            )

        out_root.mkdir(parents=True, exist_ok=True)

        splits = ["train", "val", "test"]

        if self.export_mode == "vis_ir":
            # visible/{split}/ holds RGB images + labels
            # infrared/{split}/ holds grayscale thermal images (same filenames)
            for split in splits:
                (out_root / "visible" / split).mkdir(parents=True, exist_ok=True)
                (out_root / "infrared" / split).mkdir(parents=True, exist_ok=True)
        else:
            # tiff / rgb / thermal: standard images/{split}/ + labels/{split}/
            for split in splits:
                (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
                (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        class_to_id = {name: i for i, name in enumerate(self.class_names)}

        self._write_dataset_yaml(out_root)

        # Batch-fetch all field values in one shot instead of N per-sample DB round-trips.
        print("Fetching sample fields from FiftyOne...")
        rgb_paths     = dataset.values(self.rgb_field)
        thermal_paths = dataset.values(self.thermal_field)
        splits_vals   = dataset.values(self.split_field)
        dets_vals     = dataset.values(self.bbox_field)
        sample_ids    = dataset.values("id")
        print(f"  {len(rgb_paths)} samples fetched, starting parallel export with {max_workers} workers...")

        # Thread-local S3 client so each thread reuses its own connection pool.
        thread_local = threading.local()

        def _get_s3():
            if not hasattr(thread_local, "s3"):
                thread_local.s3 = self._get_s3_client()
            return thread_local.s3

        def _open(path: str, flags: int = cv2.IMREAD_COLOR):
            path = str(path)
            if path.startswith("http://") or path.startswith("https://"):
                s3_parsed = self._parse_s3_from_url(path)
                if s3_parsed is not None:
                    bucket, key = s3_parsed
                    obj = _get_s3().get_object(Bucket=bucket, Key=key)
                    data = np.frombuffer(obj["Body"].read(), dtype=np.uint8)
                else:
                    import requests
                    resp = requests.get(path, timeout=60)
                    resp.raise_for_status()
                    data = np.frombuffer(resp.content, dtype=np.uint8)
                return cv2.imdecode(data, flags)
            return cv2.imread(path, flags)

        def _label_lines(dets, class_to_id: dict) -> list[str]:
            lines = []
            if dets is not None and hasattr(dets, "detections"):
                for det in dets.detections:
                    bb = getattr(det, "bounding_box", None)
                    if not bb or len(bb) != 4:
                        continue
                    x, y, w, h = bb
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    cls_id = self._resolve_class_id(class_to_id, getattr(det, "label", None))
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            return lines

        def process_one(idx):
            rgb_path     = rgb_paths[idx]
            thermal_path = thermal_paths[idx]
            split        = splits_vals[idx]
            dets         = dets_vals[idx]
            sample_id    = sample_ids[idx]

            if not rgb_path or not thermal_path:
                return "skip"

            if split not in splits:
                split = "train"

            stem = Path(str(rgb_path)).stem

            try:
                rgb_img = _open(str(rgb_path), cv2.IMREAD_COLOR)
                th_img  = _open(str(thermal_path), cv2.IMREAD_UNCHANGED)
            except Exception as exc:
                print(f"Warning: could not load images for sample id={sample_id}, skipping. Error: {exc}")
                return "skip"

            # Normalise thermal to single-channel grayscale.
            if th_img.ndim == 3:
                if not (np.array_equal(th_img[:, :, 0], th_img[:, :, 1]) and np.array_equal(th_img[:, :, 0], th_img[:, :, 2])):
                    raise ValueError(f"Thermal image at '{thermal_path}' has multiple channels that are not identical — expected grayscale.")
                th_img = th_img[:, :, 0]

            if th_img.shape[:2] != rgb_img.shape[:2]:
                th_img = cv2.resize(th_img, (rgb_img.shape[1], rgb_img.shape[0]))

            lines = _label_lines(dets, class_to_id)
            label_text = "\n".join(lines) + ("\n" if lines else "")

            if self.export_mode == "vis_ir":
                # Determine output extension: use image_ext if it's not .tiff, otherwise fall back to .jpg
                ext = self.image_ext if self.image_ext.lower() not in (".tiff", ".tif") else ".jpg"
                vis_img_path = out_root / "visible" / split / f"{stem}{ext}"
                ir_img_path  = out_root / "infrared" / split / f"{stem}{ext}"
                lbl_path     = out_root / "visible" / split / f"{stem}.txt"

                cv2.imwrite(str(vis_img_path), rgb_img)
                cv2.imwrite(str(ir_img_path), th_img)
                lbl_path.write_text(label_text, encoding="utf-8")
            elif self.export_mode == "rgb":
                ext = self.image_ext if self.image_ext.lower() not in (".tiff", ".tif") else ".jpg"
                out_img = out_root / "images" / split / f"{stem}{ext}"
                out_lbl = out_root / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(out_img), rgb_img)
                out_lbl.write_text(label_text, encoding="utf-8")
            elif self.export_mode == "thermal":
                ext = self.image_ext if self.image_ext.lower() not in (".tiff", ".tif") else ".jpg"
                out_img = out_root / "images" / split / f"{stem}{ext}"
                out_lbl = out_root / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(out_img), th_img)
                out_lbl.write_text(label_text, encoding="utf-8")
            else:
                import tifffile
                out_img = out_root / "images" / split / f"{stem}{self.image_ext}"
                out_lbl = out_root / "labels" / split / f"{stem}.txt"

                chw = np.concatenate([rgb_img.transpose(2, 0, 1), th_img[None, :, :]], axis=0)
                tifffile.imwrite(out_img.as_posix(), chw, metadata={"axes": "CYX"})
                out_lbl.write_text(label_text, encoding="utf-8")

            return "ok"

        total   = len(rgb_paths)
        done    = 0
        skipped = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one, i): i for i in range(total)}
            for future in as_completed(futures):
                result = future.result()
                if result == "skip":
                    skipped += 1
                done += 1
                if done % 500 == 0:
                    print(f"  Processed {done}/{total} samples...")

        print(
            f"Exported FiftyOne dataset '{self.fiftyone_dataset_name}' -> {out_root} "
            f"[mode={self.export_mode}] | "
            f"classes={len(self.class_names)} bbox_field={self.bbox_field} thermal_field={self.thermal_field} "
            f"samples={total} skipped_missing_paths={skipped}"
        )
