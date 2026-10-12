from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .fiftyone_rgbt_base import ExportMode, FiftyOneRgbtYoloDatasetDataloader

CLASS_NAMES = ("person", "car", "bus", "motorcycle", "truck")

class M3fdRgbtYoloDatasetDataloader(FiftyOneRgbtYoloDatasetDataloader):
    """M3FD FiftyOne exporter for YOLO RGBT training."""

    def __init__(
        self,
        *,
        fiftyone_dataset_name: str = "m3fd_rgbt",
        out_root: Path,
        recreate: bool = True,
        image_ext: str = ".tiff",
        export_mode: ExportMode = "tiff",
        s3_endpoint_url: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        rgb_field: str = "filepath",
        thermal_field: str = "thermal",
        bbox_field: str = "ground_truth",
        split_field: str = "split",
        class_names: Tuple[str, ...] = CLASS_NAMES,
    ) -> None:
        super().__init__(
            fiftyone_dataset_name=fiftyone_dataset_name,
            out_root=out_root,
            recreate=recreate,
            image_ext=image_ext,
            export_mode=export_mode,
            s3_endpoint_url=s3_endpoint_url,
            s3_region_name=s3_region_name,
            rgb_field=rgb_field,
            thermal_field=thermal_field,
            bbox_field=bbox_field,
            split_field=split_field,
            class_names=class_names,
        )
