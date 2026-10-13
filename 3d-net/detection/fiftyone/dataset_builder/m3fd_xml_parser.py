"""M3FD XML annotation parsing utilities.

Supports Pascal VOC-style XML files used by M3FD under:
- `detection/data/m3fd/Annotation/*.xml`

Each annotation contains:
- `<size>` with image width/height
- one or more `<object>` entries with `<name>` and `<bndbox>`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import xml.etree.ElementTree as ET


BBox = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class M3FDObject:
    label: str
    bbox: BBox
    truncated: int = 0
    difficult: int = 0


class M3FDXmlParser:
    # Canonical classes for M3FD
    CLASS_NAMES: Tuple[str, ...] = (
        "People",
        "Car",
        "Bus",
        "Lamp",
        "Motorcycle",
        "Truck",
    )

    def __init__(self, unknown_policy: Literal["skip", "error"] = "skip"):
        self.unknown_policy = unknown_policy
        self.class_name_to_id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }

    @property
    def class_names(self) -> Tuple[str, ...]:
        return self.CLASS_NAMES

    @staticmethod
    def _parse_size(root: ET.Element, xml_path: Path) -> Tuple[int, int]:
        size_elem = root.find("size")
        if size_elem is None:
            raise ValueError(f"Missing <size> in {xml_path}")

        try:
            width = int(size_elem.find("width").text)
            height = int(size_elem.find("height").text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid <size> in {xml_path}") from exc

        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive image size in {xml_path}: {width}x{height}")

        return width, height

    @staticmethod
    def _parse_bndbox(obj_elem: ET.Element) -> Optional[BBox]:
        bndbox = obj_elem.find("bndbox")
        if bndbox is None:
            return None

        try:
            x_min = float(bndbox.find("xmin").text)
            y_min = float(bndbox.find("ymin").text)
            x_max = float(bndbox.find("xmax").text)
            y_max = float(bndbox.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            return None

        if x_max < x_min or y_max < y_min:
            return None

        return x_min, y_min, x_max, y_max

    def _normalize_label(self, raw_label: str, xml_path: Path) -> Optional[str]:
        label = (raw_label or "").strip()
        if label in self.CLASS_NAMES:
            return label

        if self.unknown_policy == "error":
            raise ValueError(f"Unknown class '{label}' in {xml_path}")
        return None

    def parse(self, xml_path: str | Path) -> Tuple[int, int, List[M3FDObject]]:
        xml_path = Path(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_width, img_height = self._parse_size(root, xml_path)
        objects: List[M3FDObject] = []

        for obj in root.findall("object"):
            raw_label = obj.findtext("name", default="")
            label = self._normalize_label(raw_label, xml_path)
            if label is None:
                continue

            bbox = self._parse_bndbox(obj)
            if bbox is None:
                if self.unknown_policy == "error":
                    raise ValueError(f"Failed to parse <bndbox> in {xml_path}")
                continue

            truncated_raw = obj.findtext("truncated", default="0")
            difficult_raw = obj.findtext("difficult", default="0")
            try:
                truncated = int(truncated_raw)
            except (TypeError, ValueError):
                truncated = 0
            try:
                difficult = int(difficult_raw)
            except (TypeError, ValueError):
                difficult = 0

            objects.append(
                M3FDObject(
                    label=label,
                    bbox=bbox,
                    truncated=truncated,
                    difficult=difficult,
                )
            )

        return img_width, img_height, objects
