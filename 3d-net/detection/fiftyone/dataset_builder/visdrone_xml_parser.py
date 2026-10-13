"""VisDrone XML annotation parsing utilities.

Supports the VisDrone DroneVehicle-style XML files found under
`detection/data/visdrone_dronevehicle/**/<split>label*.xml`.

Objects may be annotated using:
- `<bndbox>`: axis-aligned box
- `<polygon>`: quadrilateral polygon (x1..x4, y1..y4; sometimes more points)
- `<point>`: a single point (x,y)

This module provides a small, reusable parser class that can be shared by:
- the YOLO dataset converter
- the FiftyOne dataset builder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Literal
import xml.etree.ElementTree as ET


BBox = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in pixels
Point = Tuple[float, float]


@dataclass(frozen=True)
class VisdroneObject:
    label: str
    bbox: BBox
    source: str  # 'bndbox' | 'polygon' | 'point'
    polygon: Optional[List[Point]] = None  # in pixels; present when source == 'polygon'


class VisdroneXmlParser:
    # Canonical classes for this dataset
    CLASS_NAMES: Tuple[str, ...] = (
        "car",
        "truck",
        "freight_car",
        "bus",
        "van",
    )

    # Internal aliases/misspellings observed in some VisDrone-style exports
    _CLASS_NAME_ALIASES: Dict[str, str] = {
        "truvk": "truck",
        "feright car": "freight_car",
        "feright_car": "freight_car",
        "feright": "freight_car",
    }

    def __init__(
        self,
        unknown_policy: Literal["skip", "error"] = "skip",
        point_box_size: float = 1.0,
    ):
        self.unknown_policy = unknown_policy
        self.point_box_size = float(point_box_size)
        self.class_name_to_id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }

    @property
    def class_names(self) -> Tuple[str, ...]:
        return self.CLASS_NAMES

    @staticmethod
    def extract_bbox_from_polygon_coords(polygon_coords: Dict[str, float]) -> BBox:
        x_coords = [polygon_coords.get(f"x{i}", 0.0) for i in range(1, 5)]
        y_coords = [polygon_coords.get(f"y{i}", 0.0) for i in range(1, 5)]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    @staticmethod
    def extract_bbox_from_points(points: Iterable[Point]) -> BBox:
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def convert_bbox_to_yolo(
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        img_width: float,
        img_height: float,
    ) -> Tuple[float, float, float, float]:
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        return x_center_norm, y_center_norm, width_norm, height_norm

    @staticmethod
    def _parse_size(root: ET.Element, xml_path: Path) -> Tuple[int, int]:
        size_elem = root.find("size")
        if size_elem is None:
            raise ValueError(f"Missing <size> in {xml_path}")

        try:
            img_width = int(size_elem.find("width").text)
            img_height = int(size_elem.find("height").text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid <size> in {xml_path}") from exc

        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Non-positive image size in {xml_path}: {img_width}x{img_height}")

        return img_width, img_height

    @staticmethod
    def _parse_bndbox(obj: ET.Element) -> Optional[BBox]:
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return None

        try:
            x_min = float(bndbox.find("xmin").text)
            y_min = float(bndbox.find("ymin").text)
            x_max = float(bndbox.find("xmax").text)
            y_max = float(bndbox.find("ymax").text)
            return x_min, y_min, x_max, y_max
        except (AttributeError, ValueError, TypeError):
            return None

    @staticmethod
    def _parse_polygon_points(obj: ET.Element) -> Optional[List[Point]]:
        polygon = obj.find("polygon")
        if polygon is None:
            return None

        points: List[Point] = []

        # Common format is x1..x4,y1..y4, but be tolerant if more exist.
        # We stop when a pair is missing.
        idx = 1
        while True:
            x_elem = polygon.find(f"x{idx}")
            y_elem = polygon.find(f"y{idx}")
            if x_elem is None or y_elem is None:
                break
            try:
                points.append((float(x_elem.text), float(y_elem.text)))
            except (TypeError, ValueError):
                return None
            idx += 1

        if len(points) < 3:
            return None
        return points

    def _parse_point_bbox(self, obj: ET.Element) -> Optional[BBox]:
        point = obj.find("point")
        if point is None:
            return None

        try:
            x = float(point.find("x").text)
            y = float(point.find("y").text)
        except (AttributeError, ValueError, TypeError):
            return None

        half = self.point_box_size / 2.0
        return x - half, y - half, x + half, y + half

    def _normalize_label(self, raw_label: str, xml_path: Path) -> Optional[str]:
        label = (raw_label or "").strip().lower()
        if not label:
            if self.unknown_policy == "error":
                raise ValueError(f"Empty class name in {xml_path}")
            return None

        # Try direct match first
        if label in self.CLASS_NAMES:
            return label

        # Try known aliases
        if label in self._CLASS_NAME_ALIASES:
            canonical = self._CLASS_NAME_ALIASES[label]
            return canonical if canonical in self.CLASS_NAMES else None

        if self.unknown_policy == "error":
            raise ValueError(f"Unknown class '{label}' in {xml_path}")
        return None

    def parse(self, xml_path: str | Path) -> Tuple[int, int, List[VisdroneObject]]:
        xml_path = Path(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_width, img_height = self._parse_size(root, xml_path)
        objects: List[VisdroneObject] = []

        for obj in root.findall("object"):
            raw_label = obj.findtext("name", default="")
            label = self._normalize_label(raw_label, xml_path)
            if label is None:
                continue

            bbox = None
            polygon_points = self._parse_polygon_points(obj)
            if obj.find("bndbox") is not None:
                bbox = self._parse_bndbox(obj)
                source = "bndbox"
            elif polygon_points is not None:
                bbox = self.extract_bbox_from_points(polygon_points)
                source = "polygon"
            elif obj.find("point") is not None:
                bbox = self._parse_point_bbox(obj)
                source = "point"
            else:
                # Unknown geometry type
                if self.unknown_policy == "error":
                    raise ValueError(f"No bndbox/polygon/point found in {xml_path}")
                continue

            if bbox is None:
                if self.unknown_policy == "error":
                    raise ValueError(f"Failed to parse geometry in {xml_path}")
                if self.unknown_policy == "skip":
                    print(f"Warning: Failed to parse geometry in {xml_path}; skipping object")
                continue

            x_min, y_min, x_max, y_max = bbox
            if x_max < x_min or y_max < y_min:
                # Skip degenerate boxes
                if self.unknown_policy == "error":
                    raise ValueError(f"Degenerate bbox in {xml_path}: {bbox}")
                if self.unknown_policy == "skip":
                    print(f"Warning: Degenerate bbox in {xml_path}: {bbox}; skipping object")
                continue

            objects.append(
                VisdroneObject(
                    label=label,
                    bbox=(x_min, y_min, x_max, y_max),
                    source=source,
                    polygon=polygon_points if source == "polygon" else None,
                )
            )

        return img_width, img_height, objects

    def parse_yolo(
        self, xml_path: str | Path
    ) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
        img_width, img_height, objects = self.parse(xml_path)

        yolo_labels: List[Tuple[int, float, float, float, float]] = []
        for obj in objects:
            class_id = self.class_name_to_id[obj.label]
            x_min, y_min, x_max, y_max = obj.bbox
            x_c, y_c, w, h = self.convert_bbox_to_yolo(
                x_min, y_min, x_max, y_max, img_width, img_height
            )
            yolo_labels.append((class_id, x_c, y_c, w, h))

        return img_width, img_height, yolo_labels
