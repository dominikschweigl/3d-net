#!/usr/bin/env python3
"""Count annotated objects per class for each dataset in detection/data/."""

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent


def count_m3fd() -> dict[str, int]:
    """M3FD: Pascal VOC XML in Annotation/."""
    ann_dir = DATA_DIR / "m3fd" / "Annotation"
    counts: dict[str, int] = defaultdict(int)
    for xml_file in sorted(ann_dir.glob("*.xml")):
        for obj in ET.parse(xml_file).getroot().findall("object"):
            counts[obj.find("name").text.strip()] += 1
    return dict(counts)


def count_flir_adas() -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """FLIR ADAS v2: COCO JSON, split by modality (rgb / thermal).

    Returns (all_splits, test_only) where each is modality -> class counts.
    """
    flir_dir = DATA_DIR / "flir_adas" / "FLIR_ADAS_v2"

    modality_split_dirs = {
        "rgb": {
            "train": flir_dir / "images_rgb_train",
            "val":   flir_dir / "images_rgb_val",
            "test":  flir_dir / "video_rgb_test",
        },
        "thermal": {
            "train": flir_dir / "images_thermal_train",
            "val":   flir_dir / "images_thermal_val",
            "test":  flir_dir / "video_thermal_test",
        },
    }

    all_counts: dict[str, dict[str, int]] = {}
    test_counts: dict[str, dict[str, int]] = {}

    for modality, split_dirs in modality_split_dirs.items():
        combined: dict[str, int] = defaultdict(int)
        test_only: dict[str, int] = defaultdict(int)
        cat_names: dict[int, str] = {}

        for split, d in split_dirs.items():
            coco_json = d / "coco.json"
            if not coco_json.exists():
                continue
            data = json.loads(coco_json.read_text())
            for cat in data.get("categories", []):
                cat_names[cat["id"]] = cat["name"]
            for ann in data.get("annotations", []):
                name = cat_names.get(ann["category_id"], f"id_{ann['category_id']}")
                combined[name] += 1
                if split == "test":
                    test_only[name] += 1

        all_counts[modality] = dict(combined)
        test_counts[modality] = dict(test_only)

    return all_counts, test_counts


def count_rtdod() -> dict[str, int]:
    """RTDOD: Pascal VOC XML in annotations/."""
    ann_dir = DATA_DIR / "rtdod" / "annotations"
    counts: dict[str, int] = defaultdict(int)
    for xml_file in sorted(ann_dir.glob("*.xml")):
        for obj in ET.parse(xml_file).getroot().findall("object"):
            counts[obj.find("name").text.strip()] += 1
    return dict(counts)


_VISDRONE_ALIASES = {
    "feright": "freight car",
    "feright car": "freight car",
    "feright_car": "freight car",
    "truvk": "truck",
}


def count_visdrone() -> dict[str, dict[str, int]]:
    """VisDrone DroneVehicle: oriented-bbox XML, split by modality (rgb / ir)."""
    base = DATA_DIR / "visdrone_dronevehicle"
    label_dirs = {
        "rgb": [base / "train/trainlabel", base / "val/vallabel", base / "test/testlabel"],
        "ir":  [base / "train/trainlabelr", base / "val/vallabelr", base / "test/testlabelr"],
    }
    results: dict[str, dict[str, int]] = {}

    for modality, dirs in label_dirs.items():
        counts: dict[str, int] = defaultdict(int)
        for d in dirs:
            if not d.exists():
                continue
            for xml_file in sorted(d.glob("*.xml")):
                for obj in ET.parse(xml_file).getroot().findall("object"):
                    raw = obj.find("name").text.strip().lower()
                    name = _VISDRONE_ALIASES.get(raw, raw)
                    counts[name] += 1
        results[modality] = dict(counts)

    return results


# ── Printing helpers ──────────────────────────────────────────────────────────

def _print_single(counts: dict[str, int], indent: str = "") -> None:
    total = sum(counts.values())
    col = 32
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{indent}  {cls:<{col}} {n:>10,}")
    print(f"{indent}  {'-' * (col + 12)}")
    print(f"{indent}  {'TOTAL':<{col}} {total:>10,}")


def print_dataset(title: str, data: dict) -> None:
    print(f"\n{'=' * 56}")
    print(f"  {title}")
    print(f"{'=' * 56}")

    # flat counts
    if data and not isinstance(next(iter(data.values())), dict):
        _print_single(data)
        return

    # per-modality counts
    combined: dict[str, int] = defaultdict(int)
    for modality, counts in data.items():
        print(f"\n  [{modality.upper()}]")
        _print_single(counts, indent="  ")
        for cls, n in counts.items():
            combined[cls] += n

    if len(data) > 1:
        print(f"\n  [COMBINED]")
        _print_single(dict(combined), indent="  ")


# ── Entry point ───────────────────────────────────────────────────────────────

DATASETS = {
    "M3FD": count_m3fd,
    "FLIR ADAS v2": count_flir_adas,
    "RTDOD": count_rtdod,
    "VisDrone DroneVehicle": count_visdrone,
}

if __name__ == "__main__":
    for name, fn in DATASETS.items():
        print(f"Counting {name} ...", end=" ", flush=True)
        result = fn()
        print("done")
        if name == "FLIR ADAS v2":
            all_counts, test_counts = result
            print_dataset("FLIR ADAS v2 — all splits", all_counts)
            print_dataset("FLIR ADAS v2 — test split", test_counts)
        else:
            print_dataset(name, result)
    print()
