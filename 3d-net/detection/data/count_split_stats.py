#!/usr/bin/env python3
"""Print image count and annotation count per train/val/test split for each dataset."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(__file__).parent


# ── per-dataset collectors ────────────────────────────────────────────────────

def stats_m3fd() -> dict:
    base = DATA_DIR / "m3fd"
    ann_dir = base / "Annotation"

    def split_stats(txt_file: Path) -> tuple[int, int]:
        names = txt_file.read_text().splitlines()
        images = len(names)
        anns = sum(
            len(ET.parse(ann_dir / (Path(name).stem + ".xml")).getroot().findall("object"))
            for name in names
            if (ann_dir / (Path(name).stem + ".xml")).exists()
        )
        return images, anns

    return {
        "train": split_stats(base / "train.txt"),
        "val":   split_stats(base / "val.txt"),
        "test":  split_stats(base / "test.txt"),
    }


def stats_flir_adas() -> dict:
    base = DATA_DIR / "flir_adas" / "FLIR_ADAS_v2"

    splits = {
        "train": {
            "rgb":     base / "images_rgb_train"     / "coco.json",
            "thermal": base / "images_thermal_train" / "coco.json",
        },
        "val": {
            "rgb":     base / "images_rgb_val"     / "coco.json",
            "thermal": base / "images_thermal_val" / "coco.json",
        },
        "test": {
            "rgb":     base / "video_rgb_test"     / "coco.json",
            "thermal": base / "video_thermal_test" / "coco.json",
        },
    }

    result = {}
    for split, modalities in splits.items():
        entry: dict[str, tuple[int, int]] = {}
        for mod, path in modalities.items():
            d = json.loads(path.read_text())
            entry[mod] = (len(d["images"]), len(d["annotations"]))
        result[split] = entry
    return result


def stats_rtdod() -> dict:
    base = DATA_DIR / "rtdod"
    ann_dir = base / "annotations"

    def split_stats(txt_file: Path) -> tuple[int, int]:
        ids = txt_file.read_text().splitlines()
        images = len(ids)
        anns = sum(
            len(ET.parse(ann_dir / f"{id_}.xml").getroot().findall("object"))
            for id_ in ids
            if (ann_dir / f"{id_}.xml").exists()
        )
        return images, anns

    return {
        "train": split_stats(base / "imagesets" / "train.txt"),
        "val":   split_stats(base / "imagesets" / "val.txt"),
        "test":  split_stats(base / "imagesets" / "test.txt"),
    }


def stats_visdrone() -> dict:
    base = DATA_DIR / "visdrone_dronevehicle"

    def count_xml_dir(d: Path) -> tuple[int, int]:
        xml_files = list(d.glob("*.xml"))
        images = len(xml_files)
        anns = sum(len(ET.parse(f).getroot().findall("object")) for f in xml_files)
        return images, anns

    result = {}
    for split in ("train", "val", "test"):
        result[split] = {
            "rgb": count_xml_dir(base / split / f"{split}label"),
            "ir":  count_xml_dir(base / split / f"{split}labelr"),
        }
    return result


# ── printers ─────────────────────────────────────────────────────────────────

COL = 10  # width of numeric columns


def _row(label: str, images: int, anns: int, indent: str = "") -> str:
    return f"{indent}  {label:<12}  {images:>{COL},}  {anns:>{COL},}"


def _header(indent: str = "") -> str:
    return f"{indent}  {'split':<12}  {'images':>{COL}}  {'annotations':>{COL}}"


def _sep(indent: str = "") -> str:
    return f"{indent}  {'-' * (12 + 2 * COL + 4)}"


def print_simple(title: str, data: dict[str, tuple[int, int]]) -> None:
    """For datasets with a single modality per split."""
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")
    print(_header())
    print(_sep())
    total_img = total_ann = 0
    for split in ("train", "val", "test"):
        imgs, anns = data[split]
        print(_row(split, imgs, anns))
        total_img += imgs
        total_ann += anns
    print(_sep())
    print(_row("TOTAL", total_img, total_ann))


def print_multimodal(title: str, data: dict[str, dict[str, tuple[int, int]]]) -> None:
    """For datasets with rgb / thermal (or ir) sub-splits."""
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")

    modalities = list(next(iter(data.values())).keys())
    grand = {m: (0, 0) for m in modalities}

    for split in ("train", "val", "test"):
        print(f"\n  [{split.upper()}]")
        print(_header(indent="  "))
        print(_sep(indent="  "))
        combined_img = combined_ann = 0
        for mod in modalities:
            imgs, anns = data[split][mod]
            print(_row(mod, imgs, anns, indent="  "))
            combined_img += imgs
            combined_ann += anns
            gi, ga = grand[mod]
            grand[mod] = (gi + imgs, ga + anns)
        print(_sep(indent="  "))
        print(_row("combined", combined_img, combined_ann, indent="  "))

    print(f"\n  [TOTAL ACROSS SPLITS]")
    print(_header(indent="  "))
    print(_sep(indent="  "))
    g_img = g_ann = 0
    for mod in modalities:
        imgs, anns = grand[mod]
        print(_row(mod, imgs, anns, indent="  "))
        g_img += imgs
        g_ann += anns
    print(_sep(indent="  "))
    print(_row("combined", g_img, g_ann, indent="  "))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Collecting stats (this may take a minute for large datasets)...")

    print("  M3FD ...", end=" ", flush=True)
    m3fd = stats_m3fd()
    print("done")

    print("  FLIR ADAS v2 ...", end=" ", flush=True)
    flir = stats_flir_adas()
    print("done")

    print("  RTDOD ...", end=" ", flush=True)
    rtdod = stats_rtdod()
    print("done")

    print("  VisDrone DroneVehicle ...", end=" ", flush=True)
    vis = stats_visdrone()
    print("done")

    print_simple("M3FD", m3fd)
    print_multimodal("FLIR ADAS v2", flir)
    print_simple("RTDOD", rtdod)
    print_multimodal("VisDrone DroneVehicle", vis)
    print()
