"""
Convert VisDrone XML dataset to YOLO format.

This script converts VisDrone drone detection dataset from XML format to YOLO compatible format.
It handles polygon-based bounding boxes and converts them to normalized YOLO format.
"""

from pathlib import Path
from typing import Literal
import argparse
import yaml
import cv2
import numpy as np
import os
import shutil

from .visdrone_xml_parser import VisdroneXmlParser


class VisDroneToYOLOConverter:
    """Converter for VisDrone XML annotations to YOLO format."""
    
    def __init__(self, data_root: str, output_root: str = None):
        """
        Initialize the converter.
        
        Args:
            data_root: Root directory containing train/test folders
            output_root: Output root directory (defaults to data_root)
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root) if output_root else Path(str(self.data_root) + '_yolo')

        # Image and label directory names inside output_root
        self.images_dir_name = 'images'
        self.labels_dir_name = 'labels'

        # Shared XML parser (also used by FiftyOne builder)
        self.xml_parser = VisdroneXmlParser(unknown_policy="skip")
        self.class_names = self.xml_parser.CLASS_NAMES


    def convert_dataset(self, split: str = 'all', mode: Literal['rgb', 'rgbt'] = 'rgb'):
        """
        High-level dataset conversion orchestrator.

        This calls the appropriate per-split conversion methods (`convert_rgb_split` and
        `convert_rgbt_split`) and then writes dataset info and a `dataset.yaml`.

        Args:
            split: Which split to convert ('train','val','test' or 'all')
            mode: Which conversion mode to run. One of 'rgb' or 'rgbt'.
        """
        splits = ['train', 'val', 'test'] if split == 'all' else [split]

        for s in splits:
            print(f"Converting split: {s}")

            # RGB-only
            rgb_img_dir = self.data_root / s / f"{s}img"
            rgb_label_dir = self.data_root / s / f"{s}label"
            if mode == 'rgb':
                if rgb_label_dir.exists():
                    print(f" Found RGB labels: {rgb_label_dir}")
                    self.convert_rgb_split(s, rgb_img_dir, rgb_label_dir)
                else:
                    print(f" No RGB labels found for split {s} at {rgb_label_dir}")

            # RGB+thermal (r suffix)
            thermal_img_dir = self.data_root / s / f"{s}imgr"
            thermal_label_dir = self.data_root / s / f"{s}labelr"
            if mode == 'rgbt':
                if thermal_label_dir.exists() and thermal_img_dir.exists():
                    print(f" Found RGBT labels: {thermal_label_dir}")
                    self.convert_rgbt_split(s, rgb_img_dir, thermal_img_dir, thermal_label_dir)
                else:
                    if thermal_label_dir.exists():
                        print(f" Thermal label dir exists but thermal images missing: {thermal_img_dir}")

        # Create dataset info and YAML
        self.create_dataset_info()

        channels = 4 if mode == 'rgbt' else 3
        self.create_yolo_dataset_yaml(channels=channels)

    def convert_rgb_split(self, split: str, img_dir: str, label_dir: str, img_ext: str = '.jpg'):
        """Convert RGB images + XML annotations to YOLO structure (copy/link images + write labels).

        Writes images to <output_root>/images/<split>/ and labels to <output_root>/labels/<split>/
        """
        img_dir = Path(img_dir)
        label_dir = Path(label_dir)

        out_img_dir = self.output_root / self.images_dir_name / split
        out_lbl_dir = self.output_root / self.labels_dir_name / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0

        for xml_path in sorted(label_dir.glob('*.xml')):
            img_path = img_dir / f"{xml_path.stem}{img_ext}"
            if not img_path.exists():
                skipped += 1
                continue

            try:
                _, _, bboxes = self.xml_parser.parse_yolo(str(xml_path))

                # Allow empty bboxes: still create empty label file
                if not bboxes:
                    bboxes = []

                # write label file
                label_out = out_lbl_dir / f"{xml_path.stem}.txt"
                with open(label_out, 'w', encoding='utf-8') as f:
                    for class_id, x_center, y_center, width, height in bboxes:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # copy or link image
                dst_img = out_img_dir / img_path.name
                if not dst_img.exists():
                    try:
                        os.link(img_path, dst_img)
                    except OSError:
                        shutil.copy2(img_path, dst_img)

                converted += 1
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
                skipped += 1

        print(f"{split}: converted {converted}, skipped {skipped}")

    def convert_rgbt_split(self, split: str, rgb_dir: str, thermal_dir: str, label_dir: str, img_ext: str = '.jpg'):
        """Convert RGB + thermal image pairs into 4-channel multi-page TIFFs and YOLO labels.

        Each output image is saved as `<stem>.tiff` containing 4 pages (R,G,B,thermal) in CHW as separate frames.
        Labels are written to <output_root>/labels/<split>/
        """
        rgb_dir = Path(rgb_dir)
        thermal_dir = Path(thermal_dir)
        label_dir = Path(label_dir)

        out_img_dir = self.output_root / self.images_dir_name / split
        out_lbl_dir = self.output_root / self.labels_dir_name / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0

        for xml_path in sorted(label_dir.glob('*.xml')):
            rgb_path = rgb_dir / f"{xml_path.stem}{img_ext}"
            thermal_path = thermal_dir / f"{xml_path.stem}{img_ext}"

            # Check existence
            if not rgb_path.exists() or not thermal_path.exists():
                skipped += 1
                continue

            try:
                _, _, bboxes = self.xml_parser.parse_yolo(str(xml_path))
                if not bboxes:
                    bboxes = []

                # write label
                label_out = out_lbl_dir / f"{xml_path.stem}.txt"
                with open(label_out, 'w', encoding='utf-8') as f:
                    for class_id, x_center, y_center, width, height in bboxes:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # load images
                rgb_img = cv2.imread(str(rgb_path))
                thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
                if rgb_img is None or thermal_img is None:
                    skipped += 1
                    continue

                # resize thermal if needed
                if thermal_img.shape[:2] != rgb_img.shape[:2]:
                    thermal_img = cv2.resize(thermal_img, (rgb_img.shape[1], rgb_img.shape[0]))

                # ensure uint8
                if rgb_img.dtype != np.uint8:
                    rgb_img = (rgb_img * 255).astype(np.uint8)
                if thermal_img.dtype != np.uint8:
                    thermal_img = (thermal_img * 255).astype(np.uint8)

                # prepare frames as 2D arrays: (n_frames, H, W)
                r, g, b = cv2.split(rgb_img)
                frames = [r, g, b, thermal_img]

                dst_img = out_img_dir / f"{xml_path.stem}.tiff"
                # cv2.imwritemulti writes multiple pages; convert frames to list of ndarray
                cv2.imwritemulti(str(dst_img), frames)

                converted += 1
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
                skipped += 1

        print(f"{split}: converted {converted}, skipped {skipped}")
    
    def create_dataset_info(self, output_dir: str = None):
        """
        Create dataset info file with class names.
        
        Args:
            output_dir: Output directory for info file
        """
        if output_dir is None:
            output_dir = self.output_root
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create classes.txt file
        classes_file = output_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            for class_name in self.xml_parser.CLASS_NAMES:
                f.write(f"{class_name}\n")
        
        print(f"Dataset info saved to {classes_file}")
    
    def create_yolo_dataset_yaml(self, output_path: str = None, channels: int = 3):
        """
        Create YOLO dataset.yaml configuration file.
        
        Args:
            output_path: Output path for dataset.yaml
        """
        if output_path is None:
            output_path = self.output_root / 'dataset.yaml'
        else:
            output_path = Path(output_path)

        # Build dictionary for yaml
        content = {
            'path': str(self.output_root),
            'train': f'{self.images_dir_name}/train',
            'val': f'{self.images_dir_name}/val',
            'test': f'{self.images_dir_name}/test',
            'names': list(self.xml_parser.CLASS_NAMES),
            'channels': channels,
        }

        # use yaml.safe_dump for consistent formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(content, f, sort_keys=False)

        print(f"YOLO dataset.yaml saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone XML annotations to YOLO format'
    )
    parser.add_argument(
        'data_root',
        help='Root directory containing train/test folders'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output root directory (defaults to data_root)'
    )
    parser.add_argument(
        '--split',
        default='all',
        choices=['train', 'test', 'val', 'all'],
        help='Dataset split to convert'
    )
    parser.add_argument(
        '--mode',
        choices=['rgb', 'rgbt', 'both'],
        default='both',
        help="Conversion mode: 'rgb' to convert RGB only, 'rgbt' to convert RGB+thermal only, 'both' to run both when available"
    )
    
    args = parser.parse_args()
    
    converter = VisDroneToYOLOConverter(args.data_root, args.output)

    converter.convert_dataset(split=args.split, mode=args.mode)


if __name__ == '__main__':
    main()