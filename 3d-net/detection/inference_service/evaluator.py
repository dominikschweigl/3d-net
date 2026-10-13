"""
Offline evaluation: matches inference detections against Pascal VOC XML ground truth
and computes precision, recall, accuracy, AP@0.5, and mAP@0.5:0.95.

Ground truth format (standard FiftyOne export):
    <dataset_root>/labels/<video>_<frame:06d>.xml  — Pascal VOC XML, absolute pixel coords
"""

import csv
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from .models import BBox, Detection

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Accumulates per-frame predictions vs. ground truth and computes AP/mAP.

    Matching is solved per class with the same greedy IoU-sorted algorithm used
    by YOLOv11's match_predictions: all overlapping (pred, GT) pairs are sorted by
    IoU descending, then duplicates are removed per prediction and per GT box.
    The matched IoU is stored so AP can be computed at any threshold without re-running inference.

    Call record_frame() per inference frame, then summarize() at the end.
    """

    _TIMING_KEYS = ("frame_provider_ms", "align_ms", "pre_ms", "infer_ms", "post_ms", "track_ms")
    # Keys that form the sequential critical path; used for total_ms.
    # align_ms and pre_ms are excluded — they overlap with frame_provider_ms in the concurrent pipeline.
    _TOTAL_KEYS = ("frame_provider_ms", "infer_ms", "post_ms", "track_ms")
    _COCO_IOU_THRESHOLDS = np.arange(0.50, 0.955, 0.05)  # 0.50, 0.55, …, 0.95

    def __init__(
        self,
        class_names: List[str],
        iou_threshold: float = 0.5,
        iou_range: bool = True,
        output_dir: Path = Path("logs/evaluation"),
        pipeline_workers: int = 1,
    ):
        self._class_names = class_names
        self._iou_threshold = iou_threshold
        self._iou_range = iou_range
        self._output_dir = Path(output_dir)
        self._pipeline_workers = pipeline_workers

        # Per-class accumulator: list of (prediction_confidence, best_matched_gt_iou).
        # best_matched_gt_iou = 0.0 means no GT box of that class could be matched.
        self._per_class_pred_ious: Dict[str, List[Tuple[float, float]]] = {
            class_name: [] for class_name in class_names
        }
        self._per_class_gt_count: Dict[str, int] = {
            class_name: 0 for class_name in class_names
        }

        self._per_timing_ms: Dict[str, List[float]] = {
            key: [] for key in self._TIMING_KEYS
        }
        self._per_fps: List[float] = []
        self._per_frame_rows: List[dict] = []
        self._telemetry_rows: List[dict] = []
        self._evaluated_frame_count = 0
        self._skipped_frame_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def load_ground_truth(xml_path: Path, class_names: List[str]) -> List[Detection]:
        """Parse a Pascal VOC XML annotation file into Detection objects (confidence=1.0)."""
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
        ground_truth: List[Detection] = []
        for obj in xml_root.iter("object"):
            class_name = (obj.findtext("name") or "").strip()
            if class_name not in class_names:
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            x_min = int(float(bndbox.findtext("xmin") or 0))
            y_min = int(float(bndbox.findtext("ymin") or 0))
            x_max = int(float(bndbox.findtext("xmax") or 0))
            y_max = int(float(bndbox.findtext("ymax") or 0))
            ground_truth.append(Detection(
                name=class_name,
                label=class_names.index(class_name),
                bbox=BBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max),
                confidence=1.0,
            ))
        return ground_truth

    def record_frame(
        self,
        frame_stem: str,
        predictions: List[Detection],
        ground_truth: List[Detection],
        timings_ms: Dict[str, float],
    ) -> None:
        """Match predictions to ground truth and accumulate statistics for this frame."""
        self._evaluated_frame_count += 1

        for gt_detection in ground_truth:
            self._per_class_gt_count[gt_detection.name] = (
                self._per_class_gt_count.get(gt_detection.name, 0) + 1
            )

        for timing_key in self._TIMING_KEYS:
            if timing_key in timings_ms:
                self._per_timing_ms[timing_key].append(timings_ms[timing_key])

        matched_iou_by_pred = np.zeros(len(predictions), dtype=float)

        for class_name in self._class_names:
            pred_indices = [
                pred_idx for pred_idx, pred in enumerate(predictions)
                if pred.name == class_name
            ]
            gt_indices = [
                gt_idx for gt_idx, gt in enumerate(ground_truth)
                if gt.name == class_name
            ]
            if not pred_indices or not gt_indices:
                continue

            iou_matrix = np.zeros((len(pred_indices), len(gt_indices)), dtype=float)
            for pred_row, pred_idx in enumerate(pred_indices):
                for gt_col, gt_idx in enumerate(gt_indices):
                    iou_matrix[pred_row, gt_col] = self._compute_iou(
                        predictions[pred_idx].bbox,
                        ground_truth[gt_idx].bbox,
                    )

            # Greedy IoU-sorted matching (mirrors YOLOv8 match_predictions, validator.py:256-265).
            matches = np.argwhere(iou_matrix > 0)
            if len(matches):
                matches = matches[iou_matrix[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                for pred_row, gt_col in matches:
                    matched_iou_by_pred[pred_indices[pred_row]] = iou_matrix[pred_row, gt_col]

        for pred_idx, predicted_detection in enumerate(predictions):
            self._per_class_pred_ious[predicted_detection.name].append(
                (predicted_detection.confidence, float(matched_iou_by_pred[pred_idx]))
            )

        true_positive_count = int(np.sum(matched_iou_by_pred >= self._iou_threshold))
        false_positive_count = len(predictions) - true_positive_count
        false_negative_count = max(0, len(ground_truth) - true_positive_count)

        self._per_frame_rows.append({
            "frame_stem": frame_stem,
            "n_gt": len(ground_truth),
            "n_pred": len(predictions),
            "tp": true_positive_count,
            "fp": false_positive_count,
            "fn": false_negative_count,
            **{key: round(timings_ms.get(key, 0.0), 3) for key in self._TIMING_KEYS},
        })

    def skip_frame(self, frame_stem: Optional[str] = None) -> None:
        """Count a frame that had no annotation file and was excluded from evaluation."""
        self._skipped_frame_count += 1

    def run_from_logs(
        self,
        stats_csv: Path,
        detections_csv: Optional[Path],
        labels_dir: Path,
    ) -> None:
        """
        Populate the evaluator from a session's CSV log files.

        Reads timing data from stats_csv and predictions from detections_csv,
        loads ground truth from <labels_dir>/<frame_stem>.xml, then calls
        record_frame() / skip_frame() for each logged frame in order.
        """
        # Build frame_stem → timings from stats CSV (one row per frame, in order).
        ordered_stems: List[str] = []
        stem_to_timings: dict = {}
        stem_to_fps: dict = {}
        stem_to_telemetry_idx: dict = {}

        with open(stats_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                frame_stem = row.get("frame_stem", "").strip()
                if not frame_stem:
                    continue
                if frame_stem not in stem_to_timings:
                    ordered_stems.append(frame_stem)
                stem_to_timings[frame_stem] = {
                    "frame_provider_ms": float(row.get("frame_provider_ms", 0) or 0),
                    "align_ms":          float(row.get("align_ms",          0) or 0),
                    "pre_ms":            float(row.get("pre_ms",            0) or 0),
                    "infer_ms":          float(row.get("infer_ms",          0) or 0),
                    "post_ms":           float(row.get("post_ms",           0) or 0),
                    "track_ms":          float(row.get("track_ms",          0) or 0),
                }
                stem_to_fps[frame_stem] = row.get("fps", "")

                telemetry_row = {
                    "frame_stem": frame_stem,
                    "hailo_temp_c": self._safe_float(row.get("hailo_temp_c")),
                    "hailo_pct": self._safe_float(row.get("hailo_pct")),
                    "cpu_temp_c": self._safe_float(row.get("cpu_temp_c")),
                    "cpu_pct": self._safe_float(row.get("cpu_pct")),
                    "ram_used_mb": self._safe_float(row.get("ram_used_mb")),
                }
                if frame_stem not in stem_to_telemetry_idx:
                    stem_to_telemetry_idx[frame_stem] = len(self._telemetry_rows)
                    self._telemetry_rows.append(telemetry_row)
                else:
                    telemetry_idx = stem_to_telemetry_idx[frame_stem]
                    self._telemetry_rows[telemetry_idx] = telemetry_row

        # Build frame_stem → list of Detection from detections CSV.
        stem_to_predictions: dict = {}
        if detections_csv and detections_csv.exists():
            with open(detections_csv, newline="") as fh:
                for row in csv.DictReader(fh):
                    frame_stem = row.get("frame_stem", "").strip()
                    if not frame_stem:
                        continue
                    stem_to_predictions.setdefault(frame_stem, []).append(
                        Detection(
                            name=row["class_name"],
                            label=int(row["label"]),
                            bbox=BBox(
                                x1=int(row["x1"]),
                                y1=int(row["y1"]),
                                x2=int(row["x2"]),
                                y2=int(row["y2"]),
                            ),
                            confidence=float(row["confidence"]),
                            track_id=int(row["track_id"]) if row.get("track_id") else None,
                        )
                    )

        for frame_stem in ordered_stems:
            annotation_path = labels_dir / f"{frame_stem}.xml"
            if not annotation_path.exists():
                self.skip_frame(frame_stem)
                continue

            ground_truth = self.load_ground_truth(annotation_path, self._class_names)
            predictions = stem_to_predictions.get(frame_stem, [])
            timings_ms = stem_to_timings[frame_stem]
            fps_val = self._safe_float(stem_to_fps.get(frame_stem))
            if fps_val is not None and fps_val > 0:
                self._per_fps.append(fps_val)

            self.record_frame(
                frame_stem=frame_stem,
                predictions=predictions,
                ground_truth=ground_truth,
                timings_ms=timings_ms,
            )

        logger.info(
            "run_from_logs: %d frames evaluated, %d skipped",
            self._evaluated_frame_count, self._skipped_frame_count,
        )

    def summarize(self) -> dict:
        """Compute final metrics, write output files, and return the summary dict."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        per_class_metrics: Dict[str, dict] = {}
        ap50_values: List[float] = []
        ap_range_values: List[float] = []

        for class_name in self._class_names:
            n_gt = self._per_class_gt_count[class_name]
            class_pred_ious = self._per_class_pred_ious[class_name]

            ap_at_threshold = self._compute_ap_at_iou(
                class_pred_ious, n_gt, self._iou_threshold
            )
            ap50_values.append(ap_at_threshold)

            mean_ap_over_range = 0.0
            if self._iou_range:
                mean_ap_over_range = float(np.mean([
                    self._compute_ap_at_iou(class_pred_ious, n_gt, iou_thresh)
                    for iou_thresh in self._COCO_IOU_THRESHOLDS
                ]))
                ap_range_values.append(mean_ap_over_range)

            # Precision/recall at the configured iou_threshold (all-confidence summary).
            true_positives = sum(
                1 for _, matched_iou in class_pred_ious
                if matched_iou >= self._iou_threshold
            )
            false_positives = len(class_pred_ious) - true_positives
            false_negatives = max(0, n_gt - true_positives)
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0 else 0.0
            )
            recall = true_positives / n_gt if n_gt > 0 else 0.0
            accuracy = (
                true_positives / (true_positives + false_positives + false_negatives)
                if (true_positives + false_positives + false_negatives) > 0 else 0.0
            )

            class_entry: dict = {
                f"AP_{int(self._iou_threshold * 100)}": round(ap_at_threshold, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "accuracy": round(accuracy, 4),
                "n_gt": n_gt,
                "n_pred": len(class_pred_ious),
            }
            if self._iou_range:
                class_entry["AP_50_95"] = round(mean_ap_over_range, 4)
            per_class_metrics[class_name] = class_entry

        map_at_threshold = float(np.mean(ap50_values)) if ap50_values else 0.0
        map_50_95 = float(np.mean(ap_range_values)) if ap_range_values else None

        summary: dict = {
            f"mAP_{int(self._iou_threshold * 100)}": round(map_at_threshold, 4),
            "per_class": per_class_metrics,
            "timing": self._compute_timing_stats(),
            "pipeline_workers": self._pipeline_workers,
            "frames_evaluated": self._evaluated_frame_count,
            "frames_skipped": self._skipped_frame_count,
        }
        if map_50_95 is not None:
            summary["mAP_50_95"] = round(map_50_95, 4)

        chart_paths = self._write_telemetry_charts()
        if chart_paths:
            summary["charts"] = chart_paths

        summary_path = self._output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Evaluation summary → %s", summary_path)

        if self._per_frame_rows:
            csv_path = self._output_dir / "per_frame.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._per_frame_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._per_frame_rows)
            logger.info("Per-frame CSV     → %s", csv_path)

        return summary

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _write_telemetry_charts(self) -> List[str]:
        if not self._telemetry_rows:
            return []

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed; skipping telemetry charts")
            return []

        frame_index = np.arange(1, len(self._telemetry_rows) + 1)
        hailo_temp = np.array([
            row["hailo_temp_c"] if row["hailo_temp_c"] is not None else np.nan
            for row in self._telemetry_rows
        ], dtype=float)
        cpu_temp = np.array([
            row["cpu_temp_c"] if row["cpu_temp_c"] is not None else np.nan
            for row in self._telemetry_rows
        ], dtype=float)
        hailo_pct = np.array([
            row["hailo_pct"] if row.get("hailo_pct") is not None else np.nan
            for row in self._telemetry_rows
        ], dtype=float)
        cpu_pct = np.array([
            row["cpu_pct"] if row["cpu_pct"] is not None else np.nan
            for row in self._telemetry_rows
        ], dtype=float)
        ram_used_mb = np.array([
            row["ram_used_mb"] if row["ram_used_mb"] is not None else np.nan
            for row in self._telemetry_rows
        ], dtype=float)

        chart_paths: List[str] = []

        if np.isfinite(hailo_temp).any() or np.isfinite(cpu_temp).any():
            fig, ax = plt.subplots(figsize=(10, 4.5))
            if np.isfinite(hailo_temp).any():
                ax.plot(frame_index, hailo_temp, label="Hailo temp (C)", linewidth=1.5)
            if np.isfinite(cpu_temp).any():
                ax.plot(frame_index, cpu_temp, label="CPU temp (C)", linewidth=1.5)
            ax.set_title("Hailo and CPU Temperature")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Temperature (C)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()

            temp_chart_path = self._output_dir / "temperature_chart.png"
            fig.savefig(temp_chart_path, dpi=150)
            plt.close(fig)
            chart_paths.append(str(temp_chart_path))
            logger.info("Telemetry chart   → %s", temp_chart_path)

        if (
            np.isfinite(hailo_pct).any()
            or np.isfinite(cpu_pct).any()
            or np.isfinite(ram_used_mb).any()
        ):
            fig, ax_cpu = plt.subplots(figsize=(10, 4.5))
            lines = []
            labels = []

            if np.isfinite(hailo_pct).any():
                hailo_line = ax_cpu.plot(
                    frame_index,
                    hailo_pct,
                    color="tab:green",
                    label="Hailo usage (%)",
                    linewidth=1.5,
                )[0]
                lines.append(hailo_line)
                labels.append(hailo_line.get_label())
            if np.isfinite(cpu_pct).any():
                cpu_line = ax_cpu.plot(
                    frame_index,
                    cpu_pct,
                    color="tab:blue",
                    label="CPU usage (%)",
                    linewidth=1.5,
                )[0]
                lines.append(cpu_line)
                labels.append(cpu_line.get_label())
            ax_cpu.set_xlabel("Frame")
            ax_cpu.set_ylabel("Utilization (%)")
            ax_cpu.grid(True, alpha=0.3)

            ax_ram = ax_cpu.twinx()
            if np.isfinite(ram_used_mb).any():
                ram_line = ax_ram.plot(
                    frame_index,
                    ram_used_mb,
                    color="tab:orange",
                    label="RAM used (MB)",
                    linewidth=1.5,
                )[0]
                lines.append(ram_line)
                labels.append(ram_line.get_label())
            ax_ram.set_ylabel("RAM used (MB)", color="tab:orange")
            ax_ram.tick_params(axis="y", labelcolor="tab:orange")

            ax_cpu.set_title("CPU and RAM Usage")
            if lines:
                ax_cpu.legend(lines, labels, loc="best")
            fig.tight_layout()

            usage_chart_path = self._output_dir / "cpu_ram_usage_chart.png"
            fig.savefig(usage_chart_path, dpi=150)
            plt.close(fig)
            chart_paths.append(str(usage_chart_path))
            logger.info("Telemetry chart   → %s", usage_chart_path)

        return chart_paths

    @staticmethod
    def _compute_iou(box_a: BBox, box_b: BBox) -> float:
        intersection_x1 = max(box_a.x1, box_b.x1)
        intersection_y1 = max(box_a.y1, box_b.y1)
        intersection_x2 = min(box_a.x2, box_b.x2)
        intersection_y2 = min(box_a.y2, box_b.y2)
        intersection_area = (
            max(0, intersection_x2 - intersection_x1)
            * max(0, intersection_y2 - intersection_y1)
        )
        area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
        area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
        union_area = area_a + area_b - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _compute_ap_at_iou(
        pred_ious: List[Tuple[float, float]],
        n_gt: int,
        iou_threshold: float,
    ) -> float:
        """Area-under-curve AP at a single IoU threshold using precision envelope."""
        if n_gt == 0 or not pred_ious:
            return 0.0

        sorted_by_confidence = sorted(pred_ious, key=lambda x: x[0], reverse=True)
        is_tp_array = np.array([
            1 if matched_iou >= iou_threshold else 0
            for _, matched_iou in sorted_by_confidence
        ])
        tp_cumulative = np.cumsum(is_tp_array)
        fp_cumulative = np.cumsum(1 - is_tp_array)

        recall_curve = tp_cumulative / n_gt
        precision_curve = tp_cumulative / (tp_cumulative + fp_cumulative)

        recall_with_endpoints = np.concatenate(([0.0], recall_curve, [1.0]))
        precision_with_endpoints = np.concatenate(([1.0], precision_curve, [0.0]))
        precision_envelope = np.maximum.accumulate(precision_with_endpoints[::-1])[::-1]
        # 101-point COCO interpolation (mirrors YOLOv8 compute_ap, metrics.py:1213-1214).
        x = np.linspace(0, 1, 101)
        return float(np.trapezoid(np.interp(x, recall_with_endpoints, precision_envelope), x))

    def _compute_timing_stats(self) -> dict:
        stats: dict = {}
        non_empty_keys = [k for k in self._TIMING_KEYS if self._per_timing_ms[k]]

        for key in non_empty_keys:
            values = np.array(self._per_timing_ms[key])
            stats[key] = {
                "mean": round(float(np.mean(values)), 3),
                "p95":  round(float(np.percentile(values, 95)), 3),
                "p99":  round(float(np.percentile(values, 99)), 3),
            }

        # total_ms uses only the sequential critical path keys; align_ms and pre_ms are
        # excluded because they run in parallel with other frames inside the frame provider.
        total_keys = [k for k in self._TOTAL_KEYS if self._per_timing_ms.get(k)]
        min_length = min((len(self._per_timing_ms[k]) for k in total_keys), default=0)
        if min_length > 0:
            total_ms_per_frame = np.array([
                sum(self._per_timing_ms[k][frame_idx] for k in total_keys)
                for frame_idx in range(min_length)
            ])
            stats["total_ms"] = {
                "mean": round(float(np.mean(total_ms_per_frame)), 3),
                "p95":  round(float(np.percentile(total_ms_per_frame, 95)), 3),
                "p99":  round(float(np.percentile(total_ms_per_frame, 99)), 3),
            }

        if self._per_fps:
            fps_arr = np.array(self._per_fps)
            stats["fps"] = {
                "mean": round(float(np.mean(fps_arr)), 2),
                "p95":  round(float(np.percentile(fps_arr, 95)), 2),
                "p99":  round(float(np.percentile(fps_arr, 99)), 2),
            }

        return stats
