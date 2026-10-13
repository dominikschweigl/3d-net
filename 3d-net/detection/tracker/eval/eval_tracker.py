"""Evaluate a single configurable tracker on the 3D-Net disk dataset using
real Hailo model inference as the detection source.

Unlike compare_trackers.py (which feeds GT boxes into every tracker), this script
runs the actual model and measures end-to-end tracking quality: HOTA / CLEAR / Identity.

GT track IDs are read from the <index> field in the Pascal VOC XML labels.
Run detection/fiftyone/download_rgbt_dataset.py first to export the dataset with
those index values stamped into the XML (requires FiftyOne on the host machine).

Usage
-----
    python eval_tracker.py --split val
    python eval_tracker.py --tracker SORT --max-age 20
    python eval_tracker.py --hef path/to/model.hef --conf 0.3
    python eval_tracker.py --split val --data-root /app/detection/data/3d_net
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in Docker / headless environments
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker
from trackers.eval import box_iou, compute_clear_metrics, compute_hota_metrics, compute_identity_metrics

DETECTION_ROOT = Path(__file__).resolve().parents[2]

# Add the detection package to the path so inference_service is importable.
sys.path.insert(0, str(DETECTION_ROOT))
from inference_service.config import (  # noqa: E402
    ByteTrackConfig,
    ModalityMode,
    OCSORTConfig,
    SORTConfig,
)
from inference_service.frame_provider import FramePair  # noqa: E402
from inference_service.preprocessor import Preprocessor  # noqa: E402


# ── Configuration — edit these to change the evaluation ───────────────────────
DATA_ROOT      = str(DETECTION_ROOT / "data" / "3d_net")
HEF_PATH       = "models/midP3_fusion_3d_net.hef"
MODALITY       = ModalityMode.RGBT      # ModalityMode.RGB | .THERMAL | .RGBT
CONF_THRESHOLD = 0.2
CLASS_NAMES    = ["person", "car", "motorcycle", "bicycle", "bus", "truck"]
TRACKER        = "ByteTrack"            # "SORT" | "ByteTrack" | "OC-SORT"
MAX_AGE        = 30
VIDEO_FPS      = 6.0                    # playback fps for output videos
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_INPUT_SIZE = 640
_FRAME_SUFFIX_LEN = 7  # "_NNNNNN"


# ──────────────────────────────────────────────────────────────────────────────
# Disk-based dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_sessions(data_root: Path, split: str) -> list[tuple[str, list[str]]]:
    """Parse <split>.txt and group frame stems by video name.

    Returns [(video_name, [sorted_stems]), …].
    Video name is the stem with the trailing _NNNNNN frame-number suffix removed.
    """
    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    lines = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    sessions: dict[str, list[str]] = {}
    for fname in lines:
        stem  = Path(fname).stem
        video = stem[:-_FRAME_SUFFIX_LEN]
        sessions.setdefault(video, []).append(stem)
    for v in sessions:
        sessions[v].sort()
    return list(sessions.items())


def _parse_gt_xml(xml_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes_xyxy, track_ids) from a Pascal VOC XML file.

    Only includes objects that have an <index> element (track ID).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, ids = [], []
    for obj in root.findall("object"):
        idx = obj.findtext("index")
        if idx is None:
            continue
        bb = obj.find("bndbox")
        boxes.append([
            float(bb.findtext("xmin")),
            float(bb.findtext("ymin")),
            float(bb.findtext("xmax")),
            float(bb.findtext("ymax")),
        ])
        ids.append(int(idx))
    if not boxes:
        return np.empty((0, 4)), np.empty(0, dtype=int)
    return np.array(boxes, dtype=float), np.array(ids, dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# Tracker wrapper — uniform interface: update(xyxy, conf, class_id, img) → (xyxy, xyxy, ids)
# ──────────────────────────────────────────────────────────────────────────────

class _TrackerWrapper:
    """Wraps SORT / ByteTrack / OC-SORT behind a single interface.

    Returns (input_xyxy, pred_xyxy, ids) where:
      input_xyxy — detection boxes matched to confirmed tracks
      pred_xyxy  — Kalman-state boxes from the tracker's internal objects
    """

    def __init__(self, tracker_cls, **kwargs):
        self._cls    = tracker_cls
        self._kwargs = kwargs
        self._tracker = tracker_cls(**kwargs)

    def reset(self):
        self._tracker = self._cls(**self._kwargs)

    def update(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        class_id: np.ndarray,
        _img: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(xyxy) == 0:
            dets = sv.Detections.empty()
        else:
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        result = self._tracker.update(dets)
        if result.tracker_id is None or len(result.tracker_id) == 0:
            return np.empty((0, 4)), np.empty((0, 4)), np.empty(0, dtype=int)

        mask        = result.tracker_id >= 0
        input_xyxy  = result.xyxy[mask]
        ids         = result.tracker_id[mask]

        internal_tracks = getattr(self._tracker, "tracks", None) or getattr(self._tracker, "trackers", [])
        id_to_state = {t.tracker_id: t.get_state_bbox() for t in internal_tracks if hasattr(t, "get_state_bbox")}
        pred_rows   = [id_to_state[tid] if tid in id_to_state else input_xyxy[i] for i, tid in enumerate(ids)]
        pred_xyxy   = np.array(pred_rows, dtype=float).reshape(-1, 4) if pred_rows else np.empty((0, 4))

        return input_xyxy, pred_xyxy, ids


def make_tracker(name: str, max_age: int) -> _TrackerWrapper:
    """Return a _TrackerWrapper for the named tracker using production defaults."""
    n = name.lower().replace("-", "").replace("_", "")
    if n == "sort":
        cfg = SORTConfig(lost_track_buffer=max_age)
        return _TrackerWrapper(
            SORTTracker,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            track_activation_threshold=cfg.track_activation_threshold,
            minimum_iou_threshold=cfg.minimum_iou_threshold,
            frame_rate=6,
        )
    if n == "bytetrack":
        cfg = ByteTrackConfig(lost_track_buffer=max_age)
        return _TrackerWrapper(
            ByteTrackTracker,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            track_activation_threshold=cfg.track_activation_threshold,
            minimum_iou_threshold=cfg.minimum_iou_threshold,
            high_conf_det_threshold=cfg.high_conf_det_threshold,
            frame_rate=6,
        )
    if n == "ocsort":
        cfg = OCSORTConfig(lost_track_buffer=max_age)
        return _TrackerWrapper(
            OCSORTTracker,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            minimum_iou_threshold=cfg.minimum_iou_threshold,
            high_conf_det_threshold=cfg.high_conf_det_threshold,
            direction_consistency_weight=cfg.direction_consistency_weight,
            delta_t=cfg.delta_t,
            frame_rate=6,
        )
    raise ValueError(f"Unknown tracker '{name}'. Choose from: SORT, ByteTrack, OC-SORT")


# ──────────────────────────────────────────────────────────────────────────────
# Hailo post-processing
# ──────────────────────────────────────────────────────────────────────────────

def _postprocess(
    batch_result,
    conf_threshold: float,
    crop_w: int,
    crop_h: int,
    lb_scale: float,
    lb_pad_x: int,
    lb_pad_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Hailo NMS output to (xyxy, conf, class_id) numpy arrays.

    batch_result is a list indexed by class; each entry is an ndarray of shape
    (N, 5) with rows [y1_norm, x1_norm, y2_norm, x2_norm, score] (YXYX normalised).
    Coordinates are de-projected from the letterboxed 640×640 space back to crop space.
    """
    xyxy_rows, conf_rows, cls_rows = [], [], []
    for cls_idx, cls_dets in enumerate(batch_result):
        if len(cls_dets) == 0:
            continue
        for det in cls_dets:
            y1_n, x1_n, y2_n, x2_n, score = det
            if score < conf_threshold:
                continue
            x1 = max(0, min(crop_w, int((x1_n * _MODEL_INPUT_SIZE - lb_pad_x) / lb_scale)))
            y1 = max(0, min(crop_h, int((y1_n * _MODEL_INPUT_SIZE - lb_pad_y) / lb_scale)))
            x2 = max(0, min(crop_w, int((x2_n * _MODEL_INPUT_SIZE - lb_pad_x) / lb_scale)))
            y2 = max(0, min(crop_h, int((y2_n * _MODEL_INPUT_SIZE - lb_pad_y) / lb_scale)))
            xyxy_rows.append([x1, y1, x2, y2])
            conf_rows.append(score)
            cls_rows.append(cls_idx)

    if not xyxy_rows:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    return (
        np.array(xyxy_rows, dtype=float),
        np.array(conf_rows, dtype=float),
        np.array(cls_rows, dtype=int),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Video visualisation
# ──────────────────────────────────────────────────────────────────────────────

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 2
_GREEN      = (0, 255,   0)   # tracker output
_RED        = (0,   0, 255)   # GT annotation


def _draw_tracking_frame(
    rgb_bgr:  np.ndarray,
    ir_bgr:   np.ndarray | None,
    tr_xyxy:  np.ndarray,
    tr_ids:   np.ndarray,
    gt_boxes: np.ndarray,
    gt_ids:   np.ndarray,
) -> np.ndarray:
    """Render one video frame: tracker boxes (green) + GT boxes (red), RGB | IR side-by-side."""

    def _draw(canvas: np.ndarray) -> np.ndarray:
        out = canvas.copy()
        for box, tid in zip(tr_xyxy, tr_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), _GREEN, _THICKNESS)
            cv2.putText(out, f"T{tid}", (x1, max(y1 - 3, 10)), _FONT, _FONT_SCALE, _GREEN, _THICKNESS)
        for box, gid in zip(gt_boxes, gt_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), _RED, _THICKNESS)
            cv2.putText(out, f"G{gid}", (x1, min(y2 + 12, canvas.shape[0] - 2)), _FONT, _FONT_SCALE, _RED, _THICKNESS)
        return out

    rgb_vis = _draw(rgb_bgr)
    if ir_bgr is not None:
        ir_vis = _draw(cv2.cvtColor(cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        return np.concatenate([rgb_vis, ir_vis], axis=1)
    return rgb_vis


# ──────────────────────────────────────────────────────────────────────────────
# Per-session evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    gt_ids:    list[np.ndarray] = field(default_factory=list)
    tr_ids:    list[np.ndarray] = field(default_factory=list)
    sims:      list[np.ndarray] = field(default_factory=list)      # GT vs input detection boxes
    sims_pred: list[np.ndarray] = field(default_factory=list)      # GT vs Kalman-predicted boxes
    latencies: list[float]      = field(default_factory=list)      # ms per frame
    det_mask:  list[bool]       = field(default_factory=list)      # True if frame had GT detections


def eval_session(
    stems: list[str],
    data_root: Path,
    tracker: _TrackerWrapper,
    pipeline,
    input_info,
    preprocessor: Preprocessor,
    conf_threshold: float,
    crop_w: int,
    crop_h: int,
    modality: ModalityMode,
    video_path: Path | None = None,
    video_fps: float = VIDEO_FPS,
) -> SessionResult:
    rgb_dir    = data_root / "rgb"
    ir_dir     = data_root / "ir"
    labels_dir = data_root / "labels"
    result     = SessionResult()

    writer: cv2.VideoWriter | None = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_w = crop_w * 2 if modality == ModalityMode.RGBT else crop_w
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (frame_w, crop_h),
        )
        logging.info("Recording video to %s", video_path)

    for stem in stems:
        # ── Load images ───────────────────────────────────────────────────────
        rgb_bgr = cv2.imread(str(rgb_dir / f"{stem}.jpg"))
        if rgb_bgr is None:
            logging.warning("Could not read RGB %s — skipping", stem)
            continue

        ir_bgr = cv2.imread(str(ir_dir / f"{stem}.jpg"))
        if ir_bgr is None and modality == ModalityMode.RGBT:
            logging.warning("Could not read IR %s — skipping", stem)
            continue

        img_h, img_w = rgb_bgr.shape[:2]
        rgb_crop  = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        gray_crop = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY) if ir_bgr is not None else np.zeros((img_h, img_w), dtype=np.uint8)

        # ── Prepare BGRT tensor ───────────────────────────────────────────────
        pair = FramePair(rgb_crop=rgb_crop, gray_crop=gray_crop)
        prep = preprocessor.prepare(0, pair)

        # ── Hailo inference ───────────────────────────────────────────────────
        t0           = time.perf_counter()
        results      = pipeline.infer({input_info.name: prep.tensor})
        batch_result = results[list(results.keys())[0]][0]
        elapsed_ms   = (time.perf_counter() - t0) * 1e3

        # ── Post-process ──────────────────────────────────────────────────────
        det_xyxy, det_conf, det_cls = _postprocess(
            batch_result, conf_threshold, crop_w, crop_h,
            prep.lb_scale, prep.lb_pad_x, prep.lb_pad_y,
        )

        # ── GT from XML ───────────────────────────────────────────────────────
        xml_path = labels_dir / f"{stem}.xml"
        if xml_path.exists():
            gt_boxes, gt_ids = _parse_gt_xml(xml_path)
        else:
            gt_boxes = np.empty((0, 4))
            gt_ids   = np.empty(0, dtype=int)

        # ── Track ─────────────────────────────────────────────────────────────
        tr_input_xyxy, tr_pred_xyxy, tr_ids = tracker.update(det_xyxy, det_conf, det_cls, rgb_bgr)

        # ── IoU similarity matrices ───────────────────────────────────────────
        if len(gt_ids) > 0 and len(tr_ids) > 0:
            sim      = box_iou(gt_boxes, tr_input_xyxy)
            sim_pred = box_iou(gt_boxes, tr_pred_xyxy)
        elif len(gt_ids) == 0 and len(tr_ids) == 0:
            sim = sim_pred = np.empty((0, 0))
        else:
            sim = sim_pred = np.zeros((len(gt_ids), len(tr_ids)))

        result.gt_ids.append(gt_ids)
        result.tr_ids.append(tr_ids)
        result.sims.append(sim)
        result.sims_pred.append(sim_pred)
        result.latencies.append(elapsed_ms)
        result.det_mask.append(len(gt_ids) > 0)

        # ── Video frame ───────────────────────────────────────────────────────
        if writer is not None:
            vis = _draw_tracking_frame(rgb_bgr, ir_bgr, tr_input_xyxy, tr_ids, gt_boxes, gt_ids)
            writer.write(vis)

    if writer is not None:
        writer.release()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation + reporting (identical to compare_trackers.py)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(all_results: list[SessionResult]) -> tuple[dict, SessionResult]:
    combined = SessionResult()
    for res in all_results:
        combined.gt_ids.extend(res.gt_ids)
        combined.tr_ids.extend(res.tr_ids)
        combined.sims.extend(res.sims)
        combined.sims_pred.extend(res.sims_pred)
        combined.latencies.extend(res.latencies)
        combined.det_mask.extend(res.det_mask)

    hota     = compute_hota_metrics(combined.gt_ids, combined.tr_ids, combined.sims)
    clear    = compute_clear_metrics(combined.gt_ids, combined.tr_ids, combined.sims)
    identity = compute_identity_metrics(combined.gt_ids, combined.tr_ids, combined.sims)

    hota_pred     = compute_hota_metrics(combined.gt_ids, combined.tr_ids, combined.sims_pred)
    clear_pred    = compute_clear_metrics(combined.gt_ids, combined.tr_ids, combined.sims_pred)
    identity_pred = compute_identity_metrics(combined.gt_ids, combined.tr_ids, combined.sims_pred)
    pred_metrics  = {k + "_pred": v for d in (hota_pred, clear_pred, identity_pred) for k, v in d.items()}

    lat     = np.array(combined.latencies)
    lat_det = lat[np.array(combined.det_mask)] if combined.det_mask else np.empty(0)

    def _lat_stats(a: np.ndarray, suffix: str) -> dict:
        if len(a) == 0:
            return {f"Lat_mean{suffix}": float("nan"), f"Lat_p50{suffix}": float("nan"), f"Lat_p95{suffix}": float("nan")}
        return {
            f"Lat_mean{suffix}": float(np.mean(a)),
            f"Lat_p50{suffix}":  float(np.percentile(a, 50)),
            f"Lat_p95{suffix}":  float(np.percentile(a, 95)),
        }

    latency_metrics = {**_lat_stats(lat, ""), **_lat_stats(lat_det, "_det")}
    metrics = {**hota, **clear, **identity, **latency_metrics, **pred_metrics}
    return metrics, combined


def print_results(tracker_name: str, metrics: dict) -> None:
    perf_keys    = ["HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDP", "IDR"]
    lat_keys_all = ["Lat_mean",     "Lat_p50",     "Lat_p95"]
    lat_keys_det = ["Lat_mean_det", "Lat_p50_det", "Lat_p95_det"]
    available    = [k for k in perf_keys if k in metrics]

    def _print_table(title: str, key_suffix: str, show_latency: bool) -> None:
        header = f"{'Metric':<10}" + "".join(f"  {k:>8}" for k in available)
        if show_latency:
            header += "  │" + "".join(f"  {k:>10}" for k in lat_keys_all) + "  │" + "".join(f"  {k:>12}" for k in lat_keys_det)
        print(f"\n{title}  [{tracker_name}]")
        print(header)
        print("-" * len(header))
        row = f"{'':10}"
        for k in available:
            row += f"  {metrics.get(k + key_suffix, float('nan')):>8.4f}"
        if show_latency:
            row += "  │"
            for k in lat_keys_all:
                row += f"  {metrics.get(k, float('nan')):>8.2f}ms"
            row += "  │"
            for k in lat_keys_det:
                row += f"  {metrics.get(k, float('nan')):>10.2f}ms"
        print(row)
        print()

    _print_table("Input detection boxes (Hailo predictions)", "",      show_latency=True)
    _print_table("Tracker-predicted bounding boxes",          "_pred", show_latency=False)


def plot_latencies(combined: SessionResult, tracker_name: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles   = ["All frames", "Frames with GT detections only"]

    for ax, title, det_only in zip(axes, titles, [False, True]):
        lat = np.array(combined.latencies)
        if det_only:
            lat = lat[np.array(combined.det_mask)]
        if len(lat) == 0:
            continue
        bins = np.logspace(np.log10(max(lat.min(), 1e-3)), np.log10(lat.max()), 50)
        ax.hist(lat, bins=bins, alpha=0.7, label=tracker_name, weights=np.ones(len(lat)) / len(lat))
        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms) — Hailo inference only")
        ax.set_ylabel("Fraction of frames")
        ax.set_title(title)
        ax.legend()

    fig.suptitle(f"Inference latency — {tracker_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved latency plot to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single tracker on the 3D-Net dataset using Hailo inference.",
    )
    parser.add_argument("--data-root",  default=DATA_ROOT,       help="Path to exported 3D-Net dataset root (must contain rgb/, ir/, labels/, <split>.txt)")
    parser.add_argument("--split",      default="val",           help="Split to evaluate: train | val | test (default: %(default)s)")
    parser.add_argument("--hef",        default=HEF_PATH,        help="Path to Hailo HEF model")
    parser.add_argument("--modality",   default=MODALITY.value,  choices=[m.value for m in ModalityMode], help="Input modality")
    parser.add_argument("--conf",       type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--tracker",    default=TRACKER,         help="Tracker to use: SORT | ByteTrack | OC-SORT")
    parser.add_argument("--max-age",    type=int, default=MAX_AGE, help="Max missed frames before a track is dropped")
    parser.add_argument("--output-dir", default=".",             help="Directory where the latency plot and videos are saved (default: cwd)")
    parser.add_argument("--video",      action="store_true",     help="Write one tracking video per session to --output-dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    modality  = ModalityMode(args.modality)
    data_root = Path(args.data_root)

    sessions = _load_sessions(data_root, args.split)
    total_frames = sum(len(s) for _, s in sessions)
    logging.info("Split '%s': %d sessions, %d frames total.", args.split, len(sessions), total_frames)

    # hailo_platform lives in the system Python dist-packages, not the venv.
    # Add the system path if it isn't already on sys.path.
    import importlib.util
    if importlib.util.find_spec("hailo_platform") is None:
        sys.path.insert(0, "/usr/lib/python3/dist-packages")

    from hailo_platform import (  # type: ignore[import]
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        HEF,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    hef          = HEF(args.hef)
    tracker      = make_tracker(args.tracker, args.max_age)
    preprocessor = Preprocessor(modality=modality)
    all_results: list[SessionResult] = []

    with VDevice() as target:
        configure_params     = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group        = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_info = hef.get_input_vstream_infos()[0]
        logging.info("Hailo input : %s  shape=%s", input_info.name, input_info.shape)

        input_params  = InputVStreamParams.make_from_network_group(network_group, quantized=True,  format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

        # Probe crop size from the first available RGB frame.
        first_stem = sessions[0][1][0]
        first_img  = cv2.imread(str(data_root / "rgb" / f"{first_stem}.jpg"))
        if first_img is None:
            raise SystemExit(f"Cannot read first frame: {data_root}/rgb/{first_stem}.jpg")
        crop_h, crop_w = first_img.shape[:2]
        logging.info("Crop size: %dx%d", crop_w, crop_h)

        with network_group.activate(network_group_params):
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                for video_name, stems in sessions:
                    logging.info("Session %s — %d frames", video_name, len(stems))
                    video_path = Path(args.output_dir) / f"{video_name}.mp4" if args.video else None
                    res = eval_session(
                        stems, data_root, tracker, pipeline,
                        input_info, preprocessor, args.conf, crop_w, crop_h, modality,
                        video_path=video_path,
                    )
                    all_results.append(res)
                    tracker.reset()

    logging.info("Computing metrics...")
    metrics, combined = aggregate(all_results)
    print_results(args.tracker, metrics)
    out_path = Path(args.output_dir) / "latency_distributions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_latencies(combined, args.tracker, str(out_path))


if __name__ == "__main__":
    main()
