from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


class ProviderMode(str, Enum):
    LIVE = "live"
    M3FD = "m3fd"
    THREE_D_NET = "3d_net"
    RTDOD = "rtdod"


class CameraConfig(BaseModel):
    ip: str = "10.0.0.230"
    # Fixed temporal offset between visible and thermal streams (seconds).
    # Positive = thermal is captured earlier than visible.
    # Measure with wiris_camera/camera_alignment/temporal_offset_estimation.py.
    temporal_offset_s: float = 0.0
    fps: float = 20.0  # native WIRIS camera frame rate


class AlignmentConfig(BaseModel):
    # Manual pixel corrections applied after the analytic FOV-based centering.
    # Tune these by visually inspecting the overlay (positive tx = thermal moves right).
    tx: float = -10.0
    ty: float = -30.0


class ModalityMode(str, Enum):
    RGB     = "rgb"      # visible only  — 3-channel RGB input
    THERMAL = "thermal"  # IR only       — grayscale replicated to 3 channels
    RGBT    = "rgbt"     # fused         — 4-channel RGB + thermal input


class ModelConfig(BaseModel):
    hef_path: str = "model.hef"
    conf_threshold: float = 0.2
    modality: ModalityMode = ModalityMode.RGBT
    # Canonical order — must match the training dataset class indices.
    # Source of truth: detection/fiftyone/merge_fiftyone_datapool.py POOL_CLASSES.
    class_names: List[str] = ["person", "car", "motorcycle", "bicycle", "bus", "truck"]


class ZmqConfig(BaseModel):
    endpoint: str = "tcp://*:5556"


class ServiceConfig(BaseModel):
    # Maximum inference rate. Actual rate is limited by camera FPS and Hailo throughput.
    max_fps: Optional[float] = 10.0
    # Frames from both cameras whose timestamps differ by more than this are dropped.
    max_sync_gap_s: Optional[float] = None  # defaults to 2 / max_fps
    # Number of concurrent worker threads inside each file-based frame provider.
    # Each worker loads and preprocesses one frame (imread + BGRT + letterbox) in parallel.
    # Increase toward 4 to approach Hailo's theoretical throughput limit.
    pipeline_workers: int = 4
    # Skip model inference entirely — only record raw streams via capture config.
    # Requires provider.mode = live and capture.enabled = true.
    record_only: bool = False


class DatasetConfig(BaseModel):
    """Shared config for all dataset-replay providers (M3FD, RTDOD, 3D-Net, …)."""
    root: str = "detection/data/m3fd"
    splits: List[str] = ["train"]
    # Video name filter for providers that organise frames by video (e.g. 3d_net).
    # None = include all videos found in the dataset.
    videos: Optional[List[str]] = None
    fps: Optional[float] = 10.0
    loop: bool = True
    # Stop after processing this many frames. None = run through the full dataset.
    max_frames: Optional[int] = None


class ProviderConfig(BaseModel):
    mode: ProviderMode = ProviderMode.LIVE
    dataset: DatasetConfig = DatasetConfig()


class TrackerMode(str, Enum):
    NONE = "none"
    SORT = "sort"
    BYTETRACK = "bytetrack"
    OCSORT = "ocsort"


class SORTConfig(BaseModel):
    lost_track_buffer: int = 30
    track_activation_threshold: float = 0.25
    minimum_consecutive_frames: int = 3
    minimum_iou_threshold: float = 0.3


class ByteTrackConfig(BaseModel):
    lost_track_buffer: int = 30
    track_activation_threshold: float = 0.7
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold: float = 0.1
    high_conf_det_threshold: float = 0.6


class OCSORTConfig(BaseModel):
    lost_track_buffer: int = 30
    minimum_consecutive_frames: int = 3
    minimum_iou_threshold: float = 0.3
    high_conf_det_threshold: float = 0.6
    direction_consistency_weight: float = 0.2
    delta_t: int = 3


class TrackerConfig(BaseModel):
    mode: TrackerMode = TrackerMode.BYTETRACK
    sort: SORTConfig = SORTConfig()
    bytetrack: ByteTrackConfig = ByteTrackConfig()
    ocsort: OCSORTConfig = OCSORTConfig()


class CaptureLayout(str, Enum):
    RGB = "rgb"
    IR = "ir"
    SIDE_BY_SIDE = "side_by_side"
    DUAL = "dual"


class CaptureConfig(BaseModel):
    """Live camera recording — handled inside LiveFrameProvider."""
    enabled: bool = False
    output_dir: str = "/app/recordings"  # must match the recordings volume in docker-compose.yaml


class InferenceRecorderConfig(BaseModel):
    """Inference-rate video recording for dataset replay providers (M3FD, RTDOD, 3D-Net)."""
    enabled: bool = False
    output_dir: str = "/app/recordings"  # must match the recordings volume in docker-compose.yaml
    layout: CaptureLayout = CaptureLayout.SIDE_BY_SIDE
    video_fps: Optional[float] = None  # None → inherits service.max_fps
    draw_detections: bool = True


class MonitoringConfig(BaseModel):
    # Number of frames in the rolling window for avg/max latency stats (stdout only).
    window_frames: int = 100
    # How often to print rolling aggregate stats to stdout.
    report_interval_s: float = 10.0
    # Directory for CSV log files (inference_stats.csv, inference_detections.csv). null = disabled.
    log_dir: Optional[str] = None
    # Refresh Hailo chip temperature/power and system CPU/RAM every N frames.
    # Keeps per-frame query overhead negligible (cached value used in between).
    hw_poll_every_frames: int = 30
    # Print per-frame status line (timing breakdown + detection list) to stdout.
    log_status_line: bool = True


class EvaluationConfig(BaseModel):
    enabled: bool = False
    iou_threshold: float = 0.5
    # If True, also compute mAP averaged over IoU thresholds 0.50:0.05:0.95 (COCO-style).
    iou_range: bool = True


class Config(BaseModel):
    provider: ProviderConfig = ProviderConfig()
    camera: CameraConfig = CameraConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    model: ModelConfig = ModelConfig()
    zmq: ZmqConfig = ZmqConfig()
    service: ServiceConfig = ServiceConfig()
    tracker: TrackerConfig = TrackerConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    capture: CaptureConfig = CaptureConfig()
    inference_recorder: InferenceRecorderConfig = InferenceRecorderConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def _apply_runtime_overrides(cfg: Config) -> Config:
    """Normalize runtime-only config values in one place."""
    dataset_cfg = cfg.provider.dataset
    service_cfg = cfg.service

    # Evaluation runs should process as fast as possible and never loop forever.
    if cfg.evaluation.enabled:
        service_cfg.max_fps = None
        dataset_cfg.fps = None
        dataset_cfg.loop = False

    return cfg


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML. Falls back to all defaults if no file provided."""
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return _apply_runtime_overrides(Config(**data))
    return _apply_runtime_overrides(Config())
