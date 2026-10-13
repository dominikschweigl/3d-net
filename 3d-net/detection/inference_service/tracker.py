"""
Adapter between the inference service detection model and the ``trackers``
library (Roboflow).  Wraps the selected tracker behind a single class that
converts between our ``List[Detection]`` and ``sv.Detections``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker

from .config import TrackerConfig, TrackerMode
from .models import BBox, Detection

logger = logging.getLogger(__name__)


class Tracker:
    """Unified wrapper around SORT / ByteTrack / OC-SORT."""

    def __init__(self, config: TrackerConfig, fps: float, class_names: List[str]) -> None:
        self._mode = config.mode
        self._class_names = class_names
        self._inner = self._create(config, fps)

    @staticmethod
    def _create(config: TrackerConfig, fps: float):
        mode = config.mode
        if mode == TrackerMode.NONE:
            return None

        if mode == TrackerMode.SORT:
            cfg = config.sort
            return SORTTracker(
                lost_track_buffer=cfg.lost_track_buffer,
                frame_rate=fps,
                track_activation_threshold=cfg.track_activation_threshold,
                minimum_consecutive_frames=cfg.minimum_consecutive_frames,
                minimum_iou_threshold=cfg.minimum_iou_threshold,
            )

        if mode == TrackerMode.BYTETRACK:
            cfg = config.bytetrack
            return ByteTrackTracker(
                lost_track_buffer=cfg.lost_track_buffer,
                frame_rate=fps,
                track_activation_threshold=cfg.track_activation_threshold,
                minimum_consecutive_frames=cfg.minimum_consecutive_frames,
                minimum_iou_threshold=cfg.minimum_iou_threshold,
                high_conf_det_threshold=cfg.high_conf_det_threshold,
            )

        if mode == TrackerMode.OCSORT:
            cfg = config.ocsort
            return OCSORTTracker(
                lost_track_buffer=cfg.lost_track_buffer,
                frame_rate=fps,
                minimum_consecutive_frames=cfg.minimum_consecutive_frames,
                minimum_iou_threshold=cfg.minimum_iou_threshold,
                high_conf_det_threshold=cfg.high_conf_det_threshold,
                direction_consistency_weight=cfg.direction_consistency_weight,
                delta_t=cfg.delta_t,
            )

        raise ValueError(f"unknown tracker mode: {mode}")

    @property
    def mode(self) -> TrackerMode:
        return self._mode

    @property
    def enabled(self) -> bool:
        return self._inner is not None

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Run the tracker on one frame's detections and return copies with track IDs."""
        if self._inner is None:
            return detections

        sv_dets = self._to_sv(detections)
        sv_result = self._inner.update(sv_dets)

        tracked: List[Detection] = []
        for i in range(len(sv_result)):
            tid: Optional[int] = None
            if sv_result.tracker_id is not None:
                tid = int(sv_result.tracker_id[i])
                # The library uses -1 for unconfirmed tracks.
                if tid < 0:
                    tid = None
            cls_id = int(sv_result.class_id[i])
            name = self._class_names[cls_id] if cls_id < len(self._class_names) else f"class_{cls_id}"
            x1, y1, x2, y2 = sv_result.xyxy[i]
            tracked.append(Detection(
                name=name,
                label=cls_id,
                bbox=BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
                confidence=float(sv_result.confidence[i]),
                track_id=tid,
            ))
        return tracked

    def reset(self) -> None:
        """Clear all track state."""
        if self._inner is not None:
            self._inner.reset()

    @staticmethod
    def _to_sv(detections: List[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections.empty()
        xyxy = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
            dtype=np.float32,
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.label for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
