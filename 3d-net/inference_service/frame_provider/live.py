import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..alignment import AlignmentTransform
from ..capture import StreamCapture
from ..config import AlignmentConfig, CameraConfig, CaptureConfig
from . import FramePair, FrameProvider

logger = logging.getLogger(__name__)


class LiveFrameProvider(FrameProvider):
    """
    Reads the two WIRIS RTSP streams, checks temporal alignment, and applies
    the analytic FOV-based spatial warp before returning crop pairs.

    If capture is enabled, every visible frame triggers a recording callback that
    aligns the pair and writes to {output_dir}/{session}_vis.mp4 + _ir.mp4 at the
    camera's native fps, fully decoupled from the inference rate.

    Args:
        camera_cfg:     Camera IP, temporal offset, and native fps.
        alignment_cfg:  Manual tx/ty correction for the spatial warp.
        max_sync_gap_s: Drop frame pairs whose timestamps differ by more than
                        this many seconds (after applying temporal_offset_s).
        capture_cfg:    Optional recording config. None or disabled = no recording.
    """

    def __init__(
        self,
        camera_cfg: CameraConfig,
        alignment_cfg: AlignmentConfig,
        max_sync_gap_s: float = 0.1,
        capture_cfg: Optional[CaptureConfig] = None,
    ):
        self._camera_cfg = camera_cfg
        self._alignment_cfg = alignment_cfg
        self._max_sync_gap_s = max_sync_gap_s
        self._capture_cfg = capture_cfg

        vis_url = f"rtsp://{camera_cfg.ip}:8554/visible"
        thm_url = f"rtsp://{camera_cfg.ip}:8554/thermal"
        self._vis_cap = StreamCapture(vis_url, name="visible")
        self._thm_cap = StreamCapture(thm_url, name="thermal")
        self._alignment = AlignmentTransform(
            manual_tx=alignment_cfg.tx,
            manual_ty=alignment_cfg.ty,
        )

        # ── Recording state ───────────────────────────────────────────────────
        self._rec_vis_writer: Optional[cv2.VideoWriter] = None
        self._rec_ir_writer: Optional[cv2.VideoWriter] = None
        self._rec_lock = threading.Lock()
        self._rec_frame_count = 0
        self._rec_vis_path: Optional[Path] = None
        self._rec_ir_path: Optional[Path] = None
        self._rec_fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._vis_cap.start()
        self._thm_cap.start()
        if self._capture_cfg and self._capture_cfg.enabled:
            self._start_recording()

    def stop(self) -> None:
        self._vis_cap.stop()
        self._thm_cap.stop()
        self._stop_recording()

    # ── FrameProvider interface ───────────────────────────────────────────────

    @property
    def crop_size(self) -> Tuple[int, int]:
        return self._alignment.crop_w, self._alignment.crop_h

    def describe(self) -> str:
        ip = self._camera_cfg.ip
        rec = ""
        if self._capture_cfg and self._capture_cfg.enabled:
            rec = f"\nRecording: {self._capture_cfg.output_dir}  ({self._camera_cfg.fps:.0f} fps)"
        return (
            f"Camera  : {ip}\n"
            f"  Visible : rtsp://{ip}:8554/visible\n"
            f"  Thermal : rtsp://{ip}:8554/thermal\n"
            f"  t-offset: {self._camera_cfg.temporal_offset_s:+.3f}s"
            f"  sync-gap: {self._max_sync_gap_s*1000:.0f}ms\n"
            f"Alignment: tx={self._alignment._M[0,2]:+.1f}px"
            f"  ty={self._alignment._M[1,2]:+.1f}px\n"
            f"  Crop    : {self._alignment.crop_w}×{self._alignment.crop_h}"
            f"{rec}"
        )

    def next_pair(self) -> Optional[FramePair]:
        vis_pair = self._vis_cap.latest
        if vis_pair is None:
            logger.debug("Streams not ready: vis=--")
            return None

        ts_vis, vis_frame = vis_pair
        ts_vis_adj = ts_vis + self._camera_cfg.temporal_offset_s
        thm_pair = self._thm_cap.closest_to(ts_vis_adj)

        if thm_pair is None:
            logger.debug("Streams not ready: thm=--")
            return None

        ts_thm, thm_frame = thm_pair
        drift = abs(ts_vis_adj - ts_thm)

        if drift > self._max_sync_gap_s:
            logger.debug(
                "Temporal sync gap %.1fms > %.1fms — skipping",
                drift * 1000, self._max_sync_gap_s * 1000,
            )
            return None

        t_align = time.monotonic()
        rgb_crop, gray_crop = self._alignment.process(vis_frame, thm_frame)
        align_s = time.monotonic() - t_align

        return FramePair(rgb_crop=rgb_crop, gray_crop=gray_crop, drift_s=drift, align_s=align_s)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        session = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self._capture_cfg.output_dir) / session
        out_dir.mkdir(parents=True, exist_ok=True)
        self._rec_vis_path = out_dir / "vis.mp4"
        self._rec_ir_path = out_dir / "ir.mp4"
        self._vis_cap._frame_callback = self._on_vis_frame
        logger.info(
            "Recording: %s + %s  (%.1f fps)",
            self._rec_vis_path, self._rec_ir_path, self._camera_cfg.fps,
        )

    def _on_vis_frame(self, ts_vis: float, vis_frame: np.ndarray) -> None:
        ts_vis_adj = ts_vis + self._camera_cfg.temporal_offset_s
        ir_pair = self._thm_cap.closest_to(ts_vis_adj)
        if ir_pair is None:
            return

        _, ir_frame = ir_pair
        rgb_crop, gray_crop = self._alignment.process(vis_frame, ir_frame)
        ir_bgr = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

        with self._rec_lock:
            if self._rec_vis_writer is None:
                h, w = rgb_crop.shape[:2]
                fps = self._camera_cfg.fps
                # AlignmentTransform.process returns RGB — VideoWriter expects BGR.
                self._rec_vis_writer = cv2.VideoWriter(
                    str(self._rec_vis_path), self._rec_fourcc, fps, (w, h)
                )
                self._rec_ir_writer = cv2.VideoWriter(
                    str(self._rec_ir_path), self._rec_fourcc, fps, (w, h)
                )
            self._rec_vis_writer.write(cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
            self._rec_ir_writer.write(ir_bgr)
            self._rec_frame_count += 1

    def _stop_recording(self) -> None:
        with self._rec_lock:
            if self._rec_vis_writer:
                self._rec_vis_writer.release()
                self._rec_vis_writer = None
            if self._rec_ir_writer:
                self._rec_ir_writer.release()
                self._rec_ir_writer = None
        if self._rec_frame_count:
            logger.info("Recording closed after %d frames", self._rec_frame_count)
