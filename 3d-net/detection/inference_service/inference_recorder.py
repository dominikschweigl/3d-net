import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import CaptureLayout, InferenceRecorderConfig
from .frame_provider import FramePair
from .models import Detection

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS = 1
_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 255, 0)


class InferenceRecorder:
    def __init__(self, cfg: InferenceRecorderConfig, session: str, crop_w: int, crop_h: int, fps: float) -> None:
        self._cfg = cfg
        self._frame_count = 0
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_ir: Optional[cv2.VideoWriter] = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = Path(cfg.output_dir) / session
        out_dir.mkdir(parents=True, exist_ok=True)

        if cfg.layout == CaptureLayout.DUAL:
            rgb_path = out_dir / "rgb.mp4"
            ir_path = out_dir / "ir.mp4"
            self._writer = cv2.VideoWriter(str(rgb_path), fourcc, fps, (crop_w, crop_h))
            self._writer_ir = cv2.VideoWriter(str(ir_path), fourcc, fps, (crop_w, crop_h))
            logger.info("InferenceRecorder: %s + %s  (%.1f fps)", rgb_path, ir_path, fps)
        else:
            if cfg.layout == CaptureLayout.SIDE_BY_SIDE:
                size = (crop_w * 2, crop_h)
            else:
                size = (crop_w, crop_h)
            out_path = out_dir / "capture.mp4"
            self._writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
            logger.info("InferenceRecorder: %s  (%.1f fps)", out_path, fps)

    def write(self, pair: FramePair, detections: List[Detection]) -> None:
        rgb = pair.rgb_crop
        ir_bgr = cv2.cvtColor(pair.gray_crop, cv2.COLOR_GRAY2BGR)

        if self._cfg.draw_detections:
            rgb = self._draw_detections(rgb, detections)
            ir_bgr = self._draw_detections(ir_bgr, detections)

        # VideoWriter expects BGR; rgb_crop is RGB.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        layout = self._cfg.layout
        if layout == CaptureLayout.RGB:
            self._writer.write(bgr)
        elif layout == CaptureLayout.IR:
            self._writer.write(ir_bgr)
        elif layout == CaptureLayout.SIDE_BY_SIDE:
            self._writer.write(np.concatenate([bgr, ir_bgr], axis=1))
        elif layout == CaptureLayout.DUAL:
            self._writer.write(bgr)
            self._writer_ir.write(ir_bgr)

        self._frame_count += 1

    @staticmethod
    def _draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            b = det.bbox
            cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), _BOX_COLOR, _THICKNESS)
            label = f"{det.name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"T{det.track_id} {label}"
            cv2.putText(out, label, (b.x1, max(b.y1 - 3, 10)), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS)
        return out

    def close(self) -> None:
        if self._writer:
            self._writer.release()
        if self._writer_ir:
            self._writer_ir.release()
        logger.info("InferenceRecorder: closed after %d frames", self._frame_count)
