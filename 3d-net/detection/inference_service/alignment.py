"""
Spatial alignment of WIRIS thermal frames into visible camera space.

Derived from wiris_camera/camera_alignment/spatial_overlay.py — same analytic
FOV-based approach, stripped of video I/O so it can be used per-frame in the
live inference loop.

Camera constants (fixed WIRIS Enterprise hardware):
  Visible:  1280×720,  HFOV=63.7°, VFOV=38.6°
  Thermal:  640×512,   HFOV=45.0°, VFOV=37.0°

The transform maps thermal pixel coordinates → visible canvas pixel coordinates
via an affine warp (scale + translate). The overlap ROI is the rectangle in the
visible frame where both cameras have valid pixels after warping.
"""

import logging
import math
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AlignmentTransform:
    """
    Pre-computed affine warp from thermal space into visible space.

    Instantiate once at startup; call process() for every frame pair.
    The returned crops are pixel-registered and ready for RGBT channel fusion.
    """

    # ── WIRIS Enterprise camera constants ────────────────────────────────────────
    _VIS_W, _VIS_H = 1280, 720
    _VIS_HFOV = 63.7
    _VIS_VFOV = 38.6

    _THM_W, _THM_H = 640, 512
    _THM_HFOV = 45.0
    _THM_VFOV = 37.0

    def __init__(self, manual_tx: float = -10.0, manual_ty: float = -30.0):
        self._M = self._build_affine(manual_tx, manual_ty)
        self._roi = self._overlap_roi(self._M)
        x0, y0, x1, y1 = self._roi
        self.crop_w = x1 - x0
        self.crop_h = y1 - y0
        logger.info(
            "Alignment: scale=(%.4f, %.4f) offset=(%.1f, %.1f) "
            "overlap=(%d,%d)–(%d,%d) crop=%dx%d",
            self._M[0, 0], self._M[1, 1],
            self._M[0, 2], self._M[1, 2],
            *self._roi, self.crop_w, self.crop_h,
        )

    def process(
        self,
        vis_frame: np.ndarray,
        thm_frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp thermal into visible space and return the overlap crop of both streams.

        Args:
            vis_frame: BGR frame from the visible RTSP stream (1280x720).
            thm_frame: BGR (pseudo-colour) or grayscale frame from the thermal RTSP stream (640x512).

        Returns:
            rgb_crop:  (crop_h, crop_w, 3) uint8 — RGB channels from visible camera.
            gray_crop: (crop_h, crop_w)    uint8 — single thermal channel, intensity only.
        """
        x0, y0, x1, y1 = self._roi

        # Warp thermal into visible canvas then crop to overlap region.
        thm_warped = cv2.warpAffine(thm_frame, self._M, (self._VIS_W, self._VIS_H))
        thm_crop = thm_warped[y0:y1, x0:x1]

        # Reduce to a single intensity channel regardless of whether the thermal
        # stream outputs pseudo-colour (3ch) or true grayscale (1ch).
        if thm_crop.ndim == 3:
            # TODO: check if all channels are identical before converting to grayscale and just return first channel if so, to avoid unnecessary colour conversion
            gray_crop = cv2.cvtColor(thm_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = thm_crop

        # Crop visible and convert BGR → RGB (Ultralytics trains with RGB).
        vis_crop = vis_frame[y0:y1, x0:x1]
        rgb_crop = cv2.cvtColor(vis_crop, cv2.COLOR_BGR2RGB)

        return rgb_crop, gray_crop
    
    def _build_affine(self, manual_tx: float, manual_ty: float) -> np.ndarray:
        """Return 2x3 affine matrix: thermal pixel → visible canvas pixel."""
        sx = math.tan(math.radians(self._THM_HFOV / 2)) / math.tan(math.radians(self._VIS_HFOV / 2))
        sy = math.tan(math.radians(self._THM_VFOV / 2)) / math.tan(math.radians(self._VIS_VFOV / 2))
        scale_x = (self._VIS_W * sx) / self._THM_W
        scale_y = (self._VIS_H * sy) / self._THM_H
        tx = (self._VIS_W * (1 - sx)) / 2 + manual_tx
        ty = (self._VIS_H * (1 - sy)) / 2 + manual_ty
        return np.array([[scale_x, 0.0, tx], [0.0, scale_y, ty]], dtype=np.float64)


    def _overlap_roi(self, M: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Return (x0, y0, x1, y1) in visible-frame coordinates: the rectangle where
        both the visible frame and the warped thermal have valid pixels.
        """
        tx, ty = M[0, 2], M[1, 2]
        sx, sy = M[0, 0], M[1, 1]
        x0 = int(math.ceil(max(0.0, tx)))
        y0 = int(math.ceil(max(0.0, ty)))
        x1 = int(math.floor(min(float(self._VIS_W), tx + sx * self._THM_W)))
        y1 = int(math.floor(min(float(self._VIS_H), ty + sy * self._THM_H)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"No overlap region: ({x0},{y0})–({x1},{y1}). Adjust tx/ty in config."
            )
        return x0, y0, x1, y1
