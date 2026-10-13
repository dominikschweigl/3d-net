"""
FrameProvider ABC and FramePair / PreparedFrame dataclasses.

Concrete implementations live in sub-modules:
  live        — LiveFrameProvider (WIRIS RTSP streams, temporal sync, spatial warp)
  m3fd        — M3fdFrameProvider (replays M3FD dataset pairs at a configurable fps)
  three_d_net — ThreeDNetFrameProvider (replays 3D-Net dataset pairs at a configurable fps)
"""

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class FramePair:
    rgb_crop:  np.ndarray   # (H, W, 3) uint8 — RGB channels from the visible camera
    gray_crop: np.ndarray   # (H, W)    uint8 — single thermal intensity channel
    drift_s:   float = 0.0  # temporal gap between the two source frames (for monitoring)
    align_s:   float = 0.0  # time spent on spatial warp + crop inside the provider
    frame_stem: Optional[str] = None            # e.g. "hausern_20_01_26_aufnahme5_000042"
    annotation_path: Optional[Path] = None      # path to labels/<stem>.xml if it exists


@dataclass
class PreparedFrame:
    seq:      int           # monotonically increasing — used by providers for ordering
    pair:     FramePair     # original crops (for recorder, monitor, status line)
    tensor:   np.ndarray    # shape (1, 640, 640, 4), BGRT uint8, ready for Hailo
    lb_scale: float
    lb_pad_x: int
    lb_pad_y: int
    t_align:  float         # seconds spent in load/warp step
    t_pre:    float         # seconds spent in BGRT + letterbox step


class FrameProvider(abc.ABC):
    """
    Supplies temporally-aligned RGBT frame pairs ready for fusion and inference.

    Lifecycle:
      configure_pipeline(n_workers)  — call once before start()
      start()                        — begin producing frames
      next_prepared()                — poll in the inference loop
      stop()                         — shut down on exit
    """

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def next_pair(self) -> Optional[FramePair]:
        """
        Return the next aligned frame pair, or None if one is not yet available
        (streams not ready) or should be skipped (temporal sync violation).
        Callers should loop and retry on None rather than treating it as an error.
        """
        ...

    @property
    @abc.abstractmethod
    def crop_size(self) -> Tuple[int, int]:
        """(width, height) of the crops returned by next_pair()."""
        ...

    @abc.abstractmethod
    def describe(self) -> str:
        """Human-readable description for the startup banner."""
        ...

    @property
    def exhausted(self) -> bool:
        """True once a non-looping provider has yielded its last frame."""
        return False

    @property
    def labels_dir(self) -> Optional[Path]:
        """Directory containing Pascal VOC XML annotation files, or None if unavailable."""
        return None

    # ── Pipeline interface ────────────────────────────────────────────────────

    def configure_pipeline(self, n_workers: int, modality: "ModalityMode" = None) -> None:
        """
        Call once before start() to enable concurrent pre-processing.
        Creates a Preprocessor and stores the worker count for concrete providers.
        """
        # Lazy import: preprocessor imports FramePair from this module, so importing
        # Preprocessor at module level here would create a circular dependency.
        from ..config import ModalityMode  # noqa: PLC0415
        from ..preprocessor import Preprocessor  # noqa: PLC0415
        if modality is None:
            modality = ModalityMode.RGBT
        self._n_pipeline_workers = n_workers
        self._preprocessor = Preprocessor(modality=modality)
        self._prepared_seq = 0

    def next_prepared(self) -> Optional[PreparedFrame]:
        """
        Return the next BGRT+letterboxed frame ready for Hailo inference, or None.

        Default implementation: sequential (calls next_pair then preprocesses inline).
        Used as-is by LiveFrameProvider; file providers override with a concurrent queue.
        """
        pair = self.next_pair()
        if pair is None:
            return None
        self._prepared_seq += 1
        return self._preprocessor.prepare(self._prepared_seq, pair)
