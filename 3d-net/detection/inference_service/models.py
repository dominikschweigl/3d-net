from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    name: str
    label: int
    bbox: BBox
    confidence: float
    track_id: Optional[int] = None


class FrameResult(BaseModel):
    """Published on the ZMQ 'detections' topic for each processed frame."""
    timestamp: datetime
    # Format: {session}_{seq:07d}, e.g. "20260414_114357_0000042".
    # Session is the UTC start time (YYYYMMDD_HHMMSS), seq is the 1-based frame counter.
    frame_id: str
    detections: List[Detection]
    # Original filename stem for dataset-replay providers (e.g. "hausern_20_01_26_000042").
    # None when running in live camera mode.
    source_file: Optional[str] = None
