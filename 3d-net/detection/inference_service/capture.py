import logging
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamCapture:
    """
    Daemon thread that continuously drains an RTSP stream.

    OpenCV's RTSP buffer accumulates frames if you don't read fast enough, causing
    stale-frame lag. This class dedicates a thread to grab() in a tight loop so the
    main inference loop always gets the freshest available frame via .latest.

    An optional frame_callback is called with every successfully decoded frame,
    allowing decoupled consumers (e.g. a video recorder) to receive every frame
    independent of the inference polling rate.
    """

    def __init__(
        self,
        url: str,
        name: str = "stream",
        frame_callback: Optional[Callable[[float, np.ndarray], None]] = None,
    ):
        self._url = url
        self._name = name
        self._frame_callback = frame_callback
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # (monotonic_timestamp, frame) — None until first successful grab
        self._latest: Optional[Tuple[float, np.ndarray]] = None
        # Rolling buffer for temporal matching (covers ~1s at 25fps)
        self._buffer: deque = deque(maxlen=30)

    @property
    def latest(self) -> Optional[Tuple[float, np.ndarray]]:
        """Thread-safe access to the most recently decoded frame and its grab timestamp."""
        with self._lock:
            return self._latest

    def closest_to(self, ts: float) -> Optional[Tuple[float, np.ndarray]]:
        """Return the buffered (timestamp, frame) whose timestamp is nearest to ts."""
        with self._lock:
            if not self._buffer:
                return None
            return min(self._buffer, key=lambda x: abs(x[0] - ts))

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._grab_loop,
            daemon=True,
            name=f"capture-{self._name}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _grab_loop(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self._url)
            if not cap.isOpened():
                logger.warning(
                    "[%s] Cannot open %s — retrying in %.1fs", self._name, self._url, backoff
                )
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, 10.0)
                continue

            backoff = 1.0
            logger.info("[%s] Connected to %s", self._name, self._url)

            while not self._stop_event.is_set():
                # grab() is cheaper than read() — it fetches without decoding.
                ok = cap.grab()
                if not ok:
                    logger.warning("[%s] grab() failed — reconnecting", self._name)
                    break
                ts = time.monotonic()
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    with self._lock:
                        self._latest = (ts, frame)
                        self._buffer.append((ts, frame))
                    if self._frame_callback is not None:
                        self._frame_callback(ts, frame)

            cap.release()
