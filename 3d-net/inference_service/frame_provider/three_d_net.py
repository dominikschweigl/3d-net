import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from . import FramePair, FrameProvider, PreparedFrame

logger = logging.getLogger(__name__)

# Suffix pattern: underscore followed by exactly 6 digits (frame number).
_FRAME_SUFFIX_LEN = 7  # "_NNNNNN"


def _video_name(stem: str) -> str:
    """Extract the video name from a frame stem like 'myvideo_000042' → 'myvideo'."""
    return stem[:-_FRAME_SUFFIX_LEN]


class ThreeDNetFrameProvider(FrameProvider):
    """
    Replays paired visible/IR video frames from the 3D-Net dataset as FramePairs.

    Dataset layout:
        <root>/rgb/<video>_<frame:06d>.jpg  — visible image (BGR)
        <root>/ir/<video>_<frame:06d>.jpg   — infrared image (BGR pseudo-colour)
        <root>/{split}.txt                  — optional per-split frame lists

    The video name is the filename prefix before the last '_NNNNNN' suffix.
    Frames with identical names in rgb/ and ir/ are treated as a matched pair.

    Args:
        root:    Path to the 3D-Net dataset root directory.
        splits:  Split name(s) to load, e.g. ["val"]. Reads <root>/<split>.txt,
                 which must contain one frame filename per line. None = all frames.
        videos:  Video name(s) to include, e.g. ["hausern_20_01_26_aufnahme5"].
                 Applied after splits; None (default) includes all videos.
        fps:     Rate at which next_pair() yields frames. If None, provider-side
                 FPS throttling is disabled.
        loop:    If True, restart from the beginning after the last frame.
    """

    def __init__(
        self,
        root: Path,
        splits: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        fps: Optional[float] = 10.0,
        loop: bool = True,
        max_frames: Optional[int] = None,
    ):
        root = Path(root)
        self._rgb_dir = root / "rgb"
        self._ir_dir = root / "ir"
        self._labels_dir = root / "labels"
        self._fps = fps
        self._loop = loop
        self._max_frames = max_frames
        self._interval = 1.0 / fps if fps and fps > 0 else 0.0

        available = self._discover_videos()
        if not available:
            raise FileNotFoundError(
                f"No video frames found under {self._rgb_dir} / {self._ir_dir}"
            )

        if splits is not None:
            split_filenames: set[str] = set()
            for split in splits:
                split_file = root / f"{split}.txt"
                if not split_file.exists():
                    raise FileNotFoundError(f"Split file not found: {split_file}")
                split_filenames.update(split_file.read_text().splitlines())
            available = {
                video: [f for f in frames if f in split_filenames]
                for video, frames in available.items()
            }
            available = {v: frames for v, frames in available.items() if frames}

        if videos is not None:
            unknown = set(videos) - available.keys()
            if unknown:
                raise ValueError(
                    f"Unknown video(s): {sorted(unknown)}. "
                    f"Available: {sorted(available.keys())}"
                )
            selected = {v: available[v] for v in videos}
        else:
            selected = available

        # Flatten: all frames across selected videos, sorted by video then frame order.
        self._filenames: List[str] = [
            name
            for video in sorted(selected.keys())
            for name in selected[video]
        ]
        self._videos = sorted(selected.keys())

        # Determine crop size from the first available frame.
        self._crop_w, self._crop_h = self._probe_size(self._filenames[0])

        self._index = 0
        self._last_yield_t: float = 0.0
        self._exhausted = False

        # Threading state — populated by configure_pipeline() + start()
        self._global_seq: int = 0   # monotonically increasing across loops
        self._index_lock = threading.Lock()
        self._prepared_q: Optional[queue.Queue] = None
        self._worker_stop = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        self._active_workers: int = 0
        self._active_lock = threading.Lock()
        self._reorder_buf: dict = {}
        self._next_expected_seq: int = 1

        fps_text = f"{fps:.1f}" if self._interval > 0 else "unlimited"

        logger.info(
            "ThreeDNetFrameProvider: %d frames from video(s) %s at %s fps (loop=%s, max_frames=%s)",
            len(self._filenames), self._videos, fps_text, loop, max_frames,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._index = 0
        self._last_yield_t = 0.0
        self._exhausted = False
        self._global_seq = 0
        self._reorder_buf.clear()
        self._next_expected_seq = 1
        self._worker_threads.clear()

        n = getattr(self, "_n_pipeline_workers", 1)
        self._prepared_q = queue.Queue(maxsize=n * 4)
        self._worker_stop.clear()
        self._active_workers = n

        for i in range(n):
            t = threading.Thread(
                target=self._file_worker,
                daemon=True,
                name=f"3dnet-worker-{i}",
            )
            self._worker_threads.append(t)
            t.start()

    def stop(self) -> None:
        self._worker_stop.set()
        for t in self._worker_threads:
            t.join(timeout=3.0)
        self._worker_threads.clear()

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    # ── FrameProvider interface ───────────────────────────────────────────────

    @property
    def labels_dir(self):
        return self._labels_dir

    @property
    def crop_size(self) -> Tuple[int, int]:
        return self._crop_w, self._crop_h

    def describe(self) -> str:
        fps_text = f"{self._fps}" if self._interval > 0 else "unlimited"
        return (
            f"Provider : ThreeDNetFrameProvider\n"
            f"  Videos  : {', '.join(self._videos)}\n"
            f"  Frames  : {len(self._filenames)}\n"
            f"  FPS     : {fps_text}\n"
            f"  Loop    : {self._loop}"
        )

    def next_pair(self) -> Optional[FramePair]:
        if self._exhausted:
            return None

        now = time.monotonic()
        if self._interval > 0 and now - self._last_yield_t < self._interval:
            return None

        filename = self._filenames[self._index]
        t_load = time.monotonic()

        rgb_bgr = cv2.imread(str(self._rgb_dir / filename))
        ir_bgr  = cv2.imread(str(self._ir_dir  / filename))

        if rgb_bgr is None or ir_bgr is None:
            logger.warning("Failed to load %s — skipping", filename)
            self._advance()
            return None

        rgb_crop  = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        gray_crop = cv2.cvtColor(ir_bgr,  cv2.COLOR_BGR2GRAY)
        align_s   = time.monotonic() - t_load

        stem = Path(filename).stem
        label_path = self._labels_dir / f"{stem}.xml"

        self._last_yield_t = time.monotonic()
        self._advance()

        return FramePair(
            rgb_crop=rgb_crop,
            gray_crop=gray_crop,
            align_s=align_s,
            frame_stem=stem,
            annotation_path=label_path if label_path.exists() else None,
        )

    def next_prepared(self) -> Optional[PreparedFrame]:
        # Serve from reorder buffer first (covers out-of-order worker completions).
        if self._next_expected_seq in self._reorder_buf:
            item = self._reorder_buf.pop(self._next_expected_seq)
            self._next_expected_seq += 1
            return item

        try:
            item = self._prepared_q.get(timeout=0.05)
        except queue.Empty:
            return None

        if item is None:
            # Sentinel: all workers finished. Callers check provider.exhausted.
            return None

        if item.seq == self._next_expected_seq:
            self._next_expected_seq += 1
            return item

        # Out-of-order arrival: buffer it and let the caller retry.
        self._reorder_buf[item.seq] = item
        return None

    # ── Internal worker ───────────────────────────────────────────────────────

    def _next_file_item(self) -> Optional[tuple]:
        """Thread-safe: atomically claim the next filename without loading it."""
        with self._index_lock:
            if self._exhausted:
                return None
            now = time.monotonic()
            if self._interval > 0 and now - self._last_yield_t < self._interval:
                return None
            filename = self._filenames[self._index]
            self._global_seq += 1
            seq = self._global_seq      # monotonically increasing; safe across loops
            self._last_yield_t = time.monotonic()
            self._advance()

        stem = Path(filename).stem
        label_path = self._labels_dir / f"{stem}.xml"
        return (
            seq,
            self._rgb_dir / filename,
            self._ir_dir  / filename,
            stem,
            label_path if label_path.exists() else None,
        )

    def _file_worker(self) -> None:
        try:
            while not self._worker_stop.is_set():
                item = self._next_file_item()
                if item is None:
                    if self._exhausted:
                        break
                    time.sleep(0.001)
                    continue

                seq, rgb_path, ir_path, stem, label_path = item

                t0 = time.monotonic()
                rgb_bgr = cv2.imread(str(rgb_path))
                ir_bgr  = cv2.imread(str(ir_path))
                if rgb_bgr is None or ir_bgr is None:
                    logger.warning("Failed to load %s — skipping", rgb_path.name)
                    continue
                rgb_crop  = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                gray_crop = cv2.cvtColor(ir_bgr,  cv2.COLOR_BGR2GRAY)
                t_align = time.monotonic() - t0

                pair = FramePair(
                    rgb_crop=rgb_crop,
                    gray_crop=gray_crop,
                    align_s=t_align,
                    frame_stem=stem,
                    annotation_path=label_path,
                )
                self._prepared_q.put(self._preprocessor.prepare(seq, pair))

        finally:
            with self._active_lock:
                self._active_workers -= 1
                if self._active_workers == 0:
                    self._prepared_q.put(None)  # sentinel: pipeline exhausted

    # ── Internals ─────────────────────────────────────────────────────────────

    def _advance(self) -> None:
        self._index += 1
        if self._max_frames is not None and self._index >= self._max_frames:
            self._exhausted = True
            logger.info("ThreeDNetFrameProvider: max_frames=%d reached", self._max_frames)
        elif self._index >= len(self._filenames):
            if self._loop:
                logger.info("ThreeDNetFrameProvider: restarting from the beginning")
                self._index = 0
            else:
                self._exhausted = True
                logger.info("ThreeDNetFrameProvider: all frames exhausted")

    def _discover_videos(self) -> Dict[str, List[str]]:
        """
        Scan rgb/ for frame files and group them by video name.

        Returns a dict mapping video_name → sorted list of filenames (basename only)
        where a matching ir/ file also exists.
        """
        videos: Dict[str, List[str]] = {}
        if not self._rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self._rgb_dir}")
        if not self._ir_dir.exists():
            raise FileNotFoundError(f"IR directory not found: {self._ir_dir}")

        _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for rgb_path in sorted(self._rgb_dir.iterdir()):
            if not rgb_path.is_file() or rgb_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            ir_path = self._ir_dir / rgb_path.name
            if not ir_path.exists():
                logger.warning("No matching IR frame for %s — skipping", rgb_path.name)
                continue
            stem = rgb_path.stem
            if len(stem) <= _FRAME_SUFFIX_LEN:
                logger.warning("Unexpected filename (too short to parse): %s", rgb_path.name)
                continue
            video = _video_name(stem)
            videos.setdefault(video, []).append(rgb_path.name)

        # Each video's frames arrive sorted because we sorted rgb_dir above.
        return videos

    def _probe_size(self, filename: str) -> Tuple[int, int]:
        """Read the first frame to determine (width, height)."""
        path = self._rgb_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read probe image: {path}")
        h, w = img.shape[:2]
        return w, h
