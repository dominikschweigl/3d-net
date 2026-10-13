"""Replay provider for the RTDOD dataset (or any VOC-format RGBT dataset).

Dataset layout (flat, paired by identical filename):
    <root>/rgb/<name>.png    — visible image
    <root>/ir/<name>.png     — infrared image
    <root>/labels/<name>.xml — Pascal VOC annotations
    <root>/train.txt         — filenames for training split (one per line)
    <root>/val.txt           — filenames for validation split
    <root>/test.txt          — filenames for test split

RGB and IR files share identical filenames so they are matched by name.
Image dimensions are probed from the first file at init time so this provider
works for any dataset that follows the layout above, regardless of resolution.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from . import FramePair, FrameProvider, PreparedFrame

logger = logging.getLogger(__name__)

_VALID_SPLITS = ("train", "val", "test")


class RtdodFrameProvider(FrameProvider):
    """
    Replays paired visible/IR images from the RTDOD dataset (or any dataset
    that follows the same VOC-format layout) as FramePairs.

    Dataset layout:
        <root>/rgb/<name>        — visible image
        <root>/ir/<name>         — infrared image (same filename)
        <root>/labels/<stem>.xml — Pascal VOC ground-truth annotations
        <root>/{train,val,test}.txt — split files, one filename per line

    Image dimensions are probed from the first RGB image so the correct
    crop_size is reported to the inference service without any hard-coding.

    Args:
        root:       Path to the dataset root directory.
        splits:     Which split(s) to load, e.g. ["val"], ["train", "val"].
        fps:        Rate at which next_pair() yields frames. None = unlimited.
        loop:       If True, restart from the beginning after the last image.
        max_frames: Stop after this many frames. None = full dataset.
    """

    def __init__(
        self,
        root: Path,
        splits: List[str] = ("val",),
        fps: Optional[float] = 10.0,
        loop: bool = True,
        max_frames: Optional[int] = None,
    ):
        root = Path(root)
        for split in splits:
            if split not in _VALID_SPLITS:
                raise ValueError(f"Unknown split {split!r}. Valid: {_VALID_SPLITS}")

        self._vis_dir   = root / "rgb"
        self._ir_dir    = root / "ir"
        self._labels_dir = root / "labels"
        self._fps        = fps
        self._loop       = loop
        self._max_frames = max_frames
        self._interval   = 1.0 / fps if fps and fps > 0 else 0.0

        self._filenames: List[str] = self._load_splits(root, splits)

        # Probe first image for actual resolution so crop_size is always correct.
        probe = cv2.imread(str(self._vis_dir / self._filenames[0]))
        if probe is None:
            raise RuntimeError(
                f"Could not read probe image: {self._vis_dir / self._filenames[0]}"
            )
        self._image_h, self._image_w = probe.shape[:2]
        del probe

        self._index = 0
        self._last_yield_t: float = 0.0
        self._exhausted = False

        # Threading state — populated by configure_pipeline() + start()
        self._global_seq: int = 0
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
            "RtdodFrameProvider: %d images from split(s) %s at %s fps "
            "(loop=%s, max_frames=%s, resolution=%dx%d)",
            len(self._filenames), splits, fps_text, loop, max_frames,
            self._image_w, self._image_h,
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
                name=f"rtdod-worker-{i}",
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
        return self._image_w, self._image_h

    def describe(self) -> str:
        fps_text = f"{self._fps}" if self._interval > 0 else "unlimited"
        return (
            f"Provider   : RtdodFrameProvider\n"
            f"  Images   : {len(self._filenames)}\n"
            f"  FPS      : {fps_text}\n"
            f"  Loop     : {self._loop}\n"
            f"  Resolution: {self._image_w}x{self._image_h}"
        )

    def next_pair(self) -> Optional[FramePair]:
        if self._exhausted:
            return None

        now = time.monotonic()
        if self._interval > 0 and now - self._last_yield_t < self._interval:
            return None

        filename = self._filenames[self._index]
        t_load = time.monotonic()

        vis_bgr = cv2.imread(str(self._vis_dir / filename))
        ir_bgr  = cv2.imread(str(self._ir_dir  / filename))

        if vis_bgr is None or ir_bgr is None:
            logger.warning("Failed to load %s — skipping", filename)
            self._advance()
            return None

        rgb_crop  = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
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
        if self._next_expected_seq in self._reorder_buf:
            item = self._reorder_buf.pop(self._next_expected_seq)
            self._next_expected_seq += 1
            return item

        try:
            item = self._prepared_q.get(timeout=0.05)
        except queue.Empty:
            return None

        if item is None:
            return None

        if item.seq == self._next_expected_seq:
            self._next_expected_seq += 1
            return item

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
            seq = self._global_seq
            self._last_yield_t = time.monotonic()
            self._advance()

        stem = Path(filename).stem
        label_path = self._labels_dir / f"{stem}.xml"
        return (
            seq,
            self._vis_dir / filename,
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

                seq, vis_path, ir_path, stem, label_path = item

                t0 = time.monotonic()
                vis_bgr = cv2.imread(str(vis_path))
                ir_bgr  = cv2.imread(str(ir_path))
                if vis_bgr is None or ir_bgr is None:
                    logger.warning("Failed to load %s — skipping", vis_path.name)
                    continue
                rgb_crop  = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
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
            logger.info("RtdodFrameProvider: max_frames=%d reached", self._max_frames)
        elif self._index >= len(self._filenames):
            if self._loop:
                logger.info("RtdodFrameProvider: restarting from the beginning")
                self._index = 0
            else:
                self._exhausted = True
                logger.info("RtdodFrameProvider: all images exhausted")

    _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def _load_splits(self, root: Path, splits: List[str]) -> List[str]:
        """Read split .txt files and return a deduplicated ordered list of filenames."""
        seen: set = set()
        filenames: List[str] = []
        for split in splits:
            txt = root / f"{split}.txt"
            if not txt.exists():
                raise FileNotFoundError(f"Split file not found: {txt}")
            for line in txt.read_text().splitlines():
                name = line.strip()
                if not name or name in seen:
                    continue
                if Path(name).suffix.lower() not in self._IMAGE_SUFFIXES:
                    logger.warning("Skipping non-image entry in split file: %s", name)
                    continue
                seen.add(name)
                filenames.append(name)
        if not filenames:
            raise ValueError(
                f"No filenames found in split(s) {splits} under {root}. "
                "Check that the .txt files are non-empty."
            )
        return filenames
