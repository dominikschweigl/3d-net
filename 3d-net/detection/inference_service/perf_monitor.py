"""
Performance monitor for the inference loop.

Per-frame latency data is written to a CSV log file on every frame so downstream
analysis can compute any statistic (percentiles, histograms, correlations with
detection count, etc.) without information loss from pre-aggregation.

A second CSV logs per-detection detail (class, confidence, bounding box) linked
to the perf CSV by a shared ``frame_id`` and ``timestamp``.

Hailo temperature, CPU temperature, and system CPU/RAM are queried once every
`hw_poll_every_frames` frames and the cached values are included in each CSV row.

A rolling-window aggregate (avg/max per stage, FPS) is printed to stdout at a
configurable interval — useful for at-a-glance monitoring without scrolling.

Perf CSV columns:
    timestamp, frame_id, drift_ms, align_ms, pre_ms, infer_ms, post_ms, track_ms,
    n_detections, hailo_temp_c, hailo_pct, cpu_temp_c, cpu_pct, ram_used_mb

Detections CSV columns:
    frame_id, timestamp, det_idx, class_name, label, confidence,
    x1, y1, x2, y2, bbox_w, bbox_h, track_id
"""

import csv
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

from .hailo_monitor.hailo_monitor_parser import HailoMonitorParser
from .models import Detection

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    logger.debug("psutil not available — CPU/RAM stats disabled")

if sys.stdout.isatty():
    GREEN, CYAN, YELLOW, BOLD, RESET = "\033[32m", "\033[36m", "\033[33m", "\033[1m", "\033[0m"
else:
    GREEN = CYAN = YELLOW = BOLD = RESET = ""

_CSV_COLUMNS = [
    "timestamp", "frame_id", "frame_stem",
    "drift_ms", "frame_provider_ms", "align_ms", "pre_ms", "infer_ms", "post_ms", "track_ms",
    "fps", "n_detections",
    "hailo_temp_c", "hailo_pct", "cpu_temp_c",
    "cpu_pct", "ram_used_mb",
]

_DET_CSV_COLUMNS = [
    "frame_id", "timestamp", "frame_stem", "det_idx",
    "class_name", "label", "confidence",
    "x1", "y1", "x2", "y2", "bbox_w", "bbox_h",
    "track_id",
]


class PerfMonitor:
    """
    Collects per-frame timings, writes them to a CSV log, and periodically
    prints rolling aggregate stats to stdout.

    Usage:
        monitor = PerfMonitor(window_frames=100, report_interval_s=10.0,
                              log_path=Path("/app/logs"))
        monitor.set_hailo_device(physical_device)   # optional, after VDevice init
        # per frame:
        monitor.record_frame(frame_id, drift_s, align_s, pre_s, infer_s, post_s, detections)
        monitor.maybe_report()
        # on shutdown:
        monitor.close()
    """

    def __init__(
        self,
        window_frames: int = 100,
        report_interval_s: float = 10.0,
        log_path: Optional[Path] = None,
        hw_poll_every_frames: int = 30,
        session: Optional[str] = None,
    ):
        self._window = window_frames
        self._report_interval_s = report_interval_s
        self._hw_poll_every = hw_poll_every_frames
        self._frame_count = 0

        # Rolling latency buffers (milliseconds) — for stdout aggregate only
        self._frame_provider_ms: Deque[float] = deque(maxlen=window_frames)
        self._align_ms:  Deque[float] = deque(maxlen=window_frames)
        self._pre_ms:    Deque[float] = deque(maxlen=window_frames)
        self._infer_ms:  Deque[float] = deque(maxlen=window_frames)
        self._post_ms:   Deque[float] = deque(maxlen=window_frames)
        self._track_ms:  Deque[float] = deque(maxlen=window_frames)
        self._frame_ts:  Deque[float] = deque(maxlen=window_frames)

        # Cached hardware + system stats (refreshed every hw_poll_every_frames)
        self._hailo_temp_c:   Optional[float] = None
        self._hailo_pct:      Optional[float] = None
        self._cpu_temp_c:     Optional[float] = None
        self._cpu_pct:        Optional[float] = None
        self._ram_used_mb:    Optional[int]   = None
        self._hailo_monitor_parser = HailoMonitorParser()

        self._hailo_device = None

        self._last_report_t = time.monotonic()

        # CSV logs — one file pair per session so runs don't clobber each other.
        self._csv_file = None
        self._csv_writer = None
        self._det_csv_file = None
        self._det_csv_writer = None
        self._stats_csv_path: Optional[Path] = None
        self._detections_csv_path: Optional[Path] = None

        if log_path is not None:
            session_dir = (log_path / session) if session else log_path
            session_dir.mkdir(parents=True, exist_ok=True)

            perf_path = session_dir / "inference_stats.csv"
            self._stats_csv_path = perf_path
            self._csv_file = open(perf_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_COLUMNS)
            self._csv_writer.writeheader()
            logger.info("PerfMonitor: logging per-frame stats to %s", perf_path)

            det_path = session_dir / "inference_detections.csv"
            self._detections_csv_path = det_path
            self._det_csv_file = open(det_path, "w", newline="")
            self._det_csv_writer = csv.DictWriter(self._det_csv_file, fieldnames=_DET_CSV_COLUMNS)
            self._det_csv_writer.writeheader()
            logger.info("PerfMonitor: logging per-detection data to %s", det_path)

    @property
    def stats_csv_path(self) -> Optional[Path]:
        return self._stats_csv_path

    @property
    def detections_csv_path(self) -> Optional[Path]:
        return self._detections_csv_path

    @property
    def current_fps(self) -> Optional[float]:
        if len(self._frame_ts) < 2:
            return None
        elapsed = self._frame_ts[-1] - self._frame_ts[0]
        return (len(self._frame_ts) - 1) / elapsed if elapsed > 0 else None

    # ── Hailo device setup ────────────────────────────────────────────────────

    def set_hailo_device(self, device) -> None:
        """
        Attach a Hailo physical device for hailo temperature monitoring.
        Call once after VDevice.configure(). Safe to skip — stats column stays empty.
        """
        self._hailo_device = device

    # ── Per-frame recording ───────────────────────────────────────────────────

    def record_frame(
        self,
        frame_id: str,
        drift_s: float,
        frame_provider_s: float,
        align_s: float,
        pre_s: float,
        infer_s: float,
        post_s: float,
        track_s: float,
        detections: List[Detection],
        frame_stem: Optional[str] = None,
    ) -> None:
        """Record one frame. Refreshes HW/system stats every hw_poll_every_frames frames."""
        # Update rolling buffers for stdout aggregate
        self._frame_provider_ms.append(frame_provider_s * 1000)
        self._align_ms.append(align_s * 1000)
        self._pre_ms.append(pre_s * 1000)
        self._infer_ms.append(infer_s * 1000)
        self._post_ms.append(post_s * 1000)
        self._track_ms.append(track_s * 1000)
        self._frame_ts.append(time.monotonic())
        self._frame_count += 1

        # Refresh HW + system stats periodically
        if self._frame_count % self._hw_poll_every == 1:
            self._refresh_hw_stats()

        ts = datetime.now(timezone.utc).isoformat()

        # Write perf CSV row
        if self._csv_writer is not None:
            fps = self.current_fps
            self._csv_writer.writerow({
                "timestamp":         ts,
                "frame_id":          frame_id,
                "frame_stem":        frame_stem or "",
                "drift_ms":          round(drift_s          * 1000, 3),
                "frame_provider_ms": round(frame_provider_s * 1000, 3),
                "align_ms":          round(align_s          * 1000, 3),
                "pre_ms":       round(pre_s    * 1000, 3),
                "infer_ms":     round(infer_s  * 1000, 3),
                "post_ms":      round(post_s   * 1000, 3),
                "track_ms":     round(track_s  * 1000, 3),
                "fps":          round(fps, 2) if fps is not None else "",
                "n_detections": len(detections),
                "hailo_temp_c": self._hailo_temp_c if self._hailo_temp_c is not None else "",
                "hailo_pct":    self._hailo_pct    if self._hailo_pct    is not None else "",
                "cpu_temp_c":   self._cpu_temp_c   if self._cpu_temp_c   is not None else "",
                "cpu_pct":      self._cpu_pct      if self._cpu_pct      is not None else "",
                "ram_used_mb":  self._ram_used_mb  if self._ram_used_mb  is not None else "",
            })

        # Write detection CSV rows
        if self._det_csv_writer is not None:
            for det_idx, det in enumerate(detections):
                b = det.bbox
                self._det_csv_writer.writerow({
                    "frame_id":   frame_id,
                    "timestamp":  ts,
                    "frame_stem": frame_stem or "",
                    "det_idx":    det_idx,
                    "class_name": det.name,
                    "label":      det.label,
                    "confidence": round(det.confidence, 4),
                    "x1":         b.x1,
                    "y1":         b.y1,
                    "x2":         b.x2,
                    "y2":         b.y2,
                    "bbox_w":     b.x2 - b.x1,
                    "bbox_h":     b.y2 - b.y1,
                    "track_id":   det.track_id if det.track_id is not None else "",
                })

    def _refresh_hw_stats(self) -> None:
        self._hailo_pct = self._hailo_monitor_parser.read_latest_utilization_pct()

        if self._hailo_device is not None:
            try:
                temp = self._hailo_device.control.get_chip_temperature()
                self._hailo_temp_c = round(temp.ts0_temperature, 1)
            except Exception as exc:
                logger.debug("PerfMonitor: hailo temperature read failed: %s", exc)

        if _PSUTIL_OK:
            try:
                self._cpu_pct = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                self._ram_used_mb = round(mem.used / 1024 / 1024)
                temps = psutil.sensors_temperatures()
                # Raspberry Pi exposes CPU temp under 'cpu_thermal'; fall back to
                # the first available sensor entry if not found.
                entry = None
                for key in ("cpu_thermal", "coretemp", "k10temp", "acpitz"):
                    if key in temps and temps[key]:
                        entry = temps[key][0]
                        break
                if entry is None:
                    for entries in temps.values():
                        if entries:
                            entry = entries[0]
                            break
                if entry is not None:
                    self._cpu_temp_c = round(entry.current, 1)
            except Exception as exc:
                logger.debug("PerfMonitor: psutil read failed: %s", exc)

    # ── Periodic stdout report ────────────────────────────────────────────────

    def maybe_report(self) -> bool:
        """Print rolling aggregate stats to stdout if the interval has elapsed."""
        if time.monotonic() - self._last_report_t < self._report_interval_s:
            return False
        self._last_report_t = time.monotonic()

        if self._csv_file:
            self._csv_file.flush()
        if self._det_csv_file:
            self._det_csv_file.flush()

        parts = [f"{BOLD}[STAT]{RESET}"]

        if len(self._frame_ts) >= 2:
            elapsed = self._frame_ts[-1] - self._frame_ts[0]
            if elapsed > 0:
                fps = (len(self._frame_ts) - 1) / elapsed
                parts.append(f"{CYAN}fps={fps:.1f}{RESET}")

        for label, buf in (
            ("frame_provider", self._frame_provider_ms),
            ("align",          self._align_ms),
            ("pre",            self._pre_ms),
            ("infer",          self._infer_ms),
            ("post",           self._post_ms),
            ("track",          self._track_ms),
        ):
            if buf:
                avg = sum(buf) / len(buf)
                mx  = max(buf)
                parts.append(f"{label}={avg:.1f}/{mx:.1f}ms")

        if self._hailo_temp_c is not None:
            colour = YELLOW if self._hailo_temp_c > 70 else GREEN
            parts.append(f"hailo={colour}{self._hailo_temp_c:.0f}°C{RESET}")
        if self._hailo_pct is not None:
            parts.append(f"hailo_pct={self._hailo_pct:.1f}%")
        if self._cpu_temp_c is not None:
            colour = YELLOW if self._cpu_temp_c > 80 else GREEN
            parts.append(f"cpu_temp={colour}{self._cpu_temp_c:.0f}°C{RESET}")
        if self._cpu_pct is not None:
            parts.append(f"cpu={self._cpu_pct:.0f}%")
        if self._ram_used_mb is not None:
            parts.append(f"ram={self._ram_used_mb}MB")

        print("  ".join(parts))
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
        if self._det_csv_file:
            self._det_csv_file.flush()
            self._det_csv_file.close()
