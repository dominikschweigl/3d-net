"""Hailo monitor file parser for extracting utilization percentages.

This parser reads files written by HAILO_MONITOR (typically in /tmp/hmon_files),
decodes them using the protobuf runtime, and extracts per-device utilization
values from ProtoMon.device_infos records.
"""

from dataclasses import dataclass
import logging
import os
import re
from pathlib import Path
from typing import Optional

from google.protobuf.message import DecodeError

from .scheduler_mon_pb2 import ProtoMon  # <-- generated from hailo_mo.proto

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceMetric:
    """Parsed utilization metrics for a single Hailo device."""
    pci: str
    chip: str | None
    utilization: float


class HailoMonitorParser:
    """
    Read Hailo utilization (%) from the newest Hailo monitor file.

    These monitor files are where Hailo writes current performance metrics during
    runtime (HAILO_MONITOR output). Parsing is done with the protobuf runtime
    using the ProtoMon schema.

    When reading a file with multiple devices, this class returns the first parsed
    device metric.
    """

    _PCI_BDF_RE = re.compile(r"^\d{4}:\d{2}:\d{2}\.\d$")

    def __init__(self, monitor_dir: Optional[Path] = None):
        self._monitor_dir = monitor_dir or Path(
            os.getenv("HAILO_MONITOR_DIR", "/tmp/hmon_files")
        )
        self._last_file: Optional[str] = None
        self._last_mtime_ns: Optional[int] = None
        self._last_metric: Optional[DeviceMetric] = None

    def read_latest_device_metric(self) -> Optional[DeviceMetric]:
        """Return the first device metric from the newest monitor file."""
        latest = self._find_latest_file()
        if latest is None:
            return None

        latest_file, mtime_ns = latest
        if self._last_file == latest_file.name and self._last_mtime_ns == mtime_ns:
            return self._last_metric

        self._last_file = latest_file.name
        self._last_mtime_ns = mtime_ns

        try:
            metrics = self._extract_metrics(latest_file.read_bytes())
        except (OSError, ValueError) as exc:
            logger.debug(
                "HailoMonitorParser: failed to parse file %s: %s",
                latest_file,
                exc,
            )
            self._last_metric = None
            return None

        self._last_metric = metrics[0] if metrics else None
        return self._last_metric

    def read_latest_utilization_pct(self) -> Optional[float]:
        """Return utilization (%) for the first parsed device."""
        metric = self.read_latest_device_metric()
        return round(metric.utilization, 2) if metric else None

    def _find_latest_file(self) -> Optional[tuple[Path, int]]:
        if not self._monitor_dir.exists():
            return None

        try:
            files = [f for f in self._monitor_dir.iterdir() if f.is_file()]
        except OSError:
            return None

        if not files:
            return None

        try:
            latest = max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))
            return latest, latest.stat().st_mtime_ns
        except OSError:
            return None

    @classmethod
    def _extract_metrics(cls, buf: bytes) -> list[DeviceMetric]:
        proto_mon = ProtoMon()

        try:
            proto_mon.ParseFromString(buf)
        except DecodeError as exc:
            raise ValueError(f"Failed to decode ProtoMon message: {exc}") from exc

        metrics: list[DeviceMetric] = []

        for device_info in proto_mon.device_infos:
            device_id = str(device_info.device_id)

            if not cls._PCI_BDF_RE.match(device_id):
                continue

            metrics.append(
                DeviceMetric(
                    pci=device_id,
                    chip=device_info.device_arch or None,
                    utilization=float(device_info.utilization),
                )
            )

        # Deduplicate
        unique = []
        seen = set()

        for m in metrics:
            key = (m.pci, m.chip, int(round(m.utilization * 1000)))
            if key not in seen:
                seen.add(key)
                unique.append(m)

        return unique

    @classmethod
    def _extract_hailo_utilization_values(cls, buf: bytes) -> list[float]:
        return [m.utilization for m in cls._extract_metrics(buf)]
