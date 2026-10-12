from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class RgbtYoloDataloader(ABC):
    """Interface for dataloaders that export YOLO-style RGBT datasets."""

    @abstractmethod
    def prepare(self) -> Path:
        """Ensure dataset exists on disk and return the path to dataset.yaml."""
        raise NotImplementedError
