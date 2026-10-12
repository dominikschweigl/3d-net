from .base import RgbtYoloDataloader
from .fiftyone_rgbt_base import FiftyOneRgbtYoloDatasetDataloader
from .m3fd_rgbt import M3fdRgbtYoloDatasetDataloader
from .visdrone_rgbt import VisDroneRgbtYoloDatasetDataloader
from .rtdod_rgbt import RtdodRgbtYoloDatasetDataloader
from .three_d_net_rgbt import ThreeDNetRgbtYoloDatasetDataloader
from .three_d_net_data_pool import ThreeDNetDataPoolRgbtYoloDatasetDataloader


__all__ = [
    "RgbtYoloDataloader",
    "FiftyOneRgbtYoloDatasetDataloader",
    "VisDroneRgbtYoloDatasetDataloader",
    "M3fdRgbtYoloDatasetDataloader",
    "RtdodRgbtYoloDatasetDataloader",
    "ThreeDNetRgbtYoloDatasetDataloader",
    "ThreeDNetDataPoolRgbtYoloDatasetDataloader",
    "CLASS_NAMES",
]

CLASS_NAMES = ("person", "car", "motorcycle", "bicycle", "bus", "truck")