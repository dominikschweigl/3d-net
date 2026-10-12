import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "converter"))

from util import read_calibration_dataset

DATASET = "rtdod_rgbt:default"  # use a real snapshot date

# Test 3-channel (RGB only) — should return (8, 640, 640, 3)
result = read_calibration_dataset(DATASET, 640, 640, image_channels=3, calibration_size=8)
assert result.shape == (8, 640, 640, 3), result.shape
print("RGB: OK", result.shape)

# Test 4-channel (RGBT) — should return (8, 640, 640, 4)
result = read_calibration_dataset(DATASET, 640, 640, image_channels=4, calibration_size=8)
assert result.shape == (8, 640, 640, 4), result.shape
assert result[..., 3].max() > 0, "thermal channel is all zeros — likely not loaded"
print("RGBT: OK", result.shape)
