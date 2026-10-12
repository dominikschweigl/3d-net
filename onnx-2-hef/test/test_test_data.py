import sys
from pathlib import Path
import fiftyone as fo

sys.path.insert(0, str(Path(__file__).parent.parent / "converter"))

from util import read_test_data

DATASET = "rtdod_rgbt:default"  # use a real snapshot date
dataset = fo.load_dataset(DATASET.split(":")[0], snapshot=DATASET.split(":")[1])
CLASSES = dataset.distinct('ground_truth.detections.label')

TEST_SIZE = 8

# Test 3-channel (RGB only) — profiling_set should be (TEST_SIZE, 640, 640, 3)
profiling_set, test_view = read_test_data(DATASET, 640, 640, image_channels=3, classes=CLASSES, size=TEST_SIZE)
assert profiling_set.shape == (TEST_SIZE, 640, 640, 3), profiling_set.shape
assert len(profiling_set) == len(test_view), "numpy array and fiftyone view length mismatch"
print("RGB: OK", profiling_set.shape)

# Test 4-channel (RGBT) — profiling_set should be (TEST_SIZE, 640, 640, 4)
profiling_set, test_view = read_test_data(DATASET, 640, 640, image_channels=4, classes=CLASSES, size=TEST_SIZE)
assert profiling_set.shape == (TEST_SIZE, 640, 640, 4), profiling_set.shape
assert len(profiling_set) == len(test_view), "numpy array and fiftyone view length mismatch"
assert profiling_set[..., 3].max() > 0, "thermal channel is all zeros — likely not loaded"
print("RGBT: OK", profiling_set.shape)
