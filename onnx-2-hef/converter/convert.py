"""
script for optimizing quantization and compiling a model to .hef file
"""

import os

from util import (
    get_input_shape,
    read_calibration_dataset,
    read_test_data,
    build_model_script,
)
from evaluate import evaluate_model
import tensorflow as tf
from hailo_sdk_client import ClientRunner
import numpy as np
from clearml import Task
from ast import literal_eval


OPTIMIZATION_LEVEL = os.getenv("OPTIMIZATION_LEVEL")
COMPRESSION_LEVEL = os.getenv("COMPRESSION_LEVEL")
COMPILATION_LEVEL = os.getenv("COMPILATION_LEVEL")
DATASET = os.getenv("DATASET")
CALIBRATION_SIZE = int(os.getenv("CALIBRATION_DATA_SIZE"))
SKIP_PROFILING = os.getenv("SKIP_PROFILING")
SKIP_EVAL = os.getenv("SKIP_EVAL")
PROFILING_DATA_SIZE = int(os.getenv("PROFILING_DATA_SIZE"))
TASK_ID = os.getenv("CLEARML_TASK_ID")

if len(tf.config.list_physical_devices("GPU")) < 1:
    print(
        "⚠️ NO GPU found. It is strongly recommended to run the optimization on a GPU, performance may be affected otherwise!"
    )

image_h, image_w, image_c = get_input_shape("/model.onnx")

task = Task.get_task(task_id=TASK_ID)
parent_task = Task.get_task(task_id=task.parent)

# upload parsed HAR and visualization immediately so they're available even if later steps fail
task.upload_artifact(name="har_parsed", artifact_object="/model.har", wait_on_upload=True)
if os.path.exists("/model.svg"):
    task.upload_artifact(name="har_visualized", artifact_object="/model.svg", wait_on_upload=True)

# build model script
if not os.path.exists("/model_script.alls"):
    mean = literal_eval(parent_task.get_parameter("data/normalize_mean"))
    std = literal_eval(parent_task.get_parameter("data/normalize_std"))
    build_model_script(mean, std, image_w)

    upload_state = task.upload_artifact(
        name="modelscript", artifact_object="/model_script.alls", wait_on_upload=True
    )
    print(f"uploaded model script: {upload_state}")

print("optimizing quantization for model /model.har...")
runner = ClientRunner(har="/model.har")

# Diagnostic: print any commands already embedded in the HAR.
# If nms_postprocess appears here AND in the model script below, that is a conflict.
try:
    embedded = runner.get_hn_model_script()
    print(f"[diagnostic] HAR embedded model script:\n{embedded}")
except Exception as e:
    print(f"[diagnostic] Could not read embedded model script: {e}")

runner.load_model_script("/model_script.alls")

runner.optimize(
    read_calibration_dataset(
        DATASET,
        image_height=image_h,
        image_width=image_w,
        image_channels=image_c,
        calibration_size=CALIBRATION_SIZE,
    )
)
runner.save_har("/model_quant.har")
print("🚀 optimization done, saved to /model_quant.har")

print("start compilation to .hef file...")
runner = ClientRunner(har="/model_quant.har", hw_arch="hailo8")
hef = runner.compile()
with open("/model.hef", "wb") as f:
    f.write(hef)
print("💾 compilation done, saved to /model.hef")
upload_state = task.upload_artifact(
    name="hef", artifact_object="/model.hef", wait_on_upload=True
)
print(f"uploaded hef: {upload_state}")


# evaluate model performance
classes = literal_eval(parent_task.get_parameter("data/classes"))
profiling_data_np, profiling_data_fo = read_test_data(
    DATASET,
    image_height=image_h,
    image_width=image_w,
    image_channels=image_c,
    classes=classes,
)
np.save("/profiling.npy", profiling_data_np[:PROFILING_DATA_SIZE, :, :, :])

if not SKIP_EVAL:
    print("🕵️‍♀️ start model testing...")
    evaluate_model("/model.har", profiling_data_np, profiling_data_fo, parent_task)
