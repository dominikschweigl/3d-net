"""
script for downloading ONNX model and NMS config file from ClearML task artifacts
"""

import sys
from clearml import Task
import shutil
import os

TASK_ID = sys.argv[1]
task = Task.get_task(task_id=TASK_ID)


parent_task = Task.get_task(task_id=task.parent)

if parent_task is None:
    raise ValueError(
        "❌️ No parent task found. Needs parent task with ONNX model artifact."
    )

if "onnx" in parent_task.artifacts:
    onnx = parent_task.artifacts["onnx"].get_local_copy()
    shutil.copyfile(onnx, "/model.onnx")
else:
    raise ValueError("❌️ No ONNX model found in the tasks parents artifact.")

if "nms" in task.artifacts:
    print(f"📄 NMS config file found")
    nms = task.artifacts["nms"].get_local_copy()
    shutil.copy(nms, "/nms_config.json")
elif "detection" in task.get_tags():
    raise ValueError(
        "❌️ No NMS config file found in the task artifact. Upload the file as an artifact named 'nms'."
    )

if "modelscript" in task.artifacts:
    print(f"📄 modelscript file found")
    nms = task.artifacts["modelscript"].get_local_copy()
    shutil.copy(nms, "/model_script.alls")

# set tags
OPTIMIZATION_LEVEL = os.getenv("OPTIMIZATION_LEVEL")
COMPRESSION_LEVEL = os.getenv("COMPRESSION_LEVEL")
COMPILATION_LEVEL = os.getenv("COMPILATION_LEVEL")

tags = task.get_tags()
tags.append(f"OPTIMIZATION_LEVEL={OPTIMIZATION_LEVEL}")
tags.append(f"COMPILATION_LEVEL={COMPILATION_LEVEL}")
tags.append(f"COMPRESSION_LEVEL={COMPRESSION_LEVEL}")
task.set_tags(tags)
