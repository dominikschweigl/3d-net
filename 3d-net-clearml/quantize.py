"""
Launch an onnx-2-hef quantization + compilation task on ClearML.

This script creates a ClearML child task that:
  1. Downloads the ONNX model from the parent training task
  2. Quantizes + compiles it to a Hailo .hef file using the onnx-2-hef docker image
  3. Uploads the resulting .har, .hef, profiling report and evaluation back to ClearML

Prerequisites
-------------
- A completed yolov11n.py training task whose task-ID you supply below.
- An NMS config JSON file (required for detection models) with at least the fields:
      {"nms_scores_th": <float>, "nms_iou_th": <float>}
- A FiftyOne dataset used for calibration/quantization, accessible via the API key below.
- All variables from .env are forwarded into the docker container automatically.
"""

import json
import os
from pathlib import Path

from clearml import Task
from dotenv import dotenv_values, load_dotenv

load_dotenv(".env")
# Use .env file when available (local runs); fall back to os.environ on remote agents
# where .env is not present on disk but all vars were injected via Docker -e flags.
env = dotenv_values(".env") if Path(".env").exists() else dict(os.environ)

# ─────────────────────────────────────────────────────────────────────────────
# Required – fill these in before running
# ─────────────────────────────────────────────────────────────────────────────

# ClearML task-ID of the completed training run to quantize.
# Find it in the ClearML UI URL or via Task.get_task(project_name=..., task_name=...).
CLEARML_PARENT_ID = os.getenv("CLEARML_QUANTIZE_TRAINING_TASK_ID")

# FiftyOne dataset used for quantization calibration.
# Dataset name is read from FIFTYONE_DATASET (same variable as training) to avoid misalignment.
# Snapshot is read from FIFTYONE_SNAPSHOT (e.g. "default" or a date string like "2025-01-15").
DATASET = f"{os.getenv('FIFTYONE_DATASET', 'rtdod_rgbt')}:{os.getenv('FIFTYONE_SNAPSHOT', 'default')}"

# ClearML queue that has a worker with GPU + the onnx-2-hef docker image available.
QUEUE = os.getenv("CLEARML_QUEUE", "training-multi")

# Path to the NMS config JSON file for detection (required for YOLOv11 detection).
# The file must contain: {"nms_scores_th": <float>, "nms_iou_th": <float>}
NMS_CONFIG_PATH = Path(os.getenv("NMS_CONFIG_PATH", "yolo11n_nms_config.json"))
MODEL_SCRIPT_PATH = Path(os.getenv("MODEL_SCRIPT_PATH", "")) if os.getenv("MODEL_SCRIPT_PATH") else None


# ─────────────────────────────────────────────────────────────────────────────
# Optional – override quantization / compilation settings
# Defaults (in brackets) are what the onnx-2-hef entrypoint uses when unset.
# ─────────────────────────────────────────────────────────────────────────────

# Optimization level: -100 (none), 0, 1, 2, 3, 4 [default: 4 for production]
OPTIMIZATION_LEVEL = os.getenv("OPTIMIZATION_LEVEL", "4")

# Compression level: 0–5 (% of weights compressed to 4-bit) [default: 0]
COMPRESSION_LEVEL = os.getenv("COMPRESSION_LEVEL", "0")

# Compiler optimization level: 0, 1, 2, "max" [default: "max"]
COMPILATION_LEVEL = os.getenv("COMPILATION_LEVEL", "max")

# Number of images used for quantization calibration [default: 2048, recommended >=1024]
CALIBRATION_DATA_SIZE = os.getenv("CALIBRATION_DATA_SIZE", "2048")

# Number of images used for profiling [default: 100, recommended 64–200]
PROFILING_DATA_SIZE = os.getenv("PROFILING_DATA_SIZE", "100")

# Set to "true" to skip profiling (faster, but no FPS/latency report) [default: "true"]
SKIP_PROFILING = os.getenv("SKIP_PROFILING", "true")


# ─────────────────────────────────────────────────────────────────────────────
# Create and launch the quantization task
# ─────────────────────────────────────────────────────────────────────────────

def create_nms_config(path: Path, scores_th: float = 0.3, iou_th: float = 0.45) -> Path:
    """Helper to create a minimal NMS config JSON if you don't have one yet."""
    config = {"nms_scores_th": scores_th, "nms_iou_th": iou_th}
    path.write_text(json.dumps(config, indent=2))
    print(f"NMS config written to {path}")
    return path


def main():
    task: Task = Task.init(
        project_name="YOLO11-3DNet",
        task_name=f"yolo11n_quantize_hef_{CLEARML_PARENT_ID}",
        task_type=Task.TaskTypes.custom,
    )

    # Link to the training task so the converter can fetch the ONNX artifact
    task.set_parent(CLEARML_PARENT_ID)
    print(f"Task initialized with ID: {task.id} and parent ID: {CLEARML_PARENT_ID}")
    parent_task: Task = Task.get_task(task_id=CLEARML_PARENT_ID)
    task.add_tags(parent_task.get_tags())

    # Upload NMS config (required for detection models)
    if not NMS_CONFIG_PATH.exists():
        create_nms_config(NMS_CONFIG_PATH)
    task.upload_artifact(name="nms", artifact_object=str(NMS_CONFIG_PATH))

    # Upload model script if present — skips auto-generation in convert.py
    if MODEL_SCRIPT_PATH is not None and MODEL_SCRIPT_PATH.exists():
        task.upload_artifact(name="modelscript", artifact_object=str(MODEL_SCRIPT_PATH))

    # Build the docker env string
    docker_env = (
        f"-e DATASET={DATASET} "
        f"-e OPTIMIZATION_LEVEL={OPTIMIZATION_LEVEL} "
        f"-e COMPRESSION_LEVEL={COMPRESSION_LEVEL} "
        f"-e COMPILATION_LEVEL={COMPILATION_LEVEL} "
        f"-e CALIBRATION_DATA_SIZE={CALIBRATION_DATA_SIZE} "
        f"-e PROFILING_DATA_SIZE={PROFILING_DATA_SIZE} "
        f"-e SKIP_PROFILING={SKIP_PROFILING} "
        f"-e SKIP_EVAL={SKIP_PROFILING} "
        + "".join([f"-e {key}='{value}' " for key, value in env.items()])
    )

    task.set_base_docker(
        "swarm.azurecr.io/onnx-2-hef:feat-4_channel_support",
        docker_arguments="--pull=always " + docker_env,
    )

    print(f"Launching quantization task on queue '{QUEUE}'...")
    print(f"  Parent training task : {CLEARML_PARENT_ID}")
    print(f"  Calibration dataset  : {DATASET}")
    print(f"  Optimization level   : {OPTIMIZATION_LEVEL}")
    print(f"  Compression level    : {COMPRESSION_LEVEL}")
    print(f"  Compilation level    : {COMPILATION_LEVEL}")
    print(f"  Calibration size     : {CALIBRATION_DATA_SIZE}")
    print(f"  Profiling size       : {PROFILING_DATA_SIZE}")
    task.execute_remotely(queue_name=str(QUEUE), exit_process=True)


if __name__ == "__main__":
    main()
