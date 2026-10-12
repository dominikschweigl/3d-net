#!/bin/bash
set -e

# set default values for environment variables if not set
if [[ -z "${OPTIMIZATION_LEVEL}" ]]; then
  export OPTIMIZATION_LEVEL=4
  echo "⚠️ Optimization level not set, defaulting to 4 (maximum). [possible values: -100, 0, 1, 2, 3, 4]"
fi
if [[ -z "${COMPILATION_LEVEL}" ]]; then
  export COMPILATION_LEVEL="max"
  echo "⚠️ Compilation optimization level not set, defaulting to 'max' (maximum). [possible values: 0, 1, 2, max]"
fi
if [[ -z "${COMPRESSION_LEVEL}" ]]; then
  export COMPRESSION_LEVEL=0
  echo "Compression optimization level not set, defaulting to 0 (minimum). [possible values: 0, 1, 2, 3, 4, 5]"
fi
if [[ -z "${CALIBRATION_DATA_SIZE}" ]]; then
  export CALIBRATION_DATA_SIZE=2048
  echo "Calibration data set size not set, defaulting to 2048. [recommended: >=1024]"
fi
if [[ -z "${PROFILING_DATA_SIZE}" ]]; then
  export PROFILING_DATA_SIZE=64
  echo "Profiling data set size not set, defaulting to 64. [recommended: 64<=size<200]"
fi

export FIFTYONE_MEDIA_CACHE_DIR="/home/swarm/data"

# download ONNX file and NMS config
python download_artifacts.py $CLEARML_TASK_ID

# parse ONNX model
if [[ -n "${END_NODE_NAMES}" ]]; then
  echo "Parsing ONNX model with end node names: ${END_NODE_NAMES}"
  yes n | hailo parser onnx /model.onnx \
    --har-path /model.har --hw-arch=hailo8 \
    --end-node-names $END_NODE_NAMES
else
  hailo parser onnx --har-path /model.har --hw-arch=hailo8 -y /model.onnx
fi
# visualize parsed HAR so node names can be inspected
hailo visualizer /model.har --out-path /model.svg --no-browser || true

# quantize, compile and evaluate
python convert.py

# profile compiled model
if [[ -z "${SKIP_PROFILING}" ]]; then
  hailo analyze-noise --data-path /profiling.npy /model_quant.har
  hailo profiler --hef /model.hef --out-path /report.html /model_quant.har
fi

# upload result artifacts to ClearML
python upload_artifacts.py $CLEARML_TASK_ID
