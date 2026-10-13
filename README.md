# 3D-Net

3D-Net is an object detection system for RGB and thermal (RGBT) images. The system finds persons and vehicles in drone video. It operates in real time on a Raspberry Pi 5 with a Hailo AI accelerator.

The detector is YOLO11n. It can use the RGB image, the thermal image, or the two images together (fusion). A tracker gives each detected object a track ID.

The detector finds these 6 classes: `person`, `car`, `motorcycle`, `bicycle`, `bus`, `truck`.

## System overview

The system has 4 stages. The output of each stage is the input of the next stage.

```
Datasets (public and self-collected)
        │   3d-net/detection/data, 3d-net/detection/fiftyone, 3d-net/detection/cvat
        ▼
FiftyOne datasets (RGB image, thermal image, labels)
        │   3d-net-clearml (GPU training on ClearML)
        ▼
Trained model (.onnx)
        │   onnx-2-hef (Hailo quantization and compilation on ClearML)
        ▼
Compiled model (.hef)
        │   3d-net/detection/inference_service (Raspberry Pi 5 + Hailo)
        ▼
Detections with track IDs (JSON on ZeroMQ)
```

1. **Data.** Scripts prepare the public datasets and the self-collected drone video. CVAT is the labeling tool. FiftyOne keeps all datasets, labels, and splits.
2. **Training.** A ClearML agent trains the model on a remote GPU. The training exports the best model as an ONNX file.
3. **Conversion.** The onnx-2-hef Docker image quantizes the ONNX model. Then it compiles the model into a Hailo `.hef` file.
4. **Inference.** The inference service reads frames from the WIRIS camera or from a dataset. It sends the detections to other programs through ZeroMQ.

## Repository structure

The repository contains 3 projects. Each project has its own README with more data.

| Folder | Purpose | Details |
|---|---|---|
| [`3d-net/detection/`](3d-net/detection) | Data tools, labeling, tracking, and the inference service | See the table below |
| [`3d-net-clearml/`](3d-net-clearml) | Training and evaluation of YOLO11n models on ClearML | [README](3d-net-clearml/README.md) |
| [`onnx-2-hef/`](onnx-2-hef) | Docker image that converts ONNX models to Hailo `.hef` files | [README](onnx-2-hef/readme.md) |

The code in `3d-net/detection/` uses paths that start with `detection/`. These paths are relative to the `3d-net/` folder. For example, run `python -m detection.inference_service` from the `3d-net/` folder.

The `3d-net/detection/` folder contains these subfolders:

| Folder | Purpose | Details |
|---|---|---|
| [`data/`](3d-net/detection/data) | Split scripts and RGB/thermal viewers for each dataset | [FLIR ADAS](3d-net/detection/data/flir_adas/README.md), [self-collected](3d-net/detection/data/self_collection/README.md), [VisDrone](3d-net/detection/data/visdrone_dronevehicle/README.md) |
| [`fiftyone/`](3d-net/detection/fiftyone) | Scripts that make FiftyOne datasets, set the splits, merge the data pool, and plot statistics | – |
| [`fiftyone/dataset_builder/`](3d-net/detection/fiftyone/dataset_builder) | One script for each dataset that makes the FiftyOne dataset | – |
| [`fiftyone/tracks/`](3d-net/detection/fiftyone/tracks) | Scripts that make the track IDs unique across all CVAT tasks | – |
| [`cvat/`](3d-net/detection/cvat) | Scripts that make CVAT labeling jobs and pull the labels back into FiftyOne | – |
| [`tracker/`](3d-net/detection/tracker) | Comparison and evaluation of the trackers SORT, ByteTrack, OC-SORT, DeepSort, and BoT-SORT | – |
| [`inference_service/`](3d-net/detection/inference_service) | Real-time detection and tracking service for the Raspberry Pi (Docker) | [README](3d-net/detection/inference_service/README.md) |
| [`inference_service/full_precision_metrics/`](3d-net/detection/inference_service/full_precision_metrics) | Results of the full-precision models and the plot scripts | [Results](3d-net/detection/inference_service/full_precision_metrics/fusion_method_performance.md) |
| [`raspberry/`](3d-net/detection/raspberry) | Network setup and Hailo setup for the Raspberry Pi, and simple inference test scripts | [README](3d-net/detection/raspberry/README.md) |

## Hardware

| Component | Function |
|---|---|
| WIRIS Enterprise camera | Dual camera with RTSP streams. Visible: 1280×720, FOV 63.7°×38.6°. Thermal: 640×512, FOV 45.0°×37.0°. |
| Raspberry Pi 5 | Operates the inference service. |
| Hailo AI accelerator (PCIe, `/dev/hailo0`) | Operates the compiled `.hef` model. The converter compiles for the `hailo8` architecture. |
| x86 Linux computer with GPU | Operates the training and the conversion (ClearML agent). The Hailo Dataflow Compiler does not operate on ARM. |

## External services

The data, training, and conversion stages need access to these services:

| Service | Function |
|---|---|
| ClearML | Operates the training and conversion tasks on remote queues (default queue: `training-multi`). Keeps the artifacts. |
| FiftyOne server | Keeps the datasets, labels, splits, and evaluations. |
| S3-compatible storage | Keeps the images (for example, the bucket `bernard-3d-net`). |
| CVAT | Labeling tool for bounding boxes and tracks. |
| Azure Container Registry (`swarm.azurecr.io`) | Keeps the Docker base images and the `onnx-2-hef` image. |
| Azure Blob Storage | Keeps the ClearML artifacts. The converter downloads the ONNX model from this storage. |

## Datasets

All FiftyOne datasets use the same fields:

| Field | Content |
|---|---|
| `filepath` | Path to the RGB image |
| `thermal` | Path to the thermal image |
| `ground_truth` | Bounding boxes (VisDrone uses `gt_bbox_thermal`) |
| `split` | `train`, `val`, or `test` |

The training scripts select the dataset with the `FIFTYONE_DATASET` variable:

| Dataset | `FIFTYONE_DATASET` | FiftyOne name | Classes | Source |
|---|---|---|---|---|
| M3FD | `m3fd_rgbt` | `m3fd_rgbt` | person, car, bus, motorcycle, truck | Public |
| RTDOD | `rtdod_rgbt` | `rtdod_rgbt` | person, car, motorcycle, bicycle, bus, truck | Public |
| VisDrone DroneVehicle | `visdrone_rgbt` | `visdrone_dronevehicle_rgbt` | car, truck, bus | Public, drone view |
| 3D-Net | `3d_net_rgbt` | `3d_net_rgbt` | person, car, motorcycle, bicycle, bus, truck | Self-collected with the WIRIS camera |
| 3D-Net data pool | `3d_net_data_pool_rgbt` | `3d_net_data_pool_rgbt` | person, car, motorcycle, bicycle, bus, truck | Merge of the 4 datasets above |

The folder [`3d-net/detection/data/flir_adas/`](3d-net/detection/data/flir_adas) contains the FLIR ADAS dataset description and a viewer. No FiftyOne dataset exists for FLIR ADAS.

## Workflow

Do the steps in sequence. Each step links to the README with the full procedure.

### Step 1: Prepare the data

For self-collected drone video, do the procedure in [`3d-net/detection/data/self_collection/README.md`](3d-net/detection/data/self_collection/README.md). The procedure has these steps:

1. Record RGB and thermal video with the live capture of the inference service. The service records video that is aligned in time and in space.
2. Copy the recordings into the session folder structure.
3. Extract paired frames and upload them to S3.
4. Make the FiftyOne dataset `3d_net_rgbt`.
5. Label the frames in CVAT. Use the scripts in [`3d-net/detection/cvat/`](3d-net/detection/cvat).
6. Merge the dataset into the data pool.

For the public datasets, use the scripts in [`3d-net/detection/fiftyone/dataset_builder/`](3d-net/detection/fiftyone/dataset_builder).

To make or update the data pool, run [`3d-net/detection/fiftyone/merge_fiftyone_datapool.py`](3d-net/detection/fiftyone/merge_fiftyone_datapool.py). You can run the merge again. It does not make duplicate samples.

### Step 2: Train the model

> **CAUTION:** The `.env` file contains secrets. Do not commit the `.env` file.

1. Go to the training project:
   ```bash
   cd 3d-net-clearml
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the ClearML credentials, for example with `clearml-init`.
4. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```
5. Set the values in `.env`. The [training README](3d-net-clearml/README.md) explains each variable.
6. Start the training with one of these scripts:
   - `python yolov11n.py` trains on one modality (3 channels). Set `MODALITY` to `rgb` or `thermal`.
   - `python yolov11nrgbt.py` trains on RGB and thermal together (4 channels). Set `FUSION_TYPE` to select the fusion architecture.

The script sends the task to the ClearML queue and stops. Then a ClearML agent does the training. To train on the local computer, set `RUN_REMOTE = False` in the script.

The values for `FUSION_TYPE` are `early`, `mid`, `midP3`, `midCMA3`, `midCTF3`, `midCTF2`, `midCFT3`, `late`, and `score`. The model files are in [`3d-net-clearml/models/`](3d-net-clearml/models).

The training uploads these artifacts to ClearML: the best checkpoint, the metrics CSV, the ONNX model, and the evaluation on the `val` and `test` splits.

The results of all fusion types on all datasets are in [`fusion_method_performance.md`](3d-net/detection/inference_service/full_precision_metrics/fusion_method_performance.md). For more evaluations by object size and light condition, use the `validate_*.py` scripts in [`3d-net-clearml/`](3d-net-clearml).

### Step 3: Convert the model to HEF

The script `quantize.py` starts the onnx-2-hef converter as a ClearML child task of the training task.

1. Get the end node names of the ONNX model:
   ```bash
   python find_end_conv_layers.py best.onnx
   ```
2. Set `END_NODE_NAMES` in `.env` to the output of step 1.
3. Start the conversion with one of these methods:
   - Automatic: set `AUTO_QUANTIZE="true"` before the training. The conversion starts after a successful training.
   - Manual: set `CLEARML_QUANTIZE_TRAINING_TASK_ID` to the ID of the training task. Then run `python quantize.py`.
4. Download the `.hef` file from the artifacts of the ClearML conversion task.

> **NOTE:** With the default optimization and compilation levels, the conversion can take several hours.

The conversion task uploads these artifacts:

| Artifact | Content |
|---|---|
| `hef` | Hailo executable model for the accelerator |
| `har` | Hailo model archive |
| `report` | Profiler report with FPS, latency, and SNR |
| `evaluation_quantized` | Evaluation of the quantized model on the test split |
| `log` | Log of the Hailo toolchain |

Examine the SNR in the report. A value of more than 10 is satisfactory. For more data, refer to the [converter README](onnx-2-hef/readme.md) and the [training README](3d-net-clearml/README.md#quantization).

### Step 4: Set up the Raspberry Pi

Do the procedure in [`3d-net/detection/raspberry/README.md`](3d-net/detection/raspberry/README.md):

1. Install the Hailo driver, firmware, HailoRT, and TAPPAS core. Use the [Raspberry Pi AI guide](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started).
2. Connect the laptop, the Raspberry Pi, and the camera in the same subnet:

| Device | IP address |
|---|---|
| Laptop (Ethernet) | `10.174.42.1` |
| Raspberry Pi | `10.174.42.89` |
| WIRIS camera | `10.174.42.250` |

> **CAUTION:** The Raspberry Pi must be in the same subnet as the camera. If it is not, the Raspberry Pi cannot receive the camera streams.

### Step 5: Run the inference service

1. Copy the `.hef` file to `3d-net/detection/inference_service/models/` on the Raspberry Pi.
2. Edit [`3d-net/detection/inference_service/config.yaml`](3d-net/detection/inference_service/config.yaml). Set these values:
   - `provider.mode`: `live` for the WIRIS camera. `m3fd`, `3d_net`, or `rtdod` to replay a dataset.
   - `model.hef_path`: path to the `.hef` file.
   - `model.modality`: `rgb`, `thermal`, or `rgbt`. The value must agree with the input channels of the model.
   - `model.class_names`: the classes in the same sequence as in the training dataset.
   - `tracker.mode`: `none`, `sort`, `bytetrack`, or `ocsort`.
3. Build the Docker image:
   ```bash
   cd 3d-net/detection/inference_service
   docker compose build
   ```
4. Start the service with one of these commands:
   - `docker compose up` for normal operation.
   - `docker compose run inference-service` for one run, for example an evaluation run.

The service gives these outputs:

| Output | Location |
|---|---|
| Detections with track IDs (JSON, `FrameResult` in [`models.py`](3d-net/detection/inference_service/models.py)) | ZeroMQ PUB socket `tcp://*:5556`, topic `detections` |
| Latency and detection logs (CSV) | `3d-net/detection/inference_service/logs/` |
| Video recordings (if `capture` or `inference_recorder` is enabled) | `3d-net/detection/inference_service/recordings/` |
| Evaluation against ground truth (if `evaluation.enabled` is `true`) | `3d-net/detection/inference_service/logs/<session>/evaluation/` |

To record aligned video for the data collection without the model, set `capture.enabled` and `service.record_only` to `true`. This mode needs `provider.mode: live`. Refer to [Step 1](#step-1-prepare-the-data).

To monitor the Hailo utilization during operation, run `hailortcli monitor` on the Raspberry Pi.

### Step 6: Evaluate the tracker (optional)

- [`3d-net/detection/tracker/compare_trackers.py`](3d-net/detection/tracker/compare_trackers.py) compares 5 trackers. It uses the ground-truth boxes as the detector input.
- [`3d-net/detection/tracker/eval/`](3d-net/detection/tracker/eval) evaluates one tracker with the Hailo model on the Raspberry Pi. Start it with Docker Compose.

The two tools calculate the HOTA, CLEAR, and Identity metrics.

Before you start the evaluation in `tracker/eval/`, export the dataset with the track IDs. Run this command from the `3d-net/` folder on a computer with FiftyOne access:

```bash
python detection/fiftyone/download_rgbt_dataset.py --output detection/data/3d_net
```

## Software requirements

| Project | Requirement |
|---|---|
| `3d-net-clearml` | Python 3.10 or higher, CUDA GPU (ClearML agent), [YOLOv11-RGBT](https://github.com/wandahangFY/YOLOv11-RGBT) fork of Ultralytics 8.3.75 |
| `onnx-2-hef` | Docker image based on Hailo AI Software Suite 2025.4. The Hailo Dataflow Compiler supports only x86 Linux and Python 3.10. |
| `3d-net/detection/inference_service` | Docker on the Raspberry Pi 5 (Python 3.13 image). The container uses HailoRT 4.23 from the host. |

## License

This project uses the MIT License. Refer to [LICENSE](LICENSE).
