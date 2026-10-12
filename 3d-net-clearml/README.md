
# 3d-net-clearml

Train and evaluate an **Ultralytics YOLO11n** detector on **RGB+Thermal (4-channel)** datasets, with experiment tracking and remote execution via **ClearML**.

This repository is intentionally small: it contains a single training entrypoint ([yolov11n.py](yolov11n.py)), a dataset exporter package ([dataloader/__init__.py](dataloader/__init__.py)) and a helper for qualitative “good/bad” sample export ([test/test_balanced_sample_export.py](test/test_balanced_sample_export.py)).

## What it does

The training script performs these steps:

- Initializes a ClearML training task (project/task name configurable in code)
- (Optional) Sends the run to a ClearML Agent queue (remote GPU execution)
- Exports a YOLO-style dataset from an existing **FiftyOne dataset** into `workspace/...` as:
	- `images/{train,val,test}/*.tiff` containing **4 channels** (RGB + thermal)
	- `labels/{train,val,test}/*.txt` in YOLO bbox format
	- `dataset.yaml` describing the dataset and `channels: 4`
- Trains YOLO11n via `ultralytics.YOLO(...).train(...)`
- Exports ONNX
- Runs evaluation on **val** and **test** splits
- Uploads key artifacts to ClearML (best checkpoint, metrics CSV, ONNX)
- Exports a balanced set of qualitative samples (“good” vs “bad”) from the **test** split

## Repo layout

- [yolov11n.py](yolov11n.py) – main entrypoint (ClearML + export + train + eval + export)
- [dataloader/base.py](dataloader/base.py) – dataloader interface (`RgbtYoloDataloader`)
- [dataloader/visdrone_rgbt.py](dataloader/visdrone_rgbt.py) – `VisDroneRgbtYoloDatasetDataloader` (FiftyOne: `visdrone_dronevehicle_rgbt`)
- [dataloader/m3fd_rgbt.py](dataloader/m3fd_rgbt.py) – `M3fdRgbtYoloDatasetDataloader` (FiftyOne: `m3fd_rgbt`)
- [dataloader/rtdod_rgbt.py](dataloader/rtdod_rgbt.py) – `RtdodRgbtYoloDatasetDataloader` (FiftyOne: `rtdod_rgbt`)
- [dataloader/three_d_net_rgbt.py](dataloader/three_d_net_rgbt.py) – `ThreeDNetRgbtYoloDatasetDataloader` (FiftyOne: `3d_net_rgbt`)
- [dataloader/three_d_net_data_pool.py](dataloader/three_d_net_data_pool.py) – `ThreeDNetDataPoolRgbtYoloDatasetDataloader` (FiftyOne: `3d_net_data_pool_rgbt`)
- [requirements.txt](requirements.txt) – Python deps
- [Dockerfile](Dockerfile) – container entry (used by ClearML Agent and for local Docker runs)

## Requirements

- Python environment compatible with Ultralytics (typically Python 3.10+)
- A working ClearML setup (local machine or remote agent as described in the [ClearML Guide](https://tec.fibery.io/search/uF07#Knowledgebase/Wiki_Page/ClearML-569))
- Access to the FiftyOne dataset referenced by the script

## Setup

### 1) Install dependencies

From the repo root:

```bash
pip install -r requirements.txt
```

### 2) Configure ClearML

Ensure your ClearML credentials are available (for example via `clearml-init`, or environment variables / agent config).

The script uses:

- `Task.init(project_name=..., task_name=...)`
- `task.set_base_docker(...)` to pin a base image
- optional `task.execute_remotely(queue_name=...)`

### 3) Create a `.env`

[yolov11n.py](yolov11n.py) loads `.env` and forwards all variables into the ClearML job’s Docker runtime (`docker_arguments=...`).

Common variables you may need:

- `CLEARML_QUEUE` (default: `training-multi`)
	- ClearML queue to submit training and quantization tasks to.
- `CLEARML_QUANTIZE_TRAINING_TASK_ID`
	- Task-ID of the completed training run to quantize. Required when running `quantize.py`. Find it in the ClearML UI URL.
- `FIFTYONE_DATASET` (default: `rtdod_rgbt`)
	- Which FiftyOne dataset to train on. One of `visdrone_rgbt`, `rtdod_rgbt`, `m3fd_rgbt`, `3d_net_rgbt`, `3d_net_data_pool_rgbt`.
	- Also used by `quantize.py` as the calibration dataset to prevent train/quantize misalignment.
- `FIFTYONE_SNAPSHOT` (default: `default`)
	- FiftyOne snapshot used by `quantize.py` for calibration. Combined with `FIFTYONE_DATASET` to form `<dataset>:<snapshot>`.
- `EPOCHS` (default: `10`)
	- Number of training epochs.
- `IMGSZ` (default: `640`)
	- Input image size for training and validation.
- `BATCH` (default: `32`)
	- Training batch size. Increase to improve GPU utilization as long as memory allows.
- `WORKERS` (default: `8`)
	- Number of dataloader worker processes.
- `TEST_SAMPLES` (default: `10`)
	- Number of qualitative balanced ("good"/"bad") samples exported after evaluation.
- `AUTO_QUANTIZE` (default: `false`)
	- Set to `"true"` to automatically launch the quantization task immediately after a successful training run. The training task's own ID is passed as the parent automatically — no need to set `CLEARML_QUANTIZE_TRAINING_TASK_ID` manually.
- `REPO_DEPLOY_TOKEN_USERNAME`, `REPO_DEPLOY_TOKEN_PASSWORD`
	- Used to build the authenticated Git URL for `task.set_repo(...)`.
	- Keep these secret; do not commit `.env`.
- `FIFTYONE_API_URI`
	- The base Docker image / container uses this to talk to the FiftyOne API.
	- The provided [Dockerfile](Dockerfile) sets a default:
		- `https://fiftyone-api.int.svc.bernard-gruppe.com`
- `FIFTYONE_API_KEY`
	- API key used by the FiftyOne Python client to authenticate against the Bernard FiftyOne server.
	- Provide it via `.env` so ClearML forwards it into the container/agent runtime.
- (Optional) S3 settings if your dataset filepaths are HTTPS URLs pointing at an S3-compatible endpoint:
	- `S3_ENDPOINT_URL` (default in code: `https://s3.de-west-1.psmanaged.com`)
	- `S3_REGION_NAME` (default in code: `de-west-1`)
	- Plus whatever credentials your environment uses for `boto3` (e.g., `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

## Dataset assumptions (important)

The exporter expects an existing FiftyOne dataset on the **Bernard FiftyOne server**. The dataset loaded depends on `FIFTYONE_DATASET`:

| `FIFTYONE_DATASET`       | FiftyOne dataset name          | Classes                                              |
|--------------------------|--------------------------------|------------------------------------------------------|
| `visdrone_rgbt`          | `visdrone_dronevehicle_rgbt`   | car, truck, bus                                      |
| `rtdod_rgbt`             | `rtdod_rgbt`                   | person, car, motorcycle, bicycle, bus, truck         |
| `m3fd_rgbt`              | `m3fd_rgbt`                    | person, car, bus, motorcycle, truck                  |
| `3d_net_rgbt`            | `3d_net_rgbt`                  | person, car, motorcycle, bicycle, bus, truck         |
| `3d_net_data_pool_rgbt`  | `3d_net_data_pool_rgbt`        | person, car, motorcycle, bicycle, bus, truck         |

To access any of them, the environment running the job (your machine for local runs, or the ClearML Agent container for remote runs) must have the FiftyOne client configured, at minimum:

- `FIFTYONE_API_URI="https://fiftyone-api.int.svc.bernard-gruppe.com"`
- `FIFTYONE_API_KEY="***"`

All datasets share the same FiftyOne field schema (see [dataloader/fiftyone_rgbt_base.py](dataloader/fiftyone_rgbt_base.py)), except VisDrone which uses `gt_bbox_thermal` for bounding boxes:

- RGB image path field: `filepath`
- thermal image path field: `thermal`
- detections field (bboxes): `ground_truth` (VisDrone: `gt_bbox_thermal`)
- split field: `split` with values in `{train,val,test}` (unknown values fall back to `train`)

If the dataset name does not exist (or the API is not reachable / credentials are missing), you’ll see an error like:

> Could not load FiftyOne dataset ‘3d_net_rgbt’

In that case, ensure the dataset exists on the Bernard FiftyOne server and that `FIFTYONE_API_URI` / `FIFTYONE_API_KEY` are set in the job environment.

## Running

### Remote run (ClearML Agent)

In [yolov11n.py](yolov11n.py), set:

- `RUN_REMOTE = True`
- `REMOTE_QUEUE = "<your-clearml-queue>"`

Then run the script once locally:

```bash
python yolov11n.py
```

ClearML will enqueue the task and the process will exit (`exit_process=True`).

## Outputs

By default the script uses:

- Dataset export root: `./workspace/visdrone_dronevehicle_yolo_rgbt`
- Results root: `workspace/.../Results/`

Each run is named like:

- `yolo11n_<clearml_task_id_prefix>`

Artifacts produced include:

- `weights/best.pt`
- `results.csv`
- exported ONNX model
- validation + test evaluation outputs (plots enabled)
- qualitative sample export under:
	- `Results/<run_name>_balanced_samples/{good,bad}/*.png`
	- plus a `report.json`

## Quantization

After training, the exported ONNX model can be compiled to a Hailo `.hef` binary for deployment on Hailo-8L hardware. The quantization pipeline is implemented in a separate repo:

**[bernard-technologies/onnx-2-hef](https://github.com/bernard-technologies/onnx-2-hef?tab=readme-ov-file)**

The [quantize.py](quantize.py) script in this repo launches that pipeline as a ClearML child task. It reads all secrets from `.env` and forwards them into the Docker container automatically.

### Additional `.env` variables required

The onnx-2-hef container needs Azure credentials to access ClearML artifact storage. Add these to your `.env`:

```
AZURE_STORAGE_CONNECTION_STRING=""   # primary blob storage connection string
AZURE_CLEARML_CONNECTION_STRING=""   # connection string for ClearML's artifact container
AZURE_CLEARML_STORAGE_SAS=""         # shared-access signature token for that container
AZURE_STORAGE_ACCOUNT_NAME=""        # storage account name
AZURE_STORAGE_ACCOUNT_KEY=""         # storage account key
```

### Hailo parser end nodes (`END_NODE_NAMES`)

The Hailo ONNX parser needs to know where to cut the model graph. By default it auto-detects end nodes (`-y`), but for YOLO11n/YOLOv8 detection models this picks the wrong nodes — it selects the final concatenated output after DFL decoding, whereas Hailo's `nms_postprocess` requires the **raw conv outputs before DFL decoding**.

Set `END_NODE_NAMES` in `.env` as a space-separated list of ONNX node names. Leave it empty to fall back to Hailo auto-detection.

**Which nodes to use**

The correct end nodes are the 6 raw Conv node names from the detection head — 2 per scale (box distribution + class scores) across 3 detection scales:

| Branch | Scale 0 (stride 8) | Scale 1 (stride 16) | Scale 2 (stride 32) |
|--------|-------------------|--------------------|--------------------|
| Box distribution (cv2) | `/model.23/cv2.0/cv2.0.2/Conv` | `/model.23/cv2.1/cv2.1.2/Conv` | `/model.23/cv2.2/cv2.2.2/Conv` |
| Class scores (cv3) | `/model.23/cv3.0/cv3.0.2/Conv` | `/model.23/cv3.1/cv3.1.2/Conv` | `/model.23/cv3.2/cv3.2.2/Conv` |

These are the **node names**, not output tensor names — i.e. `/model.23/cv2.0/cv2.0.2/Conv`, not `/model.23/cv2.0/cv2.0.2/Conv_output_0`. The DFL's own internal conv (`/model.23/dfl/conv/Conv`) must **not** be included — it is part of the DFL box decoding that Hailo's postprocessor handles itself.

**Finding the correct node names for your model**

Download `best.onnx` from the ClearML training task artifacts, then either:

- Run the helper script:
	```bash
	python find_end_conv_layers.py best.onnx
	```
	This prints the space-separated list of node names ready to paste into `.env`.

- Or open the ONNX in [Netron](https://netron.app), navigate to the `/model.23` detection head, and identify the last `Conv` nodes in the cv2 and cv3 branches before any `Reshape`/`Softmax`/`Sigmoid` operations.

### NMS config (`nms_config.json`)

Required for detection models. Tells the Hailo postprocessor when to suppress overlapping boxes:

```json
{
  "nms_scores_th": 0.3,   // confidence threshold – boxes below this are discarded
  "nms_iou_th": 0.45      // IoU threshold – overlapping boxes above this are merged
}
```

Set `NMS_CONFIG_PATH` in `.env` to point to your config file (default: `yolo11n_nms_config.json`). If the file doesn't exist yet, `quantize.py` will generate one with the defaults above.

### Model script (`MODEL_SCRIPT_PATH`)

Optional Hailo `.alls` model script uploaded as a ClearML artifact and used directly by the converter, skipping auto-generation. Set `MODEL_SCRIPT_PATH` in `.env` to point to your `.alls` file. Leave it empty (the default) to have the converter auto-generate a script from the training task's normalization parameters.

### Normalization parameters (`normalize_mean` / `normalize_std`)

These are set in the `cfg` dict in [yolov11n.py](yolov11n.py) and read by the converter to generate the Hailo model script. They define the normalization that Hailo applies **in hardware** before inference, so they must match whatever preprocessing your model was trained with.

Ultralytics YOLO expects pixels in `[0, 1]` (i.e. raw pixel value ÷ 255). The corresponding Hailo values are:

```python
"normalize_mean": [0.0, 0.0, 0.0],   # no mean subtraction
"normalize_std":  [1.0, 1.0, 1.0],   # Hailo maps std=1.0 → hardware divisor of 256 ≈ /255
```

If you fine-tune with ImageNet-style normalization instead, use:

```python
"normalize_mean": [0.485, 0.456, 0.406],
"normalize_std":  [0.229, 0.224, 0.225],
```

### Quantization / compilation settings

These optional `.env` variables control the onnx-2-hef converter. Defaults match what the converter uses when a variable is unset.

- `OPTIMIZATION_LEVEL` (default: `4`)
	- Hailo quantization optimization level. `-100` disables optimization entirely; `0`–`4` trade off speed vs. accuracy.
- `COMPRESSION_LEVEL` (default: `0`)
	- Percentage of weights compressed to 4-bit. `0` means no compression; `5` compresses the most.
- `COMPILATION_LEVEL` (default: `"max"`)
	- Hailo compiler optimization level. `0`, `1`, `2`, or `"max"`.
- `CALIBRATION_DATA_SIZE` (default: `2048`)
	- Number of images used for quantization calibration. Recommended ≥ 1024.
- `PROFILING_DATA_SIZE` (default: `100`)
	- Number of images used for profiling. Recommended 64–200.
- `SKIP_PROFILING` (default: `"true"`)
	- Set to `"true"` to skip profiling and speed up the conversion at the cost of losing the FPS/latency report.

### Running quantization

**Automatically** — set `AUTO_QUANTIZE="true"` in `.env`. The quantization task is enqueued on the same `CLEARML_QUEUE` immediately after training succeeds, with the training task's ID wired in as the parent automatically.

**Manually** — set `CLEARML_QUANTIZE_TRAINING_TASK_ID` in `.env` to the task ID of the completed training run, then:

```bash
python quantize.py
```

The resulting `.hef`, `.har`, profiling report and evaluation are uploaded as artifacts on the ClearML quantization task.

## Troubleshooting

- **Private image URLs**: if image paths are HTTPS and require S3 credentials, ensure your agent environment has valid `boto3` credentials and (if needed) set `S3_ENDPOINT_URL` / `S3_REGION_NAME`.


