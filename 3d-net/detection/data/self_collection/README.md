# Self-Collected Dataset

Scripts and data for building a paired RGB/Thermal (RGBT) dataset from drone footage captured with the WIRIS Enterprise camera.

## Process Overview

### 1. Record aligned videos

Record with the live capture of the [inference service](../../inference_service/README.md). Set in `inference_service/config.yaml`:

```yaml
provider:
  mode: live
service:
  record_only: true   # record without running the model
capture:
  enabled: true
```

Then start the service on the Raspberry Pi from `detection/inference_service/`:

```bash
docker compose run inference-service
```

For every visible frame, the service takes the thermal frame closest in time (after `camera.temporal_offset_s`), warps it into the visible image space with the FOV-based transform and the `alignment.tx`/`alignment.ty` corrections, and crops both frames to the overlapping region. Each session writes a temporally and spatially aligned video pair, named after the UTC start time:

```
inference_service/recordings/<YYYYMMDD_HHMMSS>/vis.mp4
inference_service/recordings/<YYYYMMDD_HHMMSS>/ir.mp4
```

The thermal frames are reduced to grayscale by a plain luminance conversion, so set the thermal stream of the camera to the white-hot palette before recording.

### 2. Arrange the recordings

`extract_video_samples.py` pairs the videos by sorted order and names the frames after the RGB video stem. Copy each recorded pair into a session folder with identical stems that follow the filename convention of step 6:

```
video/<location>_<YY>_<MM>_<DD>/rgb/<location>_<YY>_<MM>_<DD>_<session>.mp4   (vis.mp4)
video/<location>_<YY>_<MM>_<DD>/ir/<location>_<YY>_<MM>_<DD>_<session>.mp4    (ir.mp4)
e.g. video/hausern_05_05_26/rgb/hausern_05_05_26_aufnahme1.mp4
```

### 3. Extract paired frames

```bash
python extract_video_samples.py \
    --source video/<session_name> \
    --output img \
    --interval 3
```

Expects the aligned videos in `rgb/` and `ir/` subfolders of `--source`. Extracts every Nth frame and writes to:

```
img/rgb/<stem>_<frame_idx:06d>.jpg
img/ir/<stem>_<frame_idx:06d>.jpg
```

IR frames are converted to single-channel grayscale before saving to eliminate YUV→BGR rounding artifacts from H.264 decoding.

### 4. Verify IR images are grayscale

```bash
python check_ir_grayscale.py --ir-dir img/ir
```

Checks every image in the IR directory for the grayscale property (all channels identical). All max channel differences should be 0.

### 5. Upload to S3

Upload `img/rgb/` and `img/ir/` to the S3 bucket under the `self-collected/` prefix:

```
s3://bernard-3d-net/self-collected/rgb/<stem>_<frame_idx>.jpg
s3://bernard-3d-net/self-collected/ir/<stem>_<frame_idx>.jpg
```

### 6. Build FiftyOne dataset

```bash
python detection/fiftyone/dataset_builder/create_fiftyone_self_collected.py \
    --name 3d_net_rgbt \
    --s3-bucket bernard-3d-net \
    --s3-prefix self-collected
```

Pairs images by filename stem across the S3 `rgb/` and `ir/` prefixes and registers them as a FiftyOne dataset. Running it again is safe — new pairs are added incrementally and existing samples (including CVAT annotations) are left untouched.

Filename convention encodes metadata:
```
{location}_{YY}_{MM}_{DD}_{session}_{frame_index:06d}.jpg
e.g. hausern_20_01_26_aufnahme5_000123.jpg
```

### 7. Annotate in CVAT

Export the FiftyOne dataset to CVAT, annotate bounding boxes for the target classes (`car`, `truck`, `bus`, `person`, `motorcycle`, `bicycle`), and import the annotations back into FiftyOne.

### 8. Merge into the data pool

Run `detection/fiftyone/merge_fiftyone_datapool.py` to merge the annotated self-collected dataset into the shared RGBT data pool used for training.

## Viewing Pairs Locally

```bash
python view_image_pairs_overlay.py
```

Opens an interactive viewer showing the RGB+Thermal overlay (top) and each channel individually (bottom). Navigate with Space/Arrow keys. Optionally pass `--annotation-dir` to overlay Pascal VOC XML bounding boxes.

## File Reference

| File | Purpose |
|------|---------|
| `BWRGB.plt` | Palette file (256 RGB entries) mapping thermal intensity to color |
| `bwrgb_plt_to_whitehot.py` | Convert BWRGB pseudocolor IR video → white-hot grayscale video |
| `extract_video_samples.py` | Extract paired RGB/IR frames from aligned videos at a fixed interval |
| `check_ir_grayscale.py` | Verify extracted IR images are truly single-channel grayscale |
| `view_image_pairs_overlay.py` | Local viewer for RGB/IR image pairs with optional annotation overlay |
| `img/rgb/` | Extracted RGB frames (gitignored) |
| `img/ir/` | Extracted grayscale IR frames (gitignored) |
| `video/` | Source video recordings (gitignored) |
