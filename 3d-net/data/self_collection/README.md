# Self-Collected Dataset

Scripts and data for building a paired RGB/Thermal (RGBT) dataset from drone footage captured with the WIRIS Enterprise camera.

## Process Overview

### 1. Record videos

Fly the drone and record synchronized dual streams. Each recording session produces two `.mp4` files with identical stems placed in:

```
video/<session_name>/rgb/<stem>.mp4
video/<session_name>/ir/<stem>.mp4
```

The IR video comes off the camera in **BWRGB pseudocolor** — a 3-channel color-mapped encoding of the raw thermal data using the BWRGB palette.

### 2. Convert IR video to white-hot grayscale

```bash
python bwrgb_plt_to_whitehot.py video/<session>/ir/input.mp4 video/<session>/ir/output_whitehot.mp4
```

Reads `BWRGB.plt` (256 palette entries), builds a full `256×256×256` RGB→temperature LUT via KDTree nearest-neighbour, and maps every pixel back to a linear grayscale intensity (0 = coldest, 255 = hottest). Optionally apply CLAHE contrast enhancement with `--clahe`.

### 3. Align RGB and IR videos

The RGB and IR cameras have different fields of view and their recordings are not frame-locked, so both temporal and spatial alignment are required before frame extraction.

#### 3a. Estimate temporal offset

```bash
python wiris_camera/camera_alignment/temporal_offset_estimation.py \
    video/<session>/ir/output_whitehot.mp4 \
    video/<session>/rgb/input.mp4 \
    --check-at 30.0 \
    --test-offsets 2.5 2.6 2.7 2.8
```

Exports side-by-side JPEG snapshots for each candidate offset so you can visually inspect which one best aligns a scene with clear motion or distinct structure. Identify the best offset value for the next step.

#### 3b. Apply temporal alignment

```bash
python wiris_camera/camera_alignment/temporal_alignment.py \
    video/<session>/rgb/input.mp4 \
    video/<session>/ir/output_whitehot.mp4 \
    --output-visible video/<session>/rgb/aligned.mp4 \
    --output-thermal video/<session>/ir/aligned.mp4 \
    --manual-offset 2.6
```

Uses the offset from step 3a to pair visible frames with the nearest thermal frame on a shared timeline via a sliding window, then writes two temporally synchronized output videos.

#### 3c. Apply spatial alignment

```bash
python wiris_camera/camera_alignment/spatial_overlay.py \
    video/<session>/rgb/aligned.mp4 \
    video/<session>/ir/aligned.mp4 \
    --tx -10 --ty -30
```

Derives the thermal→visible affine transform analytically from camera FOV values (visible: 63.7°×38.6°, thermal: 45.0°×37.0°), applies optional manual `--tx`/`--ty` pixel corrections for lens baseline offset, and writes two videos cropped to the overlapping region. Use `--overlay` to write a single blended verification video instead.

### 4. Extract paired frames

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

### 5. Verify IR images are grayscale

```bash
python check_ir_grayscale.py --ir-dir img/ir
```

Checks every image in the IR directory for the grayscale property (all channels identical). All max channel differences should be 0.

### 6. Upload to S3

Upload `img/rgb/` and `img/ir/` to the S3 bucket under the `self-collected/` prefix:

```
s3://bernard-3d-net/self-collected/rgb/<stem>_<frame_idx>.jpg
s3://bernard-3d-net/self-collected/ir/<stem>_<frame_idx>.jpg
```

### 7. Build FiftyOne dataset

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

### 8. Annotate in CVAT

Export the FiftyOne dataset to CVAT, annotate bounding boxes for the target classes (`car`, `truck`, `bus`, `person`, `motorcycle`, `bicycle`), and import the annotations back into FiftyOne.

### 9. Merge into the data pool

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
| `wiris_camera/camera_alignment/temporal_offset_estimation.py` | Visually estimate the temporal offset between RGB and IR streams |
| `wiris_camera/camera_alignment/temporal_alignment.py` | Apply the temporal offset to produce synchronized video pairs |
| `wiris_camera/camera_alignment/spatial_overlay.py` | Warp thermal frames into visible image space using FOV-derived transform |
