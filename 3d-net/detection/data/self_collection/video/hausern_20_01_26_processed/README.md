# hausern_20_01_26 — Processed Videos

Videos in this directory have been processed through the following pipeline:

1. **Thermal color conversion** — thermal frames converted from BWRGB palette to white-hot grayscale without CLAHE contrast enhancement (`bwrgb_to_whitehot_lut.py`)
2. **Temporal alignment** — RGB and thermal streams aligned in time to compensate for capture offset (`temporal_alignment.py`)
3. **Spatial alignment + crop** — thermal frames warped into RGB image space via FOV-derived affine transform, then both streams cropped to the intersection region where both modalities overlap (`spatial_overlay.py`)

Only aligned pixels present in both the RGB and thermal modality after the spatial transform are retained.
