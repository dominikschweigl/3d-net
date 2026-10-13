"""
Systematically test different preprocessing combinations to find what produces detections.
"""
from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface, InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType
import cv2
import numpy as np
import os

HEF_PATH = "model_data_pool.hef"
VIS_FOLDER = "../data/m3fd/Vis"
IR_FOLDER  = "../data/m3fd/Ir"
N_IMAGES = 50

hef = HEF(HEF_PATH)

vis_files = sorted(f for f in os.listdir(VIS_FOLDER) if f.endswith(".png"))
ir_files  = set(os.listdir(IR_FOLDER))
paired    = [f for f in vis_files if f in ir_files][:N_IMAGES]

input_vstream_info = hef.get_input_vstream_infos()[0]
input_shape = input_vstream_info.shape
input_h, input_w = input_shape[0], input_shape[1]

combos = [
    # (label, quantized_input, format_type, normalize_fn, channel_order)
    ("BGRT_uint8",  True,  FormatType.UINT8,   lambda x: x.astype(np.uint8),                 "bgrt"),
    ("RGBT_uint8",  True,  FormatType.UINT8,   lambda x: x.astype(np.uint8),                 "rgbt"),
    ("BGRT_f32_01", False, FormatType.FLOAT32, lambda x: x.astype(np.float32) / 255.0,       "bgrt"),
    ("RGBT_f32_01", False, FormatType.FLOAT32, lambda x: x.astype(np.float32) / 255.0,       "rgbt"),
    ("BGRT_f32_255",False, FormatType.FLOAT32, lambda x: x.astype(np.float32),               "bgrt"),
    ("RGBT_f32_255",False, FormatType.FLOAT32, lambda x: x.astype(np.float32),               "rgbt"),
]

with VDevice() as target:
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = target.configure(hef, cfg)[0]
    ng_params = ng.create_params()

    for label, quant, fmt, norm_fn, ch_order in combos:
        print(f"\n{'='*60}")
        print(f"COMBO: {label}  (quantized={quant}, format={fmt})")
        print(f"{'='*60}")

        in_params = InputVStreamParams.make_from_network_group(ng, quantized=quant, format_type=fmt)
        out_params = OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.FLOAT32)

        total_dets = 0
        with ng.activate(ng_params):
            with InferVStreams(ng, in_params, out_params) as pipe:
                for fname in paired:
                    vis = cv2.imread(os.path.join(VIS_FOLDER, fname))
                    ir  = cv2.imread(os.path.join(IR_FOLDER,  fname), cv2.IMREAD_GRAYSCALE)
                    if vis is None or ir is None:
                        continue

                    ir_ch = ir[:, :, np.newaxis]
                    if ch_order == "bgrt":
                        fused = np.concatenate([vis, ir_ch], axis=2)       # (H,W,4) BGRT
                    else:
                        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                        fused = np.concatenate([rgb, ir_ch], axis=2)       # (H,W,4) RGBT

                    resized = cv2.resize(fused, (input_w, input_h))
                    arr = norm_fn(resized)
                    batch = np.expand_dims(arr, axis=0)

                    results = pipe.infer({input_vstream_info.name: batch})

                    for key, val in results.items():
                        if isinstance(val, list) and len(val) > 0:
                            batch_res = val[0]
                            if isinstance(batch_res, list):
                                img_dets = sum(len(c) for c in batch_res)
                                if img_dets > 0:
                                    print(f"  {fname}: {img_dets} detection(s)")
                                    for cls_i, c in enumerate(batch_res):
                                        if len(c) > 0:
                                            print(f"    class[{cls_i}]: {c[:3].round(2)}")
                                total_dets += img_dets

        print(f"  -> Total detections across {len(paired)} images: {total_dets}")

print("\nDone.")
