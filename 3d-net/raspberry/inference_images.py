from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface, InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
HEF_PATH = "yolov11n_m3fd_max_quantization_ 200_epoch.hef"
VIS_FOLDER = "./m3fd/Vis"
IR_FOLDER = "./m3fd/Ir"
OUTPUT_FOLDER = "./m3fd/Results"
MAX_IMAGES = 50
CLASS_NAMES = ("person", "car", "bus", "motorcycle", "truck")
CONF_THRESHOLD = 0.2   # must match the NMS threshold baked into the HEF

CLASS_COLORS = {
    "person":     (0,  255,   0),   # green
    "car":        (255, 128,  0),   # orange
    "bus":        (0,  128, 255),   # blue
    "motorcycle": (255,   0, 255),  # magenta
    "truck":      (0,  255, 255),   # cyan
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- LOAD HEF ---
hef = HEF(HEF_PATH)

with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # --- VSTREAM INFO ---
    print("\n--- Input VStreams ---")
    for info in hef.get_input_vstream_infos():
        print(f"  {info.name:<40} shape={info.shape} type={info.format.type}")

    print("\n--- Output VStreams ---")
    for info in hef.get_output_vstream_infos():
        print(f"  {info.name:<40} shape={info.shape} type={info.format.type}")

    input_vstream_info = hef.get_input_vstream_infos()[0]
    input_shape = input_vstream_info.shape          # (H, W, C) = (640, 640, 4)
    input_h, input_w = input_shape[0], input_shape[1]
    print(f"\nInput: {input_h}x{input_w}x{input_shape[2]}  (RGBT, UINT8)\n")

    # --- VSTREAM PARAMS: RGBT UINT8 input, FLOAT32 output ---
    input_vstreams_params = InputVStreamParams.make_from_network_group(
        network_group, quantized=True, format_type=FormatType.UINT8
    )
    output_vstreams_params = OutputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=FormatType.FLOAT32
    )

    # --- FIND PAIRED VIS / IR IMAGES ---
    vis_files = set(f for f in os.listdir(VIS_FOLDER) if f.lower().endswith(".png"))
    ir_files  = set(f for f in os.listdir(IR_FOLDER)  if f.lower().endswith(".png"))
    paired_files = sorted(vis_files & ir_files)[:MAX_IMAGES]
    print(f"Processing {len(paired_files)} paired VIS/IR images...\n")

    # --- INFERENCE LOOP ---
    with network_group.activate(network_group_params):
        with InferVStreams(
            network_group, input_vstreams_params, output_vstreams_params
        ) as infer_pipeline:

            for filename in paired_files:
                vis_img = cv2.imread(os.path.join(VIS_FOLDER, filename))
                ir_img  = cv2.imread(os.path.join(IR_FOLDER,  filename), cv2.IMREAD_GRAYSCALE)

                if vis_img is None or ir_img is None:
                    print(f"Skipping {filename}: failed to load")
                    continue

                orig_h, orig_w = vis_img.shape[:2]

                # --- FUSE: RGBT ---
                # Ultralytics trains with RGB; thermal appended as 4th channel.
                rgb_img  = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)    # (H, W, 3)
                ir_ch    = ir_img[:, :, np.newaxis]                    # (H, W, 1)
                rgbt_img = np.concatenate([rgb_img, ir_ch], axis=2)    # (H, W, 4)

                # --- PREPROCESS ---
                img_resized = cv2.resize(rgbt_img, (input_w, input_h)) # (640, 640, 4) uint8
                input_data  = {input_vstream_info.name: np.expand_dims(img_resized, axis=0)}

                # --- INFER ---
                results = infer_pipeline.infer(input_data)

                # --- POSTPROCESS ---
                # Output format: list[batch] -> list[class_idx] -> ndarray(N, 5)
                # Each detection: [y1_norm, x1_norm, y2_norm, x2_norm, score]  (Hailo NMS outputs YXYX)
                # Coordinates are normalized to [0, 1] relative to input size.
                output_key = list(results.keys())[0]
                batch_result = results[output_key][0]   # list of 5 class arrays

                detections = []
                for cls_idx, cls_dets in enumerate(batch_result):
                    if len(cls_dets) == 0:
                        continue
                    label = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                    color = CLASS_COLORS.get(label, (0, 255, 0))
                    for det in cls_dets:
                        y1_n, x1_n, y2_n, x2_n, score = det
                        if score < CONF_THRESHOLD:
                            continue
                        # Scale normalized coords to original image dimensions
                        x1 = int(x1_n * orig_w)
                        y1 = int(y1_n * orig_h)
                        x2 = int(x2_n * orig_w)
                        y2 = int(y2_n * orig_h)
                        detections.append((label, score, x1, y1, x2, y2))
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            vis_img,
                            f"{label} {score:.2f}",
                            (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                        )

                # --- PRINT SUMMARY ---
                if detections:
                    print(f"{filename}: {len(detections)} detection(s)")
                    for label, score, x1, y1, x2, y2 in detections:
                        print(f"  {label} {score:.2f}  [{x1},{y1} -> {x2},{y2}]")
                else:
                    print(f"{filename}: no detections")

                # --- SAVE ---
                out_path = os.path.join(OUTPUT_FOLDER, filename)
                cv2.imwrite(out_path, vis_img)

print(f"\nDone. Results saved to {OUTPUT_FOLDER}/")
