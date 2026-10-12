import json
import os

import onnx
import numpy as np
import fiftyone as fo
from fiftyone import ViewField as F
import cv2
from swarm_inference.preprocessing.preprocessing import get_img_size


def preprocess(image, output_height, output_width, output_channels, image_thermal=None):
    if output_channels == 4:
        if image_thermal is None:
            raise ValueError(
                "thermal is required for 4-channel (RGBT) models but was not provided."
            )
        rgb = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        thermal = cv2.resize(image_thermal, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        if thermal.ndim == 3:
            if not np.array_equal(thermal[:, :, 0], thermal[:, :, 1]) or not np.array_equal(thermal[:, :, 0], thermal[:, :, 2]):
                raise ValueError("Thermal image has 3 channels but they are not identical — expected grayscale stored as RGB.")
            thermal = thermal[:, :, 0:1]
        else:
            thermal = thermal[:, :, np.newaxis]
        return np.concatenate([rgb, thermal], axis=-1)  # (H, W, 4)

    if output_channels == 1:
        # model expects only 1 channel as input
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 2 or output_channels == 1:
        # for grayscale images, expand dimensions to make it 3D
        image = np.expand_dims(image, -1)
    resized_image = cv2.resize(
        image, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    if output_channels == 1 and resized_image.ndim == 2:
        # cv2.resize can drop the singleton channel for grayscale data.
        resized_image = np.expand_dims(resized_image, -1)
    return resized_image


def batch_data(view: fo.DatasetView, image_height: int, image_width: int, image_channels: int, calibration_size: int):
    dataset = np.zeros((calibration_size, image_height, image_width, image_channels))

    for idx, sample in enumerate(view):
        sample: fo.Sample = sample
        img = cv2.imread(sample.local_path)
        thermal = None

        if image_channels == 4:
            thermal = cv2.imread(sample.get_local_path("thermal"), cv2.IMREAD_UNCHANGED)

        dataset[idx, :, :, :] = preprocess(
            img,
            output_height=image_height,
            output_width=image_width,
            output_channels=image_channels,
            image_thermal=thermal,
        )
    return dataset


def read_calibration_dataset(
    dataset, image_height, image_width, image_channels, calibration_size=2048
):
    if ":" not in dataset:
        raise ValueError(
            "❌️ $DATASET should be in the format <dataset_name>:<snapshot_date>, e.g. classification:2025-01-15"
        )

    dataset_name, snapshot_date = dataset.split(":")
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"❌️ Dataset {dataset} not found in FiftyOne.")

    dataset: fo.Dataset = fo.load_dataset(dataset_name, snapshot=snapshot_date)
    dataset: fo.DatasetView = dataset.take(calibration_size, seed=51)

    if image_channels == 4:
        if "thermal" not in dataset.get_field_schema():
            raise ValueError(
                f"❌️ Dataset {dataset_name} does not contain a 'thermal' field required for 4-channel (RGBT) input."
            )
        view: fo.DatasetView = dataset.select_fields("ground_truth", "thermal")
        view.download_media("thermal", progress=True)
    else:
        view: fo.DatasetView = dataset.select_fields("ground_truth")

    _log_calibration_stats(view)

    return batch_data(view, image_height, image_width, image_channels, calibration_size)


def _log_calibration_stats(view: fo.DatasetView):
    split_counts = {}
    class_counts = {}

    for sample in view:
        for tag in sample.tags:
            if tag.startswith("target:"):
                split = tag[len("target:"):]
                split_counts[split] = split_counts.get(split, 0) + 1

        if sample.ground_truth is not None and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                class_counts[det.label] = class_counts.get(det.label, 0) + 1

    print(f"📊 Calibration dataset stats ({len(view)} images total):")
    if split_counts:
        for split, count in sorted(split_counts.items()):
            print(f"  {split}: {count} images")
    else:
        print("  (no target: split tags found)")

    if class_counts:
        print("  Class instances:")
        for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")
    else:
        print("  (no class annotations found)")


def read_test_data(dataset, image_height, image_width, image_channels, classes=None, size=None):
    dataset_name, _ = dataset.split(
        ":"
    )  # don't read snapshot because they are read only
    dataset = fo.load_dataset(dataset_name)
    test_view: fo.DatasetView = dataset.filter_labels("ground_truth", F("label").is_in(classes))
    test_view: fo.DatasetView = test_view.match_tags("target:test")
    if size is not None:
        test_view = test_view.take(size, seed=51)
    print(f"Found {len(test_view)} test samples in dataset {dataset_name}.")

    if image_channels == 4:
        if "thermal" not in dataset.get_field_schema():
            raise ValueError(
                f"❌️ Dataset {dataset_name} does not contain a 'thermal' field required for 4-channel (RGBT) input."
            )
        test_view: fo.DatasetView = test_view.select_fields(["ground_truth", "thermal"])
        test_view.download_media("thermal", progress=True)
    else:
        test_view: fo.DatasetView = test_view.select_fields("ground_truth")

    profiling_set = batch_data(
        test_view, image_height, image_width, image_channels, len(test_view)
    )
    return profiling_set, test_view


def get_input_shape(model_file):
    model = onnx.load(model_file)
    input_tensor = model.graph.input[0]
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_param:  # symbolic dimension
            shape.append(dim.dim_param)
        else:
            shape.append(dim.dim_value)

    if len(shape) == 4:
        return shape[-2], shape[-1], shape[1]
    elif len(shape) == 3:
        return shape[-2], shape[-1], shape[0]
    else:
        raise ValueError(f"Unsupported input rank {len(shape)} for shape {shape}")


def read_nms_parameters():
    with open("/nms_config.json", "r") as f:
        config = json.load(f)
        return config["nms_scores_th"], config["nms_iou_th"]


def normalize_box(box, model_config):
    img_h, img_w = get_img_size(
        model_config["preprocessing"]["channels"],
        model_config["preprocessing"]["input-shape"],
    )

    return [
        box[1] / img_h,
        box[0] / img_w,
        (box[3] - box[1]) / img_h,
        (box[2] - box[0]) / img_w,
    ]


def build_model_script(mean, std, image_w):

    OPTIMIZATION_LEVEL = os.getenv("OPTIMIZATION_LEVEL")
    COMPRESSION_LEVEL = os.getenv("COMPRESSION_LEVEL")
    COMPILATION_LEVEL = os.getenv("COMPILATION_LEVEL")
    CALIBRATION_SIZE = int(os.getenv("CALIBRATION_DATA_SIZE"))

    model_script = [
        f"normalization1 = normalization({[m * 256. for m in mean]}, {[s * 256. for s in std]})",
        f"model_optimization_flavor(optimization_level={OPTIMIZATION_LEVEL}, compression_level={COMPRESSION_LEVEL})",
        f"model_optimization_config(calibration, calibset_size={CALIBRATION_SIZE})",
        f"performance_param(compiler_optimization_level={COMPILATION_LEVEL})",
    ]

    if int(OPTIMIZATION_LEVEL) >= 4:
        # use full calibration set in production mode
        model_script.append(
            f"post_quantization_optimization(finetune, policy=enabled, dataset_size={CALIBRATION_SIZE})"
        )
        if image_w > 512:
            # reduce batch size to reduce memory consumption on larger models
            model_script.append(
                "post_quantization_optimization(adaround, policy=enabled, batch_size=8, cache_compression=enabled)"
            )
        else:
            model_script.append(
                "post_quantization_optimization(adaround, policy=enabled)"
            )

    if os.path.exists("/nms_config.json"):
        with open("/nms_config.json") as f:
            nms_cfg = json.load(f)
        # Re-apply sigmoid to class score outputs — the model is parsed ending at raw conv layers
        # (before sigmoid), so we must add it back explicitly for nms_postprocess to work correctly.
        for decoder in nms_cfg.get("bbox_decoders", []):
            model_script.append(f"change_output_activation({decoder['cls_layer']}, sigmoid)")
            if "precision_mode" in decoder:
                model_script.append(
                    f"quantization_param({decoder['cls_layer']}, precision_mode={decoder['precision_mode']})"
                )
        model_script.append(
            "nms_postprocess('/nms_config.json', meta_arch=yolov8, bbox_decoding_only=False)"
        )

    with open("/model_script.alls", "w", encoding="utf-8") as f:
        f.write("\n".join(model_script))
