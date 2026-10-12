import os

import numpy as np
from clearml import Task
import fiftyone as fo
from ast import literal_eval

from swarm_inference.model.model import load_model_from_clearml
from swarm_inference.model.model_hailo import (
    load_hailo_model,
    hailo_model_inference,
    InferContext,
)
from swarm_inference.postprocessing.postprocessing import postprocess

from util import read_nms_parameters, normalize_box


def evaluate_model(
    model_file: str,
    profiling_data_np: np.ndarray,
    profiling_data_fo: fo.Dataset,
    parent_task: Task,
):
    classes = literal_eval(parent_task.get_parameter("data/classes"))

    model_config = load_model_from_clearml(parent_task.id)
    transformed_input = profiling_data_np.transpose((0, 3, 1, 2))
    model = load_hailo_model(model_file)
    quantized_res = hailo_model_inference(
        model,
        transformed_input,
        InferContext.QUANTIZED,
        model_config,
    )

    eval_name = parent_task.id[0:4] + "_" + os.getenv("CLEARML_TASK_ID")[0:4]

    if "classification" in parent_task.get_tags():
        # deal with quality head
        if type(quantized_res) == list:
            for i in range(len(quantized_res)):
                if quantized_res[i].shape[-1] == len(classes):
                    quantized_res = quantized_res[i]
                    break

        eval_classification(
            profiling_data_fo,
            quantized_res,
            classes=classes,
            eval_name=eval_name,
        )
    elif "detection" in parent_task.get_tags():
        post_processed = []
        score_threshold, iou_threshold = read_nms_parameters()
        model_config["inference_platform"] = "hailo"
        for i in range(len(quantized_res)):
            (bboxes, labels), _ = postprocess(
                [quantized_res],
                model_config,
                i,
                confidence_threshold=score_threshold,
                iou_threshold=iou_threshold,
            )
            post_processed.append((bboxes, labels))

        eval_detection(
            profiling_data_fo,
            post_processed,
            classes=classes,
            eval_name=eval_name,
            model_config=model_config,
        )
    print("📝 evaluation done, saved to /evaluation.txt")


def eval_classification(fo_dataset_view, inference_result, classes=None, eval_name=""):
    for sample, result in zip(fo_dataset_view, inference_result):
        sample[f"model_{eval_name}_quantized"] = fo.Classification(
            label=classes[np.argmax(result)], confidence=np.max(result)
        )
        sample.save()

    # we selected only gt fields - before eval we need to reselect new field
    dataset_name, snapshot_date = os.getenv("DATASET").split(":")
    fo_dataset_view = fo.load_dataset(dataset_name)[fo_dataset_view]
    fo_dataset_view = fo_dataset_view.select_fields(
        ["ground_truth", f"model_{eval_name}_quantized"]
    )

    eval_result = fo_dataset_view.evaluate_classifications(
        f"model_{eval_name}_quantized",
        gt_field="ground_truth",
        classes=classes,
    )
    print("evaluation:")
    print(eval_result.print_report())

    # print to file
    with open(f"/evaluation_quantized.txt", "w") as text_file:
        print(eval_result.report(), file=text_file)


def eval_detection(
    fo_dataset_view, inference_results, classes=None, eval_name="", model_config=None
):
    for sample, sample_results in zip(fo_dataset_view, inference_results):
        detections = []
        for box, label in zip(sample_results[0], sample_results[1]):
            detections.append(
                fo.Detection(
                    label=classes[label],
                    confidence=box[4],
                    bounding_box=normalize_box(box[0:4], model_config),
                )
            )

        sample[f"model_{eval_name}_quantized"] = fo.Detections(detections=detections)
        sample.save()
    print("detections added to fiftyone, starting evaluation")

    # we selected only gt fields - before eval we need to reselect new field
    dataset_name, snapshot_date = os.getenv("DATASET").split(":")
    fo_dataset = fo.load_dataset(dataset_name)[fo_dataset_view]
    fo_dataset_view = fo_dataset.select_fields(
        ["ground_truth", f"model_{eval_name}_quantized"]
    )

    eval_result = fo_dataset_view.evaluate_detections(
        f"model_{eval_name}_quantized",
        gt_field="ground_truth",
        classes=classes,
    )
    print("evaluation:")
    print(eval_result.print_report())

    # print to file
    with open(f"/evaluation_quantized.txt", "w") as text_file:
        print(eval_result.report(), file=text_file)
