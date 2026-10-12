"""
validate_size_bins.py — Evaluate all YOLO models in pt/ on a FiftyOne dataset
and plot mAP broken down by COCO object-size categories.

Edit the CONSTANTS section and the ClearML settings below to configure the run.
"""

# ============================================================
# ClearML Task Initialization
# --------------------------------------------------------
# Must be at the top before any other imports

import os
import pathlib
from clearml import Logger, Task
from dotenv import load_dotenv, dotenv_values

load_dotenv()
env_variables = dotenv_values(".env")

# ============================================================
# CONSTANTS — edit here to configure the run
# ============================================================

# Dataset to validate on.  Options:
#   m3fd_rgbt | rtdod_rgbt | visdrone_rgbt | 3d_net_rgbt | 3d_net_data_pool_rgbt
DATASET: str = "3d_net_rgbt"

# Directory containing *.pt / *.ptrom model files (all will be evaluated).
PT_DIR_STR: str = "pt"

# Where the exported YOLO dataset will be written (reused on subsequent runs).
WORKSPACE_DIR_STR: str = "workspace/size_bins_eval"

# Export mode — "vis_ir" matches the training pipeline (separate RGB + thermal JPGs).
EXPORT_MODE: str = "vis_ir"

# Prediction settings (low conf builds a full PR curve; NMS IoU matches training).
CONF_THRESH: float = 0.001
IOU_THRESH:  float = 0.7

# Output chart path (relative to repo root).
OUT_PNG_STR: str = "size_bins_performance.png"

# COCO object-size thresholds (pixel area = width_px × height_px).
SIZE_BINS: dict = {
    "Small":  (0.0,    32 ** 2),          # area < 1 024 px²
    "Medium": (32 ** 2, 96 ** 2),         # 1 024 – 9 216 px²
    "Large":  (96 ** 2, float("inf")),    # > 9 216 px²
}

# ClearML / execution settings
CLEARML_PROJECT  = "YOLO11-3DNet"
CLEARML_TASKNAME = "yolo11n_validate_size_bins"
RUN_REMOTE       = True                 # set False to run locally
REMOTE_QUEUE     = os.getenv("CLEARML_QUEUE", "training-multi")

# ============================================================
# ClearML task init
# ============================================================

task: Task = Task.init(
    project_name=CLEARML_PROJECT,
    task_name=CLEARML_TASKNAME,
    task_type=Task.TaskTypes.testing,
    auto_connect_frameworks={"pytorch": False, "matplotlib": False},
)

task.add_tags(["detection", "validation", "size-bins", DATASET])

deploy_token_user = env_variables.get("REPO_DEPLOY_TOKEN_USERNAME", "")
deploy_token_pass = env_variables.get("REPO_DEPLOY_TOKEN_PASSWORD", "")

task.set_repo(
    f"https://{deploy_token_user}:{deploy_token_pass}@github.com/bernard-technologies/3d-net-clearml.git",
    branch="main",
)

task.set_base_docker(
    docker_image="swarm.azurecr.io/swarm-dnn-nextgen:latest-cuda-122",
    docker_arguments="".join([f' -e {key}="{value}"' for key, value in env_variables.items()]),
    docker_setup_bash_script=["apt-get install -y --no-install-recommends libgl1"],
)

# Upload every model file in pt/ (including one level of subfolders) as a ClearML
# artifact.  Subfolder names encode the dataset and are preserved in the key using
# "__" as a path separator (e.g. "3d_net__early.pt").
_pt_dir = pathlib.Path(PT_DIR_STR)
_model_files = (
    sorted(_pt_dir.glob("*.pt"))   + sorted(_pt_dir.glob("*.ptrom")) +
    sorted(_pt_dir.glob("*/*.pt")) + sorted(_pt_dir.glob("*/*.ptrom"))
)
for _mp in _model_files:
    _rel = _mp.relative_to(_pt_dir)
    _key = str(_rel).replace("\\", "/").replace("/", "__")   # "ds__model.pt" or "model.pt"
    task.upload_artifact(_key, artifact_object=str(_mp))
    print(f"Uploaded artifact: {_key}")

if RUN_REMOTE:
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

# ============================================================
# Light imports (numpy, matplotlib — available everywhere)
# ============================================================

import numpy as np  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

# ============================================================
# Colour palette (mirrors the reference plotting script)
# ============================================================

METHOD_COLORS: dict = {
    "RGB":     "#4C78A8",
    "Thermal": "#F58518",
    "Early":   "#54A24B",
    "Mid":     "#E45756",
    "MidP3":   "#72B7B2",
    "MidCMA3": "#B279A2",
    "MidCFT3": "#9D755D",
    "Late":    "#EECA3B",
    "Score":   "#BAB0AC",
}

_FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def darken(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_bar(ax, x, height, color, edge, hatch=None, alpha=1.0, width=0.1, zorder=3):
    ax.bar(
        x, height, width,
        color=color, edgecolor=edge, linewidth=0.8,
        hatch=hatch, alpha=alpha, zorder=zorder,
    )


_DATASET_DISPLAY: dict = {
    "3d_net":           "3D-Net",
    "3d_net_data_pool": "Data Pool",
}


def extract_model_name(path: pathlib.Path) -> str:
    stem = path.stem.lower()
    # Longer / more specific keywords first to avoid prefix shadowing.
    model_type = path.stem   # fallback: raw stem
    for keyword, display in [
        ("midcma3", "MidCMA3"), ("midcma",  "MidCMA3"),
        ("midcft3", "MidCFT3"), ("midcft",  "MidCFT3"),
        ("midctf3", "MidCFT3"), ("midctf",  "MidCFT3"),
        ("midp3",   "MidP3"),
        ("early",   "Early"),
        ("mid",     "Mid"),
        ("late",    "Late"),
        ("score",   "Score"),
        ("rgb",     "RGB"),
        ("thermal", "Thermal"),
    ]:
        if keyword in stem:
            model_type = display
            break
    # When the model lives in a dataset subfolder, qualify the name so same-type
    # models from different datasets are distinguishable in results and the figure.
    dataset = path.parent.name
    if dataset and dataset != "pt_downloaded":
        label = _DATASET_DISPLAY.get(dataset, dataset)
        return f"{model_type} [{label}]"
    return model_type


def _model_color(name: str, fallback_idx: int) -> str:
    return METHOD_COLORS.get(name, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])


def _extract_type(method_name: str) -> str:
    """'Early [3d_net]' → 'Early',  'Early' → 'Early'."""
    return method_name.split(" [")[0] if " [" in method_name else method_name


def _model_modality(model_name: str) -> str:
    """Return 'rgb', 'thermal', or 'rgbt' based on the display name."""
    t = _extract_type(model_name).lower()
    if t == "rgb":
        return "rgb"
    if t == "thermal":
        return "thermal"
    return "rgbt"


def _model_color_for_method(name: str, methods: list) -> str:
    """Same model type gets the same base colour; later datasets get a lighter shade."""
    model_type = _extract_type(name)
    type_idx   = next(i for i, m in enumerate(methods) if _extract_type(m) == model_type)
    base       = METHOD_COLORS.get(model_type, _FALLBACK_COLORS[type_idx % len(_FALLBACK_COLORS)])
    same_type  = [m for m in methods if _extract_type(m) == model_type]
    ds_idx     = same_type.index(name)
    if ds_idx == 0:
        return base
    if ds_idx == 1:
        return lighten(base, 0.45)
    return lighten(base, 0.70)


# ============================================================
# Ground-truth loading  (PIL + yaml only)
# ============================================================

def load_gt(dataset_yaml: pathlib.Path) -> dict:
    """
    Returns {img_stem: [(cls_id, x1, y1, x2, y2, area_px), ...]}.

    In vis_ir mode the label .txt files are co-located with the visible images
    (dataset.yaml val key → visible/val/).  In tiff/rgb/thermal mode they live
    in a separate labels/val/ tree.
    """
    cfg      = yaml.safe_load(dataset_yaml.read_text())
    ds_root  = pathlib.Path(cfg["path"])
    val_dir  = ds_root / cfg["val"]           # e.g. .../visible/val

    lbl_dir  = val_dir if "visible" in cfg["val"] else ds_root / "labels" / "val"

    _img_exts = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
    gt: dict = {}

    for txt_path in sorted(lbl_dir.glob("*.txt")):
        stem = txt_path.stem
        img_path = next(
            (
                p
                for d in (val_dir, lbl_dir)
                for ext in _img_exts
                if (p := d / f"{stem}{ext}").exists()
            ),
            None,
        )
        if img_path is None:
            print(f"  Warning: no image found for label {txt_path.name}, skipping.")
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        boxes = []
        for line in txt_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            area_px = w * img_w * h * img_h
            boxes.append((cls_id, x1, y1, x2, y2, area_px))

        gt[stem] = boxes

    total_boxes = sum(len(v) for v in gt.values())
    print(f"  Loaded GT: {len(gt)} images, {total_boxes} boxes total.")
    return gt


# ============================================================
# Metric computation  (numpy only)
# ============================================================

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(N, M) IoU matrix via Ultralytics box_iou (lazy torch import)."""
    import torch
    from ultralytics.utils.metrics import box_iou
    return box_iou(
        torch.from_numpy(boxes_a),
        torch.from_numpy(boxes_b),
    ).numpy()


def _ap_101(tp_flags: list, n_gt: int) -> float:
    """AP via Ultralytics compute_ap (lazy import, guaranteed identical results)."""
    if n_gt == 0 or not tp_flags:
        return 0.0
    from ultralytics.utils.metrics import compute_ap
    tp     = np.array(tp_flags, dtype=float)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    prec   = cum_tp / (cum_tp + cum_fp)
    rec    = cum_tp / n_gt
    ap, _, _ = compute_ap(rec, prec)
    return float(ap)


def _map_at(preds: dict, gt: dict, nc: int, size_range: tuple, iou_thresh: float) -> float:
    """
    mAP for one (size bucket, IoU threshold) combination.

    Matching replicates Ultralytics match_predictions exactly:
      1. per-image IoU matrix between ALL GT and ALL predictions
      2. zero out wrong-class pairs
      3. collect all (GT, Det) pairs above iou_thresh
      4. sort by IoU descending, deduplicate per-det then per-GT

    A prediction's bin is determined by its matched GT's area, not its own predicted
    area.  This avoids boundary errors where a correctly-localised box is a few pixels
    wider than the bin edge.  Unmatched predictions fall back to their own area.
    """
    min_a, max_a = size_range

    n_gt_per_cls: np.ndarray = np.zeros(nc, dtype=np.int64)
    all_confs:    list = []
    all_pred_cls: list = []
    all_tp:       list = []

    for stem in set(preds) | set(gt):
        img_gt    = gt.get(stem, [])
        img_preds = preds.get(stem, [])

        # Build full GT arrays — keep all objects, tag which fall in this size bin.
        if img_gt:
            gt_cls_all   = np.array([r[0]          for r in img_gt], dtype=np.int32)
            gt_boxes_all = np.array([[r[1], r[2], r[3], r[4]] for r in img_gt], dtype=np.float32)
            gt_area_all  = np.array([r[5]          for r in img_gt], dtype=np.float32)
            in_bin       = (gt_area_all >= min_a) & (gt_area_all < max_a)
        else:
            gt_cls_all   = np.zeros(0, dtype=np.int32)
            gt_boxes_all = np.zeros((0, 4), dtype=np.float32)
            in_bin       = np.zeros(0, dtype=bool)

        for c in range(nc):
            n_gt_per_cls[c] += int(np.sum(in_bin & (gt_cls_all == c)))

        if not img_preds:
            continue

        pred_cls_arr  = np.array([r[0]             for r in img_preds], dtype=np.int32)
        pred_conf_arr = np.array([r[1]             for r in img_preds], dtype=np.float32)
        pred_boxes    = np.array([[r[2], r[3], r[4], r[5]] for r in img_preds], dtype=np.float32)
        n_det         = len(img_preds)

        # Match ALL predictions against ALL GT (boundary-safe: a prediction matched to a
        # GT outside this bin is excluded rather than mis-counted as FP in this bin).
        matched_gt_for_pred = np.full(n_det, -1, dtype=np.int32)
        if len(gt_boxes_all) > 0:
            iou_mat       = _iou_matrix(gt_boxes_all, pred_boxes)       # (n_gt_all, n_det)
            correct_class = gt_cls_all[:, None] == pred_cls_arr[None, :]
            iou_mat       = iou_mat * correct_class

            matches = np.argwhere(iou_mat >= iou_thresh)
            if matches.shape[0] > 0:
                if matches.shape[0] > 1:
                    order   = iou_mat[matches[:, 0], matches[:, 1]].argsort()[::-1]
                    matches = matches[order]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # one GT per det
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # one det per GT
                matched_gt_for_pred[matches[:, 1].astype(int)] = matches[:, 0].astype(int)

        # A prediction belongs to this bin if:
        #   - matched to an in-bin GT  → TP
        #   - unmatched AND own area in-bin → FP
        # Predictions matched to out-of-bin GT are excluded (correctly detected another size).
        pred_areas       = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        matched_mask     = matched_gt_for_pred >= 0
        matched_to_inbin = np.zeros(n_det, dtype=bool)
        matched_to_inbin[matched_mask] = in_bin[matched_gt_for_pred[matched_mask]]
        unmatched_in_bin = (~matched_mask) & (pred_areas >= min_a) & (pred_areas < max_a)
        include_mask     = matched_to_inbin | unmatched_in_bin
        tp_arr           = matched_to_inbin   # unmatched_in_bin entries are FP (False)

        all_confs.extend(pred_conf_arr[include_mask].tolist())
        all_pred_cls.extend(pred_cls_arr[include_mask].tolist())
        all_tp.extend(tp_arr[include_mask].tolist())

    if not all_confs:
        return 0.0

    # Global sort by confidence descending before building per-class PR curves
    idx           = np.argsort(-np.array(all_confs))
    all_pred_cls_arr = np.array(all_pred_cls)[idx]
    all_tp_arr       = np.array(all_tp,       dtype=bool)[idx]

    ap_values = []
    for c in range(nc):
        if n_gt_per_cls[c] == 0:
            continue
        cls_mask = all_pred_cls_arr == c
        ap_values.append(_ap_101(all_tp_arr[cls_mask].tolist(), int(n_gt_per_cls[c])))
    return float(np.mean(ap_values)) if ap_values else 0.0


def compute_metrics(preds: dict, gt: dict, nc: int, size_bins: dict) -> dict:
    """Returns {size_name: {"mAP50": float, "mAP5095": float}}."""
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    results: dict = {}
    for size_name, size_range in size_bins.items():
        map50   = _map_at(preds, gt, nc, size_range, 0.5)
        map5095 = float(np.mean([_map_at(preds, gt, nc, size_range, float(t)) for t in iou_thresholds]))
        results[size_name] = {"mAP50": map50, "mAP5095": map5095}
        print(f"    {size_name:8s}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
    return results


# ============================================================
# Figure  (matplotlib + numpy only)
# ============================================================

def make_figure(results: dict, size_names: list, dataset_name: str, out_path: pathlib.Path) -> None:
    raw_methods = list(results.keys())

    _TYPE_ORDER = ["RGB", "Thermal", "Early", "Mid", "MidP3", "MidCMA3", "MidCFT3", "Late", "Score"]

    def _type_rank(m: str) -> int:
        t = _extract_type(m)
        return _TYPE_ORDER.index(t) if t in _TYPE_ORDER else len(_TYPE_ORDER)

    methods = sorted(raw_methods, key=lambda m: (_type_rank(m), m))

    # Build list of (model_type, [method, ...]) groups.
    type_groups: list = []
    for m in methods:
        t = _extract_type(m)
        if type_groups and type_groups[-1][0] == t:
            type_groups[-1][1].append(m)
        else:
            type_groups.append((t, [m]))

    bar_w            = 0.04
    inner_gap        = 0.0
    within_type_gap  = 0.005
    between_type_gap = 0.020
    slot             = 2 * bar_w + inner_gap

    # Compute each method's x-offset relative to the cluster centre.
    offsets: dict = {}
    cur = 0.0
    for _, group in type_groups:
        for mi, method in enumerate(group):
            offsets[method] = cur
            cur += slot + (within_type_gap if mi < len(group) - 1 else 0.0)
        cur += between_type_gap
    total_w = cur - between_type_gap

    # Auto-space size-bin groups so no cluster bleeds into the next.
    spacing = max(1.0, total_w + 0.4)
    x = np.arange(len(size_names)) * spacing

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

    for i, size_name in enumerate(size_names):
        for method in methods:
            color = _model_color_for_method(method, methods)
            edge  = darken(color)
            off   = offsets[method] - total_w / 2
            x50   = x[i] + off + bar_w / 2
            x5095 = x[i] + off + bar_w + inner_gap + bar_w / 2
            vals  = results[method].get(size_name)
            if vals is None:
                continue
            draw_bar(ax, x50,   vals["mAP50"],   color, edge, hatch=None,   width=bar_w)
            draw_bar(ax, x5095, vals["mAP5095"], color, edge, hatch="////", width=bar_w)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5, zorder=0)
    for i in range(len(size_names) - 1):
        ax.axvline(x=(x[i] + x[i + 1]) / 2, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    half = total_w / 2 + 0.15
    ax.set_xticks(x)
    ax.set_xticklabels(size_names)
    ax.set_ylabel("mAP")
    ax.set_xlim(x[0] - half, x[-1] + half)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"YOLO11 fusion methods — mAP per object size  ({dataset_name} val split)")

    method_handles = [
        mpatches.Patch(
            facecolor=_model_color_for_method(m, methods),
            edgecolor=darken(_model_color_for_method(m, methods)),
            label=m,
        )
        for m in methods
    ]
    metric_handles = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="black",             label="mAP@50"),
        mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="////", label="mAP@50-95"),
    ]

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.20)
    fig.legend(
        handles=method_handles, title="Model [dataset]",
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=(len(methods) + 1) // 2, frameon=False, fontsize=8.5, title_fontsize=9,
    )
    ax.legend(handles=metric_handles, loc="upper right", frameon=True, fontsize=9)

    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ============================================================
# Main  (all heavy imports live here so they only run remotely)
# ============================================================

def main() -> None:
    import inspect
    import shutil
    import sys

    # Ensure repo root is in sys.path (needed for dataloader / custom_modules).
    _repo_root = pathlib.Path(__file__).resolve().parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    # ---- Torch compat: drop unsupported weights_only BEFORE importing ultralytics ----
    import torch
    _orig_load = torch.load
    _has_wo    = "weights_only" in inspect.signature(_orig_load).parameters

    def _compat_load(*args, **kwargs):
        if "weights_only" in kwargs and not _has_wo:
            kwargs.pop("weights_only")
        return _orig_load(*args, **kwargs)

    torch.load = _compat_load

    # ---- Register ONNX-compatible modules so RGBT checkpoints load correctly ----
    from custom_modules import CrossMLCAv2Fixed, GPTFixed, onnx_fixed_modules_parse_model
    import ultralytics.nn.modules as _ul_modules
    import ultralytics.nn.tasks   as _ul_tasks

    for _cls in (GPTFixed, CrossMLCAv2Fixed):
        setattr(_ul_modules, _cls.__name__, _cls)
        setattr(_ul_tasks,   _cls.__name__, _cls)
    _ul_tasks.parse_model = onnx_fixed_modules_parse_model(_ul_tasks.parse_model)

    from ultralytics import YOLO

    from test.test_balanced_sample_export import save_balanced_test_samples
    from test.heatmap_utils import HeatmapGenerator

    from dataloader import (
        M3fdRgbtYoloDatasetDataloader,
        RtdodRgbtYoloDatasetDataloader,
        ThreeDNetDataPoolRgbtYoloDatasetDataloader,
        ThreeDNetRgbtYoloDatasetDataloader,
        VisDroneRgbtYoloDatasetDataloader,
    )

    DATASET_LOADERS = {
        "visdrone_rgbt":         VisDroneRgbtYoloDatasetDataloader,
        "rtdod_rgbt":            RtdodRgbtYoloDatasetDataloader,
        "m3fd_rgbt":             M3fdRgbtYoloDatasetDataloader,
        "3d_net_rgbt":           ThreeDNetRgbtYoloDatasetDataloader,
        "3d_net_data_pool_rgbt": ThreeDNetDataPoolRgbtYoloDatasetDataloader,
    }

    # ---- Connect config to ClearML (editable in the UI) ----
    cfg = task.connect(
        {
            "dataset":      DATASET,
            "export_mode":  EXPORT_MODE,
            "conf_thresh":  CONF_THRESH,
            "iou_thresh":   IOU_THRESH,
        },
        name="validation",
    )
    dataset_name = cfg["dataset"]

    # ---- Resolved paths ----
    workspace_dir = _repo_root / WORKSPACE_DIR_STR
    out_png       = _repo_root / OUT_PNG_STR

    # ---- Download model artifacts uploaded before execute_remotely ----
    # Ultralytics only accepts .pt suffix — rename .ptrom files after download.
    local_pt_dir = _repo_root / "pt_downloaded"
    local_pt_dir.mkdir(exist_ok=True)

    model_paths = []
    for artifact_name, artifact in task.artifacts.items():
        if not (artifact_name.endswith(".pt") or artifact_name.endswith(".ptrom")):
            continue
        local_copy = pathlib.Path(artifact.get_local_copy())
        # Decode the "__" separator back into a subfolder so path carries dataset identity.
        # "3d_net__early.pt"  →  dataset_folder="3d_net", file_part="early.pt"
        # "early.pt"          →  dataset_folder="",        file_part="early.pt"
        if "__" in artifact_name:
            dataset_folder, file_part = artifact_name.split("__", 1)
        else:
            dataset_folder, file_part = "", artifact_name
        stem = pathlib.Path(file_part).stem
        sub  = local_pt_dir / dataset_folder if dataset_folder else local_pt_dir
        sub.mkdir(exist_ok=True)
        dest = sub / f"{stem}.pt"              # always .pt so Ultralytics accepts it
        shutil.copy2(str(local_copy), str(dest))
        model_paths.append(dest)

    if not model_paths:
        raise FileNotFoundError("No .pt / .ptrom artifacts found in this ClearML task.")
    model_paths.sort()
    print(f"\nModels to evaluate: {[str(p.relative_to(local_pt_dir)) for p in model_paths]}")

    # ---- Export (or reuse) FiftyOne dataset ----
    print("\n--- Exporting dataset ---")
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASET_LOADERS)}")
    loader = DATASET_LOADERS[dataset_name](
        out_root=workspace_dir,
        recreate=False,
        export_mode=cfg["export_mode"],
        image_ext=".jpg",
    )
    dataset_yaml = loader.prepare()
    print(f"  YAML: {dataset_yaml}")

    # ---- Load ground truth ----
    print("\n--- Loading ground truth ---")
    gt = load_gt(dataset_yaml)

    ds_cfg     = yaml.safe_load(dataset_yaml.read_text())
    nc         = len(ds_cfg["names"])
    size_names = list(SIZE_BINS.keys())

    # ---- Print per-size-bin statistics for the val split ----
    print("\n--- Val-split object-size distribution ---")
    for size_name, (min_a, max_a) in SIZE_BINS.items():
        frames_with_bin: list[str] = []
        obj_count = 0
        for stem, boxes in sorted(gt.items()):
            n = sum(1 for _c, _x1, _y1, _x2, _y2, area in boxes if min_a <= area < max_a)
            if n:
                obj_count += n
                frames_with_bin.append(stem)
        print(f"  {size_name:8s}: {obj_count:5d} objects in {len(frames_with_bin):4d} frames")
        if frames_with_bin:
            print(f"            {', '.join(frames_with_bin)}")

    # Heatmap layer config — mirrors yolov11nrgbt.py _HEATMAP_CFG.
    _HEATMAP_CFG: dict = {
        "mid":     ([[6, 8, 10], [16, 18, 20]], ["rgb", "thermal"]),
        "midP3":   ([[6], [12]],                ["rgb", "thermal"]),
        "midCMA3": ([[6, 18, 26], [12, 20, 28]], ["rgb", "thermal"]),
        "midCTF3": ([[6, 8, 10], [16, 18, 20]], ["rgb", "thermal"]),
        "midCTF2": ([[6, 8, 10], [16, 18, 20]], ["rgb", "thermal"]),
        "midCFT3": ([[6, 17, 24], [12, 19, 26]], ["rgb", "thermal"]),
        "late":    ([[6, 8, 10], [18, 20, 22]], ["rgb", "thermal"]),
        "score":   ([[6, 8, 10], [18, 20, 22]], ["rgb", "thermal"]),
        "early":   ([[6, 8, 10]],               ["backbone"]),
    }
    _NAME_TO_FUSION: dict = {
        "Early":   "early",
        "Mid":     "mid",
        "MidP3":   "midP3",
        "MidCMA3": "midCMA3",
        "MidCFT3": "midCFT3",
        "Late":    "late",
        "Score":   "score",
        "RGB":     "early",     # single-backbone, reuse early heatmap layers
        "Thermal": "early",
    }

    total_samples = int(os.getenv("TEST_SAMPLES", 10))

    # ---- Pre-select a fixed pool of val-split images per size bin ----
    # Done once, model-independent, so every model is evaluated on the same frames.
    print("\n--- Pre-selecting balanced-sample pools (val split) ---")
    import random as _rnd
    _pool_ds_root  = pathlib.Path(ds_cfg["path"])
    _pool_test_sub = ds_cfg.get("val", "images/val")
    _pool_img_dir  = _pool_ds_root / _pool_test_sub
    _pool_is_vir   = "visible" in _pool_test_sub
    _pool_lbl_dir  = _pool_img_dir if _pool_is_vir else _pool_ds_root / "labels" / "val"
    _pool_img_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

    size_pools: dict = {}
    for _sname, (_min_a, _max_a) in SIZE_BINS.items():
        _eligible: list = []
        for _lbl in sorted(_pool_lbl_dir.glob("*.txt")):
            _stem = _lbl.stem
            _img  = next(
                (_pool_img_dir / f"{_stem}{_e}" for _e in _pool_img_exts
                 if (_pool_img_dir / f"{_stem}{_e}").exists()),
                None,
            )
            if _img is None:
                continue
            with Image.open(_img) as _im:
                _iw, _ih = _im.size
            for _line in _lbl.read_text().strip().splitlines():
                _p = _line.split()
                if len(_p) >= 5 and _min_a <= float(_p[3]) * _iw * float(_p[4]) * _ih < _max_a:
                    _eligible.append(_stem)
                    break
        _rng = _rnd.Random(42)
        _rng.shuffle(_eligible)
        size_pools[_sname] = _eligible[:total_samples]
        print(f"  {_sname:8s}: {len(size_pools[_sname])}/{len(_eligible)} val images selected")

    # ---- Helper: balanced samples for one size bin ----
    def _save_balanced_for_size(
        size_name: str,
        stems: list,
        yolo_model: "YOLO",
        heatmap_generator,
        balanced_dir: pathlib.Path,
        model_name: str,
        size_range: tuple,
    ) -> None:
        """
        Run save_balanced_test_samples on the pre-selected fixed pool of stems so
        that every model is evaluated on exactly the same images.
        total_samples is set to len(stems)*2 so the early-stop inside the function
        never fires — every image in the pool is always processed.
        Results are uploaded to ClearML under {model_name}/…/{size_name}/….
        """
        import tempfile

        if not stems:
            print(f"  [balanced/{size_name}] pool is empty, skipping.")
            return

        ds_root   = pathlib.Path(ds_cfg["path"])
        modality  = _model_modality(model_name)
        vis_sub   = ds_cfg.get("val", "visible/val")   # e.g. "visible/val"
        ir_sub    = vis_sub.replace("visible", "infrared")
        vis_dir   = ds_root / vis_sub
        ir_dir    = ds_root / ir_sub
        # Labels are always co-located with visible images in vis_ir export mode.
        lbl_src_dir = vis_dir

        # Source image dir depends on modality.
        if modality == "thermal":
            img_src_dir = ir_dir
        else:
            img_src_dir = vis_dir

        if not img_src_dir.is_dir():
            print(f"  [balanced/{size_name}] val image dir missing: {img_src_dir}")
            return

        _img_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
        print(f"  [balanced/{size_name}] {len(stems)} fixed images.")

        # Build a filtered temp dataset on the same filesystem (fast copy).
        with tempfile.TemporaryDirectory(dir=str(workspace_dir)) as tmp:
            tmp_root = pathlib.Path(tmp)

            if modality == "rgbt":
                # vis_ir layout: save_balanced_test_samples detects "visible" in path
                # and automatically loads the paired infrared image.
                tmp_img_dir = tmp_root / vis_sub
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                tmp_ir_dir  = tmp_root / ir_sub
                tmp_ir_dir.mkdir(parents=True, exist_ok=True)
                tmp_lbl_dir = tmp_img_dir   # labels alongside visible images
                tmp_test_sub = vis_sub
            else:
                # rgb / thermal: use a plain images/val layout so that
                # save_balanced_test_samples doesn't try to load an IR companion.
                tmp_img_dir = tmp_root / "images" / "val"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                tmp_lbl_dir = tmp_root / "labels" / "val"
                tmp_lbl_dir.mkdir(parents=True, exist_ok=True)
                tmp_ir_dir  = None
                tmp_test_sub = "images/val"

            for stem in stems:
                for src in img_src_dir.glob(f"{stem}.*"):
                    if src.suffix.lower() in _img_exts:
                        shutil.copy2(str(src), str(tmp_img_dir / src.name))
                lbl_src = lbl_src_dir / f"{stem}.txt"
                if lbl_src.exists():
                    shutil.copy2(str(lbl_src), str(tmp_lbl_dir / f"{stem}.txt"))
                if modality == "rgbt" and tmp_ir_dir:
                    for src in ir_dir.glob(f"{stem}.*"):
                        if src.suffix.lower() in _img_exts:
                            shutil.copy2(str(src), str(tmp_ir_dir / src.name))

            tmp_channels = 4 if modality == "rgbt" else 3
            tmp_yaml = tmp_root / "dataset.yaml"
            tmp_yaml.write_text(
                f"path: {tmp_root.resolve()}\n"
                f"train: {ds_cfg.get('train', '')}\n"
                f"val: {ds_cfg.get('val', '')}\n"
                f"test: {tmp_test_sub}\n"
                f"nc: {ds_cfg['nc']}\n"
                f"names: {ds_cfg['names']}\n"
                f"channels: {tmp_channels}\n"
            )

            size_out = balanced_dir / size_name
            predict_extra: dict = (
                {"use_simotm": "RGBT", "channels": 4}
                if modality == "rgbt"
                else {"channels": 3}
            )
            save_balanced_test_samples(
                model=yolo_model,
                dataset_yaml=str(tmp_yaml),
                out_dir=str(size_out),
                total_samples=len(stems) * 2,  # disable early-stop; process every image
                conf=0.25,
                iou=cfg["iou_thresh"],
                max_det=300,
                iou_match=0.50,
                seed=13,
                heatmap_generator=heatmap_generator,
                size_range=size_range,
                **predict_extra,
            )

        for category in ("good", "bad"):
            for img_path in sorted((size_out / category).glob("*.png")):
                logger.report_image(
                    title=f"{model_name}/balanced_samples/{size_name}/{category}",
                    series=img_path.stem,
                    iteration=0,
                    image=Image.open(str(img_path)).convert("RGB"),
                )
            heatmap_subdir = size_out / "heatmaps" / category
            for img_path in sorted(heatmap_subdir.glob("*.png")):
                logger.report_image(
                    title=f"{model_name}/heatmaps/{size_name}/{category}",
                    series=img_path.stem,
                    iteration=0,
                    image=Image.open(str(img_path)).convert("RGB"),
                )
        report_json = size_out / "report.json"
        if report_json.is_file():
            task.upload_artifact(
                f"balanced_report_{model_name}_{size_name}", str(report_json)
            )

    # ---- Run predictions, metrics, and qualitative samples for each model ----
    def run_predictions(model: YOLO, model_name: str) -> dict:
        """Returns {img_stem: [(cls_id, conf, x1, y1, x2, y2), ...]}."""
        ds_root   = pathlib.Path(ds_cfg["path"])
        modality  = _model_modality(model_name)
        val_sub   = ds_cfg.get("val", "visible/val")
        if modality == "thermal":
            val_sub = val_sub.replace("visible", "infrared")
        val_dir   = ds_root / val_sub

        predict_kwargs: dict = {"channels": 3} if modality != "rgbt" else {"channels": 4, "use_simotm": "RGBT"}

        results_iter = model.predict(
            source=str(val_dir),
            conf=cfg["conf_thresh"],
            iou=cfg["iou_thresh"],
            stream=True,
            verbose=False,
            **predict_kwargs,
        )
        preds: dict = {}
        for result in results_iter:
            stem = pathlib.Path(result.path).stem
            if result.boxes is None or len(result.boxes) == 0:
                preds[stem] = []
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls  = result.boxes.cls.cpu().numpy().astype(int)
            preds[stem] = [
                (int(cls[i]), float(conf[i]),
                 float(xyxy[i, 0]), float(xyxy[i, 1]),
                 float(xyxy[i, 2]), float(xyxy[i, 3]))
                for i in range(len(cls))
            ]
        total = sum(len(v) for v in preds.values())
        print(f"  Predictions: {len(preds)} images, {total} boxes total.")
        return preds

    logger = Logger.current_logger()
    all_results: dict = {}

    for model_path in model_paths:
        model_name   = extract_model_name(model_path)
        model_type   = _extract_type(model_name)
        fusion_type  = _NAME_TO_FUSION.get(model_type, "early")
        safe_name    = model_name.replace(" [", "_").replace("]", "")
        balanced_dir = workspace_dir / f"balanced_samples_{safe_name}"

        print(f"\n=== {model_name}  ({model_path.name}) ===")

        yolo_model = YOLO(str(model_path))

        print("  Running predictions …")
        preds = run_predictions(yolo_model, model_name)
        print("  Computing per-size metrics …")
        all_results[model_name] = compute_metrics(preds, gt, nc, SIZE_BINS)

        # ---- Qualitative balanced good/bad samples — one set per size bin ----
        print("  Exporting balanced samples (per size bin) …")
        layer_groups, group_labels = _HEATMAP_CFG.get(
            fusion_type, ([[6, 8, 10]], ["backbone"])
        )
        heatmap_generator = None
        try:
            heatmap_generator = HeatmapGenerator(
                model_path=str(model_path),
                device="cuda:0",
                method_name="GradCAMPlusPlus",
                layer_groups=layer_groups,
                group_labels=group_labels,
                conf_threshold=0.2,
                ratio=0.02,
                img_size=640,
            )
        except Exception as exc:
            print(f"  [heatmap] init failed ({exc}); skipping heatmaps.")

        for size_name, size_range in SIZE_BINS.items():
            _save_balanced_for_size(
                size_name=size_name,
                stems=size_pools[size_name],
                yolo_model=yolo_model,
                heatmap_generator=heatmap_generator,
                balanced_dir=balanced_dir,
                model_name=model_name,
                size_range=size_range,
            )

    # ---- Generate figure ----
    print("\n--- Generating figure ---")
    make_figure(all_results, size_names, dataset_name, out_png)

    logger.report_image(
        title="Size Bins Performance",
        series="mAP per object size",
        iteration=0,
        local_path=str(out_png),
    )
    task.upload_artifact("size_bins_performance", artifact_object=str(out_png))


if __name__ == "__main__":
    main()
