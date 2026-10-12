"""
validate_weather_conditions.py — Evaluate all YOLO models in pt/ on the RTDOD FiftyOne
dataset, restricted to two lighting-condition subsets, and plot mAP for each subset using
standard Ultralytics validation.

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

# Dataset to validate on.  Must be rtdod_rgbt (this script is RTDOD-specific).
DATASET: str = "rtdod_rgbt"

# Directory containing *.pt / *.ptrom model files (all will be evaluated).
PT_DIR_STR: str = "pt"

# Where the exported YOLO dataset will be written (reused on subsequent runs).
WORKSPACE_DIR_STR: str = "workspace/weather_conditions_eval"

# Export mode — "vis_ir" matches the training pipeline (separate RGB + thermal JPGs).
EXPORT_MODE: str = "vis_ir"

# Prediction settings.
CONF_THRESH: float = 0.001
IOU_THRESH:  float = 0.7

# Lighting-condition subsets to evaluate.  Keys become x-axis labels in the figure.
CONDITION_GROUPS: dict = {
    "Adverse Weather": ["fog", "rain"],
    "Night":           ["night", "night_rain"],
}

# Number of qualitative balanced samples to keep per (model, condition).
TOTAL_SAMPLES: int = 10

# Output chart path (relative to repo root).
OUT_PNG_STR: str = "weather_conditions_performance.png"

# ClearML / execution settings
CLEARML_PROJECT  = "YOLO11-3DNet"
CLEARML_TASKNAME = "yolo11n_validate_weather_conditions"
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

task.add_tags(["detection", "validation", "weather-conditions", DATASET])

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

# Upload every model file in pt/ (including one level of subfolders) as a ClearML artifact.
_pt_dir = pathlib.Path(PT_DIR_STR)
_model_files = (
    sorted(_pt_dir.glob("*.pt"))   + sorted(_pt_dir.glob("*.ptrom")) +
    sorted(_pt_dir.glob("*/*.pt")) + sorted(_pt_dir.glob("*/*.ptrom"))
)
for _mp in _model_files:
    _rel = _mp.relative_to(_pt_dir)
    _key = str(_rel).replace("\\", "/").replace("/", "__")
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
    "rtdod":            "RTDOD",
}


def extract_model_name(path: pathlib.Path) -> str:
    stem = path.stem.lower()
    model_type = path.stem
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
    dataset = path.parent.name
    if dataset and dataset != "pt_downloaded":
        label = _DATASET_DISPLAY.get(dataset, dataset)
        return f"{model_type} [{label}]"
    return model_type


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
# Figure  (matplotlib + numpy only)
# ============================================================

def make_figure(results: dict, cond_names: list, dataset_name: str, out_path: pathlib.Path) -> None:
    """
    results[model_name][cond_name] = {"mAP50": float, "mAP5095": float}
    cond_names — ordered list of condition group names (x-axis labels)
    """
    raw_methods = list(results.keys())

    _TYPE_ORDER = ["RGB", "Thermal", "Early", "Mid", "MidP3", "MidCMA3", "MidCFT3", "Late", "Score"]

    def _type_rank(m: str) -> int:
        t = _extract_type(m)
        return _TYPE_ORDER.index(t) if t in _TYPE_ORDER else len(_TYPE_ORDER)

    methods = sorted(raw_methods, key=lambda m: (_type_rank(m), m))

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

    offsets: dict = {}
    cur = 0.0
    for _, group in type_groups:
        for mi, method in enumerate(group):
            offsets[method] = cur
            cur += slot + (within_type_gap if mi < len(group) - 1 else 0.0)
        cur += between_type_gap
    total_w = cur - between_type_gap

    spacing = max(1.0, total_w + 0.4)
    x = np.arange(len(cond_names)) * spacing

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

    for i, cond_name in enumerate(cond_names):
        for method in methods:
            color = _model_color_for_method(method, methods)
            edge  = darken(color)
            off   = offsets[method] - total_w / 2
            x50   = x[i] + off + bar_w / 2
            x5095 = x[i] + off + bar_w + inner_gap + bar_w / 2
            vals  = results[method].get(cond_name)
            if vals is None:
                continue
            draw_bar(ax, x50,   vals["mAP50"],   color, edge, hatch=None,   width=bar_w)
            draw_bar(ax, x5095, vals["mAP5095"], color, edge, hatch="////", width=bar_w)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5, zorder=0)
    for i in range(len(cond_names) - 1):
        ax.axvline(x=(x[i] + x[i + 1]) / 2, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    half = total_w / 2 + 0.15
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names)
    ax.set_ylabel("mAP")
    ax.set_xlim(x[0] - half, x[-1] + half)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"YOLO11 fusion methods — mAP per weather condition  ({dataset_name} val split)")

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
    import random as _rnd
    import shutil
    import sys
    import tempfile

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

    from dataloader import RtdodRgbtYoloDatasetDataloader

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

    workspace_dir = _repo_root / WORKSPACE_DIR_STR
    out_png       = _repo_root / OUT_PNG_STR

    # ---- Download model artifacts uploaded before execute_remotely ----
    local_pt_dir = _repo_root / "pt_downloaded"
    local_pt_dir.mkdir(exist_ok=True)

    model_paths = []
    for artifact_name, artifact in task.artifacts.items():
        if not (artifact_name.endswith(".pt") or artifact_name.endswith(".ptrom")):
            continue
        local_copy = pathlib.Path(artifact.get_local_copy())
        if "__" in artifact_name:
            dataset_folder, file_part = artifact_name.split("__", 1)
        else:
            dataset_folder, file_part = "", artifact_name
        stem = pathlib.Path(file_part).stem
        sub  = local_pt_dir / dataset_folder if dataset_folder else local_pt_dir
        sub.mkdir(exist_ok=True)
        dest = sub / f"{stem}.pt"
        shutil.copy2(str(local_copy), str(dest))
        model_paths.append(dest)

    if not model_paths:
        raise FileNotFoundError("No .pt / .ptrom artifacts found in this ClearML task.")
    model_paths.sort()
    print(f"\nModels to evaluate: {[str(p.relative_to(local_pt_dir)) for p in model_paths]}")

    # ---- Export (or reuse) full RTDOD val split ----
    print("\n--- Exporting dataset ---")
    loader = RtdodRgbtYoloDatasetDataloader(
        out_root=workspace_dir,
        recreate=False,
        export_mode=cfg["export_mode"],
        image_ext=".jpg",
    )
    dataset_yaml = loader.prepare()
    print(f"  YAML: {dataset_yaml}")

    ds_cfg    = yaml.safe_load(dataset_yaml.read_text())
    cond_names = list(CONDITION_GROUPS.keys())

    # ---- Build stem → lighting_condition map from FiftyOne ----
    print("\n--- Building stem → lighting_condition map from FiftyOne ---")
    import fiftyone as fo

    fo_dataset  = fo.load_dataset(dataset_name)
    fo_val_view = fo_dataset.match(fo.ViewField("split") == "val")

    stem_to_cond: dict[str, str] = {}
    for sample in fo_val_view.iter_samples(progress=True):
        stem = pathlib.Path(sample.filepath).stem
        cond = (sample.get_field("lighting_condition") or "").lower()
        stem_to_cond[stem] = cond

    from collections import Counter
    cond_counts = Counter(stem_to_cond.values())
    print(f"  lighting_condition distribution (val): {dict(cond_counts)}")
    if all(v == "" for v in stem_to_cond.values()):
        print("  WARNING: all lighting_condition values are empty — check field name with"
              " fo.load_dataset('rtdod_rgbt').get_field_schema()")

    # ---- Build per-condition stem sets ----
    cond_stems: dict[str, set] = {
        name: {s for s, c in stem_to_cond.items() if c in [v.lower() for v in values]}
        for name, values in CONDITION_GROUPS.items()
    }
    for name, stems in cond_stems.items():
        print(f"  {name}: {len(stems)} val images")

    # ---- Pre-select fixed balanced-sample pools per condition ----
    print("\n--- Pre-selecting balanced-sample pools (val split) ---")
    _pool_img_dir  = pathlib.Path(ds_cfg["path"]) / ds_cfg.get("val", "images/val")
    _pool_img_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

    cond_pools: dict[str, list] = {}
    for cond_name, stems in cond_stems.items():
        eligible = [
            stem for stem in sorted(stems)
            if any((_pool_img_dir / f"{stem}{e}").exists() for e in _pool_img_exts)
        ]
        rng = _rnd.Random(42)
        rng.shuffle(eligible)
        cond_pools[cond_name] = eligible[:TOTAL_SAMPLES]
        print(f"  {cond_name}: {len(cond_pools[cond_name])}/{len(eligible)} val images selected")

    # ---- Helper: build a temp dataset from a set of stems ----
    def _make_temp_dataset(stems: list, modality: str) -> tuple[tempfile.TemporaryDirectory, pathlib.Path]:
        """
        Copy the given stems into a temp directory and return (tmp_ctx, yaml_path).
        The caller is responsible for cleaning up by calling tmp_ctx.cleanup().
        Both val: and test: in the YAML point to the same dir so model.val() and
        save_balanced_test_samples() both find the images without extra copying.
        """
        vis_sub = ds_cfg.get("val", "visible/val")
        ir_sub  = vis_sub.replace("visible", "infrared")
        ds_root = pathlib.Path(ds_cfg["path"])
        vis_dir = ds_root / vis_sub
        ir_dir  = ds_root / ir_sub

        img_src_dir = ir_dir if modality == "thermal" else vis_dir
        lbl_src_dir = vis_dir   # labels always co-located with visible images in vis_ir mode

        _img_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

        tmp_ctx  = tempfile.TemporaryDirectory(dir=str(workspace_dir))
        tmp_root = pathlib.Path(tmp_ctx.name)

        if modality == "rgbt":
            tmp_img_dir = tmp_root / vis_sub
            tmp_img_dir.mkdir(parents=True, exist_ok=True)
            tmp_ir_dir  = tmp_root / ir_sub
            tmp_ir_dir.mkdir(parents=True, exist_ok=True)
            tmp_lbl_dir = tmp_img_dir
            tmp_data_sub = vis_sub
        else:
            tmp_img_dir  = tmp_root / "images" / "val"
            tmp_img_dir.mkdir(parents=True, exist_ok=True)
            tmp_lbl_dir  = tmp_root / "labels" / "val"
            tmp_lbl_dir.mkdir(parents=True, exist_ok=True)
            tmp_ir_dir   = None
            tmp_data_sub = "images/val"

        for stem in stems:
            for src in img_src_dir.glob(f"{stem}.*"):
                if src.suffix.lower() in _img_exts:
                    shutil.copy2(str(src), str(tmp_img_dir / src.name))
            lbl_src = lbl_src_dir / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(str(lbl_src), str(tmp_lbl_dir / f"{stem}.txt"))
            if modality == "rgbt" and tmp_ir_dir is not None:
                for src in ir_dir.glob(f"{stem}.*"):
                    if src.suffix.lower() in _img_exts:
                        shutil.copy2(str(src), str(tmp_ir_dir / src.name))

        tmp_channels = 4 if modality == "rgbt" else 3
        tmp_yaml = tmp_root / "dataset.yaml"
        tmp_yaml.write_text(
            f"path: {tmp_root.resolve()}\n"
            f"train: {ds_cfg.get('train', '')}\n"
            f"val: {tmp_data_sub}\n"
            f"test: {tmp_data_sub}\n"
            f"nc: {ds_cfg['nc']}\n"
            f"names: {ds_cfg['names']}\n"
            f"channels: {tmp_channels}\n"
        )
        return tmp_ctx, tmp_yaml

    # ---- Per-model evaluation loop ----
    # all_results[model_name][cond_name] = {"mAP50": float, "mAP5095": float}
    all_results: dict[str, dict] = {}
    logger = Logger.current_logger()

    for model_path in model_paths:
        model_name   = extract_model_name(model_path)
        modality     = _model_modality(model_name)
        safe_name    = model_name.replace(" [", "_").replace("]", "")
        balanced_dir = workspace_dir / f"balanced_samples_{safe_name}"

        print(f"\n=== {model_name}  ({model_path.name}) ===")

        yolo_model = YOLO(str(model_path))
        all_results[model_name] = {}

        val_kwargs: dict = (
            {"channels": 4, "use_simotm": "RGBT"}
            if modality == "rgbt"
            else {"channels": 3}
        )

        for cond_name in CONDITION_GROUPS:
            all_stems  = sorted(cond_stems[cond_name])
            pool_stems = cond_pools[cond_name]

            print(f"  [{cond_name}] {len(all_stems)} images — building temp dataset …")

            tmp_ctx, tmp_yaml = _make_temp_dataset(all_stems, modality)
            try:
                # ---- Standard Ultralytics validation ----
                print(f"  [{cond_name}] running model.val() …")
                metrics = yolo_model.val(
                    data=str(tmp_yaml),
                    split="val",
                    conf=cfg["conf_thresh"],
                    iou=cfg["iou_thresh"],
                    verbose=False,
                    **val_kwargs,
                )
                map50   = float(metrics.box.map50)
                map5095 = float(metrics.box.map)
                all_results[model_name][cond_name] = {"mAP50": map50, "mAP5095": map5095}
                print(f"    mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

                # ---- Balanced good/bad samples (fixed pool) ----
                print(f"  [{cond_name}] exporting balanced samples …")
                cond_out = balanced_dir / cond_name
                save_balanced_test_samples(
                    model=yolo_model,
                    dataset_yaml=str(tmp_yaml),
                    out_dir=str(cond_out),
                    total_samples=len(pool_stems) * 2,
                    conf=0.25,
                    iou=cfg["iou_thresh"],
                    max_det=300,
                    iou_match=0.50,
                    seed=13,
                    heatmap_generator=None,
                    **val_kwargs,
                )
            finally:
                tmp_ctx.cleanup()

            for category in ("good", "bad"):
                for img_path in sorted((cond_out / category).glob("*.png")):
                    logger.report_image(
                        title=f"{model_name}/balanced_samples/{cond_name}/{category}",
                        series=img_path.stem,
                        iteration=0,
                        image=Image.open(str(img_path)).convert("RGB"),
                    )
            report_json = cond_out / "report.json"
            if report_json.is_file():
                task.upload_artifact(
                    f"balanced_report_{safe_name}_{cond_name}", str(report_json)
                )

    # ---- Generate figure ----
    print("\n--- Generating figure ---")
    make_figure(all_results, cond_names, dataset_name, out_png)

    logger.report_image(
        title="Weather Conditions Performance",
        series="mAP per weather condition",
        iteration=0,
        local_path=str(out_png),
    )
    task.upload_artifact("weather_conditions_performance", artifact_object=str(out_png))


if __name__ == "__main__":
    main()
