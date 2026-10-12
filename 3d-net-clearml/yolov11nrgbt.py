
# ClearML Task Initialization
# --------------------------------------------------------
# Must be at the top before any other imports

import os
from pathlib import Path
from typing import Literal
from clearml import Logger, Task
from dotenv import load_dotenv, dotenv_values

load_dotenv()
env_variables = dotenv_values(".env")

CLEARML_PROJECT  = "YOLO11-3DNet"
CLEARML_TASKNAME = "yolo11nrgbt_3dnet_train"
RUN_REMOTE       = True                 # set False to run locally
REMOTE_QUEUE     = os.getenv("CLEARML_QUEUE", "training-multi")
AUTO_QUANTIZE    = os.getenv("AUTO_QUANTIZE", "false").lower() == "true"
task: Task = Task.init(
    project_name=CLEARML_PROJECT,
    task_name=CLEARML_TASKNAME,
    task_type=Task.TaskTypes.training,
    auto_connect_frameworks={"pytorch": False, "matplotlib": False},
)

FIFTYONE_DATASET: Literal["visdrone_rgbt", "rtdod_rgbt", "m3fd_rgbt", "3d_net_rgbt", "3d_net_data_pool_rgbt"] = os.getenv("FIFTYONE_DATASET", "rtdod_rgbt")
# "mid" | "midP3" | "midCMA3" | "midCTF" | "early" | "late" | "score" — selects the fusion architecture YAML
FUSION_TYPE: Literal["mid", "midP3", "midCMA3", "midCTF3", "midCTF2", "midCFT3", "early", "late", "score"] = os.getenv("FUSION_TYPE", "mid")  # type: ignore[assignment]

task.add_tags(["detection", FIFTYONE_DATASET, f"fusion-{FUSION_TYPE}"])

deploy_token_user = env_variables.get("REPO_DEPLOY_TOKEN_USERNAME", "")
deploy_token_pass = env_variables.get("REPO_DEPLOY_TOKEN_PASSWORD", "")

task.set_repo(f"https://{deploy_token_user}:{deploy_token_pass}@github.com/bernard-technologies/3d-net-clearml.git", branch="main")

task.set_base_docker(
    docker_image="swarm.azurecr.io/swarm-dnn-nextgen:latest-cuda-122",
    docker_arguments="".join([f' -e {key}="{value}"' for key, value in env_variables.items()]),
    docker_setup_bash_script=["apt-get install -y --no-install-recommends libgl1"], # required for yolov11 RGBT ultralytics fork with non-headless opencv dependency
)

task.upload_artifact("yolov11n_pretrained", artifact_object=Path("yolo11n.pt"))

if RUN_REMOTE:
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)




def main() -> None:
    # --- Torch compat: drop unsupported weights_only BEFORE importing ultralytics ---
    import inspect
    import torch

    _orig_torch_load = torch.load
    _has_weights_only = "weights_only" in inspect.signature(_orig_torch_load).parameters

    def _compat_torch_load(*args, **kwargs):
        if "weights_only" in kwargs and not _has_weights_only:
            kwargs.pop("weights_only")
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load  # patch first!

    # --- Imports (after patch) ---
    import sys
    import os
    from pathlib import Path

    # Ensure local imports resolve reliably on remote runners (e.g. ClearML Agent)
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ultralytics import YOLO

    # Set EXPORT_ONNX=0 to skip ONNX export and use the unpatched GPT module (e.g. for performance baselines).
    # GPTFixed replaces GPT with an ONNX-compatible variant; skipping it allows training with the original module.
    _export_onnx = os.environ.get("EXPORT_ONNX", "1").strip() not in ("0", "false", "False", "no")

    if _export_onnx:
        from custom_modules import GPTFixed, CrossMLCAv2Fixed, onnx_fixed_modules_parse_model
        import ultralytics.nn.modules as _ul_modules
        import ultralytics.nn.tasks as _ul_tasks
        for _cls in (GPTFixed, CrossMLCAv2Fixed):
            setattr(_ul_modules, _cls.__name__, _cls)
            setattr(_ul_tasks, _cls.__name__, _cls)
        _ul_tasks.parse_model = onnx_fixed_modules_parse_model(_ul_tasks.parse_model)
    # else: use stock GPT module — ONNX export will be skipped at the end of training

    from dataloader import VisDroneRgbtYoloDatasetDataloader, RtdodRgbtYoloDatasetDataloader, M3fdRgbtYoloDatasetDataloader, ThreeDNetRgbtYoloDatasetDataloader, ThreeDNetDataPoolRgbtYoloDatasetDataloader, CLASS_NAMES

    from test.test_balanced_sample_export import save_balanced_test_samples
    from test.heatmap_utils import HeatmapGenerator

    load_dotenv()  # loads variables from .env into os.environ

    # =========================
    # Build YOLO dataset from FiftyOne (runs on the remote agent)
    # =========================
    dataset_base_path = "./workspace/dataset_rgbt"
    recreate_dataset = False

    DATASET_LOADERS = {
        "visdrone_rgbt":         VisDroneRgbtYoloDatasetDataloader,
        "rtdod_rgbt":            RtdodRgbtYoloDatasetDataloader,
        "m3fd_rgbt":             M3fdRgbtYoloDatasetDataloader,
        "3d_net_rgbt":           ThreeDNetRgbtYoloDatasetDataloader,
        "3d_net_data_pool_rgbt": ThreeDNetDataPoolRgbtYoloDatasetDataloader,
    }

    if FIFTYONE_DATASET not in DATASET_LOADERS:
        raise ValueError(f"Unsupported DATASET: {FIFTYONE_DATASET}")

    dataloader = DATASET_LOADERS[FIFTYONE_DATASET](
        out_root=Path(dataset_base_path),
        recreate=recreate_dataset,
        export_mode="vis_ir",
        image_ext=".jpg",
    )

    dataset_yaml = dataloader.prepare()

    print("Using dataset YAML:", dataset_yaml)
    print("Dataset contents:\n", dataset_yaml.read_text())

    _valid_fusion_types = ("mid", "midP3", "midCMA3", "midCTF3", "midCTF2", "midCFT3", "early", "late", "score")
    if FUSION_TYPE not in _valid_fusion_types:
        raise ValueError(f"FUSION_TYPE must be one of {_valid_fusion_types}, got: {FUSION_TYPE!r}")
    model_yaml = repo_root / "models" / f"yolo11n-RGBT-{FUSION_TYPE}fusion.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"RGBT model YAML not found: {model_yaml}")

    # =========================
    # ClearML task
    # =========================

    cfg = {
        "dataset_base": dataset_base_path,
        # Hailo quantization: normalization applied in hardware before inference.
        # For YOLO (ultralytics) the model expects inputs in [0, 1] (i.e. pixel / 255).
        # mean=0, std=1 maps to Hailo hardware values mean_hw=0, std_hw=256 → x/255.
        "normalize_mean": [0.0, 0.0, 0.0, 0.0],
        "normalize_std":  [1.0, 1.0, 1.0, 1.0],
        # Class names must match those in the dataset YAML used for training.
        "classes": CLASS_NAMES,
        "fiftyone_dataset": FIFTYONE_DATASET,
        "fusion_type": FUSION_TYPE,
    }
    cfg = task.connect(cfg, name="data")  # now editable in the ClearML UI

    data_root = Path(cfg["dataset_base"])
    model_variant = f"yolo11n-RGBT-{FUSION_TYPE}fusion"

    # --- Sanity checks ---
    print("CWD:", os.getcwd())
    print("DATA_ROOT:", data_root.resolve(), "exists:", data_root.exists())

    run_name = f"{model_variant}_{task.id[:8]}"
    project_name = str(data_root / "Results")

    args = dict(
        data=str(dataset_yaml),
        epochs=int(os.getenv("EPOCHS", 10)),
        imgsz=int(os.getenv("IMGSZ", 640)),
        batch=int(os.getenv("BATCH", 32)),
        workers=int(os.getenv("WORKERS", 8)),
        device=0 if RUN_REMOTE else "cpu",
        project=project_name,
        name=run_name,
        verbose=True,
        seed=0,
        use_simotm="RGBT",
        channels=4,
        patience=0,
    )
    task.set_parameter("model_variant", model_variant)
    task.connect(args)

    # =========================
    # Train
    # =========================
    pretrained_weights = task.artifacts["yolov11n_pretrained"].get_local_copy()

    model = YOLO(str(model_yaml))
    model.load(pretrained_weights)   # copies matching weights by key+shape, skips mismatches
    _ = model.train(**args)

    run_dir = Path(model.trainer.save_dir)

    best_ckpt = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"

    import plotly.express as px
    import plotly.graph_objects as go
    from PIL import Image

    logger = Logger.current_logger()

    def _log_dir_plots(save_dir, title_prefix=""):
        """Report all PNGs in a directory to the ClearML Plots tab via plotly."""
        for plot_path in sorted(Path(save_dir).glob("*.png")):
            img = Image.open(str(plot_path)).convert("RGB")
            fig = px.imshow(img)
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False)
            title = f"{title_prefix}{plot_path.stem}" if title_prefix else plot_path.stem
            logger.report_plotly(title=title, series="plot", iteration=0, figure=fig)

    def _log_per_class_ap(metrics, class_names, split):
        """Report a grouped horizontal bar chart of per-class AP to the ClearML Plots tab."""
        ap_class_idx = metrics.ap_class_index
        ap50 = metrics.box.ap50
        ap   = metrics.box.ap
        names = [class_names[i] for i in ap_class_idx]
        fig = go.Figure([
            go.Bar(name="AP50",    y=names, x=list(ap50), orientation="h"),
            go.Bar(name="AP50-95", y=names, x=list(ap),   orientation="h"),
        ])
        fig.update_layout(
            barmode="group",
            title=f"Per-class AP ({split})",
            xaxis=dict(title="AP", range=[0, 1]),
            height=max(400, len(names) * 40),
        )
        logger.report_plotly(
            title=f"per_class_AP_{split}", series="AP50 vs AP50-95", iteration=0, figure=fig
        )

    # Log all training plots (curves, confusion matrices, label distributions, results summary).
    _log_dir_plots(run_dir)


    # =========================
    # Export ONNX
    # =========================

    onnx_path = run_dir / "weights" / "best.onnx"
    if _export_onnx:
        try:
            onnx_out = Path(model.export(format="onnx"))
            onnx_path = Path(
                onnx_out if isinstance(onnx_out, (str, Path)) else onnx_path
            )
            print(f"\nONNX model exported to {onnx_path}")
        except Exception as exc:
            print(f"\nWARNING: ONNX export failed, skipping: {exc}")
            onnx_path = run_dir / "weights" / "best.onnx"
    else:
        print("\nONNX export skipped (EXPORT_ONNX=0)")


    # =========================
    # ClearML artifacts
    # =========================
    if best_ckpt.is_file():
        task.upload_artifact("best_model", str(best_ckpt))
    if results_csv.is_file():
        task.upload_artifact("training_metrics_csv", str(results_csv))
    if onnx_path.is_file():
        task.upload_artifact("onnx", str(onnx_path))


    # =========================
    # Evaluate
    # =========================
    # Ultralytics defaults to split='val'. Run both validation and test explicitly.
    eval_model = YOLO(str(best_ckpt)) if best_ckpt.is_file() else model
    for split in ("val", "test"):
        split_results = eval_model.val(
            data=str(dataset_yaml),
            split=split,
            project=args["project"],
            name=f"{run_name}_{split}",
            plots=True,
            channels=args["channels"],
            use_simotm=args["use_simotm"],
        )
        split_dir = Path(args["project"]) / f"{run_name}_{split}"
        _log_dir_plots(split_dir, title_prefix=f"{split}/")
        _log_per_class_ap(split_results, eval_model.names, split=split)


    # =========================
    # Qualitative sample export (balanced good/bad)
    # =========================
    total_samples = int(os.getenv("TEST_SAMPLES", 10))
    balanced_dir = Path(args["project"]) / f"{run_name}_balanced_samples"

    # Layer groups per fusion type. Each inner list probes one modality branch and
    # produces a separate heatmap. Indices are absolute positions in model.model.
    #
    # mid    (39 layers): dual backbone, fused at Concat layers 21-23.
    #   RGB exits: 6/8/10 (P3/P4/P5), thermal exits: 16/18/20 (P3/P4/P5).
    # midP3  (34 layers): dual backbone fused at P3 (Concat layer 13); shared P4/P5.
    #   RGB exits: 6 (P3 only), thermal exits: 12 (P3 only).
    # late   (53 layers): dual backbone + dual neck, fused only at Detect inputs.
    #   RGB exits: 6/8/10 (P3/P4/P5), thermal exits: 18/20/22 (P3/P4/P5).
    # early  (26 layers): single shared 4-ch backbone — no modality split possible.
    #   Same backbone P3/P4/P5 exits as RGB branch above: 6/8/10.
    _HEATMAP_CFG: dict[str, tuple[list[list[int]], list[str]]] = {
        "mid":     ([[6, 8, 10], [16, 18, 20]], ["rgb", "thermal"]),
        "midP3":   ([[6], [12]],                ["rgb", "thermal"]),
        # midCMA3 (33 layers): dual backbone, CrossMLCAv2 fused at P3/P4/P5.
        #   RGB exits: 6/18/26 (P3/P4/P5), thermal exits: 12/20/28 (P3/P4/P5).
        "midCMA3": ([[6, 18, 26], [12, 20, 28]], ["rgb", "thermal"]),
        # midCTF3 (26 layers): dual backbone, CrossTransformerFusion fused at P3/P4/P5.
        #   RGB exits: 6/8/10 (P3/P4/P5), thermal exits: 16/18/20 (P3/P4/P5).
        "midCTF3": ([[6, 8, 10], [16, 18, 20]], ["rgb", "thermal"]),
        # midCTF2 (26 layers): dual backbone, Concat at P3, CrossTransformerFusion at P4/P5.
        #   Same layer exits as midCTF3. RGB: 6/8/10, thermal: 16/18/20.
        "midCTF2": ([[6, 8, 10], [16, 18, 20]], ["rgb", "thermal"]),
        # midCFT3 (35 layers): dual backbone, GPT (CFT) fused at P3/P4/P5 with Add2 residual streams.
        #   RGB exits: 6/17/24 (P3/P4/P5), thermal exits: 12/19/26 (P3/P4/P5).
        "midCFT3": ([[6, 17, 24], [12, 19, 26]], ["rgb", "thermal"]),
        "late":    ([[6, 8, 10], [18, 20, 22]], ["rgb", "thermal"]),
        "score":   ([[6, 8, 10], [18, 20, 22]], ["rgb", "thermal"]),
        "early":   ([[6, 8, 10]],               ["backbone"]),
    }
    layer_groups, group_labels = _HEATMAP_CFG[FUSION_TYPE]

    heatmap_generator = None
    heatmap_device = args.get("device", "cuda:0")
    if best_ckpt.is_file():
        try:
            heatmap_generator = HeatmapGenerator(
                model_path=str(best_ckpt),
                device=heatmap_device,
                method_name="GradCAMPlusPlus",
                layer_groups=layer_groups,
                group_labels=group_labels,
                conf_threshold=0.2,
                ratio=0.02,
                img_size=args.get("imgsz", 640),
            )
        except Exception as exc:
            print(f"[heatmap] generator init failed ({exc}); skipping heatmaps.")

    save_balanced_test_samples(
        model=eval_model,
        dataset_yaml=str(dataset_yaml),
        out_dir=str(balanced_dir),
        total_samples=total_samples,
        conf=0.25,
        iou_match=0.50,
        seed=13,
        heatmap_generator=heatmap_generator,
        use_simotm=args["use_simotm"],
        channels=args["channels"],
    )

    # Upload annotated images and heatmaps to ClearML (visible under Debug Samples).
    for category in ("good", "bad"):
        for img_path in sorted((balanced_dir / category).glob("*.png")):
            logger.report_image(
                title=f"balanced_samples/{category}",
                series=img_path.stem,
                iteration=0,
                image=Image.open(str(img_path)).convert("RGB"),
            )
        heatmap_dir = balanced_dir / "heatmaps" / category
        for img_path in sorted(heatmap_dir.glob("*.png")):
            logger.report_image(
                title=f"heatmaps/{category}",
                series=img_path.stem,
                iteration=0,
                image=Image.open(str(img_path)).convert("RGB"),
            )
    report_json = balanced_dir / "report.json"
    if report_json.is_file():
        task.upload_artifact("balanced_samples_report", str(report_json))

    print("Done.")

    # =========================
    # Auto-launch quantization
    # =========================
    if AUTO_QUANTIZE:
        import subprocess
        import sys
        print(f"AUTO_QUANTIZE enabled – launching quantization task for training task {task.id}...")
        training_task_id = task.id
        task.close()  # close training task so quantize.py's Task.init() creates a fresh task
        # Strip ClearML agent vars so Task.init() in quantize.py creates a *new* task
        # instead of reconnecting to the (now-closed) training task.
        subprocess_env = {**os.environ, "CLEARML_QUANTIZE_TRAINING_TASK_ID": training_task_id}
        for _key in ("CLEARML_TASK_ID", "CLEARML_TASK_NO_REUSE"):
            subprocess_env.pop(_key, None)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "quantize.py")],
            env=subprocess_env,
            check=True,
        )


if __name__ == "__main__":
    main()
