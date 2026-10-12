from ultralytics import YOLO
import yaml
import json
import random
import cv2
import numpy as np
import tifffile

from pathlib import Path

def _read_yolo_dataset_yaml(dataset_yaml_path):
    """Minimal reader for Ultralytics-style dataset.yaml.

    Returns: (root_path: Path, test_subdir: str, names: list[str])
    """

    try:
        data = yaml.safe_load(Path(dataset_yaml_path).read_text(encoding="utf-8"))
        root = Path(data["path"]).expanduser()
        names = data.get("names") or []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        test_subdir = data.get("test", "images/test")
        return root, str(test_subdir), list(names)
    except Exception:
        # Fallback: dataset.yaml written by our exporter is simple key: value lines
        root = None
        test_subdir = "images/test"
        names = []
        for raw in Path(dataset_yaml_path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k == "path":
                root = Path(v).expanduser()
            elif k == "test":
                test_subdir = v
            elif k == "names":
                # Very small parser for Python-list-like strings
                vv = v.strip().lstrip("[").rstrip("]").strip()
                if vv:
                    names = [p.strip().strip("'\"") for p in vv.split(",")]
        if root is None:
            raise RuntimeError(f"Could not parse dataset root from {dataset_yaml_path}")
        return root, test_subdir, names


def save_balanced_test_samples(
    *,
    model: YOLO,
    dataset_yaml: str,
    out_dir: str,
    total_samples: int = 40,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
    iou_match: float = 0.50,
    seed: int = 0,
    heatmap_generator=None,
    use_simotm: str | None = None,
    channels: int | None = None,
    size_range: tuple | None = None,
) -> None:
    """Run inference on test split and save an equal number of good/bad samples.

    "Good" means: all GT boxes matched by same-class prediction with IoU>=iou_match,
    and no unmatched predictions above `conf`.
    "Bad" is everything else.

    When size_range=(min_area_px, max_area_px) is given, good/bad classification
    is size-aware: a prediction's bin is determined by the GT it matched (boundary-safe),
    not by its own area.  Visualization still draws all boxes for full context.
    """

    

    def _xywhn_to_xyxy_px(xc, yc, w, h, W, H):
        x1 = (xc - w / 2.0) * W
        y1 = (yc - h / 2.0) * H
        x2 = (xc + w / 2.0) * W
        y2 = (yc + h / 2.0) * H
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    dataset_root, test_subdir, class_names = _read_yolo_dataset_yaml(dataset_yaml)
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / test_subdir
    # vis_ir layout stores labels alongside images in visible/test/; tiff layout uses labels/test/
    labels_dir = dataset_root / test_subdir.replace("images", "labels") if test_subdir.startswith("images") else images_dir
    # vis_ir mode has paired infrared images under infrared/ mirroring visible/
    ir_images_dir = Path(str(images_dir).replace("visible", "infrared")) if "visible" in test_subdir else None
    if not images_dir.is_dir():
        raise RuntimeError(f"Missing test images folder: {images_dir}")

    out_dir = Path(out_dir)
    good_dir: Path = out_dir / "good"
    bad_dir: Path = out_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    if heatmap_generator is not None:
        (out_dir / "heatmaps" / "good").mkdir(parents=True, exist_ok=True)
        (out_dir / "heatmaps" / "bad").mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise RuntimeError(f"No test images found in {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    good = []
    bad = []
    per_image_report = []

    for img_path in image_paths:
        # Load image — TIFF may be CYX (4,H,W); JPEG/PNG is HWC
        if img_path.suffix.lower() in {".tif", ".tiff"}:
            img = tifffile.imread(img_path.as_posix())
            if img.ndim == 3 and img.shape[0] in (3, 4):
                img_hwc = np.transpose(img, (1, 2, 0))
            else:
                img_hwc = img
        else:
            rgb = cv2.imread(img_path.as_posix(), cv2.IMREAD_COLOR)  # BGR HWC
            if ir_images_dir is not None:
                ir_path = ir_images_dir / img_path.name
                ir = cv2.imread(ir_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                if ir is None:
                    raise RuntimeError(f"Missing infrared counterpart: {ir_path}")
                img_hwc = np.concatenate([rgb, ir[:, :, np.newaxis]], axis=2)  # BGRT HWC
            else:
                img_hwc = rgb

        H, W = int(img_hwc.shape[0]), int(img_hwc.shape[1])

        # GT labels
        label_path = (labels_dir / img_path.name).with_suffix(".txt")
        gt_boxes = []
        gt_cls = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
                gt_boxes.append(_xywhn_to_xyxy_px(xc, yc, w, h, W, H))
                gt_cls.append(cls_id)

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        gt_cls = np.array(gt_cls, dtype=np.int32) if gt_cls else np.zeros((0,), dtype=np.int32)

        # Predictions — always pass the file path so the fork's predictor can load the
        # infrared counterpart itself (visible→infrared path substitution happens internally)
        extra = {}
        if use_simotm is not None:
            extra["use_simotm"] = use_simotm
        if channels is not None:
            extra["channels"] = channels
        pred_results = model.predict(source=img_path.as_posix(), conf=conf, iou=iou, max_det=max_det, verbose=False, **extra)
        r0 = pred_results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_cls = np.zeros((0,), dtype=np.int32)
            pred_conf = np.zeros((0,), dtype=np.float32)
        else:
            pred_boxes = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            pred_cls = r0.boxes.cls.detach().cpu().numpy().astype(np.int32)
            pred_conf = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)

        # Compute in-bin membership masks — used for FN/FP accounting, not for pre-filtering.
        # A prediction's bin is determined by its matched GT's area (boundary-safe): a box
        # slightly wider than the bin edge is still TP if the GT it overlaps is in-bin.
        def _areas(boxes):
            if len(boxes) == 0:
                return np.zeros(0, dtype=np.float32)
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        if size_range is not None:
            min_a, max_a = size_range
            gt_in_bin   = (_areas(gt_boxes)   >= min_a) & (_areas(gt_boxes)   < max_a)
            pred_in_bin = (_areas(pred_boxes) >= min_a) & (_areas(pred_boxes) < max_a)
        else:
            gt_in_bin   = np.ones(len(gt_boxes),   dtype=bool)
            pred_in_bin = np.ones(len(pred_boxes), dtype=bool)

        # Match ALL predictions against ALL GT so boundary predictions are assigned via
        # their matched GT's size, not their own predicted area.
        matched_pred     = set()          # pred indices claimed by any GT
        matched_pred_for_gt: dict = {}    # gt_idx → pred_idx
        for gi in range(len(gt_boxes)):
            best_pi = -1
            best_iou = 0.0
            for pi in range(len(pred_boxes)):
                if pi in matched_pred:
                    continue
                if pred_cls[pi] != gt_cls[gi]:
                    continue
                iou = _iou_xyxy(gt_boxes[gi], pred_boxes[pi])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_pi >= 0 and best_iou >= iou_match:
                matched_pred.add(best_pi)
                matched_pred_for_gt[gi] = best_pi

        # TP = in-bin GTs that were matched by a prediction
        tp = sum(1 for gi in matched_pred_for_gt if gt_in_bin[gi])
        # FN = in-bin GTs that were NOT matched
        fn = int(np.sum(gt_in_bin)) - tp
        # FP = predictions matched to nothing AND whose own area is in-bin
        fp = sum(1 for pi in range(len(pred_boxes)) if pi not in matched_pred and pred_in_bin[pi])

        n_eval_gt   = int(np.sum(gt_in_bin))
        n_eval_pred = int(np.sum(pred_in_bin))

        # Classification of image quality (based on size-specific objects when size_range set)
        if n_eval_gt == 0:
            is_good = n_eval_pred == 0
        else:
            is_good = (fn == 0) and (fp == 0)

        per_image_report.append(
            {
                "image": img_path.name,
                "gt": int(len(gt_boxes)),
                "pred": int(len(pred_boxes)),
                "eval_gt": n_eval_gt,
                "eval_pred": n_eval_pred,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "good": bool(is_good),
            }
        )

        entry = (img_path, img_hwc, gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
        if is_good:
            good.append(entry)
        else:
            bad.append(entry)

        # Early stop: require enough good samples that actually contain objects before stopping.
        need_each = max(1, total_samples // 2)
        good_with_objects = sum(1 for e in good if len(e[2]) > 0)
        if good_with_objects >= need_each and len(bad) >= need_each:
            break

    need_each = max(1, total_samples // 2)
    # Prefer good samples with many GT objects; fall back to empty images only if needed.
    good_nonempty = sorted((e for e in good if len(e[2]) > 0), key=lambda e: len(e[2]), reverse=True)
    good_empty    = [e for e in good if len(e[2]) == 0]
    good = (good_nonempty + good_empty)[:need_each]
    bad  = bad[:need_each]

    def _annotate_and_save(items, dst_dir: Path, category: str):
        for (img_path, img_hwc, gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf) in items:
            # Use RGB channels for visualization
            if img_hwc.ndim == 3 and img_hwc.shape[2] >= 3:
                vis_rgb = img_hwc[:, :, :3]
            else:
                vis_rgb = np.repeat(img_hwc[:, :, None], 3, axis=2)

            if vis_rgb.dtype != np.uint8:
                vis_rgb = np.clip(vis_rgb, 0, 255).astype(np.uint8)

            vis = vis_rgb.copy()

            # Draw GT in green
            for bb, cid in zip(gt_boxes, gt_cls):
                x1, y1, x2, y2 = [int(round(v)) for v in bb.tolist()]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                cv2.putText(vis, f"GT:{name}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

            # Draw predictions in red
            for bb, cid, cf in zip(pred_boxes, pred_cls, pred_conf):
                x1, y1, x2, y2 = [int(round(v)) for v in bb.tolist()]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)
                name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                cv2.putText(
                    vis,
                    f"P:{name} {cf:.2f}",
                    (x1, min(vis.shape[0] - 5, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 220),
                    1,
                )

            out_path = (dst_dir / img_path.stem).with_suffix(".png")
            cv2.imwrite(out_path.as_posix(), vis)

            if heatmap_generator is not None:
                try:
                    for label, heatmap in heatmap_generator.generate(img_hwc):
                        # heatmap is RGB; convert to BGR for cv2.imwrite
                        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                        stem = f"{img_path.stem}_{label}"
                        hmap_path = (out_dir / "heatmaps" / category / stem).with_suffix(".png")
                        cv2.imwrite(hmap_path.as_posix(), heatmap_bgr)
                except Exception as exc:
                    print(f"[heatmap] skipped {img_path.name}: {exc}")

    _annotate_and_save(good, good_dir, "good")
    _annotate_and_save(bad, bad_dir, "bad")

    (out_dir / "report.json").write_text(json.dumps(per_image_report, indent=2), encoding="utf-8")
    print(f"Saved balanced samples to {out_dir} (good={len(good)} bad={len(bad)})")
