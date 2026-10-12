import copy

import cv2
import numpy as np
import torch
from pytorch_grad_cam import (
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    RandomCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

_CAM_METHODS = {
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "GradCAM": GradCAM,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "HiResCAM": HiResCAM,
    "LayerCAM": LayerCAM,
    "RandomCAM": RandomCAM,
    "EigenGradCAM": EigenGradCAM,
}


class _ActivationsAndGradients:
    """Forward-hook wrapper that also post-processes YOLO detect output."""

    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for layer in target_layers:
            self.handles.append(layer.register_forward_hook(self._save_activation))
            self.handles.append(layer.register_forward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())

    def _save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        raw = self.model(x)
        # raw[0]: (B, 4+nc, num_anchors) — standard Ultralytics detect head output
        result = raw[0]
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        _, indices = torch.sort(logits_.max(1)[0], descending=True)
        post_result = torch.transpose(logits_[0], 0, 1)[indices[0]]
        pre_post_boxes = torch.transpose(boxes_[0], 0, 1)[indices[0]]
        return [[post_result, pre_post_boxes]]

    def release(self):
        for h in self.handles:
            h.remove()


class _DetectTarget(torch.nn.Module):
    def __init__(self, conf: float, ratio: float):
        super().__init__()
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        scores = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            scores.append(post_result[i].max())
        # Always return a differentiable scalar so GradCAM doesn't crash on images with no detections.
        return sum(scores) if scores else post_result.max() * 0.0


class HeatmapGenerator:
    """Generates GradCAM heatmap overlays for 4-channel RGBT images.

    Initialise once with the trained model path, then call generate() per image.
    generate() returns one heatmap per layer group as a list of (label, image) pairs.

    layer_groups: list of lists of layer indices — one inner list produces one heatmap.
    group_labels: optional names for each group (used as filename suffixes).

    # ── Modality investigation configs ────────────────────────────────────────
    #
    # Mid-fusion (yolo11n-RGBT-midfusion.yaml, 39 layers):
    #   RGB backbone exits before fusion:     [6, 8, 10]   (P3/P4/P5)
    #   Thermal backbone exits before fusion: [16, 18, 20] (P3/P4/P5)
    #   Neck (current default):               [25, 28, 31, 34, 37]
    #
    #   layer_groups=[[6, 8, 10], [16, 18, 20]], group_labels=["rgb", "thermal"]
    #
    # Late-fusion (yolo11n-RGBT-latefusion.yaml, 53 layers):
    #   RGB backbone exits before RGB neck:     [6, 8, 10]   (P3/P4/P5)
    #   Thermal backbone exits before IR neck:  [18, 20, 22] (P3/P4/P5)
    #   RGB neck output P3/P4/P5:               [30, 33, 36]
    #   Thermal neck output P3/P4/P5:           [42, 45, 48]
    #
    #   layer_groups=[[6, 8, 10], [18, 20, 22]], group_labels=["rgb", "thermal"]
    #   — or for neck-level comparison —
    #   layer_groups=[[30, 33, 36], [42, 45, 48]], group_labels=["rgb_neck", "thermal_neck"]
    #
    # Early-fusion (single shared backbone, 4-ch input):
    #   No separate modality branches exist — fused from layer 0 onwards.
    #   Use neck layers directly: layer_groups=[[neck_p3, neck_p4, neck_p5]]
    # ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        method_name: str = "GradCAMPlusPlus",
        layer_groups: list[list[int]] | None = None,
        group_labels: list[str] | None = None,
        conf_threshold: float = 0.2,
        ratio: float = 0.02,
        img_size: int = 640,
        channels: int = 4,
    ):
        if layer_groups is None:
            layer_groups = [[10, 12, 14, 16, 18]]

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.img_size = img_size
        self.channels = channels
        self.group_labels = group_labels or [str(i) for i in range(len(layer_groups))]

        model_yolo = YOLO(model_path)
        model = copy.deepcopy(model_yolo.model)
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        model.task = "detect"
        if not hasattr(model, "end2end"):
            model.end2end = False

        cam_cls = _CAM_METHODS.get(method_name, GradCAMPlusPlus)
        self.target = _DetectTarget(conf_threshold, ratio)

        self.cams = []
        for indices in layer_groups:
            target_layers = [model.model[i] for i in indices]
            cam = cam_cls(model, target_layers)
            cam.activations_and_grads = _ActivationsAndGradients(model, target_layers)
            self.cams.append(cam)

    def _letterbox(self, img: np.ndarray):
        """Resize + pad to self.img_size, preserving aspect ratio. Returns (padded, top, bottom, left, right)."""
        h, w = img.shape[:2]
        r = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if (w, h) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw = (self.img_size - new_w) / 2
        dh = (self.img_size - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        channels = img.shape[2] if img.ndim == 3 else 1
        border_val = (114,) * channels
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_val)
        return img, top, bottom, left, right

    def _to_tensor(self, img_hwc: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Convert HWC numpy to normalised CHW tensor. Also returns rgb_float (H,W,3) for overlay."""
        img3c = np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[:3][::-1], dtype=np.float32) / 255.0
        rgb_float = img3c.transpose(1, 2, 0)  # (H,W,3) in [0,1]
        if self.channels == 4:
            img1c = img_hwc.transpose(2, 0, 1)[-1:].astype(np.float32) / 255.0
            img_chw = np.concatenate([img3c, img1c], axis=0)
        else:
            img_chw = img3c
        tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        return tensor, rgb_float

    def generate(self, img_hwc: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Return one GradCAM heatmap per layer group as (label, uint8 HWC RGB) pairs.

        img_hwc: HWC numpy array. For channels=4 (RGBT): 4-channel BGR+T (as loaded from TIFF).
                 For channels=3 (RGB): 3-channel BGR (as loaded from JPEG/PNG via cv2).
        """
        if img_hwc.dtype != np.uint8:
            img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)

        padded, top, bottom, left, right = self._letterbox(img_hwc)
        tensor, rgb_float = self._to_tensor(padded)
        plain_rgb = _crop_padding((rgb_float * 255).astype(np.uint8), top, bottom, left, right)

        results = []
        for label, cam in zip(self.group_labels, self.cams):
            try:
                grayscale_cam = cam(tensor, [self.target])[0]
                cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
                results.append((label, _crop_padding(cam_image, top, bottom, left, right)))
            except Exception as exc:
                print(f"[HeatmapGenerator] GradCAM failed for group '{label}' ({exc}); returning plain RGB.")
                results.append((label, plain_rgb))

        return results


def _crop_padding(img: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[top: h - bottom if bottom else h, left: w - right if right else w]
