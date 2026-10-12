import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from ultralytics.nn.modules.block import GPT, CrossMLCAv2


class GPTFixed(GPT):
    """GPT cross-modal fusion with fixed AvgPool2d replacing AdaptiveAvgPool2d.

    Drop-in replacement for GPT — identical YAML args.  On first forward call the
    actual input spatial size is detected and self.avgpool is replaced with the
    closest fixed AvgPool2d, making the module ONNX-exportable under PyTorch 2.1.

    Formula (matches AdaptiveAvgPool2d output size for any H, W):
        stride = H // out_anchors
        kernel = H - (out_anchors - 1) * stride
    Exact for P3 (80×80) and P4 (40×40) with default 8×8 anchors.
    Closest approximation for P5 (20×20 → kernel=6, stride=2 gives 8 outputs).
    """

    def forward(self, x):
        rgb_fea = x[0]
        H, W = int(rgb_fea.shape[2]), int(rgb_fea.shape[3])
        # Re-configure whenever the actual input size changes (e.g. TensorBoard traces
        # with a different dummy size before training starts, poisoning the lazy setup).
        if getattr(self, '_fixed_pool_h', None) != H or getattr(self, '_fixed_pool_w', None) != W:
            oh, ow = self.vert_anchors, self.horz_anchors
            stride_h = H // oh
            kernel_h = H - (oh - 1) * stride_h
            stride_w = W // ow
            kernel_w = W - (ow - 1) * stride_w
            self.avgpool = nn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)
            )
            self._fixed_pool_h = H
            self._fixed_pool_w = W
        return super().forward(x)


class CrossMLCAv2Fixed(CrossMLCAv2):
    """CrossMLCAv2 with all AdaptiveAvgPool2d replaced for ONNX export.

    Drop-in replacement for CrossMLCAv2 — identical YAML args.  Lazily replaces
    module-level adaptive pools and overrides forward to replace functional calls.
    All standard pyramid sizes (80, 40, 20) divide evenly by default local_size=5.
    """

    def __init__(self, in_size, local_size: int = 5,
                 gamma: int = 2, b: int = 1, local_weight: float = 0.5):
        super().__init__(in_size, local_size, gamma, b, local_weight)
        self._scale_factor = None  # set lazily on first forward call
        self._pad_h = 0
        self._pad_w = 0

    def _setup_fixed_pools(self, H: int, W: int):
        # Pad to the next multiple of local_size so exact AvgPool2d always works,
        # including rect-mode validation batches with non-divisible spatial sizes.
        pad_h = (-H) % self.local_size
        pad_w = (-W) % self.local_size
        self._pad_h = pad_h
        self._pad_w = pad_w
        H_p = H + pad_h
        W_p = W + pad_w
        k_h = H_p // self.local_size
        k_w = W_p // self.local_size
        self._scale_factor = (float(k_h), float(k_w))
        self.local_arv_pool = nn.AvgPool2d(kernel_size=(k_h, k_w), stride=(k_h, k_w))
        self.global_arv_pool = nn.AvgPool2d(
            kernel_size=self.local_size, stride=self.local_size
        )

    def forward(self, x):
        x1 = x[0] if isinstance(x, (list, tuple)) else x
        H, W = int(x1.shape[2]), int(x1.shape[3])
        # Re-configure on every input-size change (guards against TensorBoard
        # tracing with a different dummy size before training starts).
        if getattr(self, '_fixed_pool_h', None) != H or getattr(self, '_fixed_pool_w', None) != W:
            self._setup_fixed_pools(H, W)
            self._fixed_pool_h = H
            self._fixed_pool_w = W

        x1, x2 = x
        # Pad to a multiple of local_size when H or W is not evenly divisible
        # (e.g. rect-mode validation batches from non-square images).
        if self._pad_h > 0 or self._pad_w > 0:
            x1_p = F.pad(x1, (0, self._pad_w, 0, self._pad_h))
            x2_p = F.pad(x2, (0, self._pad_w, 0, self._pad_h))
        else:
            x1_p, x2_p = x1, x2

        local_arv1 = self.local_arv_pool(x1_p)
        global_arv1 = self.global_arv_pool(local_arv1)
        local_arv2 = self.local_arv_pool(x2_p)
        global_arv2 = self.global_arv_pool(local_arv2)

        b, c, m, n = x1.shape
        b_local, c_local, m_local, n_local = local_arv1.shape

        spatial_info_local1 = self.conv_cae(local_arv1)
        spatial_info_local2 = self.conv_cae(local_arv2)

        temp_local1 = local_arv1.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_local2 = local_arv2.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global1 = global_arv1.view(b, c, -1).transpose(-1, -2)
        temp_global2 = global_arv2.view(b, c, -1).transpose(-1, -2)

        y_local1 = self.conv_local(temp_local1)
        y_global1 = self.conv(temp_global1)
        y_local2 = self.conv_local(temp_local2)
        y_global2 = self.conv(temp_global2)

        y_local_transpose1 = y_local1.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose1 = y_global1.transpose(-1, -2).view(b, c, 1, 1)
        y_local_transpose2 = y_local2.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose2 = y_global2.transpose(-1, -2).view(b, c, 1, 1)

        y_local_transpose1 = spatial_info_local1 * y_local_transpose1
        y_local_transpose2 = spatial_info_local2 * y_local_transpose2

        att_local1 = y_local_transpose1.sigmoid()
        att_global1 = F.interpolate(
            y_global_transpose1.sigmoid(), size=(self.local_size, self.local_size), mode='nearest'
        )
        att_all1 = F.interpolate(
            att_global1 * (1 - self.local_weight) + att_local1 * self.local_weight,
            scale_factor=self._scale_factor, mode='nearest',
        )

        att_local2 = y_local_transpose2.sigmoid()
        att_global2 = F.interpolate(
            y_global_transpose2.sigmoid(), size=(self.local_size, self.local_size), mode='nearest'
        )
        att_all2 = F.interpolate(
            att_global2 * (1 - self.local_weight) + att_local2 * self.local_weight,
            scale_factor=self._scale_factor, mode='nearest',
        )

        # Trim attention maps back to the original spatial size when padding was applied.
        if self._pad_h > 0 or self._pad_w > 0:
            att_all1 = att_all1[:, :, :H, :W]
            att_all2 = att_all2[:, :, :H, :W]

        x1 = x1 * att_all1 + x1
        x2 = x2 * att_all2 + x2

        merged = torch.cat([x1, x2], dim=1)
        output = self.merge_conv(merged)

        return [x1, x2, output]


def onnx_fixed_modules_parse_model(parse_model_fn):
    """Decorator for ultralytics parse_model that adds support for GPTFixed and
    CrossMLCAv2Fixed.

    parse_model uses `m is GPT` / `m in frozenset({..., CrossMLCAv2})` identity
    checks to determine channel computation.  These checks never match our subclasses.

    The decorator temporarily replaces tasks.GPT → GPTFixed and
    tasks.CrossMLCAv2 → CrossMLCAv2Fixed for the duration of the call so the
    existing identity checks pick up the fixed variants automatically, then restores
    the originals.
    """
    import ultralytics.nn.tasks as _tasks

    @wraps(parse_model_fn)
    def wrapper(d, ch, verbose=True):
        saved = {
            'GPT': getattr(_tasks, 'GPT', GPT),
            'CrossMLCAv2': getattr(_tasks, 'CrossMLCAv2', CrossMLCAv2),
        }
        _tasks.GPT = GPTFixed
        _tasks.CrossMLCAv2 = CrossMLCAv2Fixed
        _tasks.GPTFixed = GPTFixed
        _tasks.CrossMLCAv2Fixed = CrossMLCAv2Fixed
        try:
            return parse_model_fn(d, ch, verbose)
        finally:
            _tasks.GPT = saved['GPT']
            _tasks.CrossMLCAv2 = saved['CrossMLCAv2']

    return wrapper
