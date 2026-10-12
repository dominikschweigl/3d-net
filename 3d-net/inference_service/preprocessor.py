import time

import cv2
import numpy as np

from .config import ModalityMode
from .frame_provider import FramePair, PreparedFrame

MODEL_INPUT_SIZE = 640


class Preprocessor:
    """
    Transforms a FramePair into a PreparedFrame ready for Hailo inference.

    Stateless after construction — safe to call from multiple threads concurrently.
    """

    def __init__(
        self,
        model_input_size: int = MODEL_INPUT_SIZE,
        modality: ModalityMode = ModalityMode.RGBT,
    ) -> None:
        self._size = model_input_size
        self._modality = modality

    def prepare(self, seq: int, pair: FramePair) -> PreparedFrame:
        t0 = time.monotonic()
        if self._modality == ModalityMode.RGBT:
            img = np.concatenate(
                [pair.rgb_crop, pair.gray_crop[:, :, np.newaxis]], axis=2
            )
        elif self._modality == ModalityMode.THERMAL:
            img = np.repeat(pair.gray_crop[:, :, np.newaxis], 3, axis=2)
        else:  # RGB
            img = pair.rgb_crop
        tensor, lb_scale, lb_pad_x, lb_pad_y = self._letterbox(img)
        tensor = np.expand_dims(tensor, axis=0)
        return PreparedFrame(
            seq=seq,
            pair=pair,
            tensor=tensor,
            lb_scale=lb_scale,
            lb_pad_x=lb_pad_x,
            lb_pad_y=lb_pad_y,
            t_align=pair.align_s,
            t_pre=time.monotonic() - t0,
        )

    def _letterbox(self, img: np.ndarray) -> tuple:
        """Scale img to fit within size×size preserving aspect ratio; pad remainder with grey (114).

        Matches Ultralytics' LetterBox used during training so the model sees the same
        geometric distribution at inference that it was trained on.

        Returns (padded_img, scale, pad_left, pad_top).
        """
        h, w = img.shape[:2]
        scale = min(self._size / w, self._size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        dw, dh = (self._size - new_w) / 2, (self._size - new_h) / 2
        pad_left = int(round(dw - 0.1))
        pad_top  = int(round(dh - 0.1))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self._size, self._size, img.shape[2]), 114, dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return padded, scale, pad_left, pad_top
