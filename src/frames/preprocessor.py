"""
SafeToon – Frame Preprocessor
Converts raw BGR frames to normalised float tensors suitable for model input.
"""

from typing import Tuple

import cv2
import numpy as np

from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)

# Normalisation constants (ImageNet, RGB order) – stored here for reproducibility
MEAN = np.array(cfg.frames.mean, dtype=np.float32)  # (3,)
STD = np.array(cfg.frames.std, dtype=np.float32)  # (3,)


def preprocess(
    frame: np.ndarray,
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Preprocess a single BGR frame (H, W, 3) uint8.

    Steps:
        1. BGR → RGB
        2. Resize to *target_size* (H, W) using bilinear interpolation
        3. Scale pixel values from [0, 255] → [0.0, 1.0]
        4. Normalise: (pixel - mean) / std  (per channel, ImageNet constants)

    Returns:
        np.ndarray of shape (H, W, 3), dtype float32, values roughly in [-3, 3].
    """
    target_size = target_size or cfg.frames.target_size  # (H, W)

    # 1. Color conversion
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Resize (cv2 expects (W, H))
    resized = cv2.resize(
        rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )

    # 3. Normalise to [0, 1]
    scaled = resized.astype(np.float32) / 255.0

    # 4. ImageNet normalisation
    normalised = (scaled - MEAN) / STD

    log.debug(
        "Preprocessed frame: shape=%s  min=%.3f  max=%.3f",
        normalised.shape,
        normalised.min(),
        normalised.max(),
    )
    return normalised


def preprocess_batch(
    frames: list[np.ndarray],
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Preprocess a list of BGR frames.
    Returns np.ndarray of shape (N, H, W, 3).
    """
    processed = [preprocess(f, target_size) for f in frames]
    batch = np.stack(processed, axis=0)
    log.info("Preprocessed batch: shape=%s", batch.shape)
    return batch
