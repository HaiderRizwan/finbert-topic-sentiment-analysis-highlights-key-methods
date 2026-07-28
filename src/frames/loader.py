"""
SafeToon – Frame Loader
Reads frame images from disk as NumPy arrays.
"""

from pathlib import Path

import cv2
import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


def load_frame(path: str | Path) -> np.ndarray:
    """
    Load an image from *path* and return a BGR NumPy array (H, W, 3) uint8.
    Raises FileNotFoundError or IOError on failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Frame not found: {p}")

    frame = cv2.imread(str(p))
    if frame is None:
        raise IOError(f"OpenCV could not read frame: {p}")

    log.debug("Loaded frame: %s  shape=%s", p.name, frame.shape)
    return frame


def load_frames_from_dir(frames_dir: str | Path) -> list[np.ndarray]:
    """
    Load all .jpg/.png frames from *frames_dir* in sorted order.
    Returns a list of BGR NumPy arrays.
    """
    frames_dir = Path(frames_dir)
    paths = sorted(frames_dir.glob("frame_*.jpg")) + sorted(
        frames_dir.glob("frame_*.png")
    )
    frames = [load_frame(p) for p in paths]
    log.info("Loaded %d frames from %s", len(frames), frames_dir)
    return frames
