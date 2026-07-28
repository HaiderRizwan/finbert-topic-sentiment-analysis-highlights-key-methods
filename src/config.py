"""
SafeToon – Centralised Configuration
All pipeline parameters live here. Import and use `cfg` everywhere.
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ─── FFmpeg auto-discovery ────────────────────────────────────────────────────
def _ensure_ffmpeg_in_path() -> None:
    """Find FFmpeg binaries and add them to PATH if not already available."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return  # already on PATH

    # Common install locations to search (Windows-centric)
    search_roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path("D:/FFmpeg"),
        Path("C:/FFmpeg"),
        Path(os.environ.get("PROGRAMFILES", "")) / "FFmpeg",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for ffmpeg_exe in root.rglob("ffmpeg.exe"):
            bin_dir = str(ffmpeg_exe.parent)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_ffmpeg_in_path()

# ─── Root paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class IngestConfig:
    """Video ingestion settings."""

    allowed_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mkv", ".webm"]
    )
    max_file_size_mb: float = 500.0  # reject files larger than this
    frame_extract_fps: float = 1.0  # frames to extract per second
    audio_chunk_sec: int = 10  # audio segment length in seconds
    audio_format: str = "wav"  # output audio format
    output_root: Path = OUTPUTS_DIR  # where job folders are created


@dataclass
class FrameConfig:
    """Frame preprocessing settings."""

    frame_format: str = "jpg"  # saved frame extension
    target_size: Tuple[int, int] = (224, 224)  # H × W for model input
    # ImageNet-style normalisation constants (mean/std per channel, RGB order)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    frame_index_path: Path = OUTPUTS_DIR / "frame_index.csv"


@dataclass
class DatasetConfig:
    """Dataset split and labelling settings.

    7-class taxonomy (AniMet-inspired):
        normal – Safe, kid-friendly content
        vl     – Violence Low    (mild slapstick, cartoon chasing)
        vm     – Violence Medium (scary scenes, moderate aggression)
        vh     – Violence High   (graphic violence, weapons, blood)
        el     – Erotic Low      (mild suggestive posing)
        em     – Erotic Medium   (partial nudity, suggestive acts)
        eh     – Erotic High     (explicit sexual content)
    """

    labels: List[str] = field(
        default_factory=lambda: ["normal", "vl", "vm", "vh", "el", "em", "eh"]
    )
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)  # train/val/test
    dataset_index_path: Path = DATA_DIR / "dataset_index.csv"
    annotation_path: Path = DATA_DIR / "annotations.csv"
    raw_dataset_root: Path = (
        Path(__file__).resolve().parent.parent / "dataset" / "archive"
    )


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# ─── Singleton ────────────────────────────────────────────────────────────────
cfg = Config()
