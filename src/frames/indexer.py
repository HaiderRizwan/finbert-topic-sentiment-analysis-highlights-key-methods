"""
SafeToon – Frame Index Builder
Scans all job output folders and builds a CSV mapping every frame to its job/video.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)

INDEX_COLUMNS = ["frame_path", "job_id", "video_path", "frame_number"]


def build_index(outputs_root: str | Path | None = None) -> pd.DataFrame:
    """
    Walk *outputs_root* (defaults to cfg.ingest.output_root) and collect info
    for every frame_XXXXXX.* file found.

    Returns a DataFrame with columns: frame_path, job_id, video_path, frame_number.
    """
    root = Path(outputs_root or cfg.ingest.output_root)
    rows: List[Dict] = []

    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir():
            continue
        meta_file = job_dir / "metadata.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            log.warning("Could not read metadata for job %s: %s", job_dir.name, e)
            continue

        job_id = meta.get("job_id", job_dir.name)
        video_path = meta.get("video_path", "")
        frames_dir = job_dir / "frames"

        if not frames_dir.exists():
            continue

        for frame_file in sorted(frames_dir.iterdir()):
            if frame_file.suffix.lower() not in (".jpg", ".png"):
                continue
            # Extract frame number from name like frame_000042.jpg
            try:
                frame_number = int(frame_file.stem.split("_")[-1])
            except ValueError:
                frame_number = -1

            rows.append(
                {
                    "frame_path": str(frame_file),
                    "job_id": job_id,
                    "video_path": video_path,
                    "frame_number": frame_number,
                }
            )

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    log.info(
        "Built frame index: %d entries across %d job(s).",
        len(df),
        df["job_id"].nunique(),
    )
    return df


def save_index(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Save the index DataFrame to a CSV file. Returns the saved path."""
    out = Path(path or cfg.frames.frame_index_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Frame index saved → %s (%d rows)", out, len(df))
    return out


def load_index(path: str | Path | None = None) -> pd.DataFrame:
    """Load and return the frame index CSV as a DataFrame."""
    p = Path(path or cfg.frames.frame_index_path)
    if not p.exists():
        raise FileNotFoundError(f"Frame index not found: {p}")
    df = pd.read_csv(p)
    log.info("Frame index loaded: %d rows from %s", len(df), p)
    return df
