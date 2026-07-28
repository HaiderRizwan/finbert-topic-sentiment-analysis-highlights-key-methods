"""
SafeToon – Ingestion Job Manager
Creates a unique job for every video ingested.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class Job:
    job_id: str
    video_path: str
    created_at: str  # ISO 8601
    frames_dir: str
    audio_dir: str
    metadata_path: str
    status: str = "created"  # created | running | done | error


def create_job(video_path: str | Path, output_root: str | Path | None = None) -> Job:
    """
    Assign a unique job_id to *video_path* and build the output folder tree:
        outputs/<job_id>/
            frames/
            audio/
            metadata.json   ← written immediately with initial state
    Returns a populated Job object.
    """
    video_path = Path(video_path)
    # Sanitize stem slightly just in case, but keep it readable
    job_id = "".join(c for c in video_path.stem if c.isalnum() or c in (" ", "_", "-")).strip()
    if not job_id:
        job_id = str(uuid.uuid4())

    base_root = Path(output_root) if output_root is not None else cfg.ingest.output_root
    job_root = base_root / job_id
    frames_dir = job_root / "frames"
    audio_dir = job_root / "audio"

    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    meta_path = job_root / "metadata.json"

    job = Job(
        job_id=job_id,
        video_path=str(video_path),
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        frames_dir=str(frames_dir),
        audio_dir=str(audio_dir),
        metadata_path=str(meta_path),
        status="created",
    )

    _write_metadata(job)
    log.info("Job created: %s → %s", job_id, job_root)
    return job


def update_job_status(job: Job, status: str, **extra) -> None:
    """Update job status and optionally merge extra fields into metadata.json."""
    job.status = status
    if extra:
        meta = _read_metadata(job.metadata_path)
        meta.update(extra)
        meta["status"] = status
        _write_raw(job.metadata_path, meta)
    else:
        _write_metadata(job)
    log.debug("Job %s → status: %s", job.job_id, status)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _write_metadata(job: Job) -> None:
    _write_raw(job.metadata_path, asdict(job))


def _write_raw(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _read_metadata(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
