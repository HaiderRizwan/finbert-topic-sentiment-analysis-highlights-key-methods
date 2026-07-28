"""
SafeToon – Ingestion Pipeline
Orchestrates: validate → create_job → extract_metadata → extract_frames → extract_audio → segment.
"""

from pathlib import Path
from typing import List

from src.ingestion.extractor import (extract_audio, extract_frames,
                                     extract_metadata, segment_audio)
from src.ingestion.job import create_job, update_job_status
from src.ingestion.validator import validate_video
from src.logger import get_logger

log = get_logger(__name__)


def run_ingestion(video_path: str | Path, fps: float | None = None, output_dir: str | Path | None = None) -> dict:
    """
    Full ingestion pipeline for a single video.
    Returns a summary dict (job_id, status, counts, paths).
    """
    video_path = Path(video_path)
    log.info("═══ Starting ingestion: %s ═══", video_path.name)

    # 1. Validate
    result = validate_video(video_path)
    if not result.valid:
        log.error("Validation failed: %s", result.reason)
        return {"status": "error", "reason": result.reason, "video": str(video_path)}

    # 2. Create job
    job = create_job(video_path, output_root=output_dir)
    update_job_status(job, "running")

    try:
        # 3. Extract metadata
        meta = extract_metadata(video_path)
        update_job_status(job, "running", video_metadata=meta)

        # 4. Extract frames
        n_frames = extract_frames(video_path, job.frames_dir, fps=fps)

        # 5. Extract audio
        audio_file = extract_audio(video_path, job.audio_dir)

        # 6. Segment audio
        n_chunks = segment_audio(audio_file, job.audio_dir)

        # 7. Finalise metadata
        update_job_status(
            job,
            "done",
            n_frames=n_frames,
            n_audio_chunks=n_chunks,
            audio_path=str(audio_file),
        )

        summary = {
            "status": "done",
            "job_id": job.job_id,
            "video": str(video_path),
            "frames_dir": job.frames_dir,
            "audio_dir": job.audio_dir,
            "metadata_json": job.metadata_path,
            "n_frames": n_frames,
            "n_audio_chunks": n_chunks,
            "video_metadata": meta,
        }
        log.info(
            "Ingestion complete for job %s (%d frames, %d audio chunks)",
            job.job_id,
            n_frames,
            n_chunks,
        )
        return summary

    except Exception as exc:
        update_job_status(job, "error", error=str(exc))
        log.exception("Ingestion failed for job %s: %s", job.job_id, exc)
        return {"status": "error", "job_id": job.job_id, "reason": str(exc)}


def run_batch(folder_path: str | Path) -> List[dict]:
    """
    Run ingestion on every valid video file in *folder_path* (non-recursive).
    Returns a list of per-video summary dicts.
    """
    folder = Path(folder_path)
    from src.config import cfg

    videos = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in cfg.ingest.allowed_extensions
    ]
    log.info("Batch ingestion: found %d video(s) in %s", len(videos), folder)
    results = []
    for v in videos:
        results.append(run_ingestion(v))
    return results
