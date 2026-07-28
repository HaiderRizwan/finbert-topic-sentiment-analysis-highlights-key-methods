"""
SafeToon – Frame & Audio Extractor
Handles metadata extraction, frame extraction, audio extraction, and audio segmentation.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2

from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)


# ─── Metadata ─────────────────────────────────────────────────────────────────


def extract_metadata(video_path: str | Path) -> Dict[str, Any]:
    """
    Use ffprobe to extract video metadata.
    Returns dict with: duration, fps, width, height, has_audio.
    """
    p = str(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        p,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    except subprocess.CalledProcessError as e:
        log.error("ffprobe failed for %s: %s", p, e.stderr.decode(errors="replace"))
        raise

    probe = json.loads(result.stdout)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # Parse fps safely
    fps_str = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0

    meta = {
        "duration": float(fmt.get("duration", 0)),
        "fps": round(fps, 3),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "has_audio": has_audio,
        "format": fmt.get("format_name", "unknown"),
        "size_bytes": int(fmt.get("size", 0)),
    }
    log.debug("Metadata for %s: %s", Path(p).name, meta)
    return meta


# ─── Frame extraction ─────────────────────────────────────────────────────────


def extract_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    fps: float | None = None,
    prefix: str = "frame",
) -> int:
    """
    Extract frames from *video_path* at *fps* frames/second using OpenCV.
    Saves frames as {prefix}_XXXXXX.jpg inside *frames_dir*.
    Returns the number of frames saved.
    """
    fps = fps or cfg.ingest.frame_extract_fps
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, round(native_fps / fps))  # skip this many frames

    frame_idx = 0  # total frames read
    saved = 0  # frames actually saved

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            filename = frames_dir / f"{prefix}_{saved:06d}.{cfg.frames.frame_format}"
            success = cv2.imwrite(str(filename), frame)
            if success:
                saved += 1
            else:
                log.warning("Failed to save frame: %s (Check path length or permissions)", filename)
        frame_idx += 1

    cap.release()
    log.info(
        "Extracted %d frames from %s (every %d native frames)",
        saved,
        Path(str(video_path)).name,
        sample_every,
    )
    return saved


# ─── Audio extraction & segmentation ─────────────────────────────────────────


def extract_audio(video_path: str | Path, audio_dir: str | Path) -> Path | None:
    """
    Extract full audio track from *video_path* and save as audio.wav in *audio_dir*.
    Returns the Path to audio.wav, or None if no audio stream.
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / "audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # drop video
        "-acodec",
        "pcm_s16le",  # standard PCM WAV
        "-ar",
        "16000",  # 16 kHz
        "-ac",
        "1",  # mono
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        if "no audio" in stderr.lower() or "Output file does not contain" in stderr:
            log.warning("No audio stream found in %s", video_path)
            return None
        log.error("Audio extraction failed for %s: %s", video_path, stderr)
        raise RuntimeError(f"FFmpeg audio extraction failed: {stderr}")

    log.info("Audio extracted → %s", out_path)
    return out_path


def segment_audio(
    audio_path: str | Path, audio_dir: str | Path, chunk_sec: int | None = None
) -> int:
    """
    Split *audio_path* into fixed-length chunks saved as chunk_XXXX.wav inside *audio_dir*.
    Returns the number of chunks created.
    """
    if audio_path is None:
        log.debug("Skipping segmentation – no audio file.")
        return 0

    chunk_sec = chunk_sec or cfg.ingest.audio_chunk_sec
    audio_dir = Path(audio_dir)
    out_pattern = str(audio_dir / "chunk_%04d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-c",
        "copy",
        out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        log.error(
            "Audio segmentation failed: %s", result.stderr.decode(errors="replace")
        )
        raise RuntimeError("FFmpeg segmentation failed")

    chunks = list(Path(audio_dir).glob("chunk_*.wav"))
    log.info("Audio segmented into %d chunk(s) of %ds each.", len(chunks), chunk_sec)
    return len(chunks)
