"""
SafeToon – Video File Validator
Checks a video file before ingestion starts.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import cfg
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    reason: Optional[str] = None  # populated when valid=False


def validate_video(path: str | Path) -> ValidationResult:
    """
    Run all validation checks on *path* and return a ValidationResult.

    Checks (in order):
        1. File exists
        2. Extension is in the allowed list
        3. File size is within the limit
        4. FFmpeg can read the file (probe without decoding)
    """
    p = Path(path)

    # 1. Existence
    if not p.exists():
        log.warning("Validation failed – file not found: %s", p)
        return ValidationResult(valid=False, reason=f"File not found: {p}")

    # 2. Extension
    ext = p.suffix.lower()
    if ext not in cfg.ingest.allowed_extensions:
        log.warning("Validation failed – unsupported extension '%s': %s", ext, p)
        return ValidationResult(
            valid=False,
            reason=f"Unsupported extension '{ext}'. Allowed: {cfg.ingest.allowed_extensions}",
        )

    # 3. File size
    size_mb = p.stat().st_size / (1024**2)
    if size_mb > cfg.ingest.max_file_size_mb:
        log.warning("Validation failed – file too large (%.1f MB): %s", size_mb, p)
        return ValidationResult(
            valid=False,
            reason=f"File too large: {size_mb:.1f} MB (limit {cfg.ingest.max_file_size_mb} MB)",
        )

    # 4. FFmpeg readability
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-i", str(p)],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace").strip()
            log.warning("Validation failed – ffprobe error for %s: %s", p, err)
            return ValidationResult(valid=False, reason=f"FFprobe error: {err}")
    except FileNotFoundError:
        log.error("ffprobe not found in PATH. Is FFmpeg installed?")
        return ValidationResult(valid=False, reason="ffprobe not found in PATH")
    except subprocess.TimeoutExpired:
        log.warning("Validation failed – ffprobe timed out: %s", p)
        return ValidationResult(valid=False, reason="FFprobe timed out")

    log.info("Validation passed: %s (%.1f MB)", p.name, size_mb)
    return ValidationResult(valid=True)
