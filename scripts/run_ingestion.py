"""
CLI – Run ingestion on a single video file or a folder.

Usage:
    python scripts/run_ingestion.py --input path/to/video.mp4
    python scripts/run_ingestion.py --input path/to/folder/
"""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import run_batch, run_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="SafeToon Video Ingestion")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a single video file or a folder containing videos",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame extraction rate (frames per second). Default: from config (1 fps)",
    )
    args = parser.parse_args()

    target = Path(args.input)
    if not target.exists():
        print(f"[ERROR] Path not found: {target}")
        sys.exit(1)

    if target.is_file():
        result = run_ingestion(target)
        print(json.dumps(result, indent=2, default=str))
    elif target.is_dir():
        results = run_batch(target)
        print(json.dumps(results, indent=2, default=str))
        ok = sum(1 for r in results if r.get("status") == "done")
        err = sum(1 for r in results if r.get("status") == "error")
        print(
            f"\n[SUMMARY] {ok} succeeded, {err} failed out of {len(results)} video(s)."
        )
    else:
        print(f"[ERROR] Not a file or directory: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
