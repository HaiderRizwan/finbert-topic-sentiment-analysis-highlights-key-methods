"""
CLI – Map extracted frames to their video-level labels.

After ingestion and annotation, this script creates a frame-level CSV mapping
every extracted frame to its parent video's label, for use as DL training input.

Output columns: frame_path, video_id, label, split, frame_number

Usage:
    python scripts/map_frames_to_labels.py
    python scripts/map_frames_to_labels.py --index data/dataset_index.csv --outputs outputs/
"""

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def load_dataset_index(index_path: Path) -> dict:
    """Load dataset_index.csv and build a lookup by video filename stem."""
    lookup = {}
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_id = row["video_id"]
            lookup[vid_id] = {
                "label": row["label"],
                "split": row["split"],
                "abs_path": row["abs_path"],
            }
    return lookup


def scan_ingested_frames(outputs_root: Path, video_lookup: dict) -> list[dict]:
    """
    Scan outputs/<job_id>/frames/ folders and match frames to video labels
    using metadata.json → video_path → dataset_index lookup.
    """
    rows = []

    for job_dir in sorted(outputs_root.iterdir()):
        if not job_dir.is_dir():
            continue

        meta_file = job_dir / "metadata.json"
        frames_dir = job_dir / "frames"

        if not meta_file.exists() or not frames_dir.exists():
            continue

        # Read metadata to get source video path
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            log.warning("Could not read metadata for %s: %s", job_dir.name, e)
            continue

        video_path = meta.get("video_path", "")
        video_id = Path(video_path).stem if video_path else ""

        # Look up label and split
        info = video_lookup.get(video_id, {})
        label = info.get("label", "unknown")
        split = info.get("split", "unknown")

        # Scan frames
        for frame_file in sorted(frames_dir.iterdir()):
            if frame_file.suffix.lower() not in (".jpg", ".png"):
                continue
            try:
                frame_number = int(frame_file.stem.split("_")[-1])
            except ValueError:
                frame_number = -1

            rows.append(
                {
                    "frame_path": str(frame_file),
                    "video_id": video_id,
                    "label": label,
                    "split": split,
                    "frame_number": frame_number,
                    "job_id": job_dir.name,
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Map extracted frames to video labels")
    parser.add_argument(
        "--index",
        "-i",
        default=str(cfg.dataset.dataset_index_path),
        help="Path to dataset_index.csv",
    )
    parser.add_argument(
        "--outputs",
        "-o",
        default=str(cfg.ingest.output_root),
        help="Path to outputs/ folder with ingested job folders",
    )
    parser.add_argument(
        "--save",
        "-s",
        default=str(cfg.frames.frame_index_path),
        help="Output path for frame-level label CSV",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    outputs_root = Path(args.outputs)
    save_path = Path(args.save)

    if not index_path.exists():
        print(f"[ERROR] Dataset index not found: {index_path}")
        print("[HINT] Run organise_dataset.py first to create the dataset index.")
        sys.exit(1)

    if not outputs_root.exists():
        print(f"[ERROR] Outputs folder not found: {outputs_root}")
        print("[HINT] Run ingestion first to extract frames.")
        sys.exit(1)

    print(f"[INFO] Loading dataset index: {index_path}")
    video_lookup = load_dataset_index(index_path)
    print(f"[INFO] Found {len(video_lookup)} videos in index")

    print(f"[INFO] Scanning ingested frames in: {outputs_root}")
    rows = scan_ingested_frames(outputs_root, video_lookup)
    print(f"[INFO] Found {len(rows)} frames across all jobs")

    if not rows:
        print("[WARNING] No frames found. Run ingestion first.")
        sys.exit(0)

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_path", "video_id", "label", "split", "frame_number", "job_id"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Frame-label mapping saved → {save_path}")
    print(f"[INFO] {len(rows)} frames mapped to labels")

    # Summary
    from collections import Counter

    label_counts = Counter(r["label"] for r in rows)
    print("\nFrame distribution by label:")
    for label in cfg.dataset.labels:
        print(f"  {label}: {label_counts.get(label, 0)}")


if __name__ == "__main__":
    main()
