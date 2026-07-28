"""
CLI – Build the annotation sheet for the Ishikawa Elsagate dataset.

Scans all video files from dataset/archive/ and generates a CSV with columns:
    video_id, filename, source_folder, source_split, original_label, new_label,
    annotator, notes

Safe videos are pre-filled with 'normal'. Elsagate videos are left blank for
manual annotation into one of: vl, vm, vh, el, em, eh.

Usage:
    python scripts/build_annotation_sheet.py
    python scripts/build_annotation_sheet.py --archive path/to/archive --output data/annotations.csv
"""

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".webm"}


def scan_videos(archive_root: Path) -> list[dict]:
    """Walk the archive folder structure and collect video metadata."""
    rows = []

    # Sort for deterministic output
    for folder in sorted(archive_root.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name  # e.g. "elsagate_videos_fold1-007"

        # Determine original label and source split from folder name
        if "elsagate" in folder_name.lower() and "video" in folder_name.lower():
            original_label = "elsagate"
        elif "safe" in folder_name.lower() and "video" in folder_name.lower():
            original_label = "safe"
        else:
            # Skip non-video folders (frames, motions)
            continue

        if "fold1" in folder_name:
            source_split = "fold1"
        elif "fold2" in folder_name:
            source_split = "fold2"
        elif "test" in folder_name:
            source_split = "test"
        else:
            source_split = "unknown"

        # Videos are inside a nested subfolder
        for subfolder in sorted(folder.iterdir()):
            if not subfolder.is_dir():
                continue
            for video_file in sorted(subfolder.iterdir()):
                if not video_file.is_file():
                    continue
                if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                # Extract video ID from filename (e.g. elsagate_0001 from elsagate_0001.mp4)
                video_id = video_file.stem

                rows.append(
                    {
                        "video_id": video_id,
                        "filename": video_file.name,
                        "source_folder": folder_name,
                        "source_split": source_split,
                        "original_label": original_label,
                        "new_label": "normal" if original_label == "safe" else "",
                        "annotator": "",
                        "notes": "",
                        "abs_path": str(video_file),
                    }
                )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write annotation rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "filename",
        "source_folder",
        "source_split",
        "original_label",
        "new_label",
        "annotator",
        "notes",
        "abs_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Annotation sheet saved → %s (%d rows)", output_path, len(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Build annotation sheet for SafeToon dataset"
    )
    parser.add_argument(
        "--archive",
        "-a",
        default=str(cfg.dataset.raw_dataset_root),
        help="Path to the dataset/archive/ folder",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(cfg.dataset.annotation_path),
        help="Output CSV path (default: data/annotations.csv)",
    )
    args = parser.parse_args()

    archive_root = Path(args.archive)
    output_path = Path(args.output)

    if not archive_root.exists():
        print(f"[ERROR] Archive folder not found: {archive_root}")
        sys.exit(1)

    print(f"[INFO] Scanning videos in: {archive_root}")
    rows = scan_videos(archive_root)

    # Print summary
    elsagate_count = sum(1 for r in rows if r["original_label"] == "elsagate")
    safe_count = sum(1 for r in rows if r["original_label"] == "safe")
    print(f"[INFO] Found {len(rows)} videos total:")
    print(f"       - {elsagate_count} elsagate (need manual labelling)")
    print(f"       - {safe_count} safe (pre-labelled as 'normal')")

    # Per-split breakdown
    from collections import Counter

    split_counts = Counter((r["source_split"], r["original_label"]) for r in rows)
    print("\n[INFO] Breakdown by split:")
    for split in ["fold1", "fold2", "test"]:
        eg = split_counts.get((split, "elsagate"), 0)
        sf = split_counts.get((split, "safe"), 0)
        print(f"       {split}: {eg} elsagate, {sf} safe")

    write_csv(rows, output_path)
    print(f"\n[DONE] Annotation sheet saved to: {output_path}")
    print(f"[NEXT] Open {output_path} in a spreadsheet editor and fill in")
    print("       'new_label' for all elsagate videos (vl/vm/vh/el/em/eh).")
    print(f"       Valid labels: {cfg.dataset.labels}")


if __name__ == "__main__":
    main()
