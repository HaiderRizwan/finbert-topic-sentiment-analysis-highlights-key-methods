"""
CLI – Organise the annotated dataset into train/val/test folders by label.

Reads the completed annotations.csv and creates symlinks (Windows junctions)
into the standard folder structure:
    data/videos/train/{normal,vl,vm,vh,el,em,eh}/
    data/videos/val/{normal,vl,vm,vh,el,em,eh}/
    data/videos/test/{normal,vl,vm,vh,el,em,eh}/

Uses the Ishikawa fold1/fold2 as train, test as test, and carves out a
validation set from fold2 (or uses a configurable ratio).

Usage:
    python scripts/organise_dataset.py
    python scripts/organise_dataset.py --annotations data/annotations.csv
    python scripts/organise_dataset.py --mode copy   # copy instead of symlink
"""

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def load_annotations(csv_path: Path) -> list[dict]:
    """Load and validate the annotations CSV."""
    if not csv_path.exists():
        print(f"[ERROR] Annotations file not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Validation
    valid_labels = set(cfg.dataset.labels)
    errors = []
    for i, row in enumerate(rows, start=2):  # row 2 = first data row
        label = row.get("new_label", "").strip().lower()
        if not label:
            errors.append(f"Row {i}: missing new_label for {row.get('video_id', '?')}")
        elif label not in valid_labels:
            errors.append(
                f"Row {i}: invalid label '{label}' for {row.get('video_id', '?')}"
            )

    if errors:
        print(f"[ERROR] Found {len(errors)} annotation errors:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    return rows


def assign_splits(rows: list[dict]) -> list[dict]:
    """
    Map Ishikawa splits to SafeToon splits:
        fold1 → train
        fold2 → split into train + val (proportionally)
        test  → test

    Maintains class balance when splitting fold2.
    """
    import random

    random.seed(42)

    # Target: 70% train, 15% val, 15% test
    # fold1 ≈ 40% of data, fold2 ≈ 40%, test ≈ 20%
    # So: fold1 → all train, fold2 → ~75% train + ~25% val, test → all test
    # This gives roughly: train ≈ 70%, val ≈ 10%, test ≈ 20%

    result = []

    # Group fold2 by label for balanced val split
    fold2_by_label = defaultdict(list)

    for row in rows:
        source = row["source_split"]
        if source == "fold1":
            row["assigned_split"] = "train"
            result.append(row)
        elif source == "test":
            row["assigned_split"] = "test"
            result.append(row)
        elif source == "fold2":
            label = row["new_label"].strip().lower()
            fold2_by_label[label].append(row)

    # Split fold2: 75% train, 25% val (to get overall ~70/15/15)
    val_ratio = 0.25
    for label, label_rows in fold2_by_label.items():
        random.shuffle(label_rows)
        n_val = max(1, int(len(label_rows) * val_ratio))
        for row in label_rows[:n_val]:
            row["assigned_split"] = "val"
            result.append(row)
        for row in label_rows[n_val:]:
            row["assigned_split"] = "train"
            result.append(row)

    return result


def create_links(rows: list[dict], output_root: Path, mode: str = "symlink") -> None:
    """Create the folder structure with symlinks or copies."""
    # Create all label directories
    for split in ["train", "val", "test"]:
        for label in cfg.dataset.labels:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for row in rows:
        split = row["assigned_split"]
        label = row["new_label"].strip().lower()
        src = Path(row["abs_path"])
        dst = output_root / split / label / row["filename"]

        if not src.exists():
            log.warning("Source not found, skipping: %s", src)
            skipped += 1
            continue

        if dst.exists():
            skipped += 1
            continue

        if mode == "symlink":
            # On Windows, create a hard link or junction (symlinks need admin)
            try:
                os.link(str(src), str(dst))
            except OSError:
                # Fallback: copy if hard link fails (e.g. cross-drive)
                shutil.copy2(str(src), str(dst))
                log.debug("Hard link failed, copied instead: %s", dst.name)
        else:
            shutil.copy2(str(src), str(dst))

        created += 1

    log.info("Created %d links/copies, skipped %d", created, skipped)


def print_distribution(rows: list[dict]) -> None:
    """Print class distribution per split."""
    dist = Counter((r["assigned_split"], r["new_label"].strip().lower()) for r in rows)

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    header = f"{'Label':<10}" + "".join(f"{'train':>8}{'val':>8}{'test':>8}")
    print(header)
    print("-" * 34)

    totals = {"train": 0, "val": 0, "test": 0}
    for label in cfg.dataset.labels:
        tr = dist.get(("train", label), 0)
        va = dist.get(("val", label), 0)
        te = dist.get(("test", label), 0)
        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te
        print(f"{label:<10}{tr:>8}{va:>8}{te:>8}")

    print("-" * 34)
    print(f"{'TOTAL':<10}{totals['train']:>8}{totals['val']:>8}{totals['test']:>8}")
    total = sum(totals.values())
    print(
        f"{'%':<10}{totals['train'] / total * 100:>7.1f}%"
        f"{totals['val'] / total * 100:>7.1f}%"
        f"{totals['test'] / total * 100:>7.1f}%"
    )
    print("=" * 60)


def save_dataset_index(rows: list[dict]) -> None:
    """Save the final dataset index CSV."""
    output_path = cfg.dataset.dataset_index_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "filename",
        "split",
        "label",
        "abs_path",
        "original_label",
        "source_split",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "filename": row["filename"],
                    "split": row["assigned_split"],
                    "label": row["new_label"].strip().lower(),
                    "abs_path": row["abs_path"],
                    "original_label": row["original_label"],
                    "source_split": row["source_split"],
                }
            )

    print(f"[INFO] Dataset index saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Organise SafeToon dataset by label")
    parser.add_argument(
        "--annotations",
        "-a",
        default=str(cfg.dataset.annotation_path),
        help="Path to annotations.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(cfg.ingest.output_root.parent / "data" / "videos"),
        help="Output root for organised videos (default: data/videos/)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks/hard-links (default) or copies",
    )
    args = parser.parse_args()

    print("[INFO] Loading annotations...")
    rows = load_annotations(Path(args.annotations))
    print(f"[INFO] Loaded {len(rows)} annotated videos")

    print("[INFO] Assigning train/val/test splits...")
    rows = assign_splits(rows)

    print_distribution(rows)

    output_root = Path(args.output)
    print(f"\n[INFO] Creating {args.mode}s in: {output_root}")
    create_links(rows, output_root, mode=args.mode)

    save_dataset_index(rows)

    print("\n[DONE] Dataset organised successfully!")
    print(
        f"[INFO] Structure: {output_root}/{{train,val,test}}/{{normal,vl,vm,vh,el,em,eh}}/"
    )


if __name__ == "__main__":
    main()
