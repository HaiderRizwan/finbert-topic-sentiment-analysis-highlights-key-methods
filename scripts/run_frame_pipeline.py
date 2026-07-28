"""
CLI – Run the full frame extraction pipeline on an ingested outputs folder.
Builds and saves the frame index CSV.

Usage:
    python scripts/run_frame_pipeline.py
    python scripts/run_frame_pipeline.py --outputs path/to/outputs/
"""

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg  # noqa: E402
from src.frames.indexer import build_index, save_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="SafeToon Frame Pipeline")
    parser.add_argument(
        "--outputs",
        "-o",
        default=str(cfg.ingest.output_root),
        help="Path to the outputs/ folder containing job sub-directories",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs)
    if not outputs_root.exists():
        print(f"[ERROR] Outputs folder not found: {outputs_root}")
        sys.exit(1)

    print(f"[INFO] Building frame index from: {outputs_root}")
    df = build_index(outputs_root)

    if df.empty:
        print("[WARNING] No frames found. Run ingestion first.")
        sys.exit(0)

    index_path = save_index(df)
    print(f"[INFO] Frame index saved → {index_path}  ({len(df)} rows)")

    # Quick sanity-check: preprocess first 5 frames of first job
    first_job = df["job_id"].iloc[0]
    job_frames = df[df["job_id"] == first_job].head(5)
    print(f"\n[INFO] Sanity-checking {len(job_frames)} frame(s) from job: {first_job}")
    for _, row in job_frames.iterrows():
        import cv2

        frame = cv2.imread(row["frame_path"])
        if frame is None:
            print(f"  [WARN] Could not read: {row['frame_path']}")
            continue
        from src.frames.preprocessor import preprocess

        processed = preprocess(frame)
        print(
            f"  frame_{row['frame_number']:06d}  shape={processed.shape}  "
            f"min={processed.min():.3f}  max={processed.max():.3f}"
        )

    print("\n[DONE] Frame pipeline complete.")


if __name__ == "__main__":
    main()
