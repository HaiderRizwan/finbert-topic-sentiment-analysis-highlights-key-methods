"""
GPU-Accelerated Data Augmentation for the Scooby-Doo Dataset.

Reads from:
    scoobidoo dataset/erotism/
    scoobidoo dataset/violent/
    scoobidoo dataset/normal/

Writes augmented images to:
    scoobidoo dataset/generated dataset/erotism/
    scoobidoo dataset/generated dataset/violent/
    scoobidoo dataset/generated dataset/normal/

Strategy:
  - Each original image is COPIED as-is to the output folder.
  - Then N augmented variants are generated per image (auto-computed from
    --target so all classes reach roughly the same count).
  - Rich set of augmentations: flip, rotate, crop/zoom, brightness,
    contrast, saturation, sharpness, blur, colour-jitter.
  - Parallel I/O via ThreadPoolExecutor; GPU used for torchvision
    tensor ops (ColorJitter, RandomPerspective, etc.) when available.

Usage:
    python scripts/augment_scoobydoo.py
    python scripts/augment_scoobydoo.py --target 1000 --workers 8
    python scripts/augment_scoobydoo.py --clean          # wipe & redo
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from tqdm import tqdm  # noqa: E402

try:
    from src.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "scoobidoo dataset"
SOURCE_CLASSES = {
    "erotism": DATASET_ROOT / "erotism",
    "violent": DATASET_ROOT / "violent",
    "normal": DATASET_ROOT / "normal",
}
OUTPUT_ROOT = DATASET_ROOT / "generated dataset"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -------------------------------------------------------------
# Augmentation helpers
# -------------------------------------------------------------
def _random_crop_resize(img: Image.Image, crop_frac: float) -> Image.Image:
    w, h = img.size
    nw = int(w * crop_frac)
    nh = int(h * crop_frac)
    left = random.randint(0, w - nw)
    top = random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)


def _add_gaussian_noise(img: Image.Image, std: float = 0.02) -> Image.Image:
    """Add subtle Gaussian noise via tensor ops."""
    t = TF.to_tensor(img)
    noise = torch.randn_like(t) * std
    t = (t + noise).clamp(0.0, 1.0)
    return TF.to_pil_image(t)


def _tv_color_jitter(img: Image.Image) -> Image.Image:
    t = TF.to_tensor(img)
    jitter = T.ColorJitter(
        brightness=random.uniform(0.1, 0.4),
        contrast=random.uniform(0.1, 0.4),
        saturation=random.uniform(0.1, 0.4),
        hue=random.uniform(0.0, 0.15),
    )
    return TF.to_pil_image(jitter(t))


def _tv_perspective(img: Image.Image) -> Image.Image:
    t = TF.to_tensor(img)
    persp = T.RandomPerspective(distortion_scale=random.uniform(0.1, 0.3), p=1.0)
    return TF.to_pil_image(persp(t))


def build_augmentation_pool() -> list:
    """
    Return a list of augmentation callables.
    PIL Image -> PIL Image; lightweight enough to run in threads.
    Randomness is captured at call time so each invocation is fresh.
    """
    return [
        # -- Geometric ------------------------------------------
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        lambda img: img.rotate(random.uniform(-20, 20), expand=False),
        lambda img: img.rotate(90),
        lambda img: img.rotate(180),
        lambda img: img.rotate(270),
        lambda img: _random_crop_resize(img, random.uniform(0.70, 0.92)),
        lambda img: _tv_perspective(img),

        # -- Colour / photometric --------------------------------
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.6)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.6)),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.4, 1.9)),
        lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.3, 2.5)),
        lambda img: _tv_color_jitter(img),

        # -- Blur / noise ----------------------------------------
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))),
        lambda img: img.filter(ImageFilter.SHARPEN),
        lambda img: _add_gaussian_noise(img, std=random.uniform(0.01, 0.05)),

        # -- Combos ---------------------------------------------
        lambda img: ImageEnhance.Brightness(
            img.transpose(Image.FLIP_LEFT_RIGHT)
        ).enhance(random.uniform(0.7, 1.3)),

        lambda img: ImageEnhance.Contrast(
            img.rotate(random.choice([-10, -5, 5, 10]))
        ).enhance(random.uniform(0.8, 1.4)),

        lambda img: _random_crop_resize(
            img.transpose(Image.FLIP_LEFT_RIGHT), random.uniform(0.80, 0.95)
        ),

        lambda img: _tv_color_jitter(
            img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        ),
    ]


# -------------------------------------------------------------
# Per-image worker
# -------------------------------------------------------------
def augment_image(
    src_path: Path,
    out_dir: Path,
    aug_pool: list,
    n_variants: int,
) -> int:
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        print(f"\n  [WARN] Cannot open {src_path.name}: {e}")
        return 0

    written = 0
    stem = src_path.stem

    # Copy original (skip if already exists)
    orig_out = out_dir / (stem + src_path.suffix)
    if not orig_out.exists():
        try:
            img.save(orig_out, quality=95)
            written += 1
        except Exception:
            pass

    # Generate augmented variants
    chosen = random.choices(aug_pool, k=n_variants)
    for idx, aug_fn in enumerate(chosen):
        out_path = out_dir / f"{stem}_aug{idx:04d}.jpg"
        if out_path.exists():
            continue
        try:
            aug_img = aug_fn(img)
            aug_img.save(out_path, quality=90)
            written += 1
        except Exception as e:
            print(f"\n  [WARN] Aug {idx} failed for {src_path.name}: {e}")

    return written


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated data augmentation for the Scooby-Doo dataset"
    )
    parser.add_argument(
        "--target", type=int, default=800,
        help="Target total images per class in the output folder (default: 800)"
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of parallel I/O threads (default: 6)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing generated dataset before running"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    print("  SafeToon - Scooby-Doo Dataset Augmentation")
    print("=" * 62)
    print(f"  Device  : {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"  Target  : {args.target} images/class")
    print(f"  Workers : {args.workers}")
    print("=" * 62)

    aug_pool = build_augmentation_pool()
    grand_total = 0

    for cls_name, src_dir in SOURCE_CLASSES.items():
        print(f"\n-- Class: {cls_name} --")

        out_dir = OUTPUT_ROOT / cls_name

        if args.clean and out_dir.exists():
            shutil.rmtree(out_dir)
            print(f"  Cleaned: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect clean source images (ignore "- Copy" duplicates)
        src_files = sorted([
            f for f in src_dir.iterdir()
            if f.suffix.lower() in IMG_EXTS
            and " - Copy" not in f.name
            and "- Copy" not in f.name
        ])

        n_src = len(src_files)
        if n_src == 0:
            print(f"  [WARN] No source images found in {src_dir}")
            continue

        # Determine how many augmented variants per image
        target = args.target
        already_in_output = len([
            f for f in out_dir.iterdir() if f.suffix.lower() in IMG_EXTS
        ])

        if already_in_output >= target:
            print(f"  Output already has {already_in_output} images >= target {target}. Skipping.")
            continue

        remaining_needed = target - already_in_output
        n_variants = max(1, round(remaining_needed / n_src))

        print(f"  Source images    : {n_src}")
        print(f"  Already in output: {already_in_output}")
        print(f"  Need ~{remaining_needed} more  ->  {n_variants} variants/image")
        expected = already_in_output + n_src + n_src * n_variants
        print(f"  Expected total   : ~{expected}")

        # Parallel processing
        cls_written = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(augment_image, f, out_dir, aug_pool, n_variants): f
                for f in src_files
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"  {cls_name}", unit="img"
            ):
                cls_written += fut.result()

        grand_total += cls_written

        final_count = len([f for f in out_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        print(f"  Written this run : {cls_written:,}")
        print(f"  Total in output  : {final_count:,}")

    print("\n" + "=" * 62)
    print(f"  Done! Total images written: {grand_total:,}")
    print(f"  Output: {OUTPUT_ROOT}")
    print("=" * 62)

    # Final summary
    print("\n  Final class counts:")
    for cls_name in SOURCE_CLASSES:
        d = OUTPUT_ROOT / cls_name
        if d.exists():
            n = len([f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS])
            print(f"    {cls_name:10s}: {n:,} images")


if __name__ == "__main__":
    main()
