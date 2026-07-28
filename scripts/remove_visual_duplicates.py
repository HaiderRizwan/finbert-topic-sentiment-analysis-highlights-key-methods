import os
from PIL import Image
import imagehash
from pathlib import Path
from tqdm import tqdm


def remove_duplicates(directory, hash_size=8, threshold=5):
    """
    Finds and removes visually duplicate images in a directory.
    Uses perceptual hashing (pHash) which is robust to minor visual differences.
    """
    print(f"\nScanning directory: {directory} (Threshold: {threshold})")

    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}

    # Store hashes to identify duplicates
    # hash -> file_path
    seen_hashes = {}
    duplicates_removed = 0

    # Ensure directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory not found - {directory}")
        return

    # Get all valid image files
    files_to_process = [f for f in dir_path.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]
    total_files = len(files_to_process)

    if total_files == 0:
        print("No images found in this directory.")
        return

    print(f"Found {total_files} images to process.")

    # Create progress bar
    progress_bar = tqdm(total=total_files, desc="Processing Frames", unit="img")

    for file_path in files_to_process:
        try:
            # Open image and compute its perceptual hash
            with Image.open(file_path) as img:
                # Convert to RGB to avoid issues with alpha channels or varied modes
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_hash = imagehash.phash(img, hash_size=hash_size)

            # Check for visual duplicates
            is_duplicate = False
            for seen_hash, original_path in seen_hashes.items():
                # Calculate Hamming distance between hashes
                # A distance <= threshold means the images are visually very similar (minor differences)
                if abs(img_hash - seen_hash) <= threshold:
                    try:
                        os.remove(file_path)
                        duplicates_removed += 1
                        is_duplicate = True
                        break
                    except Exception:
                        pass  # Ignore if failed to delete

            if not is_duplicate:
                seen_hashes[img_hash] = file_path

        except Exception:
            pass  # Ignore problematic files

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()
    print(f"Done scanning {directory}.")
    print(f"Removed {duplicates_removed} visually similar duplicates in this directory.\n")


if __name__ == "__main__":
    directories_to_scan = [
        r"c:\Users\dotsm\Desktop\FYP\SafeToon\dataset\manual dataset\erotism",
        r"c:\Users\dotsm\Desktop\FYP\SafeToon\dataset\manual dataset\normal",
        r"c:\Users\dotsm\Desktop\FYP\SafeToon\dataset\manual dataset\violent"
    ]

    # -------------------------------------------------------------------------
    # THRESHOLD EXPLANATION:
    # 0 = strictly identical images only.
    # 5 = minor visual differences (e.g. adjacent frames in a video).
    # 10 = captures larger differences.
    # Adjust this value if it is deleting too many or too few frames.
    # -------------------------------------------------------------------------
    VISUAL_DIFFERENCE_THRESHOLD = 12

    for directory in directories_to_scan:
        print("="*60)
        remove_duplicates(directory, threshold=VISUAL_DIFFERENCE_THRESHOLD)
