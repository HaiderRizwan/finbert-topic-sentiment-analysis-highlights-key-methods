import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Provide resolving for src local importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROJECT_ROOT  # noqa: E402
from scripts.train_models import get_model  # noqa: E402

CLASSES = ["erotism", "normal", "violent"]
DATASET_PATH = PROJECT_ROOT / "scoobidoo dataset"
GENERATED_PATH = DATASET_PATH / "generated dataset"

# Output directories for classified frames
OUTPUT_DIRS = {
    "erotism": DATASET_PATH / "erotism",
    "normal": DATASET_PATH / "normal",
    "violent": DATASET_PATH / "violent"
}

# Create output dirs
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Image pre-processing matching the training pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main(model_name="nasnet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing Classification Pipeline.")
    print("\n  Loading NASNet base model...")
    print(f"Model: {model_name} | Device: {device}")

    # Initialize and load weights
    model = get_model(model_name, num_classes=len(CLASSES), pretrained=False)
    weights_path = PROJECT_ROOT / "models" / f"{model_name}_best.pth"

    if not weights_path.exists():
        print(f"ERROR: Model weights not found at {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Find all frame files
    frame_files = []
    print(f"Processing frames in: {GENERATED_PATH}")
    for root, _, files in os.walk(GENERATED_PATH):
        # Only look into 'frames' folders natively created by ingestion
        if "frames" in Path(root).parts:
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    frame_files.append(Path(root) / f)

    if not frame_files:
        print("No frames found to process.")
        return

    print(f"Found {len(frame_files)} frames. Starting classification...")

    batch_size = 32
    for i in tqdm(range(0, len(frame_files), batch_size), desc="Classifying Batches"):
        batch_paths = frame_files[i:i+batch_size]
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"Failed to load frame {p}: {e}")

        if not images:
            continue

        tensor_batch = torch.stack(images).to(device)

        with torch.no_grad():
            output = model(tensor_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, top_class_idx = torch.max(probabilities, 1)

        for path, cls_idx in zip(valid_paths, top_class_idx):
            class_name = CLASSES[cls_idx.item()]
            dest_dir = OUTPUT_DIRS[class_name]

            # Create a unique filename so frames from different videos don't overwrite each other
            parent_name = path.parent.parent.name.replace(" ", "_")
            new_filename = f"{parent_name}_{path.name}"
            dest_path = dest_dir / new_filename

            # Copy frame to corresponding class folder
            shutil.copy2(path, dest_path)

    print("\n✅ Classification Complete!")
    print("Summary of frames copied into destination folders:")
    for cls, dr in OUTPUT_DIRS.items():
        count = len(list(dr.glob("*.*")))
        print(f"  - {cls}: {count} frames")


if __name__ == "__main__":
    main("nasnet")  # Utilizing nasnet by default, could also be mobilenet
