"""
Classifies every frame (.jpg) in ScoobyDoo Dataset2 using the fine-tuned
NASNet model (nasnet_scoobydoo_best.pth).

Each frame is passed through the model and copied into:
  ScoobyDoo Dataset2/classified/
    safe/
    violent/
    erotism/

Usage:
    python scripts/classify_scoobydoo2.py
"""

import os
import sys
import shutil
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torchvision import transforms  # noqa: E402

from scripts.train_models import get_model  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATASET2_DIR = PROJECT_ROOT / "ScoobyDoo Dataset2"
OUTPUT_DIR = DATASET2_DIR / "classified"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "nasnet_scoobydoo_best.pth"

CLASSES = ['erotism', 'normal', 'violent']     # alphabetical — PyTorch order
MAPPING = {
    'normal':  'safe',
    'violent': 'violent',
    'erotism': 'erotism',
}

# Same normalization as training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(device):
    print(f"Loading model from: {WEIGHTS_PATH}")
    model = get_model("nasnet", num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    print("Model loaded successfully.\n")
    return model


def classify_frames(model, device):
    # Create output category folders
    for cat in MAPPING.values():
        (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    # Gather all .jpg frames recursively (skip the classified folder itself)
    all_frames = [
        p for p in DATASET2_DIR.rglob("*.jpg")
        if "classified" not in p.parts
    ]
    print(f"Found {len(all_frames)} frames to classify.\n")

    counts = {cat: 0 for cat in MAPPING.values()}

    for frame_path in tqdm(all_frames, desc="Classifying frames"):
        try:
            img = Image.open(frame_path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_idx = torch.argmax(probs).item()

            predicted_class = CLASSES[top_idx]
            category = MAPPING[predicted_class]

            # Build a globally unique filename by always prefixing with the
            # source video folder name. Each video restarts its frame counter
            # at 0, so frame_000284 from Video A and frame_000284 from Video B
            # are completely different scenes. Without this prefix they would
            # collide and only one copy would survive in the output folder.
            #
            # frame_path layout:
            #   ScoobyDoo Dataset2 / <video_folder> / frames / frame_XXXXXX.jpg
            # So frame_path.parent       = frames/
            #    frame_path.parent.parent = <video_folder>/
            video_folder = frame_path.parent.parent.name   # e.g. "YTDowncom_...Damsels..."
            # Truncate to first 40 chars to keep filenames manageable
            video_id = video_folder[:40].rstrip("_")
            unique_name = f"{video_id}__{frame_path.name}"

            dest = OUTPUT_DIR / category / unique_name
            shutil.copy(frame_path, dest)

            counts[category] += 1

        except Exception as e:
            print(f"\n[ERROR] Could not process {frame_path.name}: {e}")

    return counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  SafeToon - ScoobyDoo Dataset 2 Frame Classifier")
    print("=" * 55)
    print(f"  Input  : {DATASET2_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Device : {device}")
    print("  Model  : nasnet_scoobydoo_best.pth")
    print("=" * 55 + "\n")

    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Model weights not found at: {WEIGHTS_PATH}")
        print("Make sure you have run finetune_scoobydoo.py first.")
        return

    model = load_model(device)
    counts = classify_frames(model, device)

    print("\n" + "=" * 55)
    print("  Classification Complete!")
    print("=" * 55)
    print(f"  safe     : {counts['safe']} frames")
    print(f"  violent  : {counts['violent']} frames")
    print(f"  erotism  : {counts['erotism']} frames")
    print(f"  Total    : {sum(counts.values())} frames")
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
