"""
Fine-tunes the existing NASNet model (trained on the original dataset)
on the augmented Scooby-Doo dataset.

Strategy: Fine-Tuning (Transfer from existing weights)
  - Loads pre-trained weights from models/nasnet_best.pth
  - Splits the Scooby-Doo generated dataset into train/val/test (70/15/15)
  - Trains the model further to update weights for Scooby-Doo content
  - Saves updated weights to models/nasnet_scoobydoo_best.pth

Usage:
    python scripts/finetune_scoobydoo.py
    python scripts/finetune_scoobydoo.py --epochs 15 --lr 0.0001
"""

from pathlib import Path
import os
import sys
import argparse
import shutil
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.train_models import get_model  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
CLASSES = ['erotism', 'normal', 'violent']
SOURCE_DIR = PROJECT_ROOT / "scoobidoo dataset" / "generated dataset"
SPLIT_DIR = PROJECT_ROOT / "scoobidoo dataset" / "split"
MODEL_DIR = PROJECT_ROOT / "models"
PRETRAINED_WEIGHTS = MODEL_DIR / "nasnet_best.pth"
BEST_OUT_PATH = MODEL_DIR / "nasnet_scoobydoo_best.pth"
CKPT_OUT_PATH = MODEL_DIR / "nasnet_scoobydoo_checkpoint.pth"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Test is the remainder (0.15)


# -------------------------------------------------------
# Step 1 — Dataset Split
# -------------------------------------------------------
def split_dataset(source_dir: Path, split_dir: Path):
    print("\n[1/3] Splitting augmented Scooby-Doo dataset...")

    if split_dir.exists():
        print(f"  Split directory already exists at: {split_dir}")
        print("  Skipping split (delete the folder to re-split).")
        return

    for subset in ["train", "val", "test"]:
        for cls in CLASSES:
            (split_dir / subset / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        cls_dir = source_dir / cls
        if not cls_dir.is_dir():
            print(f"  [WARN] Class folder not found: {cls_dir}")
            continue

        all_images = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        ]
        random.shuffle(all_images)

        n_total = len(all_images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        splits = {
            "train": all_images[:n_train],
            "val":   all_images[n_train:n_train + n_val],
            "test":  all_images[n_train + n_val:],
        }

        for subset, files in splits.items():
            for f in files:
                shutil.copy(f, split_dir / subset / cls / f.name)

        print(f"  {cls}: {n_train} train | {n_val} val | {n_total - n_train - n_val} test  (total: {n_total})")

    print("  Split complete.")


# -------------------------------------------------------
# Step 2 — Data Loaders
# -------------------------------------------------------
def get_dataloaders(split_dir: Path, batch_size: int, num_workers: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = datasets.ImageFolder(str(split_dir / "train"), transform=train_transform)
    val_data = datasets.ImageFolder(str(split_dir / "val"), transform=val_test_transform)
    test_data = datasets.ImageFolder(str(split_dir / "test"), transform=val_test_transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    print(f"\n  Dataset classes detected: {train_data.classes}")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    return train_loader, val_loader, test_loader


# -------------------------------------------------------
# Step 3 — Fine-Tuning Loop
# -------------------------------------------------------
def finetune(model, train_loader, val_loader, num_epochs, lr, device):
    MODEL_DIR.mkdir(exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Use a lower learning rate for fine-tuning so we don't
    # destroy the existing pretrained knowledge
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Mixed Precision Scaler for faster training on RTX 5070 Ti
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0

    print(f"\n[3/3] Fine-tuning for {num_epochs} epoch(s) on device: {device}")
    print(f"      Learning Rate : {lr}")
    print(f"      Batch Size    : {train_loader.batch_size}")

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        running_loss = correct_train = total_train = 0

        print(f"\n  Epoch [{epoch+1}/{num_epochs}]")
        for inputs, labels in tqdm(train_loader, desc="  Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct_train / total_train

        # --- Validation ---
        model.eval()
        val_loss = correct_val = total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="  Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (pred == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(val_loader.dataset)

        print(
            f"  Train Loss={train_loss:.4f}  Train Acc={train_acc:.2f}%  |  "
            f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.2f}%"
        )

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_acc': best_val_acc,
        }, CKPT_OUT_PATH)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_OUT_PATH)
            print(f"  --> New BEST model saved: {val_acc:.2f}%")

    print(f"\n  Fine-tuning complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Best weights: {BEST_OUT_PATH}")
    return model


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune NASNet on Scooby-Doo dataset")
    parser.add_argument("--epochs",     type=int,   default=10,     help="Number of fine-tuning epochs")
    parser.add_argument("--lr",         type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=64,     help="Batch size (optimized for 5070 Ti)")
    parser.add_argument("--clean",      action="store_true",        help="Force re-split of dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("  SafeToon - NASNet Fine-Tune on Scooby-Doo")
    print("=" * 55)
    print(f"  Device        : {device}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Learning Rate : {args.lr}")
    print(f"  Batch Size    : {args.batch_size}")
    print("=" * 55)

    # Step 1: Split
    if args.clean and SPLIT_DIR.exists():
        print(f"  Cleaning old split directory: {SPLIT_DIR}")
        shutil.rmtree(SPLIT_DIR)
    split_dataset(SOURCE_DIR, SPLIT_DIR)

    # Step 2: Data loaders
    print("\n[2/3] Building data loaders...")
    num_workers = 8 if torch.cuda.is_available() else 0
    train_loader, val_loader, test_loader = get_dataloaders(SPLIT_DIR, args.batch_size, num_workers)

    # Step 3: Load NASNet with existing weights
    print("\n  Loading NASNet base model...")
    model = get_model("nasnet", num_classes=len(CLASSES), pretrained=False)

    if PRETRAINED_WEIGHTS.exists():
        print(f"  Loading existing weights from: {PRETRAINED_WEIGHTS}")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device, weights_only=False))
        print("  Weights loaded successfully. Starting fine-tuning...")
    else:
        print(f"  [WARN] No pre-trained weights found at {PRETRAINED_WEIGHTS}")
        print("  Starting from ImageNet pretrained weights instead...")
        model = get_model("nasnet", num_classes=len(CLASSES), pretrained=True)

    # Step 4: Fine-tune
    finetune(model, train_loader, val_loader, args.epochs, args.lr, device)


if __name__ == "__main__":
    main()
