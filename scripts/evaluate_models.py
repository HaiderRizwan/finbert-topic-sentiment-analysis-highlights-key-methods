"""
evaluate_models.py
------------------
Computes comprehensive classification metrics for NASNet (Main) and NASNet
(Scooby-Doo fine-tuned) on the Train, Val, and Test splits:

  Accuracy, Precision, Recall, F1, F2, MCC, Cohen's Kappa,
  AUC-ROC (macro OvR), Confusion Matrix, per-class report.

Usage:
    python scripts/evaluate_models.py --split test
    python scripts/evaluate_models.py --split train
    python scripts/evaluate_models.py --split val
    python scripts/evaluate_models.py --split all       # runs all three
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from tqdm import tqdm  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from scripts.train_models import get_model  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────
CLASSES = ['erotism', 'normal', 'violent']
DATASET_ROOT = PROJECT_ROOT / "dataset_split"
MODEL_DIR = PROJECT_ROOT / "models"

MODELS = {
    "NASNet (Main)":        MODEL_DIR / "nasnet_best.pth",
    "NASNet (Scooby-Doo)":  MODEL_DIR / "nasnet_scoobydoo_best.pth",
}


# ─── Data Loader ──────────────────────────────────────────────────────────────
def get_loader(split: str, batch_size: int = 64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(
        root=str(DATASET_ROOT / split), transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return loader, dataset.classes


# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(model, loader, device):
    """Returns (all_labels, all_preds, all_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Inferring", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, class_names):
    avg = "macro"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, average=avg, zero_division=0)
    f0_5 = fbeta_score(y_true, y_pred, beta=0.5, average=avg, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # AUC-ROC needs probability scores
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=avg)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    return {
        "Accuracy":         acc,
        "Precision (macro)": prec,
        "Recall (macro)":   rec,
        "F1  (macro)":      f1,
        "F2  (macro)":      f2,
        "F0.5 (macro)":     f0_5,
        "MCC":              mcc,
        "Cohen Kappa":      kappa,
        "AUC-ROC (macro)":  auc,
        "_confusion_matrix": cm,
        "_report":          report,
    }


# ─── Pretty printer ───────────────────────────────────────────────────────────
def print_metrics(model_name: str, split: str, metrics: dict, class_names: list):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_name}  |  Split: {split.upper()}")
    print(sep)

    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        print(f"  {k:<22}: {v:.4f}")

    print("\n  Per-class Report:")
    print(metrics["_report"])

    cm = metrics["_confusion_matrix"]
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = "  " + "  ".join(f"{c:>10}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {class_names[i]:<10}  {row_str}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SafeToon models")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which dataset split to evaluate on (default: test)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Splits : {splits}")

    for model_name, weights_path in MODELS.items():
        print(f"\n{'-'*60}")
        print(f"Loading model: {model_name}")

        if not weights_path.exists():
            print(f"  [SKIP] Weights not found at {weights_path}")
            continue

        model = get_model("nasnet", num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        model.to(device)

        for split in splits:
            print(f"\n  Evaluating on: {split.upper()} split ...")
            loader, class_names = get_loader(split, batch_size=args.batch_size)
            y_true, y_pred, y_prob = run_inference(model, loader, device)
            metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
            print_metrics(model_name, split, metrics, class_names)


if __name__ == "__main__":
    main()
