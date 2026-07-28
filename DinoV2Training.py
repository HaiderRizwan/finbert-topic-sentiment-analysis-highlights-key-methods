import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path(r"C:/Users/dotsm/Desktop/mixed cartoon dataset/MEGA")

BATCH_SIZE = 32
EPOCHS = 12
LR = 3e-4

OUTPUT_DIR = Path("./ensemble_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder(DATASET_ROOT)
classes = dataset.classes
num_classes = len(classes)

print("Classes:", classes)

# =========================
# FEATURE EXTRACTOR (for safe split)
# =========================
feature_extractor = models.resnet18(pretrained=True)
feature_extractor.fc = nn.Identity()
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

transform_feat = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(dataset):
    feats = []
    for img, _ in tqdm(dataset, desc="Extracting features"):
        x = transform_feat(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = feature_extractor(x).cpu().numpy().flatten()
        feats.append(f)
    return np.array(feats)

# =========================
# CLUSTER-BASED SPLIT (FIXED LEAKAGE)
# =========================
print("Building leakage-safe split...")

features = extract_features(dataset)

kmeans = KMeans(n_clusters=120, random_state=42, n_init=10)
cluster_ids = kmeans.fit_predict(features)

cluster_map = {}
for i, cid in enumerate(cluster_ids):
    cluster_map.setdefault(cid, []).append(i)

clusters = list(cluster_map.keys())
random.shuffle(clusters)

n = len(clusters)

train_clusters = clusters[:int(0.7 * n)]
val_clusters   = clusters[int(0.7 * n):int(0.85 * n)]
test_clusters  = clusters[int(0.85 * n):]

def collect(cluster_list):
    out = []
    for c in cluster_list:
        out.extend(cluster_map[c])
    return out

train_idx = collect(train_clusters)
val_idx   = collect(val_clusters)
test_idx  = collect(test_clusters)

# =========================
# TRANSFORMS
# =========================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_set = Subset(datasets.ImageFolder(DATASET_ROOT, transform=train_tf), train_idx)
val_set   = Subset(datasets.ImageFolder(DATASET_ROOT, transform=eval_tf), val_idx)
test_set  = Subset(datasets.ImageFolder(DATASET_ROOT, transform=eval_tf), test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE)

# =========================
# MODELS
# =========================
convnext = models.convnext_tiny(pretrained=True)
convnext.classifier[2] = nn.Linear(768, num_classes)
convnext = convnext.to(DEVICE)

# DINO HEAD
class DinoHead(nn.Module):
    def __init__(self, in_dim=384, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

dino_head = DinoHead(384, num_classes).to(DEVICE)

# =========================
# FUSION HEAD (FIX #1)
# =========================
fusion_head = nn.Sequential(
    nn.Linear(num_classes * 2, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
).to(DEVICE)

# =========================
# LOSS (FIX #2 class imbalance ready)
# =========================
class_counts = np.bincount([dataset[i][1] for i in range(len(dataset))])
weights = 1.0 / (class_counts + 1e-6)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# =========================
# OPTIMIZER
# =========================
optimizer = torch.optim.AdamW(
    list(convnext.parameters()) +
    list(dino_head.parameters()) +
    list(fusion_head.parameters()),
    lr=LR,
    weight_decay=1e-4
)

# =========================
# DINOV2 BACKBONE (frozen)
# =========================
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
dinov2.eval()
for p in dinov2.parameters():
    p.requires_grad = False

# =========================
# FORWARD PASS (FIX #1)
# =========================
def forward_model(x):
    with torch.no_grad():
        dino_feat = dinov2.forward_features(x)["x_norm_patchtokens"].mean(dim=1)

    dino_logits = dino_head(dino_feat)
    conv_logits = convnext(x)

    fused = torch.cat([dino_logits, conv_logits], dim=1)
    out = fusion_head(fused)

    return out

# =========================
# TRAINING
# =========================
best_acc = 0
history = []

for epoch in range(EPOCHS):

    convnext.train()
    dino_head.train()
    fusion_head.train()

    total, correct, loss_sum = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = forward_model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = logits.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct/total})

    train_acc = correct / total
    train_loss = loss_sum / len(train_loader)

    # =========================
    # VALIDATION
    # =========================
    convnext.eval()
    dino_head.eval()
    fusion_head.eval()

    v_correct, v_total, v_loss = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = forward_model(x)
            loss = criterion(logits, y)

            v_loss += loss.item()
            preds = logits.argmax(1)

            v_correct += (preds == y).sum().item()
            v_total += y.size(0)

    val_acc = v_correct / v_total
    val_loss = v_loss / len(val_loader)

    print(f"\nEpoch {epoch+1} | Train {train_acc:.4f} | Val {val_acc:.4f}")

    history.append([epoch+1, train_loss, train_acc, val_loss, val_acc])

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "convnext": convnext.state_dict(),
            "dino_head": dino_head.state_dict(),
            "fusion_head": fusion_head.state_dict()
        }, OUTPUT_DIR / "best_model.pth")
        print("✔ Saved best model")

# =========================
# TEST
# =========================
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = forward_model(x)

        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print("\nREPORT")
print(classification_report(all_labels, all_preds, target_names=classes))
print(confusion_matrix(all_labels, all_preds))

# =========================
# LOGS
# =========================
with open(OUTPUT_DIR / "log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    writer.writerows(history)