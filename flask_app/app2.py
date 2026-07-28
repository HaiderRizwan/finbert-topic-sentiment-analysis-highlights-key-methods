import os
import shutil
import random
import threading
import queue
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
import io
import time

# Re-use our logic from utils.py
from utils import load_ensemble_model, CLASSES, DEVICE

app = Flask(__name__)

# CONFIG
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT_DIR / "data"
CHECKPOINT_PATH = ROOT_DIR / "DinoVsecondRun_output" / "best_DinoV2.pth"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = ROOT_DIR / "models" / "best_DinoV2.pth"
BATCH_SIZE = 16  # Processing batch size for scanning
WEB_TEMP_DIR = Path(__file__).resolve().parent / "downloads" / "web_test"
WEB_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Global State
hard_samples_queue = queue.Queue(maxsize=100)
scanning_active = False
total_processed = 0
found_hard = 0
SCAN_MODE = "hard" # "hard", "confident", or "web"

# Load Model
print("Loading model for relabeling tool...")
models_dict = load_ensemble_model(str(CHECKPOINT_PATH))
dinov2 = models_dict["dinov2"]
convnext = models_dict["convnext"]
dino_head = models_dict["dino_head"]
temp_scaler = models_dict["temp_scaler"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_hard_samples_worker():
    global scanning_active, total_processed, found_hard, SCAN_MODE
    scanning_active = True
    
    # Collect all image paths
    all_paths = []
    for cls in CLASSES:
        folder = DATASET_ROOT / cls
        if folder.exists():
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_paths.append((img_path, cls))
    
    random.shuffle(all_paths)
    
    for i in range(0, len(all_paths), BATCH_SIZE):
        batch = all_paths[i:i+BATCH_SIZE]
        
        imgs = []
        valid_batch = []
        for p, label in batch:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
                valid_batch.append((p, label))
            except:
                continue
        
        if not imgs:
            continue
            
        x = torch.stack(imgs).to(DEVICE)
        
        with torch.no_grad():
            # Ensemble logic
            dino_feat = dinov2.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
            dino_logits = dino_head(dino_feat)
            conv_logits = convnext(x)
            
            logits = (dino_logits + conv_logits) / 2
            logits = temp_scaler(logits)
            
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            
            # Margin calculation
            top2_vals, _ = torch.topk(probs, 2, dim=1)
            margins = top2_vals[:, 0] - top2_vals[:, 1]
            
        for idx in range(len(valid_batch)):
            path, true_label = valid_batch[idx]
            p_label = CLASSES[preds[idx].item()]
            p_conf = conf[idx].item()
            p_margin = margins[idx].item()
            
            total_processed += 1
            
            # MODE LOGIC
            if SCAN_MODE == "hard":
                is_match = (
                    p_conf < 0.7 or 
                    p_margin < 0.2 or 
                    p_label != true_label
                )
                status_text = "MISMATCH" if p_label != true_label else "CONFUSING" if p_conf < 0.7 else "LOW MARGIN"
            else:
                # "confident" mode
                is_match = (p_conf > 0.75 and p_label == true_label)
                status_text = "CONFIDENT"
            
            if is_match:
                found_hard += 1
                sample = {
                    "path": str(path),
                    "filename": path.name,
                    "true_label": true_label,
                    "pred_label": p_label,
                    "confidence": round(p_conf * 100, 1),
                    "margin": round(p_margin, 3),
                    "status": status_text,
                    "mode": SCAN_MODE
                }
                hard_samples_queue.put(sample) # Blocks if queue is full

    scanning_active = False

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global SCAN_MODE, found_hard, total_processed, scanning_active
    data = request.json
    new_mode = data.get("mode", "hard")
    
    if new_mode != SCAN_MODE:
        SCAN_MODE = new_mode
        # Reset queue and stats for fresh start
        while not hard_samples_queue.empty():
            try: hard_samples_queue.get_nowait()
            except: break
        found_hard = 0
        total_processed = 0
        # If not active, restart worker
        if not scanning_active:
            threading.Thread(target=get_hard_samples_worker, daemon=True).start()
            
    return jsonify({"success": True, "mode": SCAN_MODE})

@app.route('/')
def index():
    return render_template('relabel.html', classes=CLASSES)

@app.route('/next')
def get_next():
    global total_processed, found_hard, SCAN_MODE
    
    if SCAN_MODE == "web":
        # LIVE WEB TEST LOGIC
        try:
            # Fetch a random cartoon image
            img_url = f"https://loremflickr.com/640/480/cartoon?random={time.time()}"
            response = requests.get(img_url, timeout=10)
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Save locally for serving
            filename = f"web_{int(time.time())}.jpg"
            save_path = WEB_TEMP_DIR / filename
            img.save(save_path)
            
            # Predict
            x = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dino_feat = dinov2.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
                dino_logits = dino_head(dino_feat)
                conv_logits = convnext(x)
                logits = (dino_logits + conv_logits) / 2
                logits = temp_scaler(logits)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                # Margin
                top2, _ = torch.topk(probs, 2, dim=1)
                margin = top2[0,0] - top2[0,1]

            total_processed += 1
            return jsonify({
                "success": True,
                "sample": {
                    "path": str(save_path),
                    "filename": filename,
                    "true_label": "Internet",
                    "pred_label": CLASSES[pred.item()],
                    "confidence": round(conf.item() * 100, 1),
                    "margin": round(margin.item(), 3),
                    "status": "LIVE TEST",
                    "mode": "web"
                },
                "stats": {
                    "processed": total_processed,
                    "found": total_processed,
                    "scanning": False
                }
            })
        except Exception as e:
            return jsonify({"success": False, "message": f"Web fetch failed: {str(e)}", "stats": {"processed":total_processed, "found":found_hard, "scanning":False}})

    # ORIGINAL LOCAL SCAN LOGIC
    try:
        sample = hard_samples_queue.get(timeout=2)
        return jsonify({
            "success": True,
            "sample": sample,
            "stats": {
                "processed": total_processed,
                "found": found_hard,
                "scanning": scanning_active
            }
        })
    except queue.Empty:
        return jsonify({
            "success": False, 
            "message": "Scanning in progress or no more hard samples found.",
            "stats": {
                "processed": total_processed,
                "found": found_hard,
                "scanning": scanning_active
            }
        })

@app.route('/relabel', methods=['POST'])
def relabel():
    data = request.json
    old_path = Path(data['path'])
    new_label = data['new_label']
    
    if not old_path.exists():
        return jsonify({"success": False, "error": "File not found"})
        
    new_dir = DATASET_ROOT / new_label
    new_dir.mkdir(exist_ok=True)
    new_path = new_dir / old_path.name
    
    try:
        # Move file
        shutil.move(str(old_path), str(new_path))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/images/<path:filename>')
def serve_image(filename):
    # This expects an absolute path or relative from workspace
    # For safety, we'll extract the dir and filename
    path = Path(filename)
    return send_from_directory(path.parent, path.name)

if __name__ == '__main__':
    # Start scanning thread
    threading.Thread(target=get_hard_samples_worker, daemon=True).start()
    app.run(debug=True, port=5001)
