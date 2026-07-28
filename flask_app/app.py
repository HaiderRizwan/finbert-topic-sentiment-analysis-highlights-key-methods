from flask import Flask, render_template, request, jsonify
import threading
import os
import time
import shutil
from datetime import datetime
from utils import (
    load_clip_model, load_nasnet_model, load_ensemble_model,
    download_youtube_video, extract_frames, 
    classify_frames_cnn, classify_frames_clip, classify_frames_ensemble
)

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def resolve_model_path(candidates):
    for cand in candidates:
        full_p = os.path.normpath(os.path.join(ROOT_DIR, cand)) if not os.path.isabs(cand) else cand
        if os.path.exists(full_p):
            return full_p
    return os.path.normpath(os.path.join(ROOT_DIR, candidates[0]))

NASNET_PATH = resolve_model_path(["nasnet_output_train2/best_nasnet.pth", "models/nasnet_best.pth"])
NASNET_MEGA_PATH = resolve_model_path(["models/nasnet_mixed_cartoon_best.pth", "models/nasnet_best.pth"])
CLIP_PATH = resolve_model_path(["models/clip_classifier.pth", "clip_classifier.pth"])
DINOV1_PATH = resolve_model_path(["DinoV1_output/best_ensemble.pth", "models/best_ensemble.pth"])
DINOV2_PATH = resolve_model_path(["DinoVsecondRun_output/best_DinoV2.pth", "models/best_DinoV2.pth"])

# Ensure directories exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global status tracking
task_status = {
    "active": False,
    "progress": 0,
    "message": "Idle",
    "results": [],
    "logs": []
}

def log_message(msg):
    print(msg)
    task_status["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    task_status["message"] = msg

def run_pipeline(url, model_type, fps):
    global task_status
    task_status["active"] = True
    task_status["progress"] = 0
    task_status["logs"] = []
    task_status["results"] = []
    
    try:
        # 1. Download
        log_message(f"Starting YouTube download: {url}")
        task_status["progress"] = 10
        video_path = download_youtube_video(url, DOWNLOAD_DIR)
        log_message(f"Video downloaded to: {video_path}")
        
        # 2. Extract Frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(OUTPUT_DIR, f"{model_type}_{video_name}_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        
        log_message(f"Extracting frames at {fps} FPS...")
        task_status["progress"] = 30
        frame_paths = extract_frames(video_path, os.path.join(run_folder, "frames"), fps=fps)
        log_message(f"Extracted {len(frame_paths)} frames.")
        
        # 3. Classify
        task_status["progress"] = 50
        if model_type == "nasnet":
            if not os.path.exists(NASNET_PATH):
                log_message(f"ERROR: NASNet path not available ({NASNET_PATH})")
                raise FileNotFoundError(f"NASNet path not available: {NASNET_PATH}")
            log_message("Loading NASNet model...")
            model = load_nasnet_model(NASNET_PATH)
            log_message("Classifying frames with NASNet...")
            results = classify_frames_cnn(model, frame_paths)
        elif model_type == "nasnet_mega":
            if not os.path.exists(NASNET_MEGA_PATH):
                log_message(f"ERROR: NASNet Mega path not available ({NASNET_MEGA_PATH})")
                raise FileNotFoundError(f"NASNet Mega path not available: {NASNET_MEGA_PATH}")
            log_message("Loading NASNet Mega model...")
            model = load_nasnet_model(NASNET_MEGA_PATH)
            log_message("Classifying frames with NASNet Mega...")
            results = classify_frames_cnn(model, frame_paths)
        elif model_type == "dinov1":
            if not os.path.exists(DINOV1_PATH):
                log_message(f"ERROR: DINOv1 path not available ({DINOV1_PATH})")
                raise FileNotFoundError(f"DINOv1 path not available: {DINOV1_PATH}")
            log_message("Loading DINOv2 Ensemble v1...")
            models_dict = load_ensemble_model(DINOV1_PATH)
            log_message("Classifying frames with dinoV1...")
            results = classify_frames_ensemble(models_dict, frame_paths)
        elif model_type == "dinov2":
            if not os.path.exists(DINOV2_PATH):
                log_message(f"ERROR: DINOv2 path not available ({DINOV2_PATH})")
                raise FileNotFoundError(f"DINOv2 path not available: {DINOV2_PATH}")
            log_message("Loading DINOv2 Ensemble v2 (DinoV2)...")
            models_dict = load_ensemble_model(DINOV2_PATH)
            log_message("Classifying frames with dinoV2...")
            results = classify_frames_ensemble(models_dict, frame_paths)
        else:
            if not os.path.exists(CLIP_PATH):
                log_message("ERROR: CLIP path not available")
                raise FileNotFoundError("CLIP path not available")
            log_message("Loading CLIP model...")
            clip_model, classifier, preprocess = load_clip_model(CLIP_PATH)
            log_message("Classifying frames with CLIP...")
            results = classify_frames_clip(clip_model, classifier, preprocess, frame_paths)
        
        # 4. Save results and organize folders
        log_message("Organizing results into folders...")
        for res in results:
            label = res["label"]
            label_dir = os.path.join(run_folder, label)
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(res["path"], os.path.join(label_dir, os.path.basename(res["path"])))
        
        task_status["results"] = results
        task_status["progress"] = 100
        log_message(f"Processing complete! Results saved in: {run_folder}")
        
    except Exception as e:
        log_message(f"ERROR: {str(e)}")
        task_status["progress"] = 0
    finally:
        task_status["active"] = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start_task():
    if task_status["active"]:
        return jsonify({"error": "Task already running"}), 400
    
    url = request.form.get("url")
    model_type = request.form.get("model")
    fps = float(request.form.get("fps", 1))
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    thread = threading.Thread(target=run_pipeline, args=(url, model_type, fps))
    thread.start()
    return jsonify({"message": "Task started"})

@app.route("/status")
def get_status():
    return jsonify(task_status)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
