from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import os
import time
import shutil
from datetime import datetime
from utils import (
    load_clip_model, load_nasnet_model, load_ensemble_model,
    download_youtube_video, extract_frames, 
    classify_frames_cnn, classify_frames_clip, classify_frames_ensemble,
    process_video_audio, create_final_safe_video
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

# Global status tracking with dual parallel terminals
task_status = {
    "active": False,
    "progress": 0,
    "message": "Idle",
    "results": [],
    "logs": [],
    "frame_logs": [],
    "audio_logs": [],
    "audio_info": {}
}

def log_frame_msg(msg):
    print("[FRAME LOG]", msg)
    t_str = datetime.now().strftime('%H:%M:%S')
    task_status["frame_logs"].append(f"[{t_str}] {msg}")
    task_status["logs"].append(f"[{t_str}] [FRAME] {msg}")
    task_status["message"] = f"Frame: {msg}"

def log_audio_msg(msg):
    print("[AUDIO LOG]", msg)
    t_str = datetime.now().strftime('%H:%M:%S')
    task_status["audio_logs"].append(f"[{t_str}] {msg}")
    task_status["logs"].append(f"[{t_str}] [AUDIO] {msg}")

def run_pipeline(url, model_type, fps):
    global task_status
    task_status["active"] = True
    task_status["progress"] = 0
    task_status["logs"] = []
    task_status["frame_logs"] = []
    task_status["audio_logs"] = []
    task_status["results"] = []
    task_status["audio_info"] = {}
    
    try:
        # 1. Download Video
        log_frame_msg(f"Starting YouTube download: {url}")
        log_audio_msg(f"Connecting to video stream: {url}")
        task_status["progress"] = 10
        video_path = download_youtube_video(url, DOWNLOAD_DIR)
        log_frame_msg(f"Video ready: {os.path.basename(video_path)}")
        log_audio_msg(f"Video file acquired: {os.path.basename(video_path)}")
        
        # Prepare run folder
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder_name = f"{model_type}_{video_name}_{timestamp}"
        run_folder = os.path.join(OUTPUT_DIR, run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        task_status["progress"] = 20

        # Define Worker Threads for Parallel Execution
        frame_results_container = []

        def frame_worker():
            try:
                log_frame_msg(f"[15%] Extracting video frames at {fps} FPS...")
                frames_dir = os.path.join(run_folder, "frames")
                frame_paths = extract_frames(video_path, frames_dir, fps=fps)
                log_frame_msg(f"[35%] Extracted {len(frame_paths)} frames. Loading vision model ({model_type})...")

                if model_type == "nasnet":
                    if not os.path.exists(NASNET_PATH):
                        log_frame_msg(f"ERROR: NASNet path not available ({NASNET_PATH})")
                        raise FileNotFoundError(f"NASNet path not available: {NASNET_PATH}")
                    model = load_nasnet_model(NASNET_PATH)
                    log_frame_msg("[55%] Classifying frames with NASNet...")
                    raw_results = classify_frames_cnn(model, frame_paths)
                elif model_type == "nasnet_mega":
                    if not os.path.exists(NASNET_MEGA_PATH):
                        log_frame_msg(f"ERROR: NASNet Mega path not available ({NASNET_MEGA_PATH})")
                        raise FileNotFoundError(f"NASNet Mega path not available: {NASNET_MEGA_PATH}")
                    model = load_nasnet_model(NASNET_MEGA_PATH)
                    log_frame_msg("[55%] Classifying frames with NASNet Mega...")
                    raw_results = classify_frames_cnn(model, frame_paths)
                elif model_type == "dinov1":
                    if not os.path.exists(DINOV1_PATH):
                        log_frame_msg(f"ERROR: DINOv1 path not available ({DINOV1_PATH})")
                        raise FileNotFoundError(f"DINOv1 path not available: {DINOV1_PATH}")
                    models_dict = load_ensemble_model(DINOV1_PATH)
                    log_frame_msg("[55%] Classifying frames with DINOv1 Ensemble...")
                    raw_results = classify_frames_ensemble(models_dict, frame_paths)
                elif model_type == "dinov2":
                    if not os.path.exists(DINOV2_PATH):
                        log_frame_msg(f"ERROR: DINOv2 path not available ({DINOV2_PATH})")
                        raise FileNotFoundError(f"DINOv2 path not available: {DINOV2_PATH}")
                    models_dict = load_ensemble_model(DINOV2_PATH)
                    log_frame_msg("[55%] Classifying frames with DINOv2 Ensemble...")
                    raw_results = classify_frames_ensemble(models_dict, frame_paths)
                else:
                    if not os.path.exists(CLIP_PATH):
                        log_frame_msg("ERROR: CLIP path not available")
                        raise FileNotFoundError("CLIP path not available")
                    clip_model, classifier, preprocess = load_clip_model(CLIP_PATH)
                    log_frame_msg("[55%] Classifying frames with CLIP...")
                    raw_results = classify_frames_clip(clip_model, classifier, preprocess, frame_paths)

                log_frame_msg("[85%] Organizing classified frames into output folders...")
                formatted_results = []
                for res in raw_results:
                    label = res["label"]
                    label_dir = os.path.join(run_folder, label)
                    os.makedirs(label_dir, exist_ok=True)
                    fname = os.path.basename(res["path"])
                    dst_path = os.path.join(label_dir, fname)
                    shutil.copy(res["path"], dst_path)

                    # Build web-accessible URL
                    web_url = f"/media/{run_folder_name}/{label}/{fname}"
                    formatted_results.append({
                        "filename": fname,
                        "label": label,
                        "confidence": round(res["confidence"] * 100, 1),
                        "web_url": web_url
                    })

                frame_results_container.extend(formatted_results)
                log_frame_msg(f"[100%] Frame processing finished! ({len(formatted_results)} frames ready)")
            except Exception as ef:
                log_frame_msg(f"Frame processing error: {str(ef)}")

        def audio_worker():
            try:
                log_audio_msg("Starting parallel audio extraction & speech analysis...")
                audio_res = process_video_audio(video_path, run_folder, log_fn=log_audio_msg)
                if audio_res.get("success") and "cleaned_audio_path" in audio_res:
                    audio_fname = os.path.basename(audio_res["cleaned_audio_path"])
                    audio_res["web_audio_url"] = f"/media/{run_folder_name}/{audio_fname}"
                task_status["audio_info"] = audio_res
                log_audio_msg("Audio thread completed!")
            except Exception as ea:
                log_audio_msg(f"Audio processing error: {str(ea)}")

        # Launch BOTH threads in parallel
        t_frame = threading.Thread(target=frame_worker)
        t_audio = threading.Thread(target=audio_worker)

        log_frame_msg("=== Launching Parallel Frame Thread ===")
        log_audio_msg("=== Launching Parallel Audio Thread ===")

        t_frame.start()
        t_audio.start()

        # Wait for both parallel threads to finish
        while t_frame.is_alive() or t_audio.is_alive():
            time.sleep(0.5)
            # Update smooth progress
            task_status["progress"] = min(95, task_status["progress"] + 1)

        t_frame.join()
        t_audio.join()

        # 4. Synthesize Final Censored Safe Video (Intelligent Scene Cutting + Cleaned Muted Audio)
        log_frame_msg("[90%] Synthesizing final safe video with intelligent scene removal...")
        audio_info = task_status.get("audio_info", {})
        cleaned_audio_p = audio_info.get("cleaned_audio_path", os.path.join(run_folder, "extracted_audio.wav"))

        final_v_path, cut_scenes = create_final_safe_video(
            video_path=video_path,
            raw_results=frame_results_container,
            audio_path=cleaned_audio_p,
            output_dir=run_folder,
            fps=fps,
            log_fn=log_frame_msg
        )

        final_v_fname = os.path.basename(final_v_path)
        task_status["final_video_url"] = f"/media/{run_folder_name}/{final_v_fname}"
        task_status["cut_scenes"] = cut_scenes

        task_status["results"] = frame_results_container
        task_status["progress"] = 100
        task_status["message"] = "Parallel Analysis & Final Video Synthesis Complete!"
        log_frame_msg("[100%] Pipeline execution finished successfully.")
        log_audio_msg("[100%] Pipeline execution finished successfully.")

    except Exception as e:
        log_frame_msg(f"ERROR: {str(e)}")
        log_audio_msg(f"ERROR: {str(e)}")
        task_status["progress"] = 0
    finally:
        task_status["active"] = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
@app.route("/process", methods=["POST"])
@app.route("/run", methods=["POST"])
@app.route("/api/start", methods=["POST"])
def start_task():
    if task_status["active"]:
        return jsonify({"error": "Task already running"}), 400
    
    url = request.form.get("url")
    model_type = request.form.get("model", "dinov2")
    fps = float(request.form.get("fps", 1))
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    thread = threading.Thread(target=run_pipeline, args=(url, model_type, fps))
    thread.start()
    return jsonify({"message": "Task started"})

@app.route("/status")
@app.route("/api/status")
def get_status():
    return jsonify(task_status)

@app.route("/media/<path:filename>")
def serve_media(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({
        "error": "Route not found",
        "available_routes": ["/", "/start (POST)", "/process (POST)", "/status (GET)", "/media/<path:filename>"]
    }), 404

if __name__ == "__main__":
    app.run(debug=False, port=5000)
