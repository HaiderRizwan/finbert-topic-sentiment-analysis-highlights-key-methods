import traceback

import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

# Local imports
from scripts.train_models import get_model
from src.config import PROJECT_ROOT

app = Flask(__name__)
CORS(app)

CLASSES = ["erotism", "normal", "violent"]

MAPPING = {"normal": "Safe", "violent": "Violence", "erotism": "Erotism"}

# In-memory cache to prevent reloading large .pth files every single request
MODEL_CACHE = {}


def get_loaded_model(model_name: str):
    global MODEL_CACHE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name], device

    print(f"[{model_name}] Loading into cache dynamically...")
    model = get_model(model_name, num_classes=len(CLASSES), pretrained=False)
    weights_path = PROJECT_ROOT / "models" / f"{model_name}_best.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    MODEL_CACHE[model_name] = model
    return model, device


# Standard pre-processing rules mapping PyTorch exactly
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

from src.ingestion.pipeline import run_ingestion  # noqa: E402


@app.route("/")
def index():
    # Renders the UI
    return render_template("index.html")


@app.route("/ingestion")
def ingestion():
    return render_template("ingestion.html")


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    try:
        fps_val = request.form.get("fps", 1.0)
        fps = float(fps_val)
    except ValueError:
        return jsonify({"success": False, "error": "FPS must be a valid number"}), 400

    output_dir = request.form.get("output_dir", "").strip()
    if not output_dir:
        output_dir = None

    try:
        # Save temp file
        temp_dir = PROJECT_ROOT / "testing_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename
        file.save(temp_path)

        # Run pipeline
        summary = run_ingestion(temp_path, fps=fps, output_dir=output_dir)

        # Optionally cleanup
        temp_path.unlink(missing_ok=True)

        if summary.get("status") == "error":
            return jsonify(
                {
                    "success": False,
                    "error": summary.get("reason", "Unknown ingestion error"),
                }
            ), 500

        return jsonify(
            {
                "success": True,
                "job_id": summary.get("job_id"),
                "n_frames": summary.get("n_frames", 0),
                "n_audio_chunks": summary.get("n_audio_chunks", 0),
                "frames_dir": str(summary.get("frames_dir", "")),
                "audio_dir": str(summary.get("audio_dir", "")),
                "video_metadata": summary.get("video_metadata", {}),
            }
        ), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Ingestion crashed: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    model_name = request.form.get("model", "nasnet").lower()
    if model_name not in ["nasnet", "mobilenet"]:
        return jsonify({"success": False, "error": "Invalid model selection."}), 400

    try:
        # Load and verify Image
        image = Image.open(file.stream).convert("RGB")

        # Pull model (cached or fresh)
        model, device = get_loaded_model(model_name)

        # Preprocess and Batch
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Infer
        import time

        start_t = time.time()
        with torch.no_grad():
            output = model(input_batch)
        end_t = time.time()
        latency_ms = round((end_t - start_t) * 1000, 2)

        # Parse logic
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_class_idx = torch.max(probabilities, 0)

        top_class_name = CLASSES[top_class_idx]
        display_class = MAPPING.get(top_class_name, top_class_name)

        detailed_probs = {
            MAPPING.get(cls_name, cls_name): round(probabilities[i].item() * 100, 2)
            for i, cls_name in enumerate(CLASSES)
        }

        return jsonify(
            {
                "success": True,
                "prediction": display_class,
                "confidence": round(top_prob.item() * 100, 2),
                "details": detailed_probs,
                "model_used": model_name,
                "latency_ms": latency_ms,
            }
        ), 200

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Inference crashed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
