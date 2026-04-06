"""
Flask Web App for Deepfake Detection (PyTorch)
"""

import os
import sys
import uuid
import json
import torch
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename

# ---------------------------------
# Project root setup
# ---------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.video_utils import load_video
from preprocessing.face_detection import extract_faces_from_frames
from models.cnn_lstm_model import CNNLSTM
from models.ensemble_2models import Ensemble2Models
from models.ensemble_3models import Ensemble3Models

# ---------------------------------
# Flask Setup
# ---------------------------------

app = Flask(__name__)

app.config["SECRET_KEY"] = "deepfake-secret"
app.config["UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "webapp", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Model paths (try best model first, then fallback)
MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, "models", "deepfake_model_best.pth"),
    os.path.join(PROJECT_ROOT, "models", "deepfake_model.pth")
]

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------
# Model Registry - All models available
# ---------------------------------

# Single Model
_single_model = None

# Ensemble Models
_ensemble_2 = None
_ensemble_3 = None

# Current active model (default to best available)
_active_model_type = "ensemble_3"  # Can be "single", "ensemble_2", or "ensemble_3"

def get_single_model():
    global _single_model
    if _single_model is None:
        model = CNNLSTM()
        model_loaded = False

        for model_path in MODEL_PATHS:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"Loaded single model from epoch {checkpoint.get('epoch', 'unknown')}")
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"Single model loaded from: {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue

        if not model_loaded:
            print("WARNING: No single model found!")
            return None

        model.to(device)
        model.eval()
        _single_model = model
    return _single_model

def get_ensemble_2():
    global _ensemble_2
    if _ensemble_2 is None:
        try:
            _ensemble_2 = Ensemble2Models(device)
            if len(_ensemble_2.models) >= 2:
                print(f"✅ Loaded 2-model ensemble with {len(_ensemble_2.models)} models")
            else:
                print("⚠️ 2-model ensemble has insufficient models")
                _ensemble_2 = None
        except Exception as e:
            print(f"Error loading 2-model ensemble: {e}")
            _ensemble_2 = None
    return _ensemble_2

def get_ensemble_3():
    global _ensemble_3
    if _ensemble_3 is None:
        try:
            _ensemble_3 = Ensemble3Models(device)
            if len(_ensemble_3.models) >= 2:
                print(f"✅ Loaded 3-model ensemble with {len(_ensemble_3.models)} models")
                print(f"   Model A: 89.5%, Model B: 94.0%, Model C: 92.4%")
            else:
                print("⚠️ 3-model ensemble has insufficient models")
                _ensemble_3 = None
        except Exception as e:
            print(f"Error loading 3-model ensemble: {e}")
            _ensemble_3 = None
    return _ensemble_3

# Helper to get current ensemble based on active model
def get_current_ensemble():
    if _active_model_type == "ensemble_3":
        return get_ensemble_3()
    elif _active_model_type == "ensemble_2":
        return get_ensemble_2()
    return None

# ---------------------------------
# Helper
# ---------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def set_active_model(model_type):
    """Switch between different models"""
    global _active_model_type
    if model_type in ["single", "ensemble_2", "ensemble_3"]:
        _active_model_type = model_type
        return True
    return False

def get_active_model_info():
    """Get info about currently active model"""
    if _active_model_type == "single":
        model = get_single_model()
        return {
            "type": "Single CNN-LSTM",
            "available": model is not None,
            "accuracy": "89.5-94.0%"
        }
    elif _active_model_type == "ensemble_2":
        model = get_ensemble_2()
        return {
            "type": "2-Model Ensemble",
            "available": model is not None and len(model.models) >= 2,
            "accuracy": "~92.5%",
            "models": len(model.models) if model else 0
        }
    elif _active_model_type == "ensemble_3":
        model = get_ensemble_3()
        return {
            "type": "3-Model Ensemble (Smart)",
            "available": model is not None and len(model.models) >= 2,
            "accuracy": "93.4%",
            "models": len(model.models) if model else 0,
            "method": "Smart Voting (trusts Model A)"
        }
    return {"type": "Unknown", "available": False}

# ---------------------------------
# Prediction - Uses active model
# ---------------------------------

def predict_video(video_path):
    # Load frames
    frames = load_video(video_path)

    if len(frames) == 0:
        return None, None, None, None

    # Detect faces
    faces = extract_faces_from_frames(frames, resize_to=(128,128))

    if len(faces) == 0:
        return None, None, None, None

    faces = np.array(faces, dtype=np.float32) / 255.0

    SEQ_LEN = 20

    # Ensure enough frames
    if len(faces) < SEQ_LEN:
        pad = np.repeat(faces[-1:], SEQ_LEN - len(faces), axis=0)
        faces = np.concatenate([faces, pad], axis=0)

    # Sample across video
    indices = np.linspace(0, len(faces)-1, SEQ_LEN).astype(int)
    faces = faces[indices]

    # Convert shape (T,H,W,C) -> (T,C,H,W)
    faces = np.transpose(faces, (0, 3, 1, 2))

    tensor = torch.tensor(faces, dtype=torch.float32).unsqueeze(0).to(device)

    # Get active model info
    model_info = get_active_model_info()
    
    # USE ACTIVE MODEL
    if _active_model_type == "ensemble_3":
        ensemble = get_ensemble_3()
        if ensemble and len(ensemble.models) >= 3:
            try:
                result = ensemble.predict_smart(tensor)
                print(f"🤖 Smart Ensemble (3-model): {result['method']}")
                print(f"   Individual: A={result['individual_probs'][0]:.3f}, B={result['individual_probs'][1]:.3f}, C={result['individual_probs'][2]:.3f}")
                print(f"   Final: {result['prediction']} ({result['confidence']}%)")
                return result['prediction'], result['confidence'], result['probability'], result['method']
            except Exception as e:
                print(f"Smart ensemble failed: {e}, falling back")
    
    elif _active_model_type == "ensemble_2":
        ensemble = get_ensemble_2()
        if ensemble and len(ensemble.models) >= 2:
            try:
                result = ensemble.predict(tensor)
                print(f"🤖 2-Model Ensemble: {result['prediction']} ({result['confidence']}%)")
                return result['prediction'], result['confidence'], result['probability'], "2-model ensemble"
            except Exception as e:
                print(f"2-model ensemble failed: {e}")
    
    # FALLBACK TO SINGLE MODEL
    model = get_single_model()
    if model is None:
        return None, None, None, None
    
    with torch.no_grad():
        output = model(tensor)
        proba = torch.sigmoid(output).cpu().numpy()[0][0]

    if proba > 0.5:
        label = "FAKE"
        confidence = ((proba - 0.5) / 0.5) * 100
    else:
        label = "REAL"
        confidence = ((0.5 - proba) / 0.5) * 100
    
    confidence = max(0, min(100, confidence))
    confidence = round(confidence, 1)

    return label, confidence, proba, "single model"


# ---------------------------------
# Routes
# ---------------------------------

@app.route("/")
def index():
    model_info = get_active_model_info()
    return render_template("index.html", model_info=model_info)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/settings")
def settings():
    """Page to switch between models"""
    model_info = get_active_model_info()
    return render_template("settings.html", 
                          current_model=_active_model_type,
                          model_info=model_info)

@app.route("/analysis")
def analysis():
    """Performance analysis dashboard"""
    # Load training history
    history_path = os.path.join(PROJECT_ROOT, "training", "training_history_full.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = None
    
    return render_template("analysis.html", history=history)

@app.route("/switch_model/<model_type>")
def switch_model(model_type):
    """Switch between models"""
    if set_active_model(model_type):
        flash(f"Switched to {model_type} model")
    else:
        flash(f"Invalid model type: {model_type}")
    return redirect(url_for("settings"))

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        flash("No video uploaded")
        return redirect(url_for("index"))

    file = request.files["video"]

    if file.filename == "":
        flash("No video selected")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid video format")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        label, confidence, proba, method = predict_video(filepath)

        if label is None:
            flash("Could not process video (no faces detected)")
            return redirect(url_for("index"))

        model_info = get_active_model_info()
        print(f"Final prediction using {model_info['type']} ({method}): {label}, Confidence: {confidence}%")

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            probability=round(proba, 4),
            is_deepfake=(label == "FAKE"),
            filename=filename,
            model_used=f"{model_info['type']} ({method})"
        )

    except Exception as e:
        flash(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return redirect(url_for("index"))

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ---------------------------------
# API endpoints
# ---------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    file = request.files["video"]
    
    if file.filename == "":
        return jsonify({"error": "No video selected"}), 400
    
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)
    
    try:
        label, confidence, proba, method = predict_video(filepath)
        
        if label is None:
            return jsonify({"error": "Could not process video"}), 500
        
        model_info = get_active_model_info()
        
        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "probability": proba,
            "filename": filename,
            "model_used": f"{model_info['type']} ({method})"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """API endpoint for model statistics"""
    single = get_single_model()
    ensemble2 = get_ensemble_2()
    ensemble3 = get_ensemble_3()
    active = get_active_model_info()
    
    return jsonify({
        "models": {
            "single": {
                "available": single is not None,
                "accuracy": "89.5-94.0%"
            },
            "ensemble_2": {
                "available": ensemble2 is not None and len(ensemble2.models) >= 2,
                "models": len(ensemble2.models) if ensemble2 else 0,
                "accuracy": "~92.5%"
            },
            "ensemble_3": {
                "available": ensemble3 is not None and len(ensemble3.models) >= 2,
                "models": len(ensemble3.models) if ensemble3 else 0,
                "accuracy": "93.4%",
                "method": "Smart Voting (trusts Model A)"
            }
        },
        "active_model": active,
        "dataset_size": 3431,
        "real_videos": 363,
        "fake_videos": 3068,
        "framework": "PyTorch",
        "backend": "Flask"
    })


@app.route("/api/switch_model/<model_type>", methods=["POST"])
def api_switch_model(model_type):
    """API to switch models"""
    if set_active_model(model_type):
        return jsonify({
            "success": True,
            "active_model": get_active_model_info()
        })
    return jsonify({"error": "Invalid model type"}), 400


@app.route("/api/analysis/history")
def analysis_history():
    """Return training history data for charts"""
    history_path = os.path.join(PROJECT_ROOT, "training", "training_history_full.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    return jsonify({"error": "No training history found"}), 404


@app.route("/api/analysis/images")
def analysis_images():
    """Return base64 encoded images for analysis dashboard"""
    import base64
    images = {}
    
    # List of image files to serve
    image_files = [
        "ensemble_confusion_matrix.png",
        "ensemble_roc_curve.png",
        "model_comparison.png",
        "confidence_distribution.png",
        "learning_rate_schedule.png"
    ]
    
    outputs_dir = os.path.join(PROJECT_ROOT, "training", "outputs")
    
    for img_file in image_files:
        img_path = os.path.join(outputs_dir, img_file)
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                images[img_file] = base64.b64encode(f.read()).decode('utf-8')
    
    # Also try to get training curves if available
    training_curves_path = os.path.join(outputs_dir, "training_curves.png")
    if os.path.exists(training_curves_path):
        with open(training_curves_path, 'rb') as f:
            images["training_curves.png"] = base64.b64encode(f.read()).decode('utf-8')
    
    return jsonify(images)


# ---------------------------------
# Run server
# ---------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Deepfake Detection Web App")
    print("="*60)
    print(f"Device: {device}")
    
    # Load all models on startup
    single = get_single_model()
    ensemble2 = get_ensemble_2()
    ensemble3 = get_ensemble_3()
    
    print("\n📊 Available Models:")
    print(f"  • Single Model: {'✅' if single else '❌'}")
    print(f"  • 2-Model Ensemble: {'✅' if ensemble2 and len(ensemble2.models)>=2 else '❌'}")
    print(f"  • 3-Model Ensemble: {'✅' if ensemble3 and len(ensemble3.models)>=2 else '❌'}")
    
    active = get_active_model_info()
    print(f"\n🎯 Active Model: {active['type']} ({active.get('method', 'standard')})")
    
    print(f"\nUpload folder: {app.config['UPLOAD_FOLDER']}")
    print("-"*60)
    print("Routes:")
    print("  /              - Home page")
    print("  /about         - About page")
    print("  /documentation - Documentation page")
    print("  /settings      - Model selection page")
    print("  /analysis      - Performance analysis dashboard")
    print("  /switch_model/<type> - Switch model")
    print("  /predict       - Prediction form POST")
    print("  /api/*         - JSON APIs")
    print("-"*60)
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*60)

    app.run(host="0.0.0.0", port=5000, debug=True)