"""
Flask Web App for Deepfake Detection with MongoDB History
"""
import threading
import asyncio
import os
import sys
import uuid
import json
import torch
import numpy as np
import time
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId

# ---------------------------------
# Project root setup - MUST BE FIRST
# ---------------------------------

# Get the project root (two levels up from webapp folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add paths for imports
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

# Import local modules (after path is set)
from ai_agent import ai_agent
from websocket_server import start_websocket_server, broadcast_progress
from utils.video_utils import load_video
from preprocessing.face_detection import extract_faces_from_frames, extract_faces_from_bboxes
from models.cnn_lstm_model import CNNLSTM
from models.ensemble_2models import Ensemble2Models
from models.ensemble_3models import Ensemble3Models
from nim_integration import nim

# ---------------------------------
# Flask Setup
# ---------------------------------

app = Flask(__name__)
CORS(app)

@app.context_processor
def inject_global_vars():
    from datetime import datetime
    try:
        from flask import request
        endpoint = request.endpoint or 'unknown'
    except Exception:
        endpoint = 'unknown'
    
    return {
        'page_context': {'page_name': endpoint},
        'current_time': datetime.now().strftime("%I:%M %p")
    }

app.config["SECRET_KEY"] = "deepfake-secret"
app.config["UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "webapp", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Model paths
MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, "models", "deepfake_model_best.pth"),
    os.path.join(PROJECT_ROOT, "models", "deepfake_model.pth")
]

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------------------------
# Start WebSocket server in background thread
# ---------------------------------
def run_websocket():
    """Run WebSocket server with port conflict handling"""
    import socket
    import subprocess
    
    def kill_process_on_port(port):
        """Kill process using the specified port (Windows)"""
        try:
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        pid = line.strip().split()[-1]
                        subprocess.run(f'taskkill /PID {pid} /F', shell=True, 
                                     capture_output=True)
                        print(f"✅ Killed process {pid} using port {port}")
                        return True
        except Exception as e:
            print(f"⚠️ Could not kill process: {e}")
        return False
    
    def is_port_available(port):
        """Check if port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False
    
    # Try to free up port 8765
    if not is_port_available(8765):
        print("⚠️ Port 8765 is in use. Attempting to free it...")
        if kill_process_on_port(8765):
            import time
            time.sleep(1)  # Wait for OS to release the port
    
    # Try different ports if 8765 is still unavailable
    ports_to_try = [8765, 8766, 8767, 8768, 8769]
    
    for port in ports_to_try:
        if is_port_available(port):
            try:
                print(f"✅ Starting WebSocket server on port {port}")
                # Pass port to websocket server function
                asyncio.run(start_websocket_server(port=port))
                return
            except Exception as e:
                print(f"❌ Failed on port {port}: {e}")
                continue
    
    print("⚠️ WebSocket server could not start. Real-time updates disabled.")

# Start WebSocket thread
websocket_thread = threading.Thread(target=run_websocket, daemon=True)
websocket_thread.start()
print("✅ WebSocket thread started")
# ---------------------------------
# MongoDB Setup - Local Connection
# ---------------------------------

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "deepguard_ai"
COLLECTION_NAME = "detection_history"

history_collection = None

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    history_collection = db[COLLECTION_NAME]
    
    # Create index for faster queries
    history_collection.create_index("timestamp")
    history_collection.create_index("prediction")
    
    print("✅ MongoDB connected successfully on localhost:27017")
    print(f"   Database: {DB_NAME}")
    print(f"   Collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    print("   History will not be saved")
    history_collection = None

# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------
# Model Registry
# ---------------------------------

_single_model = None
_ensemble_2 = None
_ensemble_3 = None
_active_model_type = "ensemble_3"

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

def get_current_ensemble():
    if _active_model_type == "ensemble_3":
        return get_ensemble_3()
    elif _active_model_type == "ensemble_2":
        return get_ensemble_2()
    return None

# ---------------------------------
# Helper Functions
# ---------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def set_active_model(model_type):
    global _active_model_type
    if model_type in ["single", "ensemble_2", "ensemble_3"]:
        _active_model_type = model_type
        return True
    return False

def get_active_model_info():
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
# MongoDB Helper Functions
# ---------------------------------

def save_detection_to_mongodb(video_name, prediction, confidence, probability, model_used, method, file_size_mb, processing_time_sec):
    """Save detection result to MongoDB"""
    if history_collection is None:
        return None
    
    try:
        record = {
            "video_name": video_name,
            "prediction": prediction,
            "confidence": float(confidence),
            "probability": float(probability),
            "model_used": model_used,
            "method": method,
            "file_size_mb": round(file_size_mb, 2),
            "processing_time_sec": round(processing_time_sec, 2),
            "timestamp": datetime.now(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        }
        
        result = history_collection.insert_one(record)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None

def get_history_from_mongodb(limit=50, filter_type=None):
    """Retrieve detection history from MongoDB"""
    if history_collection is None:
        return [], {}
    
    try:
        query = {}
        if filter_type and filter_type in ["REAL", "FAKE"]:
            query["prediction"] = filter_type
        
        records = list(history_collection.find(query).sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string and add timestamp_iso for sorting
        for record in records:
            record["_id"] = str(record["_id"])
            if "timestamp" in record and isinstance(record["timestamp"], datetime):
                record["timestamp_iso"] = record["timestamp"].isoformat()
                record["timestamp"] = record["timestamp_iso"]
            elif "date" in record and "time" in record:
                record["timestamp_iso"] = f"{record['date']}T{record['time']}"
        
        # Get stats
        total = history_collection.count_documents({})
        real_count = history_collection.count_documents({"prediction": "REAL"})
        fake_count = history_collection.count_documents({"prediction": "FAKE"})
        
        stats = {
            "total": total,
            "real": real_count,
            "fake": fake_count
        }
        
        return records, stats
    except Exception as e:
        print(f"Error retrieving from MongoDB: {e}")
        return [], {}

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_video(video_path):
    """Process video and return prediction with proper normalization"""
    import time
    
    frames = load_video(video_path)
    
    if len(frames) == 0:
        return None, None, None, None
    
    # Try NIM face detection first (falls back automatically)
    faces = None
    if nim.enabled:
        nim_faces = nim.detect_faces_nim(frames[0])
        if nim_faces:
            # Use NIM-detected faces
            faces = extract_faces_from_bboxes(frames, nim_faces, resize_to=(128,128))
            print("✅ Using NIM face detection")
    
    # Fallback to original method
    if faces is None or len(faces) == 0:
        faces = extract_faces_from_frames(frames, resize_to=(128,128))
        print("📦 Using original face detection")
    
    if len(faces) == 0:
        return None, None, None, None
    
    # Convert to float and normalize to [0, 1]
    faces = np.array(faces, dtype=np.float32)
    if faces.max() > 1.0:
        faces = faces / 255.0
    
    # ✅ CRITICAL: ImageNet normalization (the fix!)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    faces = (faces - mean) / std
    
    print(f"✅ Preprocessing: {len(faces)} faces normalized (min={faces.min():.3f}, max={faces.max():.3f})")
    
    SEQ_LEN = 20
    
    if len(faces) < SEQ_LEN:
        pad = np.repeat(faces[-1:], SEQ_LEN - len(faces), axis=0)
        faces = np.concatenate([faces, pad], axis=0)
    
    indices = np.linspace(0, len(faces)-1, SEQ_LEN).astype(int)
    faces = faces[indices]
    
    # Transpose from (seq_len, H, W, C) to (seq_len, C, H, W)
    faces = np.transpose(faces, (0, 3, 1, 2))
    
    tensor = torch.tensor(faces, dtype=torch.float32).unsqueeze(0).to(device)
    
    model_info = get_active_model_info()
    
    if _active_model_type == "ensemble_3":
        ensemble = get_ensemble_3()
        if ensemble and len(ensemble.models) >= 3:
            try:
                result = ensemble.predict_smart(tensor)
                print(f"🤖 Smart Ensemble: {result['method']}")
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
    
    # Fallback to single model
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
# Web Routes
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
    model_info = get_active_model_info()
    return render_template("settings.html", 
                          current_model=_active_model_type,
                          model_info=model_info)

@app.route("/history")
def history():
    """History page - shows all detections"""
    records, stats = get_history_from_mongodb(limit=100)
    return render_template("history.html", records=records, stats=stats)

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/switch_model/<model_type>")
def switch_model(model_type):
    if set_active_model(model_type):
        flash(f"Switched to {model_type} model")
    else:
        flash(f"Invalid model type: {model_type}")
    return redirect(url_for("settings"))

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
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
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)

    try:
        label, confidence, proba, method = predict_video(filepath)

        if label is None:
            flash("Could not process video (no faces detected)")
            return redirect(url_for("index"))

        processing_time = time.time() - start_time
        
        model_info = get_active_model_info()
        model_used = f"{model_info['type']} ({method})"
        
        # Save to MongoDB
        save_detection_to_mongodb(
            video_name=filename,
            prediction=label,
            confidence=confidence,
            probability=proba,
            model_used=model_used,
            method=method,
            file_size_mb=file_size,
            processing_time_sec=processing_time
        )
        
        print(f"Final prediction: {label}, Confidence: {confidence}%, Time: {processing_time:.2f}s")

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            probability=round(proba, 4),
            is_deepfake=(label == "FAKE"),
            filename=filename,
            model_used=model_used,
            processing_time=round(processing_time, 2)
        )

    except Exception as e:
        flash(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return redirect(url_for("index"))

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ---------------------------------
# API Routes for AJAX/Frontend
# ---------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    start_time = time.time()
    
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    file = request.files["video"]
    
    if file.filename == "":
        return jsonify({"error": "No video selected"}), 400
    
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    
    try:
        label, confidence, proba, method = predict_video(filepath)
        
        if label is None:
            return jsonify({"error": "Could not process video"}), 500
        
        processing_time = time.time() - start_time
        
        model_info = get_active_model_info()
        model_used = f"{model_info['type']} ({method})"
        
        # Save to MongoDB
        record_id = save_detection_to_mongodb(
            video_name=filename,
            prediction=label,
            confidence=confidence,
            probability=proba,
            model_used=model_used,
            method=method,
            file_size_mb=file_size,
            processing_time_sec=processing_time
        )
        
        return jsonify({
            "success": True,
            "detection": {
                "id": record_id,
                "prediction": label,
                "confidence": confidence,
                "probability": float(proba),
                "model_used": model_used,
                "processing_time_sec": round(processing_time, 2)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/api/history", methods=["GET"])
def api_history():
    """Get detection history as JSON"""
    limit = int(request.args.get('limit', 50))
    filter_type = request.args.get('filter', None)
    
    records, stats = get_history_from_mongodb(limit=limit, filter_type=filter_type)
    
    return jsonify({
        "success": True,
        "detections": records,
        "stats": stats
    })

@app.route("/api/history/<record_id>", methods=["DELETE"])
def api_delete_history(record_id):
    """Delete a history record"""
    if history_collection is None:
        return jsonify({"error": "MongoDB not available"}), 500
    
    try:
        result = history_collection.delete_one({"_id": ObjectId(record_id)})
        return jsonify({"success": result.deleted_count > 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history/clear", methods=["DELETE"])
def api_clear_history():
    """Clear all history"""
    if history_collection is None:
        return jsonify({"error": "MongoDB not available"}), 500
    
    result = history_collection.delete_many({})
    return jsonify({"success": True, "deleted_count": result.deleted_count})

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Get system statistics"""
    records, stats = get_history_from_mongodb(limit=1)
    
    # Get average confidence
    avg_confidence = 0
    if history_collection is not None:
        pipeline = [{"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}]
        result = list(history_collection.aggregate(pipeline))
        avg_confidence = round(result[0]["avg"], 1) if result else 0
    
    return jsonify({
        "success": True,
        "stats": {
            "total_detections": stats.get("total", 0),
            "real_detections": stats.get("real", 0),
            "fake_detections": stats.get("fake", 0),
            "avg_confidence": avg_confidence,
            "model_accuracy": "93.4%",
            "dataset_size": 3431
        }
    })

# ---------------------------------
# Analysis API Routes
# ---------------------------------

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
        "learning_rate_schedule.png",
        "accuracy_curve.png",
        "loss_curve.png",
        "training_curves.png"
    ]
    
    outputs_dir = os.path.join(PROJECT_ROOT, "training", "outputs")
    
    for img_file in image_files:
        img_path = os.path.join(outputs_dir, img_file)
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                images[img_file] = base64.b64encode(f.read()).decode('utf-8')
                print(f"✅ Loaded image: {img_file}")
        else:
            print(f"⚠️ Image not found: {img_file}")
    
    return jsonify(images)

# ---------------------------------
# AI Agent Routes
# ---------------------------------

@app.route("/api/ai/explain", methods=["POST"])
def ai_explain():
    """Get AI explanation for a detection"""
    data = request.json
    video_name = data.get('video_name', 'Unknown')
    prediction = data.get('prediction', 'UNKNOWN')
    confidence = data.get('confidence', 0)
    individual_probs = data.get('individual_probs', [0, 0, 0])
    
    explanation = ai_agent.explain_detection(
        video_name, prediction, confidence, individual_probs
    )
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })

@app.route("/api/ai/analyze", methods=["GET"])
def ai_analyze():
    """Get AI analysis of detection history"""
    records, stats = get_history_from_mongodb(limit=50)
    
    # Get recent detections (last 7 days)
    recent = [r for r in records if 'date' in r]
    
    analysis = ai_agent.analyze_history(stats, recent)
    
    return jsonify({
        "success": True,
        "analysis": analysis,
        "stats": stats
    })

@app.route("/api/ai/suggest", methods=["POST"])
def ai_suggest():
    """Get action suggestions"""
    data = request.json
    prediction = data.get('prediction', 'UNKNOWN')
    confidence = data.get('confidence', 0)
    
    suggestions = ai_agent.suggest_action(prediction, confidence)
    
    return jsonify({
        "success": True,
        "suggestions": suggestions
    })

@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    """Chat with AI assistant"""
    data = request.json
    message = data.get('message', '')
    context = data.get('context', None)
    
    response = ai_agent.chat(message, context)
    
    return jsonify({
        "success": True,
        "response": response
    })

# ---------------------------------
# Run server
# ---------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Deepfake Detection Web App")
    print("="*60)
    print(f"Device: {device}")
    print(f"MongoDB: {'Connected' if history_collection is not None else 'Not Connected'}")
    
    # Load all models on startup
    single = get_single_model()
    ensemble2 = get_ensemble_2()
    ensemble3 = get_ensemble_3()
    
    print("\n📊 Available Models:")
    print(f"  • Single Model: {'✅' if single is not None else '❌'}")
    print(f"  • 2-Model Ensemble: {'✅' if ensemble2 is not None and len(ensemble2.models)>=2 else '❌'}")
    print(f"  • 3-Model Ensemble: {'✅' if ensemble3 is not None and len(ensemble3.models)>=2 else '❌'}")
    
    active = get_active_model_info()
    print(f"\n🎯 Active Model: {active['type']} ({active.get('method', 'standard')})")
    
    print(f"\nUpload folder: {app.config['UPLOAD_FOLDER']}")
    print("-"*60)
    print("Routes:")
    print("  /              - Home page")
    print("  /about         - About page")
    print("  /documentation - Documentation page")
    print("  /settings      - Model selection page")
    print("  /history       - Detection history page")
    print("  /analysis      - Performance analysis dashboard")
    print("  /predict       - Prediction form POST")
    print("  /api/*         - JSON APIs")
    print("-"*60)
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*60)

    app.run(host="0.0.0.0", port=5000, debug=True)