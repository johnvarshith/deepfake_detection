
# 🛡️ DeepGuard AI - Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0%2B-green)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-ready deepfake detection system using **CNN-LSTM architecture with attention mechanism** and **smart ensemble voting**. Achieves **93.4% accuracy** on the DFD dataset with real-time detection and persistent history tracking.

---

## 🎯 Key Features

- 🧠 **CNN-LSTM Architecture** – EfficientNet-B0 for spatial features + Bi-LSTM for temporal patterns
- 🔍 **Attention Mechanism** – Learns which frames are most important for detection
- 🤖 **Smart Ensemble Voting** – 3-model ensemble with 93.4% accuracy
- 🚀 **Real-time Detection** – 2–5 seconds per video with GPU acceleration
- 🌐 **Web Interface** – Modern UI with drag-and-drop upload, model switching, and animated confidence bars
- 📊 **Detection History** – All detections stored in MongoDB with filtering and analytics
- 📈 **Performance Dashboard** – Interactive charts for training curves, confusion matrix, ROC curve, and model comparison
- 🔄 **Model Switching** – Toggle between Single, 2-Ensemble, and 3-Ensemble models

---

## 📊 Performance Results

| Model | Accuracy | AUC |
|-------|----------|-----|
| Model A | 89.5% | 0.910 |
| Model B | 94.0% | 0.971 |
| Model C | 92.4% | 0.975 |
| **Ensemble (Smart)** | **93.4%** | **0.969** |

### Test Set (516 Videos)

| Metric | Real | Fake |
|--------|------|------|
| Precision | 61% | **99%** |
| Recall | 88% | **94%** |
| F1-Score | 72% | **96%** |

### Confusion Matrix

              Predicted
              REAL    FAKE
Actual  REAL    46       6
        FAKE    30     434


- **Fake Detection Rate:** 94% (434/464)
- **Real Detection Rate:** 88% (46/52)
- **Overall Accuracy:** 93.4%
- **ROC-AUC:** 0.969

---

## 🏗️ Architecture Overview

```
Input Video → Face Detection (MTCNN) → Frame Sampling (20 frames, 128×128)
                    ↓
         Spatial Features (EfficientNet-B0)
                    ↓
         Temporal Modeling (Bi-LSTM + Attention)
                    ↓
         Smart Ensemble (3 Models with Voting)
                    ↓
         Output: REAL / FAKE (with confidence %)
```

---

## 🚀 Quick Start

### 🔧 Prerequisites
- Python 3.8+
- CUDA GPU (recommended for real-time inference)
- MongoDB (for history tracking)
- 8GB+ RAM

### ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/johnvarshith/deepguard-ai.git
cd deepguard-ai

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 📥 Download Pretrained Models

Download from: 👉 [GitHub Releases](https://github.com/johnvarshith/deepguard-ai/releases)

Place inside `/models`:
- `deepfake_model_ensemble_A.pth`
- `deepfake_model_ensemble_B.pth`
- `deepfake_model_ensemble_C.pth`

### 🗄️ Start MongoDB (Optional for History)

```bash
# Windows (if installed as service)
net start MongoDB

# Linux/Mac
sudo systemctl start mongod

# Or run MongoDB locally
mongod --dbpath ./data
```

### ▶️ Run Web App

```bash
python webapp/app.py
```

Open: 👉 [http://localhost:5000](http://localhost:5000)

---

## 📁 Project Structure

```
deepguard-ai/
│
├── models/                          # Model architectures
│   ├── cnn_lstm_model.py            # CNN-LSTM with attention
│   ├── ensemble_3models.py          # 3-model smart ensemble
│   └── ensemble_2models.py          # 2-model ensemble
│
├── webapp/                          # Flask web application
│   ├── app.py                       # Main application with MongoDB
│   ├── templates/
│   │   ├── index.html               # Upload page
│   │   ├── result.html              # Results page
│   │   ├── about.html               # Project info
│   │   ├── documentation.html       # API docs
│   │   ├── settings.html            # Model selection
│   │   ├── history.html             # Detection history
│   │   └── analysis.html            # Performance dashboard
│   └── uploads/                     # Temporary uploads
│
├── preprocessing/                   # Face extraction
│   ├── face_detection.py            # MTCNN face detection
│   ├── extract_faces.py             # Face extraction pipeline
│   └── data_loader.py               # Dataset loader
│
├── training/                        # Training & evaluation
│   ├── train_ensemble.py            # Train 3 models
│   ├── evaluate_ensemble.py         # Evaluate ensemble
│   └── outputs/                     # PNG results
│       ├── ensemble_confusion_matrix.png
│       ├── ensemble_roc_curve.png
│       └── model_comparison.png
│
├── utils/                           # Utilities
│   └── video_utils.py               # Video loading functions
│
├── requirements.txt
└── README.md
```

---

## 🧪 Training Your Own Models

```bash
# Extract faces from videos
python preprocessing/extract_faces.py

# Train ensemble (3 models with different seeds)
python training/train_ensemble.py

# Evaluate ensemble performance
python training/evaluate_ensemble.py
```

---

## 📡 API Endpoints

### 🔹 POST `/api/predict`
Upload video for deepfake detection.

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "video=@video.mp4"
```

**Response:**
```json
{
  "success": true,
  "prediction": "FAKE",
  "confidence": 85.5,
  "probability": 0.927,
  "filename": "video.mp4",
  "model_used": "3-Model Ensemble (Smart)"
}
```

### 🔹 GET `/api/history`
Get detection history with optional filtering.

```bash
curl -X GET "http://localhost:5000/api/history?limit=50&filter=FAKE"
```

### 🔹 GET `/api/stats`
Get system statistics (total detections, accuracy, etc.)

### 🔹 DELETE `/api/history/clear`
Clear all detection history.

---

## 🖼️ Results & Visualizations

### Confusion Matrix
(training/outputs/ensemble_confusion_matrix.png)

### ROC Curve (AUC = 0.969)
<img width="800" height="600" alt="ensemble_roc_curve" src="https://github.com/user-attachments/assets/e9abbee7-781a-4fc6-bebb-bda2e2d44b9d" />


### Model Performance Comparison
<img width="1782" height="733" alt="model_comparison" src="https://github.com/user-attachments/assets/46b560ec-6568-4d54-abcd-5cbdc1653f38" />


### Training Curves
<img width="2082" height="730" alt="training_curves" src="https://github.com/user-attachments/assets/441927fa-3311-4bb8-89cd-344bc0ecabe4" />


### Confidence Distribution
<img width="1482" height="880" alt="confidence_distribution" src="https://github.com/user-attachments/assets/f97fc25a-44a6-4366-a548-b71812687611" />


### Learning Rate Schedule
<img width="1481" height="880" alt="learning_rate_schedule" src="https://github.com/user-attachments/assets/77e7aa6e-7e13-45b7-a792-67ef678d3af6" />

### Accuracy curve
<img width="1481" height="880" alt="learning_rate_schedule" src="https://github.com/user-attachments/assets/91bcbc82-a0ba-4163-a11e-d4453c409bd2" />

---

## 📊 Dataset

| Category | Count | Source |
|----------|-------|--------|
| **Real Videos** | 363 | DFD Original Sequences |
| **Fake Videos** | 3,068 | DFD Manipulated Sequences |
| **Total** | **3,431** | Google/Jigsaw Dataset |

**Scene Distribution (16 scenes):**
- exit_phone_room, hugging_happy, kitchen_pan, kitchen_still
- meeting_serious, outside_talking, podium_speech_happy
- secret_conversation, talking_against_wall, talking_angry_couch
- walking_and_outside_surprised, walking_down_indoor_hall_disgust
- walking_down_street_outside_angry, walking_outside_cafe_disgusted
- walk_down_hall_angry, and more

**Split:** 70% Training / 15% Validation / 15% Testing

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch 2.0+, EfficientNet-B0, Bi-LSTM, Attention |
| **Computer Vision** | OpenCV, MTCNN (FaceNet-PyTorch) |
| **Backend** | Flask, MongoDB (PyMongo) |
| **Frontend** | HTML5, Tailwind CSS, JavaScript, Chart.js |
| **GPU Acceleration** | CUDA, NVIDIA RTX 3050 (4GB VRAM) |
| **Visualization** | Matplotlib, Seaborn |

---

## 📈 Future Improvements

- [ ] Add more datasets (Celeb-DF, FaceForensics++)
- [ ] Implement Transformer-based architecture
- [ ] Add audio analysis for multimodal detection
- [ ] Deploy as cloud service (AWS/GCP)
- [ ] Mobile app integration
- [ ] Real-time video stream detection

---

## 👨‍💻 Author

**John Varshith**
- GitHub: [@johnvarshith](https://github.com/johnvarshith)
- Email: johnvarshith2004@gmail.com
- LinkedIn: [Connect on LinkedIn](https://linkedin.com/in/johnvarshith)

---

## 🙏 Acknowledgments

- Google/Jigsaw for DFD Dataset
- FaceNet-PyTorch for MTCNN implementation
- PyTorch Team for deep learning framework
- Flask for web framework

---

## ⭐ Star this repo if you found it useful!

```

---

This README now includes:
- ✅ MongoDB history tracking
- ✅ Analysis dashboard with charts
- ✅ All API endpoints
- ✅ Complete project structure
- ✅ All result visualizations
- ✅ Detailed dataset information
- ✅ Updated tech stack

Just copy and paste this into your `README.md` file! 🚀
