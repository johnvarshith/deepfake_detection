import torch
import numpy as np
from models.cnn_lstm_model import CNNLSTM

class Ensemble2Models:
    def __init__(self, device):
        self.device = device
        self.models = []
        self.load_models()
    
    def load_models(self):
        paths = [
            "models/deepfake_model_ensemble_A.pth",
            "models/deepfake_model_ensemble_B.pth"
        ]
        for path in paths:
            try:
                model = CNNLSTM().to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
                model.eval()
                self.models.append(model)
                print(f"✅ Loaded {path}")
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
    
    def predict(self, video_tensor):
        """Majority voting prediction"""
        if len(self.models) < 2:
            return None
        
        probs = []
        votes = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(video_tensor)
                prob = torch.sigmoid(output).cpu().numpy()[0][0]
                probs.append(prob)
                votes.append(1 if prob > 0.5 else 0)
        
        fake_votes = sum(votes)
        real_votes = len(votes) - fake_votes
        
        if fake_votes > real_votes:
            label = "FAKE"
            confidence = (fake_votes / len(votes)) * 100
        else:
            label = "REAL"
            confidence = (real_votes / len(votes)) * 100
        
        return {
            'prediction': label,
            'confidence': round(confidence, 1),
            'probability': round(float(np.mean(probs)), 4),
            'votes': f"{fake_votes}/{len(votes)} for FAKE",
            'individual_probs': [round(float(p), 4) for p in probs]
        }