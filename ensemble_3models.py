import torch
import numpy as np
from models.cnn_lstm_model import CNNLSTM

class Ensemble3Models:
    def __init__(self, device):
        self.device = device
        self.models = []
        self.load_models()
    
    def load_models(self):
        paths = [
            "models/deepfake_model_ensemble_A.pth",
            "models/deepfake_model_ensemble_B.pth",
            "models/deepfake_model_ensemble_C.pth"
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
    
    def predict_standard(self, video_tensor):
        """Standard majority voting"""
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
            'individual_probs': [round(float(p), 4) for p in probs],
            'method': 'voting'
        }
    
    def predict_smart(self, video_tensor):
        """
        Smart prediction that trusts Model A more on fake videos
        Model A is index 0, B is index 1, C is index 2
        """
        if len(self.models) < 3:
            return self.predict_standard(video_tensor)
        
        probs = []
        with torch.no_grad():
            for model in self.models:
                output = model(video_tensor)
                prob = torch.sigmoid(output).cpu().numpy()[0][0]
                probs.append(prob)
        
        # Model A (index 0) is our fake detection champion
        model_a_prob = probs[0]
        model_b_prob = probs[1]
        model_c_prob = probs[2]
        
        print(f"🤖 Smart Predictor - Raw probs: A={model_a_prob:.3f}, B={model_b_prob:.3f}, C={model_c_prob:.3f}")
        
        # CASE 1: Model A is very confident it's FAKE (>70%) and at least one other model is unsure
        if model_a_prob > 0.8 and (model_b_prob < 0.4 or model_c_prob < 0.4):
            label = "FAKE"
            # Confidence based on Model A's confidence and agreement level
            agreement = sum(1 for p in probs if p > 0.5)
            confidence = (model_a_prob * 0.8 + (agreement/3) * 0.2) * 100
            method = "smart (trusting Model A)"
        
        # CASE 2: All models agree
        elif all(p > 0.5 for p in probs) or all(p < 0.5 for p in probs):
            avg_prob = np.mean(probs)
            label = "FAKE" if avg_prob > 0.5 else "REAL"
            confidence = abs(avg_prob - 0.5) * 200
            method = "unanimous"
        
        # CASE 3: Majority voting for other cases
        else:
            votes = sum(1 for p in probs if p > 0.5)
            if votes >= 2:
                label = "FAKE"
                confidence = (votes / 3) * 100
            else:
                label = "REAL"
                confidence = ((3 - votes) / 3) * 100
            method = "majority voting"
        
        # Calculate final probability (weighted average favoring Model A)
        weights = [0.5, 0.25, 0.25]  # Model A gets 50% weight
        weighted_prob = np.average(probs, weights=weights)
        
        return {
            'prediction': label,
            'confidence': round(min(100, confidence), 1),
            'probability': round(float(weighted_prob), 4),
            'individual_probs': [round(float(p), 4) for p in probs],
            'method': method,
            'model_a_trusted': model_a_prob > 0.7
        }
    
    def predict(self, video_tensor):
        """Default predict method uses smart prediction"""
        return self.predict_smart(video_tensor)
    
    def get_model_count(self):
        return len(self.models)