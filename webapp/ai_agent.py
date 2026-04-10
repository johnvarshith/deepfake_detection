"""
DeepGuard AI Assistant - Now powered by NVIDIA NIM (Gemma-3-27B)
Answers ANY question naturally with deep project knowledge
"""

import os
import json
from datetime import datetime, timedelta
from collections import Counter

# Try to import NIM Gemma API Integration
try:
    from nim_config import nim_client
except Exception as e:
    print(f"⚠️ NIM Config Integration not available: {e}")
    nim_client = None

class DeepGuardAIAgent:
    def __init__(self):
        self.use_nim = nim_client.enabled if nim_client else False
        self.conversation_history = []
        
        if self.use_nim:
            print("✅ AI Agent using NVIDIA NIM (Gemma-3-27B)")
        else:
            print("⚠️ AI Agent using offline fallback")
            
        # Comprehensive project knowledge base
        self.project_info = {
            "name": "DeepGuard AI",
            "version": "1.0.0",
            "purpose": "Deepfake detection web application for identifying AI-manipulated videos",
            "models": {
                "ensemble": "3-model CNN-LSTM",
                "model_a": {"accuracy": 89.5, "specialty": "Fake detection specialist - best at catching deepfakes"},
                "model_b": {"accuracy": 94.0, "specialty": "Best overall performer - balanced precision and recall"},
                "model_c": {"accuracy": 92.4, "specialty": "Edge case handler - robust to video compression"},
                "overall": {"accuracy": 93.4, "dataset_size": 3431}
            },
            "tech_stack": ["Python", "Flask", "PyTorch", "MongoDB", "Tailwind CSS", "WebSockets", "OpenCV"],
            "features": [
                "Real-time video analysis",
                "Detection history with MongoDB",
                "AI-powered explanations",
                "WebSocket progress tracking",
                "Ensemble model selection"
            ]
        }
        
    def chat(self, user_message, context=None):
        """Chat with AI - Primary: NIM, Fallback: Smart responses"""
        # Try NIM first
        if self.use_nim:
            response = nim_client.chat(user_message, context)
            if response:
                return response
        
        # Fallback to smart offline responses
        return self._smart_fallback_chat(user_message, context)
        
    def explain_detection(self, video_name, prediction, confidence, individual_probs=None):
        """Explain detection using NIM"""
        context = {
            "prediction": prediction,
            "confidence": confidence,
            "video_name": video_name,
            "individual_probs": individual_probs or [0, 0, 0]
        }
        
        if self.use_nim:
            prompt = f"Explain this deepfake detection result in 2-3 sentences for a non-technical user."
            response = nim_client.chat(prompt, context)
            if response:
                return response
        
        # Fallback
        if prediction == "FAKE":
            return f"The model detected '{video_name}' as FAKE with {confidence}% confidence. This indicates suspicious artifacts typical of AI manipulation."
        return f"The model detected '{video_name}' as REAL with {confidence}% confidence. It appears to be an authentic video."
        
    def analyze_frame(self, frame_base64):
        """Analyze a video frame using Gemma-3's vision capabilities"""
        if self.use_nim:
            return nim_client.analyze_video_frame(
                frame_base64, 
                "Analyze this video frame for signs of deepfake manipulation."
            )
        return None
        
    def analyze_history(self, stats, recent_records):
        """Analyze detection history"""
        context = {"stats": stats}
        if self.use_nim:
            prompt = "Analyze these deepfake detection statistics and provide a 2-sentence summary of the overall trend."
            response = nim_client.chat(prompt, context)
            if response:
                return response
                
        return f"You have {stats.get('total', 0)} total detections: {stats.get('fake', 0)} fake and {stats.get('real', 0)} real. Keep analyzing to build a stronger dataset."
        
    def suggest_action(self, prediction, confidence):
        """Suggest action based on prediction"""
        if self.use_nim:
            prompt = f"Suggest 3 bullet-point actions given this deepfake prediction is {prediction} with {confidence}% confidence."
            response = nim_client.chat(prompt, None)
            if response:
                return response
        
        # Fallback
        if prediction == 'FAKE':
            return "• Do not share this video further.\n• Seek verification from official sources.\n• Be aware of potential misinformation."
        else:
            return "• Video appears safe to share.\n• Still apply general critical thinking."

    def _smart_fallback_chat(self, user_message, context=None):
        """Offline fallback responses"""
        msg = user_message.lower().strip()
        
        if "hello" in msg or "hi" in msg:
            return "👋 Hello! I'm DeepGuard AI Assistant. How can I help you with deepfake detection today?"
        if any(w in msg for w in ['model', 'accuracy', 'ensemble']):
            return f"🤖 We use a 3-model CNN-LSTM ensemble. Model A (89.5%) specializes in fakes, Model B (94.0%) is best overall, and Model C (92.4%) handles edge cases. Overall accuracy is 93.4%."
        if "how does it work" in msg or "process" in msg:
            return "🔬 DeepGuard extracts faces from videos, processes 20-frame sequences through three models, and uses smart voting to determine REAL vs FAKE."
        
        if context and isinstance(context, dict) and 'prediction' in context:
            pred = context['prediction']
            conf = context['confidence']
            if 'why' in msg or 'explain' in msg:
                return f"🔍 This video was classified as {pred} with {conf}% confidence by our models. It detects patterns like unnatural facial movements or lighting."
                
        return "I'm here to help with deepfake detection! You can ask about results, accuracy, how the system works, or upload a video to get started. (Currently in offline fallback mode)."

# Create singleton
ai_agent = DeepGuardAIAgent()