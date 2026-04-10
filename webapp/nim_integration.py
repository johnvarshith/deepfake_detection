"""
NVIDIA NIM Integration for DeepGuard AI
Works alongside existing pipeline without disruption
"""

import requests
import base64
import numpy as np
from PIL import Image
import io

class NIMIntegration:
    """Optional NIM acceleration - falls back to original if unavailable"""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.endpoints = {
            "face_detection": "http://localhost:8000/v1/vision/face-detection",
            "llm": "http://localhost:8001/v1/chat/completions",
            "deepfake_model": "http://localhost:8002/v1/models/deepfake/infer"
        }
        self._check_availability()
    
    def _check_availability(self):
        """Check which NIM services are available"""
        if not self.enabled:
            return
            
        for name, url in self.endpoints.items():
            try:
                response = requests.get(f"{url.split('/v1')[0]}/health", timeout=2)
                print(f"✅ NIM {name}: available")
            except:
                print(f"⚠️ NIM {name}: not available (using fallback)")
    
    def detect_faces_nim(self, frame):
        """Fast face detection using NIM"""
        if not self.enabled:
            return None
        
        try:
            # Convert frame to base64
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            response = requests.post(
                self.endpoints["face_detection"],
                json={"image": img_b64},
                timeout=5
            )
            
            if response.status_code == 200:
                faces = response.json().get("faces", [])
                return [
                    {
                        "bbox": face["bbox"],
                        "confidence": face["confidence"],
                        "landmarks": face.get("landmarks", [])
                    }
                    for face in faces
                ]
        except Exception as e:
            print(f"NIM face detection failed: {e}")
        
        return None
    
    def run_deepfake_inference(self, tensor_data):
        """Optimized inference using TensorRT"""
        if not self.enabled:
            return None
        
        try:
            response = requests.post(
                self.endpoints["deepfake_model"],
                json={
                    "inputs": [{
                        "name": "input",
                        "shape": list(tensor_data.shape),
                        "datatype": "FP32",
                        "data": tensor_data.flatten().tolist()
                    }]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()["outputs"][0]["data"]
        except Exception as e:
            print(f"NIM inference failed: {e}")
        
        return None
    
    def chat_local_llm(self, prompt, context=None):
        """Local LLM for AI Assistant"""
        if not self.enabled:
            return None
        
        try:
            messages = []
            if context:
                messages.append({"role": "system", "content": str(context)})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                self.endpoints["llm"],
                json={
                    "model": "meta/llama3-8b-instruct",
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"NIM LLM failed: {e}")
        
        return None

# Singleton instance
nim = NIMIntegration(enabled=False)  # Set to True when NIM is available
