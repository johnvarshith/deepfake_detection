"""
NVIDIA NIM API Configuration for DeepGuard AI
Uses Gemma-3-27B as the AI Assistant model
"""

import requests
import json
import base64
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class NIMClient:
    """Client for NVIDIA NIM API"""
    
    # API Configuration
    INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    # Your API Key (securely loaded)
    API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
    
    # Model Selection - Gemma-3-27B (Best overall for DeepGuard)
    MODEL = "google/gemma-3-27b-it"
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and bool(self.API_KEY)
        self.conversation_history = []
        
        if self.enabled:
            print(f"✅ NIM Client initialized with model: {self.MODEL}")
        else:
            print("⚠️ NIM Client disabled: Missing API Key or explicitly disabled.")
    
    def chat(self, message: str, context: Optional[Dict] = None, stream: bool = False) -> str:
        """
        Send a chat message to the NIM API
        """
        if not self.enabled:
            return None
        
        try:
            # Build messages with context
            messages = self._build_messages(message, context)
            
            # Prepare payload
            payload = {
                "model": self.MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.20,  # Lower = more focused responses
                "top_p": 0.70,
                "stream": stream
            }
            
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Accept": "text/event-stream" if stream else "application/json",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                if stream:
                    return self._parse_stream_response(response)
                else:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            else:
                print(f"NIM API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"NIM chat error: {e}")
            return None
    
    def _build_messages(self, message: str, context: Optional[Dict] = None) -> list:
        """Build message array with system prompt and context"""
        messages = []
        
        # System prompt with DeepGuard project context
        system_prompt = """You are DeepGuard AI Assistant, integrated into a deepfake detection web application.

PROJECT CONTEXT:
- DeepGuard AI uses a 3-model CNN-LSTM ensemble for video deepfake detection
- Model A: 89.5% accuracy (fake detection specialist)
- Model B: 94.0% accuracy (best overall performer)
- Model C: 92.4% accuracy (edge case handler)
- Overall ensemble accuracy: 93.4% on 3,431 test videos
- Tech stack: Python, Flask, PyTorch, MongoDB, Tailwind CSS

YOUR ROLE:
- Explain deepfake detection results in simple terms
- Suggest actions based on REAL/FAKE predictions
- Answer questions about deepfakes, AI, and the project
- Be helpful, friendly, and concise
- Use the context provided to give relevant responses

If asked about current detection results, use the provided context to answer specifically."""
        
        # Add context if available
        if context:
            context_str = self._format_context(context)
            if context_str:
                system_prompt += f"\n\nCURRENT CONTEXT:\n{context_str}"
                
        messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _format_context(self, context: Dict) -> str:
        """Format context dictionary into a readable string"""
        parts = []
        
        if "prediction" in context:
            parts.append(f"Recent detection: {context.get('prediction')} with {context.get('confidence', 0)}% confidence.")
            if "video_name" in context:
                parts.append(f"Video: {context['video_name']}")
            if "individual_probs" in context:
                probs = context["individual_probs"]
                parts.append(f"Model probabilities: A={probs[0]:.1%}, B={probs[1]:.1%}, C={probs[2]:.1%}")
        
        if "page" in context:
            parts.append(f"User is on the {context['page']} page.")
        
        if "stats" in context:
            stats = context["stats"]
            parts.append(f"History stats: {stats.get('total', 0)} total, {stats.get('real', 0)} real, {stats.get('fake', 0)} fake.")
        
        return " ".join(parts) if parts else ""
    
    def _parse_stream_response(self, response) -> str:
        """Parse streaming response"""
        full_response = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                        except json.JSONDecodeError:
                            pass
        return full_response
    
    def analyze_video_frame(self, frame_base64: str, question: str = "Is there anything suspicious in this frame?") -> str:
        """
        Use Gemma-3's multimodal capability to analyze a video frame
        """
        if not self.enabled:
            return None
        
        try:
            payload = {
                "model": self.MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.20
            }
            
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.INVOKE_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return None
                
        except Exception as e:
            print(f"Frame analysis error: {e}")
            return None

# Singleton instance
nim_client = NIMClient(enabled=True)
