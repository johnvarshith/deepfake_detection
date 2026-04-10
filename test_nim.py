"""
Test NVIDIA NIM Integration
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from webapp.nim_config import nim_client

# Test 1: Basic chat
print("🧪 Test 1: Basic Chat")
print("-" * 40)
response = nim_client.chat("What is DeepGuard AI?")
print(f"Response: {response}\n")

# Test 2: With context
print("🧪 Test 2: Chat with Detection Context")
print("-" * 40)
context = {
    "prediction": "FAKE",
    "confidence": 94.5,
    "video_name": "suspicious_video.mp4",
    "individual_probs": [0.92, 0.88, 0.95]
}
response = nim_client.chat("Explain this result to me", context)
print(f"Response: {response}\n")

# Test 3: Action suggestions
print("🧪 Test 3: Action Suggestions")
print("-" * 40)
response = nim_client.chat("What should I do with this video?")
print(f"Response: {response}\n")

print("✅ NIM Integration Test Complete!")
