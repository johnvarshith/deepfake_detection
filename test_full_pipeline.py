# test_full_pipeline.py
import sys
sys.path.append('.')
from webapp.app import predict_video
from webapp.ai_agent import ai_agent

# Test with a real video
video_path = "test_video.mp4"  # Replace with actual video

print("🎬 Testing Full Pipeline with Real Video")
print("="*50)

# Run prediction
label, confidence, proba, method = predict_video(video_path)

print(f"\n📊 Detection Result:")
print(f"   Prediction: {label}")
print(f"   Confidence: {confidence}%")
print(f"   Probability: {proba:.4f}")
print(f"   Method: {method}")

# Test AI explanation
print(f"\n🤖 AI Explanation:")
explanation = ai_agent.explain_detection(
    "test_video.mp4", 
    label, 
    confidence, 
    [proba, 1-proba, proba]  # Mock individual probs
)
print(f"   {explanation}")

# Test AI chat
print(f"\n💬 AI Chat Test:")
response = ai_agent.chat("Why did you make this prediction?", 
                        {"prediction": label, "confidence": confidence})
print(f"   User: Why did you make this prediction?")
print(f"   AI: {response}")