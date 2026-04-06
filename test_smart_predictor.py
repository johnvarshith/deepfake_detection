import torch
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ensemble_3models import Ensemble3Models
from utils.video_utils import load_video
from preprocessing.face_detection import extract_faces_from_frames

def test_smart_predictor(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = Ensemble3Models(device)
    
    print(f"\n📹 Testing: {video_path}")
    print("=" * 60)
    
    # Load and process video
    frames = load_video(video_path)
    faces = extract_faces_from_frames(frames, resize_to=(128,128))
    faces = np.array(faces, dtype=np.float32) / 255.0
    
    # Ensure 20 frames
    SEQ_LEN = 20
    if len(faces) < SEQ_LEN:
        pad = np.repeat(faces[-1:], SEQ_LEN - len(faces), axis=0)
        faces = np.concatenate([faces, pad], axis=0)
    
    indices = np.linspace(0, len(faces)-1, SEQ_LEN).astype(int)
    faces = faces[indices]
    faces = np.transpose(faces, (0, 3, 1, 2))
    tensor = torch.tensor(faces, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get predictions
    print("\n📊 Standard Voting:")
    standard = ensemble.predict_standard(tensor)
    print(f"   {standard['prediction']} ({standard['confidence']}%)")
    print(f"   Votes: {standard['votes']}")
    
    print("\n🤖 Smart Predictor:")
    smart = ensemble.predict_smart(tensor)
    print(f"   Method: {smart['method']}")
    print(f"   Individual: A={smart['individual_probs'][0]:.3f}, B={smart['individual_probs'][1]:.3f}, C={smart['individual_probs'][2]:.3f}")
    print(f"   Final: {smart['prediction']} ({smart['confidence']}%)")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_smart_predictor.py <video_path>")
        sys.exit(1)
    test_smart_predictor(sys.argv[1])