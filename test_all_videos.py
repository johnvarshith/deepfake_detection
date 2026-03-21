import os
import sys
import torch
import numpy as np

# Try to import tabulate, but don't fail if not available
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("⚠️  tabulate not installed. Run: pip install tabulate")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ensemble_3models import Ensemble3Models
from utils.video_utils import load_video
from preprocessing.face_detection import extract_faces_from_frames

def test_single_video(video_path, ensemble):
    """Test a single video and return results"""
    try:
        # Load and process video
        frames = load_video(video_path)
        if len(frames) == 0:
            return None, "No frames extracted"
        
        faces = extract_faces_from_frames(frames, resize_to=(128,128))
        if len(faces) == 0:
            return None, "No faces detected"
        
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
        
        # Get smart prediction
        result = ensemble.predict_smart(tensor)
        
        return result, None
    except Exception as e:
        return None, str(e)

def test_folder(folder_path, ensemble, folder_type):
    """Test all videos in a folder"""
    results = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return results
    
    # Get all video files
    video_files = []
    for ext in ['.mp4', '.MOV', '.mov', '.avi', '.mkv', '.webm']:
        video_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext.lower())])
    
    video_files = sorted(video_files)
    
    if len(video_files) == 0:
        print(f"⚠️  No video files found in {folder_path}")
        return results
    
    print(f"\n📁 Testing {folder_type} videos ({len(video_files)} files)")
    print("=" * 80)
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"  Processing: {video_file}...", end="")
        
        result, error = test_single_video(video_path, ensemble)
        
        if error:
            print(f" ❌ Error: {error}")
            results.append([video_file, "ERROR", error, "-", "-", "-", "-"])
        else:
            print(f" ✅ {result['prediction']} ({result['confidence']}%)")
            results.append([
                video_file,
                result['prediction'],
                f"{result['confidence']}%",
                f"A:{result['individual_probs'][0]:.3f}",
                f"B:{result['individual_probs'][1]:.3f}",
                f"C:{result['individual_probs'][2]:.3f}",
                result['method']
            ])
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load ensemble
    ensemble = Ensemble3Models(device)
    if len(ensemble.models) == 0:
        print("❌ No models loaded! Exiting.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(ensemble.models)} models\n")
    
    # Test folders
    test_root = os.path.join("C:\\", "games", "deepfake_detection_project", "test")
    
    fake_folder = os.path.join(test_root, "deepfake")
    real_folder = os.path.join(test_root, "video")
    
    print(f"Looking for videos in:")
    print(f"  Fake folder: {fake_folder}")
    print(f"  Real folder: {real_folder}")
    print()
    
    fake_results = test_folder(fake_folder, ensemble, "FAKE")
    real_results = test_folder(real_folder, ensemble, "REAL")
    
    # Calculate statistics
    fake_correct = sum(1 for r in fake_results if r[1] == "FAKE")
    fake_total = len([r for r in fake_results if r[1] != "ERROR"])
    real_correct = sum(1 for r in real_results if r[1] == "REAL")
    real_total = len([r for r in real_results if r[1] != "ERROR"])
    
    print("\n" + "=" * 100)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 100)
    
    if fake_results:
        print("\n🔴 FAKE VIDEOS:")
        if HAS_TABULATE:
            fake_table = [[r[0][:25], r[1], r[2], f"{r[3]} {r[4]} {r[5]}", r[6]] 
                         for r in fake_results if r[1] != "ERROR"]
            print(tabulate(fake_table, headers=["Video", "Pred", "Conf", "Individual", "Method"], 
                          tablefmt="grid", maxcolwidths=[25, 6, 6, 20, 15]))
        else:
            for r in fake_results:
                if r[1] != "ERROR":
                    print(f"  {r[0][:30]:30} → {r[1]} ({r[2]})")
    
    if real_results:
        print("\n🟢 REAL VIDEOS:")
        if HAS_TABULATE:
            real_table = [[r[0][:25], r[1], r[2], f"{r[3]} {r[4]} {r[5]}", r[6]] 
                         for r in real_results if r[1] != "ERROR"]
            print(tabulate(real_table, headers=["Video", "Pred", "Conf", "Individual", "Method"], 
                          tablefmt="grid", maxcolwidths=[25, 6, 6, 20, 15]))
        else:
            for r in real_results:
                if r[1] != "ERROR":
                    print(f"  {r[0][:30]:30} → {r[1]} ({r[2]})")
    
    print("\n" + "=" * 100)
    print("📈 PERFORMANCE METRICS")
    print("=" * 100)
    
    if fake_total > 0:
        print(f"🔴 Fake Videos: {fake_correct}/{fake_total} correct ({fake_correct/fake_total*100:.1f}%)")
    else:
        print("🔴 Fake Videos: No videos tested")
    
    if real_total > 0:
        print(f"🟢 Real Videos: {real_correct}/{real_total} correct ({real_correct/real_total*100:.1f}%)")
    else:
        print("🟢 Real Videos: No videos tested")
    
    if fake_total + real_total > 0:
        overall = (fake_correct + real_correct) / (fake_total + real_total) * 100
        print(f"🎯 Overall: {overall:.1f}%")
    
    # Save results to file
    with open("test_results.txt", "w") as f:
        f.write("DEEPFAKE DETECTION TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tested on: {fake_total + real_total} videos\n\n")
        
        if fake_results:
            f.write("FAKE VIDEOS:\n")
            for r in fake_results:
                if r[1] != "ERROR":
                    f.write(f"{r[0]}: {r[1]} ({r[2]}) - {r[3]} {r[4]} {r[5]} [{r[6]}]\n")
        
        if real_results:
            f.write("\nREAL VIDEOS:\n")
            for r in real_results:
                if r[1] != "ERROR":
                    f.write(f"{r[0]}: {r[1]} ({r[2]}) - {r[3]} {r[4]} {r[5]} [{r[6]}]\n")
    
    print(f"\n📁 Results saved to test_results.txt")