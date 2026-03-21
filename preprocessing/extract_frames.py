"""
Extract frames from video files using OpenCV.

Used for:
- Building dataset from raw videos
- Pipeline: video path -> list of frames (BGR)
"""

import os
import cv2
from utils.video_utils import MAX_FRAMES_TO_EXTRACT


def extract_frames(
    video_path,
    max_frames=MAX_FRAMES_TO_EXTRACT,
    skip_every_n=3,
):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract (for memory).
        skip_every_n: Extract every n-th frame (1 = all). Use >1 to reduce frames.

    Returns:
        List of frames as numpy arrays (H, W, 3) BGR, or empty list on error.
    """
    if not os.path.isfile(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip_every_n == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


def extract_frames_to_dir(video_path, output_dir, max_frames=MAX_FRAMES_TO_EXTRACT):
    """
    Extract frames and save as images in output_dir.
    Useful for debugging or building image datasets.

    Args:
        video_path: Path to video.
        output_dir: Directory to save frame images.
        max_frames: Max frames to save.

    Returns:
        Number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = extract_frames(video_path, max_frames=max_frames)
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(path, frame)
    return len(frames)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m preprocessing.extract_frames <video_path> [output_dir]")
        sys.exit(1)
    vpath = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "extracted_frames"
    n = extract_frames_to_dir(vpath, out)
    print(f"Saved {n} frames to {out}")
