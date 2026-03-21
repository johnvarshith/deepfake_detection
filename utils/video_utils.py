"""
Video utilities for deepfake detection project.

Provides helpers for:
- Loading videos with OpenCV
- Frame extraction parameters
- Normalization and resizing
"""

import os
import cv2
import numpy as np

FRAME_SIZE = (160,160)          # smaller images, still good for deepfakes
NUM_FRAMES_PER_SEQUENCE = 20    # fewer frames per sequence
MAX_FRAMES_TO_EXTRACT = 24      # limits RAM usage
def load_video(video_path, max_frames=100):
    """Load video frames safely"""
    if not os.path.exists(video_path):
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                frames.append(frame)
    except Exception as e:
        print(f"Error reading video: {e}")
    finally:
        cap.release()
    
    return frames


def resize_frame(frame, size=FRAME_SIZE):
    """
    Resize a single frame to target size.

    Args:
        frame: numpy array (H, W, C).
        size: (width, height) tuple. Default (128, 128).

    Returns:
        Resized frame as numpy array.
    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def normalize_frames(frames):
    """
    Normalize pixel values to [0, 1].

    Args:
        frames: numpy array of shape (N, H, W, C) or list of (H, W, C).

    Returns:
        Normalized array, same shape, float32.
    """
    arr = np.array(frames, dtype=np.float32)
    return arr / 255.0


def frames_to_sequence(frames, seq_len=NUM_FRAMES_PER_SEQUENCE):
    """
    Convert list of frames into overlapping sequences of length seq_len.

    Args:
        frames: List or array of shape (N, H, W, C).
        seq_len: Number of frames per sequence.

    Returns:
        Array of shape (num_sequences, seq_len, H, W, C).
    """
    frames = np.array(frames)
    if len(frames) < seq_len:
        # Pad by repeating last frame
        pad = np.repeat(frames[-1:], seq_len - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    sequences = []
    for i in range(0, len(frames) - seq_len + 1):
        sequences.append(frames[i : i + seq_len])
    if not sequences:
        sequences = [frames[:seq_len]]
    return np.array(sequences, dtype=np.float32)
