import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2

NUM_FRAMES_PER_SEQUENCE = 20
FRAME_SIZE = (128, 128)


def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class DeepfakeDataset(Dataset):

    def __init__(self, dataset_root, augment=True):
        self.samples = []
        self.augment = augment

        real_dir = os.path.join(dataset_root, "real")
        fake_dir = os.path.join(dataset_root, "fake")

        # -------- REAL --------
        for folder in os.listdir(real_dir):
            path = os.path.join(real_dir, folder)
            if not os.path.isdir(path):
                continue
            images = [f for f in os.listdir(path) if f.endswith(".jpg")]
            if len(images) > 0:
                self.samples.append((path, 0))

        # -------- FAKE --------
        for folder in os.listdir(fake_dir):
            path = os.path.join(fake_dir, folder)
            if not os.path.isdir(path):
                continue
            images = [f for f in os.listdir(path) if f.endswith(".jpg")]
            if len(images) > 0:
                self.samples.append((path, 1))

        print(f"Total samples: {len(self.samples)}")
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"  Real: {real_count}")
        print(f"  Fake: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def load_frames(self, folder):
        images = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
        frames = []

        # Load all frames
        for img in images:
            path = os.path.join(folder, img)
            frame = cv2.imread(path)

            if frame is None:
                continue

            frame = cv2.resize(frame, FRAME_SIZE)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        SEQ_LEN = NUM_FRAMES_PER_SEQUENCE

        # Edge case: no frames
        if len(frames) == 0:
            return np.zeros((SEQ_LEN, FRAME_SIZE[1], FRAME_SIZE[0], 3), 
                           dtype=np.float32)

        # Sample frames across the video
        if len(frames) >= SEQ_LEN:
            # Uniform sampling
            indices = np.linspace(0, len(frames)-1, SEQ_LEN).astype(int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with random frames (with replacement)
            indices = np.random.choice(len(frames), SEQ_LEN, replace=True)
            frames = [frames[i] for i in indices]

        return np.array(frames)

    def __getitem__(self, index):
        folder, label = self.samples[index]
        faces = self.load_frames(folder)

        # -------- Strong Augmentation --------
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                faces = np.flip(faces, axis=2).copy()
            
            # Random brightness adjustment
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.7, 1.3)
                faces = np.clip(faces * brightness_factor, 0, 1)
            
            # Random contrast adjustment
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.7, 1.3)
                mean = np.mean(faces, axis=(1, 2, 3), keepdims=True)
                faces = np.clip((faces - mean) * contrast_factor + mean, 0, 1)
            
            # Random saturation adjustment (convert to HSV)
            if random.random() > 0.5:
                # Convert to HSV (simplified)
                hsv_faces = []
                for i in range(faces.shape[0]):
                    hsv = cv2.cvtColor((faces[i] * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
                    hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.7, 1.3), 0, 255)
                    hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.7, 1.3), 0, 255)
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) / 255.0
                    hsv_faces.append(rgb)
                faces = np.array(hsv_faces)
            
            # Gaussian blur
            if random.random() > 0.7:
                ksize = random.choice([3, 5])
                blurred = []
                for i in range(faces.shape[0]):
                    blurred.append(cv2.GaussianBlur(faces[i], (ksize, ksize), 0))
                faces = np.array(blurred)
            
            # Add small random noise
            if random.random() > 0.7:
                noise = np.random.normal(0, 0.02, faces.shape)
                faces = np.clip(faces + noise, 0, 1)

        # (T, H, W, C) -> (T, C, H, W)
        faces = np.transpose(faces, (0, 3, 1, 2))

        return torch.tensor(faces, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)