import torch
import numpy as np
from facenet_pytorch import MTCNN
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
    image_size=128,
    margin=20,
    device=device
)


def extract_faces_from_frames(frames, resize_to=(128,128)):

    result = []

    for frame in frames:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = mtcnn(rgb)

        # ---------- if face detected ----------
        if face is not None:

            face = face.permute(1,2,0).cpu().numpy()

            # convert from [-1,1] → [0,255]
            face = ((face + 1) / 2 * 255).astype(np.uint8)

            face = cv2.resize(face, resize_to)

            result.append(face)

        # ---------- fallback: use full frame ----------
        else:

            fallback = cv2.resize(frame, resize_to)

            result.append(fallback)

    return result