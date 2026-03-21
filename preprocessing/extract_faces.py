import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from utils.video_utils import load_video, FRAME_SIZE
from preprocessing.face_detection import extract_faces_from_frames

DATASET = "dataset"
OUTPUT = "faces_dataset"

for label in ["real", "fake"]:

    src = os.path.join(DATASET, label)
    dst = os.path.join(OUTPUT, label)

    os.makedirs(dst, exist_ok=True)

    for video in os.listdir(src):

        path = os.path.join(src, video)

        frames = load_video(path)

        if len(frames) == 0:
            continue

        faces = extract_faces_from_frames(frames, resize_to=FRAME_SIZE)

        name = os.path.splitext(video)[0]

        folder = os.path.join(dst, name)

        os.makedirs(folder, exist_ok=True)

        for i, face in enumerate(faces):

            face = (face * 255).astype("uint8")

            save_path = os.path.join(folder, f"{i}.jpg")

            cv2.imwrite(save_path, face)