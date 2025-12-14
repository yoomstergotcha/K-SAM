# utils/mediapipe_face.py

import mediapipe as mp
import numpy as np

def extract_landmarks(img):
    h, w = img.shape[:2]
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True
    )

    res = mp_face.process(img)
    if not res.multi_face_landmarks:
        raise RuntimeError("No face detected")

    lm = res.multi_face_landmarks[0]
    return np.array([[p.x*w, p.y*h] for p in lm.landmark])
