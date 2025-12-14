# utils/clip.py

import numpy as np

def clip_landmarks(pts, image_shape):
    h, w = image_shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts
