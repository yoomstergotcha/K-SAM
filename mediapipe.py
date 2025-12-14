import cv2
import numpy as np

from utils.mediapipe_face import extract_landmarks
from utils.clip import clip_landmarks
from geometry.aging import apply_geometry_aging
from warp.triangulation import delaunay_triangulation
from warp.warp_triangle import warp_triangle

def geometry_age_image(img, age_src, age_tgt):
    pts = extract_landmarks(img)
    pts_aged = apply_geometry_aging(pts, age_src, age_tgt)
    pts_aged = clip_landmarks(pts_aged, img.shape)

    triangles = delaunay_triangulation(pts, img.shape)

    out = img.copy().astype(np.float32)
    for tri in triangles:
        t_src = [pts[i] for i in tri]
        t_dst = [pts_aged[i] for i in tri]
        out = warp_triangle(out, t_src, t_dst)

    return np.clip(out, 0, 255).astype(np.uint8)
