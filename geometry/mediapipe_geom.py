# geometry/mediapipe_geom.py
import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp

import mediapipe as mp

mp_face = mp.solutions.face_mesh

# FaceMesh indices (MediaPipe)
# - Face oval indices (contour-ish)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

# Eyes: a few stable points
LEFT_EYE = [33, 133, 159, 145]   # outer, inner, upper, lower
RIGHT_EYE = [362, 263, 386, 374]

# Cheek-ish anchors (mid-face). These are approximate but stable.
LEFT_CHEEK  = [50, 101, 118]     # near left cheek region
RIGHT_CHEEK = [280, 330, 347]    # near right cheek region

def _to_np_landmarks(face_landmarks, w, h):
    pts = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)
    return pts  # (468, 2)

def extract_mesh_468(img_bgr, static_image_mode=True):
    """
    Returns: pts (468,2) in pixel coords, or None if not detected.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face.FaceMesh(
        static_image_mode=static_image_mode,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as fm:
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        pts = _to_np_landmarks(res.multi_face_landmarks[0], w, h)
        return pts

def age_strength(age_src, age_tgt, max_age=80.0):
    """
    Returns a signed scalar in [-1,1]:
      negative -> younger (regression)
      positive -> older (progression)
    """
    delta = (age_tgt - age_src) / max_age
    return float(np.clip(delta, -1.0, 1.0))

def build_control_points(src_pts, age_src, age_tgt):
    """
    Make a sparse set of control points (src -> dst) for warping.
    Uses:
      - FACE_OVAL: face contour + jaw width
      - cheeks: sagging
      - eyes: opening/closing
    """
    s = age_strength(age_src, age_tgt)  # signed
    # You can tune these multipliers:
    jaw_widen   = 0.06 * s       # widen when older, narrow when younger
    cheek_sag   = 0.05 * max(s, 0.0)   # only sag when older
    eye_close   = 0.05 * max(s, 0.0)   # smaller eye opening when older
    eye_open    = 0.06 * max(-s, 0.0)  # larger eyes when younger

    src = []
    dst = []

    # --- 1) Face oval / jaw width ---
    oval = src_pts[FACE_OVAL].copy()
    cx = np.mean(oval[:, 0])

    for p in oval:
        x, y = p
        # move horizontally away/toward center for jaw widening/narrowing
        dx = (x - cx) * jaw_widen
        # slight vertical contour effect for older: lower jawline a bit
        dy = 0.02 * max(s, 0.0) * (y - np.mean(oval[:, 1]))
        src.append([x, y])
        dst.append([x + dx, y + dy])

    # --- 2) Cheek sag (older only) ---
    for idx in (LEFT_CHEEK + RIGHT_CHEEK):
        x, y = src_pts[idx]
        # sag down
        src.append([x, y])
        dst.append([x, y + cheek_sag * 128.0])  # assuming ~128-256px image scale

    # --- 3) Eye opening/closing ---
    def _eye_adjust(eye_idx, open_amt, close_amt):
        # use upper/lower to change aperture
        outer, inner, up, low = eye_idx
        # upper goes up when younger, down when older
        for idx, sign in [(up, -1.0), (low, +1.0)]:
            x, y = src_pts[idx]
            delta = (open_amt - close_amt) * 128.0
            src.append([x, y])
            dst.append([x, y + sign * delta])

        # keep corners stable (helps identity)
        for idx in [outer, inner]:
            x, y = src_pts[idx]
            src.append([x, y])
            dst.append([x, y])

    _eye_adjust(LEFT_EYE,  eye_open, eye_close)
    _eye_adjust(RIGHT_EYE, eye_open, eye_close)

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    return src, dst

def piecewise_warp(img_bgr, src_ctrl, dst_ctrl):
    """
    Piecewise affine warp using skimage.
    """
    h, w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_ctrl, src_ctrl)  # map output coords -> input coords

    warped = warp(img, tform, output_shape=(h, w), preserve_range=True)
    warped = warped.astype(np.uint8)
    warped_bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    return warped_bgr

def apply_age_geometry(img_bgr, age_src, age_tgt, upscale=256):
    """
    Main API:
      - upscales for robust mesh detection
      - extracts mesh
      - builds control points
      - warps
    """
    h0, w0 = img_bgr.shape[:2]

    if upscale is not None and max(h0, w0) < upscale:
        scale = upscale / max(h0, w0)
        img_up = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_CUBIC)
    else:
        img_up = img_bgr

    pts = extract_mesh_468(img_up, static_image_mode=True)
    if pts is None:
        return img_bgr  # fail-safe

    src_ctrl, dst_ctrl = build_control_points(pts, float(age_src), float(age_tgt))
    warped_up = piecewise_warp(img_up, src_ctrl, dst_ctrl)

    warped = cv2.resize(warped_up, (w0, h0), interpolation=cv2.INTER_AREA)
    return warped
