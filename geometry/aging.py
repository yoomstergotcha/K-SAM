# geometry/aging.py

import numpy as np
from .old import apply_old
from .young import apply_young

def apply_geometry_aging(pts, age_src, age_tgt):
    """
    Unified geometry aging function.
    """
    alpha = np.clip((age_tgt - age_src) / 40.0, -1.0, 1.0)

    if abs(alpha) < 0.05:
        return pts.copy()

    if alpha > 0:
        return apply_old(pts, strength=alpha)
    else:
        return apply_young(pts, strength=-alpha)
