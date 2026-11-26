"""Normalize rendered moments and retain explicit prediction support."""

import numpy as np


def finalize_render(raw):
    """Convert premultiplied render moments into depth/terrain/support arrays.

    The renderer must supply rgb[H,W,3], alpha[H,W], depth_sum[H,W], and
    semantic_mass[H,W,C]. Unknown class index is C; support is availability.
    """
    required = {"rgb", "alpha", "depth_sum", "semantic_mass"}
    if not required.issubset(raw):
        raise ValueError("Render packet is missing RGB, opacity, depth mass or terrain mass.")
    rgb, alpha, depth_sum, mass = (np.asarray(raw[k]) for k in ["rgb", "alpha", "depth_sum", "semantic_mass"])
    if alpha.ndim != 2 or depth_sum.shape != alpha.shape or rgb.shape != alpha.shape + (3,):
        raise ValueError("RGB, depth mass and alpha must share one H x W grid.")
    if mass.ndim != 3 or mass.shape[:2] != alpha.shape or mass.shape[2] < 1:
        raise ValueError("Terrain mass must be H x W x C with C >= 1.")
    if not all(np.isfinite(a).all() for a in [rgb, alpha, depth_sum, mass]):
        raise ValueError("Render values must be finite.")
    if (alpha < 0).any() or (alpha > 1 + 1e-5).any() or (mass < 0).any() or (depth_sum < 0).any():
        raise ValueError("Opacity and rendered masses must be nonnegative; opacity cannot exceed one.")
    if not np.allclose(mass.sum(-1), alpha, atol=1e-5, rtol=1e-5):
        raise ValueError("Terrain mass must sum to opacity; missing probability mass is not normalized away.")
    divisor = np.maximum(alpha, 1e-8)
    depth = depth_sum / divisor
    support = (alpha > 0.5) & np.isfinite(depth) & (depth > 0)
    probabilities = mass / divisor[..., None]
    classes = mass.argmax(-1).astype(np.int32)
    classes[~support] = mass.shape[2]
    return dict(rgb=rgb.copy(), alpha=alpha.copy(), depth=np.where(support, depth, 0),
                semantic_mass=mass.copy(), probabilities=np.where(support[..., None], probabilities, 0),
                classes=classes, support=support)
