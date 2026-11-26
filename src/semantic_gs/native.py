"""Optional planar Gaussian renderer; no CUDA extension is imported at package import.

The external rasterizer and the distributed CUDA patches have separate
research/noncommercial terms in licenses/Gaussian-Splatting.txt.
"""
import hashlib
import importlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


RENDERER_COMMIT = "e0ed0207b3e0669960cfad70852200a4a5847f61"
GLM_COMMIT = "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
PATCHED_SOURCE_SHA256 = {
    "cuda_rasterizer/forward.cu": "6c2dcf1795a9af86aa8e55eba25f4dc708d2ba3261af76f25d697a4821dd33d8",
    "cuda_rasterizer/backward.cu": "0014a7afb6aec9bc3b26ec2c4ad8e8f4f63e0dce2a2f209d735009acdbdeefc0",
}


@lru_cache(maxsize=1)
def verify_native_runtime():
    """Reject an unmarked extension or an extension changed since installation."""
    try:
        module = importlib.import_module("diff_surfel_rasterization")
    except ImportError as error:
        raise RuntimeError("Install the separately licensed renderer with scripts/install_native.sh.") from error
    marker = Path(module.__file__).parent / "offroad_runtime.json"
    if not marker.is_file():
        raise RuntimeError("The renderer lacks the pinned-build marker; use scripts/install_native.sh.")
    recorded = json.loads(marker.read_text())
    if (recorded.get("renderer_commit") != RENDERER_COMMIT
            or recorded.get("glm_commit") != GLM_COMMIT
            or recorded.get("patched_source_SHA256") != PATCHED_SOURCE_SHA256):
        raise RuntimeError("The renderer marker does not describe the required pinned and patched sources.")
    extension = importlib.import_module("diff_surfel_rasterization._C")
    binary_hash = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    if binary_hash != recorded.get("compiled_extension_SHA256"):
        raise RuntimeError("The renderer extension changed after its build marker was written.")
    return module


def _camera_matrices(K, camera_to_world, image_shape, device):
    """Pinhole camera matrices, including the rasterizer's half-pixel convention."""
    import torch

    height, width = image_shape
    w2c = torch.tensor(np.linalg.inv(camera_to_world), dtype=torch.float32, device=device)
    # Retain the validated float32 camera transformation path under pose revisions.
    world = torch.eye(4, dtype=torch.float32, device=device)
    world[:3, :3], world[:3, 3] = w2c[:3, :3], w2c[:3, 3]
    inverse = torch.linalg.inv(world)
    inverse[:3, 3] = (inverse[:3, 3] + torch.zeros(3, device=device)) * 1.0
    view = torch.linalg.inv(inverse).transpose(0, 1)
    fx, fy, cx, cy = (float(K[i, j]) for i, j in [(0, 0), (1, 1), (0, 2), (1, 2)])
    near, far = .01, 100.
    left = near / fx * (((2 * cx - width) / width - 1) * width / 2)
    right = near / fx * (((2 * cx - width) / width + 1) * width / 2)
    top = near / fy * (((2 * cy - height) / height + 1) * height / 2)
    bottom = near / fy * (((2 * cy - height) / height - 1) * height / 2)
    projection = torch.zeros((4, 4), dtype=torch.float32)
    projection[0, 0], projection[1, 1] = 2 * near / (right - left), 2 * near / (top - bottom)
    projection[0, 2], projection[1, 2] = (right + left) / (right - left), (top + bottom) / (top - bottom)
    projection[3, 2], projection[2, 2] = 1., far / (far - near)
    projection[2, 3] = -(far * near) / (far - near)
    projection = projection.transpose(0, 1).to(device)
    projection[2, 0] += 1 / width
    projection[2, 1] += 1 / height
    full = view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    center = view.inverse()[3, :3]
    return w2c, view, full, center


def _empty_raw(height, width, classes):
    return {
        "rgb": np.ones((height, width, 3), np.float32),
        "alpha": np.zeros((height, width), np.float32),
        "depth_sum": np.zeros((height, width), np.float32),
        "semantic_mass": np.zeros((height, width, classes), np.float32),
    }


def render_native(field, K, camera_to_world, image_shape, device="cuda"):
    """Render a constructed field with the pinned, patched planar CUDA rasterizer.

    Returns NumPy arrays containing weighted depth/semantic mass. Normalize once
    with ``finalize_render``. There is no CPU projection substitute for this call.
    """
    import torch

    height, width = map(int, image_shape)
    K, pose = np.asarray(K), np.asarray(camera_to_world)
    if (height <= 0 or width <= 0 or K.shape != (3, 3) or pose.shape != (4, 4)
            or not np.isfinite(K).all() or not np.isfinite(pose).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0)
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Use a positive image shape, zero-skew pinhole K and a rigid camera-to-world pose.")
    arrays = {key: np.asarray(field[key]) for key in ["xyz", "scales", "rotation", "rgb", "semantic", "opacity"]}
    points = len(arrays["xyz"])
    classes = arrays["semantic"].shape[1] if arrays["semantic"].ndim == 2 else 0
    shapes = {"xyz": (points, 3), "scales": (points, 2), "rotation": (points, 4),
              "rgb": (points, 3), "semantic": (points, classes), "opacity": (points, 1)}
    if classes < 1 or any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k, shape in shapes.items()):
        raise ValueError("Invalid Gaussian field shape or nonfinite attribute.")
    if (np.any(arrays["scales"] <= 0) or np.any(arrays["opacity"] < 0) or np.any(arrays["opacity"] > 1)
            or np.any(arrays["rgb"] < 0) or np.any(arrays["rgb"] > 1)
            or np.any(arrays["semantic"] < 0)
            or not np.allclose(arrays["semantic"].sum(1), 1, atol=1e-5)
            or not np.allclose(np.linalg.norm(arrays["rotation"], axis=1), 1, atol=1e-4)):
        raise ValueError("Use positive scales, RGB/opacity in[0,1], normalized terrain responses and unit rotations.")
    if not points:
        return _empty_raw(height, width, classes)
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Native Gaussian rendering requires CUDA; CPU mode only constructs and exports geometry.")
    runtime = verify_native_runtime()
    with torch.no_grad():
        tensor = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()}
        w2c, view, full, center = _camera_matrices(K, pose, (height, width), device)
        z = (tensor["xyz"] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        keep = (z >= 1) & (z <= 25)
        if not bool(keep.any()):
            return _empty_raw(height, width, classes)
        tensor = {k: v[keep].contiguous() for k, v in tensor.items()}

        def raster(attributes, background):
            settings = runtime.GaussianRasterizationSettings(
                height, width, width / (2 * float(K[0, 0])), height / (2 * float(K[1, 1])),
                background, 1., view, full, 0, center, False, False)
            image, _, aux = runtime.GaussianRasterizer(settings)(
                means3D=tensor["xyz"], means2D=torch.zeros_like(tensor["xyz"]),
                colors_precomp=attributes.contiguous(), opacities=tensor["opacity"],
                scales=tensor["scales"], rotations=tensor["rotation"])
            return image, aux

        first, parts = None, []
        for start in range(0, classes, 3):
            part = tensor["semantic"][:, start:start + 3]
            part = torch.nn.functional.pad(part, (0, 3 - part.shape[1]))
            image, aux = raster(part, torch.zeros(3, device=device))
            if first is None:
                first = aux
            elif not torch.allclose(first[1], aux[1], atol=1e-6, rtol=0):
                raise RuntimeError("Visibility differs between semantic chunks.")
            parts.append(image)
        semantic = torch.cat(parts)[:classes].permute(1, 2, 0)
        rgb, rgb_aux = raster(tensor["rgb"], torch.ones(3, device=device))
        if (not torch.allclose(first[1], rgb_aux[1], atol=1e-6, rtol=0)
                or not torch.allclose(semantic.sum(-1), first[1], atol=1e-5, rtol=1e-5)):
            raise RuntimeError("RGB and semantic visibility/mass conservation disagree.")
        raw = {"rgb": rgb.permute(1, 2, 0), "alpha": first[1], "depth_sum": first[0], "semantic_mass": semantic}
        return {k: v.detach().cpu().numpy() for k, v in raw.items()}
