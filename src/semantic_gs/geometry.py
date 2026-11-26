"""Depth-guided planar Gaussian geometry and source-image footprints."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F


def correct_depth(depth, valid, coefficients=(0.0, 0.0, 0.0, 0.0)):
    """Apply frozen coefficients (log-depth, normalized x, normalized y, bias)."""
    coefficients = depth.new_tensor(coefficients)
    if coefficients.shape != (4,) or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("Depth calibration must contain four finite coefficients.")
    h, w = depth.shape
    y, x = torch.where(valid & torch.isfinite(depth) & (depth > 0))
    xy = 2 * (torch.stack([x, y], 1).to(depth.dtype) + 0.5) / depth.new_tensor([w, h]) - 1
    features = torch.cat([depth[y, x].log()[:, None], xy], 1)
    residual = F.linear(features, coefficients[:3][None], coefficients[3:])[:, 0]
    result = depth.clone()
    result[y, x] = depth[y, x] * residual.exp()
    return result


def prepare_camera_geometry(depth, valid, K, pixels):
    """Cache camera-local points/normals at fixed (y, x) source pixels.

    Inputs are tensors. NumPy intrinsics inversion, float64 unprojection and
    float32 local points preserve the operation order of the research backend.
    """
    h, w = depth.shape
    y, x = pixels.T
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device),
                            torch.arange(w, device=depth.device), indexing="ij")
    uv1 = torch.stack([xx, yy, torch.ones_like(xx)], -1).double()
    inverse = torch.as_tensor(np.linalg.inv(K.detach().cpu().numpy()), device=depth.device)
    xyz = (uv1 @ inverse.double().T) * depth.double()[..., None]
    good = valid & torch.isfinite(depth) & (depth >= 1) & (depth <= 25)
    stencil = good.clone()
    for axis in [0, 1]:
        stencil = stencil & torch.roll(good, 1, axis) & torch.roll(good, -1, axis)
    stencil = stencil.clone()
    stencil[[0, -1], :] = False
    stencil[:, [0, -1]] = False
    dy, dx = torch.gradient(xyz, dim=(0, 1))
    normal = torch.linalg.cross(dx, dy)[y, x]
    lengths = torch.linalg.vector_norm(normal, dim=1)
    eligible = stencil[y, x] & torch.isfinite(normal).all(1) & (lengths > 0)
    fallback = normal.new_tensor([0, 0, -1]).expand_as(normal)
    normal = torch.where(eligible[:, None], normal / lengths.clamp_min(1e-30)[:, None], fallback)
    normal = torch.where(((normal * xyz[y, x]).sum(1) > 0)[:, None], -normal, normal).float()
    focal = (K[0, 0] + K[1, 1]) / 2
    retention = float(((depth[y, x] >= 1) & (depth[y, x] <= 25)).float().mean()) if len(y) else 1.0
    return dict(local=xyz[y, x].float(), normal=normal,
                radius=(depth[y, x] / focal * 4 / 2).clamp(0.02, 0.2), retention=retention)


def transform_camera_geometry(prepared, pose):
    """Use the current camera-to-map pose; world geometry is never cached."""
    world_normal = (prepared["normal"].to(pose.dtype) @ pose[:3, :3].T).double()
    world_normal = world_normal / torch.linalg.vector_norm(world_normal, dim=1)[:, None]
    q = torch.stack([1 + world_normal[:, 2], -world_normal[:, 1],
                     world_normal[:, 0], torch.zeros_like(world_normal[:, 0])], 1)
    qnorm = torch.linalg.vector_norm(q, dim=1)
    opposite = q.new_tensor([0, 1, 0, 0]).expand_as(q)
    q = torch.where((qnorm < 1e-12)[:, None], opposite, q / qnorm.clamp_min(1e-12)[:, None])
    world = (prepared["local"].to(pose.dtype) @ pose[:3, :3].T + pose[:3, 3]).float()
    return dict(xyz=world, rotation=q.float(), radius=prepared["radius"])


def _matrices(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[:, [1, 2, 3, 0]]).as_matrix()


def projected_basis(xyz, rotation, K, pose):
    """Return projected tangent basis, camera depth and camera-space tangents."""
    inverse = np.linalg.inv(pose)
    points = np.asarray(xyz, dtype=float) @ inverse[:3, :3].T + inverse[:3, 3]
    homogeneous = points @ K.T
    J = (K[None, :2, :] * homogeneous[:, 2, None, None]
         - homogeneous[:, :2, None] * K[None, 2:3, :]) / homogeneous[:, 2, None, None] ** 2
    tangent = inverse[:3, :3][None] @ _matrices(rotation)[:, :, :2]
    return J @ tangent, points[:, 2], tangent


def source_footprint(xyz, rotation, K, pose):
    """Preserve normals and set linearized source-image covariance to 4 I.

    Singular footprints are rejected, rather than clipped or silently removed.
    Quaternions use (w, x, y, z); the two scales lie in the tangent plane.
    """
    if not len(xyz):
        return dict(rotation=np.empty((0, 4), np.float32), scales=np.empty((0, 2), np.float32))
    basis, z, _ = projected_basis(xyz, rotation, K, pose)
    if not np.isfinite(basis).all() or not (z > 0).all():
        raise ValueError("Source footprints require finite projections in front of the camera.")
    _, singular, vt = np.linalg.svd(basis)
    if not (singular > 0).all():
        raise ValueError("Source footprint is rank deficient.")
    V = vt.transpose(0, 2, 1).copy()
    V[:, :, 1] *= np.linalg.det(V)[:, None]
    scales = (2 / singular).astype(np.float32)
    frame = _matrices(rotation).copy()
    frame[:, :, :2] = frame[:, :, :2] @ V
    q = Rotation.from_matrix(frame).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    if not np.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("Source footprint scales must be finite and positive.")
    return dict(rotation=q, scales=scales)
