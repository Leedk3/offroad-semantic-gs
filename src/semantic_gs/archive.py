"""Immutable source observations and query-time Gaussian construction."""

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import correct_depth, prepare_camera_geometry, source_footprint, transform_camera_geometry


@dataclass(frozen=True)
class Observation:
    """One arrived, rectified observation; probabilities share the RGB grid."""

    source_id: int
    timestamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    probabilities: np.ndarray


class SourceArchive:
    """Preserve source order while caching only immutable camera geometry."""

    def __init__(self, calibration=(0.0, 0.0, 0.0, 0.0), stride=4, device="cpu"):
        if stride != 4:
            raise ValueError("The source-footprint convention requires stride=4.")
        if np.asarray(calibration).shape != (4,) or not np.isfinite(calibration).all():
            raise ValueError("Depth calibration must contain four finite coefficients.")
        self._calibration = tuple(float(x) for x in calibration)
        self.device = torch.device(device)
        self._observations = {}
        self._cache = {}
        self._classes = None
        self.range_retention = {}

    @property
    def source_ids(self):
        return tuple(self._observations)

    @property
    def calibration(self):
        return self._calibration

    def add(self, observation):
        """Copy and validate a newly arrived source; existing sources cannot change."""
        o = observation
        if isinstance(o.source_id, bool) or not isinstance(o.source_id, (int, np.integer)) or o.source_id < 0:
            raise ValueError("source_id must be a nonnegative integer.")
        if o.source_id in self._observations:
            raise ValueError("source_id is already registered.")
        if isinstance(o.timestamp_ns, bool) or not isinstance(o.timestamp_ns, (int, np.integer)):
            raise ValueError("timestamp_ns must be an integer.")
        if self._observations and o.timestamp_ns <= next(reversed(self._observations.values())).timestamp_ns:
            raise ValueError("Observations must arrive in strictly increasing timestamp order.")
        depth = np.asarray(o.depth)
        valid, K, rgb, probabilities = (np.asarray(x) for x in [o.valid, o.K, o.rgb, o.probabilities])
        if depth.ndim != 2 or min(depth.shape) < 2 or valid.shape != depth.shape or valid.dtype != np.bool_:
            raise ValueError("depth and boolean valid must share an H x W grid of at least 2 x 2.")
        if not np.issubdtype(depth.dtype, np.floating) or not np.isfinite(depth[valid]).all() or (depth[valid] <= 0).any():
            raise ValueError("Valid axial depths must be positive finite floating-point values.")
        if rgb.shape != depth.shape + (3,) or rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 H x W x 3 on the depth grid.")
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("K must be a finite pinhole camera matrix with positive focal lengths.")
        if not np.issubdtype(K.dtype, np.floating) or abs(np.linalg.det(K)) < 1e-12:
            raise ValueError("K must be floating-point and invertible.")
        if probabilities.ndim != 3 or probabilities.shape[:2] != depth.shape or probabilities.shape[2] < 1:
            raise ValueError("probabilities must be aligned H x W x C with C >= 1.")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Terrain probabilities must be finite and nonnegative.")
        admitted = valid & (depth >= 1) & (depth <= 25)
        if not np.allclose(probabilities.sum(-1)[admitted], 1, atol=1e-5, rtol=1e-5):
            raise ValueError("Terrain probabilities must sum to one at admitted source pixels.")
        if self._classes is not None and probabilities.shape[2] != self._classes:
            raise ValueError("All sources must use the same terrain class count.")
        arrays = [np.array(x, copy=True) for x in [rgb, depth, valid, K, probabilities]]
        for array in arrays:
            array.setflags(write=False)
        self._observations[int(o.source_id)] = Observation(int(o.source_id), int(o.timestamp_ns), *arrays)
        self._classes = probabilities.shape[2]

    def _prepare(self, observation):
        depth = torch.tensor(observation.depth, dtype=torch.float32, device=self.device)
        valid = torch.tensor(observation.valid, device=self.device)
        K = torch.tensor(observation.K, device=self.device)
        h, w = depth.shape
        y, x = np.mgrid[0:h:4, 0:w:4]
        keep = observation.valid[y, x] & (observation.depth[y, x] >= 1) & (observation.depth[y, x] <= 25)
        pixels = torch.tensor(np.c_[y[keep], x[keep]], device=self.device)
        prepared = prepare_camera_geometry(correct_depth(depth, valid, self.calibration), valid, K, pixels)
        self.range_retention[observation.source_id] = prepared["retention"]
        if prepared["retention"] < 0.95:
            raise ValueError("Depth correction retained fewer than 95% of fixed source pixels in [1,25].")
        return prepared, pixels.cpu().numpy()

    def construct(self, poses, source_ids=None, use_cache=True, selective=True):
        """Build an ordered source subsequence under current camera-to-map poses.

        Eager mode builds the whole arrived archive before selecting the same
        sources. Selective mode requires poses only for the requested sources.
        """
        requested = list(self.source_ids if source_ids is None else source_ids)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in requested):
            raise ValueError("Requested source IDs must be integers.")
        if len(requested) != len(set(requested)) or any(i not in self._observations for i in requested):
            raise ValueError("Requested source IDs must be unique and already registered.")
        if requested != [i for i in self.source_ids if i in requested]:
            raise ValueError("Requested sources must preserve archive order.")
        built_ids = requested if selective else list(self.source_ids)
        groups = []
        for source_id in built_ids:
            if source_id not in poses:
                raise ValueError(f"Missing current pose for source {source_id}.")
            pose = np.asarray(poses[source_id])
            if pose.shape != (4, 4) or not np.issubdtype(pose.dtype, np.floating) or not np.isfinite(pose).all():
                raise ValueError("Poses must be finite floating-point 4 x 4 camera-to-map matrices.")
            if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5):
                raise ValueError("Each camera-to-map pose must be a rigid transform.")
            observation = self._observations[source_id]
            if use_cache and source_id in self._cache:
                prepared, pixels = self._cache[source_id]
            else:
                prepared, pixels = self._prepare(observation)
                if use_cache:
                    self._cache[source_id] = prepared, pixels
            field = transform_camera_geometry(prepared, torch.tensor(pose, device=self.device))
            xyz, rotation = (field[k].detach().cpu().numpy() for k in ["xyz", "rotation"])
            footprint = source_footprint(xyz, rotation, observation.K, pose)
            if source_id in requested:
                y, x = pixels.T
                groups.append(dict(xyz=xyz, **footprint, rgb=observation.rgb[y, x].astype(np.float32) / 255,
                                   semantic=observation.probabilities[y, x].astype(np.float32),
                                   opacity=np.full((len(pixels), 1), 0.7, np.float32),
                                   ids=np.c_[np.full(len(pixels), source_id, dtype=np.int64), pixels]))
        shapes = dict(xyz=3, rotation=4, scales=2, rgb=3, semantic=self._classes or 0, opacity=1, ids=3)
        return {key: np.concatenate([g[key] for g in groups]) if groups else np.empty((0, size), np.int64 if key == "ids" else np.float32)
                for key, size in shapes.items()}
