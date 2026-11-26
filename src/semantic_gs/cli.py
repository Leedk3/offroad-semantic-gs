"""Portable prepared-observation interface; no model downloads or inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from .archive import Observation, SourceArchive
from .readout import finalize_render


def _integer_scalar(value, name):
    value = np.asarray(value)
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be an integer scalar.")
    return int(value)


def load_manifest(path):
    path = Path(path)
    config = json.loads(path.read_text())
    observations = []
    for filename in config["sources"]:
        with np.load(path.parent / filename, allow_pickle=False) as packet:
            observations.append(Observation(
                source_id=_integer_scalar(packet["source_id"], "source_id"),
                timestamp_ns=_integer_scalar(packet["timestamp_ns"], "timestamp_ns"),
                **{key: packet[key] for key in ["rgb", "depth", "valid", "K", "probabilities"]},
            ))
    if not observations:
        raise ValueError("A manifest must contain at least one source observation.")
    if len(config["source_poses"]) != len(observations):
        raise ValueError("source_poses must correspond one-to-one with sources.")
    query = config["query"]
    query_time = _integer_scalar(query["timestamp_ns"], "query timestamp_ns")
    if any(o.timestamp_ns > query_time for o in observations):
        raise ValueError("A query cannot use future observations.")
    poses = {o.source_id: np.asarray(pose, dtype=np.float64)
             for o, pose in zip(observations, config["source_poses"])}
    return config, observations, poses


def build_map(manifest, output, eager=False, no_cache=False):
    config, observations, poses = load_manifest(manifest)
    archive = SourceArchive(calibration=config.get("calibration", [0, 0, 0, 0]))
    for observation in observations:
        archive.add(observation)
    window = config.get("window", 5)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    selected = archive.source_ids[-window:]
    field = archive.construct(poses, selected, use_cache=not no_cache, selective=not eager)
    query = config["query"]
    classes = field["semantic"].shape[1]
    names = config.get("class_names", [f"class_{i}" for i in range(classes)])
    if len(names) != classes:
        raise ValueError("class_names must match the terrain probability channel count.")
    payload = dict(field, source_ids=np.asarray(selected, dtype=np.int64),
                   K=np.asarray(query["K"], dtype=np.float64),
                   camera_to_world=np.asarray(query["camera_to_world"], dtype=np.float64),
                   image_shape=np.asarray(query["image_shape"], dtype=np.int64),
                   timestamp_ns=np.asarray(query["timestamp_ns"], dtype=np.int64),
                   class_names=np.asarray(names, dtype=np.str_),
                   calibration=np.asarray(archive.calibration))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return field, payload


def render_map(path, output):
    from .native import render_native

    with np.load(path, allow_pickle=False) as packet:
        data = dict(packet)
    raw = render_native(data, data["K"], data["camera_to_world"], tuple(data["image_shape"]))
    prediction = finalize_render(raw)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **prediction)
    return prediction


def make_demo(directory):
    """Create procedural sources and verify construction; no GS rendering."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    h, w = 48, 64
    y, x = np.mgrid[:h, :w]
    K = np.array([[70.0, 0, 31.0], [0, 65.0, 23.0], [0, 0, 1.0]])
    paths, poses = [], []
    for i in range(3):
        depth = (5 + .005 * x + .003 * y).astype(np.float32)
        valid = np.ones((h, w), bool)
        valid[:4, :4] = False
        depth[~valid] = 0
        rgb = np.stack([50 + 2 * x, 70 + 2 * y, np.full_like(x, 60 + i * 10)], -1).astype(np.uint8)
        probability = np.full((h, w, 3), .1, np.float32)
        probability[..., 0] = .8
        probability[x > w // 2] = [.1, .8, .1]
        pose = np.eye(4)
        pose[0, 3] = .1 * i
        filename = f"source_{i:03d}.npz"
        np.savez_compressed(directory / filename, source_id=i, timestamp_ns=(i + 1) * 1_000_000_000,
                            rgb=rgb, depth=depth, valid=valid, K=K, probabilities=probability)
        paths.append(filename)
        poses.append(pose.tolist())
    config = dict(sources=paths, source_poses=poses, window=2, calibration=[0, 0, 0, 0],
                  class_names=["soil", "vegetation", "rock"],
                  query=dict(K=K.tolist(), camera_to_world=poses[-1], image_shape=[h, w],
                             timestamp_ns=3_000_000_000))
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    reference, _ = build_map(manifest, directory / "map.npz")
    _, observations, pose_dict = load_manifest(manifest)
    # Verify all four construction choices on the same arrived source prefix.
    for cached in [False, True]:
        for selective in [False, True]:
            archive = SourceArchive()
            for observation in observations:
                archive.add(observation)
            field = archive.construct(pose_dict, archive.source_ids[-2:], cached, selective)
            if not all(np.array_equal(reference[key], field[key]) for key in reference):
                raise RuntimeError("Construction choices changed the requested Gaussian fields.")
    alpha = np.array([[.7, .5, 0]], np.float32)
    mass = alpha[..., None] * np.array([.8, .1, .1], np.float32)
    fixture = finalize_render(dict(rgb=np.ones((1, 3, 3), np.float32), alpha=alpha,
                                   depth_sum=alpha * 5, semantic_mass=mass))
    np.savez_compressed(directory / "readout_arithmetic_fixture.npz", **fixture)
    summary = dict(source_observations=3, selected_sources=[1, 2],
                   Gaussians=len(reference["xyz"]), four_construction_modes_identical=True,
                   native_renderer_executed=False,
                   readout_fixture="Analytical moments only; not a render of the exported map.")
    (directory / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic observations and check CPU construction.")
    demo.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build", help="Construct Gaussians from aligned prepared observations.")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--eager", action="store_true", help="Build all arrived sources before selecting.")
    build.add_argument("--no-cache", action="store_true")
    render = commands.add_parser("render", help="Use the separately installed patched CUDA renderer.")
    render.add_argument("--map", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            print(json.dumps(make_demo(args.output), indent=2))
        elif args.command == "build":
            field, _ = build_map(args.manifest, args.output, args.eager, args.no_cache)
            print(json.dumps({"Gaussians": len(field["xyz"]), "native_renderer_executed": False}))
        else:
            prediction = render_map(args.map, args.output)
            print(json.dumps({"supported_pixels": int(prediction["support"].sum())}))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"offroad-gs: {error}\n")


if __name__ == "__main__":
    main()
