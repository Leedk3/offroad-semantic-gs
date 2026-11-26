import json

import numpy as np
import pytest

from semantic_gs.cli import build_map, load_manifest, make_demo
from semantic_gs.native import _camera_matrices, render_native
from semantic_gs import finalize_render


def test_procedural_demo_and_manifest_roundtrip(tmp_path):
    summary = make_demo(tmp_path)
    assert summary["four_construction_modes_identical"]
    assert not summary["native_renderer_executed"]
    assert summary["Gaussians"] > 0
    field, packet = build_map(tmp_path / "manifest.json", tmp_path / "eager.npz", eager=True, no_cache=True)
    with np.load(tmp_path / "map.npz", allow_pickle=False) as reference:
        for key in field:
            np.testing.assert_array_equal(field[key], reference[key])
    assert set(field["ids"][:, 0]) == {1, 2}
    assert packet["class_names"].tolist() == ["soil", "vegetation", "rock"]


def test_future_source_rejected_by_query_time(tmp_path):
    make_demo(tmp_path)
    manifest = tmp_path / "manifest.json"
    config = json.loads(manifest.read_text())
    config["query"]["timestamp_ns"] = 2_000_000_000
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="future"):
        load_manifest(manifest)


def test_invalid_window_rejected(tmp_path):
    make_demo(tmp_path)
    manifest = tmp_path / "manifest.json"
    config = json.loads(manifest.read_text())
    config["window"] = 0
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="window"):
        build_map(manifest, tmp_path / "invalid.npz")


def test_native_projection_respects_integer_pixel_center():
    import torch

    K = np.array([[300., 0, 31.], [0, 260., 23.], [0, 0, 1.]])
    _, _, full, _ = _camera_matrices(K, np.eye(4), (48, 64), "cpu")
    point = torch.tensor([0., 0., 5., 1.])
    clip = point @ full
    ndc = clip[:2] / clip[3]
    pixels = ((ndc + 1) * torch.tensor([64, 48]) - 1) / 2
    np.testing.assert_allclose(pixels.numpy(), [31, 23], atol=1e-6)


def test_empty_native_field_needs_no_cuda_extension():
    field = {key: np.empty((0, size), np.float32) for key, size in
             {"xyz": 3, "scales": 2, "rotation": 4, "rgb": 3, "semantic": 3, "opacity": 1}.items()}
    prediction = finalize_render(render_native(field, np.eye(3), np.eye(4), (4, 5), device="cpu"))
    assert not prediction["support"].any()
    assert (prediction["classes"] == 3).all()


@pytest.mark.parametrize("scale", [0.0, 2.0, -1.0])
def test_native_rejects_nonrigid_query_before_cuda(scale):
    pose = np.eye(4)
    pose[0, 0] = scale
    with pytest.raises(ValueError, match="rigid"):
        render_native({}, np.eye(3), pose, (4, 5), device="cpu")
