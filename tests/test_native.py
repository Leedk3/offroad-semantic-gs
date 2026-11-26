"""Optional analytic CUDA fixture; no recorded data or model inference."""
import numpy as np
import pytest


@pytest.mark.native
def test_one_plane_matches_analytic_footprint():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    pytest.importorskip("diff_surfel_rasterization", exc_type=ImportError)
    from semantic_gs.geometry import source_footprint
    from semantic_gs.native import render_native
    from semantic_gs import finalize_render

    K = np.array([[300., 0, 31.], [0, 260., 23.], [0, 0, 1.]])
    xyz = np.array([[0., 0., 5.]], np.float32)
    rotation = np.array([[0., 1., 0., 0.]], np.float32)
    field = dict(xyz=xyz, **source_footprint(xyz, rotation, K, np.eye(4)),
                 rgb=np.array([[.2, .5, .1]], np.float32),
                 semantic=np.array([[.8, .1, .1]], np.float32),
                 opacity=np.full((1, 1), .7, np.float32))
    raw = render_native(field, K, np.eye(4), (48, 64))
    y, x = np.mgrid[:48, :64]
    expected = .7 * np.exp(-((x - 31) ** 2 + (y - 23) ** 2) / 8)
    expected[expected < 1 / 255] = 0
    np.testing.assert_allclose(raw["alpha"], expected, atol=1e-5, rtol=0)
    prediction = finalize_render(raw)
    np.testing.assert_allclose(prediction["depth"][prediction["support"]], 5, atol=1e-4, rtol=0)
