from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from semantic_gs import Observation, SourceArchive
from semantic_gs.geometry import projected_basis, source_footprint
from semantic_gs.readout import finalize_render


FIELDS = ("xyz", "rotation", "scales", "rgb", "semantic", "opacity", "ids")


def observation(source_id=0, classes=3, valid=True):
    h = w = 9
    probabilities = np.zeros((h, w, classes), np.float32)
    probabilities[..., source_id % classes] = 1
    return Observation(
        source_id=source_id,
        timestamp_ns=(source_id + 1) * 1_000_000_000,
        rgb=np.full((h, w, 3), 40 + source_id, np.uint8),
        depth=np.full((h, w), 4, np.float32),
        valid=np.full((h, w), valid, bool),
        K=np.array([[8., 0, 4], [0, 10., 4], [0, 0, 1]]),
        probabilities=probabilities,
    )


def archive(count=1, **kwargs):
    bank = SourceArchive(**kwargs)
    for index in range(count):
        bank.add(observation(index))
    return bank


def normals(quaternions):
    return Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]


def test_plane_unprojection_boundary_fallback_and_antipodal_normal():
    result = archive().construct({0: np.eye(4)})
    pixels = result["ids"][:, 1:]
    expected = np.column_stack(((pixels[:, 1] - 4) * .5,
                                (pixels[:, 0] - 4) * .4,
                                np.full(len(pixels), 4)))
    np.testing.assert_allclose(result["xyz"], expected, atol=1e-7)
    np.testing.assert_allclose(normals(result["rotation"]),
                               np.tile([0, 0, -1], (len(pixels), 1)), atol=1e-6)
    assert {tuple(p) for p in pixels} == {(y, x) for y in (0, 4, 8) for x in (0, 4, 8)}
    assert np.isfinite(result["rotation"]).all()


def test_tangent_footprint_preserves_normal_and_two_pixel_covariance():
    xyz = np.array([[0., 0, 5.], [.7, -.4, 7.], [-.2, .1, 4.]])
    q = Rotation.from_euler("xy", [[0., 0], [.6, .4], [1.1, .2]]).as_quat()[:, [3, 0, 1, 2]]
    K = np.array([[310., 0, 67.], [0, 270., 49.], [0, 0, 1.]])
    field = source_footprint(xyz, q, K, np.eye(4))
    basis, _, _ = projected_basis(xyz, field["rotation"], K, np.eye(4))
    footprint = basis * field["scales"][:, None, :]
    np.testing.assert_allclose(footprint @ footprint.transpose(0, 2, 1),
                               np.broadcast_to(4 * np.eye(2), (3, 2, 2)), atol=2e-5)
    np.testing.assert_allclose(normals(field["rotation"]), normals(q), atol=1e-6)
    assert (field["scales"] > 0).all()


def test_all_four_construction_modes_match_after_pose_revision():
    bank = archive(3)
    poses = {i: np.eye(4) for i in range(3)}
    before = bank.construct(poses, source_ids=[1, 2])
    poses[2][:3, :3] = Rotation.from_euler("y", .15).as_matrix()
    poses[2][0, 3] = 2
    reference = bank.construct(poses, source_ids=[1, 2], use_cache=False, selective=False)
    for cached in (False, True):
        for selective in (False, True):
            result = bank.construct(poses, source_ids=[1, 2], use_cache=cached, selective=selective)
            for key in FIELDS:
                np.testing.assert_array_equal(result[key], reference[key])
    assert np.unique(reference["ids"][:, 0]).tolist() == [1, 2]
    assert not np.array_equal(before["xyz"], reference["xyz"])
    expected_normal = poses[2][:3, :3] @ [0, 0, -1]
    np.testing.assert_allclose(normals(reference["rotation"])[reference["ids"][:, 0] == 2],
                               np.tile(expected_normal, (9, 1)), atol=1e-6)


def test_admitted_observations_are_immutable_for_cached_and_uncached_queries():
    bank = SourceArchive()
    frame = observation()
    bank.add(frame)
    expected = bank.construct({0: np.eye(4)})
    for array in (frame.rgb, frame.depth, frame.valid, frame.K, frame.probabilities):
        if array.flags.writeable:
            array[...] = 0
    for cached in (False, True):
        result = bank.construct({0: np.eye(4)}, use_cache=cached)
        for key in FIELDS:
            np.testing.assert_array_equal(result[key], expected[key])


def test_source_registration_order_and_query_pose_contracts():
    bank = archive(2)
    with pytest.raises(ValueError):
        bank.add(observation(1))
    with pytest.raises(ValueError, match="timestamp"):
        bank.add(replace(observation(2), timestamp_ns=1))
    with pytest.raises(ValueError):
        bank.construct({0: np.eye(4), 1: np.eye(4)}, source_ids=[2])
    with pytest.raises(ValueError):
        bank.construct({0: np.eye(4), 1: np.eye(4)}, source_ids=[1, 0])
    with pytest.raises(ValueError):
        bank.construct({0: np.eye(4)}, source_ids=[1])
    selected = bank.construct({1: np.eye(4)}, source_ids=[1], selective=True)
    assert np.unique(selected["ids"][:, 0]).tolist() == [1]
    with pytest.raises(ValueError):
        bank.construct({1: np.eye(4)}, source_ids=[1], selective=False)


def test_raw_depth_admission_precedes_calibration_and_bad_retention_is_explicit():
    frame = observation()
    frame.depth[0, 0] = .9
    bank = SourceArchive(calibration=(0., 0., 0., np.log(2)))
    bank.add(frame)
    result = bank.construct({0: np.eye(4)})
    assert (0, 0, 0) not in {tuple(row) for row in result["ids"]}
    np.testing.assert_allclose(result["xyz"][:, 2], 8., rtol=1e-6)
    invalid = archive(calibration=(0., 0., 0., np.log(10)))
    with pytest.raises(ValueError, match="retained|range"):
        invalid.construct({0: np.eye(4)})


def test_empty_admission_does_not_create_geometry():
    bank = SourceArchive()
    bank.add(observation(valid=False))
    result = bank.construct({0: np.eye(4)})
    for key in FIELDS:
        assert result[key].shape[0] == 0


def render_packet(classes=3):
    alpha = np.array([[.75, .5, 0]], np.float32)
    probability = np.zeros((1, 3, classes), np.float32)
    probability[..., -1] = 1
    return dict(rgb=np.ones((1, 3, 3), np.float32), alpha=alpha,
                depth_sum=4 * alpha, semantic_mass=probability * alpha[..., None])


def test_readout_normalizes_once_and_uses_strict_support_threshold():
    raw = render_packet()
    raw["semantic_mass"][0, 0] = np.array([.2, .3, .5]) * .75
    result = finalize_render(raw)
    np.testing.assert_array_equal(result["support"], [[True, False, False]])
    np.testing.assert_allclose(result["depth"], [[4, 0, 0]])
    np.testing.assert_allclose(result["probabilities"][0, 0], [.2, .3, .5])
    np.testing.assert_array_equal(result["probabilities"][0, 1:], 0)
    np.testing.assert_array_equal(result["classes"], [[2, 3, 3]])
    np.testing.assert_array_equal(raw["depth_sum"], [[3, 2, 0]])


def test_readout_rejects_missing_probability_mass_and_nonfinite_moments():
    raw = render_packet()
    raw["semantic_mass"] *= .5
    with pytest.raises(ValueError, match="mass|opacity"):
        finalize_render(raw)
    raw = render_packet()
    raw["depth_sum"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finalize_render(raw)


def test_unknown_class_does_not_overflow_for_large_class_sets():
    result = finalize_render(render_packet(classes=300))
    np.testing.assert_array_equal(result["classes"], [[299, 300, 300]])


def test_empty_render_remains_unknown_instead_of_free_space():
    raw = dict(rgb=np.ones((2, 3, 3), np.float32), alpha=np.zeros((2, 3), np.float32),
               depth_sum=np.zeros((2, 3), np.float32), semantic_mass=np.zeros((2, 3, 4), np.float32))
    result = finalize_render(raw)
    assert not result["support"].any()
    np.testing.assert_array_equal(result["classes"], 4)
    np.testing.assert_array_equal(result["depth"], 0)
    np.testing.assert_array_equal(result["probabilities"], 0)
