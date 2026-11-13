"""Tests for aircraft_frame.py — local body-fixed coordinate frame."""
import numpy as np
import pytest

import aircraft_io as aio
from aircraft_frame import AircraftFrame, local_keypoints


# ── synthetic GT aircraft (matches generate_input.build_ground_truth) ──────────

def _gt_points():
    """Return (P1_tail, P2_nose, P3_left, P4_right, body_dir, wing_dir)."""
    import reconstruction_utils as ru
    center = np.array([20.0, -15.0, 2000.0])
    body_dir = ru._normalize([0.3, 0.1, 1.0])
    wing_dir = ru._normalize(np.cross([0.0, 1.0, 0.0], body_dir))
    P1 = center - body_dir * 15.0
    P2 = center + body_dir * 15.0
    P3 = center - wing_dir * 20.0
    P4 = center + wing_dir * 20.0
    return P1, P2, P3, P4, body_dir, wing_dir


# ── orthonormality and handedness ──────────────────────────────────────────────

def test_R_orthonormal():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    assert np.allclose(f.R.T @ f.R, np.eye(3), atol=1e-10)

def test_R_right_handed():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    assert np.isclose(np.linalg.det(f.R), 1.0)

def test_origin_is_body_midpoint():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    assert np.allclose(f.origin, 0.5 * (P1 + P2), atol=1e-10)

# ── axis conventions ───────────────────────────────────────────────────────────

def test_local_Y_is_tail_to_nose():
    P1, P2, P3, P4, body_dir, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    assert np.allclose(f.R[:, 1], body_dir, atol=1e-10)

def test_local_Z_is_cross_X_Y():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    X, Y, Z = f.R[:, 0], f.R[:, 1], f.R[:, 2]
    assert np.allclose(np.cross(X, Y), Z, atol=1e-10)

# ── GT keypoints land on local axes ───────────────────────────────────────────

def test_body_points_on_Y_axis():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    p1l = f.to_local(P1)
    p2l = f.to_local(P2)
    assert np.allclose([p1l[0], p1l[2]], [0.0, 0.0], atol=1e-8)
    assert np.allclose([p2l[0], p2l[2]], [0.0, 0.0], atol=1e-8)

def test_wing_points_on_X_axis():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    p3l = f.to_local(P3)
    p4l = f.to_local(P4)
    assert np.allclose([p3l[1], p3l[2]], [0.0, 0.0], atol=1e-8)
    assert np.allclose([p4l[1], p4l[2]], [0.0, 0.0], atol=1e-8)

def test_body_points_symmetric_about_origin():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    p1l = f.to_local(P1)
    p2l = f.to_local(P2)
    assert np.isclose(p1l[1], -p2l[1], atol=1e-8)

def test_wing_points_symmetric_about_origin():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    p3l = f.to_local(P3)
    p4l = f.to_local(P4)
    assert np.isclose(p3l[0], -p4l[0], atol=1e-8)

# ── round-trip transforms ──────────────────────────────────────────────────────

def test_to_world_inverse_of_to_local():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    for P in [P1, P2, P3, P4]:
        assert np.allclose(f.to_world(f.to_local(P)), P, atol=1e-10)

def test_to_local_vectorized():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    pts = np.vstack([P1, P2, P3, P4])
    batch = f.to_local(pts)
    for i, P in enumerate([P1, P2, P3, P4]):
        assert np.allclose(batch[i], f.to_local(P))

def test_direction_roundtrip():
    P1, P2, P3, P4, _, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    d = np.array([0.6, 0.0, 0.8])
    assert np.allclose(f.direction_to_world(f.direction_to_local(d)), d, atol=1e-10)

def test_gt_velocity_maps_to_local_Y():
    P1, P2, P3, P4, body_dir, _ = _gt_points()
    f = AircraftFrame.from_points(P1, P2, P3, P4)
    v_local = f.direction_to_local(body_dir)
    assert np.allclose(v_local, [0, 1, 0], atol=1e-8)

# ── noisy input still gives orthonormal frame ──────────────────────────────────

def test_R_orthonormal_with_noisy_input():
    P1, P2, P3, P4, _, _ = _gt_points()
    rng = np.random.default_rng(7)
    P1n = P1 + rng.normal(0, 0.5, 3)
    P2n = P2 + rng.normal(0, 0.5, 3)
    P3n = P3 + rng.normal(0, 0.5, 3)
    P4n = P4 + rng.normal(0, 0.5, 3)
    f = AircraftFrame.from_points(P1n, P2n, P3n, P4n)
    assert np.allclose(f.R.T @ f.R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(f.R), 1.0)

# ── from_doc ───────────────────────────────────────────────────────────────────

def test_from_doc_matches_from_points():
    P1, P2, P3, P4, _, _ = _gt_points()
    doc = {
        "Coordinates_3D": {
            "P1_tail":       {"GT": aio.vec3(P1)},
            "P2_nose":       {"GT": aio.vec3(P2)},
            "P3_left_wing":  {"GT": aio.vec3(P3)},
            "P4_right_wing": {"GT": aio.vec3(P4)},
        }
    }
    f1 = AircraftFrame.from_doc(doc)
    f2 = AircraftFrame.from_points(P1, P2, P3, P4)
    assert np.allclose(f1.origin, f2.origin, atol=1e-12)
    assert np.allclose(f1.R, f2.R, atol=1e-12)

def test_from_doc_missing_gt_raises():
    P1, P2, P3, P4, _, _ = _gt_points()
    doc = {
        "Coordinates_3D": {
            "P1_tail":       {"GT": None},
            "P2_nose":       {"GT": aio.vec3(P2)},
            "P3_left_wing":  {"GT": aio.vec3(P3)},
            "P4_right_wing": {"GT": aio.vec3(P4)},
        }
    }
    with pytest.raises(ValueError):
        AircraftFrame.from_doc(doc)

# ── local_keypoints helper ─────────────────────────────────────────────────────

def test_local_keypoints_gt_matches_direct():
    P1, P2, P3, P4, _, _ = _gt_points()
    doc = {
        "Coordinates_3D": {
            "P1_tail":       aio.variant_block(GT=aio.vec3(P1)),
            "P2_nose":       aio.variant_block(GT=aio.vec3(P2)),
            "P3_left_wing":  aio.variant_block(GT=aio.vec3(P3)),
            "P4_right_wing": aio.variant_block(GT=aio.vec3(P4)),
        }
    }
    result = local_keypoints(doc, variant="GT")
    frame = AircraftFrame.from_points(P1, P2, P3, P4)
    for name, P in zip(["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"],
                        [P1, P2, P3, P4]):
        assert np.allclose(result[name], frame.to_local(P), atol=1e-10)

def test_local_keypoints_null_variant_returns_none():
    P1, P2, P3, P4, _, _ = _gt_points()
    doc = {
        "Coordinates_3D": {
            "P1_tail":       aio.variant_block(GT=aio.vec3(P1), Computed1=None),
            "P2_nose":       aio.variant_block(GT=aio.vec3(P2), Computed1=None),
            "P3_left_wing":  aio.variant_block(GT=aio.vec3(P3), Computed1=None),
            "P4_right_wing": aio.variant_block(GT=aio.vec3(P4), Computed1=None),
        }
    }
    result = local_keypoints(doc, variant="Computed1")
    assert all(v is None for v in result.values())
