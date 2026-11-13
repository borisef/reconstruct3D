"""Tests for reconstruction_utils.py — the three solvers and 6-DOF pose."""
import numpy as np
import pytest

import reconstruction_utils as ru
import aircraft_io as aio


# ── GT data fixture ────────────────────────────────────────────────────────────

def _gt():
    """Synthetic GT matching generate_input.build_ground_truth()."""
    center = np.array([20.0, -15.0, 2000.0])
    body_dir = ru._normalize([0.3, 0.1, 1.0])
    wing_dir = ru._normalize(np.cross([0.0, 1.0, 0.0], body_dir))
    P1 = center - body_dir * 15.0
    P2 = center + body_dir * 15.0
    P3 = center - wing_dir * 20.0
    P4 = center + wing_dir * 20.0
    V = body_dir
    D = float(np.linalg.norm(0.5 * (P1 + P2)))
    cam = {"W_pixels": 1920, "H_pixels": 1080, "fov_X_deg": 5.0,
           "fov_Y_deg": float(np.degrees(
               2.0 * np.arctan(np.tan(np.radians(5.0) / 2.0) * 1080 / 1920)))}
    fx, fy, cx, cy = aio.camera_intrinsics(cam)
    u1, v1 = aio.pixel_to_normalized(aio.project_pixel(P1, fx, fy, cx, cy),
                                      fx, fy, cx, cy)
    u2, v2 = aio.pixel_to_normalized(aio.project_pixel(P2, fx, fy, cx, cy),
                                      fx, fy, cx, cy)
    u3, v3 = aio.pixel_to_normalized(aio.project_pixel(P3, fx, fy, cx, cy),
                                      fx, fy, cx, cy)
    u4, v4 = aio.pixel_to_normalized(aio.project_pixel(P4, fx, fy, cx, cy),
                                      fx, fy, cx, cy)
    return dict(P1=P1, P2=P2, P3=P3, P4=P4, V=V, D=D,
                u1=u1, v1=v1, u2=u2, v2=v2,
                u3=u3, v3=v3, u4=u4, v4=v4)


# ── camera helpers ─────────────────────────────────────────────────────────────

def test_focal_length_from_fov_90deg():
    f = ru.focal_length_from_fov(100.0, np.pi / 2)
    assert f == pytest.approx(50.0)

def test_focal_length_from_fov_formula():
    W, fov = 200.0, np.radians(60.0)
    expected = W / (2.0 * np.tan(fov / 2.0))
    assert ru.focal_length_from_fov(W, fov) == pytest.approx(expected)

def test_normalize_coords_centre():
    n = ru.normalize_coords(50.0, 50.0, 50.0, 50.0, 100.0)
    assert np.allclose(n, [0.0, 0.0])

def test_normalize_coords_offset():
    n = ru.normalize_coords(60.0, 50.0, 50.0, 50.0, 100.0)
    assert np.isclose(n[0], 0.1)
    assert np.isclose(n[1], 0.0)

# ── body_solution1 ─────────────────────────────────────────────────────────────

def test_body_solution1_recovers_P1():
    g = _gt()
    P1_est, _ = ru.body_solution1(g["u1"], g["v1"], g["u2"], g["v2"],
                                   g["V"], g["D"])
    assert np.allclose(P1_est, g["P1"], atol=1e-5)

def test_body_solution1_recovers_P2():
    g = _gt()
    _, P2_est = ru.body_solution1(g["u1"], g["v1"], g["u2"], g["v2"],
                                   g["V"], g["D"])
    assert np.allclose(P2_est, g["P2"], atol=1e-5)

def test_body_solution1_midpoint_distance():
    g = _gt()
    P1_est, P2_est = ru.body_solution1(g["u1"], g["v1"], g["u2"], g["v2"],
                                        g["V"], g["D"])
    midpoint = 0.5 * (P1_est + P2_est)
    assert np.isclose(np.linalg.norm(midpoint), g["D"], atol=1e-5)

# ── body_solution2 ─────────────────────────────────────────────────────────────

def test_body_solution2_recovers_P1():
    g = _gt()
    P1_est, _ = ru.body_solution2(g["u1"], g["v1"], g["u2"], g["v2"],
                                   g["V"], g["D"])
    assert np.allclose(P1_est, g["P1"], atol=1e-4)

def test_body_solution2_recovers_P2():
    g = _gt()
    _, P2_est = ru.body_solution2(g["u1"], g["v1"], g["u2"], g["v2"],
                                   g["V"], g["D"])
    assert np.allclose(P2_est, g["P2"], atol=1e-4)

def test_body_solution2_midpoint_distance():
    g = _gt()
    P1_est, P2_est = ru.body_solution2(g["u1"], g["v1"], g["u2"], g["v2"],
                                        g["V"], g["D"])
    midpoint = 0.5 * (P1_est + P2_est)
    assert np.isclose(np.linalg.norm(midpoint), g["D"], atol=1e-4)

# ── wings_solution ─────────────────────────────────────────────────────────────

def test_wings_solution_recovers_P3():
    g = _gt()
    P3_est, _ = ru.wings_solution(g["u3"], g["v3"], g["u4"], g["v4"],
                                   g["V"], g["D"])
    assert np.allclose(P3_est, g["P3"], atol=1e-5)

def test_wings_solution_recovers_P4():
    g = _gt()
    _, P4_est = ru.wings_solution(g["u3"], g["v3"], g["u4"], g["v4"],
                                   g["V"], g["D"])
    assert np.allclose(P4_est, g["P4"], atol=1e-5)

def test_wings_solution_midpoint_distance():
    g = _gt()
    P3_est, P4_est = ru.wings_solution(g["u3"], g["v3"], g["u4"], g["v4"],
                                        g["V"], g["D"])
    midpoint = 0.5 * (P3_est + P4_est)
    assert np.isclose(np.linalg.norm(midpoint), g["D"], atol=1e-5)

# ── compute_6dof ───────────────────────────────────────────────────────────────

def test_compute_6dof_translation_is_body_centre():
    g = _gt()
    result = ru.compute_6dof(g["P1"], g["P2"], g["P3"], g["P4"])
    expected = 0.5 * (g["P1"] + g["P2"])
    assert np.allclose(result["translation"], expected, atol=1e-10)

def test_compute_6dof_R_orthonormal():
    g = _gt()
    R = ru.compute_6dof(g["P1"], g["P2"], g["P3"], g["P4"])["R"]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

def test_compute_6dof_R_right_handed():
    g = _gt()
    R = ru.compute_6dof(g["P1"], g["P2"], g["P3"], g["P4"])["R"]
    assert np.isclose(np.linalg.det(R), 1.0)

def test_compute_6dof_Y_column_is_body_direction():
    g = _gt()
    R = ru.compute_6dof(g["P1"], g["P2"], g["P3"], g["P4"])["R"]
    expected = g["V"]  # body_dir == velocity unit vector
    assert np.allclose(np.abs(R[:, 1]), np.abs(expected), atol=1e-8)
