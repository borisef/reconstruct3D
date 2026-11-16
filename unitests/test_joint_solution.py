"""Tests for reconstruction_utils.joint_solution (paper Part II)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import reconstruction_utils as ru


def _cross(L1, L2, alpha):
    """Four local keypoints of a cross aircraft (tail, nose, left, right)."""
    yw = (alpha - 0.5) * L1
    return np.array([[0.0, -L1 / 2, 0.0], [0.0, L1 / 2, 0.0],
                     [-L2 / 2, yw, 0.0], [L2 / 2, yw, 0.0]])


def _scene(alpha, D=2000.0, L1=30.0, L2=40.0, seed=0):
    """Noise-free projection of a random-pose cross at range D (body centre)."""
    rng = np.random.default_rng(seed)
    R = Rotation.random(random_state=rng).as_matrix()
    c = np.array([0.08, -0.05, 1.0])
    t = D * c / np.linalg.norm(c)              # body centre at range D
    P = _cross(L1, L2, alpha) @ R.T + t
    uv = P[:, :2] / P[:, 2:3]                   # normalized projections
    V = R[:, 1]                                 # body axis = velocity
    return P, uv, V, D


def _solve(uv, V, D):
    return np.array(ru.joint_solution(
        uv[0, 0], uv[0, 1], uv[1, 0], uv[1, 1],
        uv[2, 0], uv[2, 1], uv[3, 0], uv[3, 1], V, D))


# ── noise-free exactness ──────────────────────────────────────────────────────

def test_recovers_points_wings_at_midpoint():
    P, uv, V, D = _scene(0.5)
    assert np.allclose(_solve(uv, V, D), P, atol=1e-6)

def test_recovers_points_wings_toward_nose():
    # alpha = 0.7: the wings are NOT at the body midpoint. The sequential
    # wings solver would misplace them; the joint fit must recover them.
    P, uv, V, D = _scene(0.7)
    assert np.allclose(_solve(uv, V, D), P, atol=1e-6)

def test_recovers_points_wings_toward_tail():
    P, uv, V, D = _scene(0.25, seed=3)
    assert np.allclose(_solve(uv, V, D), P, atol=1e-6)

def test_recovers_across_random_poses():
    for seed in range(8):
        P, uv, V, D = _scene(0.6, seed=seed)
        assert np.allclose(_solve(uv, V, D), P, atol=1e-5), seed

# ── structural properties of the solution ─────────────────────────────────────

def test_body_midpoint_at_range():
    P, uv, V, D = _scene(0.65, seed=1)
    est = _solve(uv, V, D)
    centre = 0.5 * (est[0] + est[1])
    assert np.linalg.norm(centre) == pytest.approx(D, abs=1e-6)

def test_wings_perpendicular_to_body():
    P, uv, V, D = _scene(0.65, seed=2)
    est = _solve(uv, V, D)
    body = est[1] - est[0]
    wings = est[3] - est[2]
    assert abs(np.dot(body, wings)) < 1e-6

def test_body_axis_aligns_with_velocity():
    P, uv, V, D = _scene(0.55, seed=4)
    est = _solve(uv, V, D)
    body = ru._normalize(est[1] - est[0])
    assert np.linalg.norm(np.cross(body, ru._normalize(V))) < 1e-6

# ── weighting ──────────────────────────────────────────────────────────────────

def test_only_weight_ratio_matters():
    # Scaling both sigmas leaves the point estimate unchanged.
    P, uv, V, D = _scene(0.5, seed=5)
    a = _solve(uv, V, D)
    b = np.array(ru.joint_solution(
        uv[0, 0], uv[0, 1], uv[1, 0], uv[1, 1],
        uv[2, 0], uv[2, 1], uv[3, 0], uv[3, 1], V, D,
        sig_uv=10 * ru.JOINT_SIGMA_UV, sig_v=10 * ru.JOINT_SIGMA_V))
    assert np.allclose(a, b, atol=1e-7)

def test_handles_non_unit_velocity():
    # V is used by direction only; a non-unit vector must give the same result.
    P, uv, V, D = _scene(0.5, seed=6)
    a = _solve(uv, V, D)
    b = _solve(uv, 7.3 * V, D)
    assert np.allclose(a, b, atol=1e-7)
