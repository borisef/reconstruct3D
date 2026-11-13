"""Tests for the _rotation_about Rodrigues helper in GUI/viewer.py."""
import numpy as np
import pytest

from viewer import _rotation_about


def test_identity_at_zero_angle():
    R = _rotation_about([0, 0, 1], 0.0)
    assert np.allclose(R, np.eye(3))

def test_90_about_z_rotates_x_to_y():
    R = _rotation_about([0, 0, 1], np.pi / 2)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-10)
    assert np.allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-10)

def test_90_about_x_rotates_y_to_z():
    R = _rotation_about([1, 0, 0], np.pi / 2)
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-10)

def test_R_is_orthonormal():
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    R = _rotation_about(axis, 1.234)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

def test_R_is_right_handed():
    axis = np.array([1.0, 2.0, 3.0])
    R = _rotation_about(axis, 2.5)
    assert np.isclose(np.linalg.det(R), 1.0)

def test_axis_vector_is_preserved():
    axis = np.array([0.0, 1.0, 0.0])
    R = _rotation_about(axis, np.pi / 3)
    assert np.allclose(R @ axis, axis, atol=1e-10)

def test_full_360_is_identity():
    axis = np.array([0.0, 0.0, 1.0])
    R = _rotation_about(axis, 2 * np.pi)
    assert np.allclose(R, np.eye(3), atol=1e-10)

def test_accepts_unnormalized_axis():
    R1 = _rotation_about([0, 0, 1], np.pi / 4)
    R2 = _rotation_about([0, 0, 5], np.pi / 4)
    assert np.allclose(R1, R2, atol=1e-10)

def test_180_about_z_flips_x():
    R = _rotation_about([0, 0, 1], np.pi)
    assert np.allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-10)
