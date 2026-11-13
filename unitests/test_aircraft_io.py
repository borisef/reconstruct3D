"""Tests for aircraft_io.py — schema builders, camera model, file I/O."""
import json
import os

import numpy as np
import pytest

import aircraft_io as aio

# ── schema constants ───────────────────────────────────────────────────────────

def test_keypoints_3d_count():
    assert len(aio.KEYPOINTS_3D) == 4

def test_keypoints_2d_count():
    assert len(aio.KEYPOINTS_2D) == 4

def test_variants_count():
    assert len(aio.VARIANTS) == 5

def test_blocks_present():
    for b in ["Camera_parameters", "Coordinates_3D", "Coordinates_2D",
              "Velocity_vector_3D", "Range_meters"]:
        assert b in aio.BLOCKS

# ── dict builders ──────────────────────────────────────────────────────────────

def test_vec3_roundtrip():
    arr = [1.1, -2.2, 3.3]
    d = aio.vec3(arr)
    assert set(d) == {"X", "Y", "Z"}
    assert np.allclose([d["X"], d["Y"], d["Z"]], arr)

def test_vec3_none_passthrough():
    assert aio.vec3(None) is None

def test_vec3_stores_floats():
    d = aio.vec3([1, 2, 3])
    assert all(isinstance(v, float) for v in d.values())

def test_vec2_roundtrip():
    d = aio.vec2([320.5, 240.7])
    assert set(d) == {"u", "v"}
    assert np.isclose(d["u"], 320.5)
    assert np.isclose(d["v"], 240.7)

def test_vec2_none_passthrough():
    assert aio.vec2(None) is None

def test_vel_roundtrip():
    d = aio.vel([0.1, 0.2, 0.3])
    assert set(d) == {"V_x", "V_y", "V_z"}
    assert np.allclose([d["V_x"], d["V_y"], d["V_z"]], [0.1, 0.2, 0.3])

def test_vel_none_passthrough():
    assert aio.vel(None) is None

def test_scalar_passthrough():
    assert aio.scalar(3.14) == pytest.approx(3.14)

def test_scalar_none_passthrough():
    assert aio.scalar(None) is None

def test_scalar_converts_to_float():
    assert isinstance(aio.scalar(1), float)

def test_variant_block_all_keys():
    b = aio.variant_block()
    assert set(b) == set(aio.VARIANTS)

def test_variant_block_defaults_to_none():
    b = aio.variant_block()
    assert all(v is None for v in b.values())

def test_variant_block_gt_set():
    p = aio.vec3([1, 2, 3])
    b = aio.variant_block(GT=p)
    assert b["GT"] == p
    for v in ["Noisy_GT", "Computed1", "Computed2", "Computed3"]:
        assert b[v] is None

def test_variant_block_multiple_set():
    b = aio.variant_block(GT=aio.vec3([1, 2, 3]), Noisy_GT=aio.vec3([4, 5, 6]))
    assert b["GT"] is not None
    assert b["Noisy_GT"] is not None
    assert b["Computed1"] is None

def test_variant_block_3d_has_no_noisy_gt():
    b = aio.variant_block(variants=aio.VARIANTS_3D, GT=aio.vec3([1, 2, 3]))
    assert set(b) == set(aio.VARIANTS_3D)
    assert "Noisy_GT" not in b
    assert b["GT"] is not None

def test_variants_3d_constant():
    assert aio.VARIANTS_3D == ["GT", "Computed1", "Computed2", "Computed3"]

# ── camera model ───────────────────────────────────────────────────────────────

def _cam(W=1920, H=1080, fovX=5.0):
    fovY = float(np.degrees(2.0 * np.arctan(np.tan(np.radians(fovX) / 2.0) * H / W)))
    return {"W_pixels": W, "H_pixels": H, "fov_X_deg": fovX, "fov_Y_deg": fovY}


def test_camera_intrinsics_principal_point():
    fx, fy, cx, cy = aio.camera_intrinsics(_cam(1920, 1080))
    assert cx == pytest.approx(960.0)
    assert cy == pytest.approx(540.0)

def test_camera_intrinsics_focal_90deg_square():
    cam = {"W_pixels": 100, "H_pixels": 100, "fov_X_deg": 90.0, "fov_Y_deg": 90.0}
    fx, fy, cx, cy = aio.camera_intrinsics(cam)
    assert fx == pytest.approx(50.0)
    assert fy == pytest.approx(50.0)

def test_camera_intrinsics_focal_formula():
    cam = _cam(W=200, H=100, fovX=60.0)
    fx, fy, cx, cy = aio.camera_intrinsics(cam)
    expected_fx = 200.0 / (2.0 * np.tan(np.radians(60.0) / 2.0))
    assert fx == pytest.approx(expected_fx)

def test_project_pixel_principal_point():
    fx, fy, cx, cy = aio.camera_intrinsics(_cam())
    uv = aio.project_pixel([0.0, 0.0, 100.0], fx, fy, cx, cy)
    assert np.allclose(uv, [cx, cy])

def test_project_pixel_known_offset():
    cam = {"W_pixels": 100, "H_pixels": 100, "fov_X_deg": 90.0, "fov_Y_deg": 90.0}
    fx, fy, cx, cy = aio.camera_intrinsics(cam)  # fx=fy=50, cx=cy=50
    uv = aio.project_pixel([1.0, 0.0, 1.0], fx, fy, cx, cy)
    assert np.allclose(uv, [50.0 + 50.0, 50.0])  # u = fx*(1/1) + cx

def test_project_pixel_invertible():
    fx, fy, cx, cy = aio.camera_intrinsics(_cam())
    P = [10.0, -5.0, 500.0]
    uv = aio.project_pixel(P, fx, fy, cx, cy)
    n = aio.pixel_to_normalized(uv, fx, fy, cx, cy)
    assert np.allclose(n, [P[0] / P[2], P[1] / P[2]])

def test_pixel_to_normalized_center():
    fx, fy, cx, cy = aio.camera_intrinsics(_cam())
    n = aio.pixel_to_normalized([cx, cy], fx, fy, cx, cy)
    assert np.allclose(n, [0.0, 0.0])

def test_pixel_to_normalized_roundtrip():
    fx, fy, cx, cy = aio.camera_intrinsics(_cam())
    px = [1100.0, 600.0]
    n = aio.pixel_to_normalized(px, fx, fy, cx, cy)
    px2 = aio.project_pixel([n[0], n[1], 1.0], fx, fy, cx, cy)
    assert np.allclose(px2, px)

# ── file I/O ───────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    obj = {"a": 1, "b": [2, 3], "c": None, "d": {"x": 1.5}}
    p = str(tmp_path / "test.json")
    aio.save_json(p, obj)
    loaded = aio.load_json(p)
    assert loaded == obj

def test_save_produces_valid_json(tmp_path):
    p = str(tmp_path / "x.json")
    aio.save_json(p, {"k": 42})
    with open(p) as f:
        json.load(f)   # must not raise

def test_save_uses_indentation(tmp_path):
    p = str(tmp_path / "ind.json")
    aio.save_json(p, {"k": 42})
    text = open(p).read()
    assert "\n" in text  # pretty-printed

def test_load_real_input():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "inputs", "aircraft_input.json")
    doc = aio.load_json(path)
    for block in aio.BLOCKS:
        assert block in doc
