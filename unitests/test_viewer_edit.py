"""Tests for the Change-GT editing view — rigid motion, undo, frozen anchor."""
import copy
import json
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from viewer import AircraftViewer, KEYPOINTS, SCHEME_FV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


@pytest.fixture
def edit_viewer(viewer):
    viewer._on_view("Change GT")
    return viewer


def _body_len(v):
    P1 = v.point3d("P1_tail", "GT")
    P2 = v.point3d("P2_nose", "GT")
    return float(np.linalg.norm(P2 - P1))


def _wing_span(v):
    P3 = v.point3d("P3_left_wing", "GT")
    P4 = v.point3d("P4_right_wing", "GT")
    return float(np.linalg.norm(P4 - P3))


def _centre(v):
    P1 = v.point3d("P1_tail", "GT")
    P2 = v.point3d("P2_nose", "GT")
    return 0.5 * (P1 + P2)


# ── translation preserves rigid geometry ─────────────────────────────────────

def test_translation_preserves_body_length(edit_viewer):
    L0 = _body_len(edit_viewer)
    for k in ["left", "right", "up", "down", "+", "-"]:
        edit_viewer._on_key_edit(k)
    assert np.isclose(_body_len(edit_viewer), L0, atol=1e-8)

def test_translation_preserves_wing_span(edit_viewer):
    S0 = _wing_span(edit_viewer)
    for k in ["left", "right", "up", "down", "+", "-"]:
        edit_viewer._on_key_edit(k)
    assert np.isclose(_wing_span(edit_viewer), S0, atol=1e-8)

# ── rotation preserves rigid geometry and keeps centre fixed ──────────────────

def test_rotation_preserves_body_length(edit_viewer):
    L0 = _body_len(edit_viewer)
    for k in ["shift+left", "shift+right", "shift+up", "shift+down",
              "ctrl+right", "ctrl+left"]:
        edit_viewer._on_key_edit(k)
    assert np.isclose(_body_len(edit_viewer), L0, atol=1e-8)

def test_rotation_preserves_wing_span(edit_viewer):
    S0 = _wing_span(edit_viewer)
    for k in ["shift+left", "shift+up", "ctrl+right"]:
        edit_viewer._on_key_edit(k)
    assert np.isclose(_wing_span(edit_viewer), S0, atol=1e-8)

def test_rotation_keeps_centre_fixed(edit_viewer):
    C0 = _centre(edit_viewer).copy()
    for k in ["shift+left", "shift+up", "ctrl+right", "shift+down"]:
        edit_viewer._on_key_edit(k)
    assert np.allclose(_centre(edit_viewer), C0, atol=1e-8)

# ── derived fields updated after motion ──────────────────────────────────────

def test_2d_projections_updated_after_translation(edit_viewer):
    edit_viewer._on_key_edit("right")
    fx, fy, cx, cy = aio.camera_intrinsics(edit_viewer.doc["Camera_parameters"])
    for el, el2d in [("P1_tail", "p1_tail"), ("P2_nose", "p2_nose"),
                     ("P3_left_wing", "p3_left_wing"), ("P4_right_wing", "p4_right_wing")]:
        P = edit_viewer.point3d(el, "GT")
        expected = aio.project_pixel(P, fx, fy, cx, cy)
        stored = edit_viewer.doc["Coordinates_2D"][el2d]["GT"]
        assert np.allclose(expected, [stored["u"], stored["v"]], atol=1e-6)

def test_range_updated_after_translation(edit_viewer):
    edit_viewer._on_key_edit("+")
    C = _centre(edit_viewer)
    expected = float(np.linalg.norm(C))
    assert np.isclose(edit_viewer.doc["Range_meters"]["GT"], expected, atol=1e-9)

def test_velocity_direction_matches_body_after_yaw(edit_viewer):
    edit_viewer._on_key_edit("shift+left")
    P1 = edit_viewer.point3d("P1_tail", "GT")
    P2 = edit_viewer.point3d("P2_nose", "GT")
    body = (P2 - P1) / np.linalg.norm(P2 - P1)
    Vd = edit_viewer.doc["Velocity_vector_3D"]["GT"]
    Vn = np.array([Vd["V_x"], Vd["V_y"], Vd["V_z"]])
    Vn /= np.linalg.norm(Vn)
    assert np.allclose(body, Vn, atol=1e-9)

def test_velocity_direction_matches_body_after_roll(edit_viewer):
    edit_viewer._on_key_edit("ctrl+right")
    P1 = edit_viewer.point3d("P1_tail", "GT")
    P2 = edit_viewer.point3d("P2_nose", "GT")
    body = (P2 - P1) / np.linalg.norm(P2 - P1)
    Vd = edit_viewer.doc["Velocity_vector_3D"]["GT"]
    Vn = np.array([Vd["V_x"], Vd["V_y"], Vd["V_z"]])
    Vn /= np.linalg.norm(Vn)
    assert np.allclose(body, Vn, atol=1e-9)

# ── undo ──────────────────────────────────────────────────────────────────────

def test_undo_single_step_restores_p1(edit_viewer):
    orig = copy.deepcopy(edit_viewer.doc["Coordinates_3D"]["P1_tail"]["GT"])
    edit_viewer._on_key_edit("right")
    assert edit_viewer.doc["Coordinates_3D"]["P1_tail"]["GT"] != orig
    edit_viewer._on_key_edit("ctrl+z")
    assert edit_viewer.doc["Coordinates_3D"]["P1_tail"]["GT"] == orig

def test_undo_all_steps_restores_full_doc(edit_viewer):
    orig = json.loads(json.dumps(edit_viewer.doc))
    keys = ["right"] * 3 + ["shift+left"] * 2 + ["+"] * 2
    for k in keys:
        edit_viewer._on_key_edit(k)
    for _ in keys:
        edit_viewer._on_key_edit("ctrl+z")
    assert edit_viewer.doc == orig

def test_undo_on_empty_stack_is_noop(edit_viewer):
    orig = json.loads(json.dumps(edit_viewer.doc))
    edit_viewer._on_key_edit("ctrl+z")  # stack is empty
    assert edit_viewer.doc == orig

def test_undo_stack_grows_per_move(edit_viewer):
    n0 = len(edit_viewer._undo_stack)
    edit_viewer._on_key_edit("right")
    edit_viewer._on_key_edit("up")
    assert len(edit_viewer._undo_stack) == n0 + 2

def test_undo_stack_clears_on_load(edit_viewer):
    edit_viewer._on_key_edit("right")
    assert len(edit_viewer._undo_stack) >= 1
    edit_viewer._ask_open = lambda: INPUT
    edit_viewer._on_load(None)
    assert len(edit_viewer._undo_stack) == 0

# ── frozen anchor ─────────────────────────────────────────────────────────────

def test_edit_anchor_set_on_view_entry(edit_viewer):
    assert edit_viewer._edit_anchor is not None
    assert "C3" in edit_viewer._edit_anchor
    assert "Qc" in edit_viewer._edit_anchor
    assert "s_air" in edit_viewer._edit_anchor

def test_edit_anchor_qc_frozen_across_edits(edit_viewer):
    Qc0 = edit_viewer._edit_anchor["Qc"].copy()
    for k in ["right"] * 5 + ["shift+left"] * 3:
        edit_viewer._on_key_edit(k)
    assert np.allclose(edit_viewer._edit_anchor["Qc"], Qc0, atol=1e-12)

def test_edit_anchor_c3_frozen_across_edits(edit_viewer):
    C3_0 = edit_viewer._edit_anchor["C3"].copy()
    for k in ["right"] * 5 + ["shift+up"] * 2:
        edit_viewer._on_key_edit(k)
    assert np.allclose(edit_viewer._edit_anchor["C3"], C3_0, atol=1e-12)

def test_map3d_stable_for_fixed_probe_across_edits(edit_viewer):
    probe = np.array([20.0, -15.0, 2000.0])
    _, _, map3d_before, _ = edit_viewer._scheme_geometry()
    mapped_before = map3d_before(probe).copy()
    for k in ["right"] * 4 + ["shift+left"] * 2:
        edit_viewer._on_key_edit(k)
    _, _, map3d_after, _ = edit_viewer._scheme_geometry()
    assert np.allclose(map3d_before(probe), map3d_after(probe), atol=1e-12)
    assert np.allclose(map3d_after(probe), mapped_before, atol=1e-12)

def test_edit_anchor_cleared_on_view_change(edit_viewer):
    assert edit_viewer._edit_anchor is not None
    edit_viewer._on_view("3D interactive")
    assert edit_viewer._edit_anchor is None

# ── scheme geometry plane ─────────────────────────────────────────────────────

def test_scheme_plane_corners_at_fv(edit_viewer):
    plane, _, _, _ = edit_viewer._scheme_geometry()
    assert np.allclose(plane[:, 2], SCHEME_FV, atol=1e-12)

def test_scheme_plane_corners_unchanged_across_edits(edit_viewer):
    plane0, _, _, _ = edit_viewer._scheme_geometry()
    for k in ["right"] * 5 + ["shift+left"] * 3:
        edit_viewer._on_key_edit(k)
    plane1, _, _, _ = edit_viewer._scheme_geometry()
    assert np.allclose(plane0, plane1, atol=1e-12)
