"""Tests for the Change-GT wings slide (CrossPoint editing, keys w/s)."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from viewer import AircraftViewer, GT_CROSS_STEP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    v._on_view("Change GT")
    yield v
    plt.close(v.fig)


def _cross(v):
    return v._aircraft_stats("GT")["cross"]


# ── slide semantics ───────────────────────────────────────────────────────────

def test_initial_cross_is_half(viewer):
    assert _cross(viewer) == pytest.approx(0.5, abs=1e-9)

def test_w_key_moves_wings_toward_nose(viewer):
    viewer._on_key_edit("w")
    assert _cross(viewer) == pytest.approx(0.5 + GT_CROSS_STEP, abs=1e-9)

def test_s_key_moves_wings_toward_tail(viewer):
    viewer._on_key_edit("s")
    assert _cross(viewer) == pytest.approx(0.5 - GT_CROSS_STEP, abs=1e-9)

def test_cross_clamped_at_one(viewer):
    for _ in range(30):
        viewer._on_key_edit("w")
    assert _cross(viewer) == pytest.approx(1.0, abs=1e-9)

def test_cross_clamped_at_zero(viewer):
    for _ in range(30):
        viewer._on_key_edit("s")
    assert _cross(viewer) == pytest.approx(0.0, abs=1e-9)

# ── what moves and what stays ─────────────────────────────────────────────────

def test_body_points_unchanged(viewer):
    b1 = viewer.point3d("P1_tail", "GT").copy()
    b2 = viewer.point3d("P2_nose", "GT").copy()
    viewer._on_key_edit("w")
    assert np.allclose(viewer.point3d("P1_tail", "GT"), b1, atol=1e-12)
    assert np.allclose(viewer.point3d("P2_nose", "GT"), b2, atol=1e-12)

def test_wing_span_and_direction_preserved(viewer):
    before = (viewer.point3d("P4_right_wing", "GT")
              - viewer.point3d("P3_left_wing", "GT"))
    viewer._on_key_edit("w")
    after = (viewer.point3d("P4_right_wing", "GT")
             - viewer.point3d("P3_left_wing", "GT"))
    assert np.allclose(after, before, atol=1e-9)

def test_velocity_and_range_unchanged(viewer):
    vel = viewer.velocity("GT").copy()
    rng = viewer.doc["Range_meters"]["GT"]
    viewer._on_key_edit("w")
    assert np.allclose(viewer.velocity("GT"), vel, atol=1e-12)
    assert viewer.doc["Range_meters"]["GT"] == pytest.approx(rng, abs=1e-9)

def test_gt_2d_reprojected_after_slide(viewer):
    viewer._on_key_edit("w")
    assert viewer._check_gt_2d_consistency() == []   # pixels follow the 3D

# ── undo ──────────────────────────────────────────────────────────────────────

def test_undo_restores_cross(viewer):
    viewer._on_key_edit("w")
    viewer._on_key_edit("w")
    viewer._on_key_edit("ctrl+z")
    assert _cross(viewer) == pytest.approx(0.5 + GT_CROSS_STEP, abs=1e-9)
    viewer._on_key_edit("ctrl+z")
    assert _cross(viewer) == pytest.approx(0.5, abs=1e-9)

def test_no_undo_entry_when_already_clamped(viewer):
    for _ in range(11):                 # 0.5 -> 1.0 in 10 steps, then clamp
        viewer._on_key_edit("w")
    depth = len(viewer._undo_stack)
    viewer._on_key_edit("w")            # no-op at the clamp: no undo push
    assert len(viewer._undo_stack) == depth
