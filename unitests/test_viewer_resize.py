"""Tests for Change-GT wingspan (a/d) and body-length (q/e) scaling."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from viewer import AircraftViewer, GT_SCALE_STEP, GT_SIZE_MIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    v._on_view("Change GT")
    yield v
    plt.close(v.fig)


def _stats(v):
    return v._aircraft_stats("GT")


# ── wingspan (a/d) ────────────────────────────────────────────────────────────

def test_d_key_grows_wingspan(viewer):
    viewer._on_key_edit("d")
    assert _stats(viewer)["wing_span"] == pytest.approx(
        40.0 * GT_SCALE_STEP, abs=1e-9)

def test_a_key_shrinks_wingspan(viewer):
    viewer._on_key_edit("a")
    assert _stats(viewer)["wing_span"] == pytest.approx(
        40.0 / GT_SCALE_STEP, abs=1e-9)

def test_span_scaling_keeps_midpoint_and_cross(viewer):
    Wm0 = 0.5 * (viewer.point3d("P3_left_wing", "GT")
                 + viewer.point3d("P4_right_wing", "GT"))
    viewer._on_key_edit("d")
    Wm1 = 0.5 * (viewer.point3d("P3_left_wing", "GT")
                 + viewer.point3d("P4_right_wing", "GT"))
    assert np.allclose(Wm0, Wm1, atol=1e-9)
    assert _stats(viewer)["cross"] == pytest.approx(0.5, abs=1e-9)

def test_span_scaling_keeps_body(viewer):
    b1 = viewer.point3d("P1_tail", "GT").copy()
    viewer._on_key_edit("a")
    assert np.allclose(viewer.point3d("P1_tail", "GT"), b1, atol=1e-12)
    assert _stats(viewer)["body_len"] == pytest.approx(30.0, abs=1e-9)

def test_span_clamped_at_min(viewer):
    for _ in range(120):
        viewer._on_key_edit("a")
    assert _stats(viewer)["wing_span"] >= GT_SIZE_MIN

# ── body length (q/e) ─────────────────────────────────────────────────────────

def test_e_key_grows_body(viewer):
    viewer._on_key_edit("e")
    assert _stats(viewer)["body_len"] == pytest.approx(
        30.0 * GT_SCALE_STEP, abs=1e-9)

def test_q_key_shrinks_body(viewer):
    viewer._on_key_edit("q")
    assert _stats(viewer)["body_len"] == pytest.approx(
        30.0 / GT_SCALE_STEP, abs=1e-9)

def test_body_scaling_keeps_centre_range_span(viewer):
    xyz0 = _stats(viewer)["xyz"]
    rng0 = viewer.doc["Range_meters"]["GT"]
    viewer._on_key_edit("e")
    assert np.allclose(_stats(viewer)["xyz"], xyz0, atol=1e-9)
    assert viewer.doc["Range_meters"]["GT"] == pytest.approx(rng0, abs=1e-6)
    assert _stats(viewer)["wing_span"] == pytest.approx(40.0, abs=1e-9)

def test_body_scaling_preserves_cross_station(viewer):
    # Move the wings to CrossPoint 0.75, then stretch the body: the wings
    # must stay at 75% of the (longer) fuselage.
    for _ in range(5):
        viewer._on_key_edit("w")          # 0.5 -> 0.75
    assert _stats(viewer)["cross"] == pytest.approx(0.75, abs=1e-9)
    viewer._on_key_edit("e")
    assert _stats(viewer)["cross"] == pytest.approx(0.75, abs=1e-9)
    viewer._on_key_edit("q")
    viewer._on_key_edit("q")
    assert _stats(viewer)["cross"] == pytest.approx(0.75, abs=1e-9)

def test_body_clamped_at_min(viewer):
    for _ in range(120):
        viewer._on_key_edit("q")
    assert _stats(viewer)["body_len"] >= GT_SIZE_MIN

# ── shared invariants ─────────────────────────────────────────────────────────

def test_gt_2d_reprojected_after_scaling(viewer):
    viewer._on_key_edit("d")
    viewer._on_key_edit("e")
    assert viewer._check_gt_2d_consistency() == []

def test_velocity_unchanged_by_scaling(viewer):
    vel = viewer.velocity("GT").copy()
    viewer._on_key_edit("d")
    viewer._on_key_edit("q")
    assert np.allclose(viewer.velocity("GT"), vel, atol=1e-12)

def test_undo_restores_sizes(viewer):
    viewer._on_key_edit("d")
    viewer._on_key_edit("e")
    viewer._on_key_edit("ctrl+z")
    assert _stats(viewer)["body_len"] == pytest.approx(30.0, abs=1e-9)
    viewer._on_key_edit("ctrl+z")
    assert _stats(viewer)["wing_span"] == pytest.approx(40.0, abs=1e-9)
