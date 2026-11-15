"""Tests for the derived vertical tail stabilizer (fin)."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from aircraft_frame import AircraftFrame
from main import solve_document
from viewer import AircraftViewer, FIN_FRACTION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


# ── geometry ──────────────────────────────────────────────────────────────────

def test_gt_fin_starts_at_tail(viewer):
    tail, _tip = viewer._fin_segment("GT")
    assert np.allclose(tail, viewer.point3d("P1_tail", "GT"))

def test_gt_fin_length_is_fraction_of_body(viewer):
    tail, tip = viewer._fin_segment("GT")
    body_len = np.linalg.norm(viewer.point3d("P2_nose", "GT")
                              - viewer.point3d("P1_tail", "GT"))
    assert np.linalg.norm(tip - tail) == pytest.approx(
        FIN_FRACTION * body_len, abs=1e-9)
    assert body_len == pytest.approx(30.0, abs=1e-6)   # fin = 6 m

def test_gt_fin_points_along_local_z(viewer):
    tail, tip = viewer._fin_segment("GT")
    d = (tip - tail) / np.linalg.norm(tip - tail)
    frame = AircraftFrame.from_doc(viewer.doc)
    assert np.allclose(d, frame.R[:, 2], atol=1e-9)

def test_gt_fin_perpendicular_to_body_and_wings(viewer):
    tail, tip = viewer._fin_segment("GT")
    d = tip - tail
    body = viewer.point3d("P2_nose", "GT") - viewer.point3d("P1_tail", "GT")
    wings = (viewer.point3d("P4_right_wing", "GT")
             - viewer.point3d("P3_left_wing", "GT"))
    assert abs(np.dot(d, body)) < 1e-6
    assert abs(np.dot(d, wings)) < 1e-6

def test_fin_rotates_with_gt(viewer):
    viewer._on_view("Change GT")
    viewer._move_gt(axis_local=[0.0, 1.0, 0.0], angle_deg=35.0)  # roll
    tail, tip = viewer._fin_segment("GT")
    d = (tip - tail) / np.linalg.norm(tip - tail)
    frame = AircraftFrame.from_doc(viewer.doc)
    assert np.allclose(d, frame.R[:, 2], atol=1e-9)

# ── missing-data behaviour ────────────────────────────────────────────────────

def test_fin_none_for_noisy_gt(viewer):
    # Noisy_GT has no 3D keypoints, so no fin can be derived.
    assert viewer._fin_segment("Noisy_GT") is None

def test_fin_none_without_any_data(viewer):
    assert viewer._fin_segment("Computed") is None

def test_computed_fin_close_to_gt_after_gt_solve(viewer):
    solve_document(viewer.doc, params={"input_noisy": False})
    seg = viewer._fin_segment("Computed")
    assert seg is not None
    # It closely matches the GT fin for a noise-free solve.
    gt_tail, gt_tip = viewer._fin_segment("GT")
    assert np.allclose(seg[1], gt_tip, atol=0.1)

# ── rendering smoke ───────────────────────────────────────────────────────────

def test_all_views_render_with_fin(viewer):
    solve_document(viewer.doc, params={"input_noisy": False})
    viewer._sync_visibility()
    for view in ["2D", "Pinhole (scheme)", "3D interactive",
                 "3D results", "Change GT"]:
        viewer._on_view(view)   # must not raise
