"""Tests for the GT 2D/3D consistency check (auto-correct on load)."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from viewer import AircraftViewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


def _reprojection(viewer, element):
    fx, fy, cx, cy = aio.camera_intrinsics(viewer.doc["Camera_parameters"])
    return aio.project_pixel(viewer.point3d(element, "GT"), fx, fy, cx, cy)


# ── detection and correction ──────────────────────────────────────────────────

def test_input_file_is_consistent(viewer):
    assert viewer._check_gt_2d_consistency() == []

def test_stale_2d_gt_detected_and_fixed(viewer):
    viewer.doc["Coordinates_2D"]["p2_nose"]["GT"]["u"] += 25.0
    assert viewer._check_gt_2d_consistency() == ["P2_nose"]
    assert np.allclose(viewer.point2d("P2_nose", "GT"),
                       _reprojection(viewer, "P2_nose"), atol=1e-12)

def test_null_2d_gt_is_filled(viewer):
    viewer.doc["Coordinates_2D"]["p1_tail"]["GT"] = None
    assert viewer._check_gt_2d_consistency() == ["P1_tail"]
    assert np.allclose(viewer.point2d("P1_tail", "GT"),
                       _reprojection(viewer, "P1_tail"), atol=1e-12)

def test_missing_3d_gt_leaves_2d_untouched(viewer):
    # Without a 3D GT reference there is nothing to project from.
    viewer.doc["Coordinates_3D"]["P1_tail"]["GT"] = None
    viewer.doc["Coordinates_2D"]["p1_tail"]["GT"]["u"] += 25.0
    before = viewer.point2d("P1_tail", "GT").copy()
    assert viewer._check_gt_2d_consistency() == []
    assert np.allclose(viewer.point2d("P1_tail", "GT"), before)

def test_other_variants_not_touched(viewer):
    noisy_before = viewer.point2d("P2_nose", "Noisy_GT").copy()
    viewer.doc["Coordinates_2D"]["p2_nose"]["GT"]["u"] += 25.0
    viewer._check_gt_2d_consistency()
    assert np.allclose(viewer.point2d("P2_nose", "Noisy_GT"), noisy_before)

# ── info box ──────────────────────────────────────────────────────────────────

def test_ensure_pops_info_box_on_correction(viewer):
    calls = []
    viewer._show_info = lambda title, msg: calls.append((title, msg))
    viewer.doc["Coordinates_2D"]["p3_left_wing"]["GT"]["v"] -= 10.0
    assert viewer._ensure_gt_2d_consistent() == ["P3_left_wing"]
    assert len(calls) == 1
    assert "P3_left_wing" in calls[0][1]

def test_ensure_silent_when_consistent(viewer):
    calls = []
    viewer._show_info = lambda title, msg: calls.append(1)
    assert viewer._ensure_gt_2d_consistent() == []
    assert calls == []

# ── load paths ────────────────────────────────────────────────────────────────

def test_load_corrects_corrupt_file(viewer, tmp_path):
    doc = aio.load_json(INPUT)
    doc["Coordinates_2D"]["p2_nose"]["GT"]["u"] += 40.0
    bad = str(tmp_path / "bad.json")
    aio.save_json(bad, doc)
    calls = []
    viewer._show_info = lambda title, msg: calls.append(title)
    viewer._ask_open = lambda: bad
    viewer._on_load(None)
    assert calls == ["2D GT corrected"]
    assert np.allclose(viewer.point2d("P2_nose", "GT"),
                       _reprojection(viewer, "P2_nose"), atol=1e-12)

def test_init_corrects_corrupt_file(tmp_path, monkeypatch):
    doc = aio.load_json(INPUT)
    doc["Coordinates_2D"]["p1_tail"]["GT"]["v"] += 15.0
    bad = str(tmp_path / "bad.json")
    aio.save_json(bad, doc)
    calls = []
    monkeypatch.setattr(AircraftViewer, "_show_info",
                        staticmethod(lambda title, msg: calls.append(title)))
    v = AircraftViewer(bad)
    assert calls == ["2D GT corrected"]
    assert v._check_gt_2d_consistency() == []
