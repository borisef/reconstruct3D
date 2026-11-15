"""Tests for the '2D results' view (reprojected Computed vs GT/Noisy pixels)."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from viewer import AircraftViewer, KEYPOINTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


def _solve(v, noisy):
    """Run the viewer's Solve with the input-type parameter set as given."""
    v.algo_params = dict(v.algo_params, input_noisy=noisy)
    v._solve()


# ── view switching ────────────────────────────────────────────────────────────

def test_view_key_is_results2d(viewer):
    viewer._on_view("2D results")
    assert viewer.view == "results2d"

def test_renders_without_computed_data(viewer):
    viewer._on_view("2D results")   # Computed null; must not raise

def test_renders_after_solve(viewer):
    _solve(viewer, noisy=False)
    viewer._on_view("2D results")   # must not raise

# ── reprojection ──────────────────────────────────────────────────────────────

def test_projection_empty_without_computed(viewer):
    assert viewer._projected_computed_2d() == {}

def test_projection_close_to_gt_after_gt_solve(viewer):
    _solve(viewer, noisy=False)
    proj = viewer._projected_computed_2d()
    assert set(proj) == set(KEYPOINTS)
    for el in KEYPOINTS:
        gt = viewer.point2d(el, "GT")
        assert np.linalg.norm(proj[el] - gt) < 0.1, el

# ── error table ───────────────────────────────────────────────────────────────

def test_error_text_none_without_solve(viewer):
    assert viewer._results2d_error_text() is None

def test_error_text_vs_gt_after_gt_solve(viewer):
    _solve(viewer, noisy=False)
    txt = viewer._results2d_error_text()
    assert "vs GT" in txt
    assert "mean" in txt
    for el2d in ["p1_tail", "p2_nose", "p3_left_wing", "p4_right_wing"]:
        assert el2d in txt

def test_error_text_vs_noisy_after_noisy_solve(viewer):
    _solve(viewer, noisy=True)
    txt = viewer._results2d_error_text()
    assert "vs N_GT" in txt

def test_error_small_for_gt_solve(viewer):
    # Solutions lie on the source rays: reprojection error vs GT is tiny.
    _solve(viewer, noisy=False)
    txt = viewer._results2d_error_text()
    mean_line = [l for l in txt.splitlines() if "mean" in l][0]
    assert float(mean_line.split()[-1]) < 0.1

def test_error_grows_when_noise_added_after_solve(viewer):
    # Noise pressed after the solve moves the reference pixels, so the
    # reprojection error vs Noisy_GT becomes clearly nonzero.
    viewer._rng = np.random.default_rng(7)
    _solve(viewer, noisy=True)
    viewer._on_noise_target("2D")
    for _ in range(6):
        viewer._on_noise(up=True)
    txt = viewer._results2d_error_text()
    mean_line = [l for l in txt.splitlines() if "mean" in l][0]
    assert float(mean_line.split()[-1]) > 0.5

# ── solve-source tracking ─────────────────────────────────────────────────────

def test_solve_records_source(viewer):
    _solve(viewer, noisy=True)
    assert viewer._last_solve_source == "Noisy_GT"
    _solve(viewer, noisy=False)
    assert viewer._last_solve_source == "GT"

def test_load_resets_source(viewer):
    _solve(viewer, noisy=False)
    viewer._ask_open = lambda: INPUT
    viewer._on_load(None)
    assert viewer._last_solve_source is None
    assert viewer._results2d_error_text() is None
