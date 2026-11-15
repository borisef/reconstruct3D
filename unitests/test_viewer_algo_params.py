"""Tests for the Set Algo Params flow and the single-solution Solve button."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from main import DEFAULT_ALGO_PARAMS
from viewer import AircraftViewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


# ── buttons ───────────────────────────────────────────────────────────────────

def test_single_solve_button_no_solve_noisy(viewer):
    assert hasattr(viewer, "_btn_solve")
    assert not hasattr(viewer, "_btn_solve_noisy")
    assert not hasattr(viewer, "_btn_solve_gt")

def test_set_algo_params_button_exists(viewer):
    assert viewer._btn_params.label.get_text() == "Set Algo Params"

# ── parameter state ───────────────────────────────────────────────────────────

def test_defaults_on_fresh_input(viewer):
    assert viewer.algo_params == DEFAULT_ALGO_PARAMS

def test_doc_params_loaded(tmp_path, viewer):
    doc = aio.load_json(INPUT)
    doc["Algorithm_parameters"] = {"input_noisy": False,
                                   "head_tail_solution": "Average"}
    path = str(tmp_path / "with_params.json")
    aio.save_json(path, doc)
    viewer._ask_open = lambda: path
    viewer._on_load(None)
    assert viewer.algo_params == {"input_noisy": False,
                                  "head_tail_solution": "Average"}

def test_set_params_updates_state_and_doc(viewer):
    viewer._ask_algo_params = lambda: {"input_noisy": False,
                                       "head_tail_solution": "#2"}
    viewer._on_set_params(None)
    assert viewer.algo_params["input_noisy"] is False
    assert viewer.algo_params["head_tail_solution"] == "#2"
    assert viewer.doc["Algorithm_parameters"] == viewer.algo_params
    assert "[params:" in viewer._status.get_text()

def test_set_params_cancel_changes_nothing(viewer):
    before = dict(viewer.algo_params)
    viewer._ask_algo_params = lambda: None
    viewer._on_set_params(None)
    assert viewer.algo_params == before
    assert "Algorithm_parameters" not in viewer.doc

# ── Solve honours the parameters ──────────────────────────────────────────────

def test_solve_uses_input_type(viewer):
    viewer._ask_algo_params = lambda: {"input_noisy": False,
                                       "head_tail_solution": "#1"}
    viewer._on_set_params(None)
    viewer._solve()
    assert viewer._last_solve_source == "GT"
    # noise-free solve: Computed is at the GT points
    gt = viewer.point3d("P1_tail", "GT")
    assert np.linalg.norm(viewer.point3d("P1_tail", "Computed") - gt) < 0.01

def test_solve_default_is_noisy(viewer):
    viewer._solve()
    assert viewer._last_solve_source == "Noisy_GT"
    assert viewer.doc["Results"]["source"] == "Noisy_GT"

def test_solve_head_tail_choice_changes_solution(viewer):
    viewer.algo_params = dict(viewer.algo_params, head_tail_solution="#1")
    viewer._solve()
    p1 = viewer.point3d("P1_tail", "Computed").copy()
    viewer.algo_params = dict(viewer.algo_params, head_tail_solution="#2")
    viewer._solve()
    p2 = viewer.point3d("P1_tail", "Computed")
    assert np.linalg.norm(p1 - p2) > 1e-6

def test_solve_writes_params_and_results_blocks(viewer):
    viewer._solve()
    assert viewer.doc["Algorithm_parameters"] == viewer.algo_params
    res = viewer.doc["Results"]
    assert set(res) == {"source", "errors_2D_px", "table_3D"}
    assert res["errors_2D_px"]["mean"] == pytest.approx(0.0, abs=1e-6)

def test_solve_status_mentions_params(viewer):
    viewer._solve()
    assert "head/tail #1" in viewer._status.get_text()
    assert "input Noisy" in viewer._status.get_text()

# ── Computed 2D keypoints show up like any other variant ─────────────────────

def test_2d_computed_cells_enabled_after_solve(viewer):
    c = viewer._col_index("P1_tail", "2D")
    assert not viewer._cell_enabled(c, "Computed")
    viewer._solve()
    for el in ["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"]:
        assert viewer._cell_enabled(viewer._col_index(el, "2D"), "Computed"), el
        assert viewer.point2d(el, "Computed") is not None, el

def test_2d_computed_drawn_in_2d_and_scheme_views(viewer):
    viewer._solve()
    for el in ["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"]:
        viewer.visible[(viewer._col_index(el, "2D"), "Computed")] = True
    for view in ["2D", "Pinhole (scheme)", "3D interactive", "2D results"]:
        viewer._on_view(view)   # must not raise with Computed 2D checked
    # The scheme plane marker exists for the Computed pixels too.
    _, _, _, map2d = viewer._scheme_geometry()
    assert map2d("P1_tail", "Computed") is not None

def test_2d_computed_matches_projected_3d_in_viewer(viewer):
    viewer._solve()
    proj = viewer._projected_computed_2d()
    for el, p in proj.items():
        stored = viewer.point2d(el, "Computed")
        assert np.allclose(p, stored, atol=1e-9), el

def test_viewer_error_table_matches_saved_results(viewer):
    viewer.algo_params = dict(viewer.algo_params, input_noisy=True)
    viewer._solve()
    txt = viewer._results2d_error_text()
    saved = viewer.doc["Results"]["errors_2D_px"]
    mean_line = [l for l in txt.splitlines() if "mean" in l][0]
    assert float(mean_line.split()[-1]) == pytest.approx(saved["mean"],
                                                         abs=0.01)
