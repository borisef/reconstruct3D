"""Tests for the '3D results' GT-vs-Computed comparison view."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from main import solve_document
from viewer import AircraftViewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


@pytest.fixture
def solved_viewer(viewer):
    solve_document(viewer.doc, source="Noisy_GT")
    viewer._sync_visibility()
    viewer._grid.draw()
    viewer._on_view("3D results")
    return viewer


class _Key:
    def __init__(self, key):
        self.key = key


# ── view switching ────────────────────────────────────────────────────────────

def test_view_key_is_results(viewer):
    viewer._on_view("3D results")
    assert viewer.view == "results"

def test_renders_without_computed_data(viewer):
    viewer._on_view("3D results")   # input doc: Computed* all null; no raise

def test_renders_with_computed_data(solved_viewer):
    assert solved_viewer.view == "results"

# ── computed-variant selection rule ───────────────────────────────────────────

def test_default_computed_variant_is_computed1(solved_viewer):
    assert solved_viewer._results_computed_variant() == "Computed1"

def test_computed2_selected_wins(solved_viewer):
    c = solved_viewer._col_index("P1_tail", "3D")
    solved_viewer.visible[(c, "Computed2")] = True
    assert solved_viewer._results_computed_variant() == "Computed2"

def test_computed2_checked_without_data_falls_back(viewer):
    # No solve has run: Computed2 is null everywhere, so even a (stale)
    # checked cell must not select it.
    c = viewer._col_index("P1_tail", "3D")
    viewer.visible[(c, "Computed2")] = True
    viewer._on_view("3D results")
    assert viewer._results_computed_variant() == "Computed1"

# ── framing: all aircraft points in view, real-metre scale ────────────────────

def test_limits_fit_aircraft_scale(solved_viewer):
    ax = solved_viewer.ax3d
    for lo, hi in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()):
        assert 1.0 < hi - lo < 500.0   # metres around the aircraft

def test_all_points_inside_limits(solved_viewer):
    ax = solved_viewer.ax3d
    xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    for el in ["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"]:
        for variant in ["GT", "Computed1"]:
            P = solved_viewer.point3d(el, variant)
            q = solved_viewer._to3d(P)
            assert xl[0] <= q[0] <= xl[1]
            assert yl[0] <= q[1] <= yl[1]
            assert zl[0] <= q[2] <= zl[1]

def test_scheme_view_not_polluted_by_results_camera(solved_viewer):
    # Returning to the virtual scheme must recompute O(1)-unit limits, not
    # inherit the ~metres-scale results framing.
    solved_viewer._on_view("3D interactive")
    ax = solved_viewer.ax3d
    spans = [hi - lo for lo, hi in
             (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())]
    assert max(spans) < 10.0

def test_results_camera_kept_across_rerender(solved_viewer):
    ax = solved_viewer.ax3d
    ax.view_init(elev=33.0, azim=44.0, roll=5.0)
    solved_viewer.render()   # e.g. a checkbox toggle re-renders
    assert (ax.elev, ax.azim, ax.roll) == (33.0, 44.0, 5.0)

# ── aircraft stats / results table ────────────────────────────────────────────

def test_stats_gt_centre_and_sizes(viewer):
    s = viewer._aircraft_stats("GT")
    P1 = viewer.point3d("P1_tail", "GT")
    P2 = viewer.point3d("P2_nose", "GT")
    assert np.allclose(s["xyz"], 0.5 * (P1 + P2), atol=1e-10)
    assert s["body_len"] == pytest.approx(30.0, abs=1e-6)
    assert s["wing_span"] == pytest.approx(40.0, abs=1e-6)

def test_stats_gt_cross_point_is_half(viewer):
    # Synthetic GT: wings midpoint == body midpoint.
    s = viewer._aircraft_stats("GT")
    assert s["cross"] == pytest.approx(0.5, abs=1e-9)

def test_stats_cross_point_toward_nose(viewer):
    # Shift the wing pair 1/4 body length toward the nose -> cross = 0.75.
    import aircraft_io as aio
    P1 = viewer.point3d("P1_tail", "GT")
    P2 = viewer.point3d("P2_nose", "GT")
    shift = 0.25 * (P2 - P1)
    for el in ["P3_left_wing", "P4_right_wing"]:
        P = viewer.point3d(el, "GT")
        viewer.doc["Coordinates_3D"][el]["GT"] = aio.vec3(P + shift)
    s = viewer._aircraft_stats("GT")
    assert s["cross"] == pytest.approx(0.75, abs=1e-9)

def test_stats_cross_point_clamped(viewer):
    # Wings far beyond the nose still clamp to 1.0.
    import aircraft_io as aio
    P1 = viewer.point3d("P1_tail", "GT")
    P2 = viewer.point3d("P2_nose", "GT")
    shift = 2.0 * (P2 - P1)
    for el in ["P3_left_wing", "P4_right_wing"]:
        P = viewer.point3d(el, "GT")
        viewer.doc["Coordinates_3D"][el]["GT"] = aio.vec3(P + shift)
    s = viewer._aircraft_stats("GT")
    assert s["cross"] == 1.0

def test_stats_ypr_present_and_finite(viewer):
    s = viewer._aircraft_stats("GT")
    assert "ypr" in s and np.all(np.isfinite(s["ypr"]))

def test_stats_computed2_wings_fall_back_to_computed1(solved_viewer):
    s = solved_viewer._aircraft_stats("Computed2")
    assert s is not None
    assert "wing_span" in s and "cross" in s   # via Computed1 wings

def test_stats_none_without_body(viewer):
    assert viewer._aircraft_stats("Computed1") is None

def test_table_has_all_rows_and_columns(solved_viewer):
    txt = solved_viewer._results_table_text()
    for token in ["GT", "Calc1", "Calc2", "XYZ (m)", "YPR (deg)",
                  "BodyLength", "WingsSpan", "CrossPoint"]:
        assert token in txt

def test_table_dashes_without_computed(viewer):
    txt = viewer._results_table_text()
    # GT column filled, Calc1/Calc2 blank markers present
    assert "30.00" in txt       # GT body length
    assert "0.500" in txt       # GT cross point
    assert "-" in txt

def test_table_calc_values_close_to_gt_after_gt_solve(viewer):
    solve_document(viewer.doc, source="GT")
    s1 = viewer._aircraft_stats("Computed1")
    sg = viewer._aircraft_stats("GT")
    assert abs(s1["body_len"] - sg["body_len"]) < 0.01
    assert abs(s1["wing_span"] - sg["wing_span"]) < 0.01
    assert abs(s1["cross"] - sg["cross"]) < 0.001
    assert np.allclose(s1["ypr"], sg["ypr"], atol=0.1)

# ── camera keys work in the results view ──────────────────────────────────────

def test_zoom_key_shrinks_limits(solved_viewer):
    ax = solved_viewer.ax3d
    span0 = ax.get_xlim()[1] - ax.get_xlim()[0]
    solved_viewer._on_key(_Key("+"))
    span1 = ax.get_xlim()[1] - ax.get_xlim()[0]
    assert span1 < span0

def test_pan_key_shifts_limits(solved_viewer):
    ax = solved_viewer.ax3d
    xl0 = ax.get_xlim()
    solved_viewer._on_key(_Key("right"))
    assert ax.get_xlim() != xl0 or ax.get_ylim() != xl0
