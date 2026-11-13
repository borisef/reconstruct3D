"""Tests for AircraftViewer visibility and data-access logic (headless)."""
import os
import sys

import numpy as np
import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from viewer import AircraftViewer, COLUMNS, VARIANTS, ELEMENT_2D, KEYPOINTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


# ── cell enabled/disabled ─────────────────────────────────────────────────────

def test_gt_cells_enabled_for_all_columns(viewer):
    for c in range(len(COLUMNS)):
        assert viewer._cell_enabled(c, "GT"), f"col {c} GT should be enabled"

def test_noisy_gt_cells_enabled_except_3d_points(viewer):
    """Noisy_GT exists for 2D/velocity/range only; 3D keypoints have none."""
    for c, (key, kind, _lbl) in enumerate(COLUMNS):
        is_3d_point = (kind == "3D" and key in ELEMENT_2D)
        assert viewer._cell_enabled(c, "Noisy_GT") == (not is_3d_point), \
            f"col {c} ({key},{kind}) Noisy_GT enabled state wrong"

def test_computed1_disabled_for_3d_body(viewer):
    c = viewer._col_index("P1_tail", "3D")
    assert not viewer._cell_enabled(c, "Computed1")

def test_computed1_disabled_for_2d_body(viewer):
    c = viewer._col_index("P1_tail", "2D")
    assert not viewer._cell_enabled(c, "Computed1")

def test_range_scalar_cell_enabled(viewer):
    c = viewer._col_index("range", "scalar")
    assert viewer._cell_enabled(c, "GT")
    assert viewer._cell_enabled(c, "Noisy_GT")
    assert not viewer._cell_enabled(c, "Computed1")

# ── default visibility: GT only ───────────────────────────────────────────────

def test_default_gt_checked_where_enabled(viewer):
    for c in range(len(COLUMNS)):
        if viewer._cell_enabled(c, "GT"):
            assert viewer._cell_checked(c, "GT"), \
                f"col {c} GT should be checked by default"

def test_default_noisy_gt_not_checked(viewer):
    for c in range(len(COLUMNS)):
        assert not viewer._cell_checked(c, "Noisy_GT"), \
            f"col {c} Noisy_GT should not be checked by default"

def test_default_computed_not_checked(viewer):
    for c in range(len(COLUMNS)):
        for v in ["Computed1", "Computed2", "Computed3"]:
            assert not viewer._cell_checked(c, v)

# ── toggle ────────────────────────────────────────────────────────────────────

def test_toggle_flips_gt_off(viewer):
    c = viewer._col_index("P1_tail", "3D")
    assert viewer._cell_checked(c, "GT")
    viewer._toggle(c, "GT")
    assert not viewer._cell_checked(c, "GT")

def test_toggle_flips_gt_back_on(viewer):
    c = viewer._col_index("P2_nose", "3D")
    viewer._toggle(c, "GT")
    viewer._toggle(c, "GT")
    assert viewer._cell_checked(c, "GT")

def test_toggle_enables_noisy_gt(viewer):
    c = viewer._col_index("P1_tail", "2D")
    assert not viewer._cell_checked(c, "Noisy_GT")
    viewer._toggle(c, "Noisy_GT")
    assert viewer._cell_checked(c, "Noisy_GT")

# ── _col_index ────────────────────────────────────────────────────────────────

def test_col_index_p1_3d(viewer):
    c = viewer._col_index("P1_tail", "3D")
    key, kind, _ = COLUMNS[c]
    assert key == "P1_tail" and kind == "3D"

def test_col_index_p1_2d(viewer):
    c3 = viewer._col_index("P1_tail", "3D")
    c2 = viewer._col_index("P1_tail", "2D")
    assert c2 != c3

def test_col_index_velocity(viewer):
    c = viewer._col_index("velocity", "3D")
    key, kind, _ = COLUMNS[c]
    assert key == "velocity" and kind == "3D"

def test_col_index_range(viewer):
    c = viewer._col_index("range", "scalar")
    key, kind, _ = COLUMNS[c]
    assert key == "range" and kind == "scalar"

# ── _is_visible ───────────────────────────────────────────────────────────────

def test_is_visible_gt_true_by_default(viewer):
    assert viewer._is_visible("P1_tail", "3D", "GT")

def test_is_visible_noisy_false_by_default(viewer):
    assert not viewer._is_visible("P1_tail", "3D", "Noisy_GT")

def test_is_visible_after_toggle(viewer):
    viewer._toggle(viewer._col_index("P1_tail", "3D"), "GT")
    assert not viewer._is_visible("P1_tail", "3D", "GT")

# ── _sync_visibility ──────────────────────────────────────────────────────────

def test_sync_drops_checked_null_field(viewer):
    c = viewer._col_index("P1_tail", "3D")
    viewer.visible[(c, "Computed1")] = True  # force-enable a null field
    viewer._sync_visibility()
    assert not viewer.visible.get((c, "Computed1"), False)

# ── data accessors ────────────────────────────────────────────────────────────

def test_point3d_gt_returns_array(viewer):
    P = viewer.point3d("P1_tail", "GT")
    assert P is not None and P.shape == (3,)

def test_point3d_computed1_returns_none(viewer):
    assert viewer.point3d("P1_tail", "Computed1") is None

def test_point2d_gt_returns_array(viewer):
    p = viewer.point2d("P1_tail", "GT")
    assert p is not None and p.shape == (2,)

def test_velocity_gt_returns_unit_vector(viewer):
    V = viewer.velocity("GT")
    assert V is not None
    assert np.isclose(np.linalg.norm(V), 1.0, atol=1e-6)

# ── _priority_variant_2d ──────────────────────────────────────────────────────

def test_priority_variant_2d_returns_gt_by_default(viewer):
    v = viewer._priority_variant_2d(["P1_tail", "P2_nose"])
    assert v == "GT"

def test_priority_variant_2d_none_when_nothing_checked(viewer):
    for c in range(len(COLUMNS)):
        viewer.visible[(c, "GT")] = False
    assert viewer._priority_variant_2d(["P1_tail", "P2_nose"]) is None
