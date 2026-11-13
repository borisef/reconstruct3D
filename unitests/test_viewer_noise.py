"""Tests for Noise Up/Down/target in AircraftViewer."""
import json
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from viewer import AircraftViewer, NOISE_PARAMS, NOISE_FACTOR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    v._rng = np.random.default_rng(42)   # deterministic
    yield v
    plt.close(v.fig)


# ── 2D noise sigma ladder ─────────────────────────────────────────────────────

def test_2d_sigma_starts_at_default(viewer):
    assert viewer.noise_sigma["2D"] == NOISE_PARAMS["2D"]["default"]

def test_2d_sigma_up_increases(viewer):
    s0 = viewer.noise_sigma["2D"]
    viewer._on_noise_target("2D")
    viewer._on_noise(up=True)
    assert viewer.noise_sigma["2D"] > s0

def test_2d_sigma_down_decreases(viewer):
    viewer._on_noise_target("2D")
    s0 = viewer.noise_sigma["2D"]
    viewer._on_noise(up=False)
    assert viewer.noise_sigma["2D"] < s0 or viewer.noise_sigma["2D"] == 0.0

def test_2d_sigma_caps_at_max(viewer):
    viewer._on_noise_target("2D")
    for _ in range(40):
        viewer._on_noise(up=True)
    assert viewer.noise_sigma["2D"] == NOISE_PARAMS["2D"]["max"]

def test_2d_sigma_floors_at_zero(viewer):
    viewer._on_noise_target("2D")
    for _ in range(40):
        viewer._on_noise(up=False)
    assert viewer.noise_sigma["2D"] == 0.0

def test_2d_sigma_restarts_at_min_after_zero(viewer):
    viewer._on_noise_target("2D")
    for _ in range(40):
        viewer._on_noise(up=False)
    assert viewer.noise_sigma["2D"] == 0.0
    viewer._on_noise(up=True)
    assert viewer.noise_sigma["2D"] == NOISE_PARAMS["2D"]["min"]

def test_2d_noise_up_modifies_coordinates_2d(viewer):
    viewer._on_noise_target("2D")
    before = json.loads(json.dumps(viewer.doc["Coordinates_2D"]))
    viewer._on_noise(up=True)
    after = viewer.doc["Coordinates_2D"]
    assert before != after

def test_2d_noise_only_modifies_coordinates_2d(viewer):
    viewer._on_noise_target("2D")
    before = json.loads(json.dumps(viewer.doc))
    viewer._on_noise(up=True)
    after = json.loads(json.dumps(viewer.doc))
    changed = [k for k in before if before[k] != after[k]]
    assert changed == ["Coordinates_2D"]

# ── Velocity noise ────────────────────────────────────────────────────────────

def test_velocity_sigma_caps_at_max(viewer):
    viewer._on_noise_target("Velocity")
    for _ in range(40):
        viewer._on_noise(up=True)
    assert viewer.noise_sigma["Velocity"] == NOISE_PARAMS["Velocity"]["max"]

def test_velocity_noise_preserves_unit_length(viewer):
    viewer._on_noise_target("Velocity")
    for _ in range(5):
        viewer._on_noise(up=True)
    Vd = viewer.doc["Velocity_vector_3D"]["Noisy_GT"]
    noisy_norm = np.linalg.norm([Vd["V_x"], Vd["V_y"], Vd["V_z"]])
    gt = viewer.velocity("GT")
    gt_norm = float(np.linalg.norm(gt))
    assert np.isclose(noisy_norm, gt_norm, atol=1e-9)

def test_velocity_noise_only_modifies_velocity_block(viewer):
    viewer._on_noise_target("Velocity")
    before = json.loads(json.dumps(viewer.doc))
    viewer._on_noise(up=True)
    after = json.loads(json.dumps(viewer.doc))
    changed = [k for k in before if before[k] != after[k]]
    assert changed == ["Velocity_vector_3D"]

def test_velocity_sigma_floors_at_zero(viewer):
    viewer._on_noise_target("Velocity")
    for _ in range(40):
        viewer._on_noise(up=False)
    assert viewer.noise_sigma["Velocity"] == 0.0

# ── Range noise ───────────────────────────────────────────────────────────────

def test_range_sigma_caps_at_max(viewer):
    viewer._on_noise_target("Range")
    for _ in range(40):
        viewer._on_noise(up=True)
    assert viewer.noise_sigma["Range"] == NOISE_PARAMS["Range"]["max"]

def test_range_noise_always_at_least_1(viewer):
    viewer._on_noise_target("Range")
    rng = np.random.default_rng(99)
    for _ in range(20):
        viewer._rng = rng
        viewer._on_noise(up=True)
        assert viewer.doc["Range_meters"]["Noisy_GT"] >= 1.0

def test_range_noise_only_modifies_range_block(viewer):
    viewer._on_noise_target("Range")
    before = json.loads(json.dumps(viewer.doc))
    viewer._on_noise(up=True)
    after = json.loads(json.dumps(viewer.doc))
    changed = [k for k in before if before[k] != after[k]]
    assert changed == ["Range_meters"]

def test_range_sigma_floors_at_zero(viewer):
    viewer._on_noise_target("Range")
    for _ in range(40):
        viewer._on_noise(up=False)
    assert viewer.noise_sigma["Range"] == 0.0

# ── noise target switching ────────────────────────────────────────────────────

def test_noise_target_switch_does_not_affect_sigma_of_other_targets(viewer):
    viewer._on_noise_target("2D")
    for _ in range(3):
        viewer._on_noise(up=True)
    sigma_2d_before = viewer.noise_sigma["2D"]
    sigma_vel_before = viewer.noise_sigma["Velocity"]
    viewer._on_noise_target("Velocity")
    for _ in range(3):
        viewer._on_noise(up=True)
    assert viewer.noise_sigma["2D"] == sigma_2d_before
    assert viewer.noise_sigma["Velocity"] > sigma_vel_before

# ── noise is reflected wherever the noisy element is visible ──────────────────
# 2D noise -> any view except "3D results"; Velocity/Range noise -> any view
# except "2D" and "3D results". Computed* changes only on an explicit Solve.

def test_noise_never_touches_computed(viewer):
    """Noise presses re-sample Noisy_GT only; no implicit re-solve."""
    from main import solve_document
    solve_document(viewer.doc, source="Noisy_GT")
    before = json.loads(json.dumps(viewer.doc["Coordinates_3D"]))
    for target in ["2D", "Velocity", "Range"]:
        viewer._on_noise_target(target)
        viewer._on_noise(up=True)
    after = viewer.doc["Coordinates_3D"]
    for el in before:
        for v in ["Computed1", "Computed2", "Computed3"]:
            assert before[el][v] == after[el][v]

def test_2d_noise_moves_scheme_plane_marker(viewer):
    """The on-plane Noisy_GT 2D marker tracks the stored pixels (scheme views)."""
    _, _, _, map2d = viewer._scheme_geometry()
    q0 = map2d("P1_tail", "Noisy_GT").copy()
    viewer._on_noise_target("2D")
    viewer._on_noise(up=True)
    _, _, _, map2d = viewer._scheme_geometry()
    q1 = map2d("P1_tail", "Noisy_GT")
    assert not np.allclose(q0, q1)

def test_plane_marker_uses_stored_pixels(viewer):
    """map2d maps the stored pixels linearly onto the virtual plane."""
    from viewer import SCHEME_FV, SCHEME_HALF_FOV
    cam = viewer.doc["Camera_parameters"]
    W, H = cam["W_pixels"], cam["H_pixels"]
    viewer.doc["Coordinates_2D"]["p1_tail"]["Noisy_GT"] = aio.vec2([W, H])
    _, _, _, map2d = viewer._scheme_geometry()
    q = map2d("P1_tail", "Noisy_GT")
    a = SCHEME_FV * np.tan(SCHEME_HALF_FOV)
    b = a * H / W
    assert np.allclose(q, [a, b, SCHEME_FV], atol=1e-12)

def test_all_checked_velocity_variants_drawn(viewer):
    """GT and Noisy_GT velocity arrows are both drawn when both are checked."""
    c = viewer._col_index("velocity", "3D")
    viewer.visible[(c, "GT")] = True
    viewer.visible[(c, "Noisy_GT")] = True
    assert viewer._visible_velocity_variants() == ["GT", "Noisy_GT"]

def test_range_label_lists_all_checked_variants(viewer):
    c = viewer._col_index("range", "scalar")
    viewer.visible[(c, "GT")] = True
    viewer.visible[(c, "Noisy_GT")] = True
    txt = viewer._range_label_text()
    assert "GT=" in txt and "N_GT=" in txt

def test_range_label_none_when_nothing_checked(viewer):
    c = viewer._col_index("range", "scalar")
    for v in aio.VARIANTS:
        viewer.visible[(c, v)] = False
    assert viewer._range_label_text() is None

def test_range_noise_changes_range_label(viewer):
    c = viewer._col_index("range", "scalar")
    viewer.visible[(c, "Noisy_GT")] = True
    txt0 = viewer._range_label_text()
    viewer._on_noise_target("Range")
    viewer._on_noise(up=True)
    assert viewer._range_label_text() != txt0

def test_scheme_views_render_after_each_noise_target(viewer):
    """Each noise press re-renders cleanly in every non-results view."""
    for view in ["2D", "Pinhole (scheme)", "3D interactive", "Change GT"]:
        viewer._on_view(view)
        for target in ["2D", "Velocity", "Range"]:
            viewer._on_noise_target(target)
            viewer._on_noise(up=True)   # must not raise
