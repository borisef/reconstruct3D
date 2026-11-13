"""Tests for main.solve_document — Computed* fill policy and correctness."""
import copy
import json
import os

import numpy as np
import pytest

import aircraft_io as aio
from main import solve_document

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


def _doc():
    return aio.load_json(INPUT)


def _xyz(d):
    return None if d is None else np.array([d["X"], d["Y"], d["Z"]])


def _err(doc, name, variant):
    gt = _xyz(doc["Coordinates_3D"][name]["GT"])
    est = _xyz(doc["Coordinates_3D"][name][variant])
    if gt is None or est is None:
        return None
    return float(np.linalg.norm(gt - est))

# ── GT source: Computed results should match GT closely ───────────────────────

def test_gt_source_body_p1_computed1_near_gt():
    doc = _doc()
    solve_document(doc, source="GT")
    assert _err(doc, "P1_tail", "Computed1") < 0.01

def test_gt_source_body_p2_computed1_near_gt():
    doc = _doc()
    solve_document(doc, source="GT")
    assert _err(doc, "P2_nose", "Computed1") < 0.01

def test_gt_source_body_p1_computed2_near_gt():
    doc = _doc()
    solve_document(doc, source="GT")
    assert _err(doc, "P1_tail", "Computed2") < 0.01

def test_gt_source_wings_computed1_near_gt():
    doc = _doc()
    solve_document(doc, source="GT")
    assert _err(doc, "P3_left_wing", "Computed1") < 0.01
    assert _err(doc, "P4_right_wing", "Computed1") < 0.01

# ── Noisy source: Computed results have nonzero error ─────────────────────────

def test_noisy_source_p1_computed1_has_error():
    doc = _doc()
    solve_document(doc, source="Noisy_GT")
    assert _err(doc, "P1_tail", "Computed1") > 0.1

# ── Computed* fill policy ─────────────────────────────────────────────────────

def test_body_points_get_computed1_and_computed2():
    doc = _doc()
    solve_document(doc, source="GT")
    for name in ["P1_tail", "P2_nose"]:
        assert doc["Coordinates_3D"][name]["Computed1"] is not None
        assert doc["Coordinates_3D"][name]["Computed2"] is not None

def test_wing_points_get_computed1_only():
    doc = _doc()
    solve_document(doc, source="GT")
    for name in ["P3_left_wing", "P4_right_wing"]:
        assert doc["Coordinates_3D"][name]["Computed1"] is not None
        assert doc["Coordinates_3D"][name]["Computed2"] is None

def test_all_computed3_stay_null():
    doc = _doc()
    solve_document(doc, source="GT")
    for name in aio.KEYPOINTS_3D:
        assert doc["Coordinates_3D"][name]["Computed3"] is None

def test_2d_computed_stay_null():
    doc = _doc()
    solve_document(doc, source="GT")
    for name in aio.KEYPOINTS_2D:
        for v in ["Computed1", "Computed2", "Computed3"]:
            assert doc["Coordinates_2D"][name][v] is None

# ── solve does not alter GT/Noisy_GT ─────────────────────────────────────────

def test_gt_values_preserved_after_solve():
    doc = _doc()
    gt_before = json.loads(json.dumps(
        {k: doc["Coordinates_3D"][k]["GT"] for k in aio.KEYPOINTS_3D}))
    solve_document(doc, source="GT")
    for k in aio.KEYPOINTS_3D:
        assert doc["Coordinates_3D"][k]["GT"] == gt_before[k]

def test_3d_keypoints_have_no_noisy_gt():
    """3D blocks carry no Noisy_GT: noisy inputs are 2D + velocity + range."""
    doc = _doc()
    solve_document(doc, source="GT")
    for k in aio.KEYPOINTS_3D:
        assert "Noisy_GT" not in doc["Coordinates_3D"][k]

# ── error on null input ───────────────────────────────────────────────────────

def test_null_2d_source_raises_value_error():
    doc = _doc()
    for name in aio.KEYPOINTS_2D:
        doc["Coordinates_2D"][name]["Noisy_GT"] = None
    with pytest.raises(ValueError):
        solve_document(doc, source="Noisy_GT")

# ── output schema completeness ────────────────────────────────────────────────

def test_output_has_all_top_level_blocks():
    doc = _doc()
    solve_document(doc, source="GT")
    for block in aio.BLOCKS:
        assert block in doc

def test_output_3d_keypoints_have_all_variants():
    doc = _doc()
    solve_document(doc, source="GT")
    for name in aio.KEYPOINTS_3D:
        assert set(doc["Coordinates_3D"][name]) == set(aio.VARIANTS_3D)

def test_output_2d_keypoints_have_all_variants():
    doc = _doc()
    solve_document(doc, source="GT")
    for name in aio.KEYPOINTS_2D:
        assert set(doc["Coordinates_2D"][name]) == set(aio.VARIANTS)
