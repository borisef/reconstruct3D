"""Tests for main.solve_document — params, single Computed solution, Results."""
import json
import os

import numpy as np
import pytest

import aircraft_io as aio
from main import (solve_document, algo_params, DEFAULT_ALGO_PARAMS,
                  HEAD_TAIL_OPTIONS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")

GT_PARAMS = {"input_noisy": False}


def _doc():
    return aio.load_json(INPUT)


def _xyz(d):
    return None if d is None else np.array([d["X"], d["Y"], d["Z"]])


def _err(doc, name):
    gt = _xyz(doc["Coordinates_3D"][name]["GT"])
    est = _xyz(doc["Coordinates_3D"][name].get("Computed"))
    if gt is None or est is None:
        return None
    return float(np.linalg.norm(gt - est))


# ── algorithm parameters ──────────────────────────────────────────────────────

def test_default_params():
    assert DEFAULT_ALGO_PARAMS == {"input_noisy": True,
                                   "head_tail_solution": "#1"}
    assert HEAD_TAIL_OPTIONS == ["#1", "#2", "Average", "Joint with wings"]

def test_algo_params_doc_block_wins_over_defaults():
    doc = {"Algorithm_parameters": {"input_noisy": False}}
    assert algo_params(doc)["input_noisy"] is False

def test_algo_params_override_wins_over_doc():
    doc = {"Algorithm_parameters": {"head_tail_solution": "#2"}}
    p = algo_params(doc, {"head_tail_solution": "Average"})
    assert p["head_tail_solution"] == "Average"

def test_algo_params_rejects_bad_head_tail():
    with pytest.raises(ValueError):
        algo_params(override={"head_tail_solution": "#3"})

def test_algo_params_ignores_unknown_keys():
    p = algo_params(override={"not_a_param": 1})
    assert "not_a_param" not in p

# ── GT source: Computed should match GT closely ───────────────────────────────

def test_gt_source_all_points_near_gt():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    for name in aio.KEYPOINTS_3D:
        assert _err(doc, name) < 0.01, name

def test_gt_source_head_tail_2_near_gt():
    doc = _doc()
    solve_document(doc, params={"input_noisy": False,
                                "head_tail_solution": "#2"})
    assert _err(doc, "P1_tail") < 0.01
    assert _err(doc, "P2_nose") < 0.01

# ── Noisy source (the default): Computed has nonzero error ────────────────────

def test_default_source_is_noisy_and_has_error():
    doc = _doc()
    solve_document(doc)
    assert doc["Algorithm_parameters"]["input_noisy"] is True
    assert doc["Results"]["source"] == "Noisy_GT"
    assert _err(doc, "P1_tail") > 0.1

# ── head/tail solution selection ──────────────────────────────────────────────

def test_head_tail_solutions_differ_under_noise():
    d1, d2 = _doc(), _doc()
    solve_document(d1, params={"head_tail_solution": "#1"})
    solve_document(d2, params={"head_tail_solution": "#2"})
    p1a = _xyz(d1["Coordinates_3D"]["P1_tail"]["Computed"])
    p1b = _xyz(d2["Coordinates_3D"]["P1_tail"]["Computed"])
    assert np.linalg.norm(p1a - p1b) > 1e-6

def test_average_is_midpoint_of_1_and_2():
    docs = {c: _doc() for c in HEAD_TAIL_OPTIONS}
    for c, d in docs.items():
        solve_document(d, params={"head_tail_solution": c})
    for name in ["P1_tail", "P2_nose"]:
        a = _xyz(docs["#1"]["Coordinates_3D"][name]["Computed"])
        b = _xyz(docs["#2"]["Coordinates_3D"][name]["Computed"])
        avg = _xyz(docs["Average"]["Coordinates_3D"][name]["Computed"])
        assert np.allclose(avg, 0.5 * (a + b), atol=1e-9)

def test_wings_independent_of_head_tail_choice():
    d1, d2 = _doc(), _doc()
    solve_document(d1, params={"head_tail_solution": "#1"})
    solve_document(d2, params={"head_tail_solution": "#2"})
    for name in ["P3_left_wing", "P4_right_wing"]:
        assert np.allclose(_xyz(d1["Coordinates_3D"][name]["Computed"]),
                           _xyz(d2["Coordinates_3D"][name]["Computed"]))

# ── joint solution option ─────────────────────────────────────────────────────

JOINT = {"input_noisy": False, "head_tail_solution": "Joint with wings"}

def test_joint_is_valid_option():
    p = algo_params(override={"head_tail_solution": "Joint with wings"})
    assert p["head_tail_solution"] == "Joint with wings"

def test_joint_gt_solve_recovers_all_points():
    doc = _doc()
    solve_document(doc, params=JOINT)
    for name in aio.KEYPOINTS_3D:
        assert _err(doc, name) < 0.01, name

def test_joint_fills_all_computed_3d_and_2d():
    doc = _doc()
    solve_document(doc, params=JOINT)
    for name in aio.KEYPOINTS_3D:
        assert doc["Coordinates_3D"][name]["Computed"] is not None
    for name in aio.KEYPOINTS_2D:
        assert doc["Coordinates_2D"][name]["Computed"] is not None

def test_joint_params_persisted():
    doc = _doc()
    solve_document(doc, params=JOINT)
    assert doc["Algorithm_parameters"]["head_tail_solution"] == "Joint with wings"
    assert doc["Results"]["source"] == "GT"

def test_joint_noisy_solve_has_error_but_is_reasonable():
    doc = _doc()
    solve_document(doc, params={"input_noisy": True,
                                "head_tail_solution": "Joint with wings"})
    # nonzero (noisy input) but the aircraft is still recovered to ~metres
    errs = [_err(doc, name) for name in aio.KEYPOINTS_3D]
    assert max(errs) > 0.0
    assert max(errs) < 10.0

# ── fill policy ───────────────────────────────────────────────────────────────

def test_all_3d_points_get_computed():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    for name in aio.KEYPOINTS_3D:
        assert doc["Coordinates_3D"][name]["Computed"] is not None

def test_2d_computed_is_reprojection_of_3d():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    fx, fy, cx, cy = aio.camera_intrinsics(doc["Camera_parameters"])
    for el3d, el2d in zip(aio.KEYPOINTS_3D, aio.KEYPOINTS_2D):
        P = _xyz(doc["Coordinates_3D"][el3d]["Computed"])
        px = doc["Coordinates_2D"][el2d]["Computed"]
        assert px is not None
        assert np.allclose(aio.project_pixel(P, fx, fy, cx, cy),
                           [px["u"], px["v"]], atol=1e-9)

def test_2d_computed_near_gt_pixels_for_gt_solve():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    for name in aio.KEYPOINTS_2D:
        px = doc["Coordinates_2D"][name]["Computed"]
        gt = doc["Coordinates_2D"][name]["GT"]
        assert abs(px["u"] - gt["u"]) < 0.1
        assert abs(px["v"] - gt["v"]) < 0.1

# ── solve does not alter GT/Noisy_GT ─────────────────────────────────────────

def test_gt_values_preserved_after_solve():
    doc = _doc()
    gt_before = json.loads(json.dumps(
        {k: doc["Coordinates_3D"][k]["GT"] for k in aio.KEYPOINTS_3D}))
    solve_document(doc, params=GT_PARAMS)
    for k in aio.KEYPOINTS_3D:
        assert doc["Coordinates_3D"][k]["GT"] == gt_before[k]

def test_3d_keypoints_have_no_noisy_gt():
    """3D blocks carry no Noisy_GT: noisy inputs are 2D + velocity + range."""
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    for k in aio.KEYPOINTS_3D:
        assert "Noisy_GT" not in doc["Coordinates_3D"][k]

# ── error on null input ───────────────────────────────────────────────────────

def test_null_2d_source_raises_value_error():
    doc = _doc()
    for name in aio.KEYPOINTS_2D:
        doc["Coordinates_2D"][name]["Noisy_GT"] = None
    with pytest.raises(ValueError):
        solve_document(doc)   # default input is Noisy_GT

# ── output schema completeness ────────────────────────────────────────────────

def test_output_has_all_top_level_blocks():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    for block in aio.BLOCKS + aio.OPTIONAL_BLOCKS:
        assert block in doc

def test_output_keypoints_have_all_variants():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    for name in aio.KEYPOINTS_3D:
        assert set(doc["Coordinates_3D"][name]) == set(aio.VARIANTS_3D)
    for name in aio.KEYPOINTS_2D:
        assert set(doc["Coordinates_2D"][name]) == set(aio.VARIANTS)

def test_output_algorithm_parameters_block():
    doc = _doc()
    solve_document(doc, params={"input_noisy": False,
                                "head_tail_solution": "Average"})
    assert doc["Algorithm_parameters"] == {"input_noisy": False,
                                           "head_tail_solution": "Average"}

def test_doc_params_reused_on_next_solve():
    # A saved document carries its parameters; re-solving honours them.
    doc = _doc()
    solve_document(doc, params={"input_noisy": False,
                                "head_tail_solution": "#2"})
    solve_document(doc)   # no explicit params
    assert doc["Algorithm_parameters"]["head_tail_solution"] == "#2"
    assert doc["Results"]["source"] == "GT"

# ── Results block ─────────────────────────────────────────────────────────────

def test_results_errors_2d_structure():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    errs = doc["Results"]["errors_2D_px"]
    assert set(errs) == set(aio.KEYPOINTS_2D) | {"mean"}
    # Solutions lie on the source rays: reprojection error vs GT is tiny.
    assert errs["mean"] < 0.1

def test_results_table_3d_structure():
    doc = _doc()
    solve_document(doc, params=GT_PARAMS)
    table = doc["Results"]["table_3D"]
    assert set(table) == {"GT", "Computed"}
    for col in table.values():
        assert set(col) == {"XYZ_m", "YPR_deg", "BodyLength_m",
                            "WingsSpan_m", "CrossPoint"}
    assert table["GT"]["BodyLength_m"] == pytest.approx(30.0, abs=1e-6)
    assert table["GT"]["CrossPoint"] == pytest.approx(0.5, abs=1e-9)
    assert table["Computed"]["BodyLength_m"] == pytest.approx(30.0, abs=0.05)

def test_results_json_serializable():
    doc = _doc()
    solve_document(doc)
    json.dumps(doc)   # must not raise (no numpy types)
