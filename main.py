"""
main.py

Solver / algorithm library for the JSON-based aircraft 3D-2D reconstruction,
plus a small command-line solve driver. This module holds the core reconstruction
API (``solve_document``, ``algo_params``, ``aircraft_stats``, ``results_block``,
``HEAD_TAIL_OPTIONS`` ...) and is imported as ``algo`` by the GUI and the
comparison tool. The project's entry points are:

    main.py                     -- this file: solve one JSON from the CLI (below)
    main_GUI.py                 -- launch the interactive matplotlib viewer
    main_compare_algorithms.py  -- run the Monte-Carlo algorithm comparison

Usage (solve CLI):
    python main.py [INPUT_JSON] [OUTPUT_JSON] [--gt] [--head-tail CHOICE]

Reads an input document (see aircraft_io.py for the schema), solves for the 3D
keypoints from the 2D pixel measurements + velocity + range, and writes an
output document in the *exact same format*, extended with:

    Coordinates_3D.*.Computed : the ONE solution of this run
    Coordinates_2D.*.Computed : that solution reprojected to pixels
    Algorithm_parameters      : the parameters the solver ran with
    Results                   : 2D reprojection-error + 3D summary tables

Algorithm parameters (defaults in DEFAULT_ALGO_PARAMS, overridable per call
and persisted in the document):

    input_noisy        : True -> solve from Noisy_GT (2D + velocity + range),
                         False -> solve from GT.
    head_tail_solution : "#1" | "#2" | "Average" | "Joint with wings".
                         The body (head/tail) problem has a two-fold
                         ambiguity; "#1"/"#2" pick which closed-form solution
                         is reported and "Average" is the midpoint of both
                         (wings always from the separate wings solver).
                         "Joint with wings" instead fits all four keypoints in
                         one least-squares problem (paper Part II): body and
                         wings share one pose and the wing attachment point is
                         estimated rather than assumed at the body midpoint.

Everything else (Camera_parameters, GT values, the 2D GT/Noisy_GT pixels,
velocity and range) is carried through unchanged. 3D keypoints carry no
Noisy_GT -- the noisy inputs are 2D + velocity + range.

Defaults: INPUT_JSON = inputs/aircraft_input.json,
          OUTPUT_JSON = outputs/aircraft_output.json.
"""

import argparse
import os

import numpy as np

import aircraft_io as aio
import reconstruction_utils as ru


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "inputs", "aircraft_input.json")
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "aircraft_output.json")

HEAD_TAIL_OPTIONS = ["#1", "#2", "Average", "Joint with wings"]
# Which velocity vector feeds the solvers. "corrected_by_2D_vel_vec" replaces
# the noisy velocity's image-plane azimuth with the clean 2D tail->nose
# direction (keeping its depth tilt); see ru.correct_velocity_with_2d.
VELOCITY_OPTIONS = ["vel_vec", "corrected_by_2D_vel_vec"]
DEFAULT_ALGO_PARAMS = {
    "input_noisy": True,              # solve from Noisy_GT (else GT)
    "head_tail_solution": "#1",       # see HEAD_TAIL_OPTIONS
    "velocity_source": "vel_vec",     # see VELOCITY_OPTIONS
}

# Joint-solution measurement weights (head_tail_solution == "Joint with
# wings"). Only the ratio matters for the estimate; sig_uv is derived from the
# camera focal length so it is comparable to sig_v across cameras.
JOINT_SIGMA_PX = 1.0       # assumed keypoint noise, pixels
JOINT_SIGMA_V_DEG = 1.0    # assumed velocity-direction noise, degrees


def algo_params(doc=None, override=None):
    """Effective algorithm parameters: defaults <- doc block <- override.

    Unknown keys are ignored so documents stay forward-compatible; an invalid
    head_tail_solution raises ValueError.
    """
    p = dict(DEFAULT_ALGO_PARAMS)
    for src in ((doc or {}).get("Algorithm_parameters") or {}, override or {}):
        p.update({k: v for k, v in src.items() if k in p})
    if p["head_tail_solution"] not in HEAD_TAIL_OPTIONS:
        raise ValueError(f"head_tail_solution must be one of "
                         f"{HEAD_TAIL_OPTIONS}, got {p['head_tail_solution']!r}")
    if p["velocity_source"] not in VELOCITY_OPTIONS:
        raise ValueError(f"velocity_source must be one of "
                         f"{VELOCITY_OPTIONS}, got {p['velocity_source']!r}")
    p["input_noisy"] = bool(p["input_noisy"])
    return p


def _normalized_2d(doc, name, source, fx, fy, cx, cy):
    """Normalized (u, v) of a 2D keypoint variant (stored in pixels)."""
    px = doc["Coordinates_2D"][name][source]
    if px is None:
        raise ValueError(f"Coordinates_2D.{name}.{source} is null; "
                         f"cannot solve from variant '{source}'.")
    return aio.pixel_to_normalized([px["u"], px["v"]], fx, fy, cx, cy)


def corrected_velocity(doc, source):
    """The 2D-corrected velocity for ``source`` (ndarray shape (3,)), or None.

    Replaces the source velocity's image-plane azimuth with the clean 2D
    tail->nose direction, keeping the depth tilt (see
    ``ru.correct_velocity_with_2d``). Returns None when the source velocity or
    the tail/nose 2D pixels are missing -- so callers (the GUI) can show it live
    without a solve. This is the single definition ``solve_document`` also uses
    for the Computed velocity slot.
    """
    Vd = doc["Velocity_vector_3D"].get(source)
    p1 = doc["Coordinates_2D"]["p1_tail"].get(source)
    p2 = doc["Coordinates_2D"]["p2_nose"].get(source)
    if Vd is None or p1 is None or p2 is None:
        return None
    fx, fy, cx, cy = aio.camera_intrinsics(doc["Camera_parameters"])
    (u1, v1) = aio.pixel_to_normalized([p1["u"], p1["v"]], fx, fy, cx, cy)
    (u2, v2) = aio.pixel_to_normalized([p2["u"], p2["v"]], fx, fy, cx, cy)
    V = np.array([Vd["V_x"], Vd["V_y"], Vd["V_z"]])
    return ru.correct_velocity_with_2d(V, u2 - u1, v2 - v1)


def solve_document(doc, params=None):
    """Fill the Computed 3D variants in ``doc`` (mutated in place); return it.

    Solves from the 2D pixel keypoints, velocity vector and range of the
    input variant selected by the algorithm parameters (see ``algo_params``:
    defaults, the document's own Algorithm_parameters block, then ``params``).
    Exactly ONE solution is stored per keypoint; the body keypoints use the
    head/tail solution the parameters select. The Computed 2D pixels (the
    solution reprojected through the camera), the effective parameters and
    the Results tables are written into the document as well.
    """
    p = algo_params(doc, params)
    source = "Noisy_GT" if p["input_noisy"] else "GT"
    fx, fy, cx, cy = aio.camera_intrinsics(doc["Camera_parameters"])

    (u1, v1) = _normalized_2d(doc, "p1_tail", source, fx, fy, cx, cy)
    (u2, v2) = _normalized_2d(doc, "p2_nose", source, fx, fy, cx, cy)
    (u3, v3) = _normalized_2d(doc, "p3_left_wing", source, fx, fy, cx, cy)
    (u4, v4) = _normalized_2d(doc, "p4_right_wing", source, fx, fy, cx, cy)

    Vd = doc["Velocity_vector_3D"][source]
    V = np.array([Vd["V_x"], Vd["V_y"], Vd["V_z"]])
    D = doc["Range_meters"][source]

    # Velocity corrected by the clean 2D tail->nose azimuth (keeps the noisy
    # velocity's depth tilt). Always stored in the Computed velocity slot so the
    # viewer can overlay it; the solvers consume it only when selected.
    V_corr = corrected_velocity(doc, source)
    doc["Velocity_vector_3D"]["Computed"] = aio.vel(V_corr)
    if p["velocity_source"] == "corrected_by_2D_vel_vec":
        V = V_corr

    choice = p["head_tail_solution"]
    if choice == "Joint with wings":
        # One least-squares fit of all four keypoints (paper Part II).
        sig_uv = JOINT_SIGMA_PX / fx
        sig_v = np.deg2rad(JOINT_SIGMA_V_DEG)
        P1, P2, P3, P4 = ru.joint_solution(
            u1, v1, u2, v2, u3, v3, u4, v4, V, D, sig_uv=sig_uv, sig_v=sig_v)
    else:
        # Body: the head/tail ambiguity gives two closed-form solutions;
        # report the chosen one. (Wing midpoint == body midpoint == aircraft
        # centre, so the single body range D serves the wings too.)
        P1_s1, P2_s1 = ru.body_solution1(u1, v1, u2, v2, V, D)
        P1_s2, P2_s2 = ru.body_solution2(u1, v1, u2, v2, V, D)
        if choice == "#1":
            P1, P2 = P1_s1, P2_s1
        elif choice == "#2":
            P1, P2 = P1_s2, P2_s2
        else:  # "Average"
            P1, P2 = 0.5 * (P1_s1 + P1_s2), 0.5 * (P2_s1 + P2_s2)
        P3, P4 = ru.wings_solution(u3, v3, u4, v4, V, D)

    c3d = doc["Coordinates_3D"]
    c3d["P1_tail"]["Computed"] = aio.vec3(P1)
    c3d["P2_nose"]["Computed"] = aio.vec3(P2)
    c3d["P3_left_wing"]["Computed"] = aio.vec3(P3)
    c3d["P4_right_wing"]["Computed"] = aio.vec3(P4)

    # Computed 2D keypoints: the solution reprojected to pixels, so the
    # views (and saved documents) carry computed p1..p4 alongside GT/Noisy.
    c2d = doc["Coordinates_2D"]
    for el3d, el2d in zip(aio.KEYPOINTS_3D, aio.KEYPOINTS_2D):
        P = _xyz(c3d[el3d]["Computed"])
        c2d[el2d]["Computed"] = \
            aio.vec2(aio.project_pixel(P, fx, fy, cx, cy)) \
            if P is not None and P[2] > 1e-9 else None

    doc["Algorithm_parameters"] = dict(p)
    doc["Results"] = results_block(doc, source)
    return doc


def _xyz(d):
    return None if d is None else np.array([d["X"], d["Y"], d["Z"]])


# --------------------------------------------------------------------------- #
# Results (shared with the GUI's "2D results" / "3D results" views)
# --------------------------------------------------------------------------- #
def aircraft_stats(doc, variant):
    """Summary metrics of one variant's aircraft, or None without a body.

    Returns {"xyz": centre (m), "body_len", and -- when the wing points exist
    -- "wing_span", "cross" (position of the wings crossing along the body:
    0 = tail, 1 = nose, clamped to [0, 1]) and "ypr" (yaw/pitch/roll, degrees,
    from ru.compute_6dof)}.
    """
    c3d = doc["Coordinates_3D"]
    P1 = _xyz(c3d["P1_tail"].get(variant))
    P2 = _xyz(c3d["P2_nose"].get(variant))
    if P1 is None or P2 is None:
        return None
    P3 = _xyz(c3d["P3_left_wing"].get(variant))
    P4 = _xyz(c3d["P4_right_wing"].get(variant))
    out = {"xyz": 0.5 * (P1 + P2),
           "body_len": float(np.linalg.norm(P2 - P1))}
    if P3 is not None and P4 is not None:
        out["wing_span"] = float(np.linalg.norm(P4 - P3))
        body = P2 - P1
        t = float((0.5 * (P3 + P4) - P1) @ body / (body @ body))
        out["cross"] = min(1.0, max(0.0, t))
        pose = ru.compute_6dof(P1, P2, P3, P4)
        out["ypr"] = np.degrees([pose["yaw"], pose["pitch"], pose["roll"]])
    return out


def reprojection_errors_px(doc, source):
    """{p1_tail: |err|, ..., "mean": m} in pixels, or None without data.

    Error of each Computed 3D keypoint, reprojected through the camera,
    against the stored 2D pixels of ``source`` (the measurements that fed the
    solver) -- the table of the GUI's "2D results" view.
    """
    fx, fy, cx, cy = aio.camera_intrinsics(doc["Camera_parameters"])
    errs = {}
    for el3d, el2d in zip(aio.KEYPOINTS_3D, aio.KEYPOINTS_2D):
        P = _xyz(doc["Coordinates_3D"][el3d].get("Computed"))
        ref = doc["Coordinates_2D"][el2d][source]
        if P is None or P[2] <= 1e-9 or ref is None:
            continue
        proj = aio.project_pixel(P, fx, fy, cx, cy)
        errs[el2d] = float(np.linalg.norm(proj - [ref["u"], ref["v"]]))
    if not errs:
        return None
    errs["mean"] = float(np.mean(list(errs.values())))
    return errs


def keypoint_errors_3d(doc):
    """{P1_tail: |GT-Computed|, ..., "mean": m} in metres, or None without data.

    Euclidean distance between each GT and Computed 3D keypoint -- the mean is
    the single reconstruction-error number shown in the GUI's "3D results" view.
    """
    c3d = doc["Coordinates_3D"]
    errs = {}
    for el3d in aio.KEYPOINTS_3D:
        gt = _xyz(c3d[el3d].get("GT"))
        est = _xyz(c3d[el3d].get("Computed"))
        if gt is None or est is None:
            continue
        errs[el3d] = float(np.linalg.norm(gt - est))
    if not errs:
        return None
    errs["mean"] = float(np.mean(list(errs.values())))
    return errs


def _stats_json(s):
    """JSON-ready form of ``aircraft_stats`` (the 3D-results table row)."""
    if s is None:
        return None
    out = {"XYZ_m": [float(c) for c in s["xyz"]],
           "BodyLength_m": s["body_len"]}
    if "ypr" in s:
        out["YPR_deg"] = [float(c) for c in s["ypr"]]
    if "wing_span" in s:
        out["WingsSpan_m"] = s["wing_span"]
    if "cross" in s:
        out["CrossPoint"] = s["cross"]
    return out


def results_block(doc, source):
    """The Results block: the 2D- and 3D-results-view tables as JSON."""
    err3d = keypoint_errors_3d(doc)
    return {
        "source": source,
        "errors_2D_px": reprojection_errors_px(doc, source),
        "errors_3D_m": err3d,
        "mean_error_3D_m": None if err3d is None else err3d["mean"],
        "table_3D": {v: _stats_json(aircraft_stats(doc, v))
                     for v in ("GT", "Computed")},
    }


def report(doc):
    """Print Computed-vs-GT errors for a quick sanity check."""
    np.set_printoptions(precision=3, suppress=True)
    print("Computed vs GT (|err| in metres):")
    for name in aio.KEYPOINTS_3D:
        entry = doc["Coordinates_3D"][name]
        gt, est = _xyz(entry["GT"]), _xyz(entry.get("Computed"))
        if gt is None or est is None:
            continue
        err = np.linalg.norm(gt - est)
        print(f"  {name:<14} est={est}  |err|={err:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                        help="input JSON path")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                        help="output JSON path")
    parser.add_argument("--gt", action="store_true",
                        help="solve from GT instead of Noisy_GT")
    parser.add_argument("--head-tail", default=None,
                        choices=HEAD_TAIL_OPTIONS,
                        help="which body (head/tail) solution to report")
    args = parser.parse_args()

    override = {}
    if args.gt:
        override["input_noisy"] = False
    if args.head_tail is not None:
        override["head_tail_solution"] = args.head_tail

    doc = aio.load_json(args.input)
    solve_document(doc, params=override or None)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    aio.save_json(args.output, doc)

    p = doc["Algorithm_parameters"]
    print(f"Solved '{args.input}' "
          f"(input={'Noisy_GT' if p['input_noisy'] else 'GT'}, "
          f"head/tail {p['head_tail_solution']})")
    print(f"Wrote  '{args.output}'")
    report(doc)


if __name__ == "__main__":
    main()
