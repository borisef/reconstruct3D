"""
main.py

Example driver for the JSON-based aircraft 3D-2D reconstruction.

Usage:
    python main.py [INPUT_JSON] [OUTPUT_JSON] [--source VARIANT]

Reads an input document (see aircraft_io.py for the schema), solves for the 3D
keypoints from the 2D pixel measurements + velocity + range, and writes an
output document in the *exact same format*, extended with the Computed variants:

    Coordinates_3D.P1_tail / P2_nose        : Computed1 = body_solution1
                                              Computed2 = body_solution2
    Coordinates_3D.P3_left_wing / P4_right  : Computed1 = wings_solution

Everything else (Camera_parameters, GT values, the 2D block, velocity and
range, incl. their Noisy_GT) is carried through unchanged. 3D keypoints carry
no Noisy_GT -- the noisy inputs are 2D + velocity + range. 2D Computed* and
all Computed3 stay null for now.

Defaults: INPUT_JSON = inputs/aircraft_input.json,
          OUTPUT_JSON = outputs/aircraft_output.json,
          --source    = Noisy_GT  (which input variant feeds the solvers).
"""

import argparse
import os

import numpy as np

import aircraft_io as aio
import reconstruction_utils as ru


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "inputs", "aircraft_input.json")
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "aircraft_output.json")


def _normalized_2d(doc, name, source, fx, fy, cx, cy):
    """Normalized (u, v) of a 2D keypoint variant (stored in pixels)."""
    px = doc["Coordinates_2D"][name][source]
    if px is None:
        raise ValueError(f"Coordinates_2D.{name}.{source} is null; "
                         f"cannot solve from variant '{source}'.")
    return aio.pixel_to_normalized([px["u"], px["v"]], fx, fy, cx, cy)


def solve_document(doc, source="Noisy_GT"):
    """Fill the Computed* 3D variants in ``doc`` (mutated in place) and return it.

    Solves from the 2D pixel keypoints, velocity vector and range of the chosen
    input ``source`` variant.
    """
    fx, fy, cx, cy = aio.camera_intrinsics(doc["Camera_parameters"])

    (u1, v1) = _normalized_2d(doc, "p1_tail", source, fx, fy, cx, cy)
    (u2, v2) = _normalized_2d(doc, "p2_nose", source, fx, fy, cx, cy)
    (u3, v3) = _normalized_2d(doc, "p3_left_wing", source, fx, fy, cx, cy)
    (u4, v4) = _normalized_2d(doc, "p4_right_wing", source, fx, fy, cx, cy)

    Vd = doc["Velocity_vector_3D"][source]
    V = np.array([Vd["V_x"], Vd["V_y"], Vd["V_z"]])
    D = doc["Range_meters"][source]

    # Body: both solutions. (Wing midpoint == body midpoint == aircraft centre,
    # so the single body range D serves the wings too.)
    P1_s1, P2_s1 = ru.body_solution1(u1, v1, u2, v2, V, D)
    P1_s2, P2_s2 = ru.body_solution2(u1, v1, u2, v2, V, D)
    P3, P4 = ru.wings_solution(u3, v3, u4, v4, V, D)

    c3d = doc["Coordinates_3D"]
    c3d["P1_tail"]["Computed1"] = aio.vec3(P1_s1)
    c3d["P1_tail"]["Computed2"] = aio.vec3(P1_s2)
    c3d["P2_nose"]["Computed1"] = aio.vec3(P2_s1)
    c3d["P2_nose"]["Computed2"] = aio.vec3(P2_s2)
    c3d["P3_left_wing"]["Computed1"] = aio.vec3(P3)
    c3d["P4_right_wing"]["Computed1"] = aio.vec3(P4)
    return doc


def _xyz(d):
    return None if d is None else np.array([d["X"], d["Y"], d["Z"]])


def report(doc):
    """Print Computed-vs-GT errors for a quick sanity check."""
    np.set_printoptions(precision=3, suppress=True)
    print("Computed vs GT (|err| in metres):")
    rows = [
        ("P1_tail", "Computed1"), ("P1_tail", "Computed2"),
        ("P2_nose", "Computed1"), ("P2_nose", "Computed2"),
        ("P3_left_wing", "Computed1"), ("P4_right_wing", "Computed1"),
    ]
    for name, variant in rows:
        entry = doc["Coordinates_3D"][name]
        gt = _xyz(entry["GT"])
        est = _xyz(entry[variant])
        if gt is None or est is None:
            continue
        err = np.linalg.norm(gt - est)
        print(f"  {name:<14} {variant}: est={est}  |err|={err:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                        help="input JSON path")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                        help="output JSON path")
    parser.add_argument("--source", default="Noisy_GT",
                        choices=aio.VARIANTS,
                        help="input variant the solvers read (default Noisy_GT)")
    args = parser.parse_args()

    doc = aio.load_json(args.input)
    solve_document(doc, source=args.source)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    aio.save_json(args.output, doc)

    print(f"Solved '{args.input}' (source={args.source})")
    print(f"Wrote  '{args.output}'")
    report(doc)


if __name__ == "__main__":
    main()
