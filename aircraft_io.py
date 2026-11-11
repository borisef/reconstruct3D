"""
aircraft_io.py

JSON I/O helpers for the aircraft 3D-2D reconstruction.

The algorithm is driven by a single JSON document whose shape is fixed:

    Camera_parameters  : fov_X_deg, fov_Y_deg, H_pixels, W_pixels
    Coordinates_3D     : P1_tail, P2_nose, P3_left_wing, P4_right_wing
    Coordinates_2D     : p1_tail, p2_nose, p3_left_wing, p4_right_wing   (pixels)
    Velocity_vector_3D : the body/velocity direction
    Range_meters       : distance from the camera to the body midpoint (scalar)

Every keypoint / vector / range carries up to five value *variants*:

    GT, Noisy_GT, Computed1, Computed2, Computed3

any of which may be ``null``. 3D values are ``{"X","Y","Z"}``, 2D values are
``{"u","v"}`` in pixels, velocity values are ``{"V_x","V_y","V_z"}`` and the
range is a plain number.

These helpers are schema-aware builders + a camera model (pixel <-> normalized),
shared by the input generator and the later solve-and-write step.
"""

import json

import numpy as np


# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
KEYPOINTS_3D = ["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"]
KEYPOINTS_2D = ["p1_tail", "p2_nose", "p3_left_wing", "p4_right_wing"]
VARIANTS = ["GT", "Noisy_GT", "Computed1", "Computed2", "Computed3"]

BLOCKS = ["Camera_parameters", "Coordinates_3D", "Coordinates_2D",
          "Velocity_vector_3D", "Range_meters"]


# --------------------------------------------------------------------------- #
# Value builders
# --------------------------------------------------------------------------- #
def vec3(arr):
    """3D point dict {"X","Y","Z"} from a length-3 array (or None -> None)."""
    if arr is None:
        return None
    x, y, z = (float(c) for c in arr)
    return {"X": x, "Y": y, "Z": z}


def vec2(arr):
    """2D pixel dict {"u","v"} from a length-2 array (or None -> None)."""
    if arr is None:
        return None
    u, v = (float(c) for c in arr)
    return {"u": u, "v": v}


def vel(arr):
    """Velocity dict {"V_x","V_y","V_z"} from a length-3 array (or None -> None)."""
    if arr is None:
        return None
    vx, vy, vz = (float(c) for c in arr)
    return {"V_x": vx, "V_y": vy, "V_z": vz}


def scalar(value):
    """Scalar variant (e.g. range in meters); passes None through."""
    return None if value is None else float(value)


def variant_block(**vals):
    """A full 5-variant block; variants not supplied default to ``None``.

    Example: ``variant_block(GT=vec3(P), Noisy_GT=vec3(Pn))`` yields a dict with
    all of GT/Noisy_GT/Computed1/Computed2/Computed3 present, the latter three
    ``None``.
    """
    return {name: vals.get(name, None) for name in VARIANTS}


# --------------------------------------------------------------------------- #
# Camera model
# --------------------------------------------------------------------------- #
def camera_intrinsics(cam):
    """Return (fx, fy, cx, cy) in pixels from a Camera_parameters dict.

    fx = W / (2 tan(fovX/2)),  fy = H / (2 tan(fovY/2)),  cx = W/2, cy = H/2.
    """
    W = cam["W_pixels"]
    H = cam["H_pixels"]
    fovx = np.deg2rad(cam["fov_X_deg"])
    fovy = np.deg2rad(cam["fov_Y_deg"])
    fx = W / (2.0 * np.tan(fovx / 2.0))
    fy = H / (2.0 * np.tan(fovy / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


def project_pixel(P, fx, fy, cx, cy):
    """Pinhole-project a 3D point to pixel coordinates (u, v)."""
    X, Y, Z = P
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.array([u, v])


def pixel_to_normalized(px, fx, fy, cx, cy):
    """Inverse of project_pixel: pixel (u, v) -> normalized (u, v)."""
    u_px, v_px = px
    return np.array([(u_px - cx) / fx, (v_px - cy) / fy])


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
