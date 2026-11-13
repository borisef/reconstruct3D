"""
aircraft_frame.py

Local (body-fixed) aircraft coordinate frame.

Convention (right-handed):

    X : from left wing tip to right wing tip
    Y : from tail to nose (the body / flight direction)
    Z : "up" = X x Y, perpendicular to both body and wings -- a pilot sitting
        in the aircraft has +Z through their head

The origin is the aircraft centre (the body midpoint, which by construction is
also the wing midpoint).

The frame is always derived from the *ground-truth* 3D keypoints: GT is the
internal reference, and every other variant (Computed*) can be expressed in
the same GT-anchored local coordinates for comparison (3D keypoints carry no
Noisy_GT -- the noisy inputs are 2D + velocity + range only).

With the synthetic geometry (body half-length b, wing half-span w) the GT
keypoints land on the axes:

    P1_tail       = ( 0, -b, 0)      P2_nose       = (0, +b, 0)
    P3_left_wing  = (-w,  0, 0)      P4_right_wing = (+w, 0, 0)

Representation: an ``AircraftFrame`` holds ``origin`` (3,) and ``R`` (3x3),
whose *columns* are the local X/Y/Z axes expressed in camera coordinates, so

    P_camera = origin + R @ p_local        p_local = R.T @ (P_camera - origin)
"""

import numpy as np


def _normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class AircraftFrame:
    """GT-anchored local aircraft frame (origin + rotation to camera coords)."""

    def __init__(self, origin, R):
        self.origin = np.asarray(origin, dtype=float)
        self.R = np.asarray(R, dtype=float)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_points(cls, P1_tail, P2_nose, P3_left, P4_right):
        """Build the frame from four 3D keypoints (camera coordinates).

        Y is taken exactly along the body (tail -> nose); the wing direction is
        re-orthogonalized against it so R is orthonormal even if the input
        points are noisy: Z = normalize(X_raw x Y), then X = Y x Z.
        """
        P1 = np.asarray(P1_tail, dtype=float)
        P2 = np.asarray(P2_nose, dtype=float)
        P3 = np.asarray(P3_left, dtype=float)
        P4 = np.asarray(P4_right, dtype=float)

        Y = _normalize(P2 - P1)              # tail -> nose
        X_raw = _normalize(P4 - P3)          # left wing -> right wing
        Z = _normalize(np.cross(X_raw, Y))   # up (pilot's head)
        X = np.cross(Y, Z)                   # exactly orthogonal wing axis

        origin = 0.5 * (P1 + P2)             # aircraft centre
        R = np.column_stack([X, Y, Z])
        return cls(origin, R)

    @classmethod
    def from_doc(cls, doc, variant="GT"):
        """Build the frame from a JSON document's 3D keypoints.

        ``variant`` defaults to (and should normally stay) "GT": the local
        frame is the internal GT-anchored reference.
        """
        c3d = doc["Coordinates_3D"]
        pts = []
        for name in ["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"]:
            d = c3d[name].get(variant)   # 3D blocks carry no Noisy_GT key
            if d is None:
                raise ValueError(f"Coordinates_3D.{name}.{variant} is null; "
                                 f"cannot build the aircraft frame.")
            pts.append([d["X"], d["Y"], d["Z"]])
        return cls.from_points(*pts)

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #
    def to_local(self, P):
        """Camera-frame point(s) -> local aircraft coordinates.

        Accepts a (3,) point or an (N, 3) array of points.
        """
        P = np.asarray(P, dtype=float)
        return (P - self.origin) @ self.R

    def to_world(self, p_local):
        """Local aircraft coordinates -> camera-frame point(s)."""
        p_local = np.asarray(p_local, dtype=float)
        return self.origin + p_local @ self.R.T

    def direction_to_local(self, d):
        """Rotate a free vector (e.g. velocity) into the local frame."""
        return np.asarray(d, dtype=float) @ self.R

    def direction_to_world(self, d_local):
        """Rotate a local free vector back to camera coordinates."""
        return np.asarray(d_local, dtype=float) @ self.R.T


def local_keypoints(doc, variant="GT", frame=None):
    """Local coordinates of the 4 keypoints of ``variant``, in the GT frame.

    Returns {name: (3,) array or None}. ``frame`` may be passed to avoid
    rebuilding it; otherwise it is built from the document's GT points.
    """
    if frame is None:
        frame = AircraftFrame.from_doc(doc, variant="GT")
    out = {}
    for name in ["P1_tail", "P2_nose", "P3_left_wing", "P4_right_wing"]:
        d = doc["Coordinates_3D"][name].get(variant)  # no Noisy_GT key in 3D
        out[name] = None if d is None else frame.to_local([d["X"], d["Y"], d["Z"]])
    return out
