"""
reconstruction_utils.py

Implementation of every solution in the paper "3D-2D analysis of aircraft".

The aircraft is modeled as a 4-keypoint cross:
    P1 - tail   (body)
    P2 - nose   (body)
    P3 - wingtip (left)
    P4 - wingtip (right)

A calibrated pinhole camera sits at the origin [0, 0, 0]. Every 2D keypoint is
expressed in *normalized* image coordinates (u, v) and defines a ray

    P(lambda) = lambda * (u, v, 1)

The functions below recover the 3D keypoints (and the 6-DOF pose) from these
rays plus two extra pieces of information supplied by the paper's assumptions:
the body/velocity direction vector V and the range D to the target.

All functions are pure: they take and return plain numpy arrays and perform no
I/O. Section references in the docstrings point back to the paper.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _ray(u, v):
    """Direction (u, v, 1) of the camera ray through a normalized pixel."""
    return np.array([float(u), float(v), 1.0])


def _lstsq(A, b):
    """Least-squares solution of A x = b via the normal equations.

    Mirrors the paper's x = (A^T A)^{-1} A^T b. numpy.linalg.lstsq would be
    numerically preferable, but the explicit normal-equation form is kept to
    match the derivation in the document.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.inv(A.T @ A) @ A.T @ b


# --------------------------------------------------------------------------- #
# Appendix: camera helpers
# --------------------------------------------------------------------------- #
def focal_length_from_fov(W_pixel, fov, S=1.0):
    """Focal length from field of view (Appendix).

    f = W * S / (2 * tan(FOV / 2))

    Parameters
    ----------
    W_pixel : float
        Image width in pixels.
    fov : float
        Horizontal field of view in radians.
    S : float
        Pixel size. The paper works in pixels, so S = 1 and f comes out in
        pixels.
    """
    return W_pixel * S / (2.0 * np.tan(fov / 2.0))


def normalize_coords(x_pixel, y_pixel, cx, cy, f):
    """Convert pixel coordinates to normalized coordinates (Appendix).

    (u, v) = ((x_pixel - cx) / f, (y_pixel - cy) / f)

    with cx = W_pixel / 2, cy = H_pixel / 2 for a symmetric camera.
    """
    u = (x_pixel - cx) / f
    v = (y_pixel - cy) / f
    return np.array([u, v])


# --------------------------------------------------------------------------- #
# Body of aircraft - Solution #1 (direction first, then range)
# --------------------------------------------------------------------------- #
def body_solution1(u1, v1, u2, v2, V, D):
    """Recover body endpoints P1, P2 - Solution #1 (paper p.4-5).

    The body vector M = P2 - P1 is parallel to V, i.e. M = s * V for some
    scalar s. Writing P1 = lambda1 (u1, v1, 1) and P2 = lambda2 (u2, v2, 1)
    and letting lambda_tilde_i = lambda_i / s gives the 3x2 linear system

        [ u2  -u1 ] [ lambda_tilde_2 ]   [ Vx ]
        [ v2  -v1 ] [ lambda_tilde_1 ] = [ Vy ]
        [  1   -1 ]                       [ Vz ]

    solved by least squares. The range D to the midpoint then fixes the scale
    s = 2D / ||a||, after which lambda_i = s * lambda_tilde_i.

    Parameters
    ----------
    u1, v1, u2, v2 : float
        Normalized coordinates of tail (P1) and nose (P2).
    V : array_like, shape (3,)
        Body / velocity direction vector (Vx, Vy, Vz).
    D : float
        Range from the camera to the body midpoint.

    Returns
    -------
    P1, P2 : ndarray, shape (3,)
    """
    V = np.asarray(V, dtype=float)

    A = np.array([
        [u2, -u1],
        [v2, -v1],
        [1.0, -1.0],
    ])
    # [lambda_tilde_2, lambda_tilde_1]
    lam2_t, lam1_t = _lstsq(A, V)

    # Midpoint direction a = lam1_t * (u1,v1,1) + lam2_t * (u2,v2,1).
    a = lam1_t * _ray(u1, v1) + lam2_t * _ray(u2, v2)
    # 0.5 * s * ||a|| = D  ->  s = 2D / ||a||.
    s = 2.0 * D / np.linalg.norm(a)

    lam1 = s * lam1_t
    lam2 = s * lam2_t

    P1 = lam1 * _ray(u1, v1)
    P2 = lam2 * _ray(u2, v2)
    return P1, P2


# --------------------------------------------------------------------------- #
# Body of aircraft - Solution #2 (range first, then direction)
# --------------------------------------------------------------------------- #
def body_solution2(u1, v1, u2, v2, V, D):
    """Recover body endpoints P1, P2 - Solution #2 (paper p.5-7).

    Start from the 2D body midpoint, push it out along its ray to the depth
    fixed by the range D, giving the 3D midpoint Pm. Then build a symmetric
    3D segment of unknown half-length h along V that meets rays 1 and 2:

        P1 = Pm + V h
        P2 = Pm - V h

    This yields 6 equations in 3 unknowns (lambda1, lambda2, h):

        [ u1  0  -Vx ]                 [ Xm ]
        [ v1  0  -Vy ] [ lambda1 ]     [ Ym ]
        [  1  0  -Vz ] [ lambda2 ]  =  [ Zm ]
        [  0  u2  Vx ] [    h    ]     [ Xm ]
        [  0  v2  Vy ]                 [ Ym ]
        [  0   1  Vz ]                 [ Zm ]

    solved by least squares.

    Returns
    -------
    P1, P2 : ndarray, shape (3,)
    """
    V = np.asarray(V, dtype=float)
    Vx, Vy, Vz = V

    # Middle point in 2D, then its ray out to depth fixed by range D.
    um = 0.5 * (u1 + u2)
    vm = 0.5 * (v1 + v2)
    lam_m = D / np.sqrt(um * um + vm * vm + 1.0)
    Pm = lam_m * _ray(um, vm)
    Xm, Ym, Zm = Pm

    A = np.array([
        [u1, 0.0, -Vx],
        [v1, 0.0, -Vy],
        [1.0, 0.0, -Vz],
        [0.0, u2, Vx],
        [0.0, v2, Vy],
        [0.0, 1.0, Vz],
    ])
    b = np.array([Xm, Ym, Zm, Xm, Ym, Zm])
    lam1, lam2, h = _lstsq(A, b)

    P1 = lam1 * _ray(u1, v1)
    P2 = lam2 * _ray(u2, v2)
    return P1, P2


# --------------------------------------------------------------------------- #
# Wings of aircraft
# --------------------------------------------------------------------------- #
def wings_solution(u3, v3, u4, v4, V, D):
    """Recover wingtip endpoints P3, P4 (paper p.7-8).

    The wing segment P3 - P4 is perpendicular to the body direction V, and the
    range to the wing midpoint is D. With d3 = (u3,v3,1), d4 = (u4,v4,1) and

        a = (d3 . V) / (d4 . V)

    the perpendicularity constraint gives lambda4 = lambda3 * a, and the
    distance constraint ||0.5 (lambda3 d3 + lambda4 d4)|| = D reduces to

        lambda3 = 2D / sqrt(d3.d3 + a^2 d4.d4 + 2a d3.d4)

    The same routine works for any left/right pair of symmetric keypoints.

    Returns
    -------
    P3, P4 : ndarray, shape (3,)
    """
    V = np.asarray(V, dtype=float)
    d3 = _ray(u3, v3)
    d4 = _ray(u4, v4)

    a = (d3 @ V) / (d4 @ V)

    d33 = d3 @ d3
    d44 = d4 @ d4
    d34 = d3 @ d4

    lam3 = 2.0 * D / np.sqrt(d33 + a * a * d44 + 2.0 * a * d34)
    lam4 = lam3 * a

    P3 = lam3 * d3
    P4 = lam4 * d4
    return P3, P4


# --------------------------------------------------------------------------- #
# Joint solution (all four keypoints at once, paper Part II)
# --------------------------------------------------------------------------- #
# Default measurement weights for the joint solve. Only the RATIO sig_uv:sig_v
# matters for the point estimate: it sets how much the fit trusts the keypoints
# relative to the velocity direction. The absolute scale affects only the
# covariance, which this point solver does not return. The caller (main.py)
# normally derives sig_uv from the camera focal length.
JOINT_SIGMA_UV = 1.0e-4            # keypoint noise, normalized image units
JOINT_SIGMA_V = np.deg2rad(1.0)    # velocity-direction noise, radians


def joint_solution(u1, v1, u2, v2, u3, v3, u4, v4, V, D,
                   sig_uv=JOINT_SIGMA_UV, sig_v=JOINT_SIGMA_V):
    """Recover all four keypoints jointly (paper Part II, "Joint solution").

    Instead of solving the body and the wings separately, fit one parametric
    aircraft model -- body length L1, wingspan L2, wing attachment point alpha
    (0 = tail .. 1 = nose) and 6-DOF pose (R, t with ||t|| = D) -- so that its
    projection matches the four measured normalized keypoints and its body axis
    matches the velocity, in a single least-squares problem. This couples the
    body and the wings (they share one pose), averages the keypoint and
    velocity noise, and -- unlike the sequential wings solver -- does not
    assume the wings are attached at the body midpoint.

    Parameters mirror the sequential solvers: normalized coordinates of the
    four keypoints (tail, nose, left tip, right tip), the body/velocity
    direction V, and the range D to the body midpoint. ``sig_uv`` (normalized
    image units) and ``sig_v`` (radians) weight the keypoint and velocity
    residuals; only their ratio affects the result. The initial guess is the
    closed-form sequential solution (body #2 + wings), refined by
    Levenberg-Marquardt.

    Returns
    -------
    P1, P2, P3, P4 : ndarray, shape (3,)
    """
    Vhat = _normalize(V)
    uv = np.array([[u1, v1], [u2, v2], [u3, v3], [u4, v4]], dtype=float)

    def local_points(L1, L2, alpha):
        yw = (alpha - 0.5) * L1            # wing station along the body axis
        return np.array([[0.0, -L1 / 2.0, 0.0], [0.0, L1 / 2.0, 0.0],
                         [-L2 / 2.0, yw, 0.0], [L2 / 2.0, yw, 0.0]])

    def pose(theta):
        R = Rotation.from_rotvec(theta[:3]).as_matrix()
        c = _ray(theta[3], theta[4])
        t = D * c / np.linalg.norm(c)      # range is a hard constraint: |t| = D
        return R, t

    def points(theta):
        R, t = pose(theta)
        return local_points(theta[5] * D, theta[6] * D, theta[7]) @ R.T + t, R

    def residuals(theta):
        P, R = points(theta)
        uvp = P[:, :2] / P[:, 2:3]
        r_px = ((uvp - uv) / sig_uv).ravel()          # 8 reprojection residuals
        r_v = np.cross(R[:, 1], Vhat) / sig_v          # body axis vs. velocity
        return np.r_[r_px, r_v]

    # Initial guess: the closed-form sequential solution (body #2 + wings).
    P1, P2 = body_solution2(u1, v1, u2, v2, Vhat, D)
    P3, P4 = wings_solution(u3, v3, u4, v4, Vhat, D)
    x = _normalize(P4 - P3)
    y = _normalize(P2 - P1)
    z = np.cross(x, y)
    U, _, Vt = np.linalg.svd(np.column_stack((x, y, z)))
    R0 = U @ Vt                            # nearest rotation to [x y z]
    b = uv[1] - uv[0]
    wm = 0.5 * (uv[2] + uv[3])
    alpha0 = float(np.clip((wm - uv[0]) @ b / (b @ b), 0.0, 1.0))
    theta0 = np.r_[Rotation.from_matrix(R0).as_rotvec(),
                   0.5 * (uv[0] + uv[1]),
                   np.linalg.norm(P2 - P1) / D,
                   np.linalg.norm(P4 - P3) / D,
                   alpha0]

    sol = least_squares(residuals, theta0, method="lm", x_scale="jac")
    P, _ = points(sol.x)
    return P[0], P[1], P[2], P[3]


# --------------------------------------------------------------------------- #
# 6 DOF of aircraft
# --------------------------------------------------------------------------- #
def _normalize(vec):
    return np.asarray(vec, dtype=float) / np.linalg.norm(vec)


def compute_6dof(P1, P2, P3, P4):
    """Recover the aircraft 6-DOF pose from the four 3D keypoints (paper p.9).

    Local frame convention (paper section "6 DOF of aircraft"):
        * origin = body center = 0.5 (P1 + P2)
        * Y axis = tail -> nose  = P2 - P1
        * X axis = left -> right wing = P4 - P3
        * Z axis = X x Y   (up, toward the pilot's head)

    Note: the paper's axis formulas in the text are written with slightly
    inconsistent indices (e.g. P1 - P0); here we follow the explicitly stated
    semantics above, which also makes Y align with the velocity vector V.

    Returns
    -------
    dict with keys:
        translation : ndarray (3,)   body center in world coordinates
        R           : ndarray (3,3)  rotation matrix, columns [x, y, z]
        rvec        : ndarray (3,)   Rodrigues (axis-angle) vector
        pitch, yaw, roll : float     Euler angles in radians (paper formulas)
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    P3 = np.asarray(P3, dtype=float)
    P4 = np.asarray(P4, dtype=float)

    translation = 0.5 * (P1 + P2)

    y = _normalize(P2 - P1)            # tail -> nose
    x = _normalize(P4 - P3)            # left -> right wing
    z = _normalize(np.cross(x, y))     # up

    R = np.column_stack((x, y, z))

    rvec = Rotation.from_matrix(R).as_rotvec()

    # Paper's Euler formulas (indices are row, col; 0-based).
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    return {
        "translation": translation,
        "R": R,
        "rvec": rvec,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
