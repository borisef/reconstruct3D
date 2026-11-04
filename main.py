"""
main.py

Usage examples for reconstruction_utils.py, the implementation of the paper
"3D-2D analysis of aircraft".

Strategy: build a synthetic ground-truth aircraft in 3D, project its four
keypoints to normalized image coordinates with the pinhole model, derive the
inputs the paper assumes are known (body direction V and range D), then run
every reconstruction function and compare the recovered values against the
ground truth. If the implementation is correct, the recovered 3D points match
the originals up to floating-point precision.
"""

import numpy as np

import reconstruction_utils as ru


np.set_printoptions(precision=4, suppress=True)


def project(P):
    """Pinhole projection of a 3D point to normalized coordinates (u, v)."""
    X, Y, Z = P
    return X / Z, Y / Z


def build_ground_truth():
    """A synthetic aircraft placed in front of the camera.

    Returns the four 3D keypoints and the derived known quantities.
    """
    # Body center ~2 km away, slightly off-axis.
    center = np.array([150.0, -80.0, 2000.0])

    body_dir = ru._normalize([0.3, 0.1, 1.0])   # tail -> nose (== velocity V)
    wing_dir = ru._normalize(np.cross([0.0, 1.0, 0.0], body_dir))  # left->right

    body_half = 15.0   # half body length (m)
    wing_half = 20.0   # half wingspan (m)

    P1 = center - body_dir * body_half   # tail
    P2 = center + body_dir * body_half   # nose
    P3 = center - wing_dir * wing_half   # left wingtip
    P4 = center + wing_dir * wing_half   # right wingtip

    V = body_dir                         # unit velocity vector (assumption 7)
    D_body = np.linalg.norm(0.5 * (P1 + P2))   # range to body midpoint
    D_wing = np.linalg.norm(0.5 * (P3 + P4))   # range to wing midpoint
    return P1, P2, P3, P4, V, D_body, D_wing


def report(name, P_true, P_est):
    err = np.linalg.norm(P_true - P_est)
    print(f"  {name}: true={P_true}  est={P_est}  |err|={err:.3e}")


def main():
    P1, P2, P3, P4, V, D_body, D_wing = build_ground_truth()

    (u1, v1) = project(P1)
    (u2, v2) = project(P2)
    (u3, v3) = project(P3)
    (u4, v4) = project(P4)

    print("=" * 70)
    print("Ground-truth aircraft")
    print("=" * 70)
    print(f"  V (body dir) = {V}")
    print(f"  D_body = {D_body:.3f}   D_wing = {D_wing:.3f}")
    print(f"  normalized keypoints:")
    print(f"    P1 (tail) -> (u1,v1) = ({u1:.5f}, {v1:.5f})")
    print(f"    P2 (nose) -> (u2,v2) = ({u2:.5f}, {v2:.5f})")
    print(f"    P3 (lwing)-> (u3,v3) = ({u3:.5f}, {v3:.5f})")
    print(f"    P4 (rwing)-> (u4,v4) = ({u4:.5f}, {v4:.5f})")

    # ---------------------------------------------------------------- #
    # Appendix camera helpers
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Appendix: camera helpers")
    print("=" * 70)
    W_pixel, H_pixel, fov = 1920, 1080, np.deg2rad(5.0)  # small FOV (assumption 2)
    f = ru.focal_length_from_fov(W_pixel, fov, S=1.0)
    print(f"  focal_length_from_fov(W={W_pixel}, FOV=5deg) = {f:.3f} px")
    # Round-trip: take P2's normalized coords, pretend they came from pixels.
    cx, cy = W_pixel / 2.0, H_pixel / 2.0
    x_px, y_px = u2 * f + cx, v2 * f + cy
    uv = ru.normalize_coords(x_px, y_px, cx, cy, f)
    print(f"  normalize_coords round-trip -> {uv}  (expected ({u2:.5f}, {v2:.5f}))")

    # ---------------------------------------------------------------- #
    # Body - Solution #1
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Body - Solution #1 (direction first, then range)")
    print("=" * 70)
    P1e, P2e = ru.body_solution1(u1, v1, u2, v2, V, D_body)
    report("P1", P1, P1e)
    report("P2", P2, P2e)

    # ---------------------------------------------------------------- #
    # Body - Solution #2
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Body - Solution #2 (range first, then direction)")
    print("=" * 70)
    P1e2, P2e2 = ru.body_solution2(u1, v1, u2, v2, V, D_body)
    report("P1", P1, P1e2)
    report("P2", P2, P2e2)
    print(f"  solution1 vs solution2 agreement: "
          f"|dP1|={np.linalg.norm(P1e - P1e2):.3e}, "
          f"|dP2|={np.linalg.norm(P2e - P2e2):.3e}")

    # ---------------------------------------------------------------- #
    # Wings
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Wings")
    print("=" * 70)
    P3e, P4e = ru.wings_solution(u3, v3, u4, v4, V, D_wing)
    report("P3", P3, P3e)
    report("P4", P4, P4e)
    perp = np.dot(P3e - P4e, V)
    print(f"  (P3-P4).V = {perp:.3e}  (should be ~0, wings perpendicular to V)")

    # ---------------------------------------------------------------- #
    # 6 DOF
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("6 DOF")
    print("=" * 70)
    pose_true = ru.compute_6dof(P1, P2, P3, P4)
    pose_est = ru.compute_6dof(P1e, P2e, P3e, P4e)
    print(f"  translation  true={pose_true['translation']}  est={pose_est['translation']}")
    print(f"  rvec         true={pose_true['rvec']}  est={pose_est['rvec']}")
    for ang in ("pitch", "yaw", "roll"):
        print(f"  {ang:<6} true={np.rad2deg(pose_true[ang]):8.3f} deg   "
              f"est={np.rad2deg(pose_est[ang]):8.3f} deg")

    # ---------------------------------------------------------------- #
    # Derived sizes
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Derived geometry")
    print("=" * 70)
    print(f"  body length (true/est): "
          f"{np.linalg.norm(P2 - P1):.3f} / {np.linalg.norm(P2e - P1e):.3f} m")
    print(f"  wing span   (true/est): "
          f"{np.linalg.norm(P4 - P3):.3f} / {np.linalg.norm(P4e - P3e):.3f} m")


if __name__ == "__main__":
    main()
