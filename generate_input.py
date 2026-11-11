"""
generate_input.py

Build a sample input JSON (``inputs/aircraft_input.json``) for the aircraft
3D-2D reconstruction, following the schema in ``aircraft_io.py``.

The ground truth is the same synthetic aircraft used by ``main.py``: a target
~2 km away with a 30 m body and 40 m wingspan. We project its four keypoints to
pixel coordinates, derive the velocity vector and range, then add a small amount
of deterministic Gaussian noise to populate ``Noisy_GT``. All ``Computed*``
variants are left ``null`` -- they will be filled in by the later solve step.
"""

import numpy as np

import aircraft_io as aio
import reconstruction_utils as ru


# --------------------------------------------------------------------------- #
# Ground-truth aircraft (matches build_ground_truth() in main.py)
# --------------------------------------------------------------------------- #
def build_ground_truth():
    # Slightly off the optical axis but comfortably inside the (narrow-FOV) frame.
    center = np.array([20.0, -15.0, 2000.0])
    body_dir = ru._normalize([0.3, 0.1, 1.0])              # tail -> nose (== V)
    wing_dir = ru._normalize(np.cross([0.0, 1.0, 0.0], body_dir))  # left->right

    body_half = 15.0   # half body length (m)  -> 30 m body
    wing_half = 20.0   # half wingspan (m)      -> 40 m span

    P1 = center - body_dir * body_half   # tail
    P2 = center + body_dir * body_half   # nose
    P3 = center - wing_dir * wing_half   # left wingtip
    P4 = center + wing_dir * wing_half   # right wingtip

    V = body_dir
    D = np.linalg.norm(0.5 * (P1 + P2))  # range to body midpoint
    return P1, P2, P3, P4, V, D


# --------------------------------------------------------------------------- #
# Noise model (deterministic)
# --------------------------------------------------------------------------- #
SIGMA_3D = 2.0    # m per axis
SIGMA_2D = 1.5    # pixels per axis
SIGMA_V = 0.01    # per axis (velocity re-normalized afterwards)
SIGMA_RANGE = 10.0  # m


def main():
    P1, P2, P3, P4, V, D = build_ground_truth()
    pts3d = [P1, P2, P3, P4]

    # Camera: 1920x1080, 5 deg horizontal FOV; vertical FOV from the aspect.
    W, H = 1920, 1080
    fov_x = 5.0
    fov_y = np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(fov_x) / 2.0) * H / W))
    cam = {
        "fov_X_deg": float(fov_x),
        "fov_Y_deg": float(fov_y),
        "H_pixels": H,
        "W_pixels": W,
    }
    fx, fy, cx, cy = aio.camera_intrinsics(cam)

    # GT 2D pixel projections.
    pts2d = [aio.project_pixel(P, fx, fy, cx, cy) for P in pts3d]

    # Deterministic noise.
    rng = np.random.default_rng(0)
    pts3d_noisy = [P + rng.normal(0.0, SIGMA_3D, 3) for P in pts3d]
    pts2d_noisy = [p + rng.normal(0.0, SIGMA_2D, 2) for p in pts2d]
    V_noisy = ru._normalize(V + rng.normal(0.0, SIGMA_V, 3))
    D_noisy = D + rng.normal(0.0, SIGMA_RANGE)

    # Assemble the document.
    doc = {
        "Camera_parameters": cam,
        "Coordinates_3D": {
            name: aio.variant_block(GT=aio.vec3(P), Noisy_GT=aio.vec3(Pn))
            for name, P, Pn in zip(aio.KEYPOINTS_3D, pts3d, pts3d_noisy)
        },
        "Coordinates_2D": {
            name: aio.variant_block(GT=aio.vec2(p), Noisy_GT=aio.vec2(pn))
            for name, p, pn in zip(aio.KEYPOINTS_2D, pts2d, pts2d_noisy)
        },
        "Velocity_vector_3D": aio.variant_block(
            GT=aio.vel(V), Noisy_GT=aio.vel(V_noisy)
        ),
        "Range_meters": aio.variant_block(
            GT=aio.scalar(D), Noisy_GT=aio.scalar(D_noisy)
        ),
    }

    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aircraft_input.json")
    aio.save_json(out_path, doc)
    print(f"Wrote {out_path}")
    print(f"  Range GT = {D:.3f} m   Noisy = {D_noisy:.3f} m")
    print(f"  GT 2D pixels: "
          + ", ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in pts2d))


if __name__ == "__main__":
    main()
