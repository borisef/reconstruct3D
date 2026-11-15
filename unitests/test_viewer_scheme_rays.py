"""Scheme-view geometry: camera->P_i rays must cross the plane at p_i."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from main import solve_document
from viewer import AircraftViewer, KEYPOINTS, SCHEME_FV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


def _ray_plane_hit(map3d, P):
    """Where the camera->map3d(P) ray crosses the virtual plane z=SCHEME_FV."""
    Q = map3d(P)
    return Q * (SCHEME_FV / Q[2])


def test_gt_rays_pass_through_gt_plane_points(viewer):
    _, _, map3d, map2d = viewer._scheme_geometry()
    for el in KEYPOINTS:
        hit = _ray_plane_hit(map3d, viewer.point3d(el, "GT"))
        assert np.allclose(hit, map2d(el, "GT"), atol=1e-9), el

def test_centre_maps_to_anchor(viewer):
    # The aircraft centre must land exactly on the anchor point Qc.
    anchor = viewer._scheme_anchor()
    _, Qc, map3d, _ = viewer._scheme_geometry()
    assert np.allclose(map3d(anchor["C3"]), anchor["Qc"], atol=1e-9)
    assert np.allclose(Qc, anchor["Qc"], atol=1e-9)

def test_computed_rays_pass_through_their_projection(viewer):
    # Computed points have no stored pixels; their ray must still cross the
    # plane at the linear pixel map of their true projection.
    solve_document(viewer.doc, params={"input_noisy": False})
    cam = viewer.doc["Camera_parameters"]
    W, H = cam["W_pixels"], cam["H_pixels"]
    fx, fy, cx, cy = aio.camera_intrinsics(cam)
    _, _, map3d, _ = viewer._scheme_geometry()
    from viewer import SCHEME_HALF_FOV
    a = SCHEME_FV * np.tan(SCHEME_HALF_FOV)
    b = a * H / W
    for el in ["P1_tail", "P2_nose"]:
        P = viewer.point3d(el, "Computed")
        px = aio.project_pixel(P, fx, fy, cx, cy)
        expected = np.array([(px[0] - cx) / (W / 2.0) * a,
                             (px[1] - cy) / (H / 2.0) * b, SCHEME_FV])
        assert np.allclose(_ray_plane_hit(map3d, P), expected, atol=1e-9), el

def test_rays_stay_aligned_after_gt_move(viewer):
    # Edit view: after a rigid GT move the re-projected GT pixels must still
    # sit exactly where the (frozen-anchor) rays cross the plane.
    viewer._on_view("Change GT")
    for k in ["right"] * 4 + ["shift+left"] * 3 + ["up"] * 2:
        viewer._on_key_edit(k)
    _, _, map3d, map2d = viewer._scheme_geometry()
    for el in KEYPOINTS:
        hit = _ray_plane_hit(map3d, viewer.point3d(el, "GT"))
        assert np.allclose(hit, map2d(el, "GT"), atol=1e-9), el

def test_noisy_2d_marker_off_the_gt_ray(viewer):
    # Noisy pixels are measurements, not projections: their marker must NOT
    # be forced onto the GT ray.
    _, _, map3d, map2d = viewer._scheme_geometry()
    hit = _ray_plane_hit(map3d, viewer.point3d("P1_tail", "GT"))
    noisy = map2d("P1_tail", "Noisy_GT")
    assert not np.allclose(hit, noisy, atol=1e-12)
