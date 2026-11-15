"""
GUI/viewer.py

Matplotlib-based viewer for the aircraft 3D-2D reconstruction JSON documents.

Controls (left panel):
  * A compact visibility grid. Rows = variants (GT, Noisy_GT, Computed),
    labelled once on the left. Columns = elements, grouped:
        3D pts : P1 P2 P3 P4   (Coordinates_3D)
        2D pts : p1 p2 p3 p4   (Coordinates_2D)
        Vel Rng                (Velocity_vector_3D, Range_meters)
    Each cell is an independent checkbox. A cell whose underlying field is
    null/empty is disabled (greyed, non-clickable). 3D / velocity / range
    selections are accepted even though the current 2D view does not draw them.
  * A "Legends" checkbox toggling both legends at once.
  * A "Local axes" checkbox: the GT-anchored local aircraft frame (X left->right
    wing, Y tail->nose, Z up; origin [0,0,0] at the aircraft centre, between
    tail and nose) drawn as a triad in the pinhole views (b) and (c).
  * A "View" dropdown (2D / Pinhole / 3D interactive / 3D results /
    Change GT) -- one line.
  * Load JSON / Save JSON / "Solve" buttons and a status line. Solve calls
    main.solve_document(doc, params=...) with the current algorithm
    parameters; Save always asks where to save (dialog pre-filled with the
    loaded path).
  * A "Set Algo Params" button popping up a small form with the algorithm
    parameters: "Input Type: Noisy" (checkbox, default on -- the solver reads
    Noisy_GT instead of GT) and "Head/Tail solution" (#1 / #2 / Average --
    which of the two-fold-ambiguous body solutions is reported). The
    parameters persist in the document (Algorithm_parameters block), and a
    solve also writes the Results block (2D/3D results tables) so Save JSON
    exports parameters + solution + results together.
  * Noise Up / Noise Down buttons + "Noise target" radio (2D / Velocity /
    Range): each press re-samples the selected Noisy_GT block from
    GT + N(0, sigma), scaling that target's own sigma by 1.5 per press
    (2D: 0..100 px; Velocity: 0..0.5 per-axis then re-normalized;
    Range: 0..500 m, clamped positive; all snap to 0 at the bottom).
    The status line reports the current target and sigma.

Visualization:
  * (a) 2D  -- keypoints on the image plane (pixels).
  * (b) Pinhole scheme -- virtual, ray-consistent oblique projection.
  * (c) 3D interactive -- same scene in an orthographic Axes3D.
  * (d) Change GT -- same 3D scene, camera starts at preset "4"; the keyboard
    moves the GT aircraft rigidly in its local frame (arrows X/Z, +/- Y,
    shift+arrows yaw/pitch, ctrl+arrows roll, ctrl+z undo). GT 2D projections,
    velocity and range follow the moved 3D points.
  * (e) 3D results -- GT vs Computed 3D keypoints in REAL camera coordinates
    (metres), auto-framed so every aircraft point is in view. Computed
    segments reuse the body/wing colours at lower alpha; dotted connectors
    link each GT point to its computed counterpart. A results table
    (rows XYZ(m) / YPR(deg) / BodyLength / WingsSpan / CrossPoint, columns
    GT / Calc) summarises the reconstruction; CrossPoint is where the wings
    segment crosses the body, 0 = tail .. 1 = nose.

Encoding (shared by all views):
  * Point COLOUR identifies the element; point MARKER identifies the variant
    (GT = filled dot, Noisy_GT = open dot, Computed = diamond).
  * Segments are arrows: tail->nose RED, left->right wing BLUE, velocity GREEN.
    With several variants visible, the arrow uses the highest-priority visible
    one (GT > Noisy_GT > Computed).

Run:
    python GUI/viewer.py [PATH_TO_JSON]
"""

import copy
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (3d view)

# Make the project root importable when run as `python GUI/viewer.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import aircraft_io as aio  # noqa: E402
import main as algo  # noqa: E402  (solver + shared results/stats helpers)
from aircraft_frame import AircraftFrame  # noqa: E402


# --------------------------------------------------------------------------- #
# Encoding tables
# --------------------------------------------------------------------------- #
ELEMENT_2D = {
    "P1_tail": "p1_tail",
    "P2_nose": "p2_nose",
    "P3_left_wing": "p3_left_wing",
    "P4_right_wing": "p4_right_wing",
}
KEYPOINTS = list(ELEMENT_2D)

VARIANTS = aio.VARIANTS  # ["GT", "Noisy_GT", "Computed"]
VARIANT_SHORT = {"GT": "GT", "Noisy_GT": "N_GT", "Computed": "Cmp"}

# Control-grid columns: (key, kind, short label).
COLUMNS = (
    [(k, "3D", lbl) for k, lbl in zip(KEYPOINTS, ["P1", "P2", "P3", "P4"])] +
    [(k, "2D", lbl) for k, lbl in zip(KEYPOINTS, ["p1", "p2", "p3", "p4"])] +
    [("velocity", "3D", "Vel"), ("range", "scalar", "Rng")]
)
# Group super-headers: (label, first_col, last_col) inclusive.
GROUPS = [("3D pts", 0, 3), ("2D pts", 4, 7), ("aux", 8, 9)]

POINT_COLORS = {
    "P1_tail": "#ff7f0e",        # orange
    "P2_nose": "#2ca02c",        # green
    "P3_left_wing": "#9467bd",   # purple
    "P4_right_wing": "#8c564b",  # brown
}

VARIANT_MARKERS = {
    "GT":       {"marker": "o", "filled": True},
    "Noisy_GT": {"marker": "o", "filled": False},
    "Computed": {"marker": "D", "filled": True},
}

BODY_ARROW = "red"      # tail -> nose
WING_ARROW = "blue"     # left wing -> right wing
VEL_ARROW = "green"     # velocity direction

# Vertical tail stabilizer (orientation cue, cf. assets/figure1.png): a
# derived segment from the tail along local +Z, this fraction of body length.
FIN_FRACTION = 0.20

VIEW_OPTIONS = ["2D", "Pinhole (scheme)", "3D interactive", "3D results",
                "2D results", "Change GT"]
VIEW_KEY = {"2D": "2D", "Pinhole (scheme)": "pinhole", "3D interactive": "3d",
            "3D results": "results", "2D results": "results2d",
            "Change GT": "edit"}

# ----- Pinhole-scheme geometry (view b) ------------------------------------ #
# The real configuration (narrow FOV, small aircraft, huge range) cannot be
# drawn to scale: the image plane and the aircraft would be invisible specks.
# The scheme therefore uses a *virtual* configuration that keeps the picture
# ray-consistent but exaggerates angles and compresses the range:
#   * virtual focal distance f_v = 1; the aircraft centre is placed at 2 f_v
#     along its viewing ray, so camera->plane distance == plane->aircraft
#     distance;
#   * the plane half-width corresponds to an effective FOV of ~45 deg;
#   * the aircraft keeps its TRUE SHAPE (uniform scale about its centre, sized
#     relative to the plane) instead of compressing its Z extent, which would
#     flatten it (the body lies mostly along Z);
#   * each plane point is drawn at the intersection of the camera->3D-point ray
#     with the virtual plane, so rays pass through the plane exactly.
SCHEME_FV = 1.0                       # virtual focal distance
SCHEME_HALF_FOV = np.deg2rad(45.0 / 2)  # effective half-FOV of the drawn plane
SCHEME_SPAN = 0.85                    # virtual wingspan, in units of f_v
SCHEME_VEL_LEN = 0.45                 # drawn velocity arrow length (units f_v)

# Local aircraft frame triad (GT-anchored; see aircraft_frame.py). Drawn at the
# aircraft centre in the pinhole views: X left->right wing, Y tail->nose,
# Z = X x Y up. Colours avoid the body/wing/velocity arrow colours. The length
# exceeds the virtual nose (~0.32) and wing-tip (~0.43) distances so the X/Y
# axes, which are parallel to the body/wing arrows, stick out past them.
LOCAL_AXIS_LEN = 0.52                 # drawn axis length (units f_v)
LOCAL_AXIS_COLORS = {"X": "#e377c2", "Y": "#bcbd22", "Z": "#17becf"}

# Oblique orthographic basis: screen = SCHEME_M @ (x, y, z).
# Optical axis (z) runs to the upper right, image-u recedes down-right,
# image-v (down in camera convention) is drawn downward.
SCHEME_M = np.array([
    [0.45, 0.00, 0.89],    # screen x
    [-0.26, -0.90, 0.20],  # screen y
])

# ----- 3D interactive scheme (view c) -------------------------------------- #
# Same virtual scene, drawn into an Axes3D. Scene (x, y, z) is plotted as
# (z, x, -y): optical axis horizontal, true up pointing up. The default
# camera angles approximate the oblique SCHEME_M composition.
# Calibrated so the default orthographic 3D camera reproduces the oblique
# composition of the non-interactive scheme (b) (matches SCHEME_M to ~1e-5).
VIEW3D_HOME = {"elev": -21.0, "azim": -58.0, "roll": 0.0}
VIEW3D_PRESETS = {
    "1": {"elev": 0.0, "azim": 180.0, "roll": 0.0},   # plane from pinhole side
    "2": {"elev": 0.0, "azim": -90.0, "roll": 0.0},   # optical axis horizontal
}
# Preset "4" (aircraft view) is computed from the GT local frame at key press:
# local Z drawn up on screen, viewer 45 deg off the tail toward the right wing
# and slightly above the aircraft's horizontal plane, so the aircraft and the
# image plane behind it are both well seen.
VIEW3D_AIRCRAFT_AZ = -45.0    # deg off the tail (negative: left-wing side, so
                              # the nose points right and the plane sits left)
VIEW3D_AIRCRAFT_ELEV = 15.0   # deg above the aircraft's local horizontal plane
VIEW3D_ROT_STEP = 5.0     # deg per key press
VIEW3D_PAN_STEP = 0.06    # fraction of the visible range per key press
VIEW3D_ZOOM_STEP = 0.8    # range factor per zoom-in step
VIEW3D_HELP = ("mouse:  L-drag rotate   M-drag pan   R-drag/scroll zoom\n"
               "keys:  arrows pan   shift+arrows pitch/yaw   +/- zoom\n"
               "       0/Home default   1 plane view   2 axis horizontal\n"
               "       4 aircraft view (local Z up, 45°)")

# ----- "3D results" comparison view ----------------------------------------- #
# GT vs Computed keypoints drawn in REAL camera coordinates (metres), framed on
# the aircraft so every point is inside the view. The computed aircraft reuses
# the body/wing arrow colours at lower alpha so the two overlays stay easy to
# tell apart.
RESULTS_PT_ALPHA = 0.75   # computed point markers
RESULTS_SEG_ALPHA = 0.35  # computed body/wing segments (same colours as GT)

# ----- "Change GT" editing view -------------------------------------------- #
# Same 3D scene; the camera starts at preset "4" and is driven by the mouse,
# while the keys move the GT aircraft as a rigid body in its LOCAL frame
# (X wings, Y tail->nose, Z up). GT 2D projections, velocity and range follow.
GT_MOVE_STEP = 2.0        # metres per key press, along a local axis
GT_ROT_STEP = 5.0         # degrees per key press (yaw/pitch/roll)
GT_CROSS_STEP = 0.05      # CrossPoint units per key press (wings slide, w/s)
GT_SCALE_STEP = 1.05      # size factor per key press (span a/d, body q/e)
GT_SIZE_MIN = 1.0         # m, lower clamp for wingspan / body length
GT_UNDO_MAX = 200         # undo-stack depth

# 2D GT must be the pinhole projection of 3D GT; stored pixels further off
# than this (in px) are considered stale and get re-derived on load.
GT2D_TOL_PX = 1e-6

# ----- Noisy_GT re-sampling (Noise Up / Noise Down + target radio) ---------- #
# Each press re-samples the selected Noisy_GT block from GT + N(0, sigma):
#   2D       : per-axis pixels;
#   Velocity : per-axis on the unit direction, then re-normalized -- the cap
#              0.5 already means a ~30-40 deg typical direction error;
#   Range    : metres, clamped positive -- the cap 500 m is ~25% of the ~2 km
#              range.
# Sigma is multiplied/divided by NOISE_FACTOR per press; below "min" the Down
# press snaps to exactly 0 (Noisy_GT == GT) and Up restarts at "min".
# Defaults match generate_input's SIGMA_2D / SIGMA_V / SIGMA_RANGE.
NOISE_FACTOR = 1.5
NOISE_PARAMS = {
    "2D":       {"default": 1.5,  "min": 0.5,   "max": 100.0, "unit": " px"},
    "Velocity": {"default": 0.01, "min": 0.001, "max": 0.5,   "unit": ""},
    "Range":    {"default": 10.0, "min": 1.0,   "max": 500.0, "unit": " m"},
}
EDIT_HELP = ("mouse:  L-drag rotate   M-drag pan   scroll zoom   (camera)\n"
             "move GT:  ←→ X (wings)   ↑↓ Z (up/down)   "
             "+/- Y (fwd/back)\n"
             "          shift+←→ yaw   shift+↑↓ pitch   "
             "ctrl+←→ roll\n"
             "wings:  w/s slide along body (CrossPoint)   a/d span -/+\n"
             "body:   q/e length -/+\n"
             "ctrl+z undo   0/Home reset camera")
# Boxed controls legend, lower-left corner of the Change-GT view.
EDIT_CONTROLS_BOX = (
    "Change-GT controls\n"
    "Move GT    ←→  X (wings)      ↑↓  Z (up/down)\n"
    "           +/-  Y (fwd/back)      [2 m/step]\n"
    "Rotate GT  shift+←→  yaw     shift+↑↓  pitch\n"
    "           ctrl+←→   roll\n"
    "           (about centre, 5°/step; X,Y,Z kept)\n"
    "Wings      w  toward nose    s  toward tail\n"
    "           (CrossPoint ±0.05, clamped to [0,1])\n"
    "Size       a/d  wingspan -/+    q/e  body -/+\n"
    "           (x1.05 per step, about the centre)\n"
    "Undo       ctrl+z\n"
    "Camera     mouse: L-drag rotate, M-drag pan,\n"
    "           scroll zoom      0/Home reset")

AXES_RECT = [0.365, 0.07, 0.605, 0.88]


def _rotation_about(axis, angle):
    """Rodrigues rotation matrix about ``axis`` (normalized) by ``angle`` rad."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# --------------------------------------------------------------------------- #
# Custom widgets (matplotlib has no native grid-of-checkboxes or dropdown)
# --------------------------------------------------------------------------- #
class CheckGrid:
    """A compact grid of independent, individually-enableable checkboxes.

    Callbacks query live state so the grid always reflects the current document:
        is_enabled(col_index, variant) -> bool
        is_checked(col_index, variant) -> bool
        on_toggle(col_index, variant)       (called on a click of an enabled cell)
    """

    def __init__(self, ax, variants, row_labels, columns, groups,
                 is_enabled, is_checked, on_toggle):
        self.ax = ax
        self.variants = variants
        self.row_labels = row_labels
        self.columns = columns
        self.groups = groups
        self.is_enabled = is_enabled
        self.is_checked = is_checked
        self.on_toggle = on_toggle
        self.ncols = len(columns)
        self.nrows = len(variants)
        ax.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.draw()

    def _frame(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_xlim(-1.7, self.ncols + 0.1)
        self.ax.set_ylim(self.nrows + 0.1, -1.7)  # inverted: row 0 on top

    def draw(self):
        self._frame()
        ax = self.ax
        for label, c0, c1 in self.groups:
            if label == "aux":
                continue
            xc = (c0 + c1 + 1) / 2.0
            ax.text(xc, -1.25, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#333333")
            ax.plot([c0 + 0.05, c1 + 0.95], [-0.75, -0.75],
                    color="#aaaaaa", lw=0.8)
        for c, (_k, _kind, lbl) in enumerate(self.columns):
            ax.text(c + 0.5, -0.35, lbl, ha="center", va="center", fontsize=7)
        for r, v in enumerate(self.variants):
            ax.text(-0.15, r + 0.5, self.row_labels[r], ha="right",
                    va="center", fontsize=7)
            for c in range(self.ncols):
                en = self.is_enabled(c, v)
                self._cell(c, r, en, en and self.is_checked(c, v))

    def _cell(self, c, r, enabled, checked):
        x, y, s = c + 0.5, r + 0.5, 0.44
        face = "white" if enabled else "#e9e9e9"
        edge = "#333333" if enabled else "#c3c3c3"
        self.ax.add_patch(Rectangle((x - s, y - s), 2 * s, 2 * s,
                                    facecolor=face, edgecolor=edge, lw=1.0))
        if checked:
            i = s * 0.56
            self.ax.add_patch(Rectangle((x - i, y - i), 2 * i, 2 * i,
                                        facecolor="#1f77b4", edgecolor="none"))

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        c = int(np.floor(event.xdata))
        r = int(np.floor(event.ydata))
        if 0 <= c < self.ncols and 0 <= r < self.nrows:
            v = self.variants[r]
            if self.is_enabled(c, v):
                self.on_toggle(c, v)
                self.draw()
                self.ax.figure.canvas.draw_idle()


class Dropdown:
    """A one-line dropdown: a header button that expands a clickable list."""

    def __init__(self, fig, rect, options, initial, on_select):
        self.fig = fig
        self.options = options
        self.on_select = on_select
        self.current = initial
        self.expanded = False

        self.ax_header = fig.add_axes(rect)
        self.btn = Button(self.ax_header, self._label())
        self.btn.label.set_fontsize(9)
        self.btn.on_clicked(self._toggle)

        x, y, w, h = rect
        n = len(options)
        self.ax_list = fig.add_axes([x, y - h * n, w, h * n])
        self.ax_list.set_visible(False)
        self._draw_list()
        fig.canvas.mpl_connect("button_press_event", self._on_list_click)

    def _label(self):
        return f"View: {self.current}  ▾"

    def _draw_list(self):
        ax = self.ax_list
        ax.clear()
        ax.set_axis_off()
        n = len(self.options)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n)
        for i, opt in enumerate(self.options):
            yb = n - i - 1
            ax.add_patch(Rectangle((0, yb), 1, 1, facecolor="#f5f5f5",
                                   edgecolor="#cccccc"))
            ax.text(0.06, yb + 0.5, opt, va="center", ha="left", fontsize=9)

    def _toggle(self, _event):
        self.expanded = not self.expanded
        self.ax_list.set_visible(self.expanded)
        self.fig.canvas.draw_idle()

    def _on_list_click(self, event):
        if not self.expanded or event.inaxes is not self.ax_list:
            return
        n = len(self.options)
        i = n - 1 - int(np.floor(event.ydata))
        if 0 <= i < n:
            self.current = self.options[i]
            self.btn.label.set_text(self._label())
            self.expanded = False
            self.ax_list.set_visible(False)
            self.on_select(self.current)
            self.fig.canvas.draw_idle()


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #
class AircraftViewer:
    def __init__(self, path=None):
        self.path = path or self._default_path()
        self.doc = aio.load_json(self.path)
        self._ensure_gt_2d_consistent()
        self.view = "2D"
        self.show_legends = True
        self.show_local_axes = True
        self.visible = {}
        self._undo_stack = []      # GT snapshots for the Change-GT view
        self._edit_anchor = None   # frozen virtual-scene anchor (edit view)
        self.noise_target = "2D"   # radio: which Noisy_GT block to re-sample
        self.noise_sigma = {t: p["default"] for t, p in NOISE_PARAMS.items()}
        self._rng = np.random.default_rng()
        self._last_solve_source = None  # last solve's input variant; it is the
                                        # error reference of the 2D-results view
        # Algorithm parameters (Set Algo Params form); documents may carry
        # their own Algorithm_parameters block, which wins over the defaults.
        self.algo_params = algo.algo_params(self.doc)
        self._init_visibility()

        self.fig = None
        self.ax = None
        self._grid = None
        self._scene3d_drawn = False
        self._scene3d_kind = None   # "scheme" | "results": 3D scenes differ in
                                    # scale, so the kept camera is per-kind
        self._home3d = None

    @staticmethod
    def _default_path():
        out = os.path.join(_ROOT, "outputs", "aircraft_output.json")
        return out if os.path.exists(out) else \
            os.path.join(_ROOT, "inputs", "aircraft_input.json")

    # --------------------------------------------------------- visibility #
    def _init_visibility(self):
        """Default: GT checked wherever the field exists; everything else off."""
        self.visible = {}
        for c, (_k, _kind, _lbl) in enumerate(COLUMNS):
            for v in VARIANTS:
                self.visible[(c, v)] = (v == "GT") and self._cell_enabled(c, v)

    def _sync_visibility(self):
        """After load/solve: drop any selection whose field became null."""
        for c, (_k, _kind, _lbl) in enumerate(COLUMNS):
            for v in VARIANTS:
                if self.visible.get((c, v)) and not self._cell_enabled(c, v):
                    self.visible[(c, v)] = False

    def _cell_enabled(self, c, variant):
        key, kind, _lbl = COLUMNS[c]
        if kind == "2D":
            return self.doc["Coordinates_2D"][ELEMENT_2D[key]][variant] is not None
        if kind == "3D" and key in ELEMENT_2D:
            # 3D keypoints have no Noisy_GT: the noisy algorithm inputs are
            # 2D + velocity + range only.
            return (variant != "Noisy_GT"
                    and self.doc["Coordinates_3D"][key].get(variant) is not None)
        if key == "velocity":
            return self.doc["Velocity_vector_3D"][variant] is not None
        if key == "range":
            return self.doc["Range_meters"][variant] is not None
        return False

    def _cell_checked(self, c, variant):
        return bool(self.visible.get((c, variant)))

    def _toggle(self, c, variant):
        self.visible[(c, variant)] = not self.visible.get((c, variant))
        self.render()
        self.fig.canvas.draw_idle()

    def _col_index(self, key, kind):
        for c, (k, knd, _lbl) in enumerate(COLUMNS):
            if k == key and knd == kind:
                return c
        raise KeyError((key, kind))

    def _is_visible(self, key, kind, variant):
        return self._cell_checked(self._col_index(key, kind), variant)

    # --------------------------------------------------------------- data #
    def point2d(self, element, variant):
        entry = self.doc["Coordinates_2D"][ELEMENT_2D[element]][variant]
        return None if entry is None else np.array([entry["u"], entry["v"]])

    def point3d(self, element, variant):
        # .get(): 3D keypoints carry no Noisy_GT key (see aio.VARIANTS_3D)
        entry = self.doc["Coordinates_3D"][element].get(variant)
        return None if entry is None else \
            np.array([entry["X"], entry["Y"], entry["Z"]])

    def velocity(self, variant):
        entry = self.doc["Velocity_vector_3D"][variant]
        return None if entry is None else \
            np.array([entry["V_x"], entry["V_y"], entry["V_z"]])

    # --------------------------------------------- tail fin (derived) #
    @staticmethod
    def _fin_from_points(P1, P2, P3w, P4w):
        """Fin segment (tail, tip) from the four keypoints, or None.

        tip = P1 + FIN_FRACTION * |P2-P1| * Z_local with
        Z_local = normalize((P4-P3) x (P2-P1)) — the aircraft's local "up".
        """
        if any(p is None for p in (P1, P2, P3w, P4w)):
            return None
        Y = P2 - P1
        Z = np.cross(P4w - P3w, Y)
        n = np.linalg.norm(Z)
        if n < 1e-12:
            return None
        return P1, P1 + FIN_FRACTION * np.linalg.norm(Y) * (Z / n)

    def _fin_segment(self, variant):
        """Fin segment of ``variant`` in camera coordinates, or None."""
        pts = {el: self.point3d(el, variant) for el in KEYPOINTS}
        return self._fin_from_points(pts["P1_tail"], pts["P2_nose"],
                                     pts["P3_left_wing"], pts["P4_right_wing"])

    # --------------------------------------------- GT 2D/3D consistency #
    def _check_gt_2d_consistency(self):
        """Re-derive any stale 2D GT from the 3D GT; return corrected elements.

        The 2D GT pixels must always be the pinhole projection of the 3D GT
        points. Loaded files may violate this; stale (or missing) values are
        corrected in place.
        """
        fx, fy, cx, cy = aio.camera_intrinsics(self.doc["Camera_parameters"])
        fixed = []
        for el in KEYPOINTS:
            P = self.point3d(el, "GT")
            if P is None or P[2] <= 1e-9:
                continue              # nothing to project from
            proj = aio.project_pixel(P, fx, fy, cx, cy)
            stored = self.point2d(el, "GT")
            if stored is None or np.max(np.abs(stored - proj)) > GT2D_TOL_PX:
                self.doc["Coordinates_2D"][ELEMENT_2D[el]]["GT"] = aio.vec2(proj)
                fixed.append(el)
        return fixed

    def _ensure_gt_2d_consistent(self):
        """Run the GT 2D check after a load; pop an info box if it corrected."""
        fixed = self._check_gt_2d_consistency()
        if fixed:
            self._show_info(
                "2D GT corrected",
                "The 2D GT points in the loaded JSON did not match the "
                "pinhole projection of the 3D GT points for:\n\n    "
                + ", ".join(fixed) +
                "\n\nThey were re-derived from the 3D GT. The document in "
                "memory is now consistent; use Save JSON to persist it.")
        return fixed

    def _priority_variant_2d(self, elements):
        for v in VARIANTS:
            if all(self._is_visible(el, "2D", v) and self.point2d(el, v) is not None
                   for el in elements):
                return v
        return None

    # ------------------------------------------------------------- figure #
    def build_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        try:
            self.fig.canvas.manager.set_window_title("Aircraft 3D-2D viewer")
        except Exception:
            pass
        self.ax = self.fig.add_axes(AXES_RECT)
        self.ax3d = self.fig.add_axes(AXES_RECT, projection="3d")
        self.ax3d.set_visible(False)

        # Free the arrow keys from matplotlib's default back/forward bindings so
        # they can drive 3D panning.
        for km in ("keymap.back", "keymap.forward"):
            mpl.rcParams[km] = [k for k in mpl.rcParams[km]
                                if k not in ("left", "right", "up", "down")]
        # Free "s" for the wings-slide edit control (ctrl+s still saves) and
        # "q" for the body-length control (ctrl+w still quits).
        mpl.rcParams["keymap.save"] = [k for k in mpl.rcParams["keymap.save"]
                                       if k != "s"]
        mpl.rcParams["keymap.quit"] = [k for k in mpl.rcParams["keymap.quit"]
                                       if k != "q"]
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._build_controls()
        self.render()
        return self.fig

    def _build_controls(self):
        self.fig.text(0.012, 0.947, "Show  (variant × element)", fontsize=10,
                      fontweight="bold")
        ax_grid = self.fig.add_axes([0.012, 0.63, 0.25, 0.305])
        row_labels = [VARIANT_SHORT[v] for v in VARIANTS]
        self._grid = CheckGrid(ax_grid, VARIANTS, row_labels, COLUMNS, GROUPS,
                               is_enabled=self._cell_enabled,
                               is_checked=self._cell_checked,
                               on_toggle=self._toggle)

        ax_leg = self.fig.add_axes([0.015, 0.535, 0.16, 0.085])
        self._leg_check = CheckButtons(
            ax_leg, ["Legends", "Local axes"],
            [self.show_legends, self.show_local_axes])
        for lbl in self._leg_check.labels:
            lbl.set_fontsize(9)
        self._leg_check.on_clicked(self._on_checkbox)

        ax_load = self.fig.add_axes([0.015, 0.475, 0.072, 0.045])
        self._btn_load = Button(ax_load, "Load JSON")
        self._btn_load.on_clicked(self._on_load)
        ax_save = self.fig.add_axes([0.095, 0.475, 0.072, 0.045])
        self._btn_save = Button(ax_save, "Save JSON")
        self._btn_save.on_clicked(self._on_save)
        ax_slv = self.fig.add_axes([0.175, 0.475, 0.088, 0.045])
        self._btn_solve = Button(ax_slv, "Solve")
        self._btn_solve.on_clicked(lambda _e: self._solve())

        ax_prm = self.fig.add_axes([0.015, 0.42, 0.104, 0.045])
        self._btn_params = Button(ax_prm, "Set Algo Params")
        self._btn_params.on_clicked(self._on_set_params)
        ax_nu = self.fig.add_axes([0.127, 0.42, 0.064, 0.045])
        self._btn_noise_up = Button(ax_nu, "Noise Up")
        self._btn_noise_up.on_clicked(lambda _e: self._on_noise(up=True))
        ax_nd = self.fig.add_axes([0.199, 0.42, 0.064, 0.045])
        self._btn_noise_down = Button(ax_nd, "Noise Down")
        self._btn_noise_down.on_clicked(lambda _e: self._on_noise(up=False))

        for btn in (self._btn_load, self._btn_save, self._btn_solve,
                    self._btn_params, self._btn_noise_up,
                    self._btn_noise_down):
            btn.label.set_fontsize(8)

        # Noise target: which Noisy_GT block the Noise Up/Down buttons act on.
        self.fig.text(0.015, 0.398, "Noise target", fontsize=8,
                      fontweight="bold")
        ax_nt = self.fig.add_axes([0.015, 0.295, 0.10, 0.10])
        ax_nt.set_facecolor("#f2f2f2")
        self._noise_radio = RadioButtons(ax_nt, list(NOISE_PARAMS),
                                         active=0)
        for lbl in self._noise_radio.labels:
            lbl.set_fontsize(8)
        self._noise_radio.on_clicked(self._on_noise_target)

        self._status = self.fig.text(
            0.015, 0.283, self._status_text(), fontsize=7, va="top",
            family="monospace")

        # Dropdown created last so its expanded list axes is on top of every
        # other widget in z-order. The rect positions the header at the top of
        # the left panel; the list opens downward over the check grid.
        self._dropdown = Dropdown(
            self.fig, [0.012, 0.955, 0.25, 0.038], VIEW_OPTIONS, "2D",
            on_select=self._on_view)

    def _status_text(self):
        return f"file: {os.path.relpath(self.path, _ROOT)}"

    # -------------------------------------------------------------- events #
    def _on_checkbox(self, label):
        if label == "Legends":
            self.show_legends = not self.show_legends
        else:  # "Local axes"
            self.show_local_axes = not self.show_local_axes
        self.render()
        self.fig.canvas.draw_idle()

    def _on_view(self, label):
        self.view = VIEW_KEY[label]
        # Change GT: freeze the virtual-scene anchor at the entry configuration
        # (so GT edits move only the aircraft) and start from aircraft view (4).
        self._edit_anchor = self._scheme_anchor() if self.view == "edit" \
            else None
        self.render()
        if self.view == "edit":
            view = self._view3d_aircraft45()
            if view is not None:
                self.ax3d.view_init(**view)
        self.fig.canvas.draw_idle()

    def _on_load(self, _event):
        path = self._ask_open()
        if not path:
            return
        self.path = path
        self.doc = aio.load_json(path)
        self._ensure_gt_2d_consistent()
        self._undo_stack = []
        self._last_solve_source = None
        self.algo_params = algo.algo_params(self.doc)
        if self.view == "edit":    # re-anchor the edit scene to the new data
            self._edit_anchor = self._scheme_anchor()
        self._init_visibility()
        self._status.set_text(self._status_text())
        self._grid.draw()
        self.render()
        self.fig.canvas.draw_idle()

    def _solve(self):
        """Run the solver with the current algorithm parameters.

        This is the GUI's algorithm interface: ``main.solve_document(doc,
        params=...)`` reads the parameter-selected variant's 2D keypoints,
        velocity and range and fills the single Computed solution in place
        (plus the Algorithm_parameters and Results blocks).
        """
        try:
            algo.solve_document(self.doc, params=self.algo_params)
            self._last_solve_source = \
                "Noisy_GT" if self.algo_params["input_noisy"] else "GT"
            self._sync_visibility()
            self._status.set_text(self._status_text()
                                  + f"  [solved: {self._params_text()}]")
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self._status.set_text(f"solve failed: {exc}")
        self._grid.draw()
        self.render()
        self.fig.canvas.draw_idle()

    def _params_text(self):
        src = "Noisy" if self.algo_params["input_noisy"] else "GT"
        return f"input {src}, head/tail {self.algo_params['head_tail_solution']}"

    def _on_set_params(self, _event):
        """Set Algo Params button: pop the form, keep the result in the doc."""
        params = self._ask_algo_params()
        if params is None:          # cancelled (or Tk unavailable)
            return
        self.algo_params = algo.algo_params(override=params)
        # Persist with the document so Save JSON exports the parameters even
        # before a solve (a solve rewrites the block anyway).
        self.doc["Algorithm_parameters"] = dict(self.algo_params)
        self._status.set_text(self._status_text()
                              + f"  [params: {self._params_text()}]")
        self.fig.canvas.draw_idle()

    def _ask_algo_params(self):
        """Modal algorithm-parameters form; returns a params dict or None.

        Current parameters: "Input Type: Noisy" checkbox and the "Head/Tail
        solution" dropdown (#1 / #2 / Average). The form will grow with the
        algorithm.
        """
        try:
            import tkinter as tk
            from tkinter import ttk
            root = tk.Tk()
            root.title("Algorithm parameters")
            root.resizable(False, False)
            noisy = tk.BooleanVar(root, value=self.algo_params["input_noisy"])
            head_tail = tk.StringVar(
                root, value=self.algo_params["head_tail_solution"])
            frm = ttk.Frame(root, padding=12)
            frm.grid()
            ttk.Checkbutton(frm, text="Input Type: Noisy", variable=noisy) \
                .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
            ttk.Label(frm, text="Head/Tail solution:") \
                .grid(row=1, column=0, sticky="w")
            ttk.Combobox(frm, textvariable=head_tail, state="readonly",
                         values=algo.HEAD_TAIL_OPTIONS, width=9) \
                .grid(row=1, column=1, sticky="w", padx=(8, 0))
            result = {}

            def _ok():
                result["params"] = {"input_noisy": bool(noisy.get()),
                                    "head_tail_solution": head_tail.get()}
                root.destroy()

            ttk.Button(frm, text="OK", command=_ok) \
                .grid(row=2, column=0, pady=(12, 0))
            ttk.Button(frm, text="Cancel", command=root.destroy) \
                .grid(row=2, column=1, pady=(12, 0))
            root.grab_set()
            root.mainloop()
            return result.get("params")
        except Exception:
            return None

    def _on_save(self, _event):
        path = self._ask_save()
        if not path:
            return
        try:
            aio.save_json(path, self.doc)
            self.path = path
            self._status.set_text(self._status_text() + "  [saved]")
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self._status.set_text(f"save failed: {exc}")
        self.fig.canvas.draw_idle()

    def _on_noise_target(self, label):
        self.noise_target = label
        p = NOISE_PARAMS[label]
        self._status.set_text(
            self._status_text()
            + f"  [noise {label} σ={self.noise_sigma[label]:.3g}{p['unit']}]")
        self.fig.canvas.draw_idle()

    def _on_noise(self, up):
        """Noise Up/Down: adjust the selected sigma and re-sample Noisy_GT."""
        target = self.noise_target
        p = NOISE_PARAMS[target]
        s = self.noise_sigma[target]
        if up:
            s = p["min"] if s <= 0 else min(s * NOISE_FACTOR, p["max"])
        else:
            s = s / NOISE_FACTOR
            if s < p["min"]:
                s = 0.0
        self.noise_sigma[target] = s

        if target == "2D":
            for el in KEYPOINTS:
                gt = self.point2d(el, "GT")
                if gt is None:
                    continue    # cannot re-derive noise without a GT reference
                noisy = gt + self._rng.normal(0.0, s, size=2)
                self.doc["Coordinates_2D"][ELEMENT_2D[el]]["Noisy_GT"] = \
                    aio.vec2(noisy)
        elif target == "Velocity":
            V = self.velocity("GT")
            if V is not None:
                noisy = V + self._rng.normal(0.0, s, size=3)
                n = np.linalg.norm(noisy)
                if n > 1e-12:   # keep the direction unit-length, like GT
                    noisy = noisy / n * np.linalg.norm(V)
                self.doc["Velocity_vector_3D"]["Noisy_GT"] = aio.vel(noisy)
        else:                   # "Range"
            D = self.doc["Range_meters"]["GT"]
            if D is not None:
                noisy = max(1.0, D + self._rng.normal(0.0, s))
                self.doc["Range_meters"]["Noisy_GT"] = aio.scalar(noisy)

        self._status.set_text(self._status_text()
                              + f"  [noise {target} σ={s:.3g}{p['unit']}]")
        self._grid.draw()       # Noisy_GT cells may have become enabled
        self.render()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _ask_open():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Open reconstruction JSON", initialdir=_ROOT,
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            root.destroy()
            return path
        except Exception:
            return None

    @staticmethod
    def _show_info(title, message):
        """Modal info box (OK button only); no-op if Tk is unavailable."""
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message, parent=root)
            root.destroy()
        except Exception:
            pass

    def _ask_save(self):
        """Save dialog, pre-filled with the currently loaded JSON path."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Save reconstruction JSON",
                initialdir=os.path.dirname(os.path.abspath(self.path)),
                initialfile=os.path.basename(self.path),
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            root.destroy()
            return path
        except Exception:
            return None

    # ------------------------------------------------------------- render #
    def render(self):
        if self.view in ("3d", "edit", "results"):
            self.ax.set_visible(False)
            self.ax3d.set_visible(True)
            if self.view == "results":
                self._render_results_3d()
            else:
                self._render_pinhole_3d()
            return
        self.ax3d.set_visible(False)
        self.ax.set_visible(True)
        self.ax.clear()
        if self.view == "2D":
            self._render_2d()
        elif self.view == "results2d":
            self._render_results_2d()
        else:
            self._render_pinhole()

    def _render_placeholder(self, text):
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12,
                     color="#555555", transform=self.ax.transAxes)

    def _render_2d(self):
        ax = self.ax
        ax.set_axis_on()
        cam = self.doc["Camera_parameters"]
        W, H = cam["W_pixels"], cam["H_pixels"]

        ax.add_patch(Rectangle((0, 0), W, H, fill=False,
                               edgecolor="gray", linewidth=1.0))
        ax.set_xlim(-0.05 * W, 1.05 * W)
        ax.set_ylim(1.05 * H, -0.05 * H)     # image convention: v points down
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title("2D image plane")
        ax.grid(True, linestyle=":", alpha=0.4)

        for el in KEYPOINTS:
            color = POINT_COLORS[el]
            for v in VARIANTS:
                if not self._is_visible(el, "2D", v):
                    continue
                p = self.point2d(el, v)
                if p is None:
                    continue
                style = VARIANT_MARKERS[v]
                ax.plot(p[0], p[1], linestyle="none", marker=style["marker"],
                        markersize=10, markeredgecolor=color,
                        markerfacecolor=(color if style["filled"] else "none"),
                        markeredgewidth=1.6, zorder=3)

        self._draw_segment_arrow(["P1_tail", "P2_nose"], BODY_ARROW)
        self._draw_segment_arrow(["P3_left_wing", "P4_right_wing"], WING_ARROW)
        self._draw_fin_2d()
        self._draw_velocity_arrow()
        if self.show_legends:
            self._draw_legends()

    def _draw_segment_arrow(self, elements, color):
        v = self._priority_variant_2d(elements)
        if v is None:
            return
        p_from = self.point2d(elements[0], v)
        p_to = self.point2d(elements[1], v)
        self.ax.annotate("", xy=tuple(p_to), xytext=tuple(p_from),
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                         shrinkA=6, shrinkB=6), zorder=2)

    def _draw_fin_2d(self):
        """Projected tail fin in the 2D view (same variant as the body arrow).

        Derived from 3D data, so it is skipped for 2D-only variants
        (e.g. Noisy_GT, which has no 3D keypoints).
        """
        v = self._priority_variant_2d(["P1_tail", "P2_nose"])
        if v is None:
            return
        seg = self._fin_segment(v)
        if seg is None:
            return
        fx, fy, cx, cy = aio.camera_intrinsics(self.doc["Camera_parameters"])
        a = aio.project_pixel(seg[0], fx, fy, cx, cy)
        b = aio.project_pixel(seg[1], fx, fy, cx, cy)
        self.ax.plot([a[0], b[0]], [a[1], b[1]], color=BODY_ARROW, lw=2.0,
                     solid_capstyle="round", zorder=2)

    # ---------------------------------------------- 2D results view (f) #
    def _projected_computed_2d(self):
        """{el: pixel} — the Computed 3D points reprojected to the image."""
        fx, fy, cx, cy = aio.camera_intrinsics(self.doc["Camera_parameters"])
        out = {}
        for el in KEYPOINTS:
            P = self.point3d(el, "Computed")
            if P is None or P[2] <= 1e-9:
                continue
            out[el] = aio.project_pixel(P, fx, fy, cx, cy)
        return out

    def _results2d_error_text(self):
        """Pixel |err| table: solve-source 2D vs reprojected Computed, or None.

        The reference variant is the one that fed the last solve (GT or
        Noisy_GT); without a solve this session there is no reference. The
        numbers are the same table the solver writes into Results.errors_2D_px
        (recomputed live, so later noise presses show up).
        """
        src = self._last_solve_source
        if src is None:
            return None
        errs = algo.reprojection_errors_px(self.doc, src)
        if errs is None:
            return None
        lines = [f"|err| (px) vs {VARIANT_SHORT[src]}"]
        lines += [f"  {name:<14s}{e:8.2f}" for name, e in errs.items()]
        return "\n".join(lines)

    def _render_results_2d(self):
        """2D results: GT + Noisy_GT pixels vs the reprojected Computed
        points, with the pixel error norm against the solve's source."""
        ax = self.ax
        ax.set_axis_on()
        cam = self.doc["Camera_parameters"]
        W, H = cam["W_pixels"], cam["H_pixels"]
        fx, fy, cx, cy = aio.camera_intrinsics(cam)

        ax.add_patch(Rectangle((0, 0), W, H, fill=False,
                               edgecolor="gray", linewidth=1.0))
        ax.set_xlim(-0.05 * W, 1.05 * W)
        ax.set_ylim(1.05 * H, -0.05 * H)     # image convention: v points down
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title("2D results — GT / Noisy vs projected Computed (pixels)")
        ax.grid(True, linestyle=":", alpha=0.4)

        def marker(p, el, v, alpha=1.0):
            style = VARIANT_MARKERS[v]
            color = POINT_COLORS[el]
            ax.plot(p[0], p[1], linestyle="none", marker=style["marker"],
                    markersize=10, markeredgecolor=color,
                    markerfacecolor=(color if style["filled"] else "none"),
                    markeredgewidth=1.6, alpha=alpha, zorder=3)

        def arrow(a, b, color, lw, alpha=1.0):
            ax.annotate("", xy=tuple(b), xytext=tuple(a),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                        shrinkA=6, shrinkB=6, alpha=alpha),
                        zorder=2)

        def fin_px(variant, alpha=1.0):
            seg = self._fin_segment(variant)
            if seg is None:
                return
            a = aio.project_pixel(seg[0], fx, fy, cx, cy)
            b = aio.project_pixel(seg[1], fx, fy, cx, cy)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BODY_ARROW, lw=2.0,
                    solid_capstyle="round", alpha=alpha, zorder=2)

        # Measured pixels: GT (filled) and Noisy_GT (open) + GT overlay.
        for v in ("GT", "Noisy_GT"):
            for el in KEYPOINTS:
                p = self.point2d(el, v)
                if p is not None:
                    marker(p, el, v)
        pg = {el: self.point2d(el, "GT") for el in KEYPOINTS}
        if pg["P1_tail"] is not None and pg["P2_nose"] is not None:
            arrow(pg["P1_tail"], pg["P2_nose"], BODY_ARROW, 2.0)
        if pg["P3_left_wing"] is not None and pg["P4_right_wing"] is not None:
            arrow(pg["P3_left_wing"], pg["P4_right_wing"], WING_ARROW, 1.6)
        fin_px("GT")

        # Reprojected Computed points + overlay at reduced alpha.
        proj = self._projected_computed_2d()
        for el, p in proj.items():
            marker(p, el, "Computed", alpha=RESULTS_PT_ALPHA)
        if "P1_tail" in proj and "P2_nose" in proj:
            arrow(proj["P1_tail"], proj["P2_nose"], BODY_ARROW, 2.0,
                  alpha=RESULTS_SEG_ALPHA)
        if "P3_left_wing" in proj and "P4_right_wing" in proj:
            arrow(proj["P3_left_wing"], proj["P4_right_wing"],
                  WING_ARROW, 1.6, alpha=RESULTS_SEG_ALPHA)
        fin_px("Computed", alpha=RESULTS_SEG_ALPHA)

        # Dotted reference -> reprojection connectors (reference = the solve
        # source when known, else GT).
        src = self._last_solve_source or "GT"
        for el, p in proj.items():
            ref = self.point2d(el, src)
            if ref is not None:
                ax.plot([ref[0], p[0]], [ref[1], p[1]], color="#777777",
                        lw=0.8, linestyle=":", zorder=1)

        err = self._results2d_error_text()
        if err is not None:
            ax.text(0.01, 0.02, err, transform=ax.transAxes, va="bottom",
                    fontsize=8, family="monospace", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f2",
                              edgecolor="#888888", alpha=0.9))
        if not proj:
            ax.text(0.99, 0.02, "no Computed data — press Solve",
                    transform=ax.transAxes, ha="right", fontsize=9,
                    color="#aa3333")
        if self.show_legends:
            self._draw_legends()

    def _draw_velocity_arrow(self):
        # One arrow per checked variant (non-GT fainter), anchored at that
        # variant's body midpoint, so e.g. Velocity noise is visible.
        for v in self._visible_velocity_variants():
            P1, P2, Vel = (self.point3d("P1_tail", v),
                           self.point3d("P2_nose", v), self.velocity(v))
            if P1 is None or P2 is None:
                continue
            alpha = 1.0 if v == "GT" else 0.65
            mid = 0.5 * (P1 + P2)
            tip = mid + Vel / np.linalg.norm(Vel) * 300.0   # 300 m along V
            fx, fy, cx, cy = aio.camera_intrinsics(self.doc["Camera_parameters"])
            a = aio.project_pixel(mid, fx, fy, cx, cy)
            b = aio.project_pixel(tip, fx, fy, cx, cy)
            self.ax.annotate("", xy=tuple(b), xytext=tuple(a),
                             arrowprops=dict(arrowstyle="-|>", color=VEL_ARROW,
                                             lw=2.0, alpha=alpha), zorder=2)

    # ------------------------------------------------- pinhole scheme (b) #
    def _scheme_ref_variant(self):
        """Variant whose data anchors the virtual geometry (priority order)."""
        for v in VARIANTS:
            if all(self.point3d(el, v) is not None for el in KEYPOINTS):
                return v
        return None

    def _scheme_anchor(self):
        """(C3, Qc, s_air) anchoring the virtual scene to the ref variant.

        Centre ray: map the centre's pixel offset linearly into the plane,
        then put the virtual aircraft centre at twice that point (so the
        camera->plane and plane->aircraft distances are equal along the ray).
        """
        ref = self._scheme_ref_variant()
        if ref is None:
            return None
        cam = self.doc["Camera_parameters"]
        W, H = cam["W_pixels"], cam["H_pixels"]
        fv = SCHEME_FV
        a = fv * np.tan(SCHEME_HALF_FOV)
        b = a * H / W
        P = {el: self.point3d(el, ref) for el in KEYPOINTS}
        C3 = 0.5 * (P["P1_tail"] + P["P2_nose"])
        span = np.linalg.norm(P["P4_right_wing"] - P["P3_left_wing"])
        s_air = SCHEME_SPAN * fv / span
        fx, fy, cx, cy = aio.camera_intrinsics(cam)
        c_px = aio.project_pixel(C3, fx, fy, cx, cy)
        hit = np.array([(c_px[0] - cx) / (W / 2.0) * a,
                        (c_px[1] - cy) / (H / 2.0) * b, fv])
        return {"C3": C3, "Qc": 2.0 * hit, "s_air": s_air}

    def _scheme_geometry(self):
        """Virtual-space constructs for the scheme (see SCHEME_* constants).

        Returns (plane_corners 4x3, centre 3, map3d, map2d) where
        map3d(P) -> virtual 3D point (on the camera ray through its own pixel
        projection) and map2d(el, variant) -> point on the virtual plane
        (stored pixels mapped linearly; ray ∩ plane fallback), or None.
        Consequence: the camera -> P_i ray crosses the plane exactly at the
        drawn p_i whenever the stored pixels match the 3D point's projection
        (guaranteed for GT by the on-load consistency check).

        Normally the mapping re-anchors to the current ref variant each render.
        In the Change-GT view the anchor is FROZEN at entry, so moving the GT
        moves only the aircraft (and its plane projections) in the virtual
        scene -- the pinhole, plane and framing stay put.
        """
        cam = self.doc["Camera_parameters"]
        W, H = cam["W_pixels"], cam["H_pixels"]
        fv = SCHEME_FV
        a = fv * np.tan(SCHEME_HALF_FOV)          # plane half-width
        b = a * H / W                             # plane half-height
        plane = np.array([[-a, -b, fv], [a, -b, fv], [a, b, fv], [-a, b, fv]])

        ref = self._scheme_ref_variant()
        anchor = (self._edit_anchor
                  if self.view == "edit" and self._edit_anchor is not None
                  else self._scheme_anchor())
        if ref is None or anchor is None:
            return plane, None, None, None
        C3, Qc, s_air = anchor["C3"], anchor["Qc"], anchor["s_air"]

        fx, fy, cx, cy = aio.camera_intrinsics(cam)

        def map3d(Pt):
            """Virtual 3D point, placed ON the camera ray through its own
            pixel projection.

            The affine map (Qc + s_air*(P - C3)) sets the virtual depth, but
            its x/y would miss the pixel-mapped plane point: the scheme uses
            an exaggerated FOV, so scale+shift and true projection disagree.
            Re-basing the point onto the ray through the linear pixel map of
            its real projection makes camera -> P_i rays cross the plane
            exactly at p_i.
            """
            Pt = np.asarray(Pt, dtype=float)
            Q_aff = Qc + s_air * (Pt - C3)
            if Pt[2] > 1e-9 and Q_aff[2] > 1e-9:
                px = aio.project_pixel(Pt, fx, fy, cx, cy)
                hit = np.array([(px[0] - cx) / (W / 2.0) * a,
                                (px[1] - cy) / (H / 2.0) * b, fv])
                return hit * (Q_aff[2] / fv)
            return Q_aff                      # degenerate: affine fallback

        def map2d(el, variant):
            px = self.point2d(el, variant)        # stored pixels: 2D data (and
            if px is not None:                    # its noise) shows as-is
                return np.array([(px[0] - cx) / (W / 2.0) * a,
                                 (px[1] - cy) / (H / 2.0) * b, fv])
            Pt = self.point3d(el, variant)        # 3D-only fallback: ray ∩ plane
            if Pt is None:
                return None
            Q = map3d(Pt)
            if Q[2] > 1e-9:
                return Q * (fv / Q[2])
            return None

        # Current aircraft centre in virtual space (range-line endpoint).
        # Equals Qc outside the edit view; follows the moved GT inside it.
        cur_C3 = 0.5 * (self.point3d("P1_tail", ref) +
                        self.point3d("P2_nose", ref))
        return plane, map3d(cur_C3), map3d, map2d

    @staticmethod
    def _scr(p):
        """Oblique orthographic projection to the drawing plane."""
        return SCHEME_M @ np.asarray(p, dtype=float)

    def _local_axes_virtual(self, map3d):
        """GT-anchored local aircraft triad in virtual scheme space, or None.

        Returns (origin_v, {"X"|"Y"|"Z": tip_v}). The local [0,0,0] is the
        aircraft centre (body midpoint). map3d is a uniform scale + shift, so
        axis directions carry over unchanged and the drawn length is fixed in
        virtual units (LOCAL_AXIS_LEN), like the velocity arrow.
        """
        if not self.show_local_axes or map3d is None:
            return None
        if any(self.point3d(el, "GT") is None for el in KEYPOINTS):
            return None                 # frame is defined by the GT points only
        frame = AircraftFrame.from_doc(self.doc, variant="GT")
        origin_v = map3d(frame.origin)
        tips = {name: origin_v + LOCAL_AXIS_LEN * frame.R[:, i]
                for i, name in enumerate(("X", "Y", "Z"))}
        return origin_v, tips

    def _visible_velocity_variants(self):
        """All checked velocity variants with data, in priority order."""
        return [v for v in VARIANTS
                if self._is_visible("velocity", "3D", v)
                and self.velocity(v) is not None]

    def _checked_range_variants(self):
        """All checked Range variants with data, in priority order."""
        c = self._col_index("range", "scalar")
        return [v for v in VARIANTS
                if self._cell_checked(c, v)
                and self.doc["Range_meters"][v] is not None]

    def _range_label_text(self):
        """Range label listing every checked variant's value, or None."""
        parts = [f"{VARIANT_SHORT[v]}={self.doc['Range_meters'][v]:.0f}m"
                 for v in self._checked_range_variants()]
        return ("Range D: " + "  ".join(parts)) if parts else None

    def _priority_visible(self, kind, elements):
        """Highest-priority variant checked (and present) for all elements."""
        for v in VARIANTS:
            ok = True
            for el in elements:
                if not self._is_visible(el, kind, v):
                    ok = False
                    break
                has = (self.point2d(el, v) if kind == "2D"
                       else self.point3d(el, v))
                if has is None:
                    ok = False
                    break
            if ok:
                return v
        return None

    def _render_pinhole(self):
        ax = self.ax
        ax.set_axis_on()
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title("Pinhole camera scheme (virtual, not to scale)")

        plane, Qc, map3d, map2d = self._scheme_geometry()

        # Image plane (translucent, like the document figures).
        corners = np.array([self._scr(p) for p in plane])
        ax.fill(corners[:, 0], corners[:, 1], facecolor="#aec7e8",
                edgecolor="#333333", alpha=0.45, lw=1.0, zorder=1)

        # Camera centre.
        o = self._scr([0.0, 0.0, 0.0])
        ax.plot(*o, marker="o", markersize=9, markerfacecolor="#9ecae1",
                markeredgecolor="#333333", zorder=5)
        ax.annotate("[0,0,0]", o, textcoords="offset points", xytext=(-8, -14),
                    ha="right", fontsize=9)

        if Qc is None:
            ax.text(0.5, 0.02, "no complete 3D keypoint set in the document",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    color="#aa3333")
            self._finish_scheme_axes(corners, extra=[o])
            return

        pts_scr = [o]

        # Range line camera -> aircraft centre (Rng checkbox). The label lists
        # every checked variant so e.g. Range noise is visible immediately.
        c_scr = self._scr(Qc)
        rng_label = self._range_label_text()
        if rng_label is not None:
            ax.plot([o[0], c_scr[0]], [o[1], c_scr[1]], color="#444444",
                    lw=0.8, zorder=2)
            mid = 0.72 * o + 0.28 * c_scr
            ax.annotate(rng_label, mid,
                        textcoords="offset points", xytext=(2, -11),
                        ha="left", fontsize=8, rotation=0)
            pts_scr.append(c_scr)

        # 3D keypoints (visibility grid, kind "3D") + rays through the plane.
        ray_colors = {"P1_tail": BODY_ARROW, "P2_nose": BODY_ARROW,
                      "P3_left_wing": WING_ARROW, "P4_right_wing": WING_ARROW}
        for el in KEYPOINTS:
            color = POINT_COLORS[el]
            ray_done = False
            for v in VARIANTS:
                if not self._is_visible(el, "3D", v):
                    continue
                Pt = self.point3d(el, v)
                if Pt is None:
                    continue
                q = self._scr(map3d(Pt))
                pts_scr.append(q)
                style = VARIANT_MARKERS[v]
                ax.plot(*q, linestyle="none", marker=style["marker"],
                        markersize=9, markeredgecolor=color,
                        markerfacecolor=(color if style["filled"] else "none"),
                        markeredgewidth=1.5, zorder=4)
                if not ray_done:      # ray for the highest-priority variant
                    ls = "--" if ray_colors[el] == BODY_ARROW else ":"
                    ax.plot([o[0], q[0]], [o[1], q[1]], color=ray_colors[el],
                            lw=1.0, linestyle=ls, alpha=0.8, zorder=2)
                    ax.annotate(el.split("_")[0], q,
                                textcoords="offset points", xytext=(6, 6),
                                fontsize=9)
                    ray_done = True

        # 2D keypoints on the virtual plane (kind "2D").
        for el in KEYPOINTS:
            color = POINT_COLORS[el]
            for v in VARIANTS:
                if not self._is_visible(el, "2D", v):
                    continue
                q3 = map2d(el, v)
                if q3 is None:
                    continue
                q = self._scr(q3)
                style = VARIANT_MARKERS[v]
                ax.plot(*q, linestyle="none", marker=style["marker"],
                        markersize=5, markeredgecolor=color,
                        markerfacecolor=(color if style["filled"] else "none"),
                        markeredgewidth=1.1, zorder=3)

        # Segment arrows: 3D (bold, like the figures) and on-plane (thin).
        def arrow(p_from, p_to, color, lw):
            ax.annotate("", xy=tuple(p_to), xytext=tuple(p_from),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                        shrinkA=3, shrinkB=3), zorder=4)

        v3 = self._priority_visible("3D", ["P1_tail", "P2_nose"])
        if v3 is not None:
            arrow(self._scr(map3d(self.point3d("P1_tail", v3))),
                  self._scr(map3d(self.point3d("P2_nose", v3))),
                  BODY_ARROW, 2.6)
        v3w = self._priority_visible("3D", ["P3_left_wing", "P4_right_wing"])
        if v3w is not None:
            arrow(self._scr(map3d(self.point3d("P3_left_wing", v3w))),
                  self._scr(map3d(self.point3d("P4_right_wing", v3w))),
                  WING_ARROW, 2.0)
        v2 = self._priority_visible("2D", ["P1_tail", "P2_nose"])
        if v2 is not None:
            arrow(self._scr(map2d("P1_tail", v2)),
                  self._scr(map2d("P2_nose", v2)), BODY_ARROW, 1.1)
        v2w = self._priority_visible("2D", ["P3_left_wing", "P4_right_wing"])
        if v2w is not None:
            arrow(self._scr(map2d("P3_left_wing", v2w)),
                  self._scr(map2d("P4_right_wing", v2w)), WING_ARROW, 1.1)

        # Tail fin (orientation cue): bold on the virtual aircraft, thin on
        # the plane. on_plane(): map3d points sit on their own pixel ray, so
        # the ray \cap plane point IS the projected fin.
        def on_plane(P):
            Q = map3d(P)
            return Q * (SCHEME_FV / Q[2])

        if v3 is not None:
            seg = self._fin_segment(v3)
            if seg is not None:
                fa, fb = self._scr(map3d(seg[0])), self._scr(map3d(seg[1]))
                ax.plot([fa[0], fb[0]], [fa[1], fb[1]], color=BODY_ARROW,
                        lw=2.0, solid_capstyle="round", zorder=4)
                pts_scr.append(fb)
                pa, pb = self._scr(on_plane(seg[0])), self._scr(on_plane(seg[1]))
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=BODY_ARROW,
                        lw=1.1, solid_capstyle="round", zorder=3)

        # Local aircraft axes (GT-anchored, origin at the aircraft centre).
        la = self._local_axes_virtual(map3d)
        if la is not None:
            o_v, tips = la
            oq = self._scr(o_v)
            pts_scr.append(oq)
            for name, tip in tips.items():
                tq = self._scr(tip)
                pts_scr.append(tq)
                ax.annotate("", xy=tuple(tq), xytext=tuple(oq),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=LOCAL_AXIS_COLORS[name],
                                            lw=1.4), zorder=5)
                ax.annotate(f"${name}_{{loc}}$", tq,
                            textcoords="offset points", xytext=(3, 3),
                            fontsize=8, color=LOCAL_AXIS_COLORS[name])

        # Velocity (green dashed). Drawn from the camera centre, as in Figure 2
        # of the document -- V is parallel to the body, so drawing it at the
        # aircraft would hide it under the red body arrow. One arrow per
        # checked variant (non-GT fainter) so e.g. Velocity noise is visible.
        for v in self._visible_velocity_variants():
            Vel = self.velocity(v)
            alpha = 1.0 if v == "GT" else 0.65
            tip3 = Vel / np.linalg.norm(Vel) * SCHEME_VEL_LEN
            bq, tq = o, self._scr(tip3)
            ax.annotate("", xy=tuple(tq), xytext=tuple(bq),
                        arrowprops=dict(arrowstyle="-|>", color=VEL_ARROW,
                                        lw=2.0, linestyle="--", alpha=alpha),
                        zorder=4)
            ax.annotate(f"[Vx,Vy,Vz] {VARIANT_SHORT[v]}", tq,
                        textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color=VEL_ARROW, alpha=alpha)
            pts_scr.append(tq)

        if self.show_legends:
            self._draw_legends()
        self._finish_scheme_axes(corners, extra=pts_scr)

    def _finish_scheme_axes(self, corners, extra):
        pts = np.vstack([corners] + [np.atleast_2d(p) for p in extra])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.12 * (hi - lo).max()
        self.ax.set_xlim(lo[0] - pad, hi[0] + pad)
        self.ax.set_ylim(lo[1] - pad, hi[1] + pad)

    def _draw_legends(self, ax=None):
        ax = ax or self.ax
        color_handles = [
            Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                   markerfacecolor=POINT_COLORS[el],
                   markeredgecolor=POINT_COLORS[el], label=el)
            for el in KEYPOINTS]
        leg1 = ax.legend(handles=color_handles, title="element (colour)",
                         loc="upper right", fontsize=7, title_fontsize=7)
        ax.add_artist(leg1)

        marker_handles = []
        for v in VARIANTS:
            style = VARIANT_MARKERS[v]
            marker_handles.append(Line2D(
                [0], [0], marker=style["marker"], linestyle="none", markersize=9,
                markeredgecolor="#444444", markeredgewidth=1.4,
                markerfacecolor=("#444444" if style["filled"] else "none"),
                label=v))
        ax.legend(handles=marker_handles, title="variant (marker)",
                  loc="lower right", fontsize=7, title_fontsize=7)

    # ------------------------------------------- 3D interactive scheme (c) #
    @staticmethod
    def _to3d(p):
        """Scene (x, y, z) -> Axes3D data coords (z, x, -y)."""
        p = np.asarray(p, dtype=float)
        return np.array([p[2], p[0], -p[1]])

    def _render_pinhole_3d(self):
        ax = self.ax3d
        keep = None
        if self._scene3d_drawn and self._scene3d_kind == "scheme":
            keep = (ax.elev, ax.azim, ax.roll, ax.get_xlim(), ax.get_ylim(),
                    ax.get_zlim())
        self._scene3d_kind = "scheme"
        ax.clear()
        ax.set_proj_type("ortho")      # parallel projection, like view (b)
        ax.set_axis_off()
        if self.view == "edit":
            ax.set_title("Change GT — keys move the ground-truth aircraft "
                         "(virtual, not to scale)")
        else:
            ax.set_title("Pinhole camera scheme — 3D interactive "
                         "(virtual, not to scale)")

        plane, Qc, map3d, map2d = self._scheme_geometry()
        pts = []

        def P3(p):
            q = self._to3d(p)
            pts.append(q)
            return q

        # Image plane.
        poly = [self._to3d(c) for c in plane]
        ax.add_collection3d(Poly3DCollection(
            [poly], facecolor="#aec7e8", edgecolor="#333333", alpha=0.40,
            linewidths=1.0))
        for c in plane:
            pts.append(self._to3d(c))

        # Camera centre.
        o = P3([0.0, 0.0, 0.0])
        ax.scatter(*o, s=70, color="#9ecae1", edgecolor="#333333", depthshade=False)
        ax.text(o[0], o[1], o[2], "  [0,0,0]", fontsize=9)

        if Qc is None:
            self._finish_axes3d(pts, keep)
            return

        # Range line. The label lists every checked variant so e.g. Range
        # noise is visible immediately.
        rng_label = self._range_label_text()
        if rng_label is not None:
            c3 = P3(Qc)
            ax.plot([o[0], c3[0]], [o[1], c3[1]], [o[2], c3[2]],
                    color="#444444", lw=0.8)
            mid = 0.6 * o + 0.4 * c3
            ax.text(mid[0], mid[1], mid[2], "  " + rng_label, fontsize=8)

        ray_colors = {"P1_tail": BODY_ARROW, "P2_nose": BODY_ARROW,
                      "P3_left_wing": WING_ARROW, "P4_right_wing": WING_ARROW}
        for el in KEYPOINTS:
            color = POINT_COLORS[el]
            ray_done = False
            for v in VARIANTS:
                if not self._is_visible(el, "3D", v):
                    continue
                Pt = self.point3d(el, v)
                if Pt is None:
                    continue
                q = P3(map3d(Pt))
                st = VARIANT_MARKERS[v]
                ax.scatter(*q, s=55, marker=st["marker"], edgecolor=color,
                           facecolor=(color if st["filled"] else "none"),
                           depthshade=False, linewidths=1.5)
                if not ray_done:
                    ls = "--" if ray_colors[el] == BODY_ARROW else ":"
                    ax.plot([o[0], q[0]], [o[1], q[1]], [o[2], q[2]],
                            color=ray_colors[el], lw=1.0, linestyle=ls, alpha=0.8)
                    ax.text(q[0], q[1], q[2], "  " + el.split("_")[0], fontsize=9)
                    ray_done = True

        # 2D plane points.
        for el in KEYPOINTS:
            color = POINT_COLORS[el]
            for v in VARIANTS:
                if not self._is_visible(el, "2D", v):
                    continue
                q3 = map2d(el, v)
                if q3 is None:
                    continue
                q = P3(q3)
                st = VARIANT_MARKERS[v]
                ax.scatter(*q, s=22, marker=st["marker"], edgecolor=color,
                           facecolor=(color if st["filled"] else "none"),
                           depthshade=False, linewidths=1.0)

        # Segment arrows (3D keypoints + on-plane) via quiver.
        def quiver(a, b, color, lw):
            d = b - a
            ax.quiver(a[0], a[1], a[2], d[0], d[1], d[2], color=color,
                      linewidth=lw, arrow_length_ratio=0.18)

        for (els, col, lw, kind) in [
                (["P1_tail", "P2_nose"], BODY_ARROW, 2.6, "3D"),
                (["P3_left_wing", "P4_right_wing"], WING_ARROW, 2.0, "3D"),
                (["P1_tail", "P2_nose"], BODY_ARROW, 1.1, "2D"),
                (["P3_left_wing", "P4_right_wing"], WING_ARROW, 1.1, "2D")]:
            vv = self._priority_visible(kind, els)
            if vv is None:
                continue
            if kind == "3D":
                a = self._to3d(map3d(self.point3d(els[0], vv)))
                b = self._to3d(map3d(self.point3d(els[1], vv)))
            else:
                a = self._to3d(map2d(els[0], vv))
                b = self._to3d(map2d(els[1], vv))
            quiver(a, b, col, lw)

        # Tail fin (orientation cue): bold on the virtual aircraft, thin on
        # the plane (map3d points sit on their own pixel ray, so the
        # ray \cap plane point IS the projected fin).
        vfin = self._priority_visible("3D", ["P1_tail", "P2_nose"])
        if vfin is not None:
            seg = self._fin_segment(vfin)
            if seg is not None:
                fa, fb = P3(map3d(seg[0])), P3(map3d(seg[1]))
                ax.plot([fa[0], fb[0]], [fa[1], fb[1]], [fa[2], fb[2]],
                        color=BODY_ARROW, lw=2.0, solid_capstyle="round")
                Qa, Qb = map3d(seg[0]), map3d(seg[1])
                pa = P3(Qa * (SCHEME_FV / Qa[2]))
                pb = P3(Qb * (SCHEME_FV / Qb[2]))
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                        color=BODY_ARROW, lw=1.1, solid_capstyle="round")

        # Local aircraft axes (GT-anchored, origin at the aircraft centre).
        la = self._local_axes_virtual(map3d)
        if la is not None:
            o_v, tips = la
            oq = P3(o_v)
            for name, tip in tips.items():
                tq = P3(tip)
                quiver_color = LOCAL_AXIS_COLORS[name]
                d = tq - oq
                ax.quiver(oq[0], oq[1], oq[2], d[0], d[1], d[2],
                          color=quiver_color, linewidth=1.4,
                          arrow_length_ratio=0.2)
                ax.text(tq[0], tq[1], tq[2], f" ${name}_{{loc}}$",
                        fontsize=8, color=quiver_color)

        # Velocity (from camera centre, as in Figure 2). One arrow per checked
        # variant (non-GT fainter) so e.g. Velocity noise is visible.
        for v in self._visible_velocity_variants():
            Vel = self.velocity(v)
            alpha = 1.0 if v == "GT" else 0.65
            tip = P3(Vel / np.linalg.norm(Vel) * SCHEME_VEL_LEN)
            d = tip - o
            ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], color=VEL_ARROW,
                      linewidth=2.0, arrow_length_ratio=0.15, linestyle="--",
                      alpha=alpha)
            ax.text(tip[0], tip[1], tip[2],
                    f"  [Vx,Vy,Vz] {VARIANT_SHORT[v]}", fontsize=8,
                    color=VEL_ARROW, alpha=alpha)

        if self.show_legends:
            self._draw_legends(ax)
        help_text = EDIT_HELP if self.view == "edit" else VIEW3D_HELP
        ax.text2D(0.01, 0.99, help_text, transform=ax.transAxes, va="top",
                  fontsize=7, family="monospace", color="#555555")
        if self.view == "edit" and self.show_legends:
            ax.text2D(0.01, 0.01, EDIT_CONTROLS_BOX, transform=ax.transAxes,
                      va="bottom", ha="left", fontsize=8, family="monospace",
                      color="#222222",
                      bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8f8f2",
                                edgecolor="#888888", alpha=0.92))
        self._finish_axes3d(pts, keep)

    def _finish_axes3d(self, pts, keep):
        ax = self.ax3d
        pts = np.array(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = (hi - lo)
        pad = 0.12 * span.max()
        xlim = (lo[0] - pad, hi[0] + pad)
        ylim = (lo[1] - pad, hi[1] + pad)
        zlim = (lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0],
                           zlim[1] - zlim[0]))
        self._home3d = {"angles": VIEW3D_HOME,
                        "xlim": xlim, "ylim": ylim, "zlim": zlim}
        if keep is not None:
            ax.view_init(elev=keep[0], azim=keep[1], roll=keep[2])
            ax.set_xlim(keep[3]); ax.set_ylim(keep[4]); ax.set_zlim(keep[5])
        else:
            ax.view_init(**VIEW3D_HOME)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        self._scene3d_drawn = True

    # ---------------------------------------------- 3D results view (e) #
    def _aircraft_stats(self, variant):
        """Summary metrics of one variant's aircraft (see main.aircraft_stats).

        Shared with the solver so the on-screen table and the Results block
        saved in the JSON agree.
        """
        return algo.aircraft_stats(self.doc, variant)

    RESULTS_TABLE_COLS = [("GT", "GT"), ("Calc", "Computed")]

    def _results_table_text(self):
        """Monospace GT/Calc results table for the 3D results view."""
        stats = {lbl: self._aircraft_stats(v)
                 for lbl, v in self.RESULTS_TABLE_COLS}
        wid = 21

        def fmt_xyz(s):
            return "-" if s is None else \
                "{:.1f}, {:.1f}, {:.1f}".format(*s["xyz"])

        def fmt_ypr(s):
            return "-" if s is None or "ypr" not in s else \
                "{:.1f}, {:.1f}, {:.1f}".format(*s["ypr"])

        def fmt_val(s, key, nd):
            return "-" if s is None or key not in s else f"{s[key]:.{nd}f}"

        rows = [
            ("", [lbl for lbl, _ in self.RESULTS_TABLE_COLS]),
            ("XYZ (m)", [fmt_xyz(stats[l]) for l, _ in self.RESULTS_TABLE_COLS]),
            ("YPR (deg)", [fmt_ypr(stats[l]) for l, _ in self.RESULTS_TABLE_COLS]),
            ("BodyLength", [fmt_val(stats[l], "body_len", 2)
                            for l, _ in self.RESULTS_TABLE_COLS]),
            ("WingsSpan", [fmt_val(stats[l], "wing_span", 2)
                           for l, _ in self.RESULTS_TABLE_COLS]),
            ("CrossPoint", [fmt_val(stats[l], "cross", 3)
                            for l, _ in self.RESULTS_TABLE_COLS]),
        ]
        return "\n".join(name.ljust(11) + "".join(v.rjust(wid) for v in vals)
                         for name, vals in rows)

    def _render_results_3d(self):
        ax = self.ax3d
        keep = None
        if self._scene3d_drawn and self._scene3d_kind == "results":
            keep = (ax.elev, ax.azim, ax.roll, ax.get_xlim(), ax.get_ylim(),
                    ax.get_zlim())
        self._scene3d_kind = "results"
        ax.clear()
        ax.set_proj_type("ortho")
        ax.set_title("3D results — GT vs Computed "
                     "(camera coordinates, metres)")
        # Data axes are the _to3d permutation of camera coordinates.
        ax.set_xlabel("Z (m)")
        ax.set_ylabel("X (m)")
        ax.set_zlabel("-Y (m)")

        pts = []

        def P3(p):
            q = self._to3d(p)
            pts.append(q)
            return q

        def arrow(a, b, color, lw, alpha):
            d = b - a
            ax.quiver(a[0], a[1], a[2], d[0], d[1], d[2], color=color,
                      linewidth=lw, arrow_length_ratio=0.18, alpha=alpha)

        def draw(points, alpha_pt, alpha_seg):
            """points: {el: (P, variant)} -> {el: data-coords point}."""
            q3 = {}
            for el, (P, v) in points.items():
                q = P3(P)
                q3[el] = q
                st = VARIANT_MARKERS[v]
                color = POINT_COLORS[el]
                ax.scatter(*q, s=55, marker=st["marker"], edgecolor=color,
                           facecolor=(color if st["filled"] else "none"),
                           depthshade=False, linewidths=1.5, alpha=alpha_pt)
            if "P1_tail" in q3 and "P2_nose" in q3:
                arrow(q3["P1_tail"], q3["P2_nose"], BODY_ARROW, 2.6, alpha_seg)
            if "P3_left_wing" in q3 and "P4_right_wing" in q3:
                arrow(q3["P3_left_wing"], q3["P4_right_wing"], WING_ARROW,
                      2.0, alpha_seg)
            fin = self._fin_from_points(
                *(points[el][0] if el in points else None
                  for el in ("P1_tail", "P2_nose",
                             "P3_left_wing", "P4_right_wing")))
            if fin is not None:
                fa, fb = P3(fin[0]), P3(fin[1])
                ax.plot([fa[0], fb[0]], [fa[1], fb[1]], [fa[2], fb[2]],
                        color=BODY_ARROW, lw=2.0, solid_capstyle="round",
                        alpha=alpha_seg)
            return q3

        gt = {el: (self.point3d(el, "GT"), "GT") for el in KEYPOINTS
              if self.point3d(el, "GT") is not None}
        comp = {el: (self.point3d(el, "Computed"), "Computed")
                for el in KEYPOINTS
                if self.point3d(el, "Computed") is not None}

        gt_q = draw(gt, 1.0, 1.0)
        for el, q in gt_q.items():
            ax.text(q[0], q[1], q[2], "  " + el.split("_")[0], fontsize=9)
        comp_q = draw(comp, RESULTS_PT_ALPHA, RESULTS_SEG_ALPHA)

        # Dotted GT -> computed connectors.
        for el in KEYPOINTS:
            if el in gt_q and el in comp_q:
                a, b = gt_q[el], comp_q[el]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color="#777777", lw=0.8, linestyle=":")

        # Results table (rows XYZ/YPR/BodyLength/WingsSpan/CrossPoint,
        # columns GT / Calc).
        ax.text2D(0.01, 0.01, self._results_table_text(),
                  transform=ax.transAxes, va="bottom", fontsize=8,
                  family="monospace", color="#222222",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f2",
                            edgecolor="#888888", alpha=0.9))
        if not comp:
            ax.text2D(0.99, 0.30, "no Computed data — press Solve",
                      transform=ax.transAxes, ha="right", fontsize=9,
                      color="#aa3333")

        if self.show_legends:
            self._draw_legends(ax)
        ax.text2D(0.01, 0.99, VIEW3D_HELP, transform=ax.transAxes, va="top",
                  fontsize=7, family="monospace", color="#555555")

        if not pts:
            ax.set_axis_off()
            ax.text2D(0.5, 0.5, "no 3D keypoints in the document",
                      transform=ax.transAxes, ha="center", fontsize=12,
                      color="#555555")
            return
        self._finish_axes3d(pts, keep)
        if keep is None:               # start from the aircraft-centric view
            view = self._view3d_aircraft45()
            if view is not None:
                self._home3d["angles"] = view
                ax.view_init(**view)

    # 3D camera controls (no re-render; they only move the camera).
    def _zoom3d(self, factor):
        ax = self.ax3d
        for get, setr in [(ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim),
                          (ax.get_zlim, ax.set_zlim)]:
            lo, hi = get()
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
            setr(c - h, c + h)

    def _pan3d(self, dx, dy):
        ax = self.ax3d
        e, a = np.deg2rad(ax.elev), np.deg2rad(ax.azim)
        view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, view)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        sup = np.cross(view, right)
        xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        rng = max(xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0])
        s = (dx * right + dy * sup) * VIEW3D_PAN_STEP * rng
        ax.set_xlim(xl[0] + s[0], xl[1] + s[0])
        ax.set_ylim(yl[0] + s[1], yl[1] + s[1])
        ax.set_zlim(zl[0] + s[2], zl[1] + s[2])

    def _reset3d(self):
        ax, h = self.ax3d, self._home3d
        ax.view_init(**h["angles"])
        ax.set_xlim(h["xlim"]); ax.set_ylim(h["ylim"]); ax.set_zlim(h["zlim"])

    def _view3d_aircraft45(self):
        """Preset "4": aircraft-centric camera angles, or None without GT.

        The camera direction is set in the GT local frame (VIEW3D_AIRCRAFT_AZ
        off the tail, VIEW3D_AIRCRAFT_ELEV above the aircraft's horizontal
        plane, so the image plane behind the aircraft stays in view) and the
        screen-up is aligned with local Z. Local Z is generally not the Axes3D z axis, so
        this needs a roll angle: with matplotlib's convention, roll is *minus*
        the geometric angle from the roll-0 up vector to the desired up,
        measured about the viewing axis.
        """
        if any(self.point3d(el, "GT") is None for el in KEYPOINTS):
            return None
        frame = AircraftFrame.from_doc(self.doc, variant="GT")
        az = np.deg2rad(VIEW3D_AIRCRAFT_AZ)
        el = np.deg2rad(VIEW3D_AIRCRAFT_ELEV)
        w_local = np.array([np.cos(el) * np.sin(az),
                            -np.cos(el) * np.cos(az),
                            np.sin(el)])
        w = self._to3d(frame.R @ w_local)    # centre -> camera, data coords
        up = self._to3d(frame.R[:, 2])       # local Z in data coords
        e = np.arcsin(np.clip(w[2], -1.0, 1.0))
        a = np.arctan2(w[1], w[0])
        u0 = np.array([-np.sin(e) * np.cos(a),   # roll-0 screen up
                       -np.sin(e) * np.sin(a),
                       np.cos(e)])
        dp = up - np.dot(up, w) * w
        dp /= np.linalg.norm(dp)
        roll = -np.arctan2(np.dot(np.cross(u0, dp), w), np.dot(u0, dp))
        return {"elev": float(np.degrees(e)), "azim": float(np.degrees(a)),
                "roll": float(np.degrees(roll))}

    # ------------------------------------------- Change-GT editing (view d) #
    def _gt_snapshot(self):
        return copy.deepcopy({
            "p3": {el: self.doc["Coordinates_3D"][el]["GT"]
                   for el in KEYPOINTS},
            "p2": {el: self.doc["Coordinates_2D"][ELEMENT_2D[el]]["GT"]
                   for el in KEYPOINTS},
            "vel": self.doc["Velocity_vector_3D"]["GT"],
            "rng": self.doc["Range_meters"]["GT"],
        })

    def _push_undo(self):
        self._undo_stack.append(self._gt_snapshot())
        if len(self._undo_stack) > GT_UNDO_MAX:
            self._undo_stack.pop(0)

    def _undo_gt(self):
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        for el in KEYPOINTS:
            self.doc["Coordinates_3D"][el]["GT"] = snap["p3"][el]
            self.doc["Coordinates_2D"][ELEMENT_2D[el]]["GT"] = snap["p2"][el]
        self.doc["Velocity_vector_3D"]["GT"] = snap["vel"]
        self.doc["Range_meters"]["GT"] = snap["rng"]
        self.render()
        self.fig.canvas.draw_idle()

    def _refresh_gt_derived(self):
        """Re-derive GT 2D projections and GT range from the GT 3D points."""
        fx, fy, cx, cy = aio.camera_intrinsics(self.doc["Camera_parameters"])
        for el in KEYPOINTS:
            P = self.point3d(el, "GT")
            if P is not None and P[2] > 1e-9:
                self.doc["Coordinates_2D"][ELEMENT_2D[el]]["GT"] = \
                    aio.vec2(aio.project_pixel(P, fx, fy, cx, cy))
        P1 = self.point3d("P1_tail", "GT")
        P2 = self.point3d("P2_nose", "GT")
        if P1 is not None and P2 is not None:
            self.doc["Range_meters"]["GT"] = \
                aio.scalar(np.linalg.norm(0.5 * (P1 + P2)))

    def _move_gt(self, t_local=None, axis_local=None, angle_deg=0.0):
        """Rigid motion of the GT aircraft, expressed in its LOCAL frame.

        Either translate by ``t_local`` metres or rotate ``angle_deg`` about
        ``axis_local`` through the aircraft centre. GT velocity rotates with
        the body; GT 2D projections and range are re-derived.
        """
        if any(self.point3d(el, "GT") is None for el in KEYPOINTS):
            return
        self._push_undo()
        frame = AircraftFrame.from_doc(self.doc)
        c3d = self.doc["Coordinates_3D"]
        if t_local is not None:
            d = frame.R @ np.asarray(t_local, dtype=float)
            for el in KEYPOINTS:
                c3d[el]["GT"] = aio.vec3(self.point3d(el, "GT") + d)
        else:
            Rot = _rotation_about(frame.R @ np.asarray(axis_local, dtype=float),
                                  np.deg2rad(angle_deg))
            C = frame.origin
            for el in KEYPOINTS:
                c3d[el]["GT"] = aio.vec3(C + Rot @ (self.point3d(el, "GT") - C))
            V = self.velocity("GT")
            if V is not None:
                self.doc["Velocity_vector_3D"]["GT"] = aio.vel(Rot @ V)
        self._refresh_gt_derived()
        self.render()
        self.fig.canvas.draw_idle()

    def _shift_wings_gt(self, d_cross):
        """Slide the GT wing pair along the body axis (changes CrossPoint).

        ``d_cross`` is in CrossPoint units (0 = tail, 1 = nose); the result
        is clamped to [0, 1]. A non-rigid GT edit: only P3/P4 move; the body,
        velocity and range stay put, and the wings' GT 2D projections are
        re-derived.
        """
        P1 = self.point3d("P1_tail", "GT")
        P2 = self.point3d("P2_nose", "GT")
        P3w = self.point3d("P3_left_wing", "GT")
        P4w = self.point3d("P4_right_wing", "GT")
        if any(p is None for p in (P1, P2, P3w, P4w)):
            return
        body = P2 - P1
        L2 = float(np.dot(body, body))
        if L2 < 1e-12:
            return
        Wm = 0.5 * (P3w + P4w)
        cur = float(np.dot(Wm - P1, body) / L2)
        new = float(np.clip(cur + d_cross, 0.0, 1.0))
        if abs(new - cur) < 1e-12:
            return                      # already at the clamp limit
        self._push_undo()
        shift = (new - cur) * body
        for el in ("P3_left_wing", "P4_right_wing"):
            self.doc["Coordinates_3D"][el]["GT"] = \
                aio.vec3(self.point3d(el, "GT") + shift)
        self._refresh_gt_derived()
        self.render()
        self.fig.canvas.draw_idle()

    def _scale_wings_gt(self, factor):
        """Scale the GT wingspan about the wings midpoint.

        Span changes by ``factor``; the wings midpoint (and so CrossPoint),
        the body, velocity and range stay put. Clamped at GT_SIZE_MIN.
        """
        P3w = self.point3d("P3_left_wing", "GT")
        P4w = self.point3d("P4_right_wing", "GT")
        if P3w is None or P4w is None:
            return
        span = float(np.linalg.norm(P4w - P3w))
        if span < 1e-12 or factor * span < GT_SIZE_MIN:
            return
        self._push_undo()
        Wm = 0.5 * (P3w + P4w)
        for el, P in (("P3_left_wing", P3w), ("P4_right_wing", P4w)):
            self.doc["Coordinates_3D"][el]["GT"] = \
                aio.vec3(Wm + factor * (P - Wm))
        self._refresh_gt_derived()
        self.render()
        self.fig.canvas.draw_idle()

    def _scale_body_gt(self, factor):
        """Scale the GT body about its midpoint, keeping the wings' station.

        Body length changes by ``factor`` around the (fixed) centre, so the
        range is untouched. The wing pair is shifted so its CrossPoint
        parameter is preserved -- changing the length does not change where
        the wings sit on the fuselage. Clamped at GT_SIZE_MIN.
        """
        P1 = self.point3d("P1_tail", "GT")
        P2 = self.point3d("P2_nose", "GT")
        P3w = self.point3d("P3_left_wing", "GT")
        P4w = self.point3d("P4_right_wing", "GT")
        if any(p is None for p in (P1, P2, P3w, P4w)):
            return
        body = P2 - P1
        L2 = float(np.dot(body, body))
        if L2 < 1e-12 or factor * np.sqrt(L2) < GT_SIZE_MIN:
            return
        self._push_undo()
        C = 0.5 * (P1 + P2)
        t = float(np.dot(0.5 * (P3w + P4w) - P1, body) / L2)   # wings station
        P1n, P2n = C + factor * (P1 - C), C + factor * (P2 - C)
        self.doc["Coordinates_3D"]["P1_tail"]["GT"] = aio.vec3(P1n)
        self.doc["Coordinates_3D"]["P2_nose"]["GT"] = aio.vec3(P2n)
        shift = (P1n + t * (P2n - P1n)) - 0.5 * (P3w + P4w)
        for el, P in (("P3_left_wing", P3w), ("P4_right_wing", P4w)):
            self.doc["Coordinates_3D"][el]["GT"] = aio.vec3(P + shift)
        self._refresh_gt_derived()
        self.render()
        self.fig.canvas.draw_idle()

    def _on_key_edit(self, k):
        s, r = GT_MOVE_STEP, GT_ROT_STEP
        if k == "ctrl+z":
            self._undo_gt()
            return
        if k in ("w", "s"):               # wings slide along the body
            self._shift_wings_gt(GT_CROSS_STEP if k == "w" else -GT_CROSS_STEP)
            return
        if k in ("a", "d"):               # wingspan
            self._scale_wings_gt(GT_SCALE_STEP if k == "d"
                                 else 1.0 / GT_SCALE_STEP)
            return
        if k in ("q", "e"):               # body length
            self._scale_body_gt(GT_SCALE_STEP if k == "e"
                                else 1.0 / GT_SCALE_STEP)
            return
        if k in ("0", "home"):            # reset camera to the entry view
            view = self._view3d_aircraft45()
            if view is not None:
                h = self._home3d
                self.ax3d.view_init(**view)
                self.ax3d.set_xlim(h["xlim"])
                self.ax3d.set_ylim(h["ylim"])
                self.ax3d.set_zlim(h["zlim"])
                self.fig.canvas.draw_idle()
            return
        trans = {"left": [-s, 0, 0], "right": [s, 0, 0],
                 "up": [0, 0, s], "down": [0, 0, -s],
                 "+": [0, s, 0], "=": [0, s, 0],
                 "-": [0, -s, 0], "_": [0, -s, 0]}
        rots = {"shift+left": ([0, 0, 1], +r),    # yaw: nose to the left
                "shift+right": ([0, 0, 1], -r),
                "shift+up": ([1, 0, 0], +r),      # pitch: nose up
                "shift+down": ([1, 0, 0], -r),
                "ctrl+right": ([0, 1, 0], +r),    # roll: right wing down
                "ctrl+left": ([0, 1, 0], -r)}
        if k in trans:
            self._move_gt(t_local=trans[k])
        elif k in rots:
            axis, ang = rots[k]
            self._move_gt(axis_local=axis, angle_deg=ang)

    def _on_key(self, event):
        if event.key is None:
            return
        if self.view == "edit":
            self._on_key_edit(event.key)
            return
        if self.view not in ("3d", "results"):
            return
        k, ax = event.key, self.ax3d
        if k in ("0", "home"):
            self._reset3d()
        elif k in VIEW3D_PRESETS:
            ax.view_init(**VIEW3D_PRESETS[k])
        elif k == "4":
            view = self._view3d_aircraft45()
            if view is None:
                return
            ax.view_init(**view)
        elif k == "left":
            self._pan3d(-1, 0)
        elif k == "right":
            self._pan3d(1, 0)
        elif k == "up":
            self._pan3d(0, 1)
        elif k == "down":
            self._pan3d(0, -1)
        elif k == "shift+left":
            ax.view_init(ax.elev, ax.azim - VIEW3D_ROT_STEP, ax.roll)
        elif k == "shift+right":
            ax.view_init(ax.elev, ax.azim + VIEW3D_ROT_STEP, ax.roll)
        elif k == "shift+up":
            ax.view_init(ax.elev + VIEW3D_ROT_STEP, ax.azim, ax.roll)
        elif k == "shift+down":
            ax.view_init(ax.elev - VIEW3D_ROT_STEP, ax.azim, ax.roll)
        elif k in ("+", "="):
            self._zoom3d(VIEW3D_ZOOM_STEP)
        elif k in ("-", "_"):
            self._zoom3d(1.0 / VIEW3D_ZOOM_STEP)
        else:
            return
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if self.view not in ("3d", "edit", "results") \
                or event.inaxes is not self.ax3d:
            return
        self._zoom3d(VIEW3D_ZOOM_STEP if event.button == "up"
                     else 1.0 / VIEW3D_ZOOM_STEP)
        self.fig.canvas.draw_idle()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = AircraftViewer(path)
    viewer.build_figure()
    plt.show()


if __name__ == "__main__":
    main()
