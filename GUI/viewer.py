"""
GUI/viewer.py

Matplotlib-based viewer for the aircraft 3D-2D reconstruction JSON documents.

Controls (left panel):
  * A compact visibility grid. Rows = variants (GT, Noisy_GT, Computed1..3),
    labelled once on the left. Columns = elements, grouped:
        3D pts : P1 P2 P3 P4   (Coordinates_3D)
        2D pts : p1 p2 p3 p4   (Coordinates_2D)
        Vel Rng                (Velocity_vector_3D, Range_meters)
    Each cell is an independent checkbox. A cell whose underlying field is
    null/empty is disabled (greyed, non-clickable). 3D / velocity / range
    selections are accepted even though the current 2D view does not draw them.
  * A "Legends" checkbox toggling both legends at once.
  * A "View" dropdown (2D / Pinhole / 3D interactive) -- one line.
  * Load JSON / Solve buttons and a status line.

Visualization:
  * (a) 2D  -- implemented: keypoints on the image plane (pixels).
  * (b) Pinhole scheme -- placeholder.
  * (c) 3D interactive -- placeholder.

Encoding (shared by all views):
  * Point COLOUR identifies the element; point MARKER identifies the variant
    (GT = filled dot, Noisy_GT = open dot, Computed1 = diamond,
    Computed2 = square, Computed3 = triangle).
  * Segments are arrows: tail->nose RED, left->right wing BLUE, velocity GREEN.
    With several variants visible, the arrow uses the highest-priority visible
    one (GT > Noisy_GT > Computed1 > ...).

Run:
    python GUI/viewer.py [PATH_TO_JSON]
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (3d view)

# Make the project root importable when run as `python GUI/viewer.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import aircraft_io as aio  # noqa: E402


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

VARIANTS = aio.VARIANTS  # ["GT","Noisy_GT","Computed1","Computed2","Computed3"]
VARIANT_SHORT = {"GT": "GT", "Noisy_GT": "N_GT", "Computed1": "Cmp1",
                 "Computed2": "Cmp2", "Computed3": "Cmp3"}

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
    "GT":        {"marker": "o", "filled": True},
    "Noisy_GT":  {"marker": "o", "filled": False},
    "Computed1": {"marker": "D", "filled": True},
    "Computed2": {"marker": "s", "filled": True},
    "Computed3": {"marker": "^", "filled": True},
}

BODY_ARROW = "red"      # tail -> nose
WING_ARROW = "blue"     # left wing -> right wing
VEL_ARROW = "green"     # velocity direction

VIEW_OPTIONS = ["2D", "Pinhole (scheme)", "3D interactive"]
VIEW_KEY = {"2D": "2D", "Pinhole (scheme)": "pinhole", "3D interactive": "3d"}

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
VIEW3D_ROT_STEP = 5.0     # deg per key press
VIEW3D_PAN_STEP = 0.06    # fraction of the visible range per key press
VIEW3D_ZOOM_STEP = 0.8    # range factor per zoom-in step
VIEW3D_HELP = ("mouse:  L-drag rotate   M-drag pan   R-drag/scroll zoom\n"
               "keys:  arrows pan   shift+arrows pitch/yaw   +/- zoom\n"
               "       0/Home default   1 plane view   2 axis horizontal")

AXES_RECT = [0.365, 0.07, 0.605, 0.88]


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
        self.view = "2D"
        self.show_legends = True
        self.visible = {}
        self._init_visibility()

        self.fig = None
        self.ax = None
        self._grid = None
        self._scene3d_drawn = False
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
            return self.doc["Coordinates_3D"][key][variant] is not None
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
        entry = self.doc["Coordinates_3D"][element][variant]
        return None if entry is None else \
            np.array([entry["X"], entry["Y"], entry["Z"]])

    def velocity(self, variant):
        entry = self.doc["Velocity_vector_3D"][variant]
        return None if entry is None else \
            np.array([entry["V_x"], entry["V_y"], entry["V_z"]])

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
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._build_controls()
        self.render()
        return self.fig

    def _build_controls(self):
        self.fig.text(0.012, 0.965, "Show  (variant × element)", fontsize=10,
                      fontweight="bold")
        ax_grid = self.fig.add_axes([0.012, 0.63, 0.25, 0.30])
        row_labels = [VARIANT_SHORT[v] for v in VARIANTS]
        self._grid = CheckGrid(ax_grid, VARIANTS, row_labels, COLUMNS, GROUPS,
                               is_enabled=self._cell_enabled,
                               is_checked=self._cell_checked,
                               on_toggle=self._toggle)

        ax_leg = self.fig.add_axes([0.015, 0.55, 0.14, 0.045])
        self._leg_check = CheckButtons(ax_leg, ["Legends"], [self.show_legends])
        for lbl in self._leg_check.labels:
            lbl.set_fontsize(9)
        self._leg_check.on_clicked(self._on_legends)

        ax_load = self.fig.add_axes([0.015, 0.48, 0.085, 0.05])
        self._btn_load = Button(ax_load, "Load JSON")
        self._btn_load.on_clicked(self._on_load)
        ax_solve = self.fig.add_axes([0.11, 0.48, 0.085, 0.05])
        self._btn_solve = Button(ax_solve, "Solve")
        self._btn_solve.on_clicked(self._on_solve)

        self._status = self.fig.text(
            0.015, 0.445, self._status_text(), fontsize=7, va="top",
            family="monospace")

        # Dropdown last and near the bottom so its expanded list opens over empty
        # space (no widgets below it to intercept clicks).
        self._dropdown = Dropdown(
            self.fig, [0.015, 0.37, 0.18, 0.05], VIEW_OPTIONS, "2D",
            on_select=self._on_view)

    def _status_text(self):
        return f"file: {os.path.relpath(self.path, _ROOT)}"

    # -------------------------------------------------------------- events #
    def _on_legends(self, _label):
        self.show_legends = not self.show_legends
        self.render()
        self.fig.canvas.draw_idle()

    def _on_view(self, label):
        self.view = VIEW_KEY[label]
        self.render()
        self.fig.canvas.draw_idle()

    def _on_load(self, _event):
        path = self._ask_open()
        if not path:
            return
        self.path = path
        self.doc = aio.load_json(path)
        self._init_visibility()
        self._status.set_text(self._status_text())
        self._grid.draw()
        self.render()
        self.fig.canvas.draw_idle()

    def _on_solve(self, _event):
        try:
            from main import solve_document
            solve_document(self.doc, source="Noisy_GT")
            self._sync_visibility()
            self._status.set_text(self._status_text() + "  [solved]")
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self._status.set_text(f"solve failed: {exc}")
        self._grid.draw()
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

    # ------------------------------------------------------------- render #
    def render(self):
        if self.view == "3d":
            self.ax.set_visible(False)
            self.ax3d.set_visible(True)
            self._render_pinhole_3d()
            return
        self.ax3d.set_visible(False)
        self.ax.set_visible(True)
        self.ax.clear()
        if self.view == "2D":
            self._render_2d()
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

    def _draw_velocity_arrow(self):
        for v in VARIANTS:
            if not self._is_visible("velocity", "3D", v):
                continue
            P1, P2, Vel = (self.point3d("P1_tail", v),
                           self.point3d("P2_nose", v), self.velocity(v))
            if P1 is None or P2 is None or Vel is None:
                continue
            mid = 0.5 * (P1 + P2)
            tip = mid + Vel / np.linalg.norm(Vel) * 300.0   # 300 m along V
            fx, fy, cx, cy = aio.camera_intrinsics(self.doc["Camera_parameters"])
            a = aio.project_pixel(mid, fx, fy, cx, cy)
            b = aio.project_pixel(tip, fx, fy, cx, cy)
            self.ax.annotate("", xy=tuple(b), xytext=tuple(a),
                             arrowprops=dict(arrowstyle="-|>", color=VEL_ARROW,
                                             lw=2.0), zorder=2)
            return

    # ------------------------------------------------- pinhole scheme (b) #
    def _scheme_ref_variant(self):
        """Variant whose data anchors the virtual geometry (priority order)."""
        for v in VARIANTS:
            if all(self.point3d(el, v) is not None for el in KEYPOINTS):
                return v
        return None

    def _scheme_geometry(self):
        """Virtual-space constructs for the scheme (see SCHEME_* constants).

        Returns (plane_corners 4x3, Qc centre 3, map3d, map2d) where
        map3d(P) -> virtual 3D point and map2d(el, variant) -> point on the
        virtual plane (ray intersection; pixel-mapped fallback), or None.
        """
        cam = self.doc["Camera_parameters"]
        W, H = cam["W_pixels"], cam["H_pixels"]
        fv = SCHEME_FV
        a = fv * np.tan(SCHEME_HALF_FOV)          # plane half-width
        b = a * H / W                             # plane half-height
        plane = np.array([[-a, -b, fv], [a, -b, fv], [a, b, fv], [-a, b, fv]])

        ref = self._scheme_ref_variant()
        if ref is None:
            return plane, None, None, None

        P = {el: self.point3d(el, ref) for el in KEYPOINTS}
        C3 = 0.5 * (P["P1_tail"] + P["P2_nose"])
        span = np.linalg.norm(P["P4_right_wing"] - P["P3_left_wing"])
        s_air = SCHEME_SPAN * fv / span

        # Centre ray: map the centre's pixel offset linearly into the plane,
        # then put the virtual aircraft centre at twice that point (so the
        # camera->plane and plane->aircraft distances are equal along the ray).
        fx, fy, cx, cy = aio.camera_intrinsics(cam)
        c_px = aio.project_pixel(C3, fx, fy, cx, cy)
        hit = np.array([(c_px[0] - cx) / (W / 2.0) * a,
                        (c_px[1] - cy) / (H / 2.0) * b, fv])
        Qc = 2.0 * hit

        def map3d(Pt):
            return Qc + s_air * (np.asarray(Pt) - C3)

        def map2d(el, variant):
            Pt = self.point3d(el, variant)
            if Pt is not None:                    # ray \cap plane (consistent)
                Q = map3d(Pt)
                if Q[2] > 1e-9:
                    return Q * (fv / Q[2])
                return None
            px = self.point2d(el, variant)        # 2D-only fallback
            if px is None:
                return None
            return np.array([(px[0] - cx) / (W / 2.0) * a,
                             (px[1] - cy) / (H / 2.0) * b, fv])

        return plane, Qc, map3d, map2d

    @staticmethod
    def _scr(p):
        """Oblique orthographic projection to the drawing plane."""
        return SCHEME_M @ np.asarray(p, dtype=float)

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

        # Range line camera -> aircraft centre (Rng checkbox).
        c_scr = self._scr(Qc)
        rng_v = next((v for v in VARIANTS
                      if self._cell_checked(self._col_index("range", "scalar"), v)
                      and self.doc["Range_meters"][v] is not None), None)
        if rng_v is not None:
            D = self.doc["Range_meters"][rng_v]
            ax.plot([o[0], c_scr[0]], [o[1], c_scr[1]], color="#444444",
                    lw=0.8, zorder=2)
            mid = 0.72 * o + 0.28 * c_scr
            ax.annotate(f"Range D = {D:.0f} m", mid,
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

        # Velocity (green dashed). Drawn from the camera centre, as in Figure 2
        # of the document -- V is parallel to the body, so drawing it at the
        # aircraft would hide it under the red body arrow.
        for v in VARIANTS:
            if not self._is_visible("velocity", "3D", v):
                continue
            Vel = self.velocity(v)
            if Vel is None:
                continue
            tip3 = Vel / np.linalg.norm(Vel) * SCHEME_VEL_LEN
            bq, tq = o, self._scr(tip3)
            ax.annotate("", xy=tuple(tq), xytext=tuple(bq),
                        arrowprops=dict(arrowstyle="-|>", color=VEL_ARROW,
                                        lw=2.0, linestyle="--"), zorder=4)
            ax.annotate("[Vx,Vy,Vz]", tq, textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color=VEL_ARROW)
            pts_scr.append(tq)
            break

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
        if self._scene3d_drawn:
            keep = (ax.elev, ax.azim, ax.get_xlim(), ax.get_ylim(),
                    ax.get_zlim())
        ax.clear()
        ax.set_proj_type("ortho")      # parallel projection, like view (b)
        ax.set_axis_off()
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

        # Range line.
        rng_v = next((v for v in VARIANTS
                      if self._cell_checked(self._col_index("range", "scalar"), v)
                      and self.doc["Range_meters"][v] is not None), None)
        if rng_v is not None:
            c3 = P3(Qc)
            ax.plot([o[0], c3[0]], [o[1], c3[1]], [o[2], c3[2]],
                    color="#444444", lw=0.8)
            mid = 0.6 * o + 0.4 * c3
            ax.text(mid[0], mid[1], mid[2],
                    f"  Range D = {self.doc['Range_meters'][rng_v]:.0f} m",
                    fontsize=8)

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

        # Velocity (from camera centre, as in Figure 2).
        for v in VARIANTS:
            if not self._is_visible("velocity", "3D", v):
                continue
            Vel = self.velocity(v)
            if Vel is None:
                continue
            tip = P3(Vel / np.linalg.norm(Vel) * SCHEME_VEL_LEN)
            d = tip - o
            ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], color=VEL_ARROW,
                      linewidth=2.0, arrow_length_ratio=0.15, linestyle="--")
            ax.text(tip[0], tip[1], tip[2], "  [Vx,Vy,Vz]", fontsize=8,
                    color=VEL_ARROW)
            break

        if self.show_legends:
            self._draw_legends(ax)
        ax.text2D(0.01, 0.99, VIEW3D_HELP, transform=ax.transAxes, va="top",
                  fontsize=7, family="monospace", color="#555555")
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
            ax.view_init(elev=keep[0], azim=keep[1])
            ax.set_xlim(keep[2]); ax.set_ylim(keep[3]); ax.set_zlim(keep[4])
        else:
            ax.view_init(**VIEW3D_HOME)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        self._scene3d_drawn = True

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

    def _on_key(self, event):
        if self.view != "3d" or event.key is None:
            return
        k, ax = event.key, self.ax3d
        if k in ("0", "home"):
            self._reset3d()
        elif k in VIEW3D_PRESETS:
            ax.view_init(**VIEW3D_PRESETS[k])
        elif k == "left":
            self._pan3d(-1, 0)
        elif k == "right":
            self._pan3d(1, 0)
        elif k == "up":
            self._pan3d(0, 1)
        elif k == "down":
            self._pan3d(0, -1)
        elif k == "shift+left":
            ax.view_init(ax.elev, ax.azim - VIEW3D_ROT_STEP)
        elif k == "shift+right":
            ax.view_init(ax.elev, ax.azim + VIEW3D_ROT_STEP)
        elif k == "shift+up":
            ax.view_init(ax.elev + VIEW3D_ROT_STEP, ax.azim)
        elif k == "shift+down":
            ax.view_init(ax.elev - VIEW3D_ROT_STEP, ax.azim)
        elif k in ("+", "="):
            self._zoom3d(VIEW3D_ZOOM_STEP)
        elif k in ("-", "_"):
            self._zoom3d(1.0 / VIEW3D_ZOOM_STEP)
        else:
            return
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if self.view != "3d" or event.inaxes is not self.ax3d:
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
