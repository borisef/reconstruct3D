# Change log

Implementation of the paper **"3D-2D analysis of aircraft"** (single-image 3D
reconstruction of an aircraft modeled as a 4-keypoint cross).

## 2026-11-04

### Environment
- Created virtual environment at `/home/borisef/Envs/reconstruct3D`
  (Python 3.10.20).
- Installed dependencies: `numpy==2.2.6`, `scipy==1.15.3`.
- Froze them to `requirements.txt`.

### `reconstruction_utils.py` (new)
Every solution/derivation from the paper, each as a separate pure function:
- `focal_length_from_fov(W_pixel, fov, S=1.0)` — Appendix, `f = W S / (2 tan(FOV/2))`.
- `normalize_coords(x_pixel, y_pixel, cx, cy, f)` — Appendix, pixel → normalized `(u, v)`.
- `body_solution1(u1, v1, u2, v2, V, D)` — Body **Solution #1** (direction first, then
  range): 3×2 least-squares system in scaled depths, scale fixed by range `D`.
- `body_solution2(u1, v1, u2, v2, V, D)` — Body **Solution #2** (range first, then
  direction): 6×3 overcomplete least-squares system for `(lambda1, lambda2, h)`.
- `wings_solution(u3, v3, u4, v4, V, D)` — Wings: perpendicular-to-`V` + range constraint
  reduced to the quadratic for `lambda3`, then `lambda4 = lambda3 * a`.
- `compute_6dof(P1, P2, P3, P4)` — translation, rotation matrix `R` (columns x=wing dir,
  y=body dir, z=x×y), Rodrigues vector (`scipy`), and pitch/yaw/roll via the paper's
  arctan2 formulas.
- Helpers: `_ray`, `_lstsq` (explicit normal equations to mirror the paper), `_normalize`.

### `main.py` (replaced PyCharm boilerplate)
- Builds a synthetic ground-truth aircraft (~2 km range, small FOV), projects its
  keypoints with the pinhole model, then runs every function and reports recovered vs.
  true values as a round-trip sanity check.

### Verification
Ran `main.py` with the venv. Results:
- Body Solution #1: `|err| ~ 1e-10` for P1, P2.
- Body Solution #2: `|err| ~ 8e-4` (expected — the 2D-midpoint-ray approximation differs
  slightly from the exact 3D midpoint projection); agrees with Solution #1 to the same
  tolerance.
- Wings: `|err| ~ 1e-13`; `(P3-P4)·V ~ 1e-15` (perpendicularity holds).
- 6DOF: translation, Rodrigues vector, and pitch/yaw/roll all match ground truth.
- Derived body length (30 m) and wingspan (40 m) recovered exactly.

### Docs layout
- Created `docs/` folder and moved `log_changes.md` into it; all Markdown files
  live under `docs/` from now on.

### Paper rewritten as Markdown
- Added `docs/3D-2D_analysis_of_aircraft.md` — a clean, corrected rewrite of the
  source PDF (`~/Downloads/3D-2D  analysis of aircraft.pdf`) with proper LaTeX
  math, a table of contents, and a "Corrections vs. original" section.
- Corrections applied vs. the original: fixed the 6-DOF `y`-axis formula
  (`(P2-P1)/‖P2-P1‖`, was an inconsistent `(P1-P0)`); used `u,v` consistently
  (original pinhole figure had a TODO about `x,y` vs `u,v`); unified the
  Solution #2 right-hand side casing; unified notation.
- Extracted the four figures from the PDF into `assets/` (`figure1.png`,
  `figure2.png`, `figure3.png`, `figure_pinhole.png`) via `pdftoppm` (150 DPI
  page render) + Pillow crop, and referenced them from the Markdown.

### JSON I/O (input document)
Introduced a JSON-driven interface. One document holds five top-level blocks:
`Camera_parameters` (`fov_X_deg`, `fov_Y_deg`, `H_pixels`, `W_pixels`),
`Coordinates_3D` (`P1_tail`, `P2_nose`, `P3_left_wing`, `P4_right_wing`),
`Coordinates_2D` (`p1_tail`, `p2_nose`, `p3_left_wing`, `p4_right_wing`, in
**pixels**), `Velocity_vector_3D`, and `Range_meters` (scalar range to the body
midpoint). Every entry carries five variants — `GT`, `Noisy_GT`, `Computed1`,
`Computed2`, `Computed3` — any of which may be `null`.
- `aircraft_io.py` (new): schema constants, value builders (`vec3`/`vec2`/`vel`/
  `scalar`/`variant_block`), camera model (`camera_intrinsics`, `project_pixel`,
  `pixel_to_normalized`), and `load_json`/`save_json`. Reused by the later solve
  step.
- `generate_input.py` (new): builds the synthetic ground truth (~2 km range,
  30 m body / 40 m wingspan, now centered near the optical axis so all keypoints
  land inside the 1920×1080 frame), projects to pixel 2D, derives `V` and the
  range, adds deterministic Gaussian noise (`default_rng(0)`; σ = 2 m / axis 3D,
  1.5 px / axis 2D, 0.01 / axis for `V` then re-normalized, 10 m range) for
  `Noisy_GT`, and writes `inputs/aircraft_input.json` with all `Computed*` null.
- Naming note: the request listed `P2_tail`; since `P1` is the tail this is a
  typo — the schema uses `P2_nose` (the paper's nose keypoint).
- Computed-slot mapping: 3D body → `body_solution1`→`Computed1`,
  `body_solution2`→`Computed2`; 3D wings → `wings_solution`→`Computed1`. All 2D
  `Computed*` and all `Computed3` stay null for now.
- Verified: generator runs with the venv, `inputs/aircraft_input.json` parses,
  all GT 2D pixels lie within the frame, `Range_meters.GT` ≈ 2000 m, `Noisy_GT`
  differs slightly from `GT`, and all `Computed*` are null.

### JSON solve driver (`main.py` rewritten)
Replaced the synthetic round-trip demo with a JSON-driven solver/example.
- `main.py` is now a CLI: `python main.py [INPUT] [OUTPUT] [--source VARIANT]`,
  defaulting to `inputs/aircraft_input.json` → `outputs/aircraft_output.json`,
  `--source Noisy_GT`.
- `solve_document(doc, source)`: reads the chosen variant's 2D pixel keypoints,
  velocity and range; converts pixels → normalized via `aircraft_io`; runs
  `body_solution1/2` and `wings_solution`; fills the Computed slots per the
  mapping above. Output JSON is the **exact same schema as the input, extended**
  — all other blocks/variants carried through unchanged. (The single body
  `Range_meters` serves the wings too: wing midpoint == body midpoint ==
  aircraft centre.)
- Verified: solving from `--source GT` recovers the GT 3D points to ~1e-3 m
  (sanity check); solving from `Noisy_GT` gives ~3.5 m errors (consistent with
  the injected noise); output top-level/nested keys are identical to the input,
  `GT`/`Noisy_GT` preserved, body `Computed1`/`Computed2` and wing `Computed1`
  filled, 2D `Computed*` and all `Computed3` still null.

### GUI (`GUI/viewer.py`) — matplotlib viewer
Added a matplotlib-based viewer (toolkit chosen: matplotlib-only, one new
dependency; `matplotlib==3.10.9` added to `requirements.txt`). Run with
`python GUI/viewer.py [PATH_TO_JSON]`.
- **Controls** (left panel):
  - A compact **variant × element visibility grid** (custom `CheckGrid` widget).
    Rows = variants (`GT`, `Noisy_GT`, `Computed1/2/3`), labelled once on the
    left. Columns are grouped: **3D pts** `P1..P4` (`Coordinates_3D`), **2D pts**
    `p1..p4` (`Coordinates_2D`), **Vel** (`Velocity_vector_3D`) and **Rng**
    (`Range_meters`) — so 2D / 3D / velocity / range are selectable
    independently (3D/range selections are accepted even though the 2D view does
    not draw them yet). Each cell is an independent checkbox; a cell whose field
    is null/empty is **disabled (greyed, non-clickable)**. Row labels are
    abbreviated (`GT`, `N_GT`, `Cmp1`, `Cmp2`, `Cmp3`) and the grid is tightened
    and shifted left to reduce bulk.
  - A single **"Legends"** checkbox toggling both legends at once.
  - A one-line **"View" dropdown** (custom `Dropdown` widget): `2D`,
    `Pinhole (scheme)`, `3D interactive`.
  - `Load JSON` (tkinter file dialog) and `Solve` (runs `main.solve_document`,
    source `Noisy_GT`) buttons; a status line showing the current file.
  - Note: matplotlib has no native grid-of-checkboxes or dropdown, so both are
    small custom widgets drawn on their own axes with click hit-testing.
- **Encoding** (shared by all views): point **colour** = element (orange/green/
  purple/brown); point **marker** = variant (GT filled dot, Noisy_GT open dot,
  Computed1 diamond, Computed2 square, Computed3 triangle); segment **arrows**
  tail→nose **red**, left→right wing **blue**, velocity **green**. When several
  variants of a segment are visible, the arrow uses the highest-priority visible
  one (GT > Noisy_GT > Computed1 > ...).
- **Views**: (a) **2D image plane** — implemented: pixel keypoints with image
  (v-down) convention, frame border, the two segment arrows, and the velocity
  projected from the body midpoint (300 m along V). (b) **Pinhole scheme** —
  implemented (see below). (c) **3D interactive** — implemented (see below).

### Pinhole scheme view (b) implemented
Styled after Figures 1-2 of the document: camera `[0,0,0]` bottom-left,
tilted translucent image plane (~45°), aircraft to the right, drawn via an
oblique orthographic projection (`SCHEME_M`).
- **Scale handling** (real config is undrawable: 5° FOV, 40 m aircraft, 2 km
  range): a *virtual* configuration — virtual focal distance `f_v = 1` with the
  aircraft centre at `2·f_v` along its viewing ray (camera→plane distance ==
  plane→aircraft distance, per user's midpoint idea); plane sized for an
  effective ~45° FOV. Deviation from the user's original "compress Z of each 3D
  point" suggestion (agreed in discussion): the aircraft keeps its **true
  shape** (uniform scale about its centre, wingspan = `0.85·f_v`) instead of
  anisotropic Z-compression, which would flatten the mostly-along-Z body.
  Ray consistency is guaranteed by drawing each plane point at the ray∩plane
  intersection of the camera→virtual-3D-point ray (pixel-mapped fallback for
  2D-only variants).
- **Content**, all driven by the visibility grid: 3D keypoints (full-size
  markers + `P1..P4` labels) with red-dashed body rays / blue-dotted wing rays
  for the priority variant; 2D plane points (small markers) + thin on-plane
  arrows; bold red body and blue wing 3D arrows (priority variant); green
  dashed velocity drawn from the camera centre as in Figure 2 (V ∥ body, so at
  the aircraft it would hide under the body arrow); thin "Range D = ... m" line
  camera→centre, gated by the Rng checkboxes.
- Verified headlessly: GT-only scheme matches the document composition;
  enabling `Cmp1` overlays diamonds offset from the GT dots (solver error made
  visible by the virtual scaling).

### 3D interactive view (c) implemented
Same virtual scene drawn into an `Axes3D` (second axes sharing the plot rect,
shown/hidden per view). Scene `(x,y,z)` plotted as `(z, x, -y)` so the optical
axis is horizontal and true-up points up. The axes use **orthographic**
projection (`set_proj_type('ortho')`, matching parallel view (b) — matplotlib's
`Axes3D` defaults to perspective) and the default camera is **calibrated**
(`elev=-21, azim=-58, roll=0`) so the home view reproduces the oblique
composition of view (b) to ~1e-5 (angles found by matching the scheme basis
`SCHEME_M` against matplotlib's projected axis directions). Reuses
`_scheme_geometry`; markers via `scatter`, plane via `Poly3DCollection`, arrows
via `quiver`.
- **Controls** (proposed to and accepted by the user):
  - Mouse: left-drag rotate (yaw/pitch), middle-drag pan, right-drag / scroll
    zoom (Axes3D built-ins + a scroll handler; roll untouched).
  - Keys: arrows pan; shift+arrows pitch/yaw (5° steps); `+`/`-` zoom;
    `0`/`Home` reset to default; `1` plane-from-pinhole view; `2` optical-axis-
    horizontal side view. Arrow keys are removed from matplotlib's
    back/forward keymaps so they can drive panning. A help line is shown in the
    view corner.
- Camera manipulations adjust only `view_init` / axis limits (no re-render);
  the scene is rebuilt only on data/visibility/legend changes, and the current
  camera is **preserved** across such rebuilds.
- Verified headlessly: default + both presets render correctly, zoom/rotate/pan
  keys behave, view round-trips (2D↔pinhole↔3D) raise no errors, and the camera
  survives a checkbox toggle in 3D.
- Default load: `outputs/aircraft_output.json` if present, else
  `inputs/aircraft_input.json`.
- Verified headlessly (Agg): the figure builds, the 2D view renders points,
  both arrows with correct colours/directions, the velocity arrow, and both
  legends; placeholder views render their message.
