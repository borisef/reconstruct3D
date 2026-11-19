# Change log

Implementation of the paper **"3D-2D analysis of aircraft"** (single-image 3D
reconstruction of an aircraft modeled as a 4-keypoint cross).

## 2026-11-18

### GUI: live corrected velocity, velocity arrows everywhere, mean 3D error
- The `Computed` velocity slot now holds the **live** 2D-corrected velocity:
  the viewer recomputes it every render from the parameter-selected source
  (Noisy_GT / GT), so noise presses and GT edits update it immediately without
  pressing Solve. A full Solve still writes the same value, so the slot stays
  consistent with the solution. Extracted `main.corrected_velocity(doc, source)`
  as the single definition `solve_document` and the GUI both call;
  `AircraftViewer._refresh_corrected_velocity()` runs it at init, on load and at
  the top of `render()`.
- **2D view** now draws a velocity arrow for every checked variant
  (GT / Noisy_GT / Computed). Noisy_GT carries no 3D keypoints, so its arrow
  (and the Computed one) anchor on the GT body midpoint via the new
  `_velocity_anchor_3d`; previously only GT rendered.
- **3D results view** draws the velocity arrows too (checked variants, 0.8
  body-lengths long from the body midpoint) and the results table gains a single
  `MeanErr (m)` number — the mean `|GT - Computed|` over the four 3D keypoints,
  from the new `main.keypoint_errors_3d(doc)` (also surfaced in the saved
  `Results` block as `errors_3D_m` / `mean_error_3D_m`). Shows `-` before a solve.
- Velocity-arrow visibility stays controlled by the existing `Vel` cells in the
  visibility grid.
- Tests: `corrected_velocity` / `keypoint_errors_3d` in `test_solve.py`; the
  `MeanErr` line and live-velocity/anchor behaviour in `test_viewer_results.py`
  and `test_viewer_noise.py`; the `Results`-schema and 2D-noise coupling
  assertions in `test_viewer_algo_params.py` / `test_viewer_noise.py` updated for
  the new keys and the live corrected velocity.

### `corrected_by_2D_vel_vec`: fix a noisy velocity with the clean 2D body axis
- New algorithm parameter `velocity_source` (`vel_vec` default /
  `corrected_by_2D_vel_vec`) in `main.DEFAULT_ALGO_PARAMS`, listed in
  `main.VELOCITY_OPTIONS` and validated in `algo_params`. It composes with every
  head/tail solution (preprocessing step, not a separate algorithm).
- `reconstruction_utils.correct_velocity_with_2d(V, du, dv)` replaces the noisy
  velocity's image-plane azimuth with the labelled tail->nose direction
  `(u2-u1, v2-v1)` while keeping its depth (Vz) tilt and magnitude; returns `V`
  unchanged near nose-on (either the in-plane speed or the 2D step vanishing).
- `main.solve_document` computes the corrected vector after reading the velocity,
  always stores it in the `Computed` velocity slot (for visualization), and
  feeds it to the solvers only when `corrected_by_2D_vel_vec` is selected.
- GUI: "Set Algo Params" gains a "Velocity" dropdown (persisted with the other
  params); the corrected velocity is drawn in the same green but **dotted** (as
  the `Computed` velocity variant) in the 2D view and both pinhole-scheme views.
- Plan `docs/plan_corrected_velocity.md` marked implemented; benchmark condition
  in `main_compare_algorithms.py` left as follow-up.
- Tests: `test_reconstruction_utils.py` (magnitude/depth kept, oriented 2D
  direction, degenerate fallbacks), `test_solve.py` (default/validation,
  Computed velocity filled, corrected changes the solution under noise, GT solve
  stays accurate), `test_viewer_algo_params.py` (velocity_source persisted).

### Entry-point refactor: `main_GUI.py` / `main_compare_algorithms.py`
- Split the project's entry points by purpose so each `main_*` launches one
  task:
  - `main.py` — kept as the solver/algorithm library (imported as `algo` by the
    GUI and the comparison) plus its solve-one-JSON CLI; docstring updated to
    point at the two launchers.
  - `main_GUI.py` (new) — launches the interactive matplotlib viewer
    (`python main_GUI.py [PATH]`), equivalent to the old `python GUI/viewer.py`;
    a thin launcher that delegates to `GUI/viewer.py`'s `main()`.
  - `main_compare_algorithms.py` — the Monte-Carlo comparison, renamed from
    `compare_algorithms.py` (unchanged behaviour: the per-setting mean/std error
    table + histograms).
- Test renamed to `unitests/test_main_compare_algorithms.py` and its import
  updated; `docs/plan_corrected_velocity.md` reference updated. Suite: 302
  passing.

### `compare_algorithms.py` — per-setting mean/std error table
- Refocused the benchmark on a single per-sample error
  `Err = mean_i ‖GT_i − Computed_i‖` over the four 3D keypoints, reported as
  **mean ± population std** (np.std, ddof=0) per algorithm.
- Sweep is now a full grid: `alpha ∈ {0.3, 0.5, 0.8}` × `2D ∈ {0,5,10,20} px` ×
  `velocity ∈ {0,2,5,10} deg` = **48 settings**, default **100 samples** each.
- Simplified `run()` to return `{(alpha, px, vel): {algo: [Err per sample]}}`;
  dropped the six-metric / body-wing-edge machinery (`metrics_3d`,
  `edge_errors`, `METRICS`, `_frame`) in favour of `sample_error(doc)`.
- Outputs reduced to two: `outputs/algorithm_comparison.csv` — **one row per
  setting** with columns `alpha, sigma_px, sigma_vdeg, n_samples,
  mean_err_{1,2,joint}_m, std_err_{1,2,joint}_m` (48 rows) — and
  `outputs/algorithm_comparison_hist.png`, now the `Err` distribution per
  algorithm (rows = alpha) at one representative noise cell (default 10 px /
  5 deg). The stale `algorithm_comparison_summary.csv` was removed.
- Findings: noise-free, sequential Err is a systematic ~1.95 m (α=0.3) / ~2.84 m
  (α=0.8) from the off-midpoint wings while Joint is ~0; Joint leads across most
  of the grid. At the extreme velocity noise (10°) the means carry large stds
  from rare catastrophic solves — the histogram medians are robust to those.
- Tests rewritten for the new return shape / helpers (`_mean_std` population
  std, error-table columns incl. `mean_err_joint_m`, 48-row count, histogram
  PNG). Suite: 302 passing.

### `compare_algorithms.py` — alpha sweep + per-edge histograms (superseded)
- Narrowed the analysis to the three requested algorithms (`#1`, `#2`,
  `Joint with wings`; dropped `Average`, which tracked `#1`/`#2`).
- Added a **wing-attachment (alpha) sweep** to the Monte Carlo (default
  α ∈ {0.5, 0.65, 0.8}): `random_aircraft(rng, cam, alpha)` shifts the wing
  pair along the body, so for α ≠ 0.5 the wing midpoint is off the body centre
  — the regime where the sequential wings solver's midpoint assumption breaks.
- Per-keypoint 3D Euclidean error is now collected split into **body edges**
  (tail, nose) and **wing edges** (tips).
- Three outputs (in `outputs/`): `algorithm_comparison.csv` (detailed per
  (alpha, noise, algorithm) medians of the six metrics, now with an `alpha`
  column), `algorithm_comparison_summary.csv` (short body/wing edge **mean +
  median**), and `algorithm_comparison_hist.png` — a single image with a grid
  of histograms (rows = alpha, columns = body/wing edges), the three algorithms
  overlaid with median lines.
- CLI: `python compare_algorithms.py [--trials N] [--seed S]
  [--hist-noise PX DEG] [--outdir DIR]`.
- Noise sweep spans 2D keypoint noise 0..10 px and velocity-direction noise
  0..5 deg (isolated-axis sweeps plus combined mid/max); histograms default to
  the mid level 5 px / 2.5 deg.
- Findings (150-trial run): at α = 0.5 the three are comparable, but as the
  wings move off-centre the sequential wing-edge error grows to a systematic
  ~2.8 m (α = 0.65) and ~5.6 m (α = 0.8) regardless of noise, while
  **Joint with wings** stays well under ~0.6 m — it estimates the attachment
  point instead of assuming the midpoint. Body-edge errors stay similar across
  all three.
- Tests updated for the new API (alpha argument, edge collection, summary CSV,
  histogram PNG, joint-wins-on-offset-wings). Suite: 301 passing.

### `compare_algorithms.py` (new) — Monte-Carlo accuracy comparison
- Benchmarks the four `head_tail_solution` algorithms (`#1`, `#2`, `Average`,
  `Joint with wings`) against ground truth. Each trial samples a random
  aircraft pose (uniform orientation, random in-frame image position, view
  angle kept inside 20-70° to avoid nose-on/broadside degeneracy), projects the
  four keypoints, derives the GT velocity and range, adds Gaussian noise to the
  2D keypoints and the velocity direction at a chosen level (range left exact),
  then solves the SAME noisy document with all four algorithms via
  `main.solve_document`. Running them on identical noisy data per trial makes
  the comparison paired.
- 3D error metrics vs GT, aggregated as the median over trials: mean keypoint
  position error, body-centre error, body-length error, wingspan error,
  geodesic rotation error, and worst wingtip error. Reported per noise level as
  a table (best algorithm per column marked `*`) with a best-per-column tally,
  and written to `outputs/algorithm_comparison.csv`.
- CLI: `python compare_algorithms.py [--trials N] [--seed S] [--csv PATH]`
  (default 300 trials over a 9-point 2D×velocity noise sweep).
- Findings (150-trial run): the three sequential variants are nearly identical
  (they differ only in the body head/tail pick and share the wings solver);
  **Joint with wings** clearly wins under velocity-direction noise and combined
  noise — notably better rotation, wingspan and wingtip error — while under
  pure 2D-keypoint noise the sequential solvers are marginally better on a few
  metrics. This matches the paper's Part II / Section 13.3 results.
- Tests: `unitests/test_compare_algorithms.py` (structure, noise-free accuracy,
  joint-beats-sequential-rotation under velocity noise, sequential variants
  close, CSV output). Suite: 299 passing.

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

## 2026-11-12 — Local aircraft coordinate frame

### `aircraft_frame.py` added
Body-fixed right-handed frame: **X** left-wing→right-wing, **Y** tail→nose,
**Z = X×Y** "up" (through the pilot's head), origin at the aircraft centre
(body midpoint = wing midpoint). The frame is always built from the **GT**
keypoints — GT is the internal reference, and all other variants (Noisy_GT,
Computed*) are expressed in these GT-anchored local coordinates for comparison.
- `AircraftFrame`: `origin` + `R` (columns = local axes in camera coords);
  built via `from_points(...)` / `from_doc(doc)` with Y exact along the body
  and the wing axis re-orthogonalized (Z = X_raw×Y, X = Y×Z) so R stays
  orthonormal under noise. Transforms: `to_local` / `to_world` (points, also
  vectorized over (N,3)) and `direction_to_local` / `direction_to_world`
  (free vectors, e.g. velocity).
- `local_keypoints(doc, variant)`: the 4 keypoints of any variant in the GT
  frame.
- Verified on `inputs/aircraft_input.json`: det(R)=+1, R orthonormal; GT lands
  exactly on the axes (tail (0,−15,0), nose (0,15,0), wings (∓20,0,0)); local Z
  points up in the image (camera −Y dominant); GT velocity maps to local
  (0,1,0); world↔local round-trip error 0.

### GUI: "Local axes" checkbox + "Solve GT"/"Solve Noisy" buttons
- **Local axes** checkbox (stacked with "Legends", default on): draws the
  GT-anchored local aircraft triad (`aircraft_frame.AircraftFrame`) in the
  pinhole views (b) and (c) — origin [0,0,0] at the aircraft centre (between
  tail and nose), X_loc left→right wing (magenta), Y_loc tail→nose (olive),
  Z_loc up (cyan). Drawn length 0.52 virtual units, slightly past the nose /
  wing tips so X/Y stick out beyond the body/wing arrows they are parallel to.
  Skipped silently if any GT 3D keypoint is null (the frame is GT-defined).
  Not drawn in the 2D view.
- The single "Solve" button split into **Solve GT** and **Solve Noisy**, both
  routed through one interface `AircraftViewer._solve(source)` →
  `main.solve_document(doc, source=...)`, so further algorithm/source choices
  plug in at one point. Status line reports `[solved: GT]` / `[solved:
  Noisy_GT]` (or the failure + source).
- Verified headlessly: triad renders in (b) and (c) and toggles off; Solve GT
  gives |err|≈0 m, Solve Noisy ≈3.6 m on P1 Computed1 vs GT; Load/Solve row
  refitted to three buttons.

### GUI: 3D preset "4" — aircraft view (local Z up, 45°)
- New key `4` in the 3D interactive view: aircraft-centric camera computed at
  key press from the GT local frame (`aircraft_frame`), so it adapts to the
  loaded data. The camera sits 45° off the tail on the left-wing side
  (`VIEW3D_AIRCRAFT_AZ=-45`, chosen so the nose points right and the image
  plane sits left, as in the reference composition) and 15° above the
  aircraft's horizontal plane (`VIEW3D_AIRCRAFT_ELEV`); both the aircraft and
  the image plane behind it stay in view (axis limits untouched).
- Screen-up is aligned with the aircraft's local **Z**. Local Z is generally
  not the Axes3D z axis, so the preset computes a **roll**: matplotlib's roll
  is *minus* the geometric angle from the roll-0 up vector to the desired up
  about the viewing axis (sign determined numerically). Example data gives
  elev=18.8, azim=-117.1, roll=4.1.
- Roll is now preserved by shift+arrow rotations and across scene rebuilds
  (the keep-camera tuple gained `ax.roll`); `0`/Home still resets roll to 0.
  Preset skipped silently if any GT 3D keypoint is null.
- Verified headlessly: local Z projects to screen (0,1) exactly, nose points
  right, plane+camera in frame; roll survives rotate + checkbox toggle.

### GUI: "Change GT" view (d) — move the ground-truth aircraft
- New dropdown view **Change GT**: the same 3D scene as view (c), but the
  camera is parked (entry camera = preset "4", aircraft view) and the keyboard
  moves the **GT aircraft** as a rigid body in its LOCAL frame:
    * `←→` translate X (wings), `↑↓` translate Z (up/down), `+/-` translate Y
      (forward/back) — `GT_MOVE_STEP` = 2 m per press;
    * `shift+←→` yaw (about local Z), `shift+↑↓` pitch (about local X),
      `ctrl+←→` roll (about local Y) — `GT_ROT_STEP` = 5° per press, all about
      the aircraft centre; sign conventions: shift+left = nose left, shift+up =
      nose up, ctrl+right = right wing down;
    * `ctrl+z` undo (stack of GT snapshots, depth `GT_UNDO_MAX` = 200; cleared
      on Load JSON); `0`/Home resets the camera to the entry view;
    * the mouse still drives the camera (rotate/pan/scroll-zoom), consistent
      with view (c).
- Every motion keeps the document consistent: GT 2D keypoints are reprojected
  (`project_pixel`), GT velocity rotates with the body, GT range is recomputed
  to the new centre. Rotation uses a Rodrigues helper `_rotation_about`.
- Verified headlessly: entry camera equals preset 4; 10 mixed moves keep body
  30 m / wings 40 m exactly (rigid); reprojected 2D, range and velocity match
  recomputation; 10×ctrl+z restores the original document byte-exactly and an
  extra undo is a no-op; the camera survives edits and undos; view (c) keys
  are unaffected.

### Change-GT view: frozen virtual-scene anchor
- Previously the virtual scheme re-anchored to the GT every render (aircraft
  pinned at 2 f_v along its ray, span-normalized), so GT edits appeared to
  drag the whole scene instead of moving the aircraft. `_scheme_geometry` is
  split: `_scheme_anchor()` computes the (C3, Qc, s_air) anchor; in the
  Change-GT view this anchor is **frozen at view entry** (`_edit_anchor`, reset
  on view change / Load JSON), so keyboard edits move only the aircraft and
  its re-projected GT points on the image plane — the pinhole, plane and
  framing stay put. Mouse rotation still turns the whole scene (camera).
- The scheme's returned centre (range-line endpoint) is now `map3d` of the
  current ref-variant centre: identical to Qc outside the edit view, follows
  the moved GT inside it.
- Verified headlessly: plane corners and the frozen mapping are bit-identical
  across edits; the virtual centre moves and returns exactly on full undo;
  views (b)/(c) outside edit re-anchor as before (centre back at z = 2 f_v).

### Change-GT view: controls legend box
- The Change-GT view now draws a boxed controls legend in the lower-left
  corner (`EDIT_CONTROLS_BOX`, monospace in a rounded light box): Move GT
  (←→ X wings, ↑↓ Z, +/- Y, 2 m/step), Rotate GT (shift+←→ yaw, shift+↑↓
  pitch, ctrl+←→ roll — about the centre, 5°/step, position kept), Undo
  (ctrl+z) and Camera (mouse drags/scroll, 0/Home reset). Only in this view;
  the compact top help line stays.

### GUI: Save JSON + Noise Up / Noise Down buttons
- Buttons re-laid-out into two rows: Load JSON · Save JSON · Solve GT, and
  Solve Noisy · Noise Up · Noise Down (status + View dropdown moved down).
- **Save JSON** always opens a save dialog pre-filled with the loaded path
  (`asksaveasfilename`, `.json` default extension), writes the full in-memory
  document via `aircraft_io.save_json`, updates `self.path`/status (`[saved]`).
- **Noise Up / Noise Down** re-sample the 2D `Noisy_GT` pixels as
  GT + N(0, σ) per axis (persistent unseeded `default_rng`; keypoints without
  a GT 2D reference are left untouched). σ starts at 1.5 px
  (`NOISE2D_DEFAULT`, = generate_input's SIGMA_2D), each press multiplies or
  divides by 1.5 (`NOISE2D_FACTOR`), capped at 100 px (`NOISE2D_MAX`); going
  below 0.5 px (`NOISE2D_MIN`) snaps to exactly 0 (Noisy_GT == GT), and Up
  from 0 restarts at 0.5. Status line shows `[noise2D σ=… px]`; grid redrawn
  (cells can newly enable) and the active view re-rendered.
- Verified headlessly: σ sequence 1.5→…→100 (cap) →…→0 (floor) →0.5; at σ=0
  Noisy_GT equals GT exactly; empirical std at σ=8 is 8.04 px over 200
  resamples; stubbed-dialog save writes a byte-exact copy of the document.

### GUI: "Noise target" radio (2D / Velocity / Range)
- New radio group under the button rows selects which Noisy_GT block the
  Noise Up/Down buttons re-sample; each target keeps its **own sigma**
  (`NOISE_PARAMS`, shared ×1.5 step `NOISE_FACTOR`, snap-to-0 floors):
    * **2D**: default 1.5 px, cap 100 px (as before);
    * **Velocity**: per-axis sigma on the direction then re-normalized to the
      GT magnitude — default 0.01, cap 0.5 (≈30–40° typical direction error);
    * **Range**: metres, clamped ≥1 m — default 10 m, cap 500 m (~25% of the
      ~2 km range).
  Defaults match generate_input's SIGMA_2D / SIGMA_V / SIGMA_RANGE. Status
  line shows `[noise <target> σ=…]` on both radio change and button press.
- Layout: radio at [0.015, 0.295] with a bold title; status moved to 0.283 and
  the View dropdown to 0.21 (its 4-option expanded list still fits above the
  figure bottom). First placement in the gap column overlapped the 2D axes'
  tick labels — moved into the left panel stack instead.
- Verified headlessly: per-target sigmas independent; caps reached exactly
  (100 px / 0.5 / 500 m) and floors give Noisy_GT == GT bit-exactly; each
  target modifies only its own JSON block; noisy velocity stays unit-length
  (30.0° from GT at the cap); expanded dropdown does not collide.

### Unit tests: `unitests/` suite (pytest)
- New `unitests/` folder with 160 tests covering every implemented module;
  `conftest.py` forces the Agg backend and puts the repo root + `GUI/` on
  `sys.path`, so the whole suite (GUI included) runs headlessly.
- `test_aircraft_io.py` (schema constants, None-safe builders, camera
  intrinsics/projection round-trips, JSON I/O), `test_aircraft_frame.py`
  (orthonormality, axis conventions, GT keypoints on local axes, transform
  round-trips, noisy-input re-orthogonalisation, `from_doc` errors),
  `test_reconstruction_utils.py` (both body solutions + wings recover the
  synthetic GT exactly; `compute_6dof` pose), `test_solve.py` (Computed-slot
  fill policy, GT→~0 error vs Noisy→nonzero, null-source `ValueError`,
  schema completeness), `test_rotation_about.py` (Rodrigues helper),
  `test_viewer_logic.py` (grid enable/check/toggle, priority variants),
  `test_viewer_noise.py` (sigma ladders, caps/floors, block isolation),
  `test_viewer_edit.py` (rigid-motion invariants, derived GT 2D/range/
  velocity, undo byte-exactness, frozen anchor), `test_viewer_save.py`
  (stubbed-dialog save, cancel no-ops).
- `pytest>=8.0` added to requirements. Run: `python -m pytest unitests/ -q`.

### GUI: "3D results" comparison view
- New View-dropdown entry **"3D results"** (view key `results`): GT vs
  Computed 3D keypoints drawn in REAL camera coordinates (metres) in the
  interactive Axes3D, auto-framed on the aircraft so every point is in view
  (no pinhole/plane — this view is about comparing the reconstructions).
  Data axes are the usual `_to3d` permutation, labelled Z/X/−Y (m).
- GT: filled markers, full-alpha red body / blue wing arrows, point labels.
  Computed: variant markers (diamond/square) at `RESULTS_PT_ALPHA` 0.75 and
  the **same segment colours at `RESULTS_SEG_ALPHA` 0.35**, so the two
  aircraft overlay but stay distinguishable. Dotted grey GT→computed
  connectors + a monospace box report per-point |error| in metres.
- Variant rule (`_results_computed_variant`): **Computed2 when any 3D
  Computed2 cell is checked (and has data), else Computed1**; wings only ever
  have Computed1 and fall back to it (marker/readout show the actual variant).
  Without any computed data the view shows a "press Solve" note.
- Camera: enters at the aircraft-centric preset-"4" view (also stored as this
  view's 0/Home); mouse + the same keys as the 3D interactive view work
  (`_on_key`/`_on_scroll` now accept `results`). New `_scene3d_kind` tag keeps
  the remembered camera **per 3D scene kind**, so the metre-scale results
  framing and the O(1)-unit virtual-scheme framing never leak into each other.
- 12 new tests in `unitests/test_viewer_results.py` (variant rule incl.
  stale-checkbox fallback, all points inside limits, per-kind camera
  isolation, camera kept across re-renders, zoom/pan keys); full suite now
  172 passing.

### GUI: results table in the "3D results" view (per the user's ODS example)
- The per-point |error| box is replaced by a summary table matching
  `example_table_results.ods`: rows **XYZ(m) / YPR(deg) / BodyLength /
  WingsSpan / CrossPoint**, columns **GT / Calc1 / Calc2** (Computed1/2;
  missing values shown as `-`). Dotted GT→computed connectors stay.
- `_aircraft_stats(variant)`: centre XYZ, body length, wingspan, yaw/pitch/
  roll in degrees via `ru.compute_6dof`, and **CrossPoint** — the parameter
  of the wings-segment crossing along the body, `t = (Wm−P1)·(P2−P1)/|P2−P1|²`
  with Wm the wings midpoint: 0 = tail, 1 = nose (0.5 = midpoint, > 0.5
  toward the nose), clamped to [0, 1]. For Calc2 the wings fall back to
  Computed1 (the only wings solution), as in the drawn overlay.
- The "no Computed data — press Solve" note moved right so it clears the
  table; the table always shows (GT column alone before solving).
- 10 new tests (GT stats exact 30 m / 40 m / 0.5; cross point 0.75 after
  shifting wings a quarter body toward the nose; clamping; Calc2 wings
  fallback; table tokens; GT-solve Calc values ≈ GT). Suite: 182 passing.

## 2026-11-13

### GUI: noise changes reflected in every view where the noisy element is visible
Rule: a Noise Up/Down press is reflected immediately wherever the re-sampled
Noisy_GT element is drawn — 2D noise in any view except "3D results";
Velocity/Range noise in any view except "2D" and "3D results". Computed*
data changes **only** on an explicit Solve (the short-lived auto re-solve
after a Noisy solve is reverted; "3D results" is intentionally untouched by
noise presses).
- `_scheme_geometry.map2d` now maps the **stored 2D pixels** linearly onto
  the virtual plane (3D ray ∩ plane only as a fallback when pixels are
  missing). Before, the on-plane 2D markers were derived from the 3D points,
  so 2D noise never moved them in the scheme / 3D interactive / Change GT
  views. GT consistency is kept: GT edits re-project GT pixels
  (`_refresh_gt_derived`), so GT plane markers still follow GT moves.
- Velocity: all three velocity drawings (2D view arrow, scheme, 3D
  interactive) drew only the highest-priority checked variant, hiding
  Noisy_GT whenever GT was also checked. New `_visible_velocity_variants()`
  draws **one arrow per checked variant** (non-GT at alpha 0.65, labelled
  with the variant short name).
- Range: the scheme range label showed only the first checked variant. New
  `_range_label_text()` lists **every** checked variant, e.g.
  `Range D: GT=2003m  N_GT=2011m`, so Range noise is visible immediately.
- Tests: the 4 auto-re-solve tests are replaced by 8 covering the new rule
  (noise never touches Computed*, plane marker tracks pixels + moves on 2D
  noise, all checked velocity variants drawn, multi-variant range label and
  its change on Range noise, clean re-render of every non-results view per
  noise target). Suite: 190 passing.

### Schema: remove Noisy_GT from 3D keypoints (obsolete variable)
The algorithm's noisy inputs are the 2D pixels + velocity + range only; a
Noisy_GT for the 3D keypoints never feeds anything, so it is removed from the
schema, the generator, the JSON files and the GUI.
- `aircraft_io.py`: new `VARIANTS_3D = [GT, Computed1..3]`; `variant_block()`
  takes an optional `variants=` (default: the full 5-variant set). Docstring
  states the exception.
- `generate_input.py`: `SIGMA_3D` and the 3D noise draw removed; 3D blocks
  are built with `variants=aio.VARIANTS_3D`. **Both JSON files regenerated**
  (`inputs/aircraft_input.json`, `outputs/aircraft_output.json`) — note the
  deterministic 2D/velocity/range noise values shifted because the rng no
  longer spends draws on 3D noise.
- GUI: `_cell_enabled` hard-disables the N_GT cell for 3D keypoint columns
  (even if an old file still carries the key); `point3d` and the
  `aircraft_frame` helpers use `.get(variant)` so the missing key reads as
  null.
- Tests: Noisy_GT-3D expectations replaced (`"Noisy_GT" not in` 3D blocks,
  `VARIANTS_3D` schema check, grid cell disabled for 3D points, toggle test
  moved to a 2D column) + 2 new `aircraft_io` tests. Suite: 192 passing.

## 2026-11-14

### GUI: 2D GT auto-corrected to match the 3D GT projection on load
The 2D GT pixels must always be the pinhole projection of the 3D GT points.
Loaded JSON files may violate this; the viewer now verifies and corrects it.
- `_check_gt_2d_consistency()`: for every keypoint with a 3D GT (Z > 0), the
  stored 2D GT is compared to the reprojection; values further off than
  `GT2D_TOL_PX` (1e-6 px) — or missing — are re-derived in place. Runs on
  construction and on every Load JSON.
- If anything was corrected, `_show_info()` pops a modal info box (OK button
  only) listing the corrected keypoints and noting that the in-memory
  document is now consistent (Save JSON to persist). Tk-based, same
  monkeypatchable pattern as the open/save dialogs; no-op when Tk is
  unavailable.
- GT edits were already consistent (`_refresh_gt_derived` re-projects after
  every Change-GT move); noise touches Noisy_GT only.
- 9 new tests in `unitests/test_viewer_gt2d.py` (consistent file untouched,
  stale/missing 2D GT fixed to the exact reprojection, no 3D GT -> left
  alone, other variants untouched, info box on correction and silence
  otherwise, corrupt file corrected via Load and via construction).
  Suite: 201 passing.

### Scheme views: camera->P_i rays now cross the plane exactly at p_i
The virtual scheme mapped 3D points affinely (`Qc + s_air*(P - C3)`); with
the exaggerated scheme FOV that mapping disagrees with true projection, so
the ray from [0,0,0] to the mapped P_i missed the pixel-mapped p_i marker on
the image plane (visible in the Pinhole/3D-interactive views).
- `map3d` is now projection-consistent: the affine map still sets the
  virtual **depth**, but the point is re-based onto the camera ray through
  the linear pixel map of its own true projection
  (`hit(project_pixel(P)) * z_aff/fv`). Rays therefore hit the plane exactly
  at p_i whenever the stored pixels match the 3D point's projection — which
  the on-load GT consistency check now guarantees for GT; Computed points
  align with their own reprojection automatically. Noisy_GT plane markers
  (pure measurements) remain intentionally off the GT rays. The aircraft
  centre still lands exactly on Qc, so framing/anchoring is unchanged.
- 5 new tests in `unitests/test_viewer_scheme_rays.py` (GT rays through GT
  plane points, centre == anchor, Computed rays through their reprojection,
  alignment preserved after Change-GT moves under the frozen anchor, noisy
  marker off-ray). Suite: 206 passing.

### GUI: vertical tail stabilizer (fin) as an orientation cue
A derived fin segment (cf. assets/figure1.png) makes the aircraft's
orientation readable at a glance: from P1_tail along the local +Z ("up")
direction, `FIN_FRACTION = 0.20` of the body length (6 m for the 30 m GT
body). Purely visual — nothing is added to the JSON schema.
- `_fin_from_points(P1, P2, P3, P4)` / `_fin_segment(variant)`:
  tip = P1 + 0.2*|P2-P1| * normalize((P4-P3) x (P2-P1)); Computed2 wings
  fall back to Computed1, matching the drawn overlay.
- Drawn wherever the aircraft is drawn: **2D view** (projected through the
  camera, same variant as the body arrow; skipped for 2D-only variants like
  Noisy_GT which have no 3D data), **Pinhole scheme** and **3D interactive /
  Change GT** (bold on the virtual aircraft + thin on the image plane via
  the ray \cap plane point), **3D results** (on both the GT and the Computed
  overlay, at the overlay's alpha). Body colour (red), plain line (no
  arrowhead).
- 9 new tests in `unitests/test_viewer_fin.py` (starts at tail, length
  0.2*body, along local Z, perpendicular to body+wings, rotates with
  Change-GT edits, None for Noisy_GT/empty variants, Computed2 wings
  fallback, render smoke across all views). Suite: 215 passing.

### GUI: "2D results" view (f) — reprojected Computed vs measured pixels
New dropdown option between "3D results" and "Change GT". An image-plane
comparison: **GT** (filled) and **Noisy_GT** (open) measured pixels, plus the
selected Computed 3D points **reprojected through the camera** (variant
markers at the results alpha, with body/wing arrows and the tail fin for
both the GT and the computed overlay). Computed1 by default, Computed2 when
its cell is checked; Computed2 wings fall back to Computed1.
- Pixel-error box (bottom-left): per-keypoint `|measured - reprojected|` in
  px + mean, measured against **the variant that fed the last solve** (GT or
  Noisy_GT) — tracked by the re-added `_last_solve_source` (set on a
  successful solve, cleared on Load). Dotted connectors join each reference
  pixel to its reprojection. Note the errors are ~0 right after a solve
  (solutions lie on the source rays) and grow when noise is pressed
  afterwards, since the reference pixels move.
- Without Computed data the view shows the measured points and a "press
  Solve" hint; without a solve this session there is no error reference and
  the box is omitted.
- 13 new tests in `unitests/test_viewer_results2d.py` (view key, renders
  with/without data, reprojection ≈ GT after a GT solve, Computed2 wings
  fallback, error text vs GT / vs N_GT, ~0 after solve, grows after
  post-solve noise, source tracking + reset on load). Suite: 228 passing.

### Change-GT view: slide wings along the body (CrossPoint editing)
New edit controls: **w** slides the GT wing pair toward the nose, **s**
toward the tail, `GT_CROSS_STEP = 0.05` CrossPoint units per press, clamped
to [0, 1] (0 = tail, 1 = nose). The plain "s" is freed from matplotlib's
save keymap (ctrl+s still saves the figure).
- `_shift_wings_gt(d_cross)`: a non-rigid GT edit — only P3/P4 translate
  along the body direction by `(new-cur)*|body|`; body points, velocity and
  range stay put; the wings' GT 2D projections are re-derived
  (`_refresh_gt_derived`), keeping the 2D GT consistency invariant. Each
  effective press lands on the undo stack (ctrl+z); a press at the clamp is
  a no-op and pushes nothing. Wing span/direction are preserved exactly.
- Help text and the boxed controls legend list the new keys.
- 11 new tests in `unitests/test_viewer_wings_slide.py` (0.5 start, ±step
  per key, clamping both ends, body/velocity/range untouched, span and
  direction preserved, 2D GT reprojected, undo depth semantics).
  Suite: 239 passing.

### Change-GT view: wingspan (a/d) and body-length (q/e) controls
Two more non-rigid GT edits, x1.05 per press (`GT_SCALE_STEP`), clamped so
neither size drops below `GT_SIZE_MIN` (1 m); plain "q" freed from
matplotlib's quit keymap (ctrl+w still quits).
- **a / d — wingspan narrower / wider**: `_scale_wings_gt(factor)` scales
  P3/P4 about the wings midpoint; midpoint, CrossPoint, body, velocity and
  range untouched.
- **q / e — body shorter / longer**: `_scale_body_gt(factor)` scales P1/P2
  about the body midpoint (centre and range untouched) and shifts the wing
  pair so its **CrossPoint station is preserved** — resizing the fuselage
  does not move where the wings sit on it; w/s stays the dedicated
  CrossPoint control.
- Both land on the undo stack, re-derive the GT 2D projections, and are
  listed in the help text + boxed controls legend.
- 13 new tests in `unitests/test_viewer_resize.py` (exact x1.05 scaling both
  ways, midpoint/cross/body invariants for span, centre/range/span
  invariants for body, cross-station preservation at 0.75, min clamps, 2D
  reprojection, velocity untouched, undo). Suite: 252 passing.

## 2026-11-15

### Single-solution refactor: algorithm parameters + Results in the JSON
- Variants reduced to `GT / Noisy_GT / Computed`: the solver now reports
  exactly ONE solution per run (`Computed1..3` removed from `aircraft_io`,
  `main`, the GUI, both sample JSONs and all fallback logic).
- New `Algorithm_parameters` block (persisted in the document, defaults in
  `main.DEFAULT_ALGO_PARAMS`): `input_noisy` (True -> solve from Noisy_GT,
  default on) and `head_tail_solution` (`#1` / `#2` / `Average` -- which of
  the two-fold-ambiguous body solutions is reported; Average = midpoint).
- `main.solve_document(doc, params=None)` replaces the `source=` API; merge
  order is defaults <- doc block <- call params. CLI: `--gt`, `--head-tail`.
- Output JSON now also carries a `Results` block written at solve time:
  `source`, `errors_2D_px` (the 2D-results-view pixel-error table incl.
  mean) and `table_3D` (GT/Computed rows of the 3D-results table: XYZ_m,
  YPR_deg, BodyLength_m, WingsSpan_m, CrossPoint). Shared helpers
  `aircraft_stats` / `reprojection_errors_px` / `results_block` live in
  `main.py` and back both the GUI tables and the saved JSON.
- GUI: "Solve Noisy" removed; a single "Solve" button runs with the current
  parameters. New "Set Algo Params" button pops a small Tk form (Input Type:
  Noisy checkbox; Head/Tail solution dropdown) and stores the result in the
  document, so Save JSON exports parameters + solution + results together.
  Results views compare GT vs the single Computed (table columns GT/Calc).
- Tests updated throughout; new `unitests/test_viewer_algo_params.py`
  (buttons, parameter state/merge, Solve honouring the form, Results block).
  Suite: 268 passing.

### Computed 2D keypoints filled at solve time
- `solve_document` now also writes `Coordinates_2D.*.Computed`: the single
  Computed 3D solution reprojected through the camera to pixels. The 2D
  drawing paths are variant-generic, so once filled the computed p1..p4 can
  be checked in the grid and appear in every view that draws 2D keypoints
  (2D, Pinhole scheme, 3D interactive, 2D results) with the Computed diamond
  marker; saved documents carry them alongside GT/Noisy_GT.
- Tests: solver-side reprojection-equality checks in `test_solve.py`,
  viewer-side enablement/rendering checks in `test_viewer_algo_params.py`.
  Suite: 272 passing.

### "Joint with wings" head/tail option (joint single-shot solver)
- Added `reconstruction_utils.joint_solution`: the paper's Part II joint
  least-squares fit (body length, wingspan, wing attachment point alpha and
  6-DOF pose with |t|=D) of all four keypoints at once, initialised from the
  sequential solution and refined with Levenberg-Marquardt (scipy). Unlike the
  sequential path it couples body and wings through one shared pose and
  estimates where the wings meet the body instead of assuming the midpoint.
- `head_tail_solution` gained a fourth value, "Joint with wings"; when chosen,
  `main.solve_document` fills all of P1..P4 from the joint fit (keypoint vs.
  velocity weighting from `JOINT_SIGMA_PX` / `JOINT_SIGMA_V_DEG`, sig_uv
  derived from the camera focal length). The GUI's Set Algo Params dropdown
  picks it up automatically (combobox widened for the longer label); CLI:
  `--head-tail "Joint with wings"`.
- Noise-free GT solve recovers all four points to machine precision, including
  non-midpoint wing attachment (alpha != 0.5), which the sequential wings
  solver cannot.
- Tests: new `unitests/test_joint_solution.py` (exactness across poses and
  alpha, structural properties, weighting/robustness) plus joint-option cases
  in `test_solve.py` and `test_viewer_algo_params.py`. Suite: 288 passing.

### Set Algo Params remembers the last choice across runs
- The algorithm parameters chosen in the Set Algo Params form are now
  persisted to a small prefs file (`$XDG_CONFIG_HOME/reconstruct3d/
  algo_params.json`, or `~/.config/...`; override with the
  `RECONSTRUCT3D_PREFS` env var) and restored on the next launch, so the form
  opens on the values you last used instead of the built-in defaults.
- Startup priority is: built-in defaults < the loaded document's own
  `Algorithm_parameters` block < the remembered preference. The remembered
  Set-Algo-Params choice wins so it genuinely sticks across runs (otherwise the
  output JSON loaded on startup, which always carries a block, would shadow
  it). When no preference has been set yet, a document's embedded params still
  apply. Corrupt/unreadable prefs fall back to the defaults.
- `AircraftViewer(path, prefs_path=...)` gained an optional prefs path (used by
  the tests for isolation); conftest points `RECONSTRUCT3D_PREFS` at a throwaway
  temp file so the suite never touches the real user config.
- Tests: persistence round-trip, form pre-fill, doc-block-overrides-pref, and
  corrupt-prefs fallback in `test_viewer_algo_params.py`. Suite: 292 passing.

### Fix: Set Algo Params dialog now opens on the last choice
- Root cause: the viewer runs on matplotlib's Tk root, but the params form
  created a *second* `tk.Tk()` each time. Two live Tk interpreters make the
  widgets ignore their initial values, so the dialog always opened on the
  defaults regardless of the stored/remembered parameters.
- The form now opens as a `Toplevel` over the existing root (falling back to
  its own `tk.Tk()` only when there is no root), sets the checkbox and dropdown
  values explicitly from `self.algo_params`, and is made modal with
  `wait_window()`/`grab_set()` instead of a nested `mainloop()`. It therefore
  pre-fills with the current parameters both within a session and across runs
  (combined with the remembered-preference load order).
- Not exercised by the headless suite (Tk needs a display); the 293 existing
  tests still pass.
