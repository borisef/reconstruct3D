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
