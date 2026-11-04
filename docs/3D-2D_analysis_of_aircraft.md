# 3D–2D Analysis of Aircraft

Recovering the 3D position, size, and 6-DOF pose of an aircraft from a **single**
2D image, under a set of simplifying assumptions. The aircraft is modeled as a
cross ("+") with four keypoints.

> This document is a cleaned-up and corrected rewrite of the original note.
> See [Corrections vs. original](#corrections-vs-original) at the end for the
> changes. The implementation lives in `reconstruction_utils.py`, with runnable
> examples in `main.py`.

---

## Table of contents

- [Problem statement](#problem-statement)
  - [Assumptions](#assumptions)
  - [Given](#given)
  - [What we want to compute](#what-we-want-to-compute)
- [Body of aircraft](#body-of-aircraft)
  - [Solution #1 — direction first, then range](#solution-1--direction-first-then-range)
  - [Solution #2 — range first, then direction](#solution-2--range-first-then-direction)
- [Wings of aircraft](#wings-of-aircraft)
- [6 DOF of aircraft](#6-dof-of-aircraft)
- [Appendix](#appendix)
- [Corrections vs. original](#corrections-vs-original)

---

## Problem statement

### Assumptions

1. The internal parameters of the camera are known ($f$, FOV, number of pixels).
2. Pinhole perspective camera with a very small FOV, symmetric, calibrated,
   without distortions.
3. The target is at a very large distance ($> 1\,\text{km}$) relative to its size,
   and the FOV is very small.
4. The aircraft is symmetric and represented as a cross: two points on the body
   and two points at the wingtips.
5. The distance to the center of the body is known, $D$.
6. The distance to the center of the wings is the same $D$ (not strictly true,
   but we assume it).
7. The velocity vector represents the body direction and is known and noise-free.
8. The coordinates of the 4 keypoints are known and noise-free.

> In short, we assume a near-perfect world, but in some cases we still prefer a
> **least-squares (LSE)** solution to tolerate noise — an exact solution would
> often leave us with no solution at all.

### Given

- Camera: $f$, $H_{\text{pixel}}$, $W_{\text{pixel}}$. See the
  [Appendix](#appendix) for how to compute $f$.
- Pixel coordinates of the body and wing keypoints:
  $(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)$.
  We will not use the pixel coordinates directly — we convert them to
  **normalized** coordinates (see [Appendix](#appendix)):

  $$(u_1, v_1),\ (u_2, v_2) \quad\text{— normalized body keypoints (tail, nose)}$$
  $$(u_3, v_3),\ (u_4, v_4) \quad\text{— normalized wing keypoints (wingtips)}$$

  **From now on, all 2D image-plane coordinates in this document are normalized.**

- $D$ — distance from the camera focal point to the aircraft (center of body and,
  by assumption 6, center of wings).
- $\vec{V} = (V_x, V_y, V_z)$ — velocity vector of the aircraft (unit direction, or
  velocity in m/s). This is also the body direction (tail → nose).

### What we want to compute

The 3D locations of all four keypoints:

- Body endpoints: $\vec{P}_1 = (X_1, Y_1, Z_1)$ (tail), $\vec{P}_2 = (X_2, Y_2, Z_2)$ (nose).
- Wing endpoints: $\vec{P}_3 = (X_3, Y_3, Z_3)$, $\vec{P}_4 = (X_4, Y_4, Z_4)$ (wingtips).

From these we easily get the 6-DOF pose and the physical sizes of the aircraft.

**Core idea.** The camera sits at the origin. Each normalized keypoint $(u, v)$
defines a ray from the origin:

$$\vec{P}(\lambda) = \lambda\,(u, v, 1)^{\mathsf T}.$$

Reconstruction means finding the depth scalar $\lambda$ for each ray, using the
extra constraints supplied by $\vec{V}$ and $D$.

![Schematic: 3D aircraft keypoints and their projection onto the virtual image
plane. The camera is at the origin; the real image plane is on the opposite side
of $(0,0,0)$.](../assets/figure1.png)

---

## Body of aircraft

We present two solutions. The first uses the body direction $\vec{V}$ first and
the range $D$ second; the second uses the range first and the direction second.

Notation: $\vec{P}_i = \lambda_i (u_i, v_i, 1)^{\mathsf T}$, and
$\vec{P}_m$ is the body midpoint.

### Solution #1 — direction first, then range

The vector $\vec{M} = \vec{P}_2 - \vec{P}_1$ is parallel to
$\vec{V} = (V_x, V_y, V_z)$, so $\vec{M} = s\,\vec{V}$ for some scalar $s$.
Component-wise:

$$
\lambda_2 u_2 - \lambda_1 u_1 = s V_x, \qquad
\lambda_2 v_2 - \lambda_1 v_1 = s V_y, \qquad
\lambda_2 - \lambda_1 = s V_z .
$$

Let $\tilde\lambda_1 = \lambda_1 / s$ and $\tilde\lambda_2 = \lambda_2 / s$. Then

$$
\begin{bmatrix} u_2 & -u_1 \\ v_2 & -v_1 \\ 1 & -1 \end{bmatrix}
\begin{bmatrix} \tilde\lambda_2 \\ \tilde\lambda_1 \end{bmatrix}
=
\begin{bmatrix} V_x \\ V_y \\ V_z \end{bmatrix}.
$$

This is overdetermined (3 equations, 2 unknowns). Solve by least squares:

$$
A = \begin{bmatrix} u_2 & -u_1 \\ v_2 & -v_1 \\ 1 & -1 \end{bmatrix},
\qquad
b = \begin{bmatrix} V_x \\ V_y \\ V_z \end{bmatrix},
\qquad
\begin{bmatrix} \tilde\lambda_2 \\ \tilde\lambda_1 \end{bmatrix}
= (A^{\mathsf T} A)^{-1} A^{\mathsf T} b .
$$

**Recover the scale $s$ from the range $D$.** The (unscaled) midpoint direction is

$$
\vec{a} = \tilde\lambda_1 (u_1, v_1, 1)^{\mathsf T} + \tilde\lambda_2 (u_2, v_2, 1)^{\mathsf T},
$$

and the true midpoint is $\vec{P}_m = \tfrac{1}{2} s\,\vec{a}$. Imposing
$\lVert \vec{P}_m \rVert = D$:

$$
\tfrac{1}{2} s \lVert \vec{a} \rVert = D
\quad\Longrightarrow\quad
s = \frac{2D}{\lVert \vec{a} \rVert}.
$$

Finally $\lambda_1 = s\,\tilde\lambda_1$, $\lambda_2 = s\,\tilde\lambda_2$, and

$$
\vec{P}_1 = \lambda_1 (u_1, v_1, 1)^{\mathsf T}, \qquad
\vec{P}_2 = \lambda_2 (u_2, v_2, 1)^{\mathsf T}.
$$

### Solution #2 — range first, then direction

Start from the body midpoint in 2D:

$$
u_m = \tfrac{1}{2}(u_1 + u_2), \qquad v_m = \tfrac{1}{2}(v_1 + v_2).
$$

> Remark: the mid-body point from the keypoint detector may be used here instead.

Push this midpoint out along its ray to the depth fixed by the range $D$:

$$
\vec{P}_m = \lambda_m (u_m, v_m, 1)^{\mathsf T}, \qquad
\lVert \vec{P}_m \rVert = D
\ \Longrightarrow\
\lambda_m = \frac{D}{\sqrt{u_m^2 + v_m^2 + 1}} .
$$

Substituting gives $\vec{P}_m = (X_m, Y_m, Z_m)$.

Now build a symmetric 3D segment through $\vec{P}_m$, directed along $\vec{V}$,
extending by an unknown half-length $h$ on each side until it meets rays 1 and 2:

$$
\vec{P}_1 = \vec{P}_m + \vec{V} h, \qquad
\vec{P}_2 = \vec{P}_m - \vec{V} h .
$$

Using $\vec{P}_i = \lambda_i (u_i, v_i, 1)^{\mathsf T}$ yields 6 equations in the 3
unknowns $(\lambda_1, \lambda_2, h)$:

$$
\begin{aligned}
\lambda_1 u_1 - V_x h &= X_m & \lambda_2 u_2 + V_x h &= X_m \\
\lambda_1 v_1 - V_y h &= Y_m & \lambda_2 v_2 + V_y h &= Y_m \\
\lambda_1 - V_z h &= Z_m & \lambda_2 + V_z h &= Z_m
\end{aligned}
$$

In matrix form:

$$
A = \begin{bmatrix}
u_1 & 0 & -V_x \\
v_1 & 0 & -V_y \\
1 & 0 & -V_z \\
0 & u_2 & V_x \\
0 & v_2 & V_y \\
0 & 1 & V_z
\end{bmatrix},
\qquad
b = \begin{bmatrix} X_m \\ Y_m \\ Z_m \\ X_m \\ Y_m \\ Z_m \end{bmatrix},
\qquad
\begin{bmatrix} \lambda_1 \\ \lambda_2 \\ h \end{bmatrix}
= (A^{\mathsf T} A)^{-1} A^{\mathsf T} b .
$$

Then $\vec{P}_1 = \lambda_1 (u_1, v_1, 1)^{\mathsf T}$ and
$\vec{P}_2 = \lambda_2 (u_2, v_2, 1)^{\mathsf T}$.

---

## Wings of aircraft

We assume the body direction $\vec{V}$ is known and the range to the wing
midpoint is $D$. We look for the wingtips on their corresponding rays:

$$
\vec{P}_3 = \lambda_3 \vec{d}_3 = \lambda_3 (u_3, v_3, 1)^{\mathsf T}, \qquad
\vec{P}_4 = \lambda_4 \vec{d}_4 = \lambda_4 (u_4, v_4, 1)^{\mathsf T}.
$$

![Schematic: this time we seek the segment **perpendicular** to
$\vec{V}$.](../assets/figure3.png)

**Distance constraint** — the wing midpoint is at range $D$:

$$
\left\lVert \tfrac{1}{2}\big(\lambda_3 \vec{d}_3 + \lambda_4 \vec{d}_4\big) \right\rVert^2 = D^2
$$

$$
\lambda_3^2 (\vec{d}_3 \cdot \vec{d}_3)
+ \lambda_4^2 (\vec{d}_4 \cdot \vec{d}_4)
+ 2\lambda_3 \lambda_4 (\vec{d}_3 \cdot \vec{d}_4) = 4 D^2,
\tag{$*$}
$$

where

$$
\vec{d}_3 \cdot \vec{d}_3 = u_3^2 + v_3^2 + 1, \quad
\vec{d}_4 \cdot \vec{d}_4 = u_4^2 + v_4^2 + 1, \quad
\vec{d}_3 \cdot \vec{d}_4 = u_3 u_4 + v_3 v_4 + 1 .
$$

**Perpendicularity constraint** — the wing span is perpendicular to the body:

$$
(\vec{P}_3 - \vec{P}_4) \perp \vec{V}
\ \Longrightarrow\
\lambda_3\, \vec{d}_3 \cdot \vec{V} - \lambda_4\, \vec{d}_4 \cdot \vec{V} = 0
\ \Longrightarrow\
\lambda_4 = \lambda_3 \,\underbrace{\frac{\vec{d}_3 \cdot \vec{V}}{\vec{d}_4 \cdot \vec{V}}}_{a}
= \lambda_3\, a .
$$

Substituting $\lambda_4 = \lambda_3 a$ into $(*)$ gives one equation in one unknown:

$$
\lambda_3^2 \left[ (\vec{d}_3 \cdot \vec{d}_3) + a^2 (\vec{d}_4 \cdot \vec{d}_4)
+ 2a (\vec{d}_3 \cdot \vec{d}_4) \right] = 4 D^2,
$$

$$
\boxed{\ \lambda_3 = \frac{2D}{\sqrt{(\vec{d}_3 \cdot \vec{d}_3) + a^2 (\vec{d}_4 \cdot \vec{d}_4) + 2a (\vec{d}_3 \cdot \vec{d}_4)}}\ },
\qquad \lambda_4 = \lambda_3\, a .
$$

> Remark: the same procedure computes any symmetric left/right pair of keypoints.

**What's next?** With the 3D coordinates of all four keypoints we can:
(1) compute the 6-DOF pose, (2) compute body length and wingspan, (3) predict and
filter over time.

---

## 6 DOF of aircraft

Define the aircraft's local coordinate system:

- The body center $(0,0,0)_{\text{local}}$ is the midpoint of the body.
- $Y$ points from tail to nose.
- $X$ points from the left wing to the right wing.
- $Z$ points up (direction of the pilot's body).

### Translation

The body center is the midpoint, which is the translation of the local frame into
the world frame:

$$
\vec{t} = \vec{P}_m = \tfrac{1}{2}(\vec{P}_1 + \vec{P}_2) .
$$

### Rotation (axes in world coordinates)

$$
\vec{y} = \frac{\vec{P}_2 - \vec{P}_1}{\lVert \vec{P}_2 - \vec{P}_1 \rVert}
\quad\text{(tail → nose)},
$$

$$
\vec{x} = \frac{\vec{P}_4 - \vec{P}_3}{\lVert \vec{P}_4 - \vec{P}_3 \rVert}
\quad\text{(left → right wing)},
$$

$$
\vec{z} = \frac{\vec{x} \times \vec{y}}{\lVert \vec{x} \times \vec{y} \rVert}
\quad\text{(up)} .
$$

The rotation matrix has these unit axes as its columns:

$$
R = \begin{bmatrix} | & | & | \\ \vec{x} & \vec{y} & \vec{z} \\ | & | & | \end{bmatrix}
= \begin{bmatrix} r_{00} & r_{01} & r_{02} \\ r_{10} & r_{11} & r_{12} \\ r_{20} & r_{21} & r_{22} \end{bmatrix}.
$$

It can be converted to a Rodrigues (axis–angle) vector:

$$
\vec{r} = \mathrm{Rodrigues}(R) .
$$

Euler angles (0-based indices $r_{\text{row},\text{col}}$):

$$
\text{pitch} = \arcsin(-r_{20}), \qquad
\text{yaw} = \operatorname{atan2}(r_{10}, r_{00}), \qquad
\text{roll} = \operatorname{atan2}(r_{21}, r_{22}) .
$$

---

## Appendix

### Normalized coordinates

$$
(u, v) = \left( \frac{x_{\text{pixel}} - c_x}{f},\ \frac{y_{\text{pixel}} - c_y}{f} \right),
\qquad c_x = \frac{W_{\text{pixel}}}{2},\quad c_y = \frac{H_{\text{pixel}}}{2}.
$$

### Focal length from FOV

For a camera with field of view FOV, image width $W$ and pixel size $S$:

$$
f = \frac{W S}{2 \tan(\text{FOV}/2)} .
$$

Here $f$ is expressed in pixels, so $S = 1$.

### Perspective (pinhole) camera

Each pixel $(u, v)$ of a primitive perspective (pinhole) camera is represented by
a 3D ray in space. The ray starts at the optical center of the camera and
represents all points in space that project to that pixel. The optical center is
$(0,0,0)$, and the center of the virtual image plane is at $(0, 0, f)$.

$$
\vec{P}(\lambda) = \lambda\,(u, v, 1)^{\mathsf T}, \qquad \lambda > 0 .
$$

![Pinhole camera: a world point $P = (X_w, Y_w, Z_w)$ projects through the
optical center to image coordinates $(u, v)$ on the plane $z = f$. The principal
point is $(c_x, c_y)$.](../assets/figure_pinhole.png)

---

## Corrections vs. original

The math is reproduced faithfully from the original note; the following
clarity/correctness edits were applied:

1. **6-DOF $Y$ axis.** The original wrote $\vec{y} = (\vec{P}_1 - \vec{P}_0)/\lVert\cdots\rVert$
   with an undefined $\vec{P}_0$ and an index inconsistent with the stated
   semantics. It is corrected to $\vec{y} = (\vec{P}_2 - \vec{P}_1)/\lVert\vec{P}_2 - \vec{P}_1\rVert$
   (tail → nose), which also makes $\vec{y}$ align with the velocity vector $\vec{V}$.
2. **Pinhole figure labels.** The original figure carried a TODO noting that its
   $x,y$ axes should read $u,v$. The text here uses $u,v$ consistently.
3. **Matrix right-hand sides** in Solution #2 are written uniformly as
   $(X_m, Y_m, Z_m, X_m, Y_m, Z_m)$ (the original had a stray $Z_M$ casing).
4. Minor notation unified: all image coordinates are normalized $(u, v)$, all 3D
   points are $\vec{P}_i = \lambda_i (u_i, v_i, 1)^{\mathsf T}$.
