"""Joint single-shot solve for L1, L2, split point alpha and 6-DOF pose, vs. the sequential (paper) method."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
rng = np.random.default_rng(1)

W, H, FOV = 1920, 1080, np.deg2rad(1.0)
f = W/(2*np.tan(FOV/2)); cx, cy = W/2, H/2
ray = lambda uv: np.array([uv[0], uv[1], 1.0])

# ---------- model: local frame x=right, y=nose, z=up, origin = body centre ----------
def local_points(L1, L2, alpha):
    yw = (alpha-0.5)*L1                                   # wing attachment along body, alpha=0 tail, 1 nose
    return np.array([[0,-L1/2,0],[0,L1/2,0],[-L2/2,yw,0],[L2/2,yw,0]])   # tail, nose, left tip, right tip

def predict(theta, D):
    rvec, um, vm, l1, l2, alpha = theta[:3], theta[3], theta[4], theta[5], theta[6], theta[7]
    R = Rot.from_rotvec(rvec).as_matrix()
    c = ray((um, vm)); t = D*c/np.linalg.norm(c)          # range is a hard constraint: |t| = D
    P = local_points(l1*D, l2*D, alpha)@R.T + t           # sizes are solved relative to D
    return P[:,:2]/P[:,2:3], R, t

def residuals(theta, uv, Vhat, D, sig_uv, sig_V):
    uvp, R, _ = predict(theta, D)
    r_px = ((uvp-uv)/sig_uv).ravel()                      # 8 reprojection residuals
    r_v = np.cross(R[:,1], Vhat)/sig_V                    # velocity direction vs body axis (2 dof)
    return np.r_[r_px, r_v]

# ---------- sequential (paper) method, used as baseline AND as initial guess ----------
def sequential(uv, V, D):
    um = 0.5*(uv[0]+uv[1]); Pm = D*ray(um)/np.linalg.norm(ray(um))
    A = np.zeros((6,3)); A[:3,0]=ray(uv[0]); A[:3,2]=V; A[3:,1]=ray(uv[1]); A[3:,2]=-V   # sign of h fixed
    l1,l2,h = np.linalg.lstsq(A, np.r_[Pm,Pm], rcond=None)[0]
    P1, P2 = l1*ray(uv[0]), l2*ray(uv[1])
    d3, d4 = ray(uv[2]), ray(uv[3]); a = (d3@V)/(d4@V)
    l3 = 2*D/np.sqrt(d3@d3 + a*a*(d4@d4) + 2*a*(d3@d4))
    P3, P4 = l3*d3, l3*a*d4
    y = (P2-P1)/np.linalg.norm(P2-P1); x = (P4-P3)/np.linalg.norm(P4-P3); z = np.cross(x,y)
    U,_,Vt = np.linalg.svd(np.c_[x,y,z]); R = U@Vt        # nearest rotation
    return dict(P=np.array([P1,P2,P3,P4]), R=R, t=0.5*(P1+P2), L1=np.linalg.norm(P2-P1), L2=np.linalg.norm(P4-P3))

def joint(uv, V, D, sig_uv, sig_V):
    s = sequential(uv, V, D)
    b = uv[1]-uv[0]; wm = 0.5*(uv[2]+uv[3])
    alpha0 = np.clip((wm-uv[0])@b/(b@b), 0, 1)            # 2D ratio along the body segment
    um = 0.5*(uv[0]+uv[1])
    th0 = np.r_[Rot.from_matrix(s['R']).as_rotvec(), um, s['L1']/D, s['L2']/D, alpha0]
    sol = least_squares(residuals, th0, args=(uv, V, D, sig_uv, sig_V), method='lm', x_scale='jac')
    uvp, R, t = predict(sol.x, D)
    J = sol.jac; cov = np.linalg.inv(J.T@J)
    return dict(R=R, t=t, L1=sol.x[5]*D, L2=sol.x[6]*D, alpha=sol.x[7], chi2=2*sol.cost, cov=cov,
                P=local_points(sol.x[5]*D, sol.x[6]*D, sol.x[7])@R.T + t, nfev=sol.nfev)

# ---------- synthetic data ----------
def make(alpha, D=3000., L1=15., L2=10., los=(20,70)):
    while True:
        R = Rot.random(random_state=rng).as_matrix()
        cpx = np.array([rng.uniform(300,W-300), rng.uniform(200,H-200)])
        c = ray(((cpx[0]-cx)/f, (cpx[1]-cy)/f)); t = D*c/np.linalg.norm(c)
        ang = np.degrees(np.arccos(abs(R[:,1]@t)/D))
        if los[0] < ang < los[1]: break
    return dict(R=R, t=t, P=local_points(L1,L2,alpha)@R.T+t, L1=L1, L2=L2, alpha=alpha, D=D)

def observe(ac, px, vdeg):
    P = ac['P']; uv = P[:,:2]/P[:,2:3] + rng.normal(scale=px/f, size=(4,2)) if px else P[:,:2]/P[:,2:3]
    V = ac['R'][:,1].copy()
    if vdeg:
        V = V + rng.normal(scale=np.deg2rad(vdeg), size=3); V /= np.linalg.norm(V)   # isotropic direction noise
    return uv, V

rot_err = lambda Ra, Rb: np.degrees(np.linalg.norm(Rot.from_matrix(Ra.T@Rb).as_rotvec()))

# ===== 0: exactness and scale gauge =====
ac = make(0.35); uv, V = observe(ac, 0, 0)
j = joint(uv, V, ac['D'], 0.5/f, np.deg2rad(1))
print("=== noise-free, alpha=0.35 ===")
print(f"  joint: L1={j['L1']:.6f} L2={j['L2']:.6f} alpha={j['alpha']:.6f} rot err={rot_err(j['R'],ac['R']):.2e} deg, |t err|={np.linalg.norm(j['t']-ac['t']):.2e} m")
uv, V = observe(ac, 0.5, 1.0)
ja = joint(uv, V, ac['D'], 0.5/f, np.deg2rad(1)); jb = joint(uv, V, 1.03*ac['D'], 0.5/f, np.deg2rad(1))
print("=== range gauge: same noisy data solved with D and with 1.03*D ===")
print(f"  L1 ratio={jb['L1']/ja['L1']:.6f}  L2 ratio={jb['L2']/ja['L2']:.6f}  |t| ratio={np.linalg.norm(jb['t'])/np.linalg.norm(ja['t']):.6f}")
print(f"  alpha diff={jb['alpha']-ja['alpha']:.1e}  rotation diff={rot_err(ja['R'],jb['R']):.1e} deg  chi2 diff={jb['chi2']-ja['chi2']:.1e}\n")

# ===== 1: Monte Carlo, sequential vs joint =====
def mc(alpha, px, vdeg, n=1500, los=(20,70)):
    out = {k: [] for k in ['sL1','sL2','sR','sT','sW','jL1','jL2','jR','jT','jW','ja','chi2','za']}
    for _ in range(n):
        ac = make(alpha, los=los); uv, V = observe(ac, px, vdeg)
        s = sequential(uv, V, ac['D']); j = joint(uv, V, ac['D'], px/f, np.deg2rad(vdeg))
        for p, r in (('s', s), ('j', j)):
            out[p+'L1'].append(r['L1']-ac['L1']); out[p+'L2'].append(r['L2']-ac['L2'])
            out[p+'R'].append(rot_err(r['R'], ac['R'])); out[p+'T'].append(np.linalg.norm(r['t']-ac['t']))
            out[p+'W'].append(np.linalg.norm(r['P'][2:]-ac['P'][2:], axis=1).max())
        out['ja'].append(j['alpha']-alpha); out['chi2'].append(j['chi2']); out['za'].append((j['alpha']-alpha)/np.sqrt(j['cov'][7,7]))
    return {k: np.array(v) for k, v in out.items()}

rms = lambda v: np.sqrt(np.mean(v**2))
print("=== Monte Carlo: 0.5 px keypoint noise + 1 deg velocity-direction noise, view angle 20-70 deg, 1500 trials ===")
print("  alpha | method     | L1 rms [m] | L2 rms [m] | rot err med [deg] | centre err med [m] | wingtip err med [m] | alpha rms")
for alpha in [0.5, 0.4, 0.3]:
    o = mc(alpha, 0.5, 1.0)
    print(f"  {alpha:4.2f}  | sequential | {rms(o['sL1']):9.3f}  | {rms(o['sL2']):9.3f}  | {np.median(o['sR']):12.3f}      | {np.median(o['sT']):12.3f}       | {np.median(o['sW']):12.3f}        |    -")
    print(f"        | joint      | {rms(o['jL1']):9.3f}  | {rms(o['jL2']):9.3f}  | {np.median(o['jR']):12.3f}      | {np.median(o['jT']):12.3f}       | {np.median(o['jW']):12.3f}        | {rms(o['ja']):.4f}")
    if alpha == 0.4: keep = o
print(f"\n  consistency (alpha=0.4): mean chi2 = {keep['chi2'].mean():.2f} (expected 2 = 10 residual dof - 8 unknowns); "
      f"alpha error / predicted sigma: std = {keep['za'].std():.2f} (expected 1)\n")

print("=== which noise source drives which error (alpha=0.4, joint) ===")
for px, vd in [(0.5, 0.25), (0.5, 1.0), (0.5, 3.0), (2.0, 1.0)]:
    o = mc(0.4, px, vd, n=800)
    print(f"  {px:.1f} px, V {vd:4.2f} deg -> L1 rms {rms(o['jL1']):.3f} (seq {rms(o['sL1']):.3f}), L2 rms {rms(o['jL2']):.3f} (seq {rms(o['sL2']):.3f}), "
          f"rot med {np.median(o['jR']):.2f} deg (seq {np.median(o['sR']):.2f}), alpha rms {rms(o['ja']):.4f}")

print("\n=== degenerate views (alpha=0.4, 0.5 px, 1 deg), joint ===")
for los in [(2,5), (85,88), (89,89.8)]:
    o = mc(0.4, 0.5, 1.0, n=600, los=los)
    print(f"  view angle {los[0]}-{los[1]} deg: L1 rms {rms(o['jL1']):.2f} m, L2 median abs err {np.median(np.abs(o['jL2'])):.2f} m, alpha rms {rms(o['ja']):.4f}")