"""
model.py — Stokeslet kernels, ciliary beat waveform and time evolution
=======================================================================

Planar model of a row of flagella growing from a flat no-slip wall (z = 0).

Units used throughout:  length [µm], time [s], force [pN], viscosity [mPa·s]
(internally the viscosity is converted to pN·s/µm² : 1 mPa·s = 1e-3 pN·s/µm²).

Hydrodynamics
-------------
Each flagellum is discretised into N_seg regularized Stokeslets (Cortez 2001,
blob φ_ε = 15ε⁴ / 8π(r²+ε²)^{7/2}).  The no-slip wall is enforced with the
Blake (1971) image system (image Stokeslet + Stokes doublet + source doublet).
The image terms are regularized by replacing |R| → sqrt(|R|²+ε²), i.e. a
simplified version of the regularized image system of Ainley et al. (2008).
The wall velocity is exactly zero for ε → 0 and has an O(ε²/h²) residual
otherwise (`check_kernel()` prints it).

Beat waveform  (tangent angle ψ(s,φ) measured from the wall normal, +x positive)
--------------------------------------------------------------------------------
A simplified Brennen & Winet / Sleigh-type ciliary stroke:
  * effective stroke  (0 ≤ u < p_eff, u = φ/2π mod 1):
        the (almost) straight filament rotates as a rod from -θ0 to +θ0
  * recovery stroke   (p_eff ≤ u < 1):
        a bend is created at the base and propagates to the tip;
        proximal part already points backward (-θ0), distal part still forward (+θ0)
        → the filament is dragged back close to the wall, mostly tangentially.
The shape is continuous over the whole cycle.

Phase dynamics
--------------
beat_mode = "fixed" : dφ_i/dt = ω_fixed  (kinematics prescribed)
beat_mode = "load"  : dφ_i/dt = ω0 (1 - T_i / T_stall)
    T_i = Σ_k F_k · ∂X_k/∂φ_i  (generalized resistive torque conjugate to φ_i,
    pN·µm/rad; power dissipated by flagellum i = T_i φ̇_i).  F includes the
    flow produced by all other flagella, so the flagella interact.

wall_mode = "anchored" : wall at rest, thrust = Σ F_x of the flagella on the fluid
wall_mode = "swimming" : wall + bases translate with U (x).  Flagella velocities
    relative to the ambient fluid are  ∂X/∂φ φ̇ + U e_x ; U is solved every step
    from   ζ_body η U + Σ_k F_x,k = 0   (ζ_body = 6πR by default).
"""
from __future__ import annotations

import math
import sys
if sys.platform == "win32":          # Windows consoles/redirects default to cp932: print µ, ², … safely
    for _st in (sys.stdout, sys.stderr):
        try:
            _st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from dataclasses import dataclass, field, asdict, replace
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

ETA_UNIT = 1e-3          # 1 mPa·s = 1e-3 pN·s/µm²
I3 = np.eye(3)
TWO_PI = 2.0 * np.pi

# Reference d/L of the two species (used for vertical lines in plots)
VOLVOX = dict(d=9.0, L=22.0, R_body=200.0)
PLEODORINA = dict(d=20.0, L=19.0, R_body=50.0)


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class Params:
    # geometry
    N: int = 10                 # number of flagella
    d: float = 9.0              # spacing [µm]
    L: float = 22.0             # flagellum length [µm]
    a: float = 0.125            # flagellum radius [µm]
    N_seg: int = 20             # Stokeslets per flagellum
    # fluid
    eta: float = 0.84           # viscosity [mPa·s]
    eta_water: float = 0.84     # viscosity used to calibrate T_stall [mPa·s]
    # waveform
    theta0: float = 1.15        # stroke amplitude (angle from wall normal) [rad]
    p_eff: float = 0.35         # fraction of the cycle spent in the effective stroke
    bend_width: float = 0.25    # half-width of the propagating bend / L
    waveform: str = "rod"       # "rod" (rigid effective stroke + propagating bend) | "delay"
    Theta_b: float = 1.8        # "delay": total sweep of the basal tangent angle [rad]
    lag: float = 0.15           # "delay": phase delay base → tip (fraction of a cycle)
    center: float = 0.0         # "delay": mean tangent angle (tilt of the beat) [rad]
    taper: float = 1.0          # "delay": amplitude at the tip / amplitude at the base
    # beat dynamics
    beat_mode: str = "fixed"    # "fixed" | "load"
    f_fixed: float = 30.0       # beat frequency in "fixed" mode [Hz]
    f_free: float = 50.0        # ω0/2π : unloaded motor frequency in "load" mode [Hz]
    f_water: float = 30.0       # target single-flagellum frequency in water ("load")
    T_stall: float | None = None  # stall torque [pN·µm/rad]; None → calibrated from f_water
    # wall
    wall_mode: str = "anchored"  # "anchored" | "swimming"
    R_body: float = 200.0       # colony radius for ζ_body = 6πR [µm]
    zeta_body: float | None = None  # body drag coefficient [µm]; None → 6πR
    # initial phases
    init_phase: str = "random"  # "random" | "inphase" | "metachronal"
    dphi_mc: float = -np.pi / 4  # neighbour phase difference for "metachronal" [rad]
    seed: int = 0
    # numerics
    eps: float | None = None    # Stokeslet blob size [µm]; None → calibrated (see calibrate_eps)
    eps_inter: float | None = 1.0  # blob size between *different* flagella [µm] (None → eps).
                                   # The row is strictly coplanar, so out-of-phase neighbours pass
                                   # through each other; a larger blob for inter-flagellar terms keeps
                                   # the coupling finite (≈ filaments ~eps_inter apart out of plane).
    y_stagger: float = 0.0      # alternate bases at y = ±y_stagger/2 [µm] (other way to avoid crossing)
    dt: float | None = None     # time step [s]; None → period / steps_per_cycle
    steps_per_cycle: int = 60
    n_cycles: float = 20.0      # simulated time in (estimated) beat periods
    transient: float = 0.3      # fraction of the run discarded before averaging
    integrator: str = "rk2"     # "rk2" (Heun) | "euler"   (load mode)
    # output
    frame_cycles: float = 2.0   # number of final cycles stored for flow field / animation
    frames_per_cycle: int = 30

    def copy(self, **kw) -> "Params":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def eta_int(self) -> float:
        return self.eta * ETA_UNIT

    @property
    def zeta(self) -> float:
        return self.zeta_body if self.zeta_body is not None else 6.0 * np.pi * self.R_body


# =============================================================================
# Regularized Stokeslet + Blake image kernel
# =============================================================================
def kernel(x: np.ndarray, y: np.ndarray, eps, wall: bool = True) -> np.ndarray:
    """Green's tensor G[p,k,i,j]: velocity u_i(x_p) = G_ij f_j(y_k) / (8πη).

    x : (P,3) field points,  y : (K,3) source points,
    eps : scalar or (P,K) array of blob sizes.

    Blake image system  (r = x−y, R = x−y*, y* = (y1,y2,−h), h = y3):
      G_ij = S_ij(r) − S_ij(R) + 2h M_jj D_ij,
      D_ij = ∂/∂R_j [h R_i/R³ − δ_i3/R − R_i R_3/R³]
           = −δ_ij x3/R³ + 3 x3 R_i R_j/R⁵ + (δ_i3 R_j − R_i δ_j3)/R³,
    with S_ij(r) = δ_ij (r²+2ε²)/r_ε³ + r_i r_j / r_ε³ (Cortez) and
    |R| → sqrt(|R|²+ε²) in all image terms.  M = diag(1,1,−1).
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    e2 = np.broadcast_to(np.asarray(eps, float) ** 2, (x.shape[0], y.shape[0]))
    r = x[:, None, :] - y[None, :, :]
    r2 = np.einsum("pki,pki->pk", r, r)
    d3 = (r2 + e2) ** -1.5
    G = r[..., :, None] * (r * d3[..., None])[..., None, :]
    diag = (r2 + 2 * e2) * d3
    if wall:
        h = y[None, :, 2]
        x3 = x[:, None, 2] * np.ones_like(h)
        R = r.copy()
        R[..., 2] = x[:, None, 2] + y[None, :, 2]
        R2 = np.einsum("pki,pki->pk", R, R)
        D3 = (R2 + e2) ** -1.5
        D5 = D3 / (R2 + e2)
        diag = diag - (R2 + 2 * e2) * D3
        hx = 2 * h * x3
        Mv = np.array([1.0, 1.0, -1.0])
        # − R_i R_j (D3 − 6 h x3 D5 M_jj)
        coefRR = D3[..., None] - 3 * (hx * D5)[..., None] * Mv
        G -= R[..., :, None] * (R * coefRR)[..., None, :]
        # 2h M_jj D3 (δ_i3 R_j − R_i δ_j3)
        c = 2 * h * D3
        G[..., 2, :] += (c[..., None] * Mv) * R
        G[..., :, 2] += c[..., None] * R          # −R_i·(M_33 = −1)
        # δ_ij (−2h x3 D3 M_jj)
        dd = (hx * D3)
        G[..., 0, 0] -= dd
        G[..., 1, 1] -= dd
        G[..., 2, 2] += dd
    G[..., 0, 0] += diag
    G[..., 1, 1] += diag
    G[..., 2, 2] += diag
    return G


def mobility_matrix(X: np.ndarray, eps, eta_int: float, wall: bool = True) -> np.ndarray:
    """(3K,3K) matrix M with  u = M F  (point-major ordering 3k+c)."""
    K = X.shape[0]
    G = kernel(X, X, eps, wall)
    return G.transpose(0, 2, 1, 3).reshape(3 * K, 3 * K) / (8 * np.pi * eta_int)


def velocity_field(xg: np.ndarray, X: np.ndarray, F: np.ndarray, eps: float,
                   eta_int: float, wall: bool = True, chunk: int = 4000) -> np.ndarray:
    """Velocity at field points xg (P,3) due to forces F (K,3) at X (K,3)."""
    out = np.empty_like(xg, dtype=float)
    for i in range(0, xg.shape[0], chunk):
        G = kernel(xg[i:i + chunk], X, eps, wall)
        out[i:i + chunk] = np.einsum("pkij,kj->pi", G, F)
    return out / (8 * np.pi * eta_int)


def check_kernel(eps: float = 0.2, verbose: bool = True) -> dict:
    """Sanity checks of the Blake kernel: wall no-slip and the near-wall
    point-force mobilities  μ∥ = 1/(6πηh)·(−9/16),  μ⊥ = 1/(6πηh)·(−9/8)."""
    rng = np.random.default_rng(1)
    y = rng.uniform([-5, -5, 1.0], [5, 5, 5.0], size=(20, 3))
    xw = rng.uniform([-20, -20, 0], [20, 20, 0], size=(50, 3))
    G0 = kernel(xw, y, 1e-6)
    Ge = kernel(xw, y, eps)
    Gfree = kernel(xw, y, eps, wall=False)
    res0 = np.abs(G0).max() / np.abs(Gfree).max()
    rese = np.abs(Ge).max() / np.abs(Gfree).max()
    h = 3.0
    yy = np.array([[0.0, 0.0, h]])
    Gi = kernel(yy, yy, 1e-8)[0, 0] - kernel(yy, yy, 1e-8, wall=False)[0, 0]
    # convert image part to mobility × 6πη h
    mu_par = Gi[0, 0] / (8 * np.pi) * 6 * np.pi * h
    mu_perp = Gi[2, 2] / (8 * np.pi) * 6 * np.pi * h
    out = dict(wall_residual_eps0=res0, wall_residual_eps=rese,
               mu_par=mu_par, mu_par_exact=-9 / 16, mu_perp=mu_perp, mu_perp_exact=-9 / 8)
    if verbose:
        print("[kernel check]")
        print(f"  wall velocity / free velocity  (eps→0) : {res0:.2e}")
        print(f"  wall velocity / free velocity  (eps={eps}) : {rese:.2e}   (h ≥ 1 µm)")
        print(f"  image mobility ∥ ×6πηh : {mu_par:+.5f}  (Blake/Faxén −9/16 = {-9/16:+.5f})")
        print(f"  image mobility ⊥ ×6πηh : {mu_perp:+.5f}  (Blake/Lorentz −9/8 = {-9/8:+.5f})")
    return out


# =============================================================================
# ε calibration (so that the discretised filament has the drag of a radius-a rod)
# =============================================================================
def rft_coeffs(L: float, a: float, eta_int: float) -> tuple[float, float]:
    """Gray & Hancock resistive-force coefficients (λ → L) [pN·s/µm²]."""
    lg = np.log(2 * L / a)
    Cn = 4 * np.pi * eta_int / (lg + 0.5)
    Ct = 2 * np.pi * eta_int / (lg - 0.5)
    return Cn, Ct


def rod_points(L: float, N_seg: int) -> np.ndarray:
    s = (np.arange(N_seg) + 0.5) * L / N_seg
    return s


@lru_cache(maxsize=64)
def calibrate_eps(L: float, a: float, N_seg: int) -> float:
    """Choose ε such that a straight rod (free space) translating broadside has
    the slender-body drag  F = 4πηLU / (ln(L/a) + 0.5)."""
    s = rod_points(L, N_seg)
    X = np.c_[s, np.zeros_like(s), np.zeros_like(s)]
    target = 4 * np.pi * L / (np.log(L / a) + 0.5)          # for η=1, U=1
    V = np.zeros((N_seg, 3)); V[:, 2] = 1.0

    def drag(log_eps):
        M = mobility_matrix(X, np.exp(log_eps), 1.0, wall=False)
        F = np.linalg.solve(M, V.ravel()).reshape(-1, 3)
        return F[:, 2].sum() - target

    lo, hi = np.log(1e-3 * a), np.log(L / 2)
    return float(np.exp(brentq(drag, lo, hi, xtol=1e-6)))


def get_eps(p: Params) -> float:
    return p.eps if p.eps is not None else calibrate_eps(p.L, p.a, p.N_seg)


# =============================================================================
# Beat waveform
# =============================================================================
_SUB = 4  # sub-samples per segment for integrating the tangent angle (must be even)


def _base_angle(u, p: Params):
    """'delay' waveform: basal tangent angle, fast forward sweep (u < p_eff), slow return."""
    u = np.mod(u, 1.0)
    pe = p.p_eff
    a = np.where(u < pe, -np.cos(np.pi * u / pe),
                 np.cos(np.pi * (u - pe) / (1 - pe)))
    return p.center + 0.5 * p.Theta_b * a


def tangent_angle(s: np.ndarray, phi: np.ndarray, p: Params) -> np.ndarray:
    """ψ(s, φ) for arc length s (M,) and phases φ (N,) → (N, M)."""
    if p.waveform == "delay":
        u = np.mod(phi, TWO_PI)[:, None] / TWO_PI
        env = 1.0 - (1.0 - p.taper) * s[None, :] / p.L
        return p.center + env * (_base_angle(u - p.lag * s[None, :] / p.L, p) - p.center)
    u = np.mod(phi, TWO_PI)[:, None] / TWO_PI
    th = p.theta0
    # effective stroke: straight rod rotating -θ0 → +θ0
    ue = np.clip(u / p.p_eff, 0.0, 1.0)
    psi_eff = -th * np.cos(np.pi * ue) * np.ones_like(s)[None, :]
    # recovery stroke: bend propagating from base (s_b=-w) to tip (s_b=L+w)
    w = p.bend_width * p.L
    ur = np.clip((u - p.p_eff) / (1.0 - p.p_eff), 0.0, 1.0)
    sb = -w + (p.L + 2 * w) * 0.5 * (1 - np.cos(np.pi * ur))
    xi = np.clip((s[None, :] - sb) / w, -1.0, 1.0)
    S = 0.5 + 0.75 * xi - 0.25 * xi ** 3                    # C¹ smooth step 0 → 1
    psi_rec = -th + 2 * th * S
    return np.where(u < p.p_eff, psi_eff, psi_rec)


def shape(phi: np.ndarray, p: Params) -> np.ndarray:
    """Positions of the N_seg Stokeslets relative to the base, (N, N_seg, 2) in (x,z)."""
    M = p.N_seg * _SUB
    hs = p.L / M
    s_mid = (np.arange(M) + 0.5) * hs
    psi = tangent_angle(s_mid, np.atleast_1d(phi), p)
    dx = np.cumsum(hs * np.sin(psi), axis=1)
    dz = np.cumsum(hs * np.cos(psi), axis=1)
    idx = (np.arange(p.N_seg) * _SUB + _SUB // 2) - 1        # nodes at s=(k+½)Δs
    return np.stack([dx[:, idx], dz[:, idx]], axis=-1)


def shape_full(phi: np.ndarray, p: Params, n: int = 80) -> np.ndarray:
    """Finely sampled centre-line incl. the base, (N, n+1, 2) — for drawing."""
    hs = p.L / n
    s_mid = (np.arange(n) + 0.5) * hs
    psi = tangent_angle(s_mid, np.atleast_1d(phi), p)
    dx = np.concatenate([np.zeros((psi.shape[0], 1)), np.cumsum(hs * np.sin(psi), 1)], 1)
    dz = np.concatenate([np.zeros((psi.shape[0], 1)), np.cumsum(hs * np.cos(psi), 1)], 1)
    return np.stack([dx, dz], -1)


def shape_and_tangent(phi: np.ndarray, p: Params, dphi: float = 1e-5):
    """Shape (N,N_seg,2) and ∂X/∂φ (N,N_seg,2)."""
    X = shape(phi, p)
    dX = (shape(phi + dphi, p) - shape(phi - dphi, p)) / (2 * dphi)
    return X, dX


def in_effective(phi: np.ndarray, p: Params) -> np.ndarray:
    return np.mod(phi, TWO_PI) < TWO_PI * p.p_eff


def chord_sweep(p: Params, n: int = 400) -> float:
    """Total angular sweep of the base→tip chord over a cycle [rad]."""
    sh = shape_full(np.linspace(0, TWO_PI, n, endpoint=False), p)
    ang = np.arctan2(sh[:, -1, 0], sh[:, -1, 1])
    return float(ang.max() - ang.min())


def calibrate_lag(p: Params, target_chord: float) -> float:
    """'delay' waveform: lag λ such that the chord sweep equals target_chord [rad]."""
    from scipy.optimize import brentq
    f = lambda lam: chord_sweep(p.copy(lag=lam)) - target_chord
    if f(0.0) < 0:
        return 0.0
    return float(brentq(f, 0.0, 0.95, xtol=1e-4))


# =============================================================================
# Hydrodynamic solver for one instant
# =============================================================================
class System:
    """Holds the geometry and solves the instantaneous Stokes problem."""

    def __init__(self, p: Params):
        self.p = p
        self.eps = get_eps(p)
        self.base_x = np.arange(p.N) * p.d
        K = p.N * p.N_seg
        self.K = K
        owner = np.repeat(np.arange(p.N), p.N_seg)
        self.owner = owner
        if p.eps_inter is not None and p.eps_inter != self.eps and p.N > 1:
            same = owner[:, None] == owner[None, :]
            self.eps_mat = np.where(same, self.eps, max(p.eps_inter, self.eps))
        else:
            self.eps_mat = self.eps
        self.base_y = (np.arange(p.N) % 2 - 0.5) * p.y_stagger if p.N > 1 else np.zeros(1)
        # e: unit x-velocity of every point (wall/base translation)
        self.e = np.zeros(3 * K); self.e[0::3] = 1.0

    def points(self, phi):
        p = self.p
        X2, dX2 = shape_and_tangent(phi, p)
        X = np.zeros((p.N, p.N_seg, 3)); T = np.zeros_like(X)
        X[..., 0] = X2[..., 0] + self.base_x[:, None]
        X[..., 1] = self.base_y[:, None]
        X[..., 2] = X2[..., 1]
        T[..., 0] = dX2[..., 0]
        T[..., 2] = dX2[..., 1]
        return X.reshape(-1, 3), T.reshape(p.N, -1)        # T: (N, 3*N_seg)

    def tangent_matrix(self, T):
        """(3K, N) matrix whose column i is ∂X/∂φ_i."""
        p = self.p
        Tm = np.zeros((3 * self.K, p.N))
        n3 = 3 * p.N_seg
        for i in range(p.N):
            Tm[i * n3:(i + 1) * n3, i] = T[i]
        return Tm

    def solve(self, phi, omega_fixed=None, T_stall=None, omega0=None):
        """Return dict with phidot (N), U, F (K,3), X (K,3), Q (N)."""
        p = self.p
        X, T = self.points(phi)
        Tm = self.tangent_matrix(T)
        M = mobility_matrix(X, self.eps_mat, p.eta_int, wall=True)
        rhs = np.column_stack([Tm, self.e])
        Y = np.linalg.solve(M, rhs)                   # A·[T | e],  A = M⁻¹ (resistance)
        YT, Ye = Y[:, :p.N], Y[:, p.N]
        Gam = Tm.T @ YT                               # (N,N) generalized resistance
        g = Tm.T @ Ye                                 # (N,)  coupling φ̇ ↔ U
        c = self.e @ YT                               # (N,)  Σ F_x per unit φ̇_i
        b = self.e @ Ye                               # Σ F_x per unit U
        swim = p.wall_mode == "swimming"
        zeta_eta = p.zeta * p.eta_int
        if p.beat_mode == "fixed":
            phidot = np.full(p.N, omega_fixed)
            U = -(c @ phidot) / (zeta_eta + b) if swim else 0.0
        elif p.beat_mode == "load":
            k = omega0 / T_stall
            if swim:
                A = np.zeros((p.N + 1, p.N + 1))
                A[:p.N, :p.N] = np.eye(p.N) + k * Gam
                A[:p.N, p.N] = k * g
                A[p.N, :p.N] = c
                A[p.N, p.N] = zeta_eta + b
                sol = np.linalg.solve(A, np.r_[np.full(p.N, omega0), 0.0])
                phidot, U = sol[:p.N], sol[p.N]
            else:
                phidot = np.linalg.solve(np.eye(p.N) + k * Gam, np.full(p.N, omega0))
                U = 0.0
        else:
            raise ValueError(p.beat_mode)
        F = (YT @ phidot + Ye * U).reshape(-1, 3)
        Q = Tm.T @ F.ravel()
        return dict(phidot=phidot, U=U, F=F, X=X, Q=Q, Gam=Gam)


# =============================================================================
# Motor calibration
# =============================================================================
def resistance_table_single(p: Params, eta: float, n_phi: int = 96):
    """Γ(φ) of a single anchored flagellum on a uniform φ grid [pN·µm·s/rad²]."""
    q = p.copy(N=1, eta=eta, wall_mode="anchored", beat_mode="fixed")
    sysm = System(q)
    grid = np.linspace(0, TWO_PI, n_phi, endpoint=False)
    vals = np.array([sysm.solve(np.array([ph]), omega_fixed=1.0)["Gam"][0, 0] for ph in grid])
    return grid, vals


def mean_resistance_single(p: Params, eta: float, n_phi: int = 96) -> float:
    """Phase-average ⟨Γ(φ)⟩ of a single anchored flagellum."""
    return float(resistance_table_single(p, eta, n_phi)[1].mean())


def phase_map(p: Params, omega0: float, T_stall: float, n_phi: int = 96):
    """Map raw phase φ → uniformly rotating phase θ of the isolated oscillator,
        θ(φ) = 2π ∫_0^φ dφ'/φ̇₁(φ') / ∫_0^{2π} dφ'/φ̇₁,   φ̇₁ = ω0/(1+ω0Γ(φ)/T_stall).
    Returns (φ grid incl. 2π, θ grid)."""
    grid, G = resistance_table_single(p, p.eta, n_phi)
    inv = (1 + omega0 * G / T_stall) / omega0                # 1/φ̇₁
    inv = np.r_[inv, inv[0]]
    g = np.r_[grid, TWO_PI]
    cum = np.r_[0.0, np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(g))]
    return g, TWO_PI * cum / cum[-1]


def to_theta(phi: np.ndarray, pm) -> np.ndarray:
    if pm is None:
        return phi
    g, th = pm
    k = np.floor(phi / TWO_PI)
    return k * TWO_PI + np.interp(phi - k * TWO_PI, g, th)


def motor_params(p: Params) -> tuple[float, float]:
    """(ω0, T_stall). If T_stall is None it is chosen so that a single anchored
    flagellum in water (η = eta_water) beats at f_water on average:
        period = ∫dφ/φ̇ = (2π/ω0)(1 + ω0⟨Γ⟩/T_stall)."""
    omega0 = TWO_PI * p.f_free
    if p.T_stall is not None:
        return omega0, p.T_stall
    if p.f_free <= p.f_water:
        raise ValueError("f_free must be larger than f_water")
    G = mean_resistance_single(p, p.eta_water)
    return omega0, omega0 * G / (p.f_free / p.f_water - 1.0)


def estimated_frequency(p: Params, omega0: float, T_stall: float) -> float:
    if p.beat_mode == "fixed":
        return p.f_fixed
    G = mean_resistance_single(p, p.eta)
    return omega0 / (TWO_PI * (1 + omega0 * G / T_stall))


# =============================================================================
# Initial phases
# =============================================================================
def initial_phases(p: Params) -> np.ndarray:
    if p.init_phase == "inphase":
        return np.zeros(p.N)
    if p.init_phase == "metachronal":
        return np.mod(np.arange(p.N) * p.dphi_mc, TWO_PI)
    if p.init_phase == "random":
        return np.random.default_rng(p.seed).uniform(0, TWO_PI, p.N)
    raise ValueError(p.init_phase)


# =============================================================================
# Time evolution
# =============================================================================
def run(p: Params, verbose: bool = True, store_frames: bool = True) -> dict:
    """Integrate the phase dynamics and record forces.  Returns a dict of arrays."""
    sysm = System(p)
    omega0, T_stall = motor_params(p) if p.beat_mode == "load" else (TWO_PI * p.f_fixed, None)
    f_est = estimated_frequency(p, omega0, T_stall)
    period = 1.0 / f_est
    dt = p.dt if p.dt is not None else period / p.steps_per_cycle
    n_steps = int(math.ceil(p.n_cycles * period / dt))
    omega_fixed = TWO_PI * p.f_fixed

    def rhs(phi):
        return sysm.solve(phi, omega_fixed=omega_fixed, T_stall=T_stall, omega0=omega0)

    phi = initial_phases(p).astype(float)
    N = p.N
    rec = {k: np.zeros((n_steps, N)) for k in ("phi", "phidot", "Fx", "Fz", "Q")}
    rec["U"] = np.zeros(n_steps)
    rec["t"] = np.arange(n_steps) * dt
    frame_every = max(1, int(round(period / dt / p.frames_per_cycle)))
    first_frame = n_steps - int(p.frame_cycles * period / dt)
    frames = dict(t=[], phi=[], X=[], F=[], U=[])

    for n in range(n_steps):
        s = rhs(phi)
        Fk = s["F"].reshape(N, p.N_seg, 3)
        rec["phi"][n] = phi
        rec["phidot"][n] = s["phidot"]
        rec["Fx"][n] = Fk[..., 0].sum(1)
        rec["Fz"][n] = Fk[..., 2].sum(1)
        rec["Q"][n] = s["Q"]
        rec["U"][n] = s["U"]
        if store_frames and n >= first_frame and (n - first_frame) % frame_every == 0:
            frames["t"].append(n * dt); frames["phi"].append(phi.copy())
            frames["X"].append(s["X"]); frames["F"].append(s["F"]); frames["U"].append(s["U"])
        if p.beat_mode == "fixed" or p.integrator == "euler":
            phi = phi + dt * s["phidot"]
        else:  # Heun (RK2)
            k1 = s["phidot"]
            k2 = rhs(phi + dt * k1)["phidot"]
            phi = phi + 0.5 * dt * (k1 + k2)
        if verbose and (n % max(1, n_steps // 10) == 0):
            print(f"  step {n:6d}/{n_steps}  t={n*dt*1e3:7.2f} ms  "
                  f"<f_beat>={np.mean(s['phidot'])/TWO_PI:6.2f} Hz  ΣFx={rec['Fx'][n].sum():8.2f} pN")

    pm = phase_map(p, omega0, T_stall) if p.beat_mode == "load" else None
    out = dict(params=p.as_dict(), dt=dt, period_est=period, eps=sysm.eps, phase_map=pm,
               omega0=omega0, T_stall=T_stall, base_x=sysm.base_x, **rec)
    out["frames"] = {k: np.array(v) for k, v in frames.items()}
    return out


# =============================================================================
# Analysis
# =============================================================================
def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def analyze(res: dict) -> dict:
    """Effective-stroke forces, total thrust, synchrony and metachrony indices."""
    p = Params(**res["params"])
    n0 = int(p.transient * len(res["t"]))
    phi = res["phi"][n0:]
    Fx = res["Fx"][n0:]
    eff = in_effective(phi, p)
    f_i = np.array([Fx[eff[:, i], i].mean() if eff[:, i].any() else np.nan for i in range(p.N)])
    out = dict(
        f_i=f_i,                                  # effective-stroke mean x-force per flagellum
        f_mean=np.nanmean(f_i),
        F_total=np.nansum(f_i),                   # Σ_i f_i
        Fx_cycle_mean=Fx.sum(1).mean(),           # full-cycle mean of Σ F_x
        freq=res["phidot"][n0:].mean() / TWO_PI,
        freq_i=res["phidot"][n0:].mean(0) / TWO_PI,
        U_mean=res["U"][n0:].mean(),
        power=(res["Q"][n0:] * res["phidot"][n0:]).sum(1).mean(),
    )
    # synchrony is measured with the uniformly rotating phase θ(φ) of the isolated
    # oscillator (identical to φ in "fixed" mode), so that a constant time lag ↔ constant Δθ
    theta = to_theta(phi, res.get("phase_map"))
    out["theta"] = theta
    z = np.exp(1j * theta)
    out["R_sync_t"] = np.abs(z.mean(1))
    out["R_sync"] = out["R_sync_t"].mean()
    if p.N > 1:
        dphi = wrap(np.diff(theta, axis=1))       # θ_{i+1} − θ_i
        zz = np.exp(1j * dphi)
        out["dphi"] = dphi
        out["dphi_mean"] = float(np.angle(zz.mean()))
        out["dphi_coherence"] = float(np.abs(zz.mean()))
        # per-pair phase locking: 1 if every neighbour pair keeps a constant lag (any value)
        out["dphi_lock"] = float(np.abs(zz.mean(0)).mean())
        # classify: phase φ_i = ωt − k x_i ⇒ Δφ<0 → wave travels +x (direction of effective stroke)
        if out["dphi_coherence"] < 0.5:
            kind = "phase-locked, non-uniform lags" if out["dphi_lock"] > 0.8 else "disordered"
        elif abs(out["dphi_mean"]) < 0.15 * np.pi:
            kind = "synchronous (in-phase)"
        elif abs(out["dphi_mean"]) > 0.85 * np.pi:
            kind = "anti-phase"
        else:
            kind = ("symplectic MW (wave along effective stroke)" if out["dphi_mean"] < 0
                    else "antiplectic MW (wave against effective stroke)")
        out["wave_kind"] = kind
    else:
        out["dphi"] = np.zeros((len(phi), 0))
        out["dphi_mean"] = np.nan
        out["dphi_coherence"] = np.nan
        out["dphi_lock"] = np.nan
        out["wave_kind"] = "single"
    return out


# =============================================================================
# Resistive force theory check
# =============================================================================
def rft_check(p: Params | None = None, n_phi: int = 120, verbose: bool = True) -> dict:
    """Compare the force on ONE flagellum beating at f_fixed computed with
    (i) regularized Stokeslets near the wall, (ii) regularized Stokeslets in free space,
    (iii) local resistive force theory (Gray & Hancock, no wall)."""
    p = (p or Params()).copy(N=1, beat_mode="fixed", wall_mode="anchored")
    eps = get_eps(p)
    om = TWO_PI * p.f_fixed
    Cn, Ct = rft_coeffs(p.L, p.a, p.eta_int)
    ds = p.L / p.N_seg
    phis = np.linspace(0, TWO_PI, n_phi, endpoint=False)
    Fx = {"stokeslet+wall": [], "stokeslet_free": [], "RFT": []}
    for ph in phis:
        X2, dX2 = shape_and_tangent(np.array([ph]), p)
        X = np.zeros((p.N_seg, 3)); V = np.zeros_like(X)
        X[:, 0], X[:, 2] = X2[0, :, 0], X2[0, :, 1]
        V[:, 0], V[:, 2] = om * dX2[0, :, 0], om * dX2[0, :, 1]
        for key, wall in (("stokeslet+wall", True), ("stokeslet_free", False)):
            F = np.linalg.solve(mobility_matrix(X, eps, p.eta_int, wall), V.ravel()).reshape(-1, 3)
            Fx[key].append(F[:, 0].sum())
        # RFT: force on fluid  f = (Cn (I − tt) + Ct tt) v  ds
        psi = tangent_angle((np.arange(p.N_seg) + 0.5) * ds, np.array([ph]), p)[0]
        t = np.c_[np.sin(psi), np.zeros_like(psi), np.cos(psi)]
        vt = np.einsum("ki,ki->k", V, t)
        f = Cn * (V - vt[:, None] * t) + Ct * vt[:, None] * t
        Fx["RFT"].append(f[:, 0].sum() * ds)
    eff = in_effective(phis, p)
    out = {}
    for k, v in Fx.items():
        v = np.array(v)
        out[k] = dict(eff_mean=v[eff].mean(), peak=v.max(), cycle_mean=v.mean(), series=v)
    # crude hand estimate: rod rotating by 2θ0 during p_eff·T, force at mid-stroke
    w_eff = 2 * p.theta0 * np.pi / 2 / (p.p_eff / p.f_fixed)      # peak angular speed
    est = Cn * w_eff * p.L ** 2 / 2
    out["hand_estimate_peak"] = est
    out["phis"] = phis
    if verbose:
        print(f"[RFT check]  L={p.L} µm, a={p.a} µm, η={p.eta} mPa·s, f={p.f_fixed} Hz, "
              f"N_seg={p.N_seg}, ε={eps:.3f} µm")
        print(f"  C_n={Cn/p.eta_int:.3f}η, C_t={Ct/p.eta_int:.3f}η  (pN·s/µm² per mPa·s ×1e-3)")
        print(f"  {'method':18s} {'<Fx>_eff [pN]':>14s} {'peak Fx [pN]':>13s} {'<Fx>_cycle [pN]':>16s}")
        for k in Fx:
            o = out[k]
            print(f"  {k:18s} {o['eff_mean']:14.3f} {o['peak']:13.3f} {o['cycle_mean']:16.3f}")
        print(f"  hand estimate of peak (rigid rod, RFT) : {est:.3f} pN")
        r = out["stokeslet_free"]["eff_mean"] / out["RFT"]["eff_mean"]
        rw = out["stokeslet+wall"]["eff_mean"] / out["RFT"]["eff_mean"]
        print(f"  ratio Stokeslet(free)/RFT = {r:.2f},  Stokeslet(wall)/RFT = {rw:.2f}")
    return out


if __name__ == "__main__":
    check_kernel()
    p = Params()
    print(f"calibrated eps (L={p.L}, a={p.a}, N_seg={p.N_seg}) = {get_eps(p):.4f} µm")
    rft_check(p)
