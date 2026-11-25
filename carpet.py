"""
carpet.py — doubly periodic "ciliary carpet": a local model of the colony surface.

Geometry
--------
Flagella stand on the no-slip wall z = 0 on a rectangular lattice:
  * along x (beat direction, = meridian of the colony): spacing d,
    n flagella per period  →  period P = n d  (one metachronal wavelength λ = P)
  * along y (perpendicular to the beat plane, = latitude): spacing dy (default d),
    lateral neighbours beat IN PHASE (as observed for Volvox: waves travel along
    the meridian, rows of equal latitude are synchronous).
  * optionally 2 flagella per cell (biflagellate somatic cells) separated by
    `pair_sep` in y, beating together.
The lattice is infinite in both directions.  Hydrodynamic images are summed
exactly (regularized Stokeslet + Blake) for flagellum pairs whose bases are
closer than R_near; the remaining (far) images are smeared into a continuous
force density and added through a tabulated angular-averaged Blake kernel.

Because the carpet is flat and infinite, translating the whole carpet (wall +
flagella) at a speed V only adds the uniform flow V everywhere; flagellar forces
are therefore identical in the "anchored" and "free" situations, and the carpet
swims (relative to the far fluid) with the slip velocity
      U_slip = ⟨Σ_k F_x,k z_k⟩ / (η A_cell)          (exact for a planar layer)
(force density f at height h on a wall gives the plug flow f h/η above it).
For a spherical colony with uniform tangential slip U_slip, Stone & Samuel (1996)
give U_swim = (π/4) U_slip.
"""
from __future__ import annotations

import math
from functools import lru_cache

import os
import numpy as np

from model import (Params, kernel, get_eps, shape_and_tangent, in_effective, motor_params,
                   resistance_table_single, phase_map, to_theta, wrap, TWO_PI, ETA_UNIT)


# -----------------------------------------------------------------------------
# far-field (smeared) Blake kernel: angular average over a ring of radius r
# -----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def tail_table(R_near: float, zmax: float, nz: int = 28, n_r: int = 260, n_th: int = 48):
    """T[a,b,c] = ∫_{R_near}^{∞} r dr ∫ dθ G_cc(r,θ; z_a, z_b)   for c = (xx, zz)
    (diagonal; xx = yy).  Divide by the area per source point and by 8πη to use."""
    zg = np.linspace(0.0, zmax, nz)
    r = np.geomspace(R_near, 2e5, n_r)
    th = np.linspace(0, TWO_PI, n_th, endpoint=False)
    xs = (r[:, None] * np.cos(th)[None, :]).ravel()
    ys = (r[:, None] * np.sin(th)[None, :]).ravel()
    T = np.zeros((nz, nz, 2))
    w_r = np.gradient(r) * r                     # r dr
    for b, zs in enumerate(zg):
        src = np.array([[0.0, 0.0, max(zs, 1e-3)]])
        for a, zf in enumerate(zg):
            pts = np.c_[xs, ys, np.full(xs.size, zf)]
            G = kernel(pts, src, 1e-6)[:, 0]          # (n_r*n_th, 3, 3)
            gxx = G[:, 0, 0].reshape(n_r, n_th).mean(1) * TWO_PI
            gzz = G[:, 2, 2].reshape(n_r, n_th).mean(1) * TWO_PI
            T[a, b, 0] = np.sum(gxx * w_r)
            T[a, b, 1] = np.sum(gzz * w_r)
    return zg, T


def _interp2(zg, T, za, zb):
    """bilinear interpolation of T(zg,zg) at all pairs (za_i, zb_k)."""
    def idx(z):
        z = np.clip(z, zg[0], zg[-1])
        h = zg[1] - zg[0]
        i = np.minimum((z / h).astype(int), len(zg) - 2)
        return i, (z - zg[i]) / h
    ia, ta = idx(za); ib, tb = idx(zb)
    ia, ta = ia[:, None], ta[:, None, None]
    ib, tb = ib[None, :], tb[None, :, None]
    return ((1 - ta) * (1 - tb) * T[ia, ib] + ta * (1 - tb) * T[ia + 1, ib]
            + (1 - ta) * tb * T[ia, ib + 1] + ta * tb * T[ia + 1, ib + 1])


# -----------------------------------------------------------------------------
# Lattice-summed kernel tables
# -----------------------------------------------------------------------------
def _smooth_w(r, R0, R1):
    """1 for r ≤ R0, 0 for r ≥ R1, C¹ in between (near/far partition of the my=0 image row)."""
    t = np.clip((r - R0) / (R1 - R0), 0, 1)
    return 1 - t * t * (3 - 2 * t)


def _kernel_xz(dx, ylist, zf, zs):
    """Σ_{y∈ylist} G(dx, y, zf; zs) for arrays dx (m,), returns (m, 4) = xx, xz, zx, zz."""
    out = np.zeros((dx.size, 4))
    src = np.array([[0.0, 0.0, zs]])
    for ych in np.array_split(ylist, max(1, ylist.size // 64)):
        pts = np.c_[np.repeat(dx, ych.size), np.tile(ych, dx.size), np.full(dx.size * ych.size, zf)]
        G = kernel(pts, src, 1e-6)[:, 0].reshape(dx.size, ych.size, 3, 3).sum(1)
        out += np.stack([G[:, 0, 0], G[:, 0, 2], G[:, 2, 0], G[:, 2, 2]], -1)
    return out


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kernel_cache")


def _disk_cached(prefix):
    """Cache the returned tuple of arrays in .kernel_cache/<prefix>_<args>.npz."""
    def deco(fn):
        def wrapper(*args, **kw):
            key = "_".join(f"{a:g}" if isinstance(a, float) else str(a) for a in args)
            key += "".join(f"_{k}{v}" for k, v in sorted(kw.items()))
            path = os.path.join(CACHE_DIR, f"{prefix}_{key}.npz")
            if os.path.exists(path):
                with np.load(path) as z:
                    return tuple(z[f"a{i}"] for i in range(len(z.files)))
            out = fn(*args, **kw)
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, **{f"a{i}": a for i, a in enumerate(out)})
            os.replace(tmp, path)
            return out
        return lru_cache(maxsize=8)(wrapper)
    return deco


@_disk_cached("line")
def line_tables(L: float, dy: float, R0: float, R1: float, Xmax: float, nz: int = 24):
    """Row-summed kernels on a non-uniform Δx grid (Δx ≥ 0):
         lat(Δx,z,z') = Σ_{my≠0} G(Δx, my·dy, z; z')         (lateral rows, |my|·dy ≤ Xmax)
         far0(Δx,z,z') = (1 − w(|Δx|)) G(Δx, 0, z; z')         (own row, far part)
       Symmetry: xx, zz even in Δx; xz, zx odd."""
    zg = np.linspace(0.0, 1.05 * L, nz)
    xg = np.unique(np.r_[np.arange(0, 60, 0.5), np.arange(60, 300, 2.0),
                         np.arange(300, Xmax + 10, 10.0)])
    M = int(Xmax // dy)
    ylat = np.r_[-np.arange(1, M + 1)[::-1], np.arange(1, M + 1)] * dy
    lat = np.zeros((xg.size, nz, nz, 4)); far0 = np.zeros_like(lat)
    w = _smooth_w(xg, R0, R1)
    for b, zs in enumerate(zg):
        zs_ = max(zs, 1e-3)
        for a, zf in enumerate(zg):
            lat[:, a, b] = _kernel_xz(xg, ylat, zf, zs_)
            far0[:, a, b] = (1 - w)[:, None] * _kernel_xz(xg, np.array([0.0]), zf, zs_)
    return xg, zg, lat, far0


@_disk_cached("period")
def period_table(L: float, d: float, dy: float, n: int, R0: float, R1: float,
                 Xmax: float = 3000.0, dxs: float = 0.5):
    """T_P(Δx mod P, z, z') = Σ_mx [lat + far0](Δx + mx P) + smeared tail beyond Xmax."""
    xg, zg, lat, far0 = line_tables(L, dy, R0, R1, Xmax)
    S = lat + far0
    P = n * d
    nxs = int(math.ceil(P / dxs))
    xs = np.arange(nxs) * (P / nxs)                 # uniform grid that divides P exactly
    nz = zg.size
    TP = np.zeros((xs.size, nz, nz, 4))
    sign = np.array([1, -1, -1, 1.0])              # parity of xx, xz, zx, zz
    mmax = int(Xmax // P) + 1
    flat = S.reshape(xg.size, -1)
    for mx in range(-mmax, mmax + 1):
        X = xs + mx * P
        ok = np.abs(X) <= Xmax
        if not ok.any():
            continue
        ax = np.abs(X[ok])
        i = np.clip(np.searchsorted(xg, ax) - 1, 0, xg.size - 2)
        t = ((ax - xg[i]) / (xg[i + 1] - xg[i]))[:, None]
        val = ((1 - t) * flat[i] + t * flat[i + 1]).reshape(-1, nz, nz, 4)
        sg = np.where(X[ok] < 0, 1.0, 0.0)[:, None, None, None]
        val = val * (1 - sg) + val * sign * sg
        TP[ok] += val
    # smeared tail outside the summed square (|x|,|y| ≤ Xmax): annulus r > Xmax
    zt, Tt = tail_table(float(Xmax), round(1.05 * L, 3), nz=nz)
    TP[..., 0] += Tt[..., 0] / (P * dy)          # copies of one point have density 1/(P dy)
    TP[..., 3] += Tt[..., 1] / (P * dy)
    return xs, zg, TP


def blake_xz_pairs(dx, zf, zs, eps):
    """x–z block of the regularized Blake kernel for point pairs in the same y-plane
    (r_y = 0).  dx = x_f − x_s, zf field height, zs source height.  Returns xx, xz, zx, zz."""
    e2 = eps * eps
    r3 = zf - zs
    R3 = zf + zs
    r2 = dx * dx + r3 * r3
    R2 = dx * dx + R3 * R3
    d3 = (r2 + e2) ** -1.5
    D3 = (R2 + e2) ** -1.5
    D5 = D3 / (R2 + e2)
    diag = (r2 + 2 * e2) * d3 - (R2 + 2 * e2) * D3
    q = 6 * zs * zf * D5
    dd = 2 * zs * zf * D3
    c = 2 * zs * D3
    gxx = dx * dx * d3 + diag - dx * dx * (D3 - q) - dd
    gxz = dx * r3 * d3 - dx * R3 * (D3 + q) + c * dx
    gzx = r3 * dx * d3 - R3 * dx * (D3 - q) + c * dx
    gzz = r3 * r3 * d3 + diag - R3 * R3 * (D3 + q) + dd
    return gxx, gxz, gzx, gzz


class Carpet:
    """Periodic carpet (x-period P = n d, lateral spacing dy, lateral rows in phase).
    Reduced to x–z components (F_y = 0 by the y → −y symmetry)."""

    def __init__(self, p: Params, n: int, dy: float | None = None, R0: float | None = None,
                 R1: float | None = None, Xmax: float = 3000.0):
        self.p = p
        self.n = n
        self.d = p.d
        self.dy = dy if dy is not None else p.d
        self.P = n * p.d
        self.eps = get_eps(p)
        self.eps_x = max(p.eps_inter or self.eps, self.eps)
        self.R0 = R0 if R0 is not None else 1.0 * p.L
        self.R1 = R1 if R1 is not None else 2.0 * p.L
        self.nf = n
        self.npc = 1
        self.col = np.arange(n)
        self.bx = np.arange(n) * p.d
        ns = p.N_seg
        self.K = n * ns
        self.owner = np.repeat(np.arange(n), ns)
        self.A_cell = self.P * self.dy
        self.xs, self.zg, TP = period_table(round(p.L, 4), round(p.d, 4), round(self.dy, 4), n,
                                            round(self.R0, 4), round(self.R1, 4), Xmax)
        self.TP = TP
        self.TPf = np.ascontiguousarray(TP.reshape(-1, 4))
        self.mmax = int(math.ceil(self.R1 / self.P)) + 1
        same = self.owner[:, None] == self.owner[None, :]
        self.eps_same = np.where(same, self.eps, self.eps_x)

    def points(self, phi_col):
        p = self.p
        X2, dX2 = shape_and_tangent(phi_col, p)
        X = np.zeros((self.n, p.N_seg, 3)); D = np.zeros_like(X)
        X[..., 0] = X2[..., 0] + self.bx[:, None]
        X[..., 2] = X2[..., 1]
        D[..., 0] = dX2[..., 0]; D[..., 2] = dX2[..., 1]
        # reduced (x,z) tangent matrix (2K, n)
        Tm = np.zeros((2 * self.K, self.n))
        n2 = 2 * p.N_seg
        Dr = D[..., [0, 2]].reshape(self.n, -1)
        for f in range(self.n):
            Tm[f * n2:(f + 1) * n2, f] = Dr[f]
        return X.reshape(-1, 3), Tm

    def mobility(self, X):
        """(2K,2K) mobility in x–z for the infinite lattice."""
        K = self.K
        x, z = X[:, 0], X[:, 2]
        G = np.zeros((K, K, 2, 2))
        # --- near part of the own row (direct, regularized; only pairs with w > 0) ---
        Gf = G.reshape(K * K, 4)
        for mx in range(-self.mmax, self.mmax + 1):
            dxm = (x[:, None] - x[None, :] - mx * self.P).ravel()
            sel = np.nonzero(np.abs(dxm) < self.R1)[0]
            if sel.size == 0:
                continue
            i, k = np.divmod(sel, K)
            w = _smooth_w(np.abs(dxm[sel]), self.R0, self.R1)
            eps = self.eps_same.ravel()[sel] if mx == 0 else self.eps_x
            g = blake_xz_pairs(dxm[sel], z[i], z[k], eps)
            for c_, gc in enumerate(g):
                Gf[sel, c_] += w * gc
        # --- everything else from the period table (trilinear interpolation) ---
        dx = np.mod(x[:, None] - x[None, :], self.P)
        hx = self.xs[1] - self.xs[0]
        u = dx / hx
        ix = np.floor(u).astype(int); tx = u - ix
        ix = ix % self.xs.size
        ix1 = (ix + 1) % self.xs.size
        hz = self.zg[1] - self.zg[0]
        zc = np.clip(z, 0, self.zg[-1] - 1e-9)
        iz = np.minimum((zc / hz).astype(int), self.zg.size - 2); tz = zc / hz - iz
        ia, ta = iz[:, None], tz[:, None]
        ib, tb = iz[None, :], tz[None, :]
        TPf = self.TPf
        nz = self.zg.size
        acc = np.zeros((K, K, 4))
        for cx, wx in ((ix, 1 - tx), (ix1, tx)):
            bx_ = cx * (nz * nz)
            for ca, wa in ((ia, 1 - ta), (ia + 1, ta)):
                ba_ = bx_ + ca * nz
                wxa = wx * wa
                for cb, wb in ((ib, 1 - tb), (ib + 1, tb)):
                    acc += (wxa * wb)[..., None] * np.take(TPf, ba_ + cb, axis=0)
        G[..., 0, 0] += acc[..., 0]; G[..., 0, 1] += acc[..., 1]
        G[..., 1, 0] += acc[..., 2]; G[..., 1, 1] += acc[..., 3]
        return G.transpose(0, 2, 1, 3).reshape(2 * K, 2 * K) / (8 * np.pi * self.p.eta_int)

    def solve(self, phi_col, omega_fixed=None, omega0=None, T_stall=None):
        p = self.p
        X, Tm = self.points(phi_col)
        M = self.mobility(X)
        Y = np.linalg.solve(M, Tm)
        Gam = Tm.T @ Y
        if p.beat_mode == "fixed":
            phidot = np.full(self.n, omega_fixed)
        else:
            k = omega0 / T_stall
            phidot = np.linalg.solve(np.eye(self.n) + k * Gam, np.full(self.n, omega0))
        Fr = (Y @ phidot).reshape(-1, 2)
        F = np.zeros((self.K, 3)); F[:, 0] = Fr[:, 0]; F[:, 2] = Fr[:, 1]
        Q = Tm.T @ Fr.ravel()
        return dict(phidot=phidot, F=F, X=X, Q=Q, Gam=Gam)


# -----------------------------------------------------------------------------
def run_carpet(p: Params, n: int, phi0=None, dy=None, verbose=False, store_every: int = 0, **kw):
    """Time integration on the periodic carpet.  p.beat_mode: 'fixed' | 'load'.
    phi0: initial phases of the n columns (default from p.init_phase)."""
    c = Carpet(p, n, dy=dy, **kw)
    if p.beat_mode == "load":
        omega0, T_stall = motor_params(p)
        G1 = resistance_table_single(p, p.eta)[1].mean()
        f_est = omega0 / (TWO_PI * (1 + omega0 * G1 / T_stall))
    else:
        omega0, T_stall, f_est = TWO_PI * p.f_fixed, None, p.f_fixed
    period = 1 / f_est
    dt = p.dt if p.dt is not None else period / p.steps_per_cycle
    nst = int(math.ceil(p.n_cycles * period / dt))
    if phi0 is None:
        if p.init_phase == "inphase":
            phi0 = np.zeros(n)
        elif p.init_phase == "metachronal":
            phi0 = np.mod(np.arange(n) * p.dphi_mc, TWO_PI)
        else:
            phi0 = np.random.default_rng(p.seed).uniform(0, TWO_PI, n)
    phi = np.array(phi0, float)
    ns = p.N_seg
    rec = dict(t=np.arange(nst) * dt, phi=np.zeros((nst, n)), phidot=np.zeros((nst, n)),
               Fx=np.zeros((nst, c.nf)), Fz=np.zeros((nst, c.nf)), Q=np.zeros((nst, n)),
               Mz=np.zeros(nst))
    frames = []
    for s in range(nst):
        r = c.solve(phi, omega_fixed=TWO_PI * p.f_fixed, omega0=omega0, T_stall=T_stall)
        Fk = r["F"].reshape(c.nf, ns, 3)
        rec["phi"][s] = phi; rec["phidot"][s] = r["phidot"]
        rec["Fx"][s] = Fk[..., 0].sum(1); rec["Fz"][s] = Fk[..., 2].sum(1)
        rec["Q"][s] = r["Q"]
        rec["Mz"][s] = np.sum(r["F"][:, 0] * r["X"][:, 2])          # Σ F_x z
        if store_every and s % store_every == 0:
            frames.append((s * dt, phi.copy(), r["X"], r["F"]))
        if p.beat_mode == "fixed" or p.integrator == "euler":
            phi = phi + dt * r["phidot"]
        else:
            k1 = r["phidot"]
            k2 = c.solve(phi + dt * k1, omega0=omega0, T_stall=T_stall)["phidot"]
            phi = phi + 0.5 * dt * (k1 + k2)
        if verbose and s % max(1, nst // 10) == 0:
            print(f"  {s}/{nst}  f={r['phidot'].mean()/TWO_PI:.2f} Hz")
    pm = phase_map(p, omega0, T_stall) if p.beat_mode == "load" else None
    return dict(params=p.as_dict(), n=n, dy=c.dy, P=c.P, A_cell=c.A_cell, nf=c.nf, npc=c.npc,
                col=c.col, dt=dt, omega0=omega0, T_stall=T_stall, phase_map=pm, frames=frames,
                eps=c.eps, **rec)


def analyze_carpet(res, transient=None):
    p = Params(**res["params"])
    tr = p.transient if transient is None else transient
    n0 = int(tr * len(res["t"]))
    phi = res["phi"][n0:]
    col = res["col"]
    eff = in_effective(phi[:, col], p)                         # (T, nf)
    Fx = res["Fx"][n0:]
    f_eff = np.array([Fx[eff[:, f], f].mean() for f in range(res["nf"])])
    eta_int = p.eta * ETA_UNIT
    Mz = res["Mz"][n0:].mean()
    U_slip = Mz / (eta_int * res["A_cell"])
    power = (res["Q"][n0:] * res["phidot"][n0:]).sum(1).mean()      # per periodic cell
    out = dict(f_eff=float(f_eff.mean()),                           # effective-stroke mean Fx / flagellum
               F_net=float(Fx.mean()),                              # cycle-mean Fx / flagellum
               U_slip=float(U_slip),                                # µm/s
               power_per_flag=float(power / res["nf"]),             # pN·µm/s
               power_per_area=float(power / res["A_cell"]),
               freq=float(res["phidot"][n0:].mean() / TWO_PI),
               h_eff=float(Mz / Fx.sum(1).mean()) if abs(Fx.sum(1).mean()) > 1e-9 else np.nan)
    th = to_theta(phi, res.get("phase_map"))
    out["R_sync"] = float(np.abs(np.exp(1j * th).mean(1)).mean())
    # neighbour lags incl. the periodic wrap (θ_{j+1} − θ_j, j = n-1 → 0)
    dth = wrap(np.diff(np.c_[th, th[:, :1]], axis=1))
    z = np.exp(1j * dth)
    out["dphi_mean"] = float(np.angle(z.mean()))
    out["dphi_coh"] = float(np.abs(z.mean()))
    out["dphi_lock"] = float(np.abs(z.mean(0)).mean())
    # winding number = total phase advance around the period / 2π  (→ wavelength n d / |w|)
    out["winding"] = float(np.round(dth.sum(1) / TWO_PI).mean())
    return out
