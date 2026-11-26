"""
carpet2d.py — doubly periodic carpet with INDEPENDENT phases in both directions.

Periodic cell: nx columns (spacing d, along the beat) × ny lanes (spacing dy, lateral).
Every cell (ix, k) carries its own phase.  Optionally each cell has two flagella
(biflagellate somatic cell) side by side at y and y + pair_sep that beat together.
This allows a genuinely uncoordinated (random 2-D phases) reference state, which the
lane-synchronous `carpet.Carpet` cannot represent.

Hydrodynamics as in carpet.py (regularized Stokeslet + Blake image, lattice sums to
±Xmax with a smeared tail).  Pairs of points closer than R1 (horizontal distance) are
evaluated directly with a smooth partition of unity; everything else comes from
precomputed lattice tables, one per lateral offset class.

Approximation: only x–z force components are solved for (F_y = 0).  This is exact when
the lateral arrangement is mirror-symmetric (e.g. lanes in phase); otherwise the small
lateral flows are not constrained.
"""
from __future__ import annotations

import math
import numpy as np

from model import (Params, kernel, get_eps, shape_and_tangent, in_effective, motor_params,
                   resistance_table_single, phase_map, to_theta, wrap, TWO_PI, ETA_UNIT)
from carpet import _smooth_w, _disk_cached, tail_table


def blake_xz_pairs_y(dx, dy, zf, zs, eps):
    """x–z block of the regularized Blake kernel for pairs with lateral offset dy."""
    e2 = eps * eps
    r3 = zf - zs
    R3 = zf + zs
    h2 = dx * dx + dy * dy
    r2 = h2 + r3 * r3
    R2 = h2 + R3 * R3
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


@_disk_cached("line2d")
def line_table_class(L: float, y0: float, Py: float, R0: float, R1: float, Xmax: float,
                     nz: int = 24):
    """S(Δx,z,z') = Σ_m (1 − w(r_h)) G(Δx, y0 + m Py, z; z'),  r_h = sqrt(Δx² + y²),
    over |y0 + m Py| ≤ Xmax.  Δx ≥ 0 grid (xx, zz even; xz, zx odd in Δx)."""
    zg = np.linspace(0.0, 1.05 * L, nz)
    xg = np.unique(np.r_[np.arange(0, 60, 0.5), np.arange(60, 300, 2.0),
                         np.arange(300, Xmax + 10, 10.0)])
    mm = int(Xmax // Py) + 1
    ys = y0 + np.arange(-mm, mm + 1) * Py
    ys = ys[np.abs(ys) <= Xmax]
    DX, YY = np.meshgrid(xg, ys, indexing="ij")
    W = 1.0 - _smooth_w(np.hypot(DX, YY), R0, R1)
    out = np.zeros((xg.size, nz, nz, 4))
    for b, zs in enumerate(zg):
        zs_ = max(zs, 1e-3)
        for a, zf in enumerate(zg):
            g = blake_xz_pairs_y(DX, YY, zf, zs_, 1e-6)
            for c in range(4):
                out[:, a, b, c] = np.sum(W * g[c], axis=1)
    return xg, zg, out


@_disk_cached("period2d")
def period_table_class(L: float, d: float, n: int, y0: float, Py: float, R0: float,
                       R1: float, Xmax: float = 3000.0, dxs: float = 0.5):
    xg, zg, S = line_table_class(L, y0, Py, R0, R1, Xmax)
    P = n * d
    nxs = int(math.ceil(P / dxs))
    xs = np.arange(nxs) * (P / nxs)
    nz = zg.size
    TP = np.zeros((xs.size, nz, nz, 4))
    sign = np.array([1, -1, -1, 1.0])
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
        TP[ok] += val * (1 - sg) + val * sign * sg
    zt, Tt = tail_table(float(Xmax), round(1.05 * L, 3), nz=nz)
    TP[..., 0] += Tt[..., 0] / (P * Py)
    TP[..., 3] += Tt[..., 1] / (P * Py)
    return xs, zg, TP


class Carpet2D:
    def __init__(self, p: Params, nx: int, ny: int, dy: float | None = None,
                 pair_sep: float = 0.0, R0: float | None = None, R1: float | None = None,
                 Xmax: float = 3000.0):
        self.p = p
        self.nx, self.ny = nx, ny
        self.d = p.d
        self.dy = dy if dy is not None else p.d
        self.P = nx * p.d
        self.Py = ny * self.dy
        self.eps = get_eps(p)
        self.eps_x = max(p.eps_inter or self.eps, self.eps)
        self.R0 = R0 if R0 is not None else 1.0 * p.L
        self.R1 = R1 if R1 is not None else 2.0 * p.L
        self.npc = 2 if pair_sep > 0 else 1
        self.n_osc = nx * ny
        fx, fy, fo = [], [], []
        for ix in range(nx):
            for k in range(ny):
                for j in range(self.npc):
                    fx.append(ix * p.d); fy.append(k * self.dy + j * pair_sep); fo.append(ix * ny + k)
        self.fx, self.fy, self.fosc = map(np.array, (fx, fy, fo))
        self.nf = len(fx)
        ns = p.N_seg
        self.K = self.nf * ns
        self.owner = np.repeat(np.arange(self.nf), ns)
        self.A_cell = self.P * self.Py
        # lateral offset classes
        dyf = np.mod(self.fy[:, None] - self.fy[None, :], self.Py)
        dyf = np.where(np.isclose(dyf, self.Py), 0.0, dyf)
        keys = np.round(dyf, 6)
        uniq = sorted(set(keys.ravel().tolist()))
        self.classes = uniq
        cls_f = np.vectorize(lambda v: uniq.index(v))(keys)
        self.cls = np.repeat(np.repeat(cls_f, ns, 0), ns, 1)
        tabs = []
        for y0 in uniq:
            xs, zg, TP = period_table_class(round(p.L, 4), round(p.d, 4), nx, float(y0),
                                            round(self.Py, 4), round(self.R0, 4),
                                            round(self.R1, 4), Xmax)
            tabs.append(TP)
        self.xs, self.zg = xs, zg
        self.TPf = np.ascontiguousarray(np.stack(tabs).reshape(-1, 4))
        self.nxs = xs.size
        # near-field direct list: (class mask, lateral image offset)
        self.near = []
        for qi, y0 in enumerate(uniq):
            mm = int(self.R1 // self.Py) + 1
            for m in range(-mm, mm + 1):
                yi = y0 + m * self.Py
                if abs(yi) < self.R1:
                    self.near.append((self.cls == qi, yi))
        self.mmax = int(math.ceil(self.R1 / self.P)) + 1
        same = self.owner[:, None] == self.owner[None, :]
        self.same = same

    # -------------------------------------------------------------------------
    def points(self, phi_osc):
        p = self.p
        X2, dX2 = shape_and_tangent(phi_osc[self.fosc], p)
        X = np.zeros((self.nf, p.N_seg, 3)); D = np.zeros_like(X)
        X[..., 0] = X2[..., 0] + self.fx[:, None]
        X[..., 1] = self.fy[:, None]
        X[..., 2] = X2[..., 1]
        D[..., 0] = dX2[..., 0]; D[..., 2] = dX2[..., 1]
        Tm = np.zeros((2 * self.K, self.n_osc))
        n2 = 2 * p.N_seg
        Dr = D[..., [0, 2]].reshape(self.nf, -1)
        for f in range(self.nf):
            Tm[f * n2:(f + 1) * n2, self.fosc[f]] += Dr[f]
        return X.reshape(-1, 3), Tm

    def mobility(self, X):
        K = self.K
        x, z = X[:, 0], X[:, 2]
        G = np.zeros((K * K, 4))
        DXa = x[:, None] - x[None, :]
        for mask, yi in self.near:
            for mx in range(-self.mmax, self.mmax + 1):
                dxm = DXa - mx * self.P
                rh = np.sqrt(dxm * dxm + yi * yi)
                sel = np.nonzero((mask & (rh < self.R1)).ravel())[0]
                if sel.size == 0:
                    continue
                i, k = np.divmod(sel, K)
                w = _smooth_w(rh.ravel()[sel], self.R0, self.R1)
                selfpair = self.same.ravel()[sel] & (mx == 0) & (abs(yi) < 1e-9)
                eps = np.where(selfpair, self.eps, self.eps_x)
                g = blake_xz_pairs_y(dxm.ravel()[sel], yi, z[i], z[k], eps)
                for c, gc in enumerate(g):
                    G[sel, c] += w * gc
        # tables
        dx = np.mod(DXa, self.P)
        hx = self.P / self.nxs
        u = dx / hx
        ix = np.floor(u).astype(int); tx = u - ix
        ix = ix % self.nxs; ix1 = (ix + 1) % self.nxs
        nz = self.zg.size
        hz = self.zg[1] - self.zg[0]
        zc = np.clip(z, 0, self.zg[-1] - 1e-9)
        iz = np.minimum((zc / hz).astype(int), nz - 2); tz = zc / hz - iz
        ia, ta = iz[:, None], tz[:, None]
        ib, tb = iz[None, :], tz[None, :]
        base_q = self.cls * self.nxs
        acc = np.zeros((K, K, 4))
        for cx, wx in ((ix, 1 - tx), (ix1, tx)):
            bx_ = (base_q + cx) * (nz * nz)
            for ca, wa in ((ia, 1 - ta), (ia + 1, ta)):
                ba_ = bx_ + ca * nz
                wxa = wx * wa
                for cb, wb in ((ib, 1 - tb), (ib + 1, tb)):
                    acc += (wxa * wb)[..., None] * np.take(self.TPf, ba_ + cb, axis=0)
        G += acc.reshape(K * K, 4)
        G = G.reshape(K, K, 2, 2)
        return G.transpose(0, 2, 1, 3).reshape(2 * K, 2 * K) / (8 * np.pi * self.p.eta_int)

    def solve(self, phi_osc, omega_fixed=None, omega0=None, T_stall=None, noise=None):
        p = self.p
        X, Tm = self.points(phi_osc)
        M = self.mobility(X)
        Y = np.linalg.solve(M, Tm)
        Gam = Tm.T @ Y
        if p.beat_mode == "fixed":
            phidot = np.full(self.n_osc, omega_fixed)
        else:
            k = omega0 / T_stall
            phidot = np.linalg.solve(np.eye(self.n_osc) + k * Gam, np.full(self.n_osc, omega0))
        Fr = (Y @ phidot).reshape(-1, 2)
        F = np.zeros((self.K, 3)); F[:, 0] = Fr[:, 0]; F[:, 2] = Fr[:, 1]
        Q = Tm.T @ Fr.ravel()
        return dict(phidot=phidot, F=F, X=X, Q=Q, Gam=Gam)


def run_carpet2d(p: Params, nx: int, ny: int, phi0, dy=None, pair_sep=0.0, noise_D=0.0,
                 seed=0, verbose=False, **kw):
    """Time integration.  noise_D: phase diffusion constant [rad²/s] (Euler–Maruyama
    added after the deterministic Heun step; only in 'load' mode)."""
    c = Carpet2D(p, nx, ny, dy=dy, pair_sep=pair_sep, **kw)
    if p.beat_mode == "load":
        omega0, T_stall = motor_params(p)
        G1 = resistance_table_single(p, p.eta)[1].mean()
        f_est = omega0 / (TWO_PI * (1 + omega0 * G1 / T_stall))
    else:
        omega0, T_stall, f_est = TWO_PI * p.f_fixed, None, p.f_fixed
    period = 1 / f_est
    dt = p.dt if p.dt is not None else period / p.steps_per_cycle
    nst = int(math.ceil(p.n_cycles * period / dt))
    rng = np.random.default_rng(seed + 12345)
    phi = np.array(phi0, float)
    ns = p.N_seg
    rec = dict(t=np.arange(nst) * dt, phi=np.zeros((nst, c.n_osc)),
               phidot=np.zeros((nst, c.n_osc)), Fx=np.zeros((nst, c.nf)),
               Q=np.zeros((nst, c.n_osc)), Mz=np.zeros(nst))
    for s in range(nst):
        r = c.solve(phi, omega_fixed=TWO_PI * p.f_fixed, omega0=omega0, T_stall=T_stall)
        Fk = r["F"].reshape(c.nf, ns, 3)
        rec["phi"][s] = phi; rec["phidot"][s] = r["phidot"]
        rec["Fx"][s] = Fk[..., 0].sum(1); rec["Q"][s] = r["Q"]
        rec["Mz"][s] = np.sum(r["F"][:, 0] * r["X"][:, 2])
        if p.beat_mode == "fixed":
            phi = phi + dt * r["phidot"]
        else:
            k1 = r["phidot"]
            k2 = c.solve(phi + dt * k1, omega0=omega0, T_stall=T_stall)["phidot"]
            phi = phi + 0.5 * dt * (k1 + k2)
            if noise_D > 0:
                phi = phi + np.sqrt(2 * noise_D * dt) * rng.normal(size=c.n_osc)
        if verbose and s % max(1, nst // 10) == 0:
            print(f"  {s}/{nst}", flush=True)
    pm = phase_map(p, omega0, T_stall) if p.beat_mode == "load" else None
    return dict(params=p.as_dict(), nx=nx, ny=ny, P=c.P, Py=c.Py, A_cell=c.A_cell, nf=c.nf,
                n_osc=c.n_osc, npc=c.npc, fosc=c.fosc, dt=dt, omega0=omega0, T_stall=T_stall,
                phase_map=pm, **rec)


def analyze_carpet2d(res, transient=0.0):
    p = Params(**res["params"])
    n0 = int(transient * len(res["t"]))
    phi = res["phi"][n0:]
    eff = in_effective(phi[:, res["fosc"]], p)
    Fx = res["Fx"][n0:]
    f_eff = np.array([Fx[eff[:, f], f].mean() if eff[:, f].any() else np.nan
                      for f in range(res["nf"])])
    Mz = res["Mz"][n0:].mean()
    U_slip = Mz / (p.eta * ETA_UNIT * res["A_cell"])
    out = dict(f_eff=float(np.nanmean(f_eff)), F_net=float(Fx.mean()), U_slip=float(U_slip),
               Mz_per_flag=float(Mz / res["nf"]),
               power_per_flag=float((res["Q"][n0:] * res["phidot"][n0:]).sum(1).mean() / res["nf"]),
               freq=float(res["phidot"][n0:].mean() / TWO_PI))
    th = to_theta(phi, res.get("phase_map"))
    out["R_sync"] = float(np.abs(np.exp(1j * th).mean(1)).mean())
    nx, ny = res["nx"], res["ny"]
    T = th.reshape(len(th), nx, ny)
    zx = np.exp(1j * wrap(np.roll(T, -1, axis=1) - T))
    zy = np.exp(1j * wrap(np.roll(T, -1, axis=2) - T))
    out["dphi_x_mean"] = float(np.degrees(np.angle(zx.mean())))
    out["coh_x"] = float(np.abs(zx.mean()))
    out["coh_y"] = float(np.abs(zy.mean()))
    return out
