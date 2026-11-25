"""
dl_interaction.py — "How strongly do neighbouring flagella interact, as a function of d/L?"

Uses beat_mode="fixed" (prescribed kinematics, every flagellum at the same
frequency), so the motion is exactly periodic and ONE cycle suffices per condition.

  (1) ε_sim(d/L, Δφ): thrust of a metachronal row (constant neighbour phase lag Δφ)
      relative to N independent flagella.  Δφ = 0 is in-phase beating.
      → how much the phase arrangement matters at a given spacing.
  (2) hydrodynamic coupling coefficient between two flagella
      κ(d/L) = ⟨|Γ_12|⟩ / ⟨Γ_11⟩  (phase-averaged; Γ = generalized resistance matrix):
      the fraction of a flagellum's load that is caused by its neighbour's motion.

In "fixed" mode forces ∝ η, so ε_sim and κ do not depend on η.

  python dl_interaction.py                 # default grid, 4 workers
  python dl_interaction.py --fast          # N_seg=12, coarser grid
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Params, run, analyze, System, TWO_PI, VOLVOX, PLEODORINA

DL_V = VOLVOX["d"] / VOLVOX["L"]
DL_P = PLEODORINA["d"] / PLEODORINA["L"]


def base_params(a) -> Params:
    return Params(N=a.N, L=a.L, N_seg=a.N_seg, beat_mode="fixed", wall_mode="anchored",
                  steps_per_cycle=a.steps, n_cycles=1.0, transient=0.0)


def job_eps(args):
    pdict, dL, dphi, n_edge = args
    p = Params(**pdict).copy(d=dL * pdict["L"], init_phase="metachronal", dphi_mc=dphi)
    an = analyze(run(p, verbose=False, store_frames=False))
    f = an["f_i"]
    inner = f[n_edge:p.N - n_edge] if p.N > 2 * n_edge else f
    return dict(dL=dL, dphi=dphi, f_all=float(f.mean()), f_inner=float(inner.mean()),
                net_inner=float(an["Fx_cycle_mean"] / p.N))


def single_ref(pdict):
    p = Params(**pdict).copy(N=1, init_phase="inphase")
    an = analyze(run(p, verbose=False, store_frames=False))
    return float(an["f_mean"]), float(an["Fx_cycle_mean"])


def job_kappa(args):
    pdict, dL, n_phi = args
    p = Params(**pdict).copy(N=2, d=dL * pdict["L"])
    s = System(p)
    g = np.linspace(0, TWO_PI, n_phi, endpoint=False)
    G12, G11 = [], []
    for a in g:
        for b in g:
            Gm = s.solve(np.array([a, b]), omega_fixed=1.0)["Gam"]
            G12.append(0.5 * (Gm[0, 1] + Gm[1, 0])); G11.append(Gm[0, 0])
    G12, G11 = np.array(G12), np.array(G11)
    return dict(dL=dL, kappa=float(np.abs(G12).mean() / G11.mean()),
                kappa_signed=float(G12.mean() / G11.mean()))


def plot(data, path):
    dLs = np.array(data["dLs"]); dphis = np.array(data["dphis"])
    E = np.array(data["eps_inner"])            # (n_dL, n_dphi)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    # (a) heat map
    dd = np.degrees(dphis)
    im = ax[0].pcolormesh(dLs, dd, E.T, cmap="RdBu_r", shading="nearest",
                          vmin=2 - E.max(), vmax=E.max())
    ax[0].contour(dLs, dd, E.T, levels=[1.0], colors="k", linewidths=0.8)
    fig.colorbar(im, ax=ax[0], label="ε_sim (inner flagella)")
    ax[0].set_xlabel("d / L"); ax[0].set_ylabel("neighbour phase lag Δφ [deg]")
    ax[0].set_title("(a) thrust relative to independent flagella\n"
                    "Δφ<0: wave along effective stroke (symplectic)", fontsize=10)

    # (b) best / in-phase / worst
    i0 = int(np.argmin(np.abs(dphis)))
    ax[1].plot(dLs, E.max(1), "o-", color="tab:red", label="best Δφ")
    ax[1].plot(dLs, E[:, i0], "s-", color="tab:blue", label="in-phase (Δφ=0)")
    ax[1].plot(dLs, E.min(1), "v-", color="tab:gray", label="worst Δφ")
    ax[1].fill_between(dLs, E.min(1), E.max(1), color="tab:orange", alpha=0.15,
                       label="range set by phase arrangement")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_xlabel("d / L"); ax[1].set_ylabel("ε_sim")
    ax[1].set_title("(b) how much coordination matters", fontsize=10)
    ax[1].legend(fontsize=8)
    ax2 = ax[1].twinx()
    best = dd[np.argmax(E, axis=1)]
    ax2.plot(dLs, best, "k.", ms=5); ax2.set_ylabel("best Δφ [deg] (black dots)")
    ax2.set_ylim(-190, 190)

    # (c) coupling coefficient
    K = data["kappa"]
    ax[2].semilogy([k["dL"] for k in K], [k["kappa"] for k in K], "o-", color="tab:green")
    ax[2].set_xlabel("d / L"); ax[2].set_ylabel("κ = ⟨|Γ12|⟩ / ⟨Γ11⟩")
    ax[2].set_title("(c) hydrodynamic coupling between two neighbours", fontsize=10)
    ax[2].grid(alpha=0.3, which="both")

    for a in ax:
        a.axvline(DL_V, color="tab:green", ls="--", lw=1)
        a.axvline(DL_P, color="tab:purple", ls="--", lw=1)
    ax[1].text(DL_V, ax[1].get_ylim()[1], " Volvox", color="tab:green", va="top", fontsize=8)
    ax[1].text(DL_P, ax[1].get_ylim()[1], " Pleodorina", color="tab:purple", va="top", fontsize=8)
    fig.suptitle(f"fixed kinematics, N={data['params']['N']}, L={data['params']['L']} µm "
                 "(η-independent in this mode)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--L", type=float, default=22.0)
    ap.add_argument("--N_seg", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--n_dL", type=int, default=13)
    ap.add_argument("--n_dphi", type=int, default=16)
    ap.add_argument("--n_edge", type=int, default=2, help="flagella excluded at each end")
    ap.add_argument("--n_phi_kappa", type=int, default=16)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="dl_results")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "dl_interaction.json")
    if a.replot:
        with open(jpath) as fh:
            plot(json.load(fh), os.path.join(a.out, "dl_interaction.png"))
        return
    if a.fast:
        a.N_seg, a.steps, a.n_dL, a.n_dphi = 12, 40, 9, 12
    p = base_params(a)
    dLs = np.linspace(0.3, 1.5, a.n_dL)
    dphis = np.linspace(-np.pi, np.pi, a.n_dphi, endpoint=False)
    t0 = time.time()
    f1, net1 = single_ref(p.as_dict())
    tasks = [(p.as_dict(), float(x), float(y), a.n_edge) for x in dLs for y in dphis]
    ktasks = [(p.as_dict(), float(x), a.n_phi_kappa) for x in dLs]
    with Pool(a.workers) as pool:
        res = pool.map(job_eps, tasks, chunksize=2)
        kres = pool.map(job_kappa, ktasks)
    E = np.array([r["f_inner"] / f1 for r in res]).reshape(len(dLs), len(dphis))
    Eall = np.array([r["f_all"] / f1 for r in res]).reshape(len(dLs), len(dphis))
    data = dict(params=p.as_dict(), dLs=dLs.tolist(), dphis=dphis.tolist(), f_single=f1,
                eps_inner=E.tolist(), eps_all=Eall.tolist(),
                net_inner=[r["net_inner"] for r in res], net_single=net1, kappa=kres)
    with open(jpath, "w") as fh:
        json.dump(data, fh, indent=1)
    plot(data, os.path.join(a.out, "dl_interaction.png"))
    print(f"done in {(time.time()-t0)/60:.1f} min,  f_single = {f1:.2f} pN")
    i0 = int(np.argmin(np.abs(dphis)))
    print(f"{'d/L':>5s} {'in-phase':>9s} {'best':>6s} {'bestΔφ':>7s} {'worst':>6s} {'range':>6s} {'κ':>7s}")
    for k, x in enumerate(dLs):
        print(f"{x:5.2f} {E[k,i0]:9.3f} {E[k].max():6.3f} {np.degrees(dphis[E[k].argmax()]):7.1f} "
              f"{E[k].min():6.3f} {E[k].max()-E[k].min():6.3f} {kres[k]['kappa']:7.4f}")


if __name__ == "__main__":
    main()
