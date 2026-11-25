"""
sweep.py — parameter sweep over d/L (L fixed, d varied) × η, and the summary plots.

Examples
--------
  python sweep.py                                   # load / anchored, N=10, 9 × 3 conditions
  python sweep.py --fast --workers 4                # coarse numerics
  python sweep.py --beat_mode fixed --init_phase metachronal
  python sweep.py --wall_mode swimming --R_body 200
  python sweep.py --replot sweep_results/load_anchored_N10     # only redraw figures

For every η the single-flagellum reference f_single (N=1, same conditions) is run
once, and  ε_sim = F_total / (N f_single).

Outputs (in --out):
  results.json          all scalar results
  sweep_summary.png     f, F_total, ε_sim, R_sync, (U or frequency), locking vs d/L
  sweep_dphi.png        distribution of neighbour phase differences vs d/L for each η
"""
from __future__ import annotations

import os
# one BLAS thread per worker process (set before numpy is imported)
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

from model import Params, run, analyze, VOLVOX, PLEODORINA
from simulate import add_param_args, params_from_args

DL_VOLVOX = VOLVOX["d"] / VOLVOX["L"]            # 0.409
DL_PLEO = PLEODORINA["d"] / PLEODORINA["L"]      # 1.053
HIST_BINS = np.linspace(-np.pi, np.pi, 37)


def _job(task):
    kind, pdict = task
    p = Params(**pdict)
    t0 = time.time()
    res = run(p, verbose=False, store_frames=False)
    an = analyze(res)
    hist = (np.histogram(an["dphi"].ravel(), bins=HIST_BINS, density=True)[0].tolist()
            if p.N > 1 else None)
    out = dict(kind=kind, d=p.d, L=p.L, dL=p.d / p.L, eta=p.eta, N=p.N,
               f_i=an["f_i"].tolist(), f_mean=an["f_mean"], F_total=an["F_total"],
               Fx_cycle_mean=an["Fx_cycle_mean"], freq=an["freq"], U=an["U_mean"],
               power=an["power"], R_sync=an["R_sync"],
               dphi_mean=an["dphi_mean"], dphi_coherence=an["dphi_coherence"],
               dphi_lock=an["dphi_lock"], wave_kind=an["wave_kind"], dphi_hist=hist,
               elapsed=time.time() - t0)
    print(f"  done {kind:6s} d/L={out['dL']:.2f} η={p.eta:<5g} f={out['f_mean']:7.2f} pN "
          f"R={out['R_sync']:.2f} freq={out['freq']:.1f} Hz  ({out['elapsed']:.0f} s)", flush=True)
    return out


def run_sweep(base: Params, dLs, etas, workers: int):
    tasks = []
    for eta in etas:
        tasks.append(("single", base.copy(N=1, eta=eta, init_phase="inphase").as_dict()))
        for dL in dLs:
            tasks.append(("array", base.copy(eta=eta, d=float(dL * base.L)).as_dict()))
    # longest jobs (array, high η has the same #steps) first
    tasks.sort(key=lambda t: t[0] == "single")
    print(f"{len(tasks)} runs on {workers} workers ...")
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_job, tasks, chunksize=1)
    else:
        results = [_job(t) for t in tasks]
    single = {r["eta"]: r for r in results if r["kind"] == "single"}
    arr = [r for r in results if r["kind"] == "array"]
    for r in arr:
        f1 = single[r["eta"]]["f_mean"]
        r["f_single"] = f1
        r["eps_sim"] = r["F_total"] / (r["N"] * f1)
        r["U_single"] = single[r["eta"]]["U"]
        r["freq_single"] = single[r["eta"]]["freq"]
    arr.sort(key=lambda r: (r["eta"], r["dL"]))
    return arr, list(single.values())


# -----------------------------------------------------------------------------
def species_lines(ax):
    ax.axvline(DL_VOLVOX, color="tab:green", ls="--", lw=1)
    ax.axvline(DL_PLEO, color="tab:purple", ls="--", lw=1)


def plot_summary(arr, params, path):
    etas = sorted({r["eta"] for r in arr})
    cmap = plt.get_cmap("plasma")
    colors = {e: cmap(0.1 + 0.75 * i / max(1, len(etas) - 1)) for i, e in enumerate(etas)}
    swim = params["wall_mode"] == "swimming"
    load = params["beat_mode"] == "load"
    panels = [
        ("f_mean", "mean f_i (effective stroke) [pN]"),
        ("F_total", "F_total = Σ f_i [pN]"),
        ("eps_sim", "ε_sim = F_total / (N f_single)"),
        ("R_sync", "Kuramoto R_sync"),
        ("U" if swim else "freq", "swimming speed U [µm/s]" if swim else "beat frequency [Hz]"),
        ("dphi_lock", "neighbour pair locking  |⟨e^{iΔθ}⟩_t|"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, lab) in zip(axes.ravel(), panels):
        for e in etas:
            rr = [r for r in arr if r["eta"] == e]
            x = [r["dL"] for r in rr]
            y = [r[key] for r in rr]
            ax.plot(x, y, "o-", color=colors[e], label=f"η = {e:g} mPa·s")
            if key == "f_mean":
                ax.axhline(rr[0]["f_single"], color=colors[e], ls=":", lw=1)
            if key == "freq":
                ax.axhline(rr[0]["freq_single"], color=colors[e], ls=":", lw=1)
        if key == "eps_sim":
            ax.axhline(1, color="0.5", lw=0.8)
        if key in ("R_sync", "dphi_lock"):
            ax.set_ylim(0, 1.02)
        species_lines(ax)
        ax.set_xlabel("d / L"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    ax0 = axes[0, 0]
    ymax = ax0.get_ylim()[1]
    ax0.text(DL_VOLVOX, ymax, " Volvox", color="tab:green", va="top", fontsize=8)
    ax0.text(DL_PLEO, ymax, " Pleodorina", color="tab:purple", va="top", fontsize=8)
    if not load:
        axes[1, 0].set_title("(fixed mode: phases do not evolve)", fontsize=8)
    fig.suptitle(f"N={params['N']}, L={params['L']} µm, beat={params['beat_mode']}, "
                 f"wall={params['wall_mode']}, init={params['init_phase']}  "
                 "(dotted: single flagellum)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_dphi(arr, params, path):
    etas = sorted({r["eta"] for r in arr})
    if arr[0]["dphi_hist"] is None:
        return
    fig, axes = plt.subplots(1, len(etas), figsize=(4.6 * len(etas), 4), squeeze=False)
    centers = np.degrees(0.5 * (HIST_BINS[1:] + HIST_BINS[:-1]))
    for ax, e in zip(axes[0], etas):
        rr = [r for r in arr if r["eta"] == e]
        dl = np.array([r["dL"] for r in rr])
        H = np.array([r["dphi_hist"] for r in rr])
        # cell edges in d/L
        edges = np.r_[dl[0] - (dl[1] - dl[0]) / 2, 0.5 * (dl[1:] + dl[:-1]),
                      dl[-1] + (dl[-1] - dl[-2]) / 2] if len(dl) > 1 else np.r_[dl - .05, dl + .05]
        im = ax.pcolormesh(edges, np.degrees(HIST_BINS), H.T, cmap="viridis", shading="flat")
        ax.plot(dl, np.degrees([r["dphi_mean"] for r in rr]), "w.", ms=6)
        species_lines(ax)
        ax.set_title(f"η = {e:g} mPa·s"); ax.set_xlabel("d / L")
        ax.set_ylabel("Δθ = θ_{i+1} − θ_i [deg]  (<0: symplectic)")
        fig.colorbar(im, ax=ax, label="pdf")
    fig.suptitle("distribution of neighbour phase differences (white dots: circular mean)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_param_args(ap)
    ap.add_argument("--dL_min", type=float, default=0.3)
    ap.add_argument("--dL_max", type=float, default=1.5)
    ap.add_argument("--n_dL", type=int, default=9)
    ap.add_argument("--dL_list", type=float, nargs="*", default=None)
    ap.add_argument("--etas", type=float, nargs="*", default=[0.84, 2.5, 8.61])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replot", default=None, help="directory with results.json")
    args = ap.parse_args()

    if args.replot:
        with open(os.path.join(args.replot, "results.json")) as fh:
            data = json.load(fh)
        plot_summary(data["array"], data["params"], os.path.join(args.replot, "sweep_summary.png"))
        plot_dphi(data["array"], data["params"], os.path.join(args.replot, "sweep_dphi.png"))
        return

    if args.beat_mode is None:
        args.beat_mode = "load"
    base = params_from_args(args)
    dLs = args.dL_list or np.linspace(args.dL_min, args.dL_max, args.n_dL).tolist()
    out = args.out or os.path.join("sweep_results", f"{base.beat_mode}_{base.wall_mode}_N{base.N}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    arr, single = run_sweep(base, dLs, args.etas, args.workers)
    print(f"sweep finished in {(time.time()-t0)/60:.1f} min")
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(dict(params=base.as_dict(), dLs=dLs, etas=args.etas,
                       array=arr, single=single), fh, indent=1)
    print(f"\n{'η':>6s} {'d/L':>5s} {'f_mean':>8s} {'F_total':>9s} {'ε_sim':>6s} {'R':>5s} "
          f"{'lock':>5s} {'freq':>6s} {'U':>8s}  pattern")
    for r in arr:
        print(f"{r['eta']:6g} {r['dL']:5.2f} {r['f_mean']:8.2f} {r['F_total']:9.1f} "
              f"{r['eps_sim']:6.3f} {r['R_sync']:5.2f} {r['dphi_lock']:5.2f} {r['freq']:6.2f} "
              f"{r['U']:8.3f}  {r['wave_kind']}")
    plot_summary(arr, base.as_dict(), os.path.join(out, "sweep_summary.png"))
    plot_dphi(arr, base.as_dict(), os.path.join(out, "sweep_dphi.png"))
    print("figures →", out)


if __name__ == "__main__":
    main()
