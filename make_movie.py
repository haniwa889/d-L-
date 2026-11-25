"""
make_movie.py — side-by-side movie: {Volvox d/L, Pleodorina d/L} × {in-phase, metachronal}
(fixed kinematics, same L; flow speed colour map + velocity arrows; ε_sim in each title)

  python make_movie.py                       # → movies/compare_dL.mp4 (+ a still png)
  python make_movie.py --dphi -90 --cycles 3
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from model import Params, run, analyze, velocity_field, shape_full, in_effective, VOLVOX, PLEODORINA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=22.0)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--N_seg", type=int, default=16)
    ap.add_argument("--dphi", type=float, default=-90.0, help="metachronal lag [deg]")
    ap.add_argument("--cycles", type=float, default=2.0)
    ap.add_argument("--fpc", type=int, default=36, help="frames per cycle")
    ap.add_argument("--slow", type=float, default=60.0, help="slow-motion factor for playback")
    ap.add_argument("--out", default="movies")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base = Params(N=a.N, L=a.L, N_seg=a.N_seg, beat_mode="fixed", n_cycles=a.cycles,
                  transient=0.0, frame_cycles=a.cycles, frames_per_cycle=a.fpc,
                  steps_per_cycle=a.fpc)
    f1 = analyze(run(base.copy(N=1, init_phase="inphase"), verbose=False))["f_mean"]
    rows = [("Volvox", VOLVOX["d"] / VOLVOX["L"]), ("Pleodorina", PLEODORINA["d"] / PLEODORINA["L"])]
    cols = [("in-phase (Δφ=0°)", dict(init_phase="inphase")),
            (f"metachronal (Δφ={a.dphi:+.0f}°)", dict(init_phase="metachronal",
                                                       dphi_mc=np.radians(a.dphi)))]
    panels = []
    for rname, dL in rows:
        for cname, kw in cols:
            p = base.copy(d=dL * a.L, **kw)
            res = run(p, verbose=False)
            an = analyze(res)
            eps = an["f_i"][2:-2].mean() / f1
            # same window width for every panel (that of the Volvox row) → same scale
            W = (a.N - 1) * VOLVOX["d"] / VOLVOX["L"] * a.L + 2.1 * a.L
            xc = 0.5 * (a.N - 1) * p.d
            xs = np.linspace(xc - W / 2, xc + W / 2, 64)
            zs = np.linspace(0.3, 1.2 * a.L, 22)
            XX, ZZ = np.meshgrid(xs, zs)
            pts = np.c_[XX.ravel(), np.zeros(XX.size), ZZ.ravel()]
            fr = res["frames"]
            U = []
            for X, F in zip(fr["X"], fr["F"]):
                u = velocity_field(pts, X, F, res["eps"], p.eta_int)
                U.append((u[:, 0].reshape(XX.shape), u[:, 2].reshape(XX.shape)))
            panels.append(dict(p=p, res=res, fr=fr, XX=XX, ZZ=ZZ, U=U, eps=eps,
                               title=f"{rname}  d/L={dL:.2f}  |  {cname}"))
            print(f"{rname:10s} {cname:24s} ε_sim={eps:.2f}")

    vmax = np.percentile(np.concatenate([np.hypot(*u).ravel() for P in panels for u in P["U"]]), 97)
    nfr = min(len(P["fr"]["t"]) for P in panels)
    w = [P["XX"].max() - P["XX"].min() for P in panels]
    fig, axes = plt.subplots(2, 2, figsize=(15, 5.6),
                             gridspec_kw=dict(width_ratios=[1, 1], height_ratios=[1, 1]))
    axes = axes.ravel()
    quivers = []

    def draw(k):
        for ax, P in zip(axes, panels):
            ax.clear()
            ux, uz = P["U"][k]
            sp = np.hypot(ux, uz)
            ax.pcolormesh(P["XX"], P["ZZ"], sp, cmap="magma", vmin=0, vmax=vmax, shading="gouraud")
            ax.quiver(P["XX"][::2, ::2], P["ZZ"][::2, ::2], ux[::2, ::2], uz[::2, ::2], color="w",
                      alpha=0.6, scale=vmax * 22, width=0.0025)
            p = P["p"]; phi = P["fr"]["phi"][k]
            sh = shape_full(phi, p)
            eff = in_effective(phi, p)
            for i in range(p.N):
                ax.plot(i * p.d + sh[i, :, 0], sh[i, :, 1], color="cyan" if eff[i] else "lime", lw=2.2)
            ax.axhline(0, color="0.55", lw=5)
            ax.set_xlim(P["XX"].min(), P["XX"].max()); ax.set_ylim(-0.8, P["ZZ"].max())
            ax.set_aspect("equal"); ax.set_yticks([0, 10, 20])
            ax.set_title(P["title"] + f"\nthrust per flagellum ε_sim = {P['eps']:.2f}  (×single)",
                         fontsize=10)
            ax.set_xlabel("x [µm]", fontsize=8); ax.tick_params(labelsize=7)
        fig.suptitle(f"L={a.L:g} µm, {a.N} flagella, fixed beat 30 Hz (shown {a.slow:g}× slower).  "
                     f"\nsame scale in all panels; cyan: effective stroke, green: recovery stroke, colour: flow speed (max ≈ {vmax:.0f} µm/s)",
                     fontsize=10)
        return []

    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.4)
    draw(0)
    fig.savefig(os.path.join(a.out, "compare_dL_still.png"), dpi=120)
    anim = animation.FuncAnimation(fig, draw, frames=nfr, blit=False)
    fps = max(1, round(base.f_fixed * a.fpc / a.slow))
    if shutil.which("ffmpeg"):
        path = os.path.join(a.out, "compare_dL.mp4")
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000), dpi=110)
    else:
        path = os.path.join(a.out, "compare_dL.gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=70)
    print("saved", path)


if __name__ == "__main__":
    main()
