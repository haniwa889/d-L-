"""
simulate.py — run ONE condition, print the summary, save figures / animation.

Examples
--------
  python simulate.py --N 2 --beat_mode fixed                    # quick test
  python simulate.py --N 10 --beat_mode load --species volvox
  python simulate.py --N 10 --beat_mode load --wall_mode swimming --species pleodorina
  python simulate.py --rft_check                                # single flagellum vs RFT
  python simulate.py --N 10 --beat_mode fixed --init_phase metachronal --dphi_mc -0.785

Outputs go to  --out  (default results/<tag>/):
  summary.txt, timeseries.png, phase_diff.png, flow_field.png, flagella.mp4 (or .gif), run.npz
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import fields

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from model import (Params, run, analyze, rft_check, check_kernel, velocity_field,
                   shape_full, in_effective, to_theta, TWO_PI, VOLVOX, PLEODORINA, get_eps)


# -----------------------------------------------------------------------------
# argparse ↔ Params
# -----------------------------------------------------------------------------
def add_param_args(ap: argparse.ArgumentParser):
    for f in fields(Params):
        default = f.default
        typ = {int: int, float: float, str: str}.get(type(default), float)
        if default is None:
            typ = float
        ap.add_argument(f"--{f.name}", type=typ, default=None,
                        help=f"(default: {default})")
    ap.add_argument("--species", choices=["volvox", "pleodorina"], default=None,
                    help="preset d, L, R_body (overridden by explicit --d/--L/--R_body)")
    ap.add_argument("--dL", type=float, default=None, help="set d = dL × L")
    ap.add_argument("--fast", action="store_true",
                    help="coarse numerics: N_seg=12, steps_per_cycle=40")


def params_from_args(args) -> Params:
    kw = {}
    if args.species == "volvox":
        kw.update(VOLVOX)
    elif args.species == "pleodorina":
        kw.update(PLEODORINA)
    if args.fast:
        kw.update(N_seg=12, steps_per_cycle=40)
    for f in fields(Params):
        v = getattr(args, f.name, None)
        if v is not None:
            kw[f.name] = v
    p = Params(**kw)
    if args.dL is not None:
        p = p.copy(d=args.dL * p.L)
    return p


# -----------------------------------------------------------------------------
# Single-flagellum reference  f_single  (same conditions, N = 1)
# -----------------------------------------------------------------------------
def single_reference(p: Params, verbose=False) -> dict:
    q = p.copy(N=1, init_phase="inphase")
    res = run(q, verbose=verbose, store_frames=False)
    return analyze(res)


def summarize(p: Params, an: dict, an1: dict | None) -> str:
    lines = []
    lines.append(f"N={p.N}  d={p.d:.2f} µm  L={p.L:.2f} µm  d/L={p.d/p.L:.3f}  η={p.eta} mPa·s")
    lines.append(f"beat_mode={p.beat_mode}  wall_mode={p.wall_mode}  init_phase={p.init_phase}")
    lines.append(f"mean beat frequency          : {an['freq']:.2f} Hz")
    lines.append("f_i (effective-stroke mean Fx) [pN] : " + " ".join(f"{v:.2f}" for v in an["f_i"]))
    lines.append(f"mean f_i                     : {an['f_mean']:.3f} pN")
    lines.append(f"F_total = Σ f_i              : {an['F_total']:.3f} pN")
    lines.append(f"<Σ Fx> over full cycle       : {an['Fx_cycle_mean']:.3f} pN")
    lines.append(f"dissipated power             : {an['power']*1e-6:.4f} pW (= pN·µm/s ×1e-6)")
    if p.wall_mode == "swimming":
        lines.append(f"swimming speed U             : {an['U_mean']:.3f} µm/s")
    if an1 is not None:
        f1 = an1["f_mean"]
        lines.append(f"f_single (N=1)               : {f1:.3f} pN   (freq {an1['freq']:.2f} Hz)")
        lines.append(f"ε_sim = F_total/(N f_single) : {an['F_total']/(p.N*f1):.3f}")
    if p.N > 1:
        lines.append(f"Kuramoto R_sync (time mean)  : {an['R_sync']:.3f}")
        lines.append(f"neighbour Δθ: circular mean  : {an['dphi_mean']:+.3f} rad "
                     f"({np.degrees(an['dphi_mean']):+.1f}°), coherence {an['dphi_coherence']:.3f}, "
                     f"pair-locking {an['dphi_lock']:.3f}")
        lines.append(f"pattern                      : {an['wave_kind']}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_timeseries(res, an, p, path):
    t = res["t"] * 1e3
    fig, ax = plt.subplots(4 if p.N > 1 else 3, 1, figsize=(9, 10), sharex=True)
    cmap = plt.get_cmap("viridis")
    for i in range(p.N):
        c = cmap(i / max(1, p.N - 1))
        ax[0].plot(t, res["Fx"][:, i], color=c, lw=0.8)
        ax[1].plot(t, res["phidot"][:, i] / TWO_PI, color=c, lw=0.8)
    ax[0].plot(t, res["Fx"].sum(1), "k", lw=1.2, label="Σ Fx")
    ax[0].set_ylabel("Fx on fluid [pN]"); ax[0].legend(loc="upper right")
    ax[1].set_ylabel("φ̇/2π [Hz]")
    if p.wall_mode == "swimming":
        ax[2].plot(t, res["U"], "k"); ax[2].set_ylabel("U [µm/s]")
    else:
        ax[2].plot(t, res["Q"], lw=0.6); ax[2].set_ylabel("T_i [pN·µm/rad]")
    if p.N > 1:
        th = to_theta(res["phi"], res.get("phase_map"))
        ax[3].plot(t, np.abs(np.exp(1j * th).mean(1)), "k", label="R_sync(t)")
        ax[3].set_ylim(0, 1.05); ax[3].set_ylabel("Kuramoto R")
        ax3b = ax[3].twinx()
        ax3b.plot(t, np.degrees(np.angle(np.exp(1j * np.diff(th, axis=1)).mean(1))),
                  color="tab:red", lw=0.8)
        ax3b.set_ylabel("mean Δφ [deg]", color="tab:red"); ax3b.set_ylim(-185, 185)
    ax[-1].set_xlabel("t [ms]")
    t0 = p.transient * t[-1]
    for a in ax:
        a.axvspan(0, t0, color="0.9", zorder=-5)
    fig.suptitle(f"N={p.N}, d/L={p.d/p.L:.2f}, η={p.eta} mPa·s, {p.beat_mode}/{p.wall_mode}"
                 "  (grey: discarded transient)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_phase_diff(res, an, p, path):
    if p.N < 2:
        return
    n0 = int(p.transient * len(res["t"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.degrees(an["dphi"]).ravel(), bins=72, range=(-180, 180), density=True,
               color="tab:blue")
    ax[0].set_xlabel("Δθ = θ_{i+1} − θ_i  [deg]"); ax[0].set_ylabel("pdf")
    ax[0].set_title(f"{an['wave_kind']}\nmean {np.degrees(an['dphi_mean']):+.1f}°,"
                    f" coherence {an['dphi_coherence']:.2f}")
    # kymograph of phases (space–time)
    ph = np.mod(an["theta"], TWO_PI)
    t = res["t"][n0:] * 1e3
    im = ax[1].pcolormesh(np.arange(p.N), t, ph, cmap="twilight", shading="nearest",
                          vmin=0, vmax=TWO_PI, rasterized=True)
    ax[1].set_xlabel("flagellum index i"); ax[1].set_ylabel("t [ms]")
    ax[1].set_title("phase kymograph θ_i(t) mod 2π")
    fig.colorbar(im, ax=ax[1], label="φ [rad]")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def flow_grid(p, base_x, nx=70, nz=32):
    x0, x1 = base_x[0] - 1.1 * p.L, base_x[-1] + 1.1 * p.L
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(0.02 * p.L, 1.25 * p.L, nz)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.c_[XX.ravel(), np.zeros(XX.size), ZZ.ravel()]
    return XX, ZZ, pts


def draw_frame(ax, p, base_x, phi, X, F, eps, XX, ZZ, pts, vmax=None, U=0.0):
    u = velocity_field(pts, X, F, eps, p.eta_int)
    ux = u[:, 0].reshape(XX.shape); uz = u[:, 2].reshape(XX.shape)
    sp = np.hypot(ux, uz)
    vmax = vmax or np.percentile(sp, 98)
    pc = ax.pcolormesh(XX, ZZ, sp, cmap="magma", vmin=0, vmax=vmax, shading="gouraud")
    sk = 2
    ax.quiver(XX[::sk, ::sk], ZZ[::sk, ::sk], ux[::sk, ::sk], uz[::sk, ::sk],
              color="w", alpha=0.7, scale=vmax * 25, width=0.0022)
    sh = shape_full(phi, p)
    eff = in_effective(phi, p)
    for i in range(p.N):
        ax.plot(base_x[i] + sh[i, :, 0], sh[i, :, 1], color="cyan" if eff[i] else "lime", lw=2)
    ax.axhline(0, color="0.6", lw=4)
    ax.set_aspect("equal"); ax.set_xlim(XX.min(), XX.max()); ax.set_ylim(-0.5, ZZ.max())
    ax.set_xlabel("x [µm]"); ax.set_ylabel("z [µm]")
    return pc, vmax


def plot_flow(res, p, path):
    fr = res["frames"]
    if len(fr["t"]) == 0:
        return
    eps = res["eps"]
    XX, ZZ, pts = flow_grid(p, res["base_x"])
    # snapshot with largest |ΣFx| (mid effective strokes)
    idx = int(np.argmax([np.abs(F[:, 0].sum()) for F in fr["F"]]))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    pc, _ = draw_frame(ax, p, res["base_x"], fr["phi"][idx], fr["X"][idx], fr["F"][idx],
                       eps, XX, ZZ, pts)
    fig.colorbar(pc, ax=ax, label="|u| [µm/s]", shrink=0.8)
    ax.set_title(f"flow field t={fr['t'][idx]*1e3:.1f} ms   (cyan: effective, green: recovery)"
                 f"   d/L={p.d/p.L:.2f}, η={p.eta}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def make_animation(res, p, path_noext, fps=15):
    fr = res["frames"]
    if len(fr["t"]) == 0:
        return None
    eps = res["eps"]
    XX, ZZ, pts = flow_grid(p, res["base_x"], nx=56, nz=26)
    # common colour scale
    sp_all = []
    for k in range(0, len(fr["t"]), max(1, len(fr["t"]) // 6)):
        u = velocity_field(pts, fr["X"][k], fr["F"][k], eps, p.eta_int)
        sp_all.append(np.hypot(u[:, 0], u[:, 2]))
    vmax = np.percentile(np.concatenate(sp_all), 97)
    fig, ax = plt.subplots(figsize=(11, 4.4))

    def update(k):
        ax.clear()
        draw_frame(ax, p, res["base_x"], fr["phi"][k], fr["X"][k], fr["F"][k], eps, XX, ZZ, pts,
                   vmax=vmax)
        ax.set_title(f"t = {fr['t'][k]*1e3:6.1f} ms   d/L={p.d/p.L:.2f}  η={p.eta} mPa·s  "
                     f"ΣFx={fr['F'][k][:,0].sum():6.1f} pN   |u| max≈{vmax:.0f} µm/s")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(fr["t"]), blit=False)
    if shutil.which("ffmpeg"):
        path = path_noext + ".mp4"
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=110)
    else:
        path = path_noext + ".gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_param_args(ap)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_single", action="store_true", help="skip the N=1 reference run")
    ap.add_argument("--no_anim", action="store_true")
    ap.add_argument("--rft_check", action="store_true", help="only run the RFT / kernel checks")
    args = ap.parse_args()
    p = params_from_args(args)

    if args.rft_check:
        check_kernel()
        rft_check(p)
        return

    tag = args.out or os.path.join(
        "results", f"{p.beat_mode}_{p.wall_mode}_N{p.N}_dL{p.d/p.L:.2f}_eta{p.eta:g}")
    os.makedirs(tag, exist_ok=True)
    print(f"ε (blob size) = {get_eps(p):.4f} µm")
    t0 = time.time()
    res = run(p)
    an = analyze(res)
    an1 = None if (args.no_single or p.N == 1) else single_reference(p)
    if p.beat_mode == "load":
        print(f"ω0/2π = {res['omega0']/TWO_PI:.1f} Hz,  T_stall = {res['T_stall']:.1f} pN·µm/rad")
    txt = summarize(p, an, an1)
    print(txt)
    print(f"elapsed {time.time()-t0:.1f} s  → {tag}")
    with open(os.path.join(tag, "summary.txt"), "w") as fh:
        fh.write(txt + "\n\nparams:\n" + json.dumps(p.as_dict(), indent=1) + "\n")
    plot_timeseries(res, an, p, os.path.join(tag, "timeseries.png"))
    plot_phase_diff(res, an, p, os.path.join(tag, "phase_diff.png"))
    plot_flow(res, p, os.path.join(tag, "flow_field.png"))
    if not args.no_anim:
        path = make_animation(res, p, os.path.join(tag, "flagella"))
        print("animation:", path)
    np.savez_compressed(os.path.join(tag, "run.npz"),
                        **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})


if __name__ == "__main__":
    main()
