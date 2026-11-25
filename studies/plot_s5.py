"""Figures for S5 (self-organised phases on the periodic carpet)."""
import json, sys
import numpy as np
from plot_style import plt, species_lines
from common import OUT

tag = sys.argv[1] if len(sys.argv) > 1 else "s5_load"
rows = json.load(open(f"{OUT}/{tag}.json"))
ser = np.load(f"{OUT}/{tag}_series.npz")
etas = sorted({r["eta"] for r in rows})
dLs = sorted({r["dL"] for r in rows})
seeds = sorted({r["seed"] for r in rows})
ecol = {e: c for e, c in zip(etas, ("tab:blue", "tab:red", "tab:orange"))}

# ---------------------------------------------------------------- summary
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
panels = [("dphi_from_winding", "出現した隣との位相差 Δφ [度]\n(周期内の巻き数から; 負=有効打方向の波)"),
          ("dphi_coh", "位相差のそろい具合 |⟨e^{iΔθ}⟩|"),
          ("R_sync", "Kuramoto 秩序パラメータ R"),
          ("freq_rel", "拍動周波数 / 単独鞭毛の周波数"),
          ("eps_pump", "ε_pump（1本あたりポンプ量/単独）"),
          ("U_slip", "表面すべり速度 U_slip [µm/s]")]
from model import Params, motor_params, estimated_frequency
fsingle = {}
for e in etas:
    p = Params(beat_mode="load", eta=e, N_seg=rows[0]["N_seg"], L=22.0)
    om, Ts = motor_params(p)
    fsingle[e] = estimated_frequency(p, om, Ts)
for r in rows:
    r["freq_rel"] = r["freq"] / fsingle[r["eta"]]
for a, (key, lab) in zip(ax.ravel(), panels):
    for e in etas:
        for s in seeds:
            rr = sorted([r for r in rows if r["eta"] == e and r["seed"] == s], key=lambda r: r["dL"])
            off = (etas.index(e) - 0.5) * 0.015
            a.plot(np.array([r["dL"] for r in rr]) + off, [r[key] for r in rr], "o",
                   color=ecol[e], alpha=0.7, label=f"η={e:g} mPa·s" if s == seeds[0] else None)
        rr = [r for r in rows if r["eta"] == e]
        m = [np.nanmedian([r[key] for r in rr if r["dL"] == x]) for x in dLs]
        a.plot(dLs, m, "-", color=ecol[e], lw=1)
    a.set_xlabel("d / L"); a.set_ylabel(lab, fontsize=9)
    species_lines(a)
ax[0, 0].axhspan(-40, -10, color="tab:green", alpha=0.12)
ax[0, 0].legend(fontsize=8)
fig.suptitle("S5: 位相が自由に動く場合（モーターの力–速度関係, 2次元カーペット, ランダム初期位相, 各点=1シード）")
fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_summary.png"); plt.close(fig)

# ---------------------------------------------------------------- kymographs
for e in etas:
    fig, ax = plt.subplots(1, len(dLs), figsize=(3.0 * len(dLs), 4.6), sharey=False)
    for a, x in zip(np.atleast_1d(ax), dLs):
        key = f"{x}_{e}_{seeds[0]}"
        t = ser[key + "_t"]; th = ser[key + "_theta"]
        n = th.shape[1]
        a.pcolormesh(np.arange(n), t * 1e3, np.mod(th, 2 * np.pi), cmap="twilight",
                     shading="nearest", vmin=0, vmax=2 * np.pi, rasterized=True)
        r = [r for r in rows if r["dL"] == x and r["eta"] == e and r["seed"] == seeds[0]][0]
        a.set_title(f"d/L={x:.2f}\nΔφ≈{r['dphi_from_winding']:+.0f}°, coh={r['dphi_coh']:.2f}", fontsize=9)
        a.set_xlabel("列番号 (x方向)"); a.grid(False)
    np.atleast_1d(ax)[0].set_ylabel("時間 [ms]")
    fig.suptitle(f"位相の時空間図（η={e:g} mPa·s, seed {seeds[0]}）: 斜めの縞 = メタクロナル波")
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_kymo_eta{e:g}.png"); plt.close(fig)
print("saved")
