"""Figures for S2 (2-D carpet, fixed kinematics)."""
import json
import numpy as np
from plot_style import plt, species_lines
from common import OUT

rows = json.load(open(f"{OUT}/s2_carpet.json"))
dLs = sorted({r["dL"] for r in rows})
dps = sorted({r["dphi_deg"] for r in rows})


def grid(key):
    G = np.full((len(dLs), len(dps)), np.nan)
    for r in rows:
        G[dLs.index(r["dL"]), dps.index(r["dphi_deg"])] = r[key]
    return G


Ef, Ep, Us, Pr, Ui = grid("eps_f"), grid("eps_pump"), grid("U_slip"), grid("power_ratio"), grid("U_slip_indep")

# ---------------------------------------------------------------- heat maps
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, G, lab, cm, vc in ((ax[0], Ef, "有効打の力 / 単独  (ε_f)", "RdBu_r", 1.0),
                          (ax[1], Ep, "1本あたりのポンプ量 / 単独  (ε_pump)", "RdBu_r", 1.0),
                          (ax[2], Us, "表面のすべり速度 U_slip [µm/s]", "viridis", None)):
    if vc is not None:
        m = np.nanmax(np.abs(G - vc))
        im = a.pcolormesh(dLs, dps, G.T, cmap=cm, vmin=vc - m, vmax=vc + m, shading="nearest")
        a.contour(dLs, dps, G.T, levels=[vc], colors="k", linewidths=0.8)
    else:
        im = a.pcolormesh(dLs, dps, G.T, cmap=cm, shading="nearest")
    fig.colorbar(im, ax=a)
    a.set_title(lab, fontsize=11)
    a.set_xlabel("d / L"); a.set_ylabel("隣との位相差 Δφ [度]  (負: 有効打方向に進む波)")
    a.axhspan(-40, -10, color="w", alpha=0.0, lw=1.5, ec="k", ls=":")
    species_lines(a, label=False)
    a.grid(False)
fig.suptitle("2次元周期カーペット（拍動は固定）: 点線枠 = 文献から見積もった Volvox の位相差 (−10〜−40°)")
fig.tight_layout(); fig.savefig(f"{OUT}/s2_heatmaps.png"); plt.close(fig)

# ---------------------------------------------------------------- curves vs d/L
sel = [0.0, -15.0, -30.0, -45.0, -90.0]
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for k, dp in enumerate(sel):
    j = dps.index(dp)
    c = cmap(k / (len(sel) - 1) * 0.9)
    lab = f"Δφ = {dp:+.0f}°" if dp else "Δφ = 0°（全員同時）"
    ax[0, 0].plot(dLs, Ef[:, j], "o-", color=c, label=lab)
    ax[0, 1].plot(dLs, Ep[:, j], "o-", color=c, label=lab)
    ax[1, 0].plot(dLs, Us[:, j], "o-", color=c, label=lab)
    ax[1, 1].plot(dLs, Pr[:, j], "o-", color=c, label=lab)
ax[1, 0].plot(dLs, Ui[:, 0], "k--", label="相互作用なしの場合")
ax[0, 0].set_ylabel("有効打の力 / 単独鞭毛  ε_f")
ax[0, 1].set_ylabel("1本あたりのポンプ量 / 単独  ε_pump")
ax[1, 0].set_ylabel("表面すべり速度 U_slip [µm/s]")
ax[1, 1].set_ylabel("1本あたりの消費パワー / 単独")
ax[1, 0].set_yscale("log")
for a in ax.ravel():
    a.set_xlabel("d / L")
    if a is not ax[1, 0]:
        a.axhline(1, color="0.4", lw=0.8)
    species_lines(a)
ax[0, 0].legend(fontsize=8)
ax[1, 0].legend(fontsize=8)
fig.suptitle("2次元カーペット：d/L 依存性（L = 22 µm, 30 Hz, 拍動固定, 横方向の列は同位相）")
fig.tight_layout(); fig.savefig(f"{OUT}/s2_curves.png"); plt.close(fig)
print("saved s2 figures")
