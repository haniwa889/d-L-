"""Figures for S7 (stability of metachronal waves) + overall summary figure."""
import json
import numpy as np
from plot_style import plt, species_lines
from common import OUT

rows = json.load(open(f"{OUT}/s7_stability.json"))
etas = sorted({r["eta"] for r in rows})
dLs = sorted({r["dL"] for r in rows})
dps = sorted({r["dphi_deg"] for r in rows})

fig, ax = plt.subplots(1, len(etas), figsize=(6.5 * len(etas), 4.8), squeeze=False)
for a, e in zip(ax[0], etas):
    G = np.full((len(dLs), len(dps)), np.nan)
    for r in rows:
        if r["eta"] == e:
            G[dLs.index(r["dL"]), dps.index(r["dphi_deg"])] = r["lam"]
    m = np.nanmax(np.abs(G))
    im = a.imshow(G.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-m, vmax=m,
                  extent=(-0.5, len(dLs) - 0.5, -0.5, len(dps) - 0.5))
    for i in range(len(dLs)):
        for j in range(len(dps)):
            if np.isfinite(G[i, j]):
                a.text(i, j, "安定" if G[i, j] < 0 else "", ha="center", va="center", fontsize=7)
    a.set_xticks(range(len(dLs))); a.set_xticklabels([f"{x:.2f}" for x in dLs])
    a.set_yticks(range(len(dps))); a.set_yticklabels([f"{x:+.0f}°" for x in dps])
    a.set_xlabel("d / L"); a.set_ylabel("波の隣接位相差 Δφ（負: 有効打方向に進む）")
    a.set_title(f"η = {e:g} mPa·s", fontsize=11)
    a.grid(False)
    fig.colorbar(im, ax=a, label="摂動の成長率 λ [1/拍動]（負 = 安定）")
fig.suptitle("S7: メタクロナル波（各位相差）の安定性：小さな乱れが消えるか増えるか")
fig.tight_layout(); fig.savefig(f"{OUT}/s7_stability.png"); plt.close(fig)

print(f"{'eta':>5} {'d/L':>5} " + " ".join(f"{d:+5.0f}" for d in dps))
for e in etas:
    for x in dLs:
        lam = {r["dphi_deg"]: r["lam"] for r in rows if r["eta"] == e and r["dL"] == x}
        print(f"{e:5.2f} {x:5.2f} " + " ".join(f"{lam.get(d, np.nan):+.3f}" for d in dps))
