"""Figure for P5 (self-organised phases with motor law and phase noise, species geometry).
Output → studies/private/p5_noise.png (contains the user's parameters; git-ignored)."""
import json, os
import numpy as np
from plot_style import plt
from exp_params import PRIV

COL = {"Volvox": "tab:green", "Pleodorina": "tab:purple"}
JP = {"Volvox": "ボルボックス", "Pleodorina": "プレオドリナ"}
rows = json.load(open(os.path.join(PRIV, "p5_noise.json")))
etas = sorted({r["eta"] for r in rows})
Ds = sorted({r["D"] for r in rows})
LS = {0.0: "-", 1.0: "--", 4.0: ":"}

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
summary = {}
for sp in ("Volvox", "Pleodorina"):
    for D in Ds:
        g = lambda key, e: [r[key] for r in rows if r["species"] == sp and r["D"] == D and r["eta"] == e]
        x = np.array(etas) / 0.84
        coh = [np.mean(g("dphi_coh", e)) for e in etas]
        R = [np.mean(g("R_sync", e)) for e in etas]
        upb = [np.mean(g("U_per_beat", e)) for e in etas]
        base = [np.mean([r["U_per_beat"] for r in rows if r["species"] == sp and r["D"] == 4.0
                         and r["eta"] == e]) for e in etas]
        rel = np.array(upb) / np.array(base)
        lab = f"{JP[sp]} D={D:g} rad²/s"
        ax[0, 0].plot(x, coh, "o" + LS[D], color=COL[sp], label=lab)
        ax[0, 1].plot(x, R, "o" + LS[D], color=COL[sp], label=lab)
        ax[1, 0].plot(x, rel, "o" + LS[D], color=COL[sp], label=lab)
        if D == 0.0:
            for e in etas:
                v = [np.degrees(r["dphi_mean"]) for r in rows if r["species"] == sp and r["D"] == 0
                     and r["eta"] == e]
                c = [r["dphi_coh"] for r in rows if r["species"] == sp and r["D"] == 0 and r["eta"] == e]
                ax[1, 1].scatter([e / 0.84] * len(v), v, s=[400 * cc for cc in c], color=COL[sp],
                                 alpha=0.6, label=JP[sp] if e == etas[0] else None)
        summary[f"{sp} D={D:g}"] = dict(coh=coh, R=R, U_rel_to_noisy=rel.tolist(),
                                       freq=[np.mean(g("freq", e)) for e in etas])
ax[0, 0].set_ylabel("隣どうしの位相差のそろい具合（コヒーレンス）")
ax[0, 1].set_ylabel("全体の同期度 R（Kuramoto）")
ax[1, 0].set_ylabel("1拍動あたりの推進 ÷ 強いノイズ（協調なし）のとき")
ax[1, 0].axhline(1, color="0.4", lw=0.8)
ax[1, 1].set_ylabel("自発的に現れた隣どうしの位相差 Δφ [°]\n（負＝有効打方向の波、点の大きさ＝そろい具合）")
ax[1, 1].axhline(0, color="0.4", lw=0.8)
for a in ax.ravel():
    a.set_xlabel("η / η_水")
    a.legend(fontsize=7)
fig.suptitle("P5: モーター則＋位相ノイズで自発的に決まる協調（各種の d, L, 有効打振幅；水中で実測拍動数）")
fig.tight_layout()
fig.savefig(os.path.join(PRIV, "p5_noise.png"))
json.dump(summary, open(os.path.join(PRIV, "p5_summary.json"), "w"), indent=1)
for k, v in summary.items():
    print(k, {kk: np.round(vv, 2).tolist() for kk, vv in v.items()})
