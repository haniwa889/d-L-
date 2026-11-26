"""Figure for P7 (wave stability with species geometry, η within the experimental range).
Output → studies/private/p7_stability.png (git-ignored)."""
import json, os, sys
import numpy as np
from plot_style import plt
from exp_params import PRIV

tag = sys.argv[1] if len(sys.argv) > 1 else "p7_stability"
rows = json.load(open(os.path.join(PRIV, f"{tag}.json")))
JP = {"Volvox": "ボルボックス", "Pleodorina": "プレオドリナ"}
etas = sorted({r["eta"] for r in rows})
ffs = sorted({r["f_free"] for r in rows})
CM = {e: c for e, c in zip(etas, ("tab:blue", "tab:orange", "tab:red", "tab:brown"))}
LS = {ff: s for ff, s in zip(ffs, ("-", "--", ":"))}


def lam_eff(r):
    """(λ, strongly_unstable).  If the late-time fit failed because the lag spread
    saturated (≥ 0.2 period), fit the growth phase from the start until saturation."""
    if np.isfinite(r["lam"]):
        return r["lam"], False
    s = np.asarray(r["sigma"])
    sat = np.nonzero(s >= 0.2)[0]
    k = sat[0] if len(sat) else len(s)
    if k >= 3:
        m = np.arange(1, k)
        return float(np.polyfit(m, np.log(np.maximum(s[1:k], 1e-9)), 1)[0]), True
    return r["lam0"], True


fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
summ = {}
for a, sp in zip(ax[:2], ("Volvox", "Pleodorina")):
    for ff in ffs:
        for e in etas:
            rr = sorted([r for r in rows if r["species"] == sp and r["eta"] == e and r["f_free"] == ff],
                        key=lambda r: r["dphi_deg"])
            x = [r["dphi_deg"] for r in rr]
            y = [lam_eff(r)[0] for r in rr]
            strong = [lam_eff(r)[1] for r in rr]
            yc = np.clip(y, -0.1, 0.1)
            a.plot(x, yc, LS[ff], color=CM[e], marker="o", ms=4,
                   label=f"η/η_水={e/0.84:.1f}, f_free={ff:g} Hz（拍動 {np.mean([r['freq'] for r in rr]):.0f} Hz、負荷 {np.mean([r['load_frac'] for r in rr]):.2f}）")
            for xi, yi, s in zip(x, yc, strong):
                if s:
                    a.plot(xi, yi, "x", color=CM[e], ms=9, mew=2)
            summ[f"{sp} η={e} f_free={ff}"] = dict(
                dphi=x, lam=y, n_stable=int(sum(v < -0.002 for v in y)),
                n_strong_unstable=int(sum(v > 0.05 for v in y)),
                load_frac=float(np.mean([r["load_frac"] for r in rr])))
    a.axhline(0, color="k", lw=0.8)
    a.set_ylim(-0.1, 0.1)
    a.set_xlabel("波の隣どうしの位相差 Δφ [°]（負＝有効打方向の波）")
    a.set_ylabel("乱れの成長率 λ [1/拍動]\n（負＝安定、×＝数拍動で崩壊、±0.1 で打ち切り）")
    a.set_title(JP[sp], fontsize=11)
    a.legend(fontsize=6.5, loc="lower left")
# summary: number of stable waves vs motor load fraction 1 − f/f_free
b = ax[2]
MK = {50.0: "o", 100.0: "D"}
for sp, c in (("Volvox", "tab:green"), ("Pleodorina", "tab:purple")):
    pts = sorted((summ[f"{sp} η={e} f_free={ff}"]["load_frac"], summ[f"{sp} η={e} f_free={ff}"]["n_stable"], ff, e)
                 for e in etas for ff in ffs)
    b.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=c, alpha=0.4)
    for lf, ns, ff, e in pts:
        b.plot(lf, ns, MK.get(ff, "s"), color=c, ms=8, mfc=c if ff == ffs[0] else "w")
        b.annotate(f"{e/0.84:.1f}", (lf, ns), textcoords="offset points", xytext=(4, 4), fontsize=7, color=c)
    b.plot([], [], "-", color=c, label=JP[sp])
for ff in ffs:
    b.plot([], [], MK.get(ff, "s"), color="0.3", mfc="0.3" if ff == ffs[0] else "w", label=f"f_free = {ff:g} Hz")
b.set_xlabel("モーターの負荷の割合 1 − f/f_free（粘性で増える）")
b.set_ylabel(f"安定な波の数（{len({r['w'] for r in rows})} 種類中）")
b.set_title("数字 = η/η_水", fontsize=9)
b.legend(fontsize=8)
fig.suptitle("P7: 各種の配置・波形で、粘性を上げたときに協調した波（メタクロナル波）が安定に保てるか", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(PRIV, f"{tag}.png"), bbox_inches="tight"); plt.close(fig)
json.dump(summ, open(os.path.join(PRIV, f"{tag}_summary.json"), "w"), indent=1)
for k, v in summ.items():
    print(k, "stable", v["n_stable"], "strong-unstable", v["n_strong_unstable"],
          "load", round(v["load_frac"], 2), " λ:", np.round(v["lam"], 3).tolist())
