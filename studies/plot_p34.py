"""Figures comparing the model with the user's measurements (P3, P4).
Output goes to studies/private/ (contains unpublished data; git-ignored)."""
import json, os
import numpy as np
from plot_style import plt
from exp_params import species_params, PRIV, load_rows

COL = {"Volvox": "tab:green", "Pleodorina": "tab:purple"}

# ------------------------------------------------------------------ P3 bar chart
p3 = json.load(open(os.path.join(PRIV, "p3_boost.json")))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, sp in zip(ax, ("Volvox", "Pleodorina")):
    q = species_params(sp, 0.0)
    labels, vals, cols = [], [], []
    for amp, hatch in (("eff", ""), ("tip", "//")):
        rr = [r for r in p3 if r["species"] == sp and r["amp"] == amp and r["npc"] == 1]
        m = lambda st: np.mean([r["F_model"] for r in rr if r["state"] == st])
        waves = {r["w"]: r["F_model"] for r in rr if r["state"] == "wave"}
        items = [("相互作用なし", m("independent")), ("ランダム位相\n(協調なし)", m("random2D")),
                 ("全員同時", m("sync2D")), ("有効打方向の波\nΔφ=−30°", waves[2]),
                 ("逆向きの波\nΔφ=+30°", waves[-2])]
        for lab, v in items:
            labels.append(f"{lab}\n[{'有効打振幅' if amp == 'eff' else '先端振幅'}]")
            vals.append(v); cols.append(hatch)
    x = np.arange(len(vals))
    bars = a.bar(x, vals, color=[COL[sp] if h == "" else "w" for h in cols],
                 edgecolor=COL[sp], hatch=None)
    for b, h in zip(bars, cols):
        b.set_hatch(h)
    a.axhline(q["F_exp"], color="k", lw=1.5)
    a.axhspan(q["F_exp"] - q["F_exp_sd"], q["F_exp"] + q["F_exp_sd"], color="0.8", alpha=0.6,
              label=f"実測 F = {q['F_exp']:.0f} ± {q['F_exp_sd']:.0f} pN（水）")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=6.5, rotation=0)
    a.set_ylabel("群体の推進力 F = 6πηRU [pN]")
    a.set_title(f"{sp}（d/L = {q['d']/q['L']:.2f}）", fontsize=11)
    a.legend(fontsize=8); a.grid(axis="x", visible=False)
fig.suptitle("P3: 鞭毛の協調状態ごとのモデル推進力と実測値（水中、拍動固定、2次元周期カーペット＋球の遊泳換算）")
fig.tight_layout(); fig.savefig(os.path.join(PRIV, "p3_force_vs_exp.png")); plt.close(fig)

# ------------------------------------------------------------------ P4 viscosity
p4path = os.path.join(PRIV, "p4_viscosity.json")
if os.path.exists(p4path):
    p4 = json.load(open(p4path))
    rows = load_rows()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for j, sp in enumerate(("Volvox", "Pleodorina")):
        for i, amp in enumerate(("eff", "tip")):
            a = ax[i, j]
            ex = [r for r in rows if r["species"] == sp and r["PEO_wt%"] <= 0.2 + 1e-9]
            F0 = [r for r in ex if r["PEO_wt%"] == 0][0]
            xe = [r["eta/eta_w"] for r in ex if np.isfinite(r["F_colony_mean_pN"])]
            Fe = [r["F_colony_mean_pN"] / F0["F_colony_mean_pN"] for r in ex if np.isfinite(r["F_colony_mean_pN"])]
            Fs = [r["F_colony_SD"] / F0["F_colony_mean_pN"] if np.isfinite(r["F_colony_SD"]) else 0
                  for r in ex if np.isfinite(r["F_colony_mean_pN"])]
            a.errorbar(xe, Fe, yerr=Fs, fmt="ks-", lw=2, capsize=3, label="実測 F/F(水)")
            xf = [r["eta/eta_w"] for r in ex]
            a.plot(xf, [r["f_flag_mean_pN"] / F0["f_flag_mean_pN"] for r in ex], "k^--", alpha=0.6,
                   label="実測 f/f(水)（鞭毛1本の推定力）")
            for st, arg, c, lab in (("independent", 0, "tab:gray", "モデル: 相互作用なし"),
                                    ("random2D", None, COL[sp], "モデル: ランダム位相（協調なし）"),
                                    ("wave", -2, "tab:orange", "モデル: 逆向きの波 Δφ=+30°"),
                                    ("wave", 2, "tab:blue", "モデル: 有効打方向の波 Δφ=−30°")):
                rr = [r for r in p4 if r["species"] == sp and r["amp"] == amp and r["state"] == st
                      and (arg is None or r["arg"] == arg)]
                wts = sorted({r["wt"] for r in rr})
                Fm = [np.mean([r["F_model"] for r in rr if r["wt"] == w]) for w in wts]
                eta_rel = [np.mean([r["eta"] for r in rr if r["wt"] == w]) / 0.84 for w in wts]
                if Fm and Fm[0] != 0:
                    a.plot(eta_rel, np.array(Fm) / Fm[0], "o-", color=c, label=lab)
            a.axhline(1, color="0.5", lw=0.8)
            a.set_xlabel("η / η_水"); a.set_ylabel("水中の値に対する比")
            a.set_title(f"{sp}: 振幅 = {'有効打振幅' if amp == 'eff' else '先端振幅'}", fontsize=10)
            a.legend(fontsize=7)
    fig.suptitle("P4: 実測の波形変化（振幅・周波数）だけで、粘性による F の変化を説明できるか")
    fig.tight_layout(); fig.savefig(os.path.join(PRIV, "p4_viscosity_vs_exp.png")); plt.close(fig)
print("saved")
