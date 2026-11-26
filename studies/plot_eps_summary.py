"""Summary figure in terms of the user's ε = F/(N f):
 (a) species ratio ε_Pleodorina / ε_Volvox: measured vs model contributions
 (b) ε(η)/ε(water): measured vs model with measured waveform only (coordination fixed)
Output → studies/private/eps_summary.png (contains unpublished data; git-ignored)."""
import json, os
import numpy as np
from plot_style import plt
from exp_params import PRIV, load_rows

p3 = json.load(open(os.path.join(PRIV, "p3_boost.json")))
dec = json.load(open(os.path.join(PRIV, "p4_decomposition.json")))
rows = load_rows()
COL = {"Volvox": "tab:green", "Pleodorina": "tab:purple"}
JP = {"Volvox": "ボルボックス", "Pleodorina": "プレオドリナ"}

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
# (a) ratio
a = ax[0]
eps_exp = {sp: [r for r in rows if r["species"] == sp and r["PEO_wt%"] == 0][0]["epsilon_F/(N*f)"]
           for sp in COL}
labels, vals, cols = [], [], []
for amp, name in (("eff", "有効打振幅"), ("tip", "先端振幅")):
    for st, lab in (("independent", "相互作用なし\n（形 L/R, N だけ）"), ("random2D", "＋混み合い\n（協調なし）")):
        e = {sp: np.mean([r["eps_model"] for r in p3 if r["species"] == sp and r["amp"] == amp
                          and r["state"] == st and r["npc"] == 1]) for sp in COL}
        labels.append(f"モデル: {lab}\n[{name}]"); vals.append(e["Pleodorina"] / e["Volvox"])
        cols.append("tab:gray" if st == "independent" else "tab:blue")
labels.append("実測"); vals.append(eps_exp["Pleodorina"] / eps_exp["Volvox"]); cols.append("k")
x = np.arange(len(vals))
a.bar(x, vals, color=cols, alpha=0.8)
for xi, v in zip(x, vals):
    a.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=7)
a.set_ylabel("ε_プレオドリナ ÷ ε_ボルボックス（水中）")
a.set_title("(a) ε の種差：協調を入れなくてもどこまで説明できるか", fontsize=10)
a.grid(axis="x", visible=False)

# (b) viscosity
b = ax[1]
for sp in ("Volvox", "Pleodorina"):
    ex = [r for r in rows if r["species"] == sp and r["PEO_wt%"] <= 0.2 + 1e-9]
    e0 = ex[0]["epsilon_F/(N*f)"]
    ok = [r for r in ex if np.isfinite(r["epsilon_F/(N*f)"])]
    xe = [r["eta/eta_w"] for r in ok]
    ye = [r["epsilon_F/(N*f)"] / e0 for r in ok]
    se = [y * (r["F_colony_SD"] / r["F_colony_mean_pN"] if np.isfinite(r["F_colony_SD"]) else 0)
          for y, r in zip(ye, ok)]
    b.errorbar(xe, ye, yerr=se, fmt="s-", color=COL[sp], lw=2.2, capsize=3, label=f"{JP[sp]} 実測")
    v = np.array(dec[f"{sp}_eff_random2D_None"])
    f0 = ex[0]["f_flag_mean_pN"]
    fr = np.array([r["f_flag_mean_pN"] / f0 for r in ex])
    b.plot(v[:, 0], v[:, 1] / v[0, 1] / fr, "o--", color=COL[sp], alpha=0.75,
           label=f"{JP[sp]} モデル（実測の波形変化だけ、協調は固定）")
b.axhline(1, color="0.5", lw=0.8)
b.set_xlabel("η / η_水"); b.set_ylabel("ε / ε_水")
b.set_title("(b) 粘性による ε の変化：波形変化だけでどこまで説明できるか", fontsize=10)
b.legend(fontsize=8)
fig.suptitle("考察の指標 ε = F/(N f) で見た、モデルと実測の比較")
fig.tight_layout(); fig.savefig(os.path.join(PRIV, "eps_summary.png")); plt.close(fig)
print("ratios:", dict(zip([l.replace("\n", " ") for l in labels], np.round(vals, 2))))
