"""P4 analysis: decompose F_model = η · f_beat · I(shape, coordination).
I (impulse per beat per unit viscosity) is taken from the model with the measured amplitude;
the beat frequency is estimated from the reliably measured effective-stroke time assuming the
water value of the effective-stroke fraction (f_alt = p_eff(water) / t_eff(η)).
Output (private): p4_decomposition.json and p4_viscosity_vs_exp.png"""
import json, os
import numpy as np
from plot_style import plt
from exp_params import species_params, PRIV, load_rows

p4 = json.load(open(os.path.join(PRIV, "p4_viscosity.json")))
rows = load_rows()
COL = {"Volvox": "tab:green", "Pleodorina": "tab:purple"}
STATES = (("independent", 0, "tab:gray", "相互作用なし"),
          ("random2D", None, None, "ランダム位相（協調なし）"),
          ("wave", -2, "tab:orange", "逆向きの波 Δφ=+30°"),
          ("wave", 2, "tab:blue", "有効打方向の波 Δφ=−30°"),
          ("wave", 0, "tab:red", "全員同時"))
out = {}
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
for j, sp in enumerate(("Volvox", "Pleodorina")):
    q0 = species_params(sp, 0.0)
    pe0 = q0["p_eff"]
    for i, amp in enumerate(("eff", "tip")):
        a = ax[i, j]
        ex = [r for r in rows if r["species"] == sp and r["PEO_wt%"] <= 0.2 + 1e-9]
        F0 = [r for r in ex if r["PEO_wt%"] == 0][0]
        okF = [r for r in ex if np.isfinite(r["F_colony_mean_pN"])]
        a.errorbar([r["eta/eta_w"] for r in okF], [r["F_colony_mean_pN"] / F0["F_colony_mean_pN"] for r in okF],
                   yerr=[(r["F_colony_SD"] if np.isfinite(r["F_colony_SD"]) else 0) / F0["F_colony_mean_pN"] for r in okF],
                   fmt="ks-", lw=2.2, capsize=3, label="実測 F/F(水)", zorder=5)
        a.plot([r["eta/eta_w"] for r in ex], [r["f_flag_mean_pN"] / F0["f_flag_mean_pN"] for r in ex],
               "k^--", alpha=0.55, label="実測 f/f(水)")
        for st, arg, c, lab in STATES:
            rr = [r for r in p4 if r["species"] == sp and r["amp"] == amp and r["state"] == st
                  and (arg is None or r["arg"] == arg)]
            wts = sorted({r["wt"] for r in rr})
            pred = []
            for w in wts:
                rw = [r for r in rr if r["wt"] == w]
                I = np.mean([r["F_model"] / (r["eta"] * r["f_beat"]) for r in rw])
                q = species_params(sp, w)
                f_alt = pe0 / q["t_eff"]
                pred.append((q["eta"] / 0.84, q["eta"] * f_alt * I, I))
            pred = np.array(pred)
            if pred[0, 1] == 0:
                continue
            out[f"{sp}_{amp}_{st}_{arg}"] = pred.tolist()
            a.plot(pred[:, 0], pred[:, 1] / pred[0, 1], "o-", color=c or COL[sp], label="モデル: " + lab)
        a.axhline(1, color="0.5", lw=0.8)
        a.set_ylim(-0.5, 3.0)
        a.set_xlabel("η / η_水"); a.set_ylabel("水中の値に対する比")
        a.set_title(f"{sp}: 振幅 = {'有効打振幅' if amp == 'eff' else '先端振幅'}（測定値を各濃度で使用）",
                    fontsize=10)
        a.legend(fontsize=7, loc="upper left")
fig.suptitle("P4: 実測の波形変化だけで F(η) を予測（協調状態は固定）。周波数 = 水の有効打割合 / 実測の振下時間")
fig.tight_layout(); fig.savefig(os.path.join(PRIV, "p4_viscosity_vs_exp.png")); plt.close(fig)
json.dump(out, open(os.path.join(PRIV, "p4_decomposition.json"), "w"), indent=1)
for k, v in out.items():
    if "random2D" in k or "independent" in k:
        v = np.array(v)
        print(k, " η/ηw:", np.round(v[:, 0], 2), " F/F0:", np.round(v[:, 1] / v[0, 1], 2),
              " I/I0:", np.round(v[:, 2] / v[0, 2], 2))

# ---------------------------------------------------------------- absolute values (eff amplitude)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, sp in zip(ax, ("Volvox", "Pleodorina")):
    ex = [r for r in rows if r["species"] == sp and r["PEO_wt%"] <= 0.2 + 1e-9]
    okF = [r for r in ex if np.isfinite(r["F_colony_mean_pN"])]
    a.errorbar([r["eta/eta_w"] for r in okF], [r["F_colony_mean_pN"] for r in okF],
               yerr=[r["F_colony_SD"] if np.isfinite(r["F_colony_SD"]) else 0 for r in okF],
               fmt="ks-", lw=2.2, capsize=3, label="実測 F", zorder=5)
    for st, arg, c, lab in STATES[:4] + (STATES[4],):
        k = f"{sp}_eff_{st}_{arg}"
        if k not in out:
            continue
        v = np.array(out[k])
        a.plot(v[:, 0], v[:, 1], "o-", color=c or COL[sp], label="モデル: " + lab)
    a.set_xlabel("η / η_水"); a.set_ylabel("群体の推進力 F [pN]")
    a.set_title(f"{sp}（振幅 = 有効打振幅、各濃度の実測波形）", fontsize=10)
    a.axhline(0, color="0.5", lw=0.8)
    a.legend(fontsize=7)
fig.suptitle("P4: 群体推進力の絶対値 — モデル（協調状態を固定）と実測")
fig.tight_layout(); fig.savefig(os.path.join(PRIV, "p4_absolute_vs_exp.png")); plt.close(fig)
