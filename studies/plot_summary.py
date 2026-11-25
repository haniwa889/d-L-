"""One-page summary figure of the d/L study (Japanese labels)."""
import json, os
import numpy as np
from plot_style import plt, species_lines
from common import OUT

s2 = json.load(open(f"{OUT}/s2_carpet.json"))
s5 = json.load(open(f"{OUT}/s5_load.json"))
s6 = json.load(open(f"{OUT}/s6_swimming.json"))
dL2 = sorted({r["dL"] for r in s2})

fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

# (A) shielding + sensitivity to coordination
a = ax[0, 0]
get = lambda x, dp: [r["eps_pump"] for r in s2 if r["dL"] == x and r["dphi_deg"] == dp][0]
band = np.array([[get(x, dp) for dp in (0.0, -15.0, -30.0, -45.0)] for x in dL2])
a.fill_between(dL2, band.min(1), band.max(1), color="tab:green", alpha=0.2,
               label="Δφ = −45〜0°（有効打方向の波）の範囲")
a.plot(dL2, band[:, 0], "o-", color="#3b0f70", label="Δφ = 0°（全員同時）")
a.plot(dL2, [get(x, 30.0) for x in dL2], "s--", color="tab:orange", label="Δφ = +30°（逆向きの波）")
a.axhline(1, color="0.5", lw=0.8)
a.set_xlabel("d / L"); a.set_ylabel("ε_pump（1本あたりポンプ量 / 単独）")
a.set_title("(A) 位相のそろえ方で効率がどれだけ変わるか（拍動固定）", fontsize=10)
a.legend(fontsize=8, loc="lower right"); species_lines(a)

# (B) self-organised states
a = ax[0, 1]
for e, c in ((0.84, "tab:blue"), (8.61, "tab:red")):
    rr = [r for r in s5 if r["eta"] == e]
    xs = sorted({r["dL"] for r in rr})
    a.plot([r["dL"] for r in rr], [r["eps_pump"] for r in rr], "o", color=c, alpha=0.5)
    a.plot(xs, [np.mean([r["eps_pump"] for r in rr if r["dL"] == x]) for x in xs], "-", color=c,
           label=f"η = {e:g} mPa·s")
a.set_xlabel("d / L"); a.set_ylabel("ε_pump（自己組織化した状態）")
a.set_title("(B) 位相が自由に動くとき：間隔が広いほど1本の効率が高い", fontsize=10)
a.legend(fontsize=8); species_lines(a)

# (C) viscosity sensitivity of propulsion per beat
a = ax[1, 0]
xs = sorted({r["dL"] for r in s5})
m = lambda e, x, k: np.mean([r[k] for r in s5 if r["eta"] == e and r["dL"] == x])
ratio = [(m(8.61, x, "U_slip") / m(8.61, x, "freq")) / (m(0.84, x, "U_slip") / m(0.84, x, "freq"))
         for x in xs]
a.plot(xs, ratio, "o-", color="k")
a.set_xlabel("d / L"); a.set_ylabel("(U/f) at η=8.61 ÷ (U/f) at η=0.84")
a.set_title("(C) 予測：粘性を上げたとき「1拍動あたりの推進」の落ち方", fontsize=10)
a.set_ylim(0.5, 1.0); species_lines(a)

# (D) experiment-style epsilon contains a geometric factor
a = ax[1, 1]
names = ["Volvox", "Pleodorina"]
geo = [s6[n]["rows"][0]["geom_factor"] for n in names]
dLsp = [s6[n]["dL"] for n in names]
a.bar([0, 1], geo, color=["tab:green", "tab:purple"])
for i, g in enumerate(geo):
    a.text(i, g, f"{g:.3f}", ha="center", va="bottom")
a.set_xticks([0, 1]); a.set_xticklabels([f"{n}\nR={s6[n]['params']['R']:.0f} µm, L={s6[n]['params']['L']:.0f} µm"
                                         for n in names])
a.set_ylabel("ε_drag / ε_pump（幾何学因子 ∝ L/R）")
a.set_title(f"(D) 6πηRU/(N f) 型のεは、相互作用が同じでも {geo[1]/geo[0]:.1f} 倍違う", fontsize=10)
a.grid(False)

fig.suptitle("d/L と鞭毛間の流体相互作用：シミュレーションのまとめ（2次元周期カーペット）", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/summary.png"); plt.close(fig)
print("saved summary.png")
