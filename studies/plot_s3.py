"""Figures / tables for S3 (robustness, which variable controls the interaction)."""
import json
import numpy as np
from plot_style import plt, species_lines
from common import OUT

s3 = json.load(open(f"{OUT}/s3_robustness.json"))
s2 = json.load(open(f"{OUT}/s2_carpet.json"))
base = [r for r in s2 if r["w"] in (0, 1, 3, 6)]
for r in base:
    r["group"] = "base"
WCOL = {0: "#3b0f70", 1: "#3e4a89", 3: "#35b779", 6: "#b5de2b"}
WLAB = {0: "Δφ=0°", 1: "Δφ=−15°", 3: "Δφ=−45°", 6: "Δφ=−90°"}

fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
# (a) collapse in L
for key, a in (("eps_f", ax[0, 0]), ("eps_pump", ax[1, 0])):
    for w in (0, 1, 3, 6):
        for L, mk, ls in ((15.0, "s", ":"), (22.0, "o", "-"), (30.0, "^", "--")):
            rr = [r for r in s3 + base if r["w"] == w and abs(r["L"] - L) < 1e-6 and
                  r["group"] in ("a", "base") and r["dL"] <= 1.06 and abs(r["dy"] - r["d"]) < 1e-6
                  and r["theta0"] == 1.15 and r["p_eff"] == 0.35 and r["bend_width"] == 0.25
                  and r["N_seg"] == 12]
            rr = sorted(rr, key=lambda r: r["dL"])
            if L == 22.0:
                rr = [r for r in rr if r["dL"] in (0.41, 0.7, 1.05)]
            a.plot([r["dL"] for r in rr], [r[key] for r in rr], mk + ls, color=WCOL[w],
                   label=f"{WLAB[w]}, L={L:g}" if w in (0, 3) else None)
    a.set_xlabel("d / L"); species_lines(a, label=False)
ax[0, 0].set_title("(a) Lを変えても d/L が同じなら同じ値か？", fontsize=10)
ax[0, 0].set_ylabel("ε_f（有効打の力/単独）"); ax[1, 0].set_ylabel("ε_pump（ポンプ量/単独）")
ax[0, 0].legend(fontsize=7)

# (b) anisotropy
combos = [(0.41, 0.41), (0.41, 1.05), (0.41, 3.0), (1.05, 0.41), (1.05, 1.05), (1.05, 3.0)]
xx = np.arange(len(combos))
for key, a in (("eps_f", ax[0, 1]), ("eps_pump", ax[1, 1])):
    for k, w in enumerate((0, 1, 3, 6)):
        vals = []
        for dx, dy in combos:
            r = [r for r in s3 if r["group"] == "b" and r["w"] == w and abs(r["dL"] - dx) < 1e-6
                 and abs(r["dy"] / r["L"] - dy) < 1e-3]
            vals.append(r[0][key] if r else np.nan)
        a.bar(xx + (k - 1.5) * 0.2, vals, 0.2, color=WCOL[w], label=WLAB[w])
    a.set_xticks(xx); a.set_xticklabels([f"dx/L={dx}\ndy/L={dy}" for dx, dy in combos], fontsize=7)
    a.axhline(1, color="0.4", lw=0.8)
ax[0, 1].set_title("(b) 拍動方向の間隔 dx と横方向の間隔 dy のどちらが効くか", fontsize=10)
ax[0, 1].legend(fontsize=7)

# (c) beat amplitude
ths = (0.8, 1.0, 1.15, 1.35)
for key, a in (("eps_f", ax[0, 2]), ("eps_pump", ax[1, 2])):
    for th, mk in zip(ths, ("v", "s", "o", "^")):
        for w in (0, 3):
            if th == 1.15:
                rr = [r for r in base if r["w"] == w]
            else:
                rr = [r for r in s3 if r["group"] == "c" and r["theta0"] == th and r["w"] == w]
            rr = sorted(rr, key=lambda r: r["dL"])
            x = [r["dL"] / np.sin(th) for r in rr]        # d / (L sinθ0): spacing / beat half-width
            a.plot(x, [r[key] for r in rr], mk + "-", color=WCOL[w], alpha=0.4 + 0.6 * (th - 0.8) / 0.55,
                   label=f"θ0={th}, {WLAB[w]}")
    a.set_xlabel("d / (L sinθ0)   （間隔 / 拍動の片側振れ幅）")
ax[0, 2].set_title("(c) 振幅θ0を変えたとき d/(L sinθ0) でまとまるか", fontsize=10)
ax[0, 2].legend(fontsize=6, ncol=2)
fig.suptitle("S3: 頑健性と「d/L が本当に支配変数か」の検証（2次元カーペット, 拍動固定）")
fig.tight_layout(); fig.savefig(f"{OUT}/s3_robustness.png"); plt.close(fig)

# --- c: same data vs d/L (for comparison of collapse quality)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for a, xfun, lab in ((axs[0], lambda r, th: r["dL"], "d / L"),
                     (axs[1], lambda r, th: r["dL"] / np.sin(th), "d / (L sinθ0)")):
    for th, mk in zip(ths, ("v", "s", "o", "^")):
        for w in (0, 3):
            rr = ([r for r in base if r["w"] == w] if th == 1.15 else
                  [r for r in s3 if r["group"] == "c" and r["theta0"] == th and r["w"] == w])
            rr = sorted(rr, key=lambda r: r["dL"])
            a.plot([xfun(r, th) for r in rr], [r["eps_pump"] for r in rr], mk + "-", color=WCOL[w],
                   alpha=0.4 + 0.6 * (th - 0.8) / 0.55, label=f"θ0={th}, {WLAB[w]}")
    a.set_xlabel(lab); a.set_ylabel("ε_pump")
axs[0].legend(fontsize=6, ncol=2)
fig.suptitle("振幅θ0を変えた場合：横軸を d/L にするか d/(L sinθ0) にするか")
fig.tight_layout(); fig.savefig(f"{OUT}/s3_amplitude_collapse.png"); plt.close(fig)

# --- tables d, e
print("waveform variants (group d) and resolution (group e):")
for r in sorted([r for r in s3 if r["group"] in ("d", "e")],
                key=lambda r: (r["group"], r["dL"], r["w"], r["p_eff"], r["bend_width"])):
    b = [q for q in base if abs(q["dL"] - r["dL"]) < 1e-6 and q["w"] == r["w"]]
    b = b[0] if b else None
    print(f"{r['group']} dL={r['dL']:.2f} Δφ={r['dphi_deg']:+5.0f} p_eff={r['p_eff']} bw={r['bend_width']} "
          f"Nseg={r['N_seg']}: eps_f={r['eps_f']:.3f} eps_pump={r['eps_pump']:.3f} "
          f"(base {b['eps_f']:.3f}/{b['eps_pump']:.3f})" if b else "")
