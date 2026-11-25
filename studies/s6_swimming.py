"""S6: from the carpet to the swimming colony.

Flat-carpet limit (L ≪ R): the flagellar forces are the same whether the surface is
anchored or swimming; the colony swims with  U_swim ≈ (π/4) U_slip  (Stone & Samuel
1996, uniform tangential slip on a sphere).  We evaluate, for each species
(own d, L, R) and each neighbour lag Δφ:
  U_slip, U_swim,
  ε_f, ε_pump (interaction measures, independent of R),
  ε_drag = 6πηR U_swim / (N_cells f1)   ("experiment-style": drag of the colony
           divided by N × single-flagellum effective-stroke force)
         = (3π/8) ε_pump (M1/f1) / R    → contains the geometric factor ~ L/R.
Carpet data (L = 22 µm) are rescaled to the species' L at the same d/L:
forces ∝ L², U_slip ∝ L (same frequency).  Output: studies/out/s6_swimming.json + fig."""
import json
import numpy as np
from common import OUT, single_ref
from model import ETA_UNIT
from plot_style import plt

SPECIES = {
    "Volvox": dict(d=9.0, L=22.0, R=200.0, color="tab:green"),
    "Pleodorina": dict(d=20.0, L=19.0, R=50.0, color="tab:purple"),
}
rows = json.load(open(f"{OUT}/s2_carpet.json"))
dLs = np.array(sorted({r["dL"] for r in rows}))
dps = sorted({r["dphi_deg"] for r in rows})
eta = 0.84
out = {}
for name, s in SPECIES.items():
    dL = s["d"] / s["L"]
    ref22 = single_ref(22.0, 12)
    ref = single_ref(s["L"], 12)
    res = []
    for dp in dps:
        rr = sorted([r for r in rows if r["dphi_deg"] == dp], key=lambda r: r["dL"])
        x = np.array([r["dL"] for r in rr])
        ep = float(np.interp(dL, x, [r["eps_pump"] for r in rr]))
        ef = float(np.interp(dL, x, [r["eps_f"] for r in rr]))
        # U_slip at the species' own L: U = ε_pump M1(L) / (η d²)
        U_slip = ep * ref["M1"] / (eta * ETA_UNIT * s["d"] ** 2)
        U_swim = np.pi / 4 * U_slip
        Ncell = 4 * np.pi * s["R"] ** 2 / s["d"] ** 2
        eps_drag = 6 * np.pi * eta * ETA_UNIT * s["R"] * U_swim / (Ncell * ref["f1"])
        res.append(dict(dphi=dp, eps_f=ef, eps_pump=ep, U_slip=U_slip, U_swim=U_swim,
                        N_cells=Ncell, eps_drag=eps_drag,
                        geom_factor=3 * np.pi / 8 * ref["M1"] / ref["f1"] / s["R"]))
    out[name] = dict(params=s, dL=dL, f1=ref["f1"], M1=ref["M1"], rows=res)
json.dump(out, open(f"{OUT}/s6_swimming.json", "w"), indent=1)

print(f"{'species':10s} {'Δφ':>6s} {'ε_f':>6s} {'ε_pump':>7s} {'U_slip':>8s} {'U_swim':>8s} {'ε_drag':>8s} {'geom':>7s}")
for name, o in out.items():
    for r in o["rows"]:
        if r["dphi"] in (0, -15, -30, -45, -90, 15, 30, 45):
            print(f"{name:10s} {r['dphi']:6.0f} {r['eps_f']:6.3f} {r['eps_pump']:7.3f} {r['U_slip']:8.1f} "
                  f"{r['U_swim']:8.1f} {r['eps_drag']:8.4f} {r['geom_factor']:7.4f}")
    print(f"  N_cells={o['rows'][0]['N_cells']:.0f}, f1={o['f1']:.1f} pN, M1/f1={o['M1']/o['f1']:.2f} µm")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for name, o in out.items():
    c = o["params"]["color"]
    x = [r["dphi"] for r in o["rows"]]
    ax[0].plot(x, [r["eps_pump"] for r in o["rows"]], "o-", color=c, label=f"{name} (d/L={o['dL']:.2f})")
    ax[1].plot(x, [r["U_swim"] for r in o["rows"]], "o-", color=c, label=name)
    ax[2].plot(x, [r["eps_drag"] for r in o["rows"]], "o-", color=c, label=name)
for a in ax:
    a.axvspan(-40, -10, color="tab:green", alpha=0.12)
    a.set_xlabel("隣との位相差 Δφ [度]（負: 有効打方向に進む波）")
    a.legend(fontsize=8)
ax[0].set_ylabel("ε_pump（1本あたりのポンプ量 / 単独）")
ax[1].set_ylabel("推定遊泳速度 U_swim [µm/s]"); ax[1].set_yscale("symlog", linthresh=10)
ax[2].set_ylabel("ε_drag = 6πηRU / (N f_単独)")
ax[0].set_title("相互作用の効率（Rによらない）", fontsize=10)
ax[1].set_title("群体の遊泳速度（平板近似, 30 Hz）", fontsize=10)
ax[2].set_title("実験的な定義のε（L/R の幾何因子を含む）", fontsize=10)
fig.suptitle("S6: カーペットから群体の遊泳へ（緑の帯: 文献から見積もった Volvox の位相差）")
fig.tight_layout(); fig.savefig(f"{OUT}/s6_swimming.png"); plt.close(fig)
