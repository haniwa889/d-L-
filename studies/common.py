"""Shared helpers for the study scripts (carpet runs + single-flagellum references)."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "studies", "out")
os.makedirs(OUT, exist_ok=True)

import numpy as np
from functools import lru_cache
from model import Params, System, shape_and_tangent, in_effective, TWO_PI, ETA_UNIT, get_eps
from carpet import run_carpet, analyze_carpet

DL_VOLVOX = 9.0 / 22.0
DL_PLEO = 20.0 / 19.0


@lru_cache(maxsize=64)
def single_ref(L=22.0, N_seg=12, theta0=1.15, p_eff=0.35, bend_width=0.25, f=30.0, eta=0.84,
               n_phi=80):
    """Isolated anchored flagellum, fixed beat: effective-stroke mean Fx (f1), cycle-mean Fx,
    cycle-mean Σ F_x z (M1, pumping moment) and power (P1)."""
    p = Params(N=1, L=L, N_seg=N_seg, theta0=theta0, p_eff=p_eff, bend_width=bend_width,
               f_fixed=f, eta=eta, beat_mode="fixed")
    s = System(p)
    om = TWO_PI * f
    phis = np.linspace(0, TWO_PI, n_phi, endpoint=False)
    Fx, Mz, Pw = [], [], []
    for ph in phis:
        r = s.solve(np.array([ph]), omega_fixed=om)
        Fx.append(r["F"][:, 0].sum()); Mz.append(np.sum(r["F"][:, 0] * r["X"][:, 2]))
        Pw.append(r["Q"][0] * om)
    Fx, Mz, Pw = map(np.array, (Fx, Mz, Pw))
    eff = in_effective(phis, p)
    return dict(f1=float(Fx[eff].mean()), F1=float(Fx.mean()), M1=float(Mz.mean()),
                P1=float(Pw.mean()))


def carpet_point(dL, w, n=24, L=22.0, N_seg=12, theta0=1.15, p_eff=0.35, bend_width=0.25,
                 dy_factor=1.0, steps=40, eta=0.84, f=30.0):
    """Fixed-kinematics carpet with winding number w (Δφ = −2π w / n per neighbour;
    w>0: symplectic wave, travelling in +x = effective-stroke direction)."""
    d = dL * L
    p = Params(L=L, d=d, N_seg=N_seg, theta0=theta0, p_eff=p_eff, bend_width=bend_width,
               beat_mode="fixed", steps_per_cycle=steps, n_cycles=1.0, transient=0.0,
               eta=eta, f_fixed=f)
    phi0 = np.mod(-np.arange(n) * TWO_PI * w / n, TWO_PI)
    res = run_carpet(p, n, phi0=phi0, dy=dy_factor * d)
    a = analyze_carpet(res)
    ref = single_ref(L, N_seg, theta0, p_eff, bend_width, f, eta)
    eta_i = eta * ETA_UNIT
    A_flag = res["A_cell"] / n                                  # area per flagellum
    out = dict(dL=dL, d=d, L=L, w=w, n=n, dphi_deg=-360.0 * w / n, dy=dy_factor * d,
               theta0=theta0, p_eff=p_eff, bend_width=bend_width, N_seg=N_seg,
               f_eff=a["f_eff"], F_net=a["F_net"], U_slip=a["U_slip"],
               power_flag=a["power_per_flag"], **{k + "_single": v for k, v in ref.items()})
    out["eps_f"] = a["f_eff"] / ref["f1"]                        # effective-stroke force ratio
    # pumping per flagellum relative to an isolated flagellum: (U_slip η A_flag) / M1
    out["eps_pump"] = a["U_slip"] * eta_i * A_flag / ref["M1"]
    out["power_ratio"] = a["power_per_flag"] / ref["P1"]
    # slip velocity if the flagella did not interact at all (dilute-carpet limit)
    out["U_slip_indep"] = ref["M1"] / (eta_i * A_flag)
    out["U_per_power"] = a["U_slip"] / (a["power_per_flag"] / A_flag)   # µm/s per (pN·µm/s/µm²)
    return out
