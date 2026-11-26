"""P4: viscosity dependence with the MEASURED kinematics at each PEO concentration.

Question: can the measured change of the waveform (amplitude, timing) alone explain the
colony force F(η) of each species, or is a loss of coordination required?

For each species and wt% (0–0.2), the model uses the measured stroke amplitude θ0(η)
(tip amplitude, or effective-stroke amplitude), effective-stroke fraction and beat
frequency; flagellar length fixed at the water value.  For fixed kinematics in a
Newtonian fluid  F_model ∝ η · f_beat · I(shape sequence, coordination),  so each
coordination state gives a prediction for F(η)/F(0) that does NOT invoke coordination
changes.  Compared with the measured F(η)/F(0) and f(η)/f(0).

Output → studies/private/p4_viscosity.json
"""
from common import OUT
import json, os, time, argparse
from multiprocessing import Pool
import numpy as np
from model import Params, TWO_PI, ETA_UNIT
from carpet import run_carpet, analyze_carpet, Carpet
from carpet2d import run_carpet2d, analyze_carpet2d, Carpet2D
from exp_params import species_params, f_formula, PRIV
from p3_boost import isolated

NS, SPC = 10, 40
WTS = [0.0, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2]


def params_at(sp, wt, amp):
    q0 = species_params(sp, 0.0, amp)
    q = species_params(sp, wt, amp)
    p = Params(L=q0["L"], d=q0["d"], N_seg=NS, theta0=q["theta0"], p_eff=q["p_eff"],
               f_fixed=q["f_beat"], eta=q["eta"], beat_mode="fixed", steps_per_cycle=SPC,
               n_cycles=1.0, transient=0.0)
    return q0, q, p


def job(t):
    sp, wt, amp, state, arg = t
    q0, q, p = params_at(sp, wt, amp)
    if not (0.02 < p.p_eff < 0.95):          # unreliable beat-frequency data
        p = p.copy(p_eff=q0["p_eff"])
    eta_i = q["eta"] * ETA_UNIT
    if state == "independent":
        iso = isolated(p)
        U = iso["M"] / (eta_i * q0["d"] ** 2)
    elif state == "random2D":
        ph = np.random.default_rng(arg).uniform(0, TWO_PI, 48)
        a = analyze_carpet2d(run_carpet2d(p, 8, 6, ph))
        U = a["U_slip"]
    else:
        n = 24
        ph = np.mod(-np.arange(n) * TWO_PI * arg / n, TWO_PI)
        a = analyze_carpet(run_carpet(p, n, phi0=ph), transient=0.0)
        U = a["U_slip"]
    F = 6 * np.pi * eta_i * q0["R"] * np.pi / 4 * U
    out = dict(species=sp, wt=wt, amp=amp, state=state, arg=arg, eta=q["eta"],
               theta0=q["theta0"], p_eff_used=p.p_eff, f_beat=q["f_beat"], U_slip=U,
               U_swim=np.pi / 4 * U, F_model=F, F_exp=q["F_exp"], U_exp=q["U_exp"],
               f_exp=q["f_exp"], f_formula_meas=f_formula(q["eta"], q["omega_exp"], q0["L"]))
    print(f"{sp:10s} {amp} wt={wt:<4} {state:11s} {arg:>2}  θ0={np.degrees(q['theta0']):5.1f}°  "
          f"U={U:7.1f}  F={F:7.1f} pN", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", nargs="*", default=["tip", "eff"])
    a = ap.parse_args()
    tasks = []
    for sp in ("Volvox", "Pleodorina"):
        for amp in a.amps:
            for wt in WTS:
                tasks.append((sp, wt, amp, "independent", 0))
                for s in range(2):
                    tasks.append((sp, wt, amp, "random2D", s))
                for w in (0, 2, -2):
                    tasks.append((sp, wt, amp, "wave", w))
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(rows, open(os.path.join(PRIV, "p4_viscosity.json"), "w"), indent=1)
    print(f"saved ({(time.time()-t0)/60:.1f} min)")
