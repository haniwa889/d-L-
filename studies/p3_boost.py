"""P3: does coordination give extra thrust?  (species-specific geometry and waveform)

For each species (own d, L, R, N, measured waveform), prescribed beat (fixed kinematics,
measured beat frequency), compare the carpet's slip velocity / colony force for
  independent  : flagella do not interact (sum of isolated flagella)
  random2D     : random phases in BOTH directions (no coordination)   [carpet2d]
  random_lanes : random along the beat, lateral rows in phase           [carpet]
  sync         : all flagella in phase
  wave w       : metachronal wave, n = 24 per wavelength, Δφ = −15° w (lanes in phase)
Optional: two flagella per cell (pair_sep).  Colony force F = 6πηR (π/4) U_slip,
ε_model = F / (N f_formula) with the user's f definition.

Outputs (contain the user's parameters) → studies/private/p3_boost.json
"""
from common import OUT, single_ref
import json, os, time, argparse
from multiprocessing import Pool
import numpy as np
from model import Params, TWO_PI, ETA_UNIT, System, in_effective
from carpet import run_carpet, analyze_carpet, Carpet
from carpet2d import run_carpet2d, analyze_carpet2d, Carpet2D
from exp_params import species_params, f_formula, PRIV

NS = 10
SPC = 40


def make_params(sp, wt, amp):
    q = species_params(sp, wt, amp)
    p = Params(L=q["L"], d=q["d"], N_seg=NS, theta0=q["theta0"], p_eff=q["p_eff"],
               f_fixed=q["f_beat"], eta=q["eta"], beat_mode="fixed", steps_per_cycle=SPC,
               n_cycles=1.0, transient=0.0)
    return q, p


def isolated(p):
    s = System(p.copy(N=1))
    om = TWO_PI * p.f_fixed
    ph = np.linspace(0, TWO_PI, 80, endpoint=False)
    Fx, Mz = [], []
    for x in ph:
        r = s.solve(np.array([x]), omega_fixed=om)
        Fx.append(r["F"][:, 0].sum()); Mz.append(np.sum(r["F"][:, 0] * r["X"][:, 2]))
    Fx = np.array(Fx)
    return dict(f_eff=float(Fx[in_effective(ph, p)].mean()), F_net=float(Fx.mean()),
                M=float(np.mean(Mz)))


def finish(q, p, iso, state, U_slip, f_eff, npc=1, extra=None):
    eta_i = q["eta"] * ETA_UNIT
    U_swim = np.pi / 4 * U_slip
    F = 6 * np.pi * eta_i * q["R"] * U_swim
    omega_m = 2 * q["theta0"] / q["t_eff"]
    ff = f_formula(q["eta"], omega_m, q["L"])
    A_flag = q["d"] ** 2 / npc
    out = dict(species=q["species"], wt=q["wt"], state=state, npc=npc, U_slip=U_slip,
               U_swim=U_swim, F_model=F, f_formula_model=ff, f_eff_model=f_eff,
               eps_model=F / (q["N"] * ff),
               eps_pump=U_slip * eta_i * A_flag / iso["M"],
               U_indep=iso["M"] / (eta_i * A_flag))
    if extra:
        out.update(extra)
    return out


def job(t):
    sp, wt, amp, state, arg, npc = t
    q, p = make_params(sp, wt, amp)
    iso = isolated(p)
    if state == "independent":
        A_flag = q["d"] ** 2 / npc
        r = finish(q, p, iso, state, iso["M"] / (q["eta"] * ETA_UNIT * A_flag), iso["f_eff"], npc)
    elif state in ("random2D", "sync2D"):
        nx, ny = 8, 6
        rng = np.random.default_rng(arg)
        ph = rng.uniform(0, TWO_PI, nx * ny) if state == "random2D" else np.zeros(nx * ny)
        res = run_carpet2d(p, nx, ny, ph, pair_sep=(2.0 if npc == 2 else 0.0))
        a = analyze_carpet2d(res)
        r = finish(q, p, iso, state, a["U_slip"], a["f_eff"], npc, dict(seed=arg))
    else:  # lane-synchronous carpet: random_lanes / wave
        n = 24
        if state == "random_lanes":
            ph = np.random.default_rng(arg).uniform(0, TWO_PI, n)
        else:
            ph = np.mod(-np.arange(n) * TWO_PI * arg / n, TWO_PI)
        res = run_carpet(p, n, phi0=ph)
        a = analyze_carpet(res, transient=0.0)
        r = finish(q, p, iso, state, a["U_slip"], a["f_eff"], 1,
                   dict(w=arg if state == "wave" else None, seed=arg if state != "wave" else None,
                        dphi=-360.0 * arg / n if state == "wave" else None))
    r["amp"] = amp
    print(f"{sp:10s} {amp} {state:12s} {str(arg):>3s} npc={npc}  U_slip={r['U_slip']:8.1f}  "
          f"F={r['F_model']:8.1f} pN  eps={r['eps_model']:.4f}  eps_pump={r['eps_pump']:.3f}", flush=True)
    return r


def build(key):
    sp, amp, npc = key
    q, p = make_params(sp, 0.0, amp)
    Carpet(p, 24)
    Carpet2D(p, 8, 6, pair_sep=(2.0 if npc == 2 else 0.0))
    return key


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", nargs="*", default=["tip", "eff"])
    a = ap.parse_args()
    tasks = []
    for sp in ("Volvox", "Pleodorina"):
        for amp in a.amps:
            tasks.append((sp, 0.0, amp, "independent", 0, 1))
            tasks.append((sp, 0.0, amp, "independent", 0, 2))
            for s in range(4):
                tasks.append((sp, 0.0, amp, "random2D", s, 1))
            tasks.append((sp, 0.0, amp, "sync2D", 0, 1))
            for s in range(2):
                tasks.append((sp, 0.0, amp, "random2D", s, 2))
            tasks.append((sp, 0.0, amp, "sync2D", 0, 2))
            for s in range(3):
                tasks.append((sp, 0.0, amp, "random_lanes", s, 1))
            for w in (0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6):
                tasks.append((sp, 0.0, amp, "wave", w, 1))
    t0 = time.time()
    keys = sorted({(sp, amp, npc) for sp, _, amp, _, _, npc in tasks})
    with Pool(4) as pool:
        for k in pool.imap_unordered(build, keys):
            print("tables ready", k, f"{time.time()-t0:.0f}s", flush=True)
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(rows, open(os.path.join(PRIV, "p3_boost.json"), "w"), indent=1)
    print(f"saved ({(time.time()-t0)/60:.1f} min)")
