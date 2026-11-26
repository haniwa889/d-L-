"""P5: does coordination (and its effect on thrust) disappear when viscosity slows the beat?

Hypothesis to test (user): η↑ → ω↓ → weaker flow on neighbours → coordination lost →
thrust "boost" lost (Volvox), while Pleodorina is hardly coordinated anyway.

Model: species geometry (d, L) and effective-stroke-amplitude waveform, phases driven by
the motor law dφ/dt = ω0 (1 − T/T_stall) (T_stall: single flagellum beats at the
measured water frequency; ω0 = 2π·50 Hz) plus phase noise of diffusion constant D.
Random initial phases, lane-synchronous carpet (n = 24 per period).
Measured: beat frequency, coordination (coherence of neighbour lags, R), and the slip
velocity per beat relative to the uncoordinated (random-phase) value in water.

Output → studies/private/p5_noise.json
"""
from common import OUT
import json, os, time, argparse
from multiprocessing import Pool
import numpy as np
from model import Params, TWO_PI, ETA_UNIT
from carpet import run_carpet, analyze_carpet, Carpet
from exp_params import species_params, PRIV


def job(t):
    sp, eta, D, seed, cyc = t
    q = species_params(sp, 0.0, "eff")
    p = Params(L=q["L"], d=q["d"], N_seg=10, theta0=q["theta0"], p_eff=q["p_eff"],
               eta=eta, eta_water=0.84, f_water=q["f_beat"], f_free=50.0, beat_mode="load",
               steps_per_cycle=40, n_cycles=cyc, transient=0.5, seed=seed, init_phase="random")
    t0 = time.time()
    res = run_carpet(p, 24, noise_D=D, noise_seed=seed)
    a = analyze_carpet(res, transient=0.5)
    out = dict(species=sp, eta=eta, D=D, seed=seed, **a, elapsed=time.time() - t0)
    out["U_per_beat"] = a["U_slip"] / a["freq"]
    print(f"{sp:10s} η={eta:<5} D={D:<4} seed={seed}  f={a['freq']:.1f} Hz  U={a['U_slip']:.0f}  "
          f"coh={a['dphi_coh']:.2f} Δφ={np.degrees(a['dphi_mean']):+.0f}° R={a['R_sync']:.2f} "
          f"({out['elapsed']/60:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=float, default=200)
    ap.add_argument("--etas", type=float, nargs="*", default=[0.84, 1.77, 3.26])
    ap.add_argument("--Ds", type=float, nargs="*", default=[0.0, 1.0, 4.0])
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    tasks = [(sp, e, D, s, a.cycles) for s in range(a.seeds) for D in a.Ds for e in a.etas
             for sp in ("Volvox", "Pleodorina")]
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(rows, open(os.path.join(PRIV, "p5_noise.json"), "w"), indent=1)
    print(f"saved ({(time.time()-t0)/60:.1f} min)")
