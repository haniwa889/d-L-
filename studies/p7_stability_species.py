"""P7: stability of metachronal waves with each species' own geometry and waveform,
within the viscosity range of the experiments (η/η_w ≤ ~3.9).

Question: when viscosity rises, does the coordinated (wave) state of the dense Volvox
carpet stop being stable, while the sparse Pleodorina carpet is hardly affected?

Same method as S7 (exact travelling wave + 0.1 rad perturbation, growth rate λ of the
spread of neighbour lags per beat), but with species d, L and the effective-stroke
amplitude waveform.  Motor: linear torque–speed law, single flagellum beats at the
measured water frequency, ω0 = 2π·f_free (f_free = 50 Hz; 100 Hz as a sensitivity check).  The waveform is kept at the water value so
that only viscosity (through the load on the motor) changes.

Output (contains the user's parameters) → studies/private/p7_stability.json
"""
from common import OUT
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from model import Params, TWO_PI, motor_params, phase_map
from carpet import run_carpet, Carpet
from s7_stability import crossing_times
from exp_params import species_params, PRIV


def make_params(sp, eta, f_free, cyc, spc, ns):
    q = species_params(sp, 0.0, "eff")
    return Params(L=q["L"], d=q["d"], N_seg=ns, theta0=q["theta0"], p_eff=q["p_eff"],
                  eta=eta, eta_water=0.84, f_water=q["f_beat"], f_free=f_free,
                  beat_mode="load", steps_per_cycle=spc, n_cycles=cyc, transient=0.0)


def job(task):
    sp, eta, w, f_free, n, ns, cyc, spc, amp, seed = task
    p = make_params(sp, eta, f_free, cyc, spc, ns)
    rng = np.random.default_rng(seed)
    pert = rng.normal(0, amp, n); pert -= pert.mean()
    om, Ts = motor_params(p)
    g, th = phase_map(p, om, Ts)
    theta0 = np.mod(-TWO_PI * w * np.arange(n) / n, TWO_PI)
    phi0 = np.interp(theta0, th, g) + pert
    phi0 -= phi0.min()
    t0 = time.time()
    res = run_carpet(p, n, phi0=phi0)
    T = crossing_times(res["t"], res["phi"])
    lag = np.diff(np.c_[T, T[:, :1]], axis=1)
    per = np.median(np.diff(T, axis=0))
    lag = (lag + per / 2) % per - per / 2
    sig = lag.std(1) / per
    m = np.arange(len(sig))
    ok = (sig > 1e-7) & (sig < 0.2) & (m >= len(sig) // 4)
    lam = float(np.polyfit(m[ok], np.log(sig[ok]), 1)[0]) if ok.sum() >= 6 else np.nan
    # early growth (first beats) for strongly unstable cases
    k = min(8, len(sig) - 1)
    lam0 = float(np.polyfit(m[1:k + 1], np.log(np.maximum(sig[1:k + 1], 1e-9)), 1)[0])
    half = len(res["Mz"]) // 2
    out = dict(species=sp, eta=eta, w=w, f_free=f_free, dphi_deg=-360.0 * w / n, n=n,
               lam=lam, lam0=lam0, sigma=sig.tolist(), period=float(per), freq=float(1 / per),
               U_slip_mean=float(np.mean(res["Mz"][half:]) / (eta * 1e-3 * res["A_cell"])),
               load_frac=float(1 - (1 / per) / f_free), elapsed=time.time() - t0)
    print(f"{sp:10s} η={eta:<5g} f_free={f_free:<4g} Δφ={out['dphi_deg']:+5.0f}°  "
          f"λ={lam:+.4f} λ0={lam0:+.3f}/beat  f={out['freq']:.1f} Hz "
          f"({out['elapsed']/60:.1f} min)", flush=True)
    return out


def build(sp):
    p = make_params(sp, 0.84, 50.0, 1, 40, 10)
    Carpet(p, 24)
    return sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=float, default=60)
    ap.add_argument("--etas", type=float, nargs="*", default=[0.84, 1.77, 3.26])
    ap.add_argument("--ffree", type=float, nargs="*", default=[50.0])
    ap.add_argument("--ws", type=int, nargs="*", default=[0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6])
    ap.add_argument("--tag", default="p7_stability")
    a = ap.parse_args()
    with Pool(2) as pool:
        list(pool.map(build, ["Volvox", "Pleodorina"]))
    tasks = [(sp, eta, w, ff, 24, 10, a.cycles, 40, 0.1, 1)
             for ff in a.ffree for eta in a.etas for sp in ("Volvox", "Pleodorina") for w in a.ws]
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(rows, open(os.path.join(PRIV, f"{a.tag}.json"), "w"), indent=1)
    print(f"saved ({(time.time()-t0)/60:.1f} min)")
