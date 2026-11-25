"""S7: linear stability of metachronal waves on the periodic carpet (beat_mode = "load").

For identical oscillators on a periodic lattice every travelling wave with integer
winding w is an exact periodic solution.  We start from wave w plus a small random
phase perturbation (rms 0.1 rad) and follow the lags between successive crossing
times of φ = 2πm by neighbouring columns (Poincaré section).  For a perfect wave
these lags are all equal; the spread σ_m (std over columns) grows (unstable) or
decays (stable) exponentially: σ_m ~ exp(λ m).  λ (per beat) < 0 → stable wave.

Output: studies/out/s7_stability.json
"""
from common import OUT
import argparse, json, time
from multiprocessing import Pool
import numpy as np
from model import Params, TWO_PI
from carpet import run_carpet, Carpet


def crossing_times(t, phi):
    """times at which each column's (unwrapped) phase passes 2π m."""
    out = []
    for j in range(phi.shape[1]):
        ph = phi[:, j]
        m0, m1 = int(np.ceil(ph[0] / TWO_PI)), int(np.floor(ph[-1] / TWO_PI))
        levels = TWO_PI * np.arange(m0, m1 + 1)
        out.append(np.interp(levels, ph, t))
    m = min(len(x) for x in out)
    return np.array([x[:m] for x in out]).T            # (cycles, n)


def job(task):
    dL, eta, w, n, ns, cyc, spc, amp, seed = task
    L = 22.0
    p = Params(L=L, d=dL * L, N_seg=ns, eta=eta, beat_mode="load", steps_per_cycle=spc,
               n_cycles=cyc, transient=0.0)
    rng = np.random.default_rng(seed)
    pert = rng.normal(0, amp, n); pert -= pert.mean()
    phi0 = -TWO_PI * w * np.arange(n) / n + pert
    phi0 -= phi0.min()                                   # all phases ≥ 0 at t=0
    t0 = time.time()
    res = run_carpet(p, n, phi0=phi0)
    T = crossing_times(res["t"], res["phi"])             # (m, n)
    # lags between neighbours, measured as time differences (periodic wrap)
    lag = np.diff(np.c_[T, T[:, :1]], axis=1)
    per = np.median(np.diff(T, axis=0))
    lag = (lag + per / 2) % per - per / 2                # wrap into one period
    sig = lag.std(1) / per                               # spread in units of the period
    m = np.arange(len(sig))
    ok = (sig > 1e-6) & (sig < 0.05)                     # linear regime only
    ok[:2] = False
    lam = float(np.polyfit(m[ok], np.log(sig[ok]), 1)[0]) if ok.sum() >= 4 else np.nan
    out = dict(dL=dL, eta=eta, w=w, dphi_deg=-360.0 * w / n, n=n, lam=lam,
               sigma=sig.tolist(), period=float(per), freq=float(1 / per),
               U_slip_mean=float(np.mean(res["Mz"][len(res["Mz"]) // 2:]) /
                                 (eta * 1e-3 * res["A_cell"])),
               elapsed=time.time() - t0)
    print(f"d/L={dL:.2f} η={eta:<5g} Δφ={out['dphi_deg']:+5.0f}°  λ={lam:+.4f}/beat  "
          f"f={out['freq']:.1f} Hz ({out['elapsed']/60:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=float, default=40)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--N_seg", type=int, default=10)
    ap.add_argument("--spc", type=int, default=40)
    ap.add_argument("--amp", type=float, default=0.1)
    ap.add_argument("--dLs", type=float, nargs="*", default=[0.3, 0.41, 0.6, 0.8, 1.05, 1.3])
    ap.add_argument("--etas", type=float, nargs="*", default=[0.84, 8.61])
    ap.add_argument("--ws", type=int, nargs="*", default=[0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6])
    ap.add_argument("--tag", default="s7_stability")
    a = ap.parse_args()
    for dL in a.dLs:
        Carpet(Params(L=22.0, d=dL * 22.0, N_seg=a.N_seg), a.n)
    tasks = [(dL, eta, w, a.n, a.N_seg, a.cycles, a.spc, a.amp, 1)
             for eta in a.etas for dL in a.dLs for w in a.ws]
    with Pool(4) as pool:
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(rows, open(f"{OUT}/{a.tag}.json", "w"), indent=1)
    print("saved")
