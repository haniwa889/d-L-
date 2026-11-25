"""S5: self-organisation of phases on the periodic carpet (beat_mode = "load").

Each column of flagella (n = 24 per period, lateral rows in phase) is a phase
oscillator  dφ/dt = ω0 (1 − T/T_stall)  driven by the full hydrodynamic load.
Random initial phases; several seeds; d/L × η.  Measures the emergent pattern:
winding number w (wavelength λ = n d / |w|, neighbour lag Δφ ≈ −360° w / n),
coherence, R_sync, slip velocity, beat frequency.

  python s5_carpet_load.py [--cycles 150] [--seeds 3] [--quick]
Output: studies/out/s5_load.json  (+ phase time series in s5_load_series.npz)
"""
from common import OUT, single_ref          # sets OMP/BLAS threads = 1 before numpy import
import argparse, json, time
from multiprocessing import Pool
import numpy as np
from model import Params, TWO_PI, ETA_UNIT, to_theta, wrap
from carpet import run_carpet, analyze_carpet, Carpet


def job(t):
    dL, eta, seed, n, ns, cyc, spc = t
    L = 22.0
    p = Params(L=L, d=dL * L, N_seg=ns, eta=eta, beat_mode="load", steps_per_cycle=spc,
               n_cycles=cyc, transient=0.6, seed=seed, init_phase="random")
    t0 = time.time()
    res = run_carpet(p, n)
    a = analyze_carpet(res, transient=0.6)
    th = to_theta(res["phi"], res["phase_map"])
    dth = wrap(np.diff(np.c_[th, th[:, :1]], axis=1))
    wind_t = np.round(dth.sum(1) / TWO_PI)                 # winding number vs time
    k = max(1, len(res["t"]) // 600)                       # down-sample for storage
    ref = single_ref(L, ns, eta=eta, f=a["freq"])
    A_flag = res["A_cell"] / n
    out = dict(dL=dL, eta=eta, seed=seed, n=n, N_seg=ns, cycles=cyc, **a,
               winding_final=float(wind_t[-1]),
               winding_frac_last=float(np.mean(wind_t[int(0.6 * len(wind_t)):] == wind_t[-1])),
               dphi_from_winding=float(-360.0 * wind_t[-1] / n),
               eps_pump=float(a["U_slip"] * eta * ETA_UNIT * A_flag / ref["M1"]),
               T_stall=float(res["T_stall"]), elapsed=time.time() - t0)
    series = dict(t=res["t"][::k], theta=th[::k], wind=wind_t[::k])
    print(f"d/L={dL:.2f} η={eta:<5g} seed={seed}  w={out['winding_final']:+.0f} "
          f"(Δφ≈{out['dphi_from_winding']:+.0f}°) coh={a['dphi_coh']:.2f} R={a['R_sync']:.2f} "
          f"f={a['freq']:.1f}Hz U={a['U_slip']:.0f}µm/s  ({out['elapsed']/60:.1f} min)", flush=True)
    return out, series


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=float, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--N_seg", type=int, default=10)
    ap.add_argument("--spc", type=int, default=40)
    ap.add_argument("--dLs", type=float, nargs="*", default=[0.3, 0.41, 0.6, 0.8, 1.05, 1.3])
    ap.add_argument("--etas", type=float, nargs="*", default=[0.84, 8.61])
    ap.add_argument("--tag", default="s5_load")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    # build kernel tables once (cached on disk) before forking
    for dL in a.dLs:
        Carpet(Params(L=22.0, d=dL * 22.0, N_seg=a.N_seg), a.n)
    tasks = [(dL, eta, s, a.n, a.N_seg, a.cycles, a.spc)
             for s in range(a.seeds) for eta in a.etas for dL in a.dLs]
    with Pool(a.workers) as pool:
        res = pool.map(job, tasks, chunksize=1)
    rows = [r for r, _ in res]
    json.dump(rows, open(f"{OUT}/{a.tag}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/{a.tag}_series.npz",
                        **{f"{r['dL']}_{r['eta']}_{r['seed']}_{k}": v
                           for (r, s) in res for k, v in s.items()})
    print("saved", f"{OUT}/{a.tag}.json")
