"""S1: dependence of the per-flagellum effective-stroke force on the row length N
(finite row, fixed kinematics).  Inner flagella = middle half of the row."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from model import Params, run, analyze

base = Params(N_seg=12, beat_mode="fixed", steps_per_cycle=40, n_cycles=1, transient=0)

def job(t):
    N, dL, dp = t
    p = base.copy(N=N, d=dL * base.L, init_phase="metachronal", dphi_mc=np.radians(dp))
    a = analyze(run(p, verbose=False, store_frames=False))
    f = a["f_i"]; q = N // 4
    return dict(N=N, dL=dL, dphi=dp, f_all=float(f.mean()), f_inner=float(f[q:N - q].mean()),
                f_center=float(f[N // 2 - 1:N // 2 + 1].mean()), net=float(a["Fx_cycle_mean"] / N),
                f_i=f.tolist())

if __name__ == "__main__":
    f1 = analyze(run(base.copy(N=1), verbose=False, store_frames=False))["f_mean"]
    tasks = [(N, dL, dp) for N in (10, 20, 40) for dL in (0.41, 0.7, 1.05, 1.5)
             for dp in (0, -15, -30, -60, -90)]
    tasks.sort(key=lambda t: -t[0])
    with Pool(4) as pool:
        res = pool.map(job, tasks, chunksize=1)
    os.makedirs("studies/out", exist_ok=True)
    json.dump(dict(f_single=f1, res=res), open("studies/out/s1_row_size.json", "w"), indent=1)
    print("f_single", f1)
    for r in sorted(res, key=lambda r: (r["dL"], r["dphi"], r["N"])):
        print(f"dL={r['dL']:.2f} dphi={r['dphi']:4d} N={r['N']:3d}  inner={r['f_inner']/f1:.3f} "
              f"center={r['f_center']/f1:.3f} all={r['f_all']/f1:.3f}")
