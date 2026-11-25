"""S2: 2-D periodic carpet with prescribed (fixed) kinematics.
d/L × metachronal lag Δφ (n = 24 flagella per wavelength, windings w → Δφ = −15° w).
Lateral rows (spacing dy = d) beat in phase.  Output: studies/out/s2_carpet.json"""
import json, os, sys, time
from multiprocessing import Pool
from common import carpet_point, OUT
import numpy as np

DLS = [0.3, 0.35, 0.41, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05, 1.2, 1.35, 1.5]
WS = [0, 1, 2, 3, 4, 6, 8, 12, -1, -2, -3, -4, -6, -8]


def job(dL):
    t0 = time.time()
    path = f"{OUT}/s2_parts/dL{dL:.3f}.json"
    if os.path.exists(path):
        return json.load(open(path))
    rows = [carpet_point(dL, w) for w in WS]
    os.makedirs(f"{OUT}/s2_parts", exist_ok=True)
    json.dump(rows, open(path, "w"))
    print(f"d/L={dL:.2f} done in {time.time()-t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(4) as pool:
        res = pool.map(job, sorted(DLS, key=lambda x: x), chunksize=1)
    rows = [r for rr in res for r in rr]
    json.dump(rows, open(f"{OUT}/s2_carpet.json", "w"), indent=1)
    print(f"{'d/L':>5s} {'Δφ':>6s} {'eps_f':>6s} {'eps_pump':>8s} {'U_slip':>8s} {'P/P1':>6s}")
    for r in sorted(rows, key=lambda r: (r["dL"], r["dphi_deg"])):
        print(f"{r['dL']:5.2f} {r['dphi_deg']:6.1f} {r['eps_f']:6.3f} {r['eps_pump']:8.3f} "
              f"{r['U_slip']:8.1f} {r['power_ratio']:6.3f}")
