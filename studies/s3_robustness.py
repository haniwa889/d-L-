"""S3: is d/L the controlling variable, and are the trends robust?
 (a) collapse: same d/L with L = 15, 22, 30 µm
 (b) anisotropy: spacing along the beat (dx) vs lateral (dy) varied separately
 (c) beat amplitude θ0 (envelope width 2 L sinθ0) — does d/(L sinθ0) collapse better?
 (d) waveform: fraction of effective stroke p_eff, bend width
 (e) resolution: N_seg = 20 at key points
Output: studies/out/s3_robustness.json"""
import json, time
from multiprocessing import Pool
from common import carpet_point, OUT
from model import Params
from carpet import Carpet

W4 = [0, 1, 3, 6]          # Δφ = 0, −15, −45, −90°
W3 = [0, 1, 3]
tasks = []
for L in (15.0, 30.0):
    for dL in (0.41, 0.7, 1.05):
        for w in W4:
            tasks.append(("a", dict(dL=dL, w=w, L=L)))
for dxL, dyL in ((0.41, 1.05), (1.05, 0.41), (0.41, 3.0), (1.05, 3.0), (0.41, 0.41), (1.05, 1.05)):
    for w in W4:
        tasks.append(("b", dict(dL=dxL, w=w, dy_factor=dyL / dxL)))
for th in (0.8, 1.0, 1.35):
    for dL in (0.3, 0.41, 0.6, 0.8, 1.05, 1.5):
        for w in W3:
            tasks.append(("c", dict(dL=dL, w=w, theta0=th)))
for kw in (dict(p_eff=0.25), dict(p_eff=0.5), dict(bend_width=0.15), dict(bend_width=0.4)):
    for dL in (0.41, 1.05):
        for w in W3:
            tasks.append(("d", dict(dL=dL, w=w, **kw)))
for dL in (0.41, 1.05):
    for w in W4:
        tasks.append(("e", dict(dL=dL, w=w, N_seg=20)))


def table_key(kw):
    L = kw.get("L", 22.0); d = kw["dL"] * L
    return (L, d, kw.get("dy_factor", 1.0) * d)


def build(key):
    L, d, dy = key
    Carpet(Params(L=L, d=d, N_seg=12), 24, dy=dy)
    return key


def job(t):
    grp, kw = t
    r = carpet_point(**kw)
    r["group"] = grp
    return r


if __name__ == "__main__":
    t0 = time.time()
    keys = sorted({table_key(kw) for _, kw in tasks})
    with Pool(4) as pool:
        for k in pool.imap_unordered(build, keys):
            print("tables", k, f"{time.time()-t0:.0f}s", flush=True)
        rows = pool.map(job, tasks, chunksize=1)
    json.dump(rows, open(f"{OUT}/s3_robustness.json", "w"), indent=1)
    print(f"done {len(rows)} runs in {(time.time()-t0)/60:.1f} min")
