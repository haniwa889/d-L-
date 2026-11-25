import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from model import *
from carpet import *
def job(t):
    dL, ns, w = t
    p=Params(N_seg=ns,beat_mode='fixed',steps_per_cycle=40,n_cycles=1,transient=0,d=dL*22)
    t0=time.time(); n=24
    r=run_carpet(p,n,phi0=np.mod(-np.arange(n)*TWO_PI*w/n,TWO_PI))
    a=analyze_carpet(r)
    return (dL,ns,w,a['f_eff'],a['F_net'],a['U_slip'],a['power_per_flag'],time.time()-t0)
if __name__=="__main__":
    tasks=[(dL,ns,w) for dL in (0.41,1.05) for w in (1,2,6) for ns in (8,10,12,16)]
    with Pool(4) as pool: res=pool.map(job,tasks,chunksize=1)
    for r in sorted(res): print("dL=%.2f Nseg=%2d winding=%d  f_eff=%.2f F_net=%.2f U_slip=%.1f P=%.0f (%.0fs)"%r)
