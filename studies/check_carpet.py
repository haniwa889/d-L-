import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from model import *
from carpet import *
def job(t):
    dL, n, Rn, tail, ns = t
    p=Params(N_seg=ns,beat_mode='fixed',steps_per_cycle=40,n_cycles=1,transient=0,d=dL*22)
    t0=time.time()
    r=run_carpet(p,n,phi0=np.mod(-np.arange(n)*TWO_PI/n,TWO_PI),R_near=Rn*22,tail=tail)
    a=analyze_carpet(r)
    return (dL,n,Rn,tail,ns,a['f_eff'],a['F_net'],a['U_slip'],time.time()-t0)
if __name__=="__main__":
    tasks=[(dL,12,Rn,tail,12) for dL in (0.41,1.05) for Rn,tail in ((2,False),(3,False),(5,False),(2,True),(3,True),(5,True))]
    tasks+=[(dL,12,3,True,ns) for dL in (0.41,1.05) for ns in (16,20)]
    with Pool(4) as pool: res=pool.map(job,tasks,chunksize=1)
    for r in res: print("dL=%.2f n=%d Rnear=%dL tail=%d Nseg=%d  f_eff=%.2f F_net=%.3f U_slip=%.2f  (%.0fs)"%r)
