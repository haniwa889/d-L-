import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from model import *
from carpet import Carpet, tail_table
def brute(c, X, XB=1000.0):
    p=c.p; K=X.shape[0]
    G=np.zeros((K,K,3,3))
    owner=c.owner; same=owner[:,None]==owner[None,:]
    for mx in range(-int(XB//c.P)-1, int(XB//c.P)+2):
        for my in range(-int(XB//c.dy), int(XB//c.dy)+1):
            o=np.array([mx*c.P,my*c.dy,0.])
            if abs(o[0])>XB+c.P: continue
            eps=np.where(same&(mx==0)&(my==0),c.eps,c.eps_x)
            G+=kernel(X,X+o,eps)
    zt,Tt=tail_table(float(XB+0.5*c.P), round(1.05*p.L,3), nz=24)
    from carpet import _interp2
    Tz=_interp2(zt,Tt,X[:,2],X[:,2])/(c.P*c.dy)
    G[...,0,0]+=Tz[...,0]; G[...,2,2]+=Tz[...,1]
    Gr=G[:,:,[0,2]][:,:,:,[0,2]]
    return Gr.transpose(0,2,1,3).reshape(2*K,2*K)/(8*np.pi*p.eta_int)
for d,n in [(9.0,12),(23.1,6)]:
    p=Params(N_seg=10,beat_mode='fixed',d=d)
    c=Carpet(p,n)
    phi=np.mod(-np.arange(n)*TWO_PI/n+0.7,TWO_PI)
    X,Tm=c.points(phi)
    M1=c.mobility(X); t=time.time(); M2=brute(c,X); print('brute',time.time()-t,'s')
    om=TWO_PI*30
    for name,M in (('table',M1),('brute',M2)):
        F=np.linalg.solve(M,Tm@np.full(n,om)).reshape(-1,2)
        Mz=np.sum(F[:,0]*X[:,2]); Us=Mz/(p.eta_int*c.A_cell)
        Fx=F[:,0].reshape(n,-1).sum(1)
        print(f"d={d} n={n} {name}: Fx per flagellum={np.round(Fx,2)}  U_slip={Us:.2f}")
    print('rel matrix diff', np.abs(M1-M2).max()/np.abs(M2).max())
