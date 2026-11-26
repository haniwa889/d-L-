"""Carpet2D with lanes in phase must reproduce Carpet (lane-synchronous)."""
from common import *
import time
import numpy as np
from model import Params, TWO_PI
from carpet import Carpet
from carpet2d import Carpet2D
p = Params(N_seg=10, d=9.0, L=22.0)
n = 12
phi = np.mod(-np.arange(n) * TWO_PI / n + 0.3, TWO_PI)
c1 = Carpet(p, n)
X, Tm = c1.points(phi)
M1 = c1.mobility(X)
om = TWO_PI * 30
F1 = np.linalg.solve(M1, Tm @ np.full(n, om)).reshape(-1, 2)
print("Carpet   U_slip=%.2f  Fx=%s" % ((F1[:, 0] * X[:, 2]).sum() / (p.eta_int * c1.A_cell),
                                       np.round(F1[:, 0].reshape(n, -1).sum(1)[:4], 2)))
for ny in (1, 2):
    t = time.time()
    c2 = Carpet2D(p, n, ny)
    print("setup", time.time() - t)
    ph2 = np.repeat(phi, ny)                         # lanes in phase
    X2, Tm2 = c2.points(ph2)
    t = time.time(); M2 = c2.mobility(X2); print("mobility", time.time() - t)
    F2 = np.linalg.solve(M2, Tm2 @ np.full(c2.n_osc, om)).reshape(-1, 2)
    Us = (F2[:, 0] * X2[:, 2]).sum() / (p.eta_int * c2.A_cell)
    Fx = F2[:, 0].reshape(c2.nf, -1).sum(1)
    print(f"Carpet2D ny={ny} U_slip={Us:.2f}  Fx(lane0)={np.round(Fx[::ny][:4], 2)}")
