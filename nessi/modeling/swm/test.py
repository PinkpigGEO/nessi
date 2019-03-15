import numpy as np
import matplotlib.pyplot as plt
from cmodel import cmodext, cmodlame, cmodbuo, cmodpml
from cacquisition import cricker, csrcspread
from cmarching import evolution
import time

# generate pseudo-model
n1 = 101
n2 = 101
dh = 0.5
npml = 10
nt = 2500
dt = 0.0001
xs = 25.
zs = 25.

# Create models
modvp = np.ones((n1, n2), dtype=np.float32)
modvs = np.ones((n1, n2), dtype=np.float32)
modro = np.ones((n1, n2), dtype=np.float32)
modvp[:, :] = 1000.
modvs[:, :] = 750.
modro[:, :] = 1500.

# Extend models
tstart = time.time()
modvpe = cmodext(n1, n2, npml, modvp)
modvse = cmodext(n1, n2, npml, modvs)
modroe = cmodext(n1, n2, npml, modro) 
tend = time.time()
print('extend models', tend-tstart)

# Convert models
n1e = n1+2*npml
n2e = n2+2*npml
tstart = time.time()
bux, buz = cmodbuo(n1e, n2e, modroe)
mu, lbd, lbdmu = cmodlame(n1e, n2e, modvpe, modvse, modroe)
tend = time.time()
print('convert models', tend-tstart)
