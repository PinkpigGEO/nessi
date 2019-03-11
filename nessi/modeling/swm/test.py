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
modvpe = cmodext(modvp, n1, n2, npml)
modvse = cmodext(modvs, n1, n2, npml)
modroe = cmodext(modro, n1, n2, npml)

# Convert models
mu, lbd, lbdmu = cmodlame(modvpe, modvse, modroe)
bux, buz = cmodbuo(modroe)
bux, buz = cmodbuo(modroe)

# Calculate PML
isurf = 1
ppml = 2
apml = 800.
pmlx0, pmlx1, pmlz0, pmlz1 = cmodpml(n1, n2, dh, isurf, npml, ppml, apml)

# Source
sigma = -1
f0 = 15.
t0 = 0.1
acq = np.zeros((2, 2), dtype=np.float32)
acq[0, 0] = 10.; acq[0, 1] = 10.
acq[0, 0] = 40.; acq[0, 1] = 40.
tsrc = cricker(nt, dt, f0, t0)
gsrc = csrcspread(n1, n2, dh, npml, xs, zs, sigma)
srctype = 1
isurf = 0

# Marching
ux, uz = evolution(mu, lbd, lbdmu, bux, buz, pmlx0, pmlx1, pmlz0, pmlz1, npml, isurf, srctype, tsrc, gsrc, dh, nt, dt)

plt.subplot(121)
plt.imshow(ux, aspect='auto', cmap='gray')
plt.colorbar()
plt.subplot(122)
plt.imshow(uz, aspect='auto', cmap='gray')
plt.colorbar()
plt.show()
