import numpy as np
import matplotlib.pyplot as plt

from nessi.modeling.swm import cmodext, cmodlame, cmodbuo, cmodpml
from nessi.modeling.swm import cacqpos, cricker, csrcspread
from nessi.modeling.swm import evolution

# Model parameters
n1 = 401
n2 = 401
dh = 0.5
npml = 20
ppml = 3
apml = 800.
isurf = 1
vp = 2000.
vs = 1000.
ro = 2000.

# Create models
n1e = n1+2*npml
n2e = n2+2*npml
modvp = np.zeros((n1, n2), dtype=np.float32)
modvs = np.zeros((n1, n2), dtype=np.float32)
modro = np.zeros((n1, n2), dtype=np.float32)
modvp[:, :] = vp
modvs[:, :] = vs
modro[:, :] = ro

# Extend models
modvpe = cmodext(n1, n2, npml, modvp)
modvse = cmodext(n1, n2, npml, modvs)
modroe = cmodext(n1, n2, npml, modro)

# Elastic parameter models
bux, buz = cmodbuo(n1e, n2e, modroe)
mu, lbd, lbdmu = cmodlame(n1e, n2e, modvpe, modvse, modroe)

#PML
pmlx0, pmlx1, pmlz0, pmlz1 = cmodpml(n1, n2, dh, isurf, npml, ppml, vp) #np.amax(modvpe))

# Acquisition
acq = np.zeros((2, 2), dtype=np.float32)
acq[:, :] = 10.
acqui = cacqpos(n1, n2, dh, npml, acq)

# Source
xs = 100.
zs = 100.
nt = 2500
dt = 0.0001
t0 = 0.1
f0 = 25.
ricker = cricker(nt, dt, f0, t0)
gridsrc = csrcspread(n1, n2, dh, npml, xs, zs, -1)


# Marching
ux, uz = evolution(mu, lbd, lbdmu, bux, buz, pmlx0, pmlx1, pmlz0, pmlz1, npml, isurf, 2, ricker, gridsrc, dh, nt, dt)

plt.imshow(uz, aspect='auto', vmin=-1.e-6, vmax=1.e-6)
plt.colorbar()
plt.show()

#evolution(np.ndarray[DTYPE_f, ndim=2] mu, np.ndarray[DTYPE_f, ndim=2] lbd, np.ndarray[DTYPE_f, ndim=2] lbdmu, np.ndarray[DTYPE_f, ndim=2] bux, np.ndarray[DTYPE_f, ndim=2] buz, np.ndarray[DTYPE_f, ndim=2] pmlx0, np.ndarray[DTYPE_f, ndim=2] pmlx1, np.ndarray[DTYPE_f, ndim=2] pmlz0, np.ndarray[DTYPE_f, ndim=2] pmlz1, int npml, int isurf, int srctype, np.ndarray[DTYPE_f, ndim=1] tsrc, np.ndarray[DTYPE_f, ndim=2] gsrc, DTYPE_f dh, int nt, DTYPE_f dt)
