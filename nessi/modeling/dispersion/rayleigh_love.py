import numpy as np
import matplotlib.pyplot as plt

from nessi.modeling.dispersion import Disp

disp = Disp()

disp.initmodel(3)

disp.vp[0] = 400.; disp.vs[0] = 200.; disp.ro[0] = 2000.; disp.hl[0] = 25.
disp.vp[1] = 1000.; disp.vs[1] = 500.; disp.ro[1] = 2000.; disp.hl[1] = 25.
disp.vp[2] = 2000.; disp.vs[2] = 1000.; disp.ro[2] = 2000.; disp.hl[2] = 0.

disp.initdiag(1., 50., 100, 200., 1000., 400)

disp.lovediag()

# Dispersion curves
#disp.initcurve(10., 50., 100, 200., 1000.)
disp.nmodes = 5
disp.lovecurves()

freq = np.linspace(10., 50., 100)

plt.imshow(disp.diagram.swapaxes(1, 0), aspect='auto', cmap='jet', vmin=-2.0, vmax=2.0, origin='lower', extent=[10., 50., 200., 1000.] )
plt.colorbar()
plt.plot(freq, disp.curves[:,0], color='black')
plt.plot(freq, disp.curves[:,1], color='black')
plt.plot(freq, disp.curves[:,2], color='black')
plt.plot(freq, disp.curves[:,3], color='black')
plt.plot(freq, disp.curves[:,4], color='black')
plt.show()
