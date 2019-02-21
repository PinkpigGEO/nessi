# -------------------------------------------------------------------
# Filename: csources.pyx
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Source related functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

def lsrcinv1d(dcal, scal, dobs):
  """
  Linear source inversion using stabilized deconvolution for 1D signals.

  :param dobs: observed data
  :param dcal: calculated data
  :param scal: source used for calculated data
  """

  # Statements for indexing
  cdef Py_ssize_t i

  # Statements
  cdef int ns, nsobs, nscal, nssrc

  # Get the number of time samples
  nsobs = len(dobs)
  nscal = len(dcal)
  nssrc = len(scal)

  # Check signal lenght validity
  if(nsobs != nscal or nsobs != nssrc or nscal != nsrc):
    print("Signals must have the same lenght.")
  else:
    ns = nsobs

def lsrcinv2d(np.ndarray[float, ndim=2, mode='c'] dcal, np.ndarray[float, ndim=1, mode='c'] scal, np.ndarray[float, ndim=2, mode='c']dobs, int axis=0):
    """
    Linear source inversion using stabilized deconvolution for 2D signals.

    :param dobs: observed data
    :param dcal: calculated data
    :param scal: source used for calculated data
    :param axis: time axis if dobs is a 2D array
    """

    # Statements for indexing
    cdef Py_ssize_t i

    # Statements
    cdef int ns=0, ntrac=0

    # Get number of time samples and number of traces
    if axis == 0:
        ns = np.size(dobs, axis=0)
        ntrac = np.size(dobs, axis=1)
    if axis == 1:
        ns = np.size(dobs, axis=1)
        ntrac = np.size(dobs, axis=0)

    # Fast Fourier transform
    gobs = np.fft.rfft(dobs, axis=axis)
    gcal = np.fft.rfft(dcal, axis=axis)
    gscal = np.fft.rfft(scal)
    nfft = np.size(gobs, axis=axis)

    # Linear source inversion
    num = np.zeros(nfft, dtype=np.complex64)
    den = np.zeros(nfft, dtype=np.complex64)

    if ntrac == 1:
        for iw in range(0, nfft):
            num[iw] += gcal[iw]*np.conj(gobs[iw])
            den[iw] += gcal[iw]*np.conj(gcal[iw])
    else:
        if axis == 0:
            for iw in range(0, nfft):
                for itrac in range(0, ntrac):
                    num[iw] += gcal[iw][itrac]*np.conj(gobs[iw][itrac])
                    den[iw] += gcal[iw][itrac]*np.conj(gcal[iw][itrac])
        if axis == 1:
            for iw in range(0, nfft):
                for itrac in range(0, ntrac):
                    num[iw] += gcal[itrac][iw]*np.conj(gobs[itrac][iw])
                    den[iw] += gcal[itrac][iw]*np.conj(gcal[itrac][iw])

    # Estimated source
    gsinv = np.zeros(nfft, dtype=np.complex64)
    gcorrector = np.zeros(nfft, dtype=np.complex64)
    for iw in range(0, nfft):
        if den[iw] != complex(0., 0.):
            gsinv[iw] = gscal[iw]*np.conj(num[iw]/den[iw])
            gcorrector[iw] = num[iw]/den[iw]
    sinv = np.float32(np.fft.irfft(gsinv, n=ns))

    return sinv, gcorrector
