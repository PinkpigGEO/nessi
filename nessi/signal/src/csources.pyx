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

def lsrcinv2d(np.ndarray[float, ndim=2] dcal, np.ndarray[float, ndim=1] dsrc, np.ndarray[float, ndim=2] dobs):
  """
  Linear source inversion using stabilized deconvolution for 2D signals.
  Return the estimated source (float) and the corrector factor array (complex).

  :param dobs: observed data
  :param dcal: calculated data
  :param dsrc: source used for calculated data
  """

  # Statements for indexing
  cdef Py_ssize_t i

  # Get the number of time samples
  cdef int ns = np.size(dobs, axis=1)
  cdef int nr = np.size(dobs, axis=0)

  # Fast Fourier transform
  cdef np.ndarray[float complex, ndim=2] gobs = np.ndarray(np.fft.rfft(dobs, axis=1), dtype=np.complex64)
  cdef np.ndarray[float complex, ndim=2] gcal = np.ndarray(np.fft.rfft(dcal, axis=1), dtype=np.complex64)
  cdef np.ndarray[float complex, ndim=1] gsrc = np.ndarray(np.fft.rfft(dsrc, axis=0), dtype=np.complex64)
  cdef int nw = len(gsrc)

  # Initialize arrays
  cdef np.ndarray[float complex, ndim=1] gcorrector = np.zeros(nw, dtype=np.complex64)
  cdef np.ndarray[float complex, ndim=1] gsrcest = np.zeros(nw, dtype=np.complex64)

  # Initialize parameters
  cdef np.ndarray[float complex, ndim=1] num = np.zeros(nw, dtype=np.complex64)
  cdef np.ndarray[float complex, ndim=1] den = np.zeros(nw, dtype=np.complex64)

  # Loop over traces
  for ir in range(0, nr):
    # Loop over frequencies
    for iw in range(0, nw):
      num[iw] = num[iw]+gcal[ir][iw]*np.conj(gobs[ir][iw])
      den[iw] = den[iw]+gcal[ir][iw]*np.conj(gcal[ir][iw])

  # Get the corrector factors and the estimated source wavelet
  # Loop over frequencies
  for iw in range(0, nw):
    if den[iw] != complex(0., 0.):
      gcorrector[iw] = num[iw]/den[iw]
  gsrcest[iw] = gsrc[iw]*np.conj(gcorrector[iw])

  # Inverse Fourier transform for the estimated source
  cdef np.ndarray[float, ndim=1] dsrcest = np.ndarray(np.fft.irfft(gsrcest, axis=0, n=ns), dtype=np.float32)

  return dsrcest, gcorrector
