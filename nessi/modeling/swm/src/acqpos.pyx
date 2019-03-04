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

def acqpos(int n1, int n2, int npml, float dh, np.ndarray[float, ndim=2, mode='c'] acq):
  """
  Calculate the position of the receivers on the FD grid.

  :param n1: number of grid points in the first dimension (z)
  :param n2: number of grid points in the second dimension (x)
  :param npml: number of grid points for PMLs
  :param dh: distance between grid points
  :param acq: acquisition array (x, z)
  """

  # Statements
  cdef Py_ssize_t irec
  cdef float xmax, zmax, lpml
  cdef float xpos, zpos

  # Get the number of receivers
  cdef int nrec = np.size(acq, axis=0)

  # Declare receiver position array
  cdef np.ndarray[int, ndim=2, mode='c'] recpos = np.zeros(nrec, dtype=np.int16)

  # Calculate maximum lenghts
  xmax = float(n2+2*npml-1)*dh
  zmax = float(n1+2*npml-1)*dh
  lpml = float(npml)*dh

  # Calculate receiver position on the extended grid
  for irec in range(0, nrec):
    xpos = acq[irec, 0]+lpml
    zpos = acq[irec, 1]+lpml
    recpos[irec, 0] = int(xpos/dh)
    recpos[irec, 1] = int(zpos/dh)

  return recpos
