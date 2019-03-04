# -------------------------------------------------------------------
# Filename: cmodels.pyx
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Parameter model related functions.

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

def modext(model, npml):
  """
  Extent input parameter model with PML.

  :param model: input model to extent
  :param npml: number of points in PML
  """

  # Statements
  cdef Py_ssize_t i1, i2

  # Get input model size in points
  cdef int n1 = np.size(model, axis=0)
  cdef int n2 = np.size(model, axis=1)

  # Initialize extended model
  cdef int n1e = int(n1+2*npml)
  cdef int n2e = int(n2+2*npml)
  cdef np.ndarray[float, ndim=2, mode='c'] model_ext = np.zeros((n1e, n2e), dtype=np.float32)

  # Fill extended model in the original model
  for i2 in range(0, n2):
    for i1 in range(0, n1):
      model_ext[npml+i1, npml+i2] = model[i1, i2]

  # Extent model in the first dimension
  for i2 in range(npml, n2+npml+1):
    for i1 in range(0, npml):
      model_ext[i1, i2] = model_ext[npml+1, i2]
      model_ext[n1+npml+i1, i2] = model_ext[npml+n1, i2]

  # Extent model in the second dimension
  for i1 in range(0, n1e):
    for i2 in range(0, npml):
      model_ext[i1, i2] = model_ext[i1, npml+1]
      model_ext[i1, n2+npml+i2] = model_ext[i1, npml+n2]

  return model_ext

# TO DO:
# - modbuo
# - modlame
