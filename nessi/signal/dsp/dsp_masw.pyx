#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: disp_masw.pyx
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
Multichannel Analysis of Surface Wave.
"""

# Import modules
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t DTYPE_f
ctypedef np.complex64_t DTYPE_c

@cython.boundscheck(False)
@cython.wraparound(False)


def cymasw(np.ndarray[DTYPE_c, ndim=2] gtraces, np.ndarray[DTYPE_f, ndim=1] offset, np.ndarray[DTYPE_f, ndim=1] vel, np.ndarray[DTYPE_f, ndim=1] frq, int iwmin, int whitening, int normalize):
    """
    Calculate the dispersion diagram using MASW method.
    Return the calculated dispersion diagram.

    :param gtraces: two-dimensional array containing the Fourier transform of
      the data on which performs MASW
    :param offset: one-dimensional array containing source-receiver offset for
      each trace
    :param nv: one-dimensional array containing velocities to test
    :param nw: one-dimensional array containing frequencies to test
    :param whithening: whitening True/False
    :param normalize: normalize the diagram by frequency (default False)
    """

    cdef Py_ssize_t i, iv, ir, iw

    # Get dimensions from arrays
    # - nr: number of traces
    # - nv: number of velocity samples
    # - nw: number of frequency samples
    cdef int nr = np.size(gtraces, axis=0)
    cdef int nv = np.size(vel)
    cdef int nw = np.size(frq)

    # Initialize temporary and dispersion diagram arrays
    cdef np.ndarray[DTYPE_c, ndim=1] tmp = np.zeros(nw, dtype=np.complex64)
    cdef np.ndarray[DTYPE_f, ndim=2] disp = np.zeros((nv, nw), dtype=np.float32)

    # Initialiaze variables
    cdef DTYPE_c phase

    # Loop over velocities
    for iv in range(0, nv):
        # Initialize temporary vector
        for iw in range(0, nw):
            tmp[iw] = complex(0., 0.)
        # Loop over traces
        for ir in range(0, nr):
            # Loop over frequencies
            for iw in range(0, nw):
                # Calculate the phase
                phase = complex(0., 1.)*2.*np.pi*offset[ir]*frq[iw]/vel[iv]
                # Stack over frequencies and receivers
                if whitening == 1:
                  tmp[iw] = tmp[iw]+gtraces[ir, iw+iwmin]/np.amax(np.abs(gtraces[:, iw+iwmin]))*np.exp(phase)
                else: # whitening == False
                  tmp[iw] = tmp[iw]+gtraces[ir, iw+iwmin]*np.exp(phase)
        # Stack over velocities
        for iw in range(0, nw):
            disp[iv,iw] = disp[iv, iw]+np.abs(tmp[iw])

    # Normalize
    if normalize  == 1:
      for iw in range(0, nw):
        # Get the maximum value for the current frequency
        awmax = np.amax(disp[:, iw])
        # Loop over velocities
        for iv in range(0, nv):
            disp[iv, iw] = disp[iv, iw]/awmax

    return disp
