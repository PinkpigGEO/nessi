#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: stream.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
Functions to prepare models for elastic 2D PSV modeling.
"""

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)

def cacqpos(int n1, int n2, float dh, int npml, np.ndarray[float, ndim=2] acq):
    """
    Convert (x,z) receiver positions in extended gird index

    :param n1: number of grid points in the first dimension
    :param n2: number of grid points in the second dimension
    :param dh: space sampling
    :param npml: number of grid points for PML
    :param acq: numpy array containing positions of receiver in (x,z)
    """

    cdef Py_ssize_t i1, i2

    # Get the number of receivers
    cdef int nrec = np.size(acq, axis=0)

    # Declare output array
    cdef np.ndarray[int, ndim=2] recpos = np.zeros((nrec, 2), dtype=np.int16)

    # Parameters
    lpml = float(npml)*dh

    # Convert position in index
    for irec in range(0, nrec):
        recpos[irec, 0] = int((acq[irec, 0]+lpml)/dh)
        recpos[irec, 1] = int((acq[irec, 1]+lpml)/dh)

    return recpos


def cricker(nt, dt, f0, t0):
    """
    Calculate Ricker source time function.

    :param nt: number of time samples
    :param dt: time sampling
    :param f0: peak frequency
    :param t0: delay time
    """

    cdef Py_ssize_t i1, i2

    # Declare output array
    cdef np.ndarray[float, ndim=1] src = np.zeros(nt, dtype=np.float32)

    # Declare variables
    cdef float t = 0.
    cdef float sigma = 0.

    # Calculate source
    for it in range(0, nt):
        t = float(it)*dt
        sigma = (np.pi*f0*t)*(np.pi*f0*t)
        src[it] = (1.-2.*sigma)*np.exp(-1.*sigma)

    return src

def csrcspread(int n1, int n2, float dh, int npml, float xs, float zs, float sigma):
    """
    Calculate the grid for source application

    :param n1: number of grid points in the first dimension
    :param n2: number of grid points in the second dimension
    :param dh: space sampling
    :param xs: position of the source along the x axis
    :param zs: position of the source along the z axis
    :param sigma: spreading
    """

    cdef Py_ssize_t i1, i2

    # Calculate parameters
    cdef int n1e = n1+2*npml
    cdef int n2e = n2+2*npml

    # Index of the source position
    cdef int ix = int(xs/dh)+npml
    cdef int iz = int(zs/dh)+npml

    # Declare output grid
    cdef np.ndarray[float, ndim=2] srcgrid = np.zeros((n1e, n2e), dtype=np.float32)

    # Declare variable
    cdef float x, z
    cdef float p1, p2
    cdef float betasum = 0.

    # Fill grid
    if sigma <= 0:
        srcgrid[iz, ix] = 1.
    else:
        for i2 in range(npml, n2+npml):
            x = float(i2)*dh
            for i1 in range(npml, n1+npml):
                z = float(i1)*dh
                p1 = (z-zs)*(z-zs)/(sigma*sigma)
                p2 = (x-xs)*(x-xs)/(sigma*sigma)
                srcgrid[i1, i2] = np.exp(-1*p1-p2)
                betasum = betasum+np.exp(-1*p1-p2)
        for i2 in range(0, n2e):
            for i1 in range(0, n1e):
                srcgrid[i1, i2] = srcgrid[i1, i2]/betasum

    return srcgrid
