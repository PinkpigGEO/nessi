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

def cmodext(np.ndarray[float, ndim=2] mod, int n1, int n2, int npml):
    """
    Return an extended model (with PML).

    :param mod: 2D numpy array, input physical parameter model
    :param n1: number of points in the first dimension
    :param n2: number of points in the second dimension
    :param npml: number of points in PML
    """

    cdef Py_ssize_t i1, i2

    # Calculate the number of points for the extended model
    cdef int n1e = n1+2*npml
    cdef int n2e = n2+2*npml

    # Declare the output extended model
    cdef np.ndarray[float, ndim=2] modext = np.zeros((n1e, n2e), dtype=np.float32)

    # Fill the extended model with the original model
    for i2 in range(0, n2):
        for i1 in range(0, n1):
            modext[i1+npml, i2+npml] = mod[i1, i2]

    # Fill PML
    for i2 in range(0, npml):
        for i1 in range(0, n1):
            modext[i1+npml, i2] = mod[i1, 0]
            modext[i1+npml, i2+n2+npml] = mod[i1, n2-1]

    for i2 in range(0, n2e):
        for i1 in range(0, npml):
            modext[i1, i2] = modext[npml, i2]
            modext[i1+n1+npml, i2] = modext[n1+npml-1, i2]

    return modext


def cmodbuo(np.ndarray[float, ndim=2] modro):
    """
    Return buoyancy arrays bux and buz

    :param modro: 2D numpy array, density
    """

    cdef Py_ssize_t i1, i2

    # Calculate the number of points for the extended model
    cdef int n1 = np.size(modro, axis=0)
    cdef int n2 = np.size(modro, axis=1)

    # Declare arrays
    cdef np.ndarray[float, ndim=2] bux = np.zeros((n1, n2), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] buz = np.zeros((n1, n2), dtype=np.float32)

    # Fill bux
    for i2 in range(0, n2-1):
        for i1 in range(0, n1):
            bux[i1, i2] = 0.5*(1./modro[i1, i2]+1./modro[i1, i2+1])
    for i1 in range(0, n1):
        bux[i1, n2-1] = 1./modro[i1, n2-1]

    # Fill buz
    for i2 in range(0, n2):
        for i1 in range(0, n1-1):
            bux[i1, i2] = 0.5*(1./modro[i1, i2]+1./modro[i1+1, i2])
    for i1 in range(0, n2):
        bux[n1-1, i2] = 1./modro[n1-1, i2]

    return bux, buz

def cmodlame(np.ndarray[float, ndim=2] modvp, np.ndarray[float, ndim=2] modvs, np.ndarray[float, ndim=2] modro):
    """
    Return Lamé parameter models

    :param modvp: 2D numpy array, P-wave velocity
    :param modvs: 2D numpy array, S-wave velocity
    :param modro: 2D numpy array, density
    """

    # Declare variables
    cdef Py_ssize_t i1, i2

    # Get the input arrays dimensions
    cdef int n1 = np.size(modvp, axis=0)
    cdef int n2 = np.size(modvs, axis=1)

    # Declare output arrays
    cdef np.ndarray[float, ndim=2] mu = np.zeros((n1, n2), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] mu0 = np.zeros((n1, n2), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] lbd = np.zeros((n1, n2), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] lbdmu = np.zeros((n1, n2), dtype=np.float32)

    # Calculate mu0
    for i2 in range(0, n2):
        for i1 in range(0, n1):
          mu0[i1, i2] = modvs[i1, i2]*modvs[i1, i2]*modro[i1, i2]

    # Calculate mu
    for i2 in range(1, n2-1):
        for i1 in range(1, n1-1):
          mu[i1, i2] = 1./(0.25*(1./mu0[i1, i2]+1./mu0[i1+1,i2]+1./mu0[i1,i2+1]+1./mu0[i1+1,i2+1]))
    for i1 in range(0, n1):
        mu[i1, 0] = mu[i1, 1]
        mu[i1, n2-1] = mu[i1, n2-2]
    for i2 in range(0, n2):
        mu[0, i2] = mu[1, i2]
        mu0[n1-1, i2] = mu[n1-2, i2]

    # Calculate lbd and lbdmu
    for i2 in range(0, n2):
        for i1 in range(0, n1):
            lbd[i1, i2] = modvp[i1, i2]*modvp[i1, i2]*modro[i1, i2]-2.*mu0[i1, i2]
            lbdmu[i1, i2] = lbd[i1, i2]+2.*mu0[i1, i2]

    return mu, lbd, lbdmu

def cmodpml(int n1, int n2, float dh, int isurf, int npml, int ppml, float apml):
    """
    Calculate Perfect Matched Layer coefficients on a 2D grid.
    Return four grids.

    :param n1: number of points in the first dimension without PML
    :param n2: number of points in the second dimension without PML
    :param dh: space sampling of the grid
    :param isurf: free surface (0=no, 1=yes)
    :param npml: number of PML points
    :param ppml: power of the PML
    :param apml: amplitude of the PML
    """

    # Declare variables
    cdef Py_ssize_t i1, i2, ipml

    # Calculate extended dimensions
    cdef int n1e = n1+2*npml
    cdef int n2e = n2+2*npml

    # Declare output arrays
    cdef np.ndarray[float, ndim=2] pmlx0 = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] pmlx1 = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] pmlz0 = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] pmlz1 = np.zeros((n1e, n2e), dtype=np.float32)

    # Initialize parameters
    cdef float R = 0.0001                                 # Reflection coefficient
    cdef float L = float(npml-1)*dh                       # Lenght of the PML
    cdef float D = float(ppml+1)*apml*np.log(1./R)/(2./L) # Damping

    # Initialize variables
    cdef float val0 = 0.
    cdef float val1 = 0.
    cdef float val2 = 0.

    # Calculate PML coefficients
    for ipml in range(0, npml+1):
        val0 = float(npml-ipml)*dh
        val1 = float(npml-ipml)*dh-(dh/2.)
        val2 = float(npml-ipml)*dh+(dh/2.)
        # PMLZ
        for i2 in range(0, n2e):
            if isurf == 0:
                pmlz0[ipml, i2] = 0.5*D*(val0/L)**ppml
                pmlz0[ipml, i2] = 0.5*D*(val1/L)**ppml
            pmlz0[n1e-1-ipml, i2] = 0.5*D*(val0/L)**ppml
            pmlz1[n1e-1-ipml, i2] = 0.5*D*(val1/L)**ppml
        # PMLX
        for i1 in range(0, n1e):
            pmlx0[i1, ipml] = 0.5*D*(val0/L)**ppml
            pmlx0[i1, n2e-1-ipml] = 0.5*D*(val0/L)**ppml
            pmlx1[i1, ipml] = 0.5*D*(val1/L)**ppml
            pmlx1[i1, n2e-1-ipml] = 0.5*D*(val2/L)**ppml

    return pmlx0, pmlx1, pmlz0, pmlz1
