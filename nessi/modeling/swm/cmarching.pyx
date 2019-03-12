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

ctypedef np.float32_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)

def dxforward(np.ndarray[DTYPE_f, ndim=2] f, int n1, int n2):
    """
    Fourth-order forward finite-difference along the x axis

    :param f: input 2D numpy array to derive
    :param n1: number of points in the first dimension
    :param n2: number of points in the second dimension
    """

    cdef Py_ssize_t i1, i2

    # Declare parameters
    cdef float c1=9./8.
    cdef float c2 = -1./24.

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] deriv = np.zeros((n1, n2), dtype=np.float32)

    # Loop over grid points
    for i2 in range(1, n2-2):
        for i1 in range(0, n1):
            deriv[i1, i2] = c1*(f[i1, i2+1]-f[i1, i2])+c2*(f[i1,i2+2]-f[i1, i2-1])

    # 2nd order derivative
    for i1 in range(0, n1):
        deriv[i1, 0] = f[i1, 1]-f[i1, 0]
        deriv[i1, n2-2] = f[i1, n2-1]-f[i1, n2-2]

    return deriv

def dxbackward(np.ndarray[DTYPE_f, ndim=2] f, int n1, int n2):
    """
    Fourth-order backward finite-difference along the x axis

    :param f: input 2D numpy array to derive
    :param n1: number of points in the first dimension
    :param n2: number of points in the second dimension
    """

    cdef Py_ssize_t i1, i2

    # Declare parameters
    cdef float c1=9./8.
    cdef float c2 = -1./24.

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] deriv = np.zeros((n1, n2), dtype=np.float32)

    # Loop over grid points
    for i2 in range(2, n2-1):
        for i1 in range(0, n1):
            deriv[i1, i2] = c1*(f[i1, i2]-f[i1, i2-1])+c2*(f[i1,i2+1]-f[i1, i2-2])

    # 2nd order derivative
    for i1 in range(0, n1):
        deriv[i1, 1] = f[i1, 1]-f[i1, 0]
        deriv[i1, n2-1] = f[i1, n2-1]-f[i1, n2-2]

    return deriv

def dzforward(np.ndarray[DTYPE_f, ndim=2] f, int n1, int n2, int npml, int isurf):
    """
    Fourth-order forward finite-difference along the z axis

    :param f: input 2D numpy array to derive
    :param n1: number of points in the first dimension
    :param n2: number of points in the second dimension
    """

    cdef Py_ssize_t i1, i2

    # Declare parameters
    cdef float c1=9./8.
    cdef float c2 = -1./24.

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] deriv = np.zeros((n1, n2), dtype=np.float32)

    # Parameters
    cdef int ibeg = 1
    if isurf == 1:
        ibeg = npml+1

    # Loop over grid points
    for i2 in range(0, n2):
        for i1 in range(ibeg, n1-2):
            deriv[i1, i2] = c1*(f[i1+1, i2]-f[i1, i2])+c2*(f[i1+2,i2]-f[i1-1, i2])

    # 2nd order derivative
    for i2 in range(0, n2):
        deriv[ibeg-1, i2] = f[ibeg, i2]-f[ibeg-1, i2]
        deriv[n1-2, i2] = f[n1-1, i2]-f[n1-2, i2]

    return deriv

def dzbackward(np.ndarray[DTYPE_f, ndim=2] f, int n1, int n2, int npml, int isurf):
    """
    Fourth-order backward finite-difference along the z axis

    :param f: input 2D numpy array to derive
    :param n1: number of points in the first dimension
    :param n2: number of points in the second dimension
    """

    cdef Py_ssize_t i1, i2

    # Declare parameters
    cdef float c1=9./8.
    cdef float c2 = -1./24.

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] deriv = np.zeros((n1, n2), dtype=np.float32)

    # Parameters
    cdef int ibeg = 2
    if isurf == 1:
        ibeg = npml+2

    # Loop over grid points
    for i2 in range(0, n2):
        for i1 in range(ibeg, n1-1):
            deriv[i1, i2] = c1*(f[i1, i2]-f[i1-1, i2])+c2*(f[i1+1,i2]-f[i1-2, i2])

    # 2nd order derivative
    for i2 in range(0, n2):
        deriv[ibeg-1, i2] = f[ibeg-1, i2]-f[ibeg-2, i2]
        deriv[n1-1, i2] = f[n1-1, i2]-f[n1-2, i2]

    return deriv

def evolution(np.ndarray[DTYPE_f, ndim=2] mu, np.ndarray[DTYPE_f, ndim=2] lbd, np.ndarray[DTYPE_f, ndim=2] lbdmu, np.ndarray[DTYPE_f, ndim=2] bux, np.ndarray[DTYPE_f, ndim=2] buz, np.ndarray[DTYPE_f, ndim=2] pmlx0, np.ndarray[DTYPE_f, ndim=2] pmlx1, np.ndarray[DTYPE_f, ndim=2] pmlz0, np.ndarray[DTYPE_f, ndim=2] pmlz1, int npml, int isurf, int srctype, np.ndarray[DTYPE_f, ndim=1] tsrc, np.ndarray[DTYPE_f, ndim=2] gsrc, DTYPE_f dh, int nt, DTYPE_f dt):
    """
    Marching
    """

    cdef Py_ssize_t it, i1e, i2e

    # Get arrays dimensions
    cdef int n1e = np.size(mu, axis=0)
    cdef int n2e = np.size(mu, axis=1)

    # Declare derivative arrays
    cdef np.ndarray[DTYPE_f, ndim=2] d1a = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] d2a = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] d1b = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] d2b = np.zeros((n1e, n2e), dtype=np.float32)

    # Declare wavefield arrays
    cdef np.ndarray[DTYPE_f, ndim=2] ux = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] uz = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] txx = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] txz = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] tzz = np.zeros((n1e, n2e), dtype=np.float32)

    # Declare split wavefield arrays
    cdef np.ndarray[DTYPE_f, ndim=2] uxx = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] uxz = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] uzx = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] uzz = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] txxx = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] txxz = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] tzzx = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] tzzz = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] txzx = np.zeros((n1e, n2e), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] txzz = np.zeros((n1e, n2e), dtype=np.float32)

    # Loop over time samples
    for it in range(0, nt):
        print(it+1, nt) #, np.amax(np.abs(ux)))
        # [Ux-Uz] Calculate derivatives
        d1a = dzbackward(txz, n1e, n2e, npml, isurf)
        d2a = dxforward(txx, n1e, n2e)
        d1b = dzforward(tzz, n1e, n2e, npml, isurf)
        d2b = dxbackward(txz, n1e, n2e)
        # [Ux-Uz] Calculate
        for i2e in range(0, n2e):
            for i1e in range(0, n1e):
                # [Ux] Split
                uxx[i1e, i2e] = ((1./dt-pmlx1[i1e, i2e])*uxx[i1e, i2e]+(1./dh)*bux[i1e, i2e]*d2a[i1e, i2e])/(1./dt+pmlx1[i1e, i2e])
                uxz[i1e, i2e] = ((1./dt-pmlz0[i1e, i2e])*uxz[i1e, i2e]+(1./dh)*bux[i1e, i2e]*d1a[i1e, i2e])/(1./dt+pmlz0[i1e, i2e])
                # [Uz] Split
                uzx[i1e, i2e] = ((1./dt-pmlx0[i1e, i2e])*uzx[i1e, i2e]+(1./dh)*buz[i1e, i2e]*d2b[i1e, i2e])/(1./dt+pmlx0[i1e, i2e])
                uzz[i1e, i2e] = ((1./dt-pmlz1[i1e, i2e])*uzz[i1e, i2e]+(1./dh)*buz[i1e, i2e]*d1b[i1e, i2e])/(1./dt+pmlz1[i1e, i2e])
                # [Ux-Uz] Complete
                ux[i1e, i2e] = uxx[i1e, i2e]+uxz[i1e, i2e]
                uz[i1e, i2e] = uzx[i1e, i2e]+uzz[i1e, i2e]
        # [Uz] Free surface condition
        #if isurf == 1:
        #    for i2e in range(0, n2e-2):
        #        uz[npml-1, i2e+1] = uz[npml, i2e+1]+lbd[npml, i2e+1]/lbdmu[npml, i2e+1]*(ux[npml, i2e+1]-ux[npml, i2e])

        # [Txx-Tzz-Txz] Calculate derivatives
        d1a = dzbackward(uz, n1e, n2e, npml, isurf)
        d2a = dxbackward(ux, n1e, n2e)
        d1b = dzforward(ux, n1e, n2e, npml, isurf)
        d2b = dxforward(uz, n1e, n2e)
        # [Txx-Tzz-Txz] calculate
        for i2e in range(0, n2e):
            for i1e in range(0, n1e):
                # [Txx] Split
                txxx[i1e, i2e] = ((1./dt-pmlx0[i1e, i2e])*txxx[i1e, i2e]+(1./dh)*lbdmu[i1e, i2e]*d2a[i1e, i2e])/(1./dt+pmlx0[i1e, i2e])
                txxz[i1e, i2e] = ((1./dt-pmlz0[i1e, i2e])*txxz[i1e, i2e]+(1./dh)*lbd[i1e, i2e]*d1a[i1e, i2e])/(1./dt+pmlz0[i1e, i2e])
                # [Tzz] Split
                tzzx[i1e, i2e] = ((1./dt-pmlx0[i1e, i2e])*tzzx[i1e, i2e]+(1./dh)*lbd[i1e, i2e]*d2a[i1e, i2e])/(1./dt+pmlx0[i1e, i2e])
                tzzz[i1e, i2e] = ((1./dt-pmlz0[i1e, i2e])*tzzz[i1e, i2e]+(1./dh)*lbdmu[i1e, i2e]*d1a[i1e, i2e])/(1./dt+pmlz0[i1e, i2e])
                # [Txz] Split
                txzx[i1e, i2e] = ((1./dt-pmlx1[i1e, i2e])*txzx[i1e, i2e]+(1./dh)*mu[i1e, i2e]*d2b[i1e, i2e])/(1./dt+pmlx1[i1e, i2e])
                txzz[i1e, i2e] = ((1./dt-pmlz1[i1e, i2e])*txzz[i1e, i2e]+(1./dh)*mu[i1e, i2e]*d1b[i1e, i2e])/(1./dt+pmlz1[i1e, i2e])
                # [Txx-Tzz] Source
                if(srctype == 1):
                    txxx[i1e, i2e] = txxx[i1e, i2e]+tsrc[it]*gsrc[i1e, i2e]/(dh*dh*dt)
                    tzzx[i1e, i2e] = tzzx[i1e, i2e]+tsrc[it]*gsrc[i1e, i2e]/(dh*dh*dt)
                if(srctype == 2):
                    txxx[i1e, i2e] = txxx[i1e, i2e]+tsrc[it]*gsrc[i1e, i2e]/(dh*dh*dt)
                if(srctype == 3):
                    tzzx[i1e, i2e] = tzzx[i1e, i2e]+tsrc[it]*gsrc[i1e, i2e]/(dh*dh*dt)
                # [Txx-Tzz-Txz] Complete
                txx[i1e, i2e] = txxx[i1e, i2e] + txxz[i1e, i2e]
                tzz[i1e, i2e] = tzzx[i1e, i2e] + tzzz[i1e, i2e]
                txz[i1e, i2e] = txzx[i1e, i2e] + txzz[i1e, i2e]

        # [Tzz-Txz] Free surface condition
        if isurf == 1:
            for i2e in range(0, n2e):
                # [Tzz]
                tzz[npml, i2e] = 0.
                tzz[npml-1, i2e] = -tzz[npml+1, i2e]
                # [Txz]
                txz[npml-1, i2e] = -txz[npml, i2e]
                txz[npml-2, i2e] = -txz[npml+1, i2e]

    return ux, uz
