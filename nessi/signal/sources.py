#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: sources.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018 Damien Pageot
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

def lsrcinv(dcal, scal, dobs, axis=0):
    """
    Linear source inversion using stabilized deconvolution.

    :param dobs: observed data
    :param dcal: calculated data
    :param scal: source used for calculated data
    :param axis: time axis if dobs is a 2D array
    """

    # Get number of time samples and numner of traces
    if np.ndim(dobs) == 1:
        ns = np.size(dobs)
        ntrac = 1
        axis=0
    else:
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
    nfft = len(gscal)

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
