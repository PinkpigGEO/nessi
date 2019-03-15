#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: filtering.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018, 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Data filtering functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def sin2filter(object, **options):
    """
    Applies a zero-phase, sine-squared tapered filter (adapted from the
    sufilter command - Seismic Unix 44R1).

    :param dobs: input data
    :param freq: array (1D) of filter frequencies (Hz)
    :param amps: array (1D) of filter amplitudes
    """

    # Get the number of traces, the number of sample and the time sampling
    nd = np.ndim(object.traces)
    if nd == 1:
        ntrac = 1
        ns = object.header[0]['ns']
        dt = object.header[0]['dt']/1000000.
        axis = 0
    if nd == 2:
        ntrac = len(object.header)
        ns = object.header[0]['ns']
        dt = object.header[0]['dt']/1000000.
        axis = 1

    # Get options
    freq = options.get('freq', [])
    amps = options.get('amps', [])

    # Calculate the Nyquist frequency
    fnyq = 0.5/dt

    # Fast Fourier transform
    gobs = np.fft.rfft(object.traces, axis=axis)

    # Get the number of frequency samples
    nfft = np.size(gobs, axis=axis)

    # Get the frequency array
    ftmp = np.fft.rfftfreq(ns, dt)

    # Get the frequency sampling
    df = ftmp[1]

    # Get the number of filter frequencies
    npoly = len(freq)

    # Integer filter frequencies
    intfreq = np.zeros(npoly, dtype=np.int)
    for ipoly in range(0, npoly):
        intfreq[ipoly] = int(freq[ipoly]/df)

    # Initialize the polygonal filter with sin^2 tapering
    pfilt = np.zeros(nfft, dtype=np.float32)

    # From 0 to first filter frequency
    for ifreq in range(0, intfreq[0]):
        pfilt[ifreq] = amps[0]

    # Middle frequencies
    for ipoly in range(0, npoly-1):

        c = 0.5*np.pi/float(intfreq[ipoly+1]-intfreq[ipoly])

        # Increasing amplitude
        if amps[ipoly] < amps[ipoly+1]:
            for ifreq in range(intfreq[ipoly], intfreq[ipoly+1]):
                s = np.sin(c*float(ifreq-intfreq[ipoly]))
                a = amps[ipoly+1]-amps[ipoly]
                pfilt[ifreq] = amps[ipoly]+a*s*s

        # Decreasing amplitude
        if amps[ipoly] > amps[ipoly+1]:
            for ifreq in range(intfreq[ipoly], intfreq[ipoly+1]):
                s = np.sin(c*float(intfreq[ipoly]-ifreq))
                a = amps[ipoly]-amps[ipoly+1]
                pfilt[ifreq] = amps[ipoly]-a*s*s

        # Stable amplitude
        if amps[ipoly] == amps[ipoly+1]:
            for ifreq in range(intfreq[ipoly], intfreq[ipoly+1]):
                pfilt[ifreq] = amps[ipoly]

    # From the last filter frequency to the last frequency
    for ifreq in range(intfreq[-1], nfft):
        pfilt[ifreq] = amps[-1]

    # Apply filter and Inverse Fast Fourier Transform
    gfiltered = np.zeros(nfft, dtype=np.complex64)
    if nd == 1:
        gfiltered[:] = gobs[:]*pfilt[:]
        object.traces[:] = np.fft.irfft(gfiltered, n=ns)
    if nd == 2:
        for itrac in range(0, ntrac):
            gfiltered[:] = gobs[itrac, :]*pfilt[:]
            object.traces[itrac, :] = np.fft.irfft(gfiltered, n=ns)
