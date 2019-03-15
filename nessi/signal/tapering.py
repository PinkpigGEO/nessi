#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: tapering.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018, 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Data tapering functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def _linear(n, ntap1, ntap2):
    """
    Linear taper type.

    :param n: taper total number of points
    :param ntap1: number of points to taper at the begining
    :param ntap2: number of points to tper at the end
    """

    # Initialize taper function
    ftap = np.zeros(n, dtype=np.float32)
    ftap[:] = 1

    # Create the taper function
    for i in range(0, ntap1):
        if i == 0:
            ftap[i] = 0.
        else:
            ftap[i] = float(i)/float(ntap1)
    for i in range(0, ntap2):
        if i == 0:
            ftap[n-i-1] = 0.
        else:
            ftap[n-i-1] = float(i)/float(ntap2)

    return ftap

def _sine(n, ntap1, ntap2):
    """
    Sine taper type.

    :param n: taper total number of points
    :param ntap1: number of points to taper at the begining
    :param ntap2: number of points to tper at the end
    """

    # Initialize taper function
    ftap = np.zeros(n, dtype=np.float32)
    ftap[:] = 1

    # Create the taper function
    for i in range(0, ntap1):
        if i == 0:
            ftap[i] = 0.
        else:
            ftap[i] = np.sin(np.pi*float(i)/float(ntap1)/2.)
    for i in range(0, ntap2):
        if i == 0:
            ftap[n-i-1] = 0.
        else:
            ftap[n-i-1] = np.sin(np.pi*float(i)/float(ntap2)/2.)

    return ftap

def _cosine(n, ntap1, ntap2):
    """
    Cosine taper type.

    :param n: taper total number of points
    :param ntap1: number of points to taper at the begining
    :param ntap2: number of points to tper at the end
    """

    # Initialize taper function
    ftap = np.zeros(n, dtype=np.float32)
    ftap[:] = 1

    # Create the taper function
    for i in range(0, ntap1):
        if i == 0:
            ftap[i] = 0.
        else:
            ftap[i] = 0.5*(1.0-np.cos(np.pi*float(i)/float(ntap1)))
    for i in range(0, ntap2):
        if i == 0:
            ftap[n-i-1] = 0.
        else:
            ftap[n-i-1] = 0.5*(1.0-np.cos(np.pi*float(i)/float(ntap2)))

    return ftap

def time_taper(data, dt=0.01, **options):
    """
    Taper the start and/or the end of data to zero.

    :param data: numpy array
    :param dt: time sampling (default=0.01)
    :param tbeg: (optional) length of taper (ms) at trace start (=0.).
    :param tend: (optional) length of taper (ms) at trace end (=0).
    :param type: (optional) 'linear'(default), 'sine', 'cosine'
    """

    # Get options
    tbeg = options.get('tbeg', 0)
    tend = options.get('tend', 0)
    type = options.get('type', 'linear')

    # Get the number of dimensions
    ndim = np.ndim(data)

    # Get the dimensions
    if ndim == 1:
        ns = np.size(data, axis=0)
    if ndim == 2:
        ntrac = np.size(data, axis=0)
        ns = np.size(data, axis=1)

    # Calculate the number of points to taper at begining and at end
    if(tbeg !=0. or tend !=0.):
        ntap1 = int(tbeg/1000./dt)+1
        ntap2 = int(tend/1000./dt)+1

    # Calculate the taper function
    if type == 'linear':
        ftap = _linear(ns, ntap1, ntap2)
    if type == 'sine':
        ftap = _sine(ns, ntap1, ntap2)
    if type == 'cosine':
        ftap = _cosine(ns, ntap1, ntap2)

    # Apply the taper function
    if ndim == 1:
        data[:] *= ftap[:]
    else:
        for itrac in range(0, ntrac):
            data[itrac, :] *= ftap[:]

    return data
