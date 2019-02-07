#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: tapering.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018 Damien Pageot
# ------------------------------------------------------------------
"""
Data tapering functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def time_taper(object, **options):
    """
    Tapering data.

    :param object: the Stream object containing traces to taper
    :param tbeg: length of taper (ms) at trace start (=0.).
    :param tend: length of taper (ms) at trace end (=0).
    :param type: 'linear'(default), 'sine', 'cosine'
    """

    # Get options
    tbeg = options.get('tbeg', 0)
    tend = options.get('tend', 0)
    type = options.get('type', 'linear')

    # Get ns and dt from header
    ns = object.header[0]['ns']
    dt = object.header[0]['dt']/1000000.

    # Get number of traces
    ntrac = len(object.header)

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
    if ntrac == 1:
        object.traces[:] *= ftap[:]
    else:
        for itrac in range(0, ntrac):
            object.traces[itrac, :] *= ftap[:]

def taper1d(dobs, ntap1, ntap2, min=1.0, type='linear', axis=0):
    """
    Taper data.
    """

    dobstaper = np.zeros(np.shape(dobs), dtype=np.float32)

    # Get the number of dimensions of dobs
    if dobs.ndim == 1:
        n = len(dobs)
    if dobs.ndim == 2:
        if axis == 0:
            n = np.size(dobs, axis=0)
        else:
            n = np.size(dobs, axis=1)

    # Calculate the taper function
    if type == 'linear':
        ftap = _linear(n, ntap1, ntap2)
    if type == 'sine':
        ftap = _sine(n, ntap1, ntap2)
    if type == 'cosine':
        ftap = _cosine(n, ntap1, ntap2)

    # Apply the taper function
    if dobs.ndim == 1:
        dobstaper[:] = dobs[:]*ftap[:]
    else:
        if axis == 0:
            for i in range(0, np.size(dobs, axis=1)):
                dobstaper[:, i] = dobs[:, i]*ftap[:]
        if axis == 1:
            for i in range(0, np.size(dobs, axis=0)):
                dobstaper[i, :] = dobs[i, :]*ftap[:]

    return dobstaper
