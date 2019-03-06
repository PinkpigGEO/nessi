#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: windowing.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018, 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Data windowing functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import copy
import numpy as np

def window_trace(data, **options):
    """
    Window along trace.

    :param data: numpy array
    :param vmin: minimum value to pass
    :param vmax: maximum value to pass
    :param dv: sampling along the axis to window (default=1.)
    :param fv: value of the first sample along the axis to window (default=0.)
    """

    # Get the number of dimensions
    ndim = np.ndim(data)


def window_data(object, **options):
    """
    Window along traces.

    :param object: input Stream object containing traces to window.
    :param vmin: (optional) minimum time to pass in second (default tmin=0.0).
    :param vmax: (optional) maximum time to pass in second (default tmax=0.0).
    """

    # Create a copy of the Stream object
    objcopy = copy.deepcopy(object)

    # Get number of time sample and time sampling from header
    ns = objcopy.header[0]['ns']
    dt = objcopy.header[0]['dt']/1000000.
    delrt = objcopy.header[0]['delrt']/1000.

    # Get the number of traces
    if objcopy.traces.ndim == 1:
        ntrac = 1
    if objcopy.traces.ndim == 2:
        ntrac = np.size(objcopy.traces, axis=0)

    # Get options
    tmin = options.get('tmin', 0.)
    tmax = options.get('tmax', 0.)

    # Calculate the index of tmin and tmax and the size of the new data array
    itmin = np.int((tmin-delrt)/dt)
    itmax = np.int((tmax-delrt)/dt)
    nsnew = itmax-itmin+1
    if tmin < 0:
        delrtnew = int(tmin*1000.)
    else:
        delrtnew = 0

    # Slice the data array and update the header
    if objcopy.traces.ndim == 1:
        traces = np.zeros(nsnew, dtype=np.float32, order='C')
        traces[:] = objcopy.traces[itmin:itmax+1]
        objcopy.traces = traces
        objcopy.header[0]['ns'] = nsnew
        objcopy.header[0]['delrt'] = delrtnew
    else:
        traces = np.zeros((ntrac, nsnew), dtype=np.float32, order='C')
        traces[:, :] = objcopy.traces[:, itmin:itmax+1]
        objcopy.traces = traces
        objcopy.header[:]['ns'] = nsnew
        objcopy.header[:]['delrt'] = delrtnew

    return objcopy

def space_window(object, **options): #dobs, imin=0.0, imax=0.0, axis=0):
    """
    Window traces in space.

    :param dobs: input data to window
    :param imin: (optional) minimum value of trace to pass (=0)
    :param tmax: (optional) maximum value of trace to pass (=0)
    """

    # Get options
    key = options.get('key', 'tracf')
    vmin = options.get('vmin', 0)
    vmax = options.get('vmax', len(object.header))

    # Get trace indices corresponding to keyword values
    imin = np.argmin(np.abs(object.header[key][:]-vmin))
    imax = np.argmin(np.abs(object.header[key][:]-vmax))

    # Windowing
    object.header = object.header[imin:imax]
    object.traces = object.traces[imin:imax, :]
    for itrac in range(0, len(object.header)):
        object.header[itrac]['tracf'] = itrac+1
