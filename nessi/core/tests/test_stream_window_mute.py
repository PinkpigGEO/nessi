#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_stream_window.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the windowing/muting methods of the Stream class.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.core import Stream

# List of test functions
# - test_stream_wind_1d()
# - test_stream_wind_2d()
# - test_stream_windkey_2d()
# - test_stream_kill_one_trace_2d()
# - test_stream_kill_multi_traces_2d()

def test_stream_wind_1d():
    """
    Test the ``wind`` method of the Stream class for 1D data.
    """

    # Create one-dimensionnal data of size 'ns'
    ns = 1000
    dt = 0.001
    data = np.ones(ns)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Windowing
    object.wind(tmin=0.1, tmax=0.2)

    # Attempted result
    itmin = int(0.1/dt)
    itmax = int(0.2/dt)
    nsnew = itmax-itmin+1

    # Testing the stream object members initialization
    np.testing.assert_equal(object.header[0]['ns'], nsnew)
    np.testing.assert_equal(np.size(object.traces, axis=0), nsnew)

def test_stream_wind_2d():
    """
    Test the ``wind`` method of the Stream class for 2D data.
    """

    # Create two-dimensionnal data of size '(nr,ns)'
    nr = 100
    ns = 1000
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Windowing
    object.wind(tmin=0.1, tmax=0.2)

    # Attempted result
    itmin = int(0.1/dt)
    itmax = int(0.2/dt)
    nsnew = itmax-itmin+1

    # Testing the stream object members initialization
    np.testing.assert_equal(object.header[0]['ns'], nsnew)
    np.testing.assert_equal(np.size(object.traces, axis=0), nr)
    np.testing.assert_equal(np.size(object.traces, axis=1), nsnew)

def test_stream_windkey_2d():
    """
    Test the ``windkey`` method of the Stream class for 2D data.
    """

    # Create two-dimensionnal data of size '(nr,ns)'
    nr = 100
    ns = 1000
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Windowing
    vmin = 10
    vmax = 20
    object.windkey(key='tracf', vmin=vmin, vmax=vmax)

    # Attempted result
    nrnew = vmax-vmin+1

    # Testing the stream object members initialization
    np.testing.assert_equal(object.header[0]['ns'], ns)
    np.testing.assert_equal(object.header[0]['tracf'], 1)
    np.testing.assert_equal(object.header[0]['tracl'], 10)
    np.testing.assert_equal(np.size(object.traces, axis=0), nrnew)
    np.testing.assert_equal(np.size(object.traces, axis=1), ns)

def test_stream_kill_one_trace_2d():
    """
    Test the ``kill`` method of the Stream class for 2D data.
    """

    # Create two-dimensionnal data of size '(nr,ns)'
    nr = 100
    ns = 1000
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Killing
    object.kill(key='tracf', a=10)

    # Attempted result
    dkill = np.zeros(ns, dtype=np.float32)

    # Testing the stream object members initialization
    np.testing.assert_equal(object.traces[9,:], dkill)

def test_stream_kill_multi_traces_2d():
    """
    Test the ``kill`` method of the Stream class for 2D data.
    """

    # Create two-dimensionnal data of size '(nr,ns)'
    nr = 100
    ns = 1000
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Killing
    object.kill(key='tracf', a=10, count=10)

    # Attempted result
    dkill = np.zeros((10, ns), dtype=np.float32)

    # Testing the stream object members initialization
    np.testing.assert_equal(object.traces[9:19,:], dkill)

if __name__ == "__main__" :
    np.testing.run_module_suite()
