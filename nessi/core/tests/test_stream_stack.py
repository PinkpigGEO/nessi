#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_stream_stack.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the stacking method of the Stream class.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.core import Stream

# List of test functions
# - test_stream_stack_2d()
# - test_stream_stack_mean_2d()
# - test_stream_stack_weight_2d()

def test_stream_stack_2d():
    """
    Test the ``stack`` method of the Stream class for 2D data.
    """

    # Create one-dimensionnal data of size 'ns'
    ns = 1000
    nr = 100
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Stacking
    object.stack()

    # Attempted result
    dstack = np.ones(ns, dtype=np.float32)
    dstack *= float(nr)

    # Testing the stream object members initialization
    np.testing.assert_equal(np.size(object.header, axis=0), 1)
    np.testing.assert_equal(object.header[0]['ns'], ns)
    np.testing.assert_equal(object.traces, dstack)

def test_stream_stack_mean_2d():
    """
    Test the ``stack`` method of the Stream class for 2D data.
    """

    # Create one-dimensionnal data of size 'ns'
    ns = 1000
    nr = 100
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Stacking
    object.stack(mean=True)

    # Attempted result
    dstack = np.ones(ns, dtype=np.float32)

    # Testing the stream object members initialization
    np.testing.assert_equal(np.size(object.header, axis=0), 1)
    np.testing.assert_equal(object.header[0]['ns'], ns)
    np.testing.assert_equal(object.traces, dstack)

def test_stream_stack_weight_2d():
    """
    Test the ``stack`` method of the Stream class for 2D data.
    """

    # Create one-dimensionnal data of size 'ns'
    ns = 1000
    nr = 100
    dt = 0.001
    data = np.ones((nr, ns), dtype=np.float32)

    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data, dt=dt)

    # Stacking
    weight= np.zeros(nr, dtype=np.float32)
    weight[:] = 10.
    object.stack(weight=weight)

    # Attempted result
    dstack = np.ones(ns, dtype=np.float32)
    dstack *= 1000.

    # Testing the stream object members initialization
    np.testing.assert_equal(np.size(object.header, axis=0), 1)
    np.testing.assert_equal(object.header[0]['ns'], ns)
    np.testing.assert_equal(object.traces, dstack)

if __name__ == "__main__" :
    np.testing.run_module_suite()
