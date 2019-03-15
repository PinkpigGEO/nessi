#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_tapering.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018, 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the tapering functions (nessi.signal.tapering)

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.core import Stream

# List of test functions
# - test_stream_taper_linear_1d()
# - test_stream_taper_sine_1d()
# - test_stream_taper_cosine_1d()

def test_stream_taper_linear_1d():
    """
    signal.tapering.time_taper testing for linear taper.
    """

    # Create initial data trace
    ns = 128   # number of time sample
    dt = 0.01
    data = np.ones((ns), dtype=np.float32)

    # Create Stream object
    object = Stream()
    object.create(data, dt=0.01)

    # Define the taper
    tbeg = float(16*dt)*1000.
    tend = float(16*dt)*1000.

    # Tapering
    object.taper(tbeg=tbeg, tend=tend, type='linear')

    # Attempted output
    output = np.array([ 0.        , 0.05882353, 0.11764706, 0.17647059, 0.23529412,
                        0.29411766, 0.35294119, 0.41176471, 0.47058824, 0.52941179,
                        0.58823532, 0.64705884, 0.70588237, 0.7647059 , 0.82352942,
                        0.88235295, 0.94117647, 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 0.94117647, 0.88235295, 0.82352942, 0.7647059 ,
                        0.70588237, 0.64705884, 0.58823532, 0.52941179, 0.47058824,
                        0.41176471, 0.35294119, 0.29411766, 0.23529412, 0.17647059,
                        0.11764706, 0.05882353, 0.        ], dtype=np.float32)

    # Testing
    np.testing.assert_allclose(object.traces, output, atol=1.e-4)

def test_taper_sine_1d():
    """
    signal.tapering.time_taper testing for sine taper.
    """

    # Create initial data trace
    ns = 128   # number of time sample
    dt = 0.01
    data = np.ones((ns), dtype=np.float32)

    # Create Stream object
    object = Stream()
    object.create(data, dt=0.01)

    # Define the taper
    tbeg = float(16*dt)*1000.
    tend = float(16*dt)*1000.

    # Tapering
    object.taper(tbeg=tbeg, tend=tend, type='sine')

    # Attempted output
    output = np.array([ 0.        , 0.09226836, 0.18374951, 0.27366298, 0.36124167,
                        0.44573835, 0.52643216, 0.60263461, 0.67369562, 0.7390089 ,
                        0.7980172 , 0.85021716, 0.8951633 , 0.93247223, 0.96182567,
                        0.9829731 , 0.99573416, 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 0.99573416, 0.9829731 , 0.96182567, 0.93247223,
                        0.8951633 , 0.85021716, 0.7980172 , 0.7390089 , 0.67369562,
                        0.60263461, 0.52643216, 0.44573835, 0.36124167, 0.27366298,
                        0.18374951, 0.09226836, 0.        ], dtype=np.float32)

    # Testing
    np.testing.assert_allclose(object.traces, output, atol=1.e-4)

def test_taper_cosine_1d():
    """
    signal.tapering.time_taper testing for cosine taper.
    """

    # Create initial data trace
    ns = 128   # number of time sample
    dt = 0.01
    data = np.ones((ns), dtype=np.float32)

    # Create Stream object
    object = Stream()
    object.create(data, dt=0.01)

    # Define the taper
    tbeg = float(16*dt)*1000.
    tend = float(16*dt)*1000.

    # Tapering
    object.taper(tbeg=tbeg, tend=tend, type='cosine')

    # Attempted output
    output = np.array([ 0.        , 0.00851345, 0.03376389, 0.07489143, 0.13049555,
                        0.19868268, 0.27713081, 0.36316851, 0.45386583, 0.54613417,
                        0.63683152, 0.72286916, 0.80131733, 0.86950445, 0.92510855,
                        0.96623611, 0.99148655, 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 1.        , 1.        , 1.        , 1.        ,
                        1.        , 0.99148655, 0.96623611, 0.92510855, 0.86950445,
                        0.80131733, 0.72286916, 0.63683152, 0.54613417, 0.45386583,
                        0.36316851, 0.27713081, 0.19868268, 0.13049555, 0.07489143,
                        0.03376389, 0.00851345, 0.        ], dtype=np.float32)

    # Testing
    np.testing.assert_allclose(object.traces, output, atol=1.e-4)

if __name__ == "__main__" :
    np.testing.run_module_suite()
