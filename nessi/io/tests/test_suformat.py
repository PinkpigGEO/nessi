#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_stream_main.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the main methods of the Stream class.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.io import suread

# List of test functions
# - test_suformat_suread_1d()
# - test_suformat_suread_2d()

def test_suformat_suread_1d():
    """
    Test the ``suread`` method.
    """

    # Read the Seismic Unix test file
    sdata = suread('data/nessi_test_data_1d.su')

    # Testing
    np.testing.assert_equal(np.size(sdata.header), 1)
    np.testing.assert_equal(sdata.header[0]['ns'], 5000)
    np.testing.assert_equal(sdata.header[0]['dt'], 100)
    np.testing.assert_equal(sdata.header[0]['trid'], 1)
    np.testing.assert_equal(sdata.header[0]['year'], 2019)

def test_suformat_suread_2d():
    """
    Test the ``suread`` method.
    """

    # Read the Seismic Unix test file
    sdata = suread('data/nessi_test_data_2d.su')

    # Testing
    np.testing.assert_equal(np.size(sdata.header), 24)
    np.testing.assert_equal(sdata.header[0]['ns'], 5000)
    np.testing.assert_equal(sdata.header[0]['dt'], 100)
    np.testing.assert_equal(sdata.header[0]['trid'], 1)
    np.testing.assert_equal(sdata.header[0]['year'], 2019)

if __name__ == "__main__" :
    np.testing.run_module_suite()
