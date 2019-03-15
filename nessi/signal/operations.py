#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: operations.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Collection of simple operations on data.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def avg(object):
    """
    Remove average value from signals.

    :param object: input stream object.
    """

    # Get the number of traces and number of samples per trace
    nd = object.traces.ndim
    if nd == 1:
        ntrac = 1
        ns = object.header['ns']
    if nd == 2:
        ntrac = np.size(object.traces, axis=0)
        ns = object.header[0]['ns']

    # If 'nd == 1'
    if nd == 1:
        avg = np.sum(object.traces[:])/float(ns)
        object.trace[:] -= avg
    # If 'nd == 2', loop over traces
    if nd == 2:
        for itrac in range(0, ntrac):
            avg = np.sum(object.traces[itrac, :])/float(ns)
            object.traces[itrac, :] -= avg
