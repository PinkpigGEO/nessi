#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: corcoef.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Correlation coefficient calculation.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def ccoef(signal1, signal2):
    """
    Function which returns the value of the correlation coefficient for two
    signals.

    :param signal1: first signal
    :param signal2: second signal
    """

    # Get parameters
    ns = len(signal1)

    # Calculate average
    average1 = np.average(signal1)
    average2 = np.average(signal2)

    # Calculate sub-coefficients
    sigma1 = 0.
    sigma2 = 0.
    sigma12 = 0.
    for i in range(0, ns):
        sigma1 += (signal1[i]-average1)**2
        sigma2 += (signal2[i]-average2)**2
        sigma12 += (signal1[i]-average1)*(signal2[i]-average2)

    # Finalize calculation
    sigma1 = np.sqrt(sigma1/float(ns))
    sigma2 = np.sqrt(sigma2/float(ns))
    sigma12 = sigma12/float(ns)

    coef = sigma12/(sigma1*sigma2)
    
    return coef
