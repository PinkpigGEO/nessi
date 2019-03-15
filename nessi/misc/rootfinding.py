#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: rootfinding.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Root finding utilities.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def frayleigh(vp, vs, vr):
    """
    Function which must return 0 when Rayleigh wave velocity is the true one.

    :param vp: P-wave velocity
    :param vs: S-wave velocity
    :param vr: Rayleigh wave velocity
    """
    vp2 = vp*vp
    vs2 = vs*vs
    vr2 = vr*vr
    A = (vr/vs)**6-8.*(vr/vs)**4
    B = (8.*(vr/vs)**2)*(1.+2.*(1.-vs2/vp2))
    C = 16.*(1.-vs2/vp2)
    return A+B-C

def vrayleigh(vp, vs):
    """
    Estimate Rayleigh wave velocity from P-wave and S-wave velocities using
    the secante method.

    :param vp: P-wave velocity
    :param vs: S-wave velocity
    :return vr: Rayleigh wave velocity
    """

    # Estimate minimum and maximum value for Rayleigh wave velocity
    x0 = 0.80*vs
    x1 = 1.20*vs

    # Estimate Rayleigh wave velocity using the secante method
    while(np.abs(x0-x1)>0.1):
        fx0 = frayleigh(vp, vs, x0)
        fx1 = frayleigh(vp, vs, x1)
        x = x1-fx1*(x1-x0)/(fx1-fx0)
        x0 = x1
        x1 = x

    return (x0+x1)/2.
