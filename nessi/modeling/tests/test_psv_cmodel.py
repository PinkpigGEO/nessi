#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_psv_cmodel.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the PSV model manipulation methods.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.modeling.swm import cmodext
from nessi.modeling.swm import cmodbuo
from nessi.modeling.swm import cmodlame

# List of test functions
# - test_psv_cmodext()
# - test_psv_cmodbuo()

def test_psv_cmodext():
    """
    Test the ``cmodext`` method.
    """

    # Model parameters
    n1 = 101
    n2 = 101
    npml = 20

    # Create two-dimensionnal fake model
    fakemod = np.ones((n1, n2), dtype=np.float32)

    # Extend model
    fakemodext = cmodext(n1, n2, npml, fakemod)

    # Get parameters from resulting model
    n1e = np.size(fakemodext, axis=0)
    n2e = np.size(fakemodext, axis=1)

    # Testing
    np.testing.assert_equal(n1e, n1+2*npml)
    np.testing.assert_equal(n2e, n2+2*npml)
    np.testing.assert_equal(np.amin(fakemodext), np.amin(fakemod))
    np.testing.assert_equal(np.amax(fakemodext), np.amax(fakemod))

def test_psv_modbuo():
    """
    Test the ``cmodbuo`` method.
    """

    # Model parameters
    n1 = 101
    n2 = 101
    npml = 20

    # Create two-dimensionnal fake model
    fakemod = np.ones((n1, n2), dtype=np.float32)
    fakemod[:, :] = 1000.

    # Buoyancy
    fakemodbux, fakemodbuz = cmodbuo(n1, n2, fakemod)

    # Get parameters from the resulting model
    fakemeanx = np.mean(fakemodbux)
    fakemeanz = np.mean(fakemodbuz)

    # Testing
    np.testing.assert_allclose(fakemeanx, 1./1000., atol=1.e-10)
    np.testing.assert_allclose(fakemeanz, 1./1000., atol=1.e-10)
    np.testing.assert_allclose(np.amin(fakemodbux), 1./1000., atol=1.e-10)
    np.testing.assert_allclose(np.amin(fakemodbuz), 1./1000., atol=1.e-10)
    np.testing.assert_allclose(fakemeanx, fakemeanz, atol=1.e-10)
    np.testing.assert_allclose(fakemodbux, fakemodbuz, atol=1.e-10)

def test_psv_modlame():
    """
    Test the ``cmodlame`` method
    """

    # Model Parameters
    n1 = 101
    n2 = 101
    vp = 1500.
    vs = 900.
    ro = 1500.

    # Create fakemodels
    fakemodvp = np.zeros((n1, n2), dtype=np.float32)
    fakemodvs = np.zeros((n1, n2), dtype=np.float32)
    fakemodro = np.zeros((n1, n2), dtype=np.float32)
    fakemodvp[:, :] = vp
    fakemodvs[:, :] = vs
    fakemodro[:, :] = ro

    # Lamé models
    fakemu, fakelbd, fakelbdmu = cmodlame(n1, n2, fakemodvp, fakemodvs, fakemodro)

    # Calculate values for testing
    mu = vs*vs*ro
    lbd = vp*vp*ro-2.*mu
    lbdmu = lbd+2.*mu

    # Testing
    np.testing.assert_allclose(fakemu[0, 0], mu, atol=1.e-6)
    np.testing.assert_allclose(fakelbd[0, 0], lbd, atol=1.e-6)
    np.testing.assert_allclose(fakelbdmu[0, 0], lbdmu, atol=1.e-6)
    
if __name__ == "__main__" :
    np.testing.run_module_suite()
