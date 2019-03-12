#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_psv_modeling.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the PSV modeling.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import matplotlib.pyplot as plt
import numpy as np
from nessi.modeling.swm import cmodext
from nessi.modeling.swm import cmodbuo
from nessi.modeling.swm import cmodlame
from nessi.modeling.swm import cmodpml
from nessi.modeling.swm import dxforward

# List of test functions
# - test_psv_cmodext()
# - test_psv_cmodbuo()
# - test_psv_cmodlame()
# - test_psv_cmodpml()
# - test_psv_dxforward()

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

def test_psv_modpml():
    """
    Test ``modpml`` method
    """

    # Model parameters
    n1 = 101
    n2 = 101
    dh = 0.5
    npml = 20
    apml = 800.
    ppml = 3

    # PML
    pmlx0, pmlx1, pmlz0, pmlz1 = cmodpml(n1, n2, dh, 0, npml, ppml, apml)

    # Attempted output
    attempted_x0 = np.array([8.1642914e+04, 6.9998586e+04, 5.9517684e+04, 5.0138945e+04,
                               4.1801168e+04, 3.4443105e+04, 2.8003516e+04, 2.2421184e+04,
                               1.7634865e+04, 1.3583338e+04, 1.0205364e+04, 7.4397104e+03,
                               5.2251460e+03, 3.5004395e+03, 2.2043582e+03, 1.2756705e+03,
                               6.5314325e+02, 2.7554477e+02, 8.1642906e+01, 1.0205363e+01,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                               0.0000000e+00, 1.0205363e+01, 8.1642906e+01, 2.7554477e+02,
                               6.5314325e+02, 1.2756705e+03, 2.2043582e+03, 3.5004395e+03,
                               5.2251460e+03, 7.4397104e+03, 1.0205364e+04, 1.3583338e+04,
                               1.7634865e+04, 2.2421184e+04, 2.8003516e+04, 3.4443105e+04,
                               4.1801168e+04, 5.0138945e+04, 5.9517684e+04, 6.9998586e+04,
                               8.1642914e+04], dtype=np.float32)

    # testing
    np.testing.assert_allclose(pmlx0[0,:], attempted_x0[:], atol=1.e-6)

def test_psv_dxforward():
    """
    test ``dxforward`` method
    """

    # Parameters
    n1 = 101
    n2 = 101

    # Declare array
    fakefunction = np.zeros((n1, n2), dtype=np.float32)
    testfunction = np.zeros((n1, n2), dtype=np.float32)

    # Function
    for i2 in range(0, n2):
        x = -np.pi+float((i2+1)/n2)*2.*np.pi
        fakefunction[:, i2] = np.cos(x)
        testfunction[:, i2] = -np.sin(x)

    # dxforward
    dx = dxforward(fakefunction, n1, n2)

    dh = 2.*np.pi/float(n2)
    plt.plot(testfunction[0, :], color='black')
    plt.plot(fakefunction[0, :], color='green')
    plt.plot(dx[0, :]/dh, color='red')
    plt.show()

if __name__ == "__main__" :
    np.testing.run_module_suite()
