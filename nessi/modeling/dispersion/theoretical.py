#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: theoretical.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Theoretical Love and Rayleigh wave dispersion calculation.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from .rootfinding import vrayleigh

class Disp():
    """
    Theoretical dispersion curve calculation.
    """

    def __init__(self):
        """
        Initialization
        """
        self.initialization = 1
        self.nmodes = 1

    def initmodel(self, nl):
        """
        Initialize model definition arrays.

        :param nl: number of layers
        """

        # Initialize arrays
        self.vp = np.zeros(nl, dtype=np.float32)
        self.vs = np.zeros(nl, dtype=np.float32)
        self.ro = np.zeros(nl, dtype=np.float32)
        self.hl = np.zeros(nl, dtype=np.float32)


    def initcurve(self, fmin, fmax, nf, vmin, vmax):
        """
        Initialize dispersion curve parameters.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :param nf: number of frequencies
        :param vmin: minimum velocity
        :param vmax: maximum velocity
        """

        self.freq = np.linspace(fmin, fmax, nf)
        self.vmin = vmin
        self.vmax = vmax
        self.curveinit = 1


    def initdiag(self, fmin, fmax, nf, vmin, vmax, nv):
        """
        Initialize dispersion diagram parameters.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :param nf: number of frequencies samples
        :param vmin: minimum velocity
        :param vmax: maximum velocity
        :param nv: number of velocity samples
        """

        self.freq = np.linspace(fmin, fmax, nf)
        self.vel = np.linspace(vmin, vmax, nv)
        self.diaginit = 1

    def lovedet(self, f, v):
        """
        Calculate L21(z0) component for a given frequency-velocity couple
        """
        nl = len(self.vp)

        # Loop over layers
        for il in range(0, nl):

            # Wavenumbers
            kt = (2.*np.pi*f/self.vs[nl-il-1])
            ke = (2.*np.pi*f/v)

            # Delay
            nu = np.sqrt(np.abs(ke*ke-kt*kt))
            mu = self.ro[nl-il-1]*self.vs[nl-il-1]**2

            # Initialize l21 & l22
            l21t = 0.
            l22t = 0.

            # Bottom layer
            if il == 0: #Bottom layer
                if nu != 0.: #complex(0., 0.):
                    l21t = 1./2. #nu*self.ro[nl-il-1]/(2.*nu*self.ro[nl-il-1])
                    l22t = 1./(2.*nu*mu) #1./(2.*nu*self.ro[nl-il-1]*self.vs[nl-il-1]**2)
            # Layer
            else:
                A = nu*self.hl[nl-il-1]
                # Pure imaginary
                if ke < kt:
                    G11 = np.cos(A)
                    G12 = np.sin(A)/(nu*mu)
                    G21 = -np.sin(A)*(nu*mu)
                    G22 = np.cos(A)
                # Zero
                elif ke == kt:
                    G11 = 1.
                    G12 = 0.
                    G21 = 0.
                    G22 = 1.
                # Pure real
                else:
                    G11 = (1.+np.exp(-2.*A))/2.
                    G12 = (1.-np.exp(-2.*A))/(2.*nu*mu)
                    G21 = -(1.-np.exp(-2.*A))/2.*(nu*mu)
                    G22 = (1.+np.exp(-2.*A))/2.

                # Calculate temporary l21 & l22
                l21t = l21*G11+l22*G21
                l22t = l21*G12+l22*G22

            # Assign l21 & l22
            l21 = l21t
            l22 = l22t

        return l21

    def lovecurves(self):
        """
        Estimate theoretical Love wave dispersion curves.
        """

        # Initializations
        nw = len(self.freq)
        nl = len(self.vp)

        # Initialize curves
        self.curves = np.zeros((nw, self.nmodes), dtype=np.float32)

        # Loop over frequencies (from high to low)
        for iw in range(0, nw):
            # Polarity under fundamental mode
            f0 = self.freq[nw-iw-1] #1./(2.*np.pi)
            v0 = np.amin(self.vs/1.05)
            det0 = self.lovedet(f0, v0)
            polarity0 = np.sign(det0)

            # Estimate dv
            imin = np.argmin(self.vs)
            vr = vrayleigh(self.vp[imin], self.vs[imin])
            dv = (self.vs[imin]-vr)/(2.*self.vs[imin])

            # Loop over modes
            for imode in range(0, self.nmodes):
                # Starting velocity
                v1 = v0
                polarity1 = polarity0
                while polarity1 == polarity0:
                    print(imode, iw, 'polarity search')
                    # Increase velocity
                    v1 += dv
                    # Calculate determinant
                    det1 = self.lovedet(f0, v1)
                    # Get polarity
                    polarity1 = np.sign(det1)
                v0 = v1-dv

                # Estimate Love wave velocity using the secante method
                iter = 0
                while(np.abs(v0-v1)>dv/2. and iter < 10):
                    print(imode, iw, 'velocity search', iter)
                    #print(iter, imode, iw, v0, v1)
                    fx0 = self.lovedet(f0, v0)
                    fx1 = self.lovedet(f0, v1)
                    v = v1-fx1*(v1-v0)/(fx1-fx0)
                    v0 = v1
                    v1 = v
                    iter += 1
                #print(iw, imode, v0, polarity0, v1, polarity1)
                print(iw, imode, v0, np.sign(fx0), v1, np.sign(fx1))

                polarity0 = polarity1
                if np.sign(fx0) == np.sign(fx1):
                    self.curves[nw-iw-1, imode] = v #0.
                else:
                    self.curves[nw-iw-1, imode] = v

                v0 = v/0.999

    def lovediag(self):
        """
        Calculate theoretical Love wave dispersion diagram.
        """

        # Initializations
        nw = len(self.freq)
        nv = len(self.vel)
        nl = len(self.vp)

        # Initialize dispersion diagram array
        self.diagram = np.zeros((nw, nv), dtype=np.float32)

        # Loop over frequencies
        for iw in range(0, nw):
            # Loop over velocities
            for iv in range(0, nv):
                # Store result in dispersion diagram array
                self.diagram[iw, iv] = self.lovedet(self.freq[iw], self.vel[iv])

    def rayleighdiag(self):
        """
        Calculate theoretical Rayleigh dispersion diagram.
        """

        # Initializations
        nw = len(self.freq)
        nv = len(self.vel)
        nl = len(self.vp)

        # Initialize dispersion diagram array
        self.diagram = np.zeros((nw, nv), dtype=np.float32)
