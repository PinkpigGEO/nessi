#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: stream.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
Class to handle dataset using a Seismic Unix CWP (rev.0) data structure.
"""

# Import modules
import os
import sys
import copy
import numpy as np
from scipy.signal import resample

from nessi.graphics import ximage, xwigg
from nessi.signal import stack as sustack
from nessi.signal import time_window, space_window
from nessi.signal import time_taper
from nessi.signal import sin2filter
from nessi.signal import lsrcinv
from nessi.signal import avg

class Stream():
    """
    Class to handle seismic dataset. The data structure use a classic
    ``numpy`` array to store data and a ``numpy`` array with a custom
    datatype to store trace header values. The header array follows the
    Seismic Unix CWP (revision 0) structure. Further, the data structure
    embedded a history log (in text format) which stores all signal
    processing applyed to the data.
    """

    def __init__(self):
        """
        Initialize the NESSI data structure. The data structure is divided in
        header, traces, and data processing history. Additionnaly, object
        variable exist to store the path of the original file and the file
        format to access to additionnal informations if needed.
        """

        # This data type follows the Seismic Unix CWP (revision 0)
        # binary header structure.
        sudtype = np.dtype([
            ('tracl', np.int32), ('tracr', np.int32), ('fldr', np.int32), \
            ('tracf', np.int32), ('ep', np.int32), ('cdp', np.int32), \
            ('cdpt', np.int32), ('trid', np.int16), ('nvs', np.int16), \
            ('nhs', np.int16), ('duse', np.int16), ('offset', np.int32), \
            ('gelev', np.int32), ('selev', np.int32), ('sdepth', np.int32), \
            ('gdel', np.int32), ('sdel', np.int32), ('swdep', np.int32), \
            ('gwdep', np.int32), ('scalel', np.int16), ('scalco', np.int16), \
            ('sx', np.int32), ('sy', np.int32), ('gx', np.int32), \
            ('gy', np.int32), ('counit', np.int16), ('wevel', np.int16), \
            ('swevel', np.int16), ('sut', np.int16), ('gut', np.int16), \
            ('sstat', np.int16), ('gstat', np.int16), ('tstat', np.int16), \
            ('laga', np.int16), ('lagb', np.int16), ('delrt', np.int16), \
            ('muts', np.int16), ('mute', np.int16), ('ns', np.uint16), \
            ('dt', np.uint16), ('gain', np.int16), ('igc', np.int16), \
            ('igi', np.int16), ('corr', np.int16), ('sfs', np.int16), \
            ('sfe', np.int16), ('slen', np.int16), ('styp', np.int16), \
            ('stas', np.int16), ('stae', np.int16), ('tatyp', np.int16), \
            ('afilf', np.int16), ('afils', np.int16), ('nofilf', np.int16), \
            ('nofils', np.int16), ('lcf', np.int16), ('hcf', np.int16), \
            ('lcs', np.int16), ('hcs', np.int16), ('year', np.int16), \
            ('day', np.int16), ('hour', np.int16), ('minute', np.int16), \
            ('sec', np.int16), ('timebas', np.int16), ('trwf', np.int16), \
            ('grnors', np.int16), ('grnofr', np.int16), ('grnlof', np.int16), \
            ('gaps', np.int16), ('otrav', np.int16), ('d1', np.float32),\
            ('f1', np.float32), ('d2', np.float32), ('f2', np.float32), \
            ('ungpow', np.float32), ('unscale', np.float32), ('ntr', np.int32), \
            ('mark', np.int16), ('shortpad', np.int16), \
            ('unassignedInt1', np.int32), ('unassignedInt2', np.int32), \
            ('unassignedInt3', np.int32), ('unassignedInt4', np.int32), \
            ('unassignedFloat1', np.float32), ('unassignedFloat2', np.float32), \
            ('unassignedFloat3', np.float32)])

        # Orignal file
        self.origin = ''
        self.format = ''
        self.endian = ''

        # Initialize empty header and traces members
        self.header = np.zeros(1, dtype=sudtype, order='C')
        self.traces = np.zeros(1, dtype=np.float32, order='C')

        # Initialize history log member.
        self.history = '>> History log\n'

    def savehist(self, fname, path='.'):
        """
        Write the history log in a text file.

        :param fname: filename to write the history log without the .txt
            extension.
        :param fpath: (optional) path where the text file will be saved.
            By default, the file is saved in the current directory.
        """

        # Test if path exist


        # Open the history log text file to write
        fhist = open(path+'/'+fname, 'w')

        # Write history to history log text file
        fhist.write(self.history)

        # Close the history log text file
        fhist.close()

    def create(self, data, **options):
        """
        Create a stream object from a numpy array. The recommanded parameters
        are ``trid`` and ``dt`` for time data and ``trid``, ``n1`` and ``d1``
        for others. If no optional parameter is given, the header is filled with
        the default parameters:
        ``dt = 0.01 ms``
        ``trid = 1 (seismic data)``
        If ``trid`` is given for non time data but ``d1`` and/or ``d2`` not
        given:
        ``d1 = 1``
        ``d2 = 1``

        By default, for trid=1,  ``ns = np.size(data, axis=1)`` and
        ``n1 = np.size(data, axis=1)`` for trid !=0.

        Trace dependant keyword values can be set afterward.

        :param data: numpy array containing the data.
        :param trid: trace identification (default 1, seismic data)
        :param dt: time sampling if trid=1 (default=0.01 s)
        :param d1: for trid != 1 (default=1)
        :param d2: for trid != 1 (default=1)
        """

        # Get data array size and resize the stream objects header and traces
        # If data array is one dimensional
        if np.ndim(data) == 1:
            # Get the number of samples
            n1 = len(data)
            nd = 1
            # Resize header and traces arrays
            self.header.resize(1)
            self.traces.resize(n1)
        # If data array is two dimensional
        if np.ndim(data) == 2:
            # Get the number of samples in each dimension
            n2 = np.size(data, axis=0) # it corresponds to the number of traces
            n1 = np.size(data, axis=1)
            nd = 2
            # Resize header and traces arrays
            self.header.resize(n2)
            self.traces.resize(n2, n1)

        # Get trace identification code
        trid = options.get('trid', 1)

        # Edit header for trace identification and trace numbering
        if nd == 1:
            self.header['trid'] = trid
            self.header['tracl'] = 1
            self.header['tracf'] = 1
            self.header['tracr'] = 1
        if nd == 2:
            for i2 in range(0, n2):
                self.header[i2]['trid'] = trid
                self.header[i2]['tracl'] = i2+1
                self.header[i2]['tracf'] = i2+1
                self.header[i2]['tracr'] = i2+1

        # If trid=1 (seismic data), get the time sampling
        if trid == 1:
            # Get time sampling
            dt = options.get('dt', 0.01)
            # Edit header
            if nd == 1:
                self.header['ns'] = n1
                self.header['dt'] = int(dt*1000000.)
            if nd == 2:
                self.header[:]['ns'] = n1
                self.header[:]['dt'] = int(dt*1000000.)

        # If trid !=1 (non-seismic data), get the sampling in the first
        # (and 2nd if 2D array) dimensions
        if trid != 1:
            # Get sampling in the 1st and 2nd dimensions
            d1 = options.get('d1', 1)
            d2 = options.get('d2', 1)
            # Edit header
            if nd == 1:
                self.header['n1'] = n1
                self.header['n2'] = n2
                self.header['d1'] = d1
                self.header['d2'] = d2
            if nd == 2:
                self.header[:]['n1'] = n1
                self.header[:]['n2'] = n2
                self.header[:]['d1'] = d1
                self.header[:]['d2'] = d2

        # Fill traces
        if nd == 1:
            self.traces[:] = data[:]
        if nd == 2:
            self.traces[:, :] = data[:, :]

    def copy(self):
        """
        Return a copy of the Stream object.

        The Stream.copy method returns a copy of the original Stream object
        such as they are two different objects ``obj1 != obj2`` but contain
        exactly the same informations ``obj1.header == obj2.header`` and
        ``obj1.traces == obj2.traces``
        """
        return copy.deepcopy(self)

    def operation(self, type=' '):
        """
        Call simple opertion functions to apply on data.

        :param type: type of operation:
            - 'avg': remove average value for each trace of the Stream object
        """

        if type == 'avg':
            avg(self)

    def write(self, fname, path='.'):
        """
        Write the stream object on disk as a Seismic Unix CWP file (rev.0).

        :param fname: output file name without the ``.su`` extension.
        :param path: path to write the file (default is the current directory)
        """

        # Open file to write
        sufile = open(path+'/'+fname+'.su', 'wb')

        # Get the number of traces
        ntrac = len(self.header)

        # If one trace only
        if ntrac == 1:
            sufile.write(self.header[:])
            sufile.write(self.traces[:])
        else:
            for itrac in range(0, ntrac):
                sufile.write(self.header[itrac])
                sufile.write(self.traces[itrac, :])

        # Close file
        sufile.close()

    def wind(self, type='time', **options):
        """
        Windowing traces in time or space.

        :param type: 'time' (default) or 'space' windowing.
        :param vmin: minimum value to pass (in time or space)
        :param vmax: maximum value to pass (in time or space)
        """

        if type == 'time':
            time_window(self, **options)
        if type == 'space':
            space_window(self, **options)

    def kill(self, key=' ', a=1, min=0, count=1):
        """
        Zero out traces.
        If min= is set it overrides selecting traces by header.

        :param key: SU header keyword
        :param a: header value identifying traces to kill
        :param min: first trace to kill
        :param count: number of traces to kill
        """

        # Get the number of traces
        ntrac = self.traces.shape[0]

        # Kill traces from min to min+icount
        if key == ' ':
            for icount in range(0, count):
                if min+icount < ntrac:
                    self.traces[min+icount, :] = 0.
        # Kill traces with the given header value
        else:
            if key != ' ':
                for itrac in range(0, ntrac):
                    if self.header[itrac][key] == a:
                        self.traces[itrac, :] = 0.

    def resample(self, nso, dto):
        """
        Resample data.
        """
        # Get values from header
        ns = self.header[0]['ns']
        dt = self.header[0]['dt']/1000000.

        # Calculate time lenght for the old data
        t_old = float(ns-1)*dt

        # Calculate time lenght for the resampled data
        t_resamp = float(nso-1)*dto

        # Calculate the number of time samples of the old trace to resample
        nsamp = int(t_resamp/dt)+1

        # Resampling
        if nsamp > ns:
            print('Impossible to resample \n')
        else:
            if np.ndim(self.traces) == 1:
                traces = resample(self.traces[:,:nsamp], num=nso)
                self.traces = traces
            else:
                traces = resample(self.traces[:,:nsamp], num=nso, axis=1 )
                self.traces = np.ascontiguousarray(traces, dtype=np.float32)

        # Edit header
        self.header[:]['ns'] = int(nso)
        self.header[:]['dt'] = int(dto*1000000.)

    def image(self, **options):
        ximage(self, **options)

    def wiggle(self, **options):
        xwigg(self, **options)

    def taper(self, **options):
        time_taper(self, **options)

    def stack(self, **options):
        sustack(self, **options)

    def pfilter(self, **options):
        sin2filter(self, **options)

    def normalize(self, **options):
        """
        Normalize traces by traces or by maximum.

        :param mode: default(='max') or trace
        """

        # Get options
        mode = options.get('mode', 'max')

        if np.ndim(self.traces) == 1:
            # One trace: norm=max
            ampmax = np.abs(np.amax(self.traces))
            self.trace[0, :] /= ampmax
        else:
            if mode == 'max':
                ampmax = np.abs(np.amax(self.traces[:, :]))
                self.traces[:, :] /= ampmax
            if mode == 'trace':
                ntrac = np.size(self.traces, axis=0)
                # Loop over traces
                for itrac in range(0, ntrac):
                    ampmax = np.abs(np.amax(self.traces[itrac, :]))
                    self.traces[itrac, :] /= ampmax

    def specfx(self):
        """
        Fourier spectrum (time to frequency) of traces using the numpy.fft functions.
        """

        # Get number of time samples and numner of traces
        if np.ndim(self.traces) == 1:
            fftaxis = 0
        else:
            fftaxis = 1

        # Amplitude of the real Fourier transform
        self.traces = np.absolute(np.fft.rfft(self.traces, axis=1)) #fftaxis))

        # Get the frequency vector
        ns = self.header[0]['ns']
        dt = self.header[0]['dt']/1000000.
        frqv = np.fft.rfftfreq(ns, dt)

        # Update the SU header
        self.header[:]['ns'] = len(frqv)
        self.header[:]['d1'] = frqv[1]-frqv[0]
        self.header[:]['dt'] = 0
        self.header[:]['trid'] = 118 # Amplitude of complex trace from 0 to Nyquist

    def specfk(self):
        """
        FK spectrum of traces using the numpy.fft functions.
        """

        # Get dx from header, if not set dx=1.0
        d2 = self.header[0]['d2']
        if d2 == 0:
            d2 = 1.0
        ntrac = len(self.header)

        # Amplitude of the real Fourier transform
        self.traces = np.fft.rfft(self.traces, axis=1)
        self.traces = np.fft.fft(self.traces, axis=0)
        self.traces = np.flip(np.fft.fftshift(self.traces, axes=0), axis=0)
        self.traces = np.absolute(self.traces)

        # Get the frequency and K vectors
        ns = self.header[0]['ns']
        dt = self.header[0]['dt']/1000000.
        frqv = np.fft.rfftfreq(ns, dt)
        wavv = np.fft.fftfreq(ntrac, d2)
        # Centering
        wavv = np.fft.fftshift(wavv)

        # Update the SU header
        self.header[:]['ns'] = len(frqv)
        self.header[:]['d1'] = frqv[1]-frqv[0]
        self.header[:]['d2'] = np.abs(wavv[1]-wavv[0])
        self.header[:]['dt'] = 0
        self.header[:]['f1'] = frqv[0] #frqv[1]-frqv[0]
        self.header[:]['f2'] = wavv[0]
        self.header[:]['trid'] = 122 # Amplitude of complex trace from 0 to Nyquist

def susrcinv(dcal, scal, dobs):
    """
    Linear source inversion using SU files only.
    Return the estimated source and the corrected data in SU format.

    :param dcal: calculated data in SU format
    :param dobs: observed data in SU format
    :param scal: source used for calculated data in SU format
    """

    # SU parameters
    ns = dcal.header['ns'][0]
    dt = dcal.header['dt'][0]/1000000.

    # Get the number of traces
    if dobs.traces.ndim == 1:
        ntrac = 1
        naxis = 0
    if dobs.traces.ndim == 2:
        ntrac = np.size(dobs.traces, axis=0)
        naxis = 1

    # Linear source inversion
    srcest, corrector = lsrcinv(dcal.traces, scal.traces[:], dobs.traces, axis=naxis)

    # Calculated data correction
    gcal = np.fft.rfft(dcal.traces, axis=naxis)
    nw = np.size(gcal, axis=naxis)
    if naxis == 0:
        gcorrected = np.zeros((nw), dtype=np.complex64)
        gcorrected[:] = gcal[:]*np.conj(corrector[:])
        dcorrected = np.fft.irfft(gcorrected, n=ns, axis=0)
    if naxis == 1:
        gcorrected = np.zeros((ntrac, nw), dtype=np.complex64)
        for itrac in range(0, ntrac):
            gcorrected[itrac, :] = gcal[itrac, :]*np.conj(corrector[:])
        dcorrected = np.float32(np.fft.irfft(gcorrected, n=ns, axis=1))

    # Create outputs
    susrcest = Stream(); susrcest.create(srcest, dt=dt)
    sucorrected = Stream(); sucorrected.create(dcorrected, dt=dt)

    return susrcest, sucorrected
