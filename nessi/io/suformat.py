#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: suformat.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
Support of Seismic Unix format.
"""

# Import modules
import os
import sys
import numpy as np
from nessi.core import Stream

def _sutype():
    """
    Numpy custom data structure for Seismic Unix CWP (rev.0).
    """
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

    return sudtype

def _check_format(filename):
    """
    Determine endianess, number of trace samples and number of traces.

    :param filename: name of the Seismic Unix file
    """

    # Open Seismic Unix file
    sufile = open(filename, 'rb')

    # Read the Binary Seismic Unix file and get the size
    bdata = sufile.read()
    bsize = os.stat(filename).st_size

    # Get value of ns considering 'little endian' format (nsl) and
    # 'big endian' format (nsb).
    nsl = np.frombuffer(bdata, dtype='<h', count=1, offset=114)[0]
    nsb = np.frombuffer(bdata, dtype='>h', count=1, offset=114)[0]

    # Determining endianess, number of samples and number of traces
    if(bsize%((nsl*4)+240) == 0): # Little Endian Format
        endian = 'l'
        ns = nsl
        ntrac = int(bsize/((nsl*4)+240))
    else:
        if(bsize%((nsb*4)+240) == 0): # Big Endian Format
            endian = 'b'
            ns = nsb
            ntrac = int(bsize/((nsb*4)+240))
        else:
            sys.exit("Unable to read "+self.filename+"\n")

    # Close the Seismic Unix File
    sufile.close()

    return endian, ns, ntrac

def suread(fname):
    """
    Read Seismic Unix files and store in NESSI data structure.

    :param fname: Seismic Unix filename and path

    .. rubric:: Basic usage

    >>> # Import suread from nessi package
    >>> from nessi.io import suread
    >>> # Read Seismic Unix file
    >>> sdata = suread('NESSI_TEST_DATA.su')
    """

    # Test if file exist


    # Check file
    endian, ns, ntrac = _check_format(fname)

    # Initialize stream
    sudata = Stream()
    sudata.origin = fname
    sudata.header.resize(ntrac) #, dtype=sudata.sutype)
    sudata.endian = endian

    if ntrac == 1:
        sudata.traces = np.zeros(ns, dtype=np.float32, order='C')
    else:
        sudata.traces = np.zeros((ntrac, ns), dtype=np.float32, order='C') #, dtype=np.float32)

    # Endianess parameters
    if sudata.endian == 'b': # Big endian
        sudtype = _sutype().newbyteorder()
        npdtype = '>f4'
    if sudata.endian == 'l': # Little endian
        sudtype = _sutype()
        npdtype = '<f4'

    # Open the file to read
    file = open(fname, 'rb')

    if ntrac == 1:
        # Get header
        bhdr = file.read(240)
        sudata.header[0] = np.frombuffer(bhdr, dtype=sudtype, count=1)[0]

        # Get data
        btrc = file.read(ns*4)
        sudata.traces[:] = np.frombuffer(btrc, dtype=(npdtype, ns), count=1)[0]

        # TRID default value (=1 seismic data)
        if sudata.header[0]['trid'] == 0:
            sudata.header[0]['trid'] = 1

        # NS keyword header value
        sudata.header[0]['ns'] = ns

        # DT default value (=0.04s)
        if sudata.header[0]['dt'] == 0:
            # Time sampling (default dt=0.04s)
            sudata.header[0]['dt'] = int(0.04*1000000.)
    else:
        # Loop over traces
        for itrac in range(0, ntrac):

            # Get header
            bhdr = file.read(240)
            sudata.header[itrac] = np.frombuffer(bhdr, dtype=sudtype, count=1)[0]

            # Get data
            btrc = file.read(ns*4)
            sudata.traces[itrac, :] = np.frombuffer(btrc, dtype=(npdtype, ns), count=1)[0]

            # TRID default value (=1 seismic data)
            if sudata.header[itrac]['trid'] == 0:
                sudata.header[itrac]['trid'] = 1

            # NS keyword header value
            sudata.header[itrac]['ns'] = ns

            # DT default value (=0.04s)
            if sudata.header[itrac]['dt'] == 0:
                # Time sampling (default dt=0.04s)
                sudata.header[itrac]['dt'] = int(0.04*1000000.)

    if endian == 'b':
        sudata.header.newbyteorder()

    # Close the file
    file.close()

    return sudata

def suwrite(object, fname, path='.'):
    """
    Write a stream object on disk as a Seismic Unix CWP file (rev.0).

    :param object: stream object
    :param fname: output file name without the ``.su`` extension.
    :param path: path to write the file (default is the current directory)

    .. rubric:: Basic usage

    >>> import numpy as np
    >>> from nessi.io import suwrite
    >>> # Generate fake data
    >>> ns = 256    # Number of sample
    >>> dt = 0.0001 # Time sampling
    >>> fakedata = np.ones(ns, dtype=np.float32)
    >>> # Create a Stream object containing the data
    >>> sdata = Stream()
    >>> sdata.create(fakedata, dt=dt)
    >>> # Write SU file on disk
    >>> suwrite(sdata, 'my_sufile')

    """

    # Open file to write
    sufile = open(path+'/'+fname+'.su', 'wb')

    # Get the number of traces
    ntrac = len(object.header)

    # If one trace only
    if ntrac == 1:
        sufile.write(object.header[:])
        sufile.write(object.traces[:])
    else:
        for itrac in range(0, ntrac):
            sufile.write(object.header[itrac])
            sufile.write(object.traces[itrac, :])

    # Close file
    sufile.close()
