#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: ximage.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
2D surface image from dataset.

TODO => xgraph, xpolar
     => difference xwigb xwigp ?

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

def ximage(object, **options):
    """
    matplotlib.pyplot.imshow adapted to plot SU files

    :param key: header keyword (default tracl)
    :param bclip: data values outside of [bclip,wclip] are clipped
    :param wclip: data values outside of [bclip,wclip] are clipped
    :param clip: clip used to determine bclip and wclip
    :param legend: colorbar 0=no colorbar (default) 1=colorbar
    :param label1: x-axis label
    :param label2: y-axis label
    :param title: title of the image
    :param cmap: color map (defautl 'gray'): gray, jet, ...
    """

    # Get options
    key = options.get('key', 'tracl')
    clip = options.get('clip', None)
    bclip = options.get('bclip', None)
    wclip = options.get('wclip', None)
    label1 = options.get('label1', ' ')
    label2 = options.get('label2', ' ')
    title = options.get('title', ' ')
    cmap = options.get('cmap', 'gray')
    legend = options.get('legend', 0)
    style = options.get('style', 'normal')
    interpolation = options.get('interpolation', None)

    # Check clip, bclip and wclip.
    # If clip, bclip and wclip are fixed, the clip value is ignored.
    if(clip == None and bclip == None and clip == None):
        # If no image clip value is specified bclip and wclip are fixed to the
        # minimum and maximum values in dataset, respectively.
        bclip = np.amin(object.traces)
        wclip = np.amax(object.traces)
    else:
        # If clip is defined but not bclip and wclip, bclip and wclip are fixed
        # from the clip value. If bclip is fixed and not wclip, wclip is fixed
        # as the opposite of bclip and reciproquely if wclip is fixed but not
        # bclip
        if(clip != None and bclip == None and wclip == None):
            bclip = -1.*clip
            wclip = clip
        if(bclip != None and wclip == None):
            wclip = -1.*bclip
        if(bclip == None and wclip != None):
            bclip = -1.*wclip

    # Get ns and dt from header
    ns = object.header[0]['ns']
    dt = float(object.header[0]['dt']/1000000.)
    if dt != 0:
        y0 = float(object.header[0]['delrt'])/1000.
        y1 = float(ns-1)*dt+y0
        x0 = object.header[0][key]
        x1 = object.header[-1][key]

    if object.header[0]['trid'] == 118:
        # Get d1
        d1 = float(object.header[0]['d1'])
        y0 = 0.
        y1 = float(ns-1)*d1
        x0 = object.header[0][key]
        x1 = object.header[-1][key]

    if object.header[0]['trid'] == 122:
        # Get d1
        d1 = float(object.header[0]['d1'])
        y0 = 0.
        y1 = float(ns-1)*d1
        # Get d2
        d2 = float(object.header[0]['d2'])
        x0 = float(object.header[0]['f2'])
        x1 = x0+float(len(object.header)-1)*d2

    if object.header[0]['trid'] == 132:
        # Get d1
        d1 = float(object.header[0]['d1'])
        y0 = float(object.header[0]['f1'])
        y1 = y0+float(ns-1)*d1
        # Get d2
        d2 = float(object.header[0]['d2'])
        x0 = float(object.header[0]['f2'])
        x1 = x0+float(len(object.header)-1)*d2

    # Add labels to axes and title to figure even if they are empty strings.
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(title)

    if style == 'normal':
        # Plot surface
        plt.imshow(object.traces.swapaxes(1,0), aspect='auto', cmap=cmap,
                    extent=[x0, x1, y1, y0],
                    vmin=bclip, vmax=wclip, interpolation=interpolation)

    if style == 'masw':
        # Plot surface
        plt.imshow(object.traces, origin='bottom-left', aspect='auto', cmap=cmap,
                    extent=[y0, y1, x0, x1],
                    vmin=bclip, vmax=wclip, interpolation=interpolation)

    # Add legend
    if legend == 1:
        plt.colorbar()

def xwigg(object, **options):
    """
    Wiggle for SU files

    :param clip: clip used to determine outside values to be clipped [-clip, clip]
    :param key: header keyword (default tracl)
    :param label1: x-axis label
    :param label2: y-axis label
    :param title: title of the image
    :param tracecolor: color of the traces
    :param tracestyle: style of the traces ('--', ':', ...)
    :param tracewidth: width of the traces
    :param skip: number of traces to skip for each plotted trace
    :param xcur: factor to increase trace amplitudes on output
    :param x1beg: start vlaue for the 1st axis
    :param x1end: end value for the 1st axis
    """

    # Get options
    clip = options.get('clip', -1)
    key = options.get('key', 'tracl')
    label1 = options.get('label1', ' ')
    label2 = options.get('label2', ' ')
    title = options.get('title', ' ')
    tracecolor = options.get('tracecolor', 'black')
    tracestyle = options.get('tracestyle', '-')
    tracewidth = options.get('tracewidth', 2)
    skip = options.get('skip', 1)
    xcur = options.get('xcur', 1)
    x1beg = options.get('x1beg', np.nan)
    x1end = options.get('x1end', np.nan)

    # Get ns and dt from header
    ns = object.header[0]['ns']
    dt = float(object.header[0]['dt']/1000000.)
    ntrac = len(object.header)
    if dt != 0:
        y0 = float(object.header[0]['delrt'])/1000.
        y1 = float(ns-1)*dt+y0
        x0 = object.header[0][key]
        x1 = object.header[-1][key]
        d2 = 1.

    if object.header[0]['trid'] == 118:
        # Get d1
        d1 = float(object.header[0]['d1'])
        y0 = 0.
        y1 = float(ns-1)*d1
        x0 = object.header[0][key]
        x1 = object.header[-1][key]

    if object.header[0]['trid'] == 122:
        # Get d1
        d1 = float(object.header[0]['d1'])
        y0 = 0.
        y1 = float(ns-1)*d1
        # Get d2
        d2 = float(object.header[0]['d2'])
        x0 = float(object.header[0]['f2'])
        x1 = x0+float(len(object.header)-1)*d2

    # Add labels
    plt.xlabel(label1)
    plt.ylabel(label2)

    # Add axes
    plt.title(title)

    # Get the normalization parameter (for output)
    y = np.linspace(y0, y1, ns)
    if clip >= 0. :
        norm = clip
    else:
        norm = np.amax(np.abs(object.traces))

    # Windowing 1st axis
    if x1beg != np.nan:
        y0 = x1beg
    if x1end != np.nan:
        y1 = x1end
        
    # Plot the traces
    plt.ylim(y1, y0)
    for itrac in range(0, ntrac, skip):
        wig = object.traces[itrac]/norm*d2*float(skip-1)*xcur
        plt.plot(wig+x0+float(itrac)*d2, y, color=tracecolor, linestyle=tracestyle, linewidth=tracewidth)
