# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: Convenience import for nessi.signal
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018 Damien Pageot
# ------------------------------------------------------------------
"""
Initialization file for nessi.signal.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

# Import nessi.signal classes and functions
from .tapering import time_taper
from .filtering import sin2filter
from .sources import lsrcinv
from .operations import avg

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
