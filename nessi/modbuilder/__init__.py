# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: Convenience import for nessi.modbuilder
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018 Damien Pageot
# ------------------------------------------------------------------
"""
Initialization file for nessi.modbuilder .

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

# Import nessi.modbuilder.interp2d classes and functions
from .cinterp2d import cvoronoi
from .cinterp2d import cinvdist
from .cinterp2d import csibsons

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
