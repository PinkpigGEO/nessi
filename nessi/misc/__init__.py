# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: Convenience import for nessi.dispersion
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Initialization file for nessi.io.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

# Import nessi.io classes and functions
from .rootfinding import vrayleigh
from .corcoef import ccoef

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
