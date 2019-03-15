# -*- coding: utf-8 -*-
"""
nessi.dispersion
================================================
"""
from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)

from .rootfinding import vrayleigh
from .theoretical import Disp

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
