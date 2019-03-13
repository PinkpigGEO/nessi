#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_swarm.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018, 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the Swarm mathods (nessi.globopt)

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.globopt import Swarm


def test_init_pspace_one_line():
    """
    swarm.init_pspace test for one point search.
    """

    # Attempted output
    output = np.zeros((1, 2, 3), dtype=np.float32)
    output[0,0,0] = -3.0
    output[0,0,1] = 3.0
    output[0,0,2] = 0.6
    output[0,1,0] = -3.0
    output[0,1,1] = 3.0
    output[0,1,2] = 0.6

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Testing the parameter space initialization
    np.testing.assert_equal(swarm.pspace, output)

def test_init_pspace_multi_line():
    """
    swarm.init_pspace test for multipoints search.
    """

    # Attempted output
    output = np.zeros((3, 2, 3), dtype=np.float32)
    output = np.array([
            [[-3.0, 3.0, 0.6], [-3.0, 3.0, 0.6]],
            [[-3.0, 3.0, 0.6], [-3.0, 3.0, 0.6]],
            [[-3.0, 3.0, 0.6], [-3.0, 3.0, 0.6]]], dtype=np.float32)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_multi_line.txt')

    # Testing the parameter space initialization
    np.testing.assert_equal(swarm.pspace, output)

def test_init_particles():
    """
    swarm.init_particles test.
    """

    # Attempted output
    output = np.array([
            [[-0.49786797,  1.321947  ], [-2.9993138 , -1.1860045 ], [-2.1194646 , -2.4459684 ]],
            [[-1.8824388 , -0.9266356 ], [-0.61939514,  0.23290041], [-0.4848329 ,  1.111317  ]]], dtype=np.float32)

    # Initialize the random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_multi_line.txt')

    # Initialize particles
    swarm.init_particles(2)

    # Testing particle initializations
    np.testing.assert_equal(swarm.current, output)

def test_get_gbest_full():
    """
    swarm.get_best tests.
    """

    # Attempted output
    output = np.array([[-0.4848329 ,  1.111317  ]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[:] = 1.0
    swarm.misfit[5] = 0.0

    # Testing get_gbest for topology: full
    np.testing.assert_equal(swarm.get_gbest(topology='full'), output)

def test_get_gbest_ring():
    """
    swarm.get_best tests.
    """

    # Attempted output
    output = np.array([[-0.4848329 ,  1.111317  ]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[:] = 1.0
    swarm.misfit[5] = 0.0

    # Testing get_gbest for topology: ring
    np.testing.assert_equal(swarm.get_gbest(topology='ring', indv=4), output)

def test_get_gbest_toroidal():
    """
    swarm.get_best tests.
    """

    # Attempted output
    output = np.array([[-0.4848329 ,  1.111317  ]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[:] = 1.0
    swarm.misfit[5] = 0.0

    # Testing get_gbest for topology: toroidal
    np.testing.assert_equal(swarm.get_gbest(topology='toroidal', indv=8, ndim=3), output)

def test_update_weight_full():
    """
    swarm.update with inertia weight and full topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -1.2868801 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-2.0357487,  -1.5266356 ]],
        [[-1.2193952,  -0.3670996 ]],
        [[-1.0848329,   0.511317  ]],
        [[-1.7870306,   1.6687046 ]],
        [[-2.7573261,   0.42280507]],
        [[-0.83671474, -0.24786106]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=0, topology='full')
    np.testing.assert_equal(swarm.current, output)

def test_update_weight_ring():
    """
    swarm.update with inertia weight and ring topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -1.2868801 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-2.0357487,  -1.5266356 ]],
        [[-0.34661528,  0.7379111 ]],
        [[-0.4848329,   1.111317  ]],
        [[-1.722132,    1.7665863 ]],
        [[-2.8356745,   1.0228051 ]],
        [[-0.98696524,  0.952139  ]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=0, topology='ring')
    np.testing.assert_equal(swarm.current, output)

def test_update_weight_toroidal():
    """
    swarm.update with inertia weight and toroidal topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -1.2868801 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-1.2824388,  -0.3266356 ]],
        [[-0.34661528,  0.7379111 ]],
        [[-1.0848329,   0.511317  ]],
        [[-1.8154657,   1.7281866 ]],
        [[-2.8356745,   1.0228051 ]],
        [[-0.83671474, -0.24786106]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=0, topology='toroidal', ndim=3)
    np.testing.assert_equal(swarm.current, output)

def test_update_constriction_full():
    """
    swarm.update with constriction and full topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -1.2596235 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-1.9943244,  -1.5266356 ]],
        [[-1.2193952,  -0.3670996 ]],
        [[-1.0848329,   0.511317  ]],
        [[-1.7833169,   1.6687046 ]],
        [[-2.7784958,   0.42280507]],
        [[-0.74469984, -0.24786106]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=1, topology='full')
    np.testing.assert_equal(swarm.current, output)

def test_update_constriction_ring():
    """
    swarm.update with constriction and ring topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -1.2596235 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-1.9943244,  -1.5266356 ]],
        [[-0.4203204,   0.60145724]],
        [[-0.4848329,   1.111317  ]],
        [[-1.7359539,   1.9022588 ]],
        [[-2.8356745,   1.0228051 ]],
        [[-0.8543527,   0.952139  ]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=1, topology='ring')
    np.testing.assert_equal(swarm.current, output)

def test_update_constriction_toroidal():
    """
    swarm.update with constriction and toroidal topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -1.2596235 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-1.2824388,  -0.3266356 ]],
        [[-0.4203204,   0.60145724]],
        [[-1.0848329,   0.511317  ]],
        [[-1.8040688,   1.8742346 ]],
        [[-2.8356745,   1.0228051 ]],
        [[-0.74469984, -0.24786106]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=1, topology='toroidal', ndim=3)
    np.testing.assert_equal(swarm.current, output)

def test_update_weight_full_pupd():
    """
    swarm.update with inertia weight and full topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.9993138,  -1.1860045 ]],
        [[-2.1194646,  -2.4459684 ]],
        [[-1.8824388,  -0.9266356 ]],
        [[-1.1416478,  -0.30718902]],
        [[-1.0848329,   0.511317  ]],
        [[-1.7732865,   2.2687047 ]],
        [[-2.8356745,   1.0228051 ]],
        [[-1.0961711,  -0.24786106]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.update(control=0, topology='full', pupd=0.5)
    np.testing.assert_equal(swarm.current, output)

def test_fiupdate_weight_full():
    """
    swarm.fiupdate with inertia weight and full topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -0.5860045 ]],
        [[-1.5194646,  -1.8459684 ]],
        [[-1.2824388,  -0.3266356 ]],
        [[-1.2193952,   0.5599115 ]],
        [[-1.0848329,   0.511317  ]],
        [[-1.7580183,   1.6687046 ]],
        [[-2.2356744,   0.42280507]],
        [[-1.0961711,  -0.17463902]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.fiupdate(control=0, topology='full')
    np.testing.assert_equal(swarm.current, output)

def test_fiupdate_weight_ring():
    """
    swarm.fiupdate with inertia weight and ring topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -0.5860045 ]],
        [[-2.7194648,  -1.8459684 ]],
        [[-1.63892,    -1.5266356 ]],
        [[-1.2193952,   0.8329004 ]],
        [[-1.0848329,   1.7113171 ]],
        [[-1.5215923,   1.6687046 ]],
        [[-2.2356744,   0.42280507]],
        [[-1.0961711,   0.952139  ]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.fiupdate(control=0, topology='ring')
    np.testing.assert_equal(swarm.current, output)

def test_fiupdate_weight_toroidal():
    """
    swarm.fiupdate with inertia weight and toroidal topology options.
    """

    # Attempted output
    output = np.array([
        [[-1.097868,    0.72194695]],
        [[-2.399314,   -0.5860045 ]],
        [[-1.5194646,  -1.8459684 ]],
        [[-1.2824388,  -0.3266356 ]],
        [[-1.2193952,   0.16728637]],
        [[-1.0848329,   0.511317  ]],
        [[-1.1732864,   1.6687046 ]],
        [[-2.2356744,   0.42280507]],
        [[-1.0961711,   0.952139  ]]], dtype=np.float32)

    # Initialize random number generator
    np.random.seed(1)

    # Initialize the swarm class
    swarm = Swarm()

    # Initialize the parameter space
    swarm.init_pspace('data/pspace_one_line.txt')

    # Initialize particles
    swarm.init_particles(9)

    # Fake evaluation
    swarm.history[:, :, :] = swarm.current[:, :, :]
    swarm.misfit[0] = 1.00; swarm.misfit[1] = 0.28
    swarm.misfit[2] = 0.01; swarm.misfit[3] = 0.72
    swarm.misfit[4] = 0.53; swarm.misfit[5] = 0.02
    swarm.misfit[6] = 0.64; swarm.misfit[7] = 0.21
    swarm.misfit[8] = 0.98

    # Testing update method
    swarm.fiupdate(control=0, topology='toroidal', ndim=3)
    np.testing.assert_equal(swarm.current, output)

if __name__ == "__main__" :
    np.testing.run_module_suite()
