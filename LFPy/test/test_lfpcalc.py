#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""
from __future__ import division
import os
import unittest
import numpy as np
import LFPy
from LFPy import lfpcalc

class testLfpCalc(unittest.TestCase):
    """
    test class LFPy.lfpcalc
    """
    def test_return_dist_from_segment(self):
        cell = TestCell()
        dist, clostest_point = lfpcalc.return_dist_from_segments(
                                           cell.xstart, cell.ystart,
                                           cell.zstart, cell.xend,
                                           cell.yend, cell.zend, [1, 10, 0])
        np.testing.assert_equal([10], dist)
        np.testing.assert_equal([1, 0, 0], clostest_point.T[0])

        dist, clostest_point = lfpcalc.return_dist_from_segments(
                                           cell.xstart, cell.ystart,
                                           cell.zstart, cell.xend,
                                           cell.yend, cell.zend, [-1, 10, 0])
        np.testing.assert_equal([np.sqrt(101)], dist)
        np.testing.assert_equal([0, 0, 0], clostest_point.T[0])


    def test_calc_lfp_pointsource(self):
        sigma = 0.3
        cell = TestCell()
        np.testing.assert_equal(1./(4*np.pi*sigma),
                                lfpcalc.calc_lfp_pointsource(cell,
                                x=0.5, y=0, z=1, sigma=sigma,
                                r_limit=cell.diam/2))

    def test_calc_lfp_pointsource_anisotropic(self):

        sigma = [0.6, 0.3, 0.45]
        cell = TestCell()
        cell.xmid = cell.ymid = cell.zmid = np.array([1.2])
        sigma_r = np.sqrt(sigma[1] * sigma[2] * 1.2**2
                        + sigma[0] * sigma[2] * 1.2**2
                        + sigma[0] * sigma[1] * 1.2**2)

        phi_analytic = 1./(4*np.pi * sigma_r)
        np.testing.assert_equal(phi_analytic,
                                lfpcalc.calc_lfp_pointsource_anisotropic(cell,
                                x=0, y=0, z=0, sigma=sigma,
                                r_limit=cell.diam/2))


    def test_deltaS_calc(self):
        cell = TestCell()
        cell.yend[0] = 5
        ds = lfpcalc._deltaS_calc(cell.xstart, cell.xend,
                                  cell.ystart, cell.yend,
                                  cell.zstart, cell.zend)
        np.testing.assert_equal(ds, np.sqrt(26))



class TestCell(object):
    """Cell like object with attributes for predicting extracellular potentials,
    but with:
        - 1 compartment
        - position in (0,0,0)
        - length 1
        - diam 1
        - current amplitude 1
        - 1 timestep
    """
    def __init__(self):
        self.imem = np.array([[1.]])
        self.xstart = np.array([0.])
        self.ystart = np.array([0.])
        self.zstart = np.array([0.])
        self.xmid = np.array([0.5])
        self.ymid = np.array([0.])
        self.zmid = np.array([0.])
        self.xend = np.array([1.])
        self.yend = np.array([0.])
        self.zend = np.array([0.])
        self.diam = np.array([1.])
