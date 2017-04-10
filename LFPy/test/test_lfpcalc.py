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

class testLfpCalc(unittest.TestCase):
    """
    test class LFPy.lfpcalc
    """
    def test_calc_lfp_pointsource(self):
        sigma = 0.3
        cell = TestCell()
        np.testing.assert_equal(1./(4*np.pi*sigma),
                                LFPy.lfpcalc.calc_lfp_pointsource(cell,
                                    x=1, y=0, z=0, sigma=sigma,
                                    r_limit=cell.diam/2))

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
        self.xmid = np.array([0.])
        self.ymid = np.array([0.])
        self.zmid = np.array([0.])
        self.xend = np.array([0.])
        self.yend = np.array([0.])
        self.zend = np.array([0.])
        self.diam = np.array([1.])
