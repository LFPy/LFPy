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


import unittest
import numpy as np
import LFPy


class testAliasMethod(unittest.TestCase):
    """
    test LFPy.alias_method methods
    """

    def test_alias_method_00(self):
        """deterministic probabilities 0.0 and 1.0"""
        idx = np.arange(2)
        probs = np.arange(2).astype(float)
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(
            LFPy.alias_method.alias_method(
                idx, probs, nidx), bins)

        self.assertEqual(nidx, hist[1])

    def test_alias_method_01(self):
        """probabilities 0.25 and 0.75"""
        idx = np.arange(2)
        probs = np.array([0.25, 0.75])
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(
            LFPy.alias_method.alias_method(
                idx, probs, nidx), bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 7 decimal places
        self.assertAlmostEqual(
            np.corrcoef(
                probs, hist.astype(float))[
                0, 1], 1., places=7)

    def test_alias_method_02(self):
        """probabilities 0.75 and 0.25"""
        idx = np.arange(2)
        probs = np.array([0.75, 0.25])
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(
            LFPy.alias_method.alias_method(
                idx, probs, nidx), bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 7 decimal places
        self.assertAlmostEqual(
            np.corrcoef(
                probs, hist.astype(float))[
                0, 1], 1., places=7)

    def test_alias_method_03(self):
        '''over range of normalized probabilities'''
        size = 5
        idx = np.arange(size)
        probs = np.arange(size).astype(float)**2
        probs /= probs.sum()
        nidx = 1000000
        bins = np.arange(probs.size + 1)

        hist, _ = np.histogram(
            LFPy.alias_method.alias_method(
                idx, probs, nidx), bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 5 decimal places
        self.assertAlmostEqual(
            np.corrcoef(
                probs, hist.astype(float))[
                0, 1], 1., places=4)

    def test_alias_method_04(self):
        """deterministic probabilities 1.0 and 0.0"""
        idx = np.arange(2)
        probs = np.arange(2).astype(float)[::-1]
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(
            LFPy.alias_method.alias_method(
                idx, probs, nidx), bins)

        self.assertEqual(nidx, hist[0])
