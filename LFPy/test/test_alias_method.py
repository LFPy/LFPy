#!/usr/bin/env python
"""A few tests for LFPy, most importantly the calculations of
extracellular field potentials

"""

from __future__ import division
import os
from warnings import warn
import unittest
import numpy as np
from scipy.integrate import quad
from scipy import real, imag
import LFPy
import neuron
import pickle
from warnings import warn
import random

# for nosetests to run load the SinSyn sinusoid synapse currrent mechanism
neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))

class testLFPy(unittest.TestCase):
    """
    
    """


    def test_alias_method_00(self):
        """deterministic probabilities 0.0 and 1.0"""
        idx = np.arange(2)
        probs = np.arange(2).astype(float)
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        self.assertEqual(nidx, hist[1])

    def test_alias_method_01(self):
        """probabilities 0.25 and 0.75"""
        idx = np.arange(2)
        probs = np.array([0.25, 0.75])
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 7 decimal places
        self.assertAlmostEqual(np.corrcoef(probs, hist.astype(float))[0, 1], 1., places=7)

    def test_alias_method_02(self):
        """probabilities 0.75 and 0.25"""
        idx = np.arange(2)
        probs = np.array([0.75, 0.25])
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 7 decimal places
        self.assertAlmostEqual(np.corrcoef(probs, hist.astype(float))[0, 1], 1., places=7)

    def test_alias_method_03(self):
        '''over range of normalized probabilities'''
        size = 5
        idx = np.arange(size)
        probs = np.arange(size).astype(float)**2
        probs /= probs.sum()
        nidx = 1000000
        bins = np.arange(probs.size + 1)

        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 5 decimal places
        self.assertAlmostEqual(np.corrcoef(probs, hist.astype(float))[0, 1], 1., places=4)


    def test_alias_method_04(self):
        """deterministic probabilities 1.0 and 0.0"""
        idx = np.arange(2)
        probs = np.arange(2).astype(float)[::-1]
        nidx = 1000000
        bins = np.arange(3)

        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        self.assertEqual(nidx, hist[0])
