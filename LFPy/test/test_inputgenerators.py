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
import unittest
import numpy as np
import scipy.stats as st
import LFPy


class testInputGenerators(unittest.TestCase):
    """
    test LFPy.inputgenerators module
    """
    def test_inputgenerators_get_rand_spike_times(self):
        """test LFPy.inputgenerators.get_rand_spike_times"""
        nsyn = 10
        nspikes = 100
        tstart = 0.
        tstop=1000.        
        spt = LFPy.inputgenerators.get_rand_spike_times(nsyn=nsyn, nspikes=nspikes, tstart=tstart, tstop=tstop)
        
        self.assertEqual(spt.shape, (nsyn, nspikes))
        self.assertTrue((spt.max() <= tstop) & (spt.min() >= tstart))

    def test_inputgenerators_stationary_poisson(self):
        """test LFPy.inputgenerators.stationary_poisson"""
        nsyn = 10
        rate = 10.
        tstart = 0.
        tstop = 10000.
        
        spt = LFPy.inputgenerators.stationary_poisson(nsyn=nsyn, rate=rate, tstart=tstart, tstop=tstop)
        
        self.assertEqual(len(spt), nsyn)
        for x in spt:
            if x.size > 0:
                self.assertTrue((x.min() >= tstart) & (x.max() <= tstop))

    def test_inputgenerators_stationary_gamma(self):
        """test LFPy.inputgenerators.stationary_gamma"""
        tstart = 0.
        tstop = 10000.
        
        spt = LFPy.inputgenerators.stationary_gamma(tstart=tstart, tstop=tstop)
        
        for x in spt:
            if x.size > 0:
                self.assertTrue((x.min() >= tstart) & (x.max() <= tstop))

    def test_get_activation_times_from_scipy_stats(self):
        """test LFPy.inputgenerators.test_get_activation_times_from_scipy_stats
        """
        n = 10
        tstart = 0.
        tstop = 1E4
        distribution = st.expon
        rvs_args = dict(loc=0, scale=100.)
        times = LFPy.inputgenerators.get_activation_times_from_scipy_stats(
            n=n, tstart=tstart, tstop=tstop, rvs_args=rvs_args, maxiter=1E6
        )
        self.assertTrue(len(times) == n)
        for t in times:
            self.assertTrue((t.min() >= tstart) & (t.max() <= tstop))
        
        