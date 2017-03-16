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

    """
    def test_inputgenerators_get_rand_spike_times(self):
        """test LFPy.inputgenerators.get_rand_spike_times"""
        synpos = np.arange(10)
        nspikes = 100
        tstart = 0.
        tstop=1000.        
        spt = LFPy.inputgenerators.get_rand_spike_times(synpos=synpos, nspikes=nspikes, tstart=tstart, tstop=tstop)
        
        self.assertEqual(spt.shape, (synpos.size, nspikes))
        self.assertTrue((spt.max() <= tstop) & (spt.min() >= tstart))
