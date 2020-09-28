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
import scipy.stats as st
import LFPy


class testInputGenerators(unittest.TestCase):
    """
    test LFPy.inputgenerators module
    """

    def test_get_activation_times_from_distribution(self):
        """test LFPy.inputgenerators.get_activation_times_from_distribution
        """
        n = 10
        tstart = 0.
        tstop = 1E4
        distribution = st.expon
        rvs_args = dict(loc=0, scale=100.)
        times = LFPy.inputgenerators.get_activation_times_from_distribution(
            n=n, tstart=tstart, tstop=tstop, distribution=distribution,
            rvs_args=rvs_args, maxiter=1E6
        )
        self.assertTrue(len(times) == n)
        for t in times:
            self.assertTrue((t.min() >= tstart) & (t.max() <= tstop))
