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


import os
import unittest
import LFPy
import pickle


class testTools(unittest.TestCase):
    """
    test LFPy.tools methods
    """

    def test_tools_load_00(self):
        filename = 'test.cpickle'
        x = object()
        f = open(filename, 'wb')
        pickle.dump(x, f)
        f.close()
        self.assertTrue(isinstance(LFPy.tools.load(filename), object))
        os.remove(filename)

    def test_tools_noise_brown(self):
        ncols = 3
        nrows = 2
        self.assertEqual(
            LFPy.tools.noise_brown(
                ncols, nrows).shape, (nrows, ncols))
