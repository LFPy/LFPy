#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2020 Computational Neuroscience Group, NMBU.

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
import numpy as np
import LFPy


class testMorphologyImport(unittest.TestCase):
    """
    test that importing a morphology in NEURON results in the correct
    morphology in LFPy.Cell
    """

    def test_morphology_import_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'L23E_p23.hoc'),
                         nsegs_method=None)
        data = np.c_[cell.x[:, 0], cell.x.mean(axis=-1), cell.x[:, -1],
                     cell.y[:, 0], cell.y.mean(axis=-1), cell.y[:, -1],
                     cell.z[:, 0], cell.z.mean(axis=-1), cell.z[:, -1]]
        expected_data = np.array([
            [0., 0., 0., 0., 0., 0., -10., 0., 10.],
            [0., 50., 100., 0., -50., -100., 10., 55., 100.],
            [0., -50., -100., 0., 50., 100., 10., 55., 100.],
            [0., 50., 100., 0., 50., 100., -10., -55., -100.],
            [0., -50., -100., 0., -50., -100., -10., -55., -100.],
            [0., 0., 0., 0., 0., 0., 10., 167., 324.]])

        np.testing.assert_equal(data, expected_data)
