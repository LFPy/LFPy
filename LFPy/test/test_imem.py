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
import unittest
import os
import numpy as np
import LFPy


class testImem(unittest.TestCase):
    def test_cell_v_init_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         cm=1.,
                         Ra=150.,
                         passive=True,
                         )
        cell.simulate(rec_imem=True)

        np.testing.assert_allclose(
            cell.imem, np.zeros_like(
                cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_01(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         cm=1.,
                         Ra=150.,
                         v_init=-70.,
                         passive=True,
                         passive_parameters=dict(e_pas=-70, g_pas=0.001)
                         )
        cell.simulate(rec_imem=True)

        np.testing.assert_allclose(
            cell.imem, np.zeros_like(
                cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_02(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         cm=1.,
                         Ra=150.,
                         v_init=0.,
                         passive=True,
                         passive_parameters=dict(e_pas=0., g_pas=0.001)
                         )
        cell.simulate(rec_imem=True)

        np.testing.assert_allclose(
            cell.imem, np.zeros_like(
                cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_03(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         cm=1.,
                         Ra=150.,
                         passive=False,
                         )
        cell.simulate(rec_imem=True)

        np.testing.assert_allclose(
            cell.imem, np.zeros_like(
                cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_04(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         cm=1.,
                         Ra=150.,
                         v_init=0.,
                         passive=False,
                         )
        cell.simulate(rec_imem=True)

        np.testing.assert_allclose(
            cell.imem, np.zeros_like(
                cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)
