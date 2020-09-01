#!/usr/bin/env python

from __future__ import division
import unittest
import os
import numpy as np
import LFPy

class test_imem(unittest.TestCase):
    def test_cell_v_init_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks.hoc' ),
                         cm=1.,
                         Ra=150.,
                         passive=True,
                         )
        cell.simulate(rec_imem=True)
        
        np.testing.assert_allclose(cell.imem, np.zeros_like(cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_01(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks.hoc' ),
                         cm=1.,
                         Ra=150.,
                         v_init=-70.,
                         passive=True,
                         passive_parameters = dict(e_pas=-70, g_pas=0.001)
                         )
        cell.simulate(rec_imem=True)
        
        np.testing.assert_allclose(cell.imem, np.zeros_like(cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_02(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks.hoc' ),
                         cm=1.,
                         Ra=150.,
                         v_init=0.,
                         passive=True,
                         passive_parameters = dict(e_pas=0., g_pas=0.001)
                         )
        cell.simulate(rec_imem=True)
        
        np.testing.assert_allclose(cell.imem, np.zeros_like(cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_03(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks.hoc' ),
                         cm=1.,
                         Ra=150.,
                         passive=False,
                         )
        cell.simulate(rec_imem=True)
        
        np.testing.assert_allclose(cell.imem, np.zeros_like(cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)

    def test_cell_v_init_04(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks.hoc' ),
                         cm=1.,
                         Ra=150.,
                         v_init=0.,
                         passive=False,
                         )
        cell.simulate(rec_imem=True)
        
        np.testing.assert_allclose(cell.imem, np.zeros_like(cell.imem), atol=1E-10)
        np.testing.assert_allclose(cell.somav, cell.v_init)
