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
import neuron


class testPointProcess(unittest.TestCase):
    """

    """
    def test_PointProcess_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        pp = LFPy.PointProcess(cell=cell, idx=0)
        self.assertTrue(np.alltrue(np.array([pp.x, pp.y, pp.z])==cell.somapos))
        self.assertEqual(pp.idx, 0)
    

class testSynapse(unittest.TestCase):
    """

    """
    pass
    # def test_Synapse_00(self):
    #     cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
    #                                              'ball_and_sticks.hoc'))
    #     syn = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSyn')
        
        
        

class testStimIntElectrode(unittest.TestCase):
    """

    """
    def test_StimIntElectrode_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        stim = LFPy.StimIntElectrode(cell=cell, idx=0, pptype='IClamp',
                            amp=1., dur=20., delay=10.,
                            record_potential=True,
                            record_current=True)
        cell.simulate(rec_istim=True, rec_vmemstim=True)
        # stim.collect_potential(cell) 
        gt = np.zeros(cell.tvec.size)
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = 1.
        np.testing.assert_equal(gt, stim.i)
        np.testing.assert_equal(cell.somav, stim.v)

    def test_StimIntElectrode_01(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'), dt=1.)
        stim = LFPy.StimIntElectrode(cell=cell,
                                     record_potential=True,
                                     **{'idx' : 0,
                                        'pptype' : 'VClamp',
                                        'amp[0]' : -65,
                                        'dur[0]' : 10,
                                        'amp[1]' : -55.,
                                        'dur[1]' : 20,
                                        'amp[2]' : -65,
                                        'dur[2]' : 10,
                                   })
        cell.simulate(rec_vmemstim=True)
        # stim.collect_potential(cell) 
        gt = np.zeros(cell.tvec.size)-65.
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = -55.
        np.testing.assert_allclose(gt, cell.somav, rtol=1E-3)
        np.testing.assert_equal(cell.somav, stim.v)

    def test_StimIntElectrode_02(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'), dt=1.)
        stim = LFPy.StimIntElectrode(cell=cell,
                                     record_potential=True,
                                     **{'idx' : 0,
                                        'pptype' : 'SEClamp',
                                        'amp1' : -65,
                                        'dur1' : 10,
                                        'amp2' : -55.,
                                        'dur2' : 20,
                                        'amp3' : -65,
                                        'dur3' : 10,
                                   })
        cell.simulate(rec_vmemstim=True)
        # stim.collect_potential(cell) 
        gt = np.zeros(cell.tvec.size)-65.
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = -55.
        np.testing.assert_allclose(gt, cell.somav, rtol=1E-2)
        np.testing.assert_equal(cell.somav, stim.v)
