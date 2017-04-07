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

# for nosetests to run load mechanisms
neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))

class testPointProcess(unittest.TestCase):
    """
    test class LFPy.PointProcess
    """
    def test_PointProcess_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        pp = LFPy.PointProcess(cell=cell, idx=0)
        self.assertTrue(np.alltrue(np.array([pp.x, pp.y, pp.z])==cell.somapos))
        self.assertEqual(pp.idx, 0)
    

class testSynapse(unittest.TestCase):
    """
    test class LFPy.Synapse
    """
    def test_Synapse_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        syn = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                           weight=1., tau=5., record_current=True,
                           record_potential=True)
        syn.set_spike_times(np.array([10.]))
        cell.simulate(rec_isyn=True, rec_vmemsyn=True)
        
        i = np.zeros(cell.tvec.size)
        i[cell.tvec > 10.] = -np.exp(-np.arange((cell.tvec > 10.).sum())*cell.dt / 5.)

        np.testing.assert_allclose(i, syn.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn.v)
        
        

class testStimIntElectrode(unittest.TestCase):
    """
    test class LFPy.StimIntElectrode
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
                                                 'ball_and_sticks.hoc'), dt=1.,
                         v_init=-65.,
                         Ra = 150.,
                         cm = 1.,
                         passive=True,
                         passive_parameters=dict(g_pas=1./30000, e_pas=-65)
                         )
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
                                                 'ball_and_sticks.hoc'), dt=1.,
                         v_init=-65.,
                         Ra = 150.,
                         cm = 1.,
                         passive=True,
                         passive_parameters=dict(g_pas=1./30000, e_pas=-65)
                         )
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


    
    
