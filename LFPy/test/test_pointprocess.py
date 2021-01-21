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

import sys
import os
import posixpath
import unittest
import numpy as np
import LFPy
import neuron

# for nosetests to run load mechanisms
if "win32" in sys.platform:
    pth = os.path.join(LFPy.__path__[0], 'test', 'nrnmech.dll')
    pth = pth.replace(os.sep, posixpath.sep)
    if pth not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth)
        neuron.nrn_dll_loaded.append(pth)
else:
    neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))


class testPointProcess(unittest.TestCase):
    """
    test class LFPy.PointProcess
    """

    def test_PointProcess_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        pp = LFPy.PointProcess(cell=cell, idx=0)
        self.assertTrue(np.alltrue(
            np.array([pp.x, pp.y, pp.z]) == cell.somapos))
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
        cell.simulate()

        i = np.zeros(cell.tvec.size)
        i[cell.tvec > 10.] = - \
            np.exp(-np.arange((cell.tvec > 10.).sum()) * cell.dt / 5.)

        np.testing.assert_allclose(i, syn.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn.v)

    def test_Synapse_01(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        syn0 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=5., record_current=True,
                            record_potential=True)
        syn0.set_spike_times(np.array([10.]))

        syn1 = LFPy.Synapse(cell=cell, idx=1, syntype='ExpSynI',
                            weight=1., tau=5., record_current=True,
                            record_potential=False)
        syn1.set_spike_times(np.array([20.]))

        syn2 = LFPy.Synapse(cell=cell, idx=2, syntype='ExpSynI',
                            weight=1., tau=5., record_current=False,
                            record_potential=True)
        syn2.set_spike_times(np.array([30.]))

        syn3 = LFPy.Synapse(cell=cell, idx=3, syntype='ExpSynI',
                            weight=1., tau=5., record_current=False,
                            record_potential=False)
        syn3.set_spike_times(np.array([40.]))

        cell.simulate()

        i = np.zeros(cell.tvec.size)
        i[cell.tvec > 10.] = - \
            np.exp(-np.arange((cell.tvec > 10.).sum()) * cell.dt / 5.)

        np.testing.assert_allclose(i, syn0.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn0.v)

        self.assertTrue(hasattr(syn1, 'i'))
        i = np.zeros(cell.tvec.size)
        i[cell.tvec > 20.] = - \
            np.exp(-np.arange((cell.tvec > 20.).sum()) * cell.dt / 5.)
        self.assertFalse(hasattr(syn1, 'v'))
        np.testing.assert_allclose(i, syn1.i, rtol=1E-1)

        self.assertFalse(hasattr(syn2, 'i'))
        self.assertTrue(hasattr(syn2, 'v'))

        self.assertFalse(hasattr(syn3, 'i'))
        self.assertFalse(hasattr(syn3, 'v'))

    def test_Synapse_02(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        t0 = 10.
        t1 = 30.
        tau = 5.
        syn0 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)
        syn0.set_spike_times_w_netstim(
            noise=0., start=t0 - 1, number=1)  # -1 to acct for delay

        syn1 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)
        syn1.set_spike_times(np.array([t1]))

        cell.simulate()

        i0 = np.zeros(cell.tvec.size)
        i0[cell.tvec > t0] = - \
            np.exp(-np.arange((cell.tvec > t0).sum()) * cell.dt / tau)

        i1 = np.zeros(cell.tvec.size)
        i1[cell.tvec > t1] = - \
            np.exp(-np.arange((cell.tvec > t1).sum()) * cell.dt / tau)

        np.testing.assert_allclose(i0, syn0.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn0.v)

        np.testing.assert_allclose(i1, syn1.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn1.v)

    def test_Synapse_03(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        t0 = 10.
        t1 = 30.
        tau = 5.
        syn0 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)
        syn0.set_spike_times_w_netstim(
            noise=0., start=t0 - 1, number=1)  # -1 to acct for delay

        syn1 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)
        syn1.set_spike_times_w_netstim(
            noise=0., start=t1 - 1, number=1)  # -1 to acct for delay

        cell.simulate()

        i0 = np.zeros(cell.tvec.size)
        i0[cell.tvec > t0] = - \
            np.exp(-np.arange((cell.tvec > t0).sum()) * cell.dt / tau)

        i1 = np.zeros(cell.tvec.size)
        i1[cell.tvec > t1] = - \
            np.exp(-np.arange((cell.tvec > t1).sum()) * cell.dt / tau)

        np.testing.assert_allclose(i0, syn0.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn0.v)

        np.testing.assert_allclose(i1, syn1.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn1.v)

    def test_Synapse_04(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        t0 = 10.
        t1 = 30.
        tau = 5.
        syn0 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)

        syn1 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)

        syn0.set_spike_times_w_netstim(
            noise=0., start=t0 - 1, number=1)  # -1 to acct for delay
        syn1.set_spike_times(np.array([t1]))

        cell.simulate()

        i0 = np.zeros(cell.tvec.size)
        i0[cell.tvec > t0] = - \
            np.exp(-np.arange((cell.tvec > t0).sum()) * cell.dt / tau)

        i1 = np.zeros(cell.tvec.size)
        i1[cell.tvec > t1] = - \
            np.exp(-np.arange((cell.tvec > t1).sum()) * cell.dt / tau)

        np.testing.assert_allclose(i0, syn0.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn0.v)

        np.testing.assert_allclose(i1, syn1.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn1.v)

    def test_Synapse_05(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        t0 = 10.
        t1 = 30.
        tau = 5.
        syn0 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)

        syn1 = LFPy.Synapse(cell=cell, idx=0, syntype='ExpSynI',
                            weight=1., tau=tau, record_current=True,
                            record_potential=True)

        syn1.set_spike_times(np.array([t1]))
        syn0.set_spike_times_w_netstim(
            noise=0., start=t0 - 1, number=1)  # -1 to acct for delay

        cell.simulate()

        i0 = np.zeros(cell.tvec.size)
        i0[cell.tvec > t0] = - \
            np.exp(-np.arange((cell.tvec > t0).sum()) * cell.dt / tau)

        i1 = np.zeros(cell.tvec.size)
        i1[cell.tvec > t1] = - \
            np.exp(-np.arange((cell.tvec > t1).sum()) * cell.dt / tau)

        np.testing.assert_allclose(i0, syn0.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn0.v)

        np.testing.assert_allclose(i1, syn1.i, rtol=1E-1)
        np.testing.assert_equal(cell.somav, syn1.v)


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
        cell.simulate()
        gt = np.zeros(cell.tvec.size)
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = 1.
        np.testing.assert_equal(gt, stim.i)
        np.testing.assert_equal(cell.somav, stim.v)

    def test_StimIntElectrode_01(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'), dt=1.,
                         v_init=-65.,
                         Ra=150.,
                         cm=1.,
                         passive=True,
                         passive_parameters=dict(g_pas=1. / 30000, e_pas=-65)
                         )
        stim = LFPy.StimIntElectrode(cell=cell,
                                     record_potential=True,
                                     **{'idx': 0,
                                        'pptype': 'VClamp',
                                        'amp': [-65, -55, -65],
                                        'dur': [10, 20, 10],
                                        })
        cell.simulate()
        gt = np.zeros(cell.tvec.size) - 65.
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = -55.
        np.testing.assert_allclose(gt, cell.somav, rtol=1E-3)
        np.testing.assert_equal(cell.somav, stim.v)

    def test_StimIntElectrode_02(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'), dt=1.,
                         v_init=-65.,
                         Ra=150.,
                         cm=1.,
                         passive=True,
                         passive_parameters=dict(g_pas=1. / 30000, e_pas=-65)
                         )
        stim = LFPy.StimIntElectrode(cell=cell,
                                     record_potential=True,
                                     **{'idx': 0,
                                        'pptype': 'SEClamp',
                                        'amp1': -65,
                                        'dur1': 10,
                                        'amp2': -55.,
                                        'dur2': 20,
                                        'amp3': -65,
                                        'dur3': 10,
                                        })
        cell.simulate()
        gt = np.zeros(cell.tvec.size) - 65.
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = -55.
        np.testing.assert_allclose(gt, cell.somav, rtol=1E-2)
        np.testing.assert_equal(cell.somav, stim.v)

    def test_StimIntElectrode_03(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        stim0 = LFPy.StimIntElectrode(cell=cell, idx=0, pptype='IClamp',
                                      amp=1., dur=20., delay=10.,
                                      record_potential=True,
                                      record_current=True)

        stim1 = LFPy.StimIntElectrode(cell=cell, idx=1, pptype='IClamp',
                                      amp=1., dur=20., delay=30.,
                                      record_potential=True,
                                      record_current=False)

        stim2 = LFPy.StimIntElectrode(cell=cell, idx=2, pptype='IClamp',
                                      amp=1., dur=20., delay=50.,
                                      record_potential=False,
                                      record_current=True)

        stim3 = LFPy.StimIntElectrode(cell=cell, idx=3, pptype='IClamp',
                                      amp=1., dur=20., delay=70.,
                                      record_potential=False,
                                      record_current=False)

        cell.simulate()
        gt = np.zeros(cell.tvec.size)
        gt[(cell.tvec > 10.) & (cell.tvec <= 30.)] = 1.
        np.testing.assert_equal(gt, stim0.i)
        np.testing.assert_equal(cell.somav, stim0.v)

        self.assertTrue(hasattr(stim1, 'v'))
        self.assertTrue(cell.tvec.shape == stim1.v.shape)
        self.assertFalse(hasattr(stim2, 'v'))
        self.assertFalse(hasattr(stim3, 'v'))
        self.assertFalse(hasattr(stim1, 'i'))
        self.assertTrue(hasattr(stim2, 'i'))
        self.assertTrue(cell.tvec.shape == stim2.i.shape)
        self.assertFalse(hasattr(stim3, 'i'))
