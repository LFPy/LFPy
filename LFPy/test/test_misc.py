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
import numpy as np
import neuron


class testMisc(unittest.TestCase):
    """
    test misc. code
    """

    def test_neuron_record_i_membrane_methods_00(self):
        '''not a test of LFPy per se, but we're using this method for
        calculating with the i_membrane_ attribute on each time step'''
        # sections
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')

        # connect sections
        dend.connect(soma, 1, 0)

        # geometry
        soma.L = 30.
        soma.diam = 30.
        soma.nseg = 1
        dend.L = 500.
        dend.diam = 2.
        dend.nseg = 50

        # biophysical parameters
        for sec in [soma, dend]:
            sec.Ra = 100
            sec.cm = 1
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 0.0002
                seg.pas.e = -65.

        # stimulus
        syn = neuron.h.ExpSyn(0.5, sec=dend)
        syn.e = 0.
        syn.tau = 2.

        # generators
        ns = neuron.h.NetStim(0.5)
        ns.noise = 1.
        ns.start = 0.
        ns.number = 1000
        ns.interval = 10.
        nc = neuron.h.NetCon(ns, syn)
        nc.weight[0] = .01

        # integrator
        cvode = neuron.h.CVode()
        cvode.use_fast_imem(1)

        # record
        i_membrane_control = []  # record currents using Vector.record method
        i_membrane_fadvance = []  # record seg.i_membrane_ at each timestep
        for sec in [soma, dend]:
            for seg in sec:
                i = neuron.h.Vector()
                i.record(seg._ref_i_membrane_)
                i_membrane_control.append(i)
                i_membrane_fadvance.append([])

        # Simulation control
        neuron.h.dt = 2**-4  # simulation time resolution
        tstop = 500.  # simulation duration
        v_init = -65.  # membrane voltage(s) at t = 0

        def initialize():
            neuron.h.finitialize(v_init)
            neuron.h.fcurrent()

        def collect_i_membrane():
            j = 0
            for sec in [soma, dend]:
                for seg in sec:
                    i_membrane_fadvance[j].append(seg.i_membrane_)
                    j += 1

        def integrate():
            while neuron.h.t < tstop:
                collect_i_membrane()
                neuron.h.fadvance()
            collect_i_membrane()  # otherwise shape mismatch

        initialize()
        integrate()

        i_membrane_control = np.array(i_membrane_control)
        i_membrane_fadvance = np.array(i_membrane_fadvance)

        np.testing.assert_equal(i_membrane_control, i_membrane_fadvance)

    def test_neuron_record_i_membrane_methods_01(self):
        '''not a test of LFPy per se, but we're using this method for
        calculating with the i_membrane_ attribute on each time step'''
        # sections
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')

        # connect sections
        dend.connect(soma, 1, 0)

        # geometry
        soma.L = 30.
        soma.diam = 30.
        soma.nseg = 1
        dend.L = 500.
        dend.diam = 2.
        dend.nseg = 50

        # biophysical parameters
        for sec in [soma, dend]:
            sec.Ra = 100
            sec.cm = 1
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 0.0002
                seg.pas.e = -65.

        # stimulus
        syn = neuron.h.ExpSyn(0.5, sec=dend)
        syn.e = 0.
        syn.tau = 2.

        # generators
        ns = neuron.h.NetStim(0.5)
        ns.noise = 1.
        ns.start = 0.
        ns.number = 1000
        ns.interval = 10.
        nc = neuron.h.NetCon(ns, syn)
        nc.weight[0] = .01

        # integrator
        cvode = neuron.h.CVode()
        cvode.use_fast_imem(1)

        # record
        i_membrane_control = []  # record currents using Vector.record method
        i_membrane_fadvance = []  # record seg.i_membrane_ at each timestep
        for sec in [soma, dend]:
            for seg in sec:
                i = neuron.h.Vector()
                i.record(seg._ref_i_membrane_)
                i_membrane_control.append(i)
                i_membrane_fadvance.append([])

        # Simulation control
        neuron.h.dt = 2**-4          # simulation time resolution
        tstop = 500.        # simulation duration
        v_init = -65.       # membrane voltage(s) at t = 0

        def initialize():
            neuron.h.finitialize(v_init)
            neuron.h.fcurrent()
            neuron.h.frecord_init()
            neuron.h.t = -100.  # force simulations to start at some negative t

        def collect_i_membrane():
            j = 0
            for sec in [soma, dend]:
                for seg in sec:
                    i_membrane_fadvance[j].append(seg.i_membrane_)
                    j += 1

        def integrate():
            while neuron.h.t < tstop:
                collect_i_membrane()
                neuron.h.fadvance()
            collect_i_membrane()  # otherwise shape mismatch

        initialize()
        integrate()

        i_membrane_control = np.array(i_membrane_control)
        i_membrane_fadvance = np.array(i_membrane_fadvance)

        np.testing.assert_equal(i_membrane_control, i_membrane_fadvance)
