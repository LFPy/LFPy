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
from LFPy.test.common import *

# for nosetests to run load the SinSyn sinusoid synapse currrent mechanism
if "win32" in sys.platform:
    pth = os.path.join(LFPy.__path__[0], 'test', 'nrnmech.dll')
    pth = pth.replace(os.sep, posixpath.sep)
    if not pth in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth)
        neuron.nrn_dll_loaded.append(pth)
else:
    neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))

class testRecExtElectrode(unittest.TestCase):
    """
    test class LFPy.RecExtElectrode
    """

    def test_method_pointsource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulation(method='pointsource')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)

    def test_method_linesource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulation(method='linesource')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)


    def test_method_soma_as_point(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulation(method='soma_as_point')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)



    def test_method_pointsource_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationDotprodcoeffs(method='pointsource')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)


    def test_method_linesource_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationDotprodcoeffs(method='linesource')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)


    def test_method_soma_as_point_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationDotprodcoeffs(method='soma_as_point')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)


    def test_method_pointsource_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='soma_as_point')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)


    def test_method_linesource_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='linesource')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)


    def test_method_soma_as_point_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='soma_as_point')

        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)

        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        np.testing.assert_allclose(LFP_analytic, LFP_LFPy, rtol=0,
                                   atol=abs(LFP_analytic).max() / 10.)

    def test_sigma_inputs(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : 0,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }
        stick = LFPy.Cell(**stickParams)

        electrodeParams = {
            'sigma' : [0.3, 0.3, 0.3, 0.3],
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
        }

        np.testing.assert_raises(ValueError, LFPy.RecExtElectrode, **electrodeParams)

    def test_bad_cell_position_in_slice(self):
        electrodeParams = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': 200,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11),
            'method': "pointsource",
        }

        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : -10,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }
        stick = LFPy.Cell(**stickParams)
        stick.set_rotation(y=np.pi/2)
        stick.simulate(rec_imem=True)

        stick.set_pos(z=-100)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA.calc_lfp)

        stick.set_pos(z=300)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA.calc_lfp)

    def test_sqeeze_cell_and_bad_position(self):
        electrodeParams = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': 200,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11),
            'method': "pointsource",
            'squeeze_cell_factor': None,
        }

        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : -10,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }
        stick = LFPy.Cell(**stickParams)
        stick.set_rotation(y=np.pi/2)
        stick.simulate(rec_imem=True)

        stick.set_pos(z=1)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA.test_cell_extent)

        stick.set_pos(z=199)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA.test_cell_extent)

        electrodeParams = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': 200,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11),
            'method': "pointsource",
            'squeeze_cell_factor': 0.1,
        }

        stick.set_pos(z=-1)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA.test_cell_extent)

        stick.set_pos(z=201)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA.test_cell_extent)


    def test_return_comp_outside_slice(self):
        electrodeParams = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': 200,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11),
            'method': "pointsource",
            'squeeze_cell_factor': None,
        }

        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : -10,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }
        stick = LFPy.Cell(**stickParams)
        stick.set_rotation(y=np.pi/2)
        stick.set_pos(z=100)
        stick.simulate(rec_imem=True)
        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA._return_comp_outside_slice)
        true_bad_comp = np.array([2, 3, 6])

        stick.zstart[true_bad_comp] = 1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zstart above")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.zstart[true_bad_comp] = 100

        stick.zstart[true_bad_comp] = -1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zstart below")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.zstart[true_bad_comp] = 100

        stick.zend[true_bad_comp] = 1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zend above")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.zend[true_bad_comp] = 100

        stick.zend[true_bad_comp] = -1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zend below")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.zend[true_bad_comp] = 100

    def test_position_shifted_slice(self):
        electrodeParams = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': 200,
            'z_shift': -200,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11) - 100,
            'method': "pointsource",
            'squeeze_cell_factor': None,
        }

        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : -10,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
        }
        stick = LFPy.Cell(**stickParams)
        stick.set_rotation(y=np.pi/2)
        stick.set_pos(z=-100)

        MEA = LFPy.RecMEAElectrode(stick, **electrodeParams)
        MEA.test_cell_extent()

    def test_slice_shift_invariance_pointsource(self):
        h = 200
        z_shift_1 = 0
        z_shift_2 = -352

        electrodeParams_1 = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': h,
            'z_shift': z_shift_1,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11) + z_shift_1,
            'squeeze_cell_factor': None,
        }

        electrodeParams_2 = {
            'sigma_T' : 0.3,
            'sigma_S' : 1.5,
            'sigma_G' : 0.0,
            'h': h,
            'z_shift': z_shift_2,
            'x' : np.linspace(0, 1000, 11),
            'y' : np.zeros(11),
            'z' : np.zeros(11) + z_shift_2,
            'squeeze_cell_factor': None,
        }
        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : -100.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : -np.pi/2,
            'bias' : 0.,
            'record_current' : True
        }
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : -10,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
        }
        stick = LFPy.Cell(**stickParams)
        stick.set_rotation(y=np.pi/2)

        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 0),
                                   **stimParams)

        stick.simulate(rec_imem=True)

        methods = ["pointsource", "linesource", "soma_as_point"]

        for method in methods:
            electrodeParams_1["method"] = method
            electrodeParams_2["method"] = method


            stick.set_pos(z=z_shift_1 + h/2)
            MEA_shift_1 = LFPy.RecMEAElectrode(stick, **electrodeParams_1)
            MEA_shift_1.calc_lfp()

            stick.set_pos(z=z_shift_2 + h/2)
            MEA_shift_2 = LFPy.RecMEAElectrode(stick, **electrodeParams_2)
            MEA_shift_2.calc_lfp()

            np.testing.assert_allclose(MEA_shift_1.LFP,
                                       MEA_shift_2.LFP, rtol=1E-7)

    def test_isotropic_version_of_anisotropic_methods(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : 0,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }
        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : -100.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : -np.pi/2,
            'bias' : 0.,
            'record_current' : True
        }

        isotropic_electrodeParams = {
            'sigma' : 0.3,
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
        }
        anisotropic_electrodeParams = isotropic_electrodeParams.copy()
        anisotropic_electrodeParams["sigma"] = [isotropic_electrodeParams["sigma"]] * 3


        methods = ["pointsource", "linesource", "soma_as_point"]

        for method in methods:
            isotropic_electrodeParams["method"] = method
            anisotropic_electrodeParams["method"] = method

            isotropic_electrode = LFPy.RecExtElectrode(**isotropic_electrodeParams)
            anisotropic_electrode = LFPy.RecExtElectrode(**anisotropic_electrodeParams)

            stick = LFPy.Cell(**stickParams)
            stick.set_pos(z=-stick.zstart[0])

            synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                                   **stimParams)
            stick.simulate([isotropic_electrode, anisotropic_electrode],
                           rec_imem=True, rec_vmem=True)

            np.testing.assert_allclose(isotropic_electrode.LFP,
                                       anisotropic_electrode.LFP, rtol=1E-7)

    def test_compare_anisotropic_lfp_methods(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : 0,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }
        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : -100.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : -np.pi/2,
            'bias' : 0.,
            'record_current' : True
        }

        electrodeParams = {
            'sigma' : [0.3, 0.3, 0.45],
            'x' : np.array([0, 1000]),
            'y' : np.zeros(2),
            'z' : np.zeros(2),

        }

        ps_electrodeParams = electrodeParams.copy()
        ls_electrodeParams = electrodeParams.copy()
        sap_electrodeParams = electrodeParams.copy()

        ps_electrodeParams["method"] = "pointsource"
        ls_electrodeParams["method"] = "linesource"
        sap_electrodeParams["method"] = "soma_as_point"

        electrode_ps = LFPy.RecExtElectrode(**ps_electrodeParams)
        electrode_ls = LFPy.RecExtElectrode(**ls_electrodeParams)
        electrode_sap = LFPy.RecExtElectrode(**sap_electrodeParams)

        stick = LFPy.Cell(**stickParams)
        stick.set_pos(z=-stick.zstart[0])

        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        stick.simulate([electrode_ps, electrode_ls, electrode_sap],
                       rec_imem=True, rec_vmem=True)

        # Test that distant electrode is independent of choice of method
        np.testing.assert_almost_equal(electrode_ps.LFP[1,:],
                                   electrode_ls.LFP[1,:])

        np.testing.assert_almost_equal(electrode_ps.LFP[1,:],
                                   electrode_sap.LFP[1,:])

        # Hack to test that LFP close to stick is dependent on choice of method
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 electrode_ps.LFP[0,:], electrode_ls.LFP[0,:])

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 electrode_ps.LFP[0,:], electrode_sap.LFP[0,:])

    def test_electrical_stimulation(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'passive': True,
            'tstart' : 0,
            'tstop' : 20,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1000,

        }

        electrodeParams = {
            'sigma' : 0.3,
            'x' : np.array([0, 1000]),
            'y' : np.zeros(2),
            'z' : np.zeros(2),
        }

        def sim1():
            electrode1 = LFPy.RecExtElectrode(**electrodeParams)
            stick1 = LFPy.Cell(**stickParams)
            stick1.set_pos(z=-stick1.zstart[0])
            v1, t_ext1 = electrode1.probe.set_current_pulses(0, width1=0.1,
                                                             amp1=10000,
                                                             dt=stickParams['dt'],
                                                             t_stop=stick1.tstop,
                                                             interpulse=0.2)
            stick1.enable_extracellular_stimulation(electrode1, t_ext=t_ext1)
            stick1.simulate(electrode=electrode1,
                            rec_imem=True, rec_vmem=True)
            neuron.h('forall delete_section()')
            return v1, electrode1.LFP, stick1.vmem

        def sim2():
            electrode2 = LFPy.RecExtElectrode(**electrodeParams)
            stick2 = LFPy.Cell(**stickParams)
            stick2.set_pos(z=-stick2.zstart[0])
            v2, t_ext2 = electrode2.probe.set_current_pulses(0, width1=0.1,
                                                             amp1=10000,
                                                             dt=stickParams['dt'],
                                                             t_stop=stick2.tstop,
                                                             interpulse=0.2)
            stick2.enable_extracellular_stimulation(electrode2, t_ext=t_ext2, n=10)
            stick2.simulate(electrode=electrode2,
                            rec_imem=True, rec_vmem=True)
            neuron.h('forall delete_section()')
            return v2, electrode2.LFP, stick2.vmem

        v1, LFP1, vmem1 = sim1()
        v2, LFP2, vmem2 = sim2()

        # Test that distant electrode is independent of choice of method
        np.testing.assert_almost_equal(v1, v2)

        np.testing.assert_almost_equal(LFP1, LFP2)

        np.testing.assert_almost_equal(vmem1, vmem2)
