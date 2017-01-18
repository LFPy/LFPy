#!/usr/bin/env python
"""A few tests for LFPy, most importantly the calculations of
extracellular field potentials

"""

import os
from warnings import warn
import unittest
import numpy as np
from scipy.integrate import quad
from scipy import real, imag
import LFPy
import neuron
import pickle
from warnings import warn

# for nosetests to run load the SinSyn sinusoid synapse currrent mechanism
neuron.load_mechanisms(LFPy.__path__[0])

class testLFPy(unittest.TestCase):
    """
    A set of test functions for each method of calculating the LFP, where the
    model outcome from LFPy is compared with analytically obtained results for
    a stick neuron with sinusoid synaptic current input at the end, and LFP
    calculated alongside the neuron.
    
    The tests should pass with 3 significant numbers in the LFP, as effects
    of compartmentalising the stick is prominent.
    
    Some tests of cell.tvec is also executed
    """
    
    def test_method_pointsource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulation(method='pointsource')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_linesource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulation(method='linesource')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_som_as_point(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulation(method='som_as_point')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)


    def test_method_pointsource_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulationDotprodcoeffs(method='pointsource')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_linesource_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulationDotprodcoeffs(method='linesource')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_som_as_point_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulationDotprodcoeffs(method='som_as_point')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)


    def test_method_pointsource_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='som_as_point')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_linesource_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='linesource')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)

    def test_method_som_as_point_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='som_as_point')
        
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)

    
    def test_tvec_00(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : 0.,
            'tstopms' : 100.,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)
        
    def test_cell_tvec_01(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : 0.,
            'tstopms' : 100.,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_02(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : 0.,
            'tstopms' : 10000.,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_03(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : 0.,
            'tstopms' : 10000.,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        
        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])


    def test_cell_tvec_04(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : 0,
            'tstopms' : 100,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)
    
    def test_cell_tvec_05(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : 0.,
            'tstopms' : 100.,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertAlmostEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_06(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : 0,
            'tstopms' : 10000,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)
    
    def test_cell_tvec_07(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : 0.,
            'tstopms' : 10000.,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_08(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : -100,
            'tstopms' : 100,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)

        
    def test_cell_tvec_09(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : -100,
            'tstopms' : 100,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_10(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : -100,
            'tstopms' : 10000,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)

        
    def test_cell_tvec_11(self):
        stickParams = {
            'dt' : 2**-3,
            'tstartms' : -100,
            'tstopms' : 10000,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_12(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : -100,
            'tstopms' : 100,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_13(self):
        stickParams = {
            'dt' : 0.10,
            'tstartms' : -100,
            'tstopms' : 100,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertAlmostEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_14(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : -100,
            'tstopms' : 10000,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_15(self):
        stickParams = {
            'dt' : 0.1,
            'tstartms' : -100,
            'tstopms' : 10000,
        }
        
        tvec = self.stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstopms'],
                    stickParams['tstopms']/stickParams['dt'] + 1)
        
        for i in range(tvec.size):
            self.assertAlmostEqual(tvec[i], tvec_numpy[i])    

    def test_alias_method_01(self):
        """deterministic probabilities 0.0 and 1.0"""
        idx = np.arange(2)
        probs = np.arange(2).astype(float)
        nidx = 1000000
        bins = np.arange(3)
        
        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)
        
        self.assertEqual(nidx, hist[1])
        
    def test_alias_method_02(self):
        """probabilities 0.25 and 0.75"""
        idx = np.arange(2)
        probs = np.array([0.25, 0.75])
        nidx = 1000000
        bins = np.arange(3)
        
        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 7 decimal places
        self.assertAlmostEqual(np.corrcoef(probs, hist.astype(float))[0, 1], 1., places=7)

    def test_alias_method_03(self):
        """probabilities 0.75 and 0.25"""
        idx = np.arange(2)
        probs = np.array([0.75, 0.25])
        nidx = 1000000
        bins = np.arange(3)
        
        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 7 decimal places
        self.assertAlmostEqual(np.corrcoef(probs, hist.astype(float))[0, 1], 1., places=7)

    def test_alias_method_04(self):
        '''over range of normalized probabilities'''
        size = 5
        idx = np.arange(size)
        probs = np.arange(size).astype(float)**2
        probs /= probs.sum()
        nidx = 1000000
        bins = np.arange(probs.size + 1)
        
        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 5 decimal places
        self.assertAlmostEqual(np.corrcoef(probs, hist.astype(float))[0, 1], 1., places=4)


    def test_alias_method_05(self):
        """deterministic probabilities 1.0 and 0.0"""
        idx = np.arange(2)
        probs = np.arange(2).astype(float)[::-1]
        nidx = 1000000
        bins = np.arange(3)
        
        hist, _ = np.histogram(LFPy.alias_method.alias_method(idx, probs, nidx),
                               bins)
        
        self.assertEqual(nidx, hist[0])
    
    def test_cell_set_pos_00(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        np.testing.assert_allclose(cell.somapos, [0, 0, 0])
        
    def test_cell_set_pos_01(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_02(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                          pt3d=True)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_03(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_04(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                          pt3d=True)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])


    def test_cell_set_pos_05(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        np.testing.assert_allclose(cell.somapos,
                                   [cell.xmid[0], cell.ymid[0], cell.zmid[0]])
            

    def test_cell_set_pos_06(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                         pt3d=True)
        np.testing.assert_allclose(cell.somapos,
                                   [cell.xmid[0], cell.ymid[0], cell.zmid[0]])


    def test_cell_set_rotation_00(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around x-axis
        cell.set_rotation(x=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.ystart, -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, -yends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, -zends, atol=1e-07)


    def test_cell_set_rotation_01(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around y-axis
        cell.set_rotation(y=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, -xends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, -zends, atol=1e-07)


    def test_cell_set_rotation_02(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        # test rotation 180 deg around z-axis
        cell.set_rotation(z=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, -xends, atol=1e-07)
        np.testing.assert_allclose(cell.ystart, -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, -yends, atol=1e-07)

    def test_cell_set_rotation_03(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                         pt3d=True)
        
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around x-axis
        cell.set_rotation(x=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.ystart, -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, -yends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, -zends, atol=1e-07)


    def test_cell_set_rotation_04(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                         pt3d=True)
        
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around y-axis
        cell.set_rotation(y=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, -xends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, -zends, atol=1e-07)


    def test_cell_set_rotation_05(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                         pt3d=True)
        
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        # test rotation 180 deg around z-axis
        cell.set_rotation(z=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, -xends, atol=1e-07)
        np.testing.assert_allclose(cell.ystart, -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, -yends, atol=1e-07)

    def test_cell_set_rotation_06(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'stick.hoc'))

        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation: 90 deg around x-axis, 90 deg around y-axis, 90 deg around z-axis
        cell.set_rotation(x=np.pi / 2., y=np.pi, z=np.pi / 4.)
        # revert rotation: -90 deg around x-axis, -90 deg around y-axis, -90 deg around z-axis, rotation_order='zyx'
        cell.set_rotation(x=-np.pi / 2., y=-np.pi, z=-np.pi / 4., rotation_order='zyx')
        # assert that x-, y- and z-coordinates are same as beginning, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, xends, atol=1e-07)
        np.testing.assert_allclose(cell.ystart, ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, yends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, zends, atol=1e-07)

    def test_cell_chiral_morphology_00(self):
        '''test LFPy.Cell.chiral_morphology()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around x-axis
        cell.chiral_morphology(axis='x')
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, -xends, atol=1e-07)
        np.testing.assert_allclose(cell.ystart, ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, yends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, zends, atol=1e-07)


    def test_cell_chiral_morphology_00(self):
        '''test LFPy.Cell.chiral_morphology()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around x-axis
        cell.chiral_morphology(axis='y')
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.xstart, xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, xends, atol=1e-07)
        np.testing.assert_allclose(cell.ystart, -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, -yends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, zends, atol=1e-07)
        
    def test_cell_chiral_morphology_00(self):
        '''test LFPy.Cell.chiral_morphology()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        
        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around x-axis
        cell.chiral_morphology(axis='z')
        # assert that y- and z-coordinates are inverted, using absolute

        # tolerances
        np.testing.assert_allclose(cell.xstart, xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.xmid, xmids, atol=1e-07)
        np.testing.assert_allclose(cell.xend, xends, atol=1e-07)
        np.testing.assert_allclose(cell.ystart, ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.ymid, ymids, atol=1e-07)
        np.testing.assert_allclose(cell.yend, yends, atol=1e-07)
        np.testing.assert_allclose(cell.zstart, -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.zmid, -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.zend, -zends, atol=1e-07)


    def test_cell_get_rand_prob_area_norm_00(self):
        '''test LFPy.Cell.get_rand_prob_area_norm()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                 'ball_and_sticks.hoc' ))
        p = cell.get_rand_prob_area_norm()
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertTrue(p.min() >= 0.)
        self.assertTrue(p.max() <= 1.)
        

    def test_cell_get_rand_prob_area_norm_from_idx(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                 'ball_and_sticks.hoc' ))
        p = cell.get_rand_prob_area_norm_from_idx(idx=cell.get_idx(section='allsec'))
        self.assertListEqual(cell.get_rand_prob_area_norm().tolist(), p.tolist())
        

    def test_cell_get_rand_prob_area_norm_from_idx_00(self):
        '''test LFPy.Cell.get_rand_prob_area_norm()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        p = cell.get_rand_prob_area_norm_from_idx(idx=np.array([0]))
        np.testing.assert_equal(p, np.array([1.]))

    
    def test_cell_get_intersegment_vector_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        idx0 = 0
        idx1 = 1
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)

        self.assertListEqual(vector,
                            [cell.xmid[idx1] - cell.xmid[idx0],
                             cell.ymid[idx1] - cell.ymid[idx0],
                             cell.zmid[idx1] - cell.zmid[idx0]])


    def test_cell_get_intersegment_distance_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        idx0 = 0
        idx1 = 1
        distance = cell.get_intersegment_distance(idx0=idx0, idx1=idx1)
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)
        
        self.assertEqual(np.sqrt(np.array(vector)**2).sum(), distance)


    def test_cell_get_idx_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                         nsegs_method=None)
        self.assertListEqual(cell.get_idx(section='soma').tolist(), [0])
        self.assertListEqual(cell.get_idx(section='soma[0]').tolist(), [0])
        self.assertListEqual(cell.get_idx(section='dend[0]').tolist(), [1])
        self.assertListEqual(cell.get_idx(section='dend[1]').tolist(), [2])
        self.assertListEqual(cell.get_idx(section='dend[2]').tolist(), [3])
        self.assertListEqual(cell.get_idx(section='dend').tolist(), [1, 2, 3])
        self.assertListEqual(cell.get_idx(section='allsec').tolist(),
                             [0, 1, 2, 3])
        self.assertListEqual(cell.get_idx(section=['soma', 'dend']).tolist(),
                             [0, 1, 2, 3])
        self.assertListEqual(cell.get_idx(section='apic').tolist(), [])


    def test_cell_get_closest_idx_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ),
                         nsegs_method=None)
        self.assertEqual(cell.get_closest_idx(x=0, y=0, z=0),
                             cell.get_idx(section='soma')[0])
                         
        self.assertEqual(cell.get_closest_idx(x=-25, y=0, z=175),
                             cell.get_idx(section='dend[1]')[0])
        

    def test_cell_get_idx_children_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
    
        np.testing.assert_array_equal(cell.get_idx_children(parent='soma[0]'),
                                      cell.get_idx(section='dend[0]'))
        
      
    def test_cell_get_idx_parent_children_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        np.testing.assert_array_equal(cell.get_idx_parent_children(parent='soma[0]'),
                                      cell.get_idx(section=['soma[0]',
                                                            'dend[0]']))
    
    
    def test_cell_get_idx_name_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))    
        np.testing.assert_array_equal(cell.get_idx_name(idx=np.array([0])),
                                                np.array([[0, 'soma[0]', 0.5]],
                                                         dtype=object))


    def test_cell_get_rand_idx_area_norm_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        idx = cell.get_rand_idx_area_norm(nidx=1000000)
        
        
        # compute histogram and correlate with segment area
        bins = np.arange(cell.totnsegs+1)
        hist, bin_edges = np.histogram(idx, bins=bins)
        
        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 5 decimal places
        self.assertAlmostEqual(np.corrcoef(cell.area, hist)[0, 1], 1., places=5)

        # check if min and max is in the range of segment indices
        self.assertEqual(idx.min(), 0)
        self.assertEqual(idx.max(), cell.totnsegs-1)
    
    
    def test_cell_set_synapse_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        cell.set_synapse(idx=0, syntype='ExpSyn', record_curret=False,
                         record_potential=False, weight=1.,
                         **dict(e=10., tau=2.))
        
        self.assertEqual(cell.synlist[0].hname(), 'ExpSyn[0]')
        self.assertEqual(len(cell.synlist), 1)
        self.assertEqual(len(cell.netconlist), 1)
        self.assertEqual(len(cell.netstimlist), 1)
        self.assertEqual(cell.synlist[0].e, 10.)
        self.assertEqual(cell.synlist[0].tau, 2.)
        self.assertEqual(cell.netconlist[0].weight[0], 1.)
        
    
    def test_cell_set_point_process_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        cell.set_point_process(idx=0, pptype='IClamp', record_current=False,
                               **dict(delay=1., amp=1.))
        self.assertEqual(cell.stimlist[0].hname(), 'IClamp[0]')
        self.assertEqual(len(cell.stimlist), 1)
        self.assertEqual(cell.stimlist[0].delay, 1.)
        self.assertEqual(cell.stimlist[0].amp, 1.)
        
    
    def test_cell_strip_hoc_objects_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        cell.strip_hoc_objects()
        for attribute in dir(cell):
            self.assertNotEqual(str(type(getattr(cell, attribute))),
                                'hoc.HocObject')

    def test_cell_cellpickler_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0],
                                                  'ball_and_sticks.hoc' ))
        cell_pickle = cell.cellpickler(filename=None, pickler=pickle.dumps)
        pickled_cell = pickle.loads(cell_pickle)
        
        for attribute in dir(cell):
            if attribute.startswith('__') or attribute.startswith('_'):
                pass
            else:
                self.assertEqual(type(getattr(cell, attribute)),
                                 type(getattr(pickled_cell, attribute)))

    
    ######## Functions used by tests: ##########################################
    def stickSimulationTesttvec(self, **kwargs):
        stick = LFPy.Cell(morphology = os.path.join(LFPy.__path__[0],
                                                    'stick.hoc'), verbose=True,
                          **kwargs)
        stick.simulate(rec_imem=False)    
        return stick.tvec
    
    def stickSimulation(self, method):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'dt' : 0.01,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
            
        }
        
        electrodeParams = {
            'sigma' : 0.3,
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
            'method' : method
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
        
        
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        
        stick = LFPy.Cell(**stickParams)
        
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        stick.simulate(electrode, rec_imem=True, rec_istim=True, rec_vmem=True)
        
        return electrode.LFP

    def stickSimulationAveragingElectrode(self,
                        contactRadius, contactNPoints, method):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'dt' : 0.01,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
            
        }
        
        N = np.empty((11, 3))
        for i in range(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
        electrodeParams = {
            'sigma' : 0.3,
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
            'r' : contactRadius,
            'n' : 10,
            'N' : N,
            'method' : method
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
        
        
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        
        stick = LFPy.Cell(**stickParams)
        
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        stick.simulate(electrode, rec_imem=True, rec_istim=True, rec_vmem=True)
        
        return electrode.LFP

    def stickSimulationDotprodcoeffs(self, method):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'dt' : 0.01,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
            
        }
        
        electrodeParams = {
            'sigma' : 0.3,
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
            'method' : method
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
        
        
        
        stick = LFPy.Cell(**stickParams)
        #dummy variables for mapping
        stick.imem = np.eye(stick.totnsegs)
        stick.tvec = np.arange(stick.totnsegs)*stick.dt
        
        electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
        electrode.calc_lfp()
        #not needed anymore:
        del stick.imem, stick.tvec
        
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        stick.simulate(dotprodcoeffs=electrode.LFP,
                       rec_imem=True, rec_istim=True, rec_vmem=True)
        
        return stick.dotprodresults[0]

    
    def analytical_LFP(self, time=np.linspace(0, 100, 1001),
                       stickLength=1000.,
                       stickDiam=2.,
                       Rm=30000.,
                       Cm=1.,
                       Ri=150.,
                       stimFrequency=100.,
                       stimAmplitude=1.,
                       sigma=0.3,
                       electrodeR=100.,
                       electrodeZ=0.):
        """
        Will calculate the analytical LFP from a dendrite stick aligned with z-axis.
        The synaptic current is always assumed to be at the end of the stick, i.e.
        Zin = stickLength.
        
        Arguments:
        ::
            
            time : The LFP is calculated for values in this np.array (ms)
            stickLength : length of stick (mum)
            stickDiam : diameter of stick (mum)
            Rm : Membrane resistivity (Ohm * cm2)
            Cm : Membrane capacitance (muF/cm2)
            Ri : Intracellular resistivity (Ohm*cm)
            stimFrequency : Frequency of cosine synapse current (Hz)
            stimAmplitude : Amplitude of cosine synapse current (nA)
            sigma : Extracellular conductivity (muS/mum)
            electrodeR : Radial distance from stick (mum)
            electrodeZ : Longitudal distance along stick(mum)
        """    
        Gm = 1. / Rm            # specific membrane conductivity (S/cm2)
        gm = 1E2 * np.pi * stickDiam / Rm     # absolute membrane conductance (muS / mum)
        ri = 1E-2 * 4. * Ri / (np.pi * stickDiam**2) # intracellular resistance  (Mohm/mum)
        
        Lambda = 1E2 / np.sqrt(gm * ri) # Electrotonic length constant of stick (mum)
        Ginf = 10 / (ri * Lambda)   # iinfinite stick input cond (muS)?
        
        tau_m = Rm * Cm / 1000        # membrane time constant (ms)
        Omega = 2 * np.pi * stimFrequency * tau_m / 1000 #impedance
        Zel = electrodeZ / Lambda    # z-position of extracellular point, in units of Lambda
        L = stickLength / Lambda      # Length of stick in units of Lambda
        Rel = electrodeR / Lambda    # extracellular, location along x-axis, or radius, in units of Lambda
        q = np.sqrt(1 + 1j*Omega)	    # Note: j is sqrt(-1)
        Yin = q * Ginf * np.tanh(q * L)	    # Admittance, Zin is input position?
        Zin = stickLength / Lambda  # unitless location of synapse
        
        PhiExImem = np.empty(time.size)
        PhiExInput = np.empty(time.size)
        
        def i_mem(z): #z is location at stick
            return gm * q**2 * np.cosh(q * z) / np.cosh(q * L) * stimAmplitude / Yin
            
        def f_to_integrate(z):
            return 1E-3 / (4 * np.pi * sigma) * i_mem(z) \
                / np.sqrt(Rel**2 + (z - Zel)**2)
        
        #calculate contrib from membrane currents
        Vex_imem = -self.complex_quadrature(f_to_integrate, 0, L, epsabs=1E-20)
        
        #adding contrib from input current to Vex
        Vex_input = stimAmplitude / (4 * np.pi * sigma * Lambda * np.sqrt(Rel**2 + (Zin-Zel)**2))
        
        PhiExImemComplex = Vex_imem * np.exp(1j * 2 * np.pi * stimFrequency *
                                                  time / 1000)
        PhiExInputComplex = Vex_input * np.exp(1j * 2 * np.pi * stimFrequency *
                                                 time / 1000)
        
        #Using only real component
        PhiExImem = PhiExImemComplex.real
        PhiExInput = PhiExInputComplex.real
        
        PhiEx = PhiExImem + PhiExInput
        return PhiEx
    
    def complex_quadrature(self, func, a, b, **kwargs):
        """
        Will return the complex integral value.
        """
        def real_func(x):
            return real(func(x))
        def imag_func(x):
            return imag(func(x))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return real_integral[0] + 1j*imag_integral[0]



def _test(verbosity=2):
    """
    Run tests for the LFPy module implemented using the unittest module.

    Notes
    -----
    if the NEURON extension file LFPy/sinsyn.mod could not be compiled using the
    neuron-provided nrnivmodl script (linux/OSX) upon installation of LFPy,
    tests will fail. Consider reinstalling LFPy e.g., issuing
    
        >>> pip install LFPy --upgrade
    
    or
    
        >>> cd /path/to/LFPy/sources
        >>> python setup.py install

    Parameters
    ----------
        verbosity : int
            unittest.TextTestRunner verbosity level
    """
    #check if sinsyn.mod is compiled, if it isn't, some tests will fail
    if not hasattr(neuron.h, 'SinSyn'):
        warn('tests will fail because the sinsyn.mod mechanism is not compiled')

    #load and execute testing suite
    suite = unittest.TestLoader().loadTestsFromTestCase(testLFPy)

    unittest.TextTestRunner(verbosity=verbosity).run(suite)