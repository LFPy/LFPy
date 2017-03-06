#!/usr/bin/env python
"""A few tests for LFPy, most importantly the calculations of
extracellular field potentials

"""

from __future__ import division
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
import random

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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
        time = np.linspace(0, 100, 100*2**6+1)
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
            'tstopms' : 10000,
        }

        try:
            self.stickSimulationTesttvec(**stickParams)
        except AssertionError:
            pass

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

        self.assertTrue('ExpSyn' in cell.synlist[0].hname())
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

    def test_single_dend(self):
        '''
        Check Kirchhoff in single dend.
        '''
        neuron.h('forall delete_section()')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend2.connect(dend1(1.), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[dend1, dend2])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[1], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[1], cell.imem[0], rtol=1E-5)
        np.testing.assert_almost_equal(iaxial[2], cell.imem[1], decimal=9)
        np.testing.assert_allclose(iaxial[2], cell.imem[1], rtol=1E-5)

    def test_soma_dend_mid(self):
        '''
        Check Kirchhoff in soma when single dend connected to soma mid.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')
        dend.connect(soma(0.5), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_allclose(iaxial[0], np.zeros(cell.tvec.size))
        np.testing.assert_almost_equal(-iaxial[2]-iaxial[3], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[2]-iaxial[3], cell.imem[0], rtol=1E-4)

    def test_soma_dend_end(self):
        '''
        Check Kirchhoff in soma when single dend connected to soma end.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')
        dend.connect(soma(1.0), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[1], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[1], cell.imem[0], rtol=1E-4)

    def test_soma_dend_rand_conn_not_0(self):
        '''
        Check Kirchhoff in soma when single dend connected to random soma point.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')
        dend.connect(soma(random.uniform(1e-2, 1.)), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[2], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[2], cell.imem[0], rtol=1E-4)

    def test_soma_dends_mid(self):
        '''
        Check Kirchhoff in soma when two dends connected to soma mid.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(0.5), 0)
        dend2.connect(soma(0.5), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend1, dend2])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[2]-iaxial[4], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[2]-iaxial[4], cell.imem[0], rtol=1E-4)

    def test_soma_dends_end(self):
        '''
        Check Kirchhoff in soma when two dends connected to soma end.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(1.), 0)
        dend2.connect(soma(1.), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend1, dend2])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[1]-iaxial[3], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[1]-iaxial[3], cell.imem[0], rtol=1E-4)

    def test_soma_dends_diff(self):
        '''
        Check Kirchhoff in soma when two dends connected to diff soma points.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(1.0), 0)
        dend2.connect(soma(.5), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend1, dend2])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[2]-iaxial[4], cell.imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[2]-iaxial[4], cell.imem[0], rtol=1E-4)

    def test_soma_y_diff(self):
        '''
        Check Kirchhoff in mid dend when two dends connected to dend.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend1.connect(soma(1.0), 0)
        dend2.connect(dend1(.5), 0)
        dend3.connect(dend1(1.), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend1, dend2, dend3])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)
        np.testing.assert_almost_equal(-iaxial[1]+iaxial[4]+iaxial[6], -cell.imem[1], decimal=9)
        np.testing.assert_allclose(-iaxial[1]+iaxial[4]+iaxial[6], -cell.imem[1], rtol=1E-4)

    def test_3_dends_soma(self):
        '''
        Check Kirchhoff in soma when three dends connected to soma.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend1.connect(soma(1.0), 0)
        dend2.connect(soma(.5), 0)
        dend3.connect(soma(.8), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections()
        d_list, iaxial = cell.get_axial_currents_from_vmem()
        np.testing.assert_almost_equal(-iaxial[2]-iaxial[4]-iaxial[6], cell.imem[0], decimal=9)

    def test_len_iaxial(self):
        '''
        Check that len(iaxial) = cell.totnsegs - 1
        '''
        soma = neuron.h.Section(name='soma[0]')
        dend1 = neuron.h.Section(name='dend1[0]')
        dend2 = neuron.h.Section(name='dend2[0]')
        dend3 = neuron.h.Section(name='dend3[0]')
        dend1.connect(soma(1.0), 0)
        dend2.connect(soma(.5), 0)
        dend3.connect(soma(0.8), 0)
        cell, synapse, d_list, iaxial = self.cell_w_synapse_from_sections(sections=[soma, dend1, dend2, dend3])
        self.assertEqual(iaxial.shape[0], cell.totnsegs*2)

    def test_decompose_dipole(self):
        '''Test radial and tangential parts of dipole sums to dipole'''
        P1 = np.array([[1., 1., 1.]])
        p_rad, p_tan = self.decompose_dipole(P1)
        np.testing.assert_equal(p_rad + p_tan, P1)

    def test_rad_dipole(self):
        '''Test that radial part of decomposed dipole is correct'''
        P1 = np.array([[1., 1., 1.]])
        p_rad, p_tan = self.decompose_dipole(P1)
        np.testing.assert_equal(p_rad, np.array([[0., 0., 1.]]))

    def test_tan_dipole(self):
        '''Test that tangential part of decomposed dipole is correct'''
        P1 = np.array([[1., 1., 1.]])
        p_rad, p_tan = self.decompose_dipole(P1)
        np.testing.assert_equal(p_tan, np.array([[1., 1., 0.]]))

    def test_calc_theta(self):
        '''Test theta: angle between rz and r'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 0., 90.], [0., 0., -90.],[0., 70., 0.], [0., -70., 0.], [0., 10., 10.]])
        fs = self.make_class_object(rz1, r_el)
        theta = fs.calc_theta()
        np.testing.assert_almost_equal(theta, np.array([0., np.pi, np.pi/2, np.pi/2, np.pi/4]))

    def test_calc_phi(self):
        '''Test phi: azimuthal angle between rx and rxy'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 1., 0], [-1., -1., 1.],[1., 1., 4.]])
        fs = self.make_class_object(rz1, r_el)
        P1 = np.array([[0., 1., 0.], [1., 0., 1.]])
        phi = fs.calc_phi(P1)
        np.testing.assert_almost_equal(phi, np.array([[np.pi/2, np.pi], [5*np.pi/4, 7*np.pi/4], [np.pi/4, 3*np.pi/4]]))

    def test_rad_sign(self):
        '''Test if radial dipole points inwards or outwards'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 0., 90.]])
        fs = self.make_class_object(rz1, r_el)
        P1 = np.array([[0., 0., 1.], [0., 0., -2.]])
        s_vector = fs._sign_rad_dipole(P1)
        np.testing.assert_almost_equal(s_vector, np.array([1., -1.]))

    def test_MEG_00(self):
        '''test LFPy.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((11, 3))
        current_dipole_moment[:, 0] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0],
                       current_dipole_moment.shape[0], 3))
        gt[1, :, 2] = 1./4/np.pi
        gt[2, :, 1] = -1./4/np.pi
        gt[4, :, 2] = -1./4/np.pi
        gt[5, :, 1] = 1./4/np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_01(self):
        '''test LFPy.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((11, 3))
        current_dipole_moment[:, 1] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0],
                       current_dipole_moment.shape[0], 3))
        gt[0, :, 2] = -1./4/np.pi
        gt[2, :, 0] = 1./4/np.pi
        gt[3, :, 2] = 1./4/np.pi
        gt[5, :, 0] = -1./4/np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_02(self):
        '''test LFPy.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((11, 3))
        current_dipole_moment[:, 2] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        # ground truth
        gt = np.zeros((sensor_locations.shape[0],
                       current_dipole_moment.shape[0], 3))
        gt[0, :, 1] = 1./4/np.pi
        gt[1, :, 0] = -1./4/np.pi
        gt[3, :, 1] = -1./4/np.pi
        gt[4, :, 0] = 1./4/np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_03(self):
        '''test LFPy.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((1, 3))
        current_dipole_moment[:, 0] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0],
                       current_dipole_moment.shape[0], 3))
        gt[1, :, 2] = 1./4/np.pi
        gt[2, :, 1] = -1./4/np.pi
        gt[4, :, 2] = -1./4/np.pi
        gt[5, :, 1] = 1./4/np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_04(self):
        '''test LFPy.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((1, 3))
        current_dipole_moment[:, 1] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0],
                       current_dipole_moment.shape[0], 3))
        gt[0, :, 2] = -1./4/np.pi
        gt[2, :, 0] = 1./4/np.pi
        gt[3, :, 2] = 1./4/np.pi
        gt[5, :, 0] = -1./4/np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_05(self):
        '''test LFPy.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((1, 3))
        current_dipole_moment[:, 2] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0],
                       current_dipole_moment.shape[0], 3))
        gt[0, :, 1] = 1./4/np.pi
        gt[1, :, 0] = -1./4/np.pi
        gt[3, :, 1] = -1./4/np.pi
        gt[4, :, 0] = 1./4/np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))


    def test_cell_simulate_current_dipole_moment_00(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : 0,
            'tstopms' : 100,
            'dt' : 0.1,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
        }

        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : 0.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : 0,
            'bias' : 0.,
            'record_current' : False
        }
        for idx in range(31): #31 segments
            if idx != 15: # no net dipole moment because of stick symmetry
                stick = LFPy.Cell(**stickParams)
                synapse = LFPy.StimIntElectrode(stick, idx=idx,
                                       **stimParams)
                stick.simulate(rec_imem=True, rec_current_dipole_moment=True)
                p = np.dot(stick.imem.T, np.c_[stick.xmid, stick.ymid, stick.zmid])
                np.testing.assert_allclose(p, stick.current_dipole_moment)


    def test_cell_simulate_current_dipole_moment_01(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
        }

        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : 0.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : 0,
            'bias' : 0.,
            'record_current' : False
        }

        for idx in range(31): #31 segments
            if idx != 15: # no net dipole moment because of stick symmetry
                stick = LFPy.Cell(**stickParams)
                synapse = LFPy.StimIntElectrode(stick, idx=idx,
                                       **stimParams)
                stick.simulate(rec_imem=True, rec_current_dipole_moment=True)
                p = np.dot(stick.imem.T, np.c_[stick.xmid, stick.ymid, stick.zmid])
                np.testing.assert_allclose(p, stick.current_dipole_moment)

    def test_cell_simulate_current_dipole_moment_02(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
        }

        stimParams = {
            'e' : 0,                                # reversal potential
            'syntype' : 'Exp2Syn',                   # synapse type
            'tau1' : 0.1,                              # syn. time constant
            'tau2' : 2.,                              # syn. time constant
            'weight' : 0.01,
        }

        for idx in range(31): #31 segments
            if idx != 15: # no net dipole moment because of stick symmetry
                stick = LFPy.Cell(**stickParams)
                synapse = LFPy.Synapse(stick, idx=idx,
                                       **stimParams)
                synapse.set_spike_times(np.array([10., 20., 30., 40., 50.]))
                stick.simulate(rec_imem=True, rec_current_dipole_moment=True)
                p = np.dot(stick.imem.T, np.c_[stick.xmid, stick.ymid, stick.zmid])
                np.testing.assert_allclose(p, stick.current_dipole_moment)

    def test_cell_tstart_00(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
        }

        stimParams = {
            'pptype' : 'SinSyn',
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'bias' : 0.,
            'record_current' : False
        }

        stick0 = LFPy.Cell(tstartms=0, tstopms=200, **stickParams)
        synapse0 = LFPy.StimIntElectrode(stick0,
                                         stick0.get_closest_idx(0, 0, 1000),
                                         delay=0, phase=0.,
                                         **stimParams)
        stick0.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)


        stick1 = LFPy.Cell(tstartms=-100, tstopms=100, **stickParams)
        synapse1 = LFPy.StimIntElectrode(stick1,
                                         stick1.get_closest_idx(0, 0, 1000),
                                         delay=-100, phase=0.,
                                         **stimParams)
        stick1.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)

        inds = stick0.tvec >= 100
        np.testing.assert_allclose(stick0.vmem[:, inds], stick1.vmem)
        np.testing.assert_allclose(stick0.imem[:, inds], stick1.imem)
        np.testing.assert_allclose(stick0.current_dipole_moment[inds, :],
                                   stick1.current_dipole_moment)


    def test_cell_with_recextelectrode_00(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : 0,
            'tstopms' : 100,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,

        }

        electrodeParams = {
            'sigma' : 0.3,
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
            'method' : 'pointsource'
        }

        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : 0.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : 0,
            'bias' : 0.,
            'record_current' : False
        }

        stick = LFPy.Cell(**stickParams)
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        stick.simulate(electrode, rec_imem=True)

        electrode1 = LFPy.RecExtElectrode(cell=stick, **electrodeParams)
        electrode1.calc_lfp()

        np.testing.assert_allclose(electrode.LFP, electrode1.LFP)


    def test_cell_with_recextelectrode_01(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'stick.hoc'),
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'dt' : 2**-4,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,

        }

        electrodeParams = {
            'sigma' : 0.3,
            'x' : np.ones(11) * 100.,
            'y' : np.zeros(11),
            'z' : np.linspace(1000, 0, 11),
            'method' : 'pointsource'
        }

        stimParams = {
            'pptype' : 'SinSyn',
            'delay' : 0.,
            'dur' : 1000.,
            'pkamp' : 1.,
            'freq' : 100.,
            'phase' : 0,
            'bias' : 0.,
            'record_current' : False
        }

        stick = LFPy.Cell(**stickParams)
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        stick.simulate(electrode, rec_imem=True)

        electrode1 = LFPy.RecExtElectrode(cell=stick, **electrodeParams)
        electrode1.calc_lfp()

        np.testing.assert_allclose(electrode.LFP, electrode1.LFP)


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
        i_membrane_control = []     # record currents using Vector.record method
        i_membrane_fadvance = []    # record seg._i_membrane_ at each timestep
        for sec in [soma, dend]:
            for seg in sec:
                i = neuron.h.Vector()
                i.record(seg._ref_i_membrane_)
                i_membrane_control.append(i)
                i_membrane_fadvance.append([])

        # Simulation control
        neuron.h.dt =  2**-4          # simulation time resolution
        tstop = 500.        # simulation duration
        v_init = -65        # membrane voltage(s) at t = 0

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
            collect_i_membrane() # otherwise shape mismatch

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
        i_membrane_control = []     # record currents using Vector.record method
        i_membrane_fadvance = []    # record seg._i_membrane_ at each timestep
        for sec in [soma, dend]:
            for seg in sec:
                i = neuron.h.Vector()
                i.record(seg._ref_i_membrane_)
                i_membrane_control.append(i)
                i_membrane_fadvance.append([])

        # Simulation control
        neuron.h.dt = 2**-4          # simulation time resolution
        tstop = 500.        # simulation duration
        v_init = -65        # membrane voltage(s) at t = 0

        def initialize():
            neuron.h.finitialize(v_init)
            neuron.h.fcurrent()
            neuron.h.frecord_init()
            neuron.h.t = -100. # force simulations to start at some negative t

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
            collect_i_membrane() # otherwise shape mismatch

        initialize()
        integrate()

        i_membrane_control = np.array(i_membrane_control)
        i_membrane_fadvance = np.array(i_membrane_fadvance)

        np.testing.assert_equal(i_membrane_control, i_membrane_fadvance)

    def test_get_dipole_potential(self):
        sigma = 0.3
        r = np.array([[0., 0., 1.], [0., 1., 0.]])
        p = np.array([[0., 0., 4*np.pi*0.3], [0., 4*np.pi*0.3, 0.]])
        inf_model = LFPy.InfiniteVolumeConductor(sigma)
        phi = inf_model.get_dipole_potential(p, r)
        np.testing.assert_allclose(phi, np.array([[1., 0.], [0., 1.]]))

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
            'dt' : 2**-6,
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
            'dt' : 2**-6,
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
            'dt' : 2**-6,
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


    def cell_w_synapse_from_sections(self, sections=None):
        '''
        Make cell and synapse objects, set spike, simulate and return cell
        '''
        cellParams = {
            'morphology': None,
            'rm' : 30000,
            'cm' : 1.0,
            'Ra' : 150,
            'dt' : 2**-6,
            'tstartms' : -50,
            'tstopms' : 50,
            'delete_sections' : False
        }

        synapse_parameters = {'e': 0.,
                          'syntype': 'ExpSyn',
                          'tau': 5.,
                          'weight': .001,
                          'record_current': True,
                          'idx': 1}

        cell = LFPy.Cell(**cellParams)
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([1.]))
        cell.simulate(rec_imem = True, rec_isyn = True, rec_vmem = True)
        d_list, iaxial = cell.get_axial_currents_from_vmem()
        return cell, synapse, d_list, iaxial

    def make_class_object(self, rz1, r_el):
        '''Return class object fs'''
        radii = [79., 80., 85., 90.]
        sigmas = [0.3, 0.015, 15, 0.3]
        fs = LFPy.FourSphereVolumeConductor(radii, sigmas, r_el, rz1)
        return fs

    def decompose_dipole(self, P1):
        '''Return decomposed current dipole'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 0., 90.]])
        fs = self.make_class_object(rz1, r_el)
        p_rad, p_tan = fs._decompose_dipole(P1)
        return p_rad, p_tan

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
