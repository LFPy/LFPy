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

import os
import posixpath
import sys
import unittest
import numpy as np
import LFPy
import neuron
import pickle
import random
from copy import copy

# for nosetests to run load mechanisms
if "win32" in sys.platform:
    pth = os.path.join(LFPy.__path__[0], 'test', 'nrnmech.dll')
    pth = pth.replace(os.sep, posixpath.sep)
    if pth not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth)
        neuron.nrn_dll_loaded.append(pth)
else:
    neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))


class testCell(unittest.TestCase):
    """
    test class LFPy.Cell
    """

    def test_cell_tvec_00(self):
        stickParams = {
            'dt': 2**-3,
            'tstart': 0.,
            'tstop': 100.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_01(self):
        stickParams = {
            'dt': 2**-3,
            'tstart': 0.,
            'tstop': 10000.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_02(self):
        stickParams = {
            'dt': 0.1,
            'tstart': 0,
            'tstop': 100,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_03(self):
        stickParams = {
            'dt': 0.1,
            'tstart': 0,
            'tstop': 10000,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_04(self):
        stickParams = {
            'dt': 0.2,
            'tstart': 0,
            'tstop': 100,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_05(self):
        stickParams = {
            'dt': 0.2,
            'tstart': 0,
            'tstop': 10000,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_06(self):
        stickParams = {
            'dt': 2**-3,
            'tstart': -100,
            'tstop': 100,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_07(self):
        stickParams = {
            'dt': 2**-3,
            'tstart': -100,
            'tstop': 10000,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(
            0, stickParams['tstop'], int(
                stickParams['tstop'] / stickParams['dt']) + 1)

        np.testing.assert_allclose(tvec, tvec_numpy, atol=1E-9)

    def test_cell_tvec_08(self):
        stickParams = {
            'dt': 0.1,
            'tstart': -100,
            'tstop': 10000,
        }

        try:
            stickSimulationTesttvec(**stickParams)
        except AssertionError:
            pass

    def test_cell_tvec_09(self):
        stickParams = {
            'dt': 0.2,
            'tstart': -100,
            'tstop': 10000,
        }

        try:
            stickSimulationTesttvec(**stickParams)
        except AssertionError:
            pass

    def test_cell_set_pos_00(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        np.testing.assert_allclose(cell.somapos, [0, 0, 0])

    def test_cell_set_pos_01(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_02(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         pt3d=True)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_03(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_04(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         pt3d=True)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_05(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        np.testing.assert_allclose(cell.somapos,
                                   [cell.x[0].mean(axis=-1),
                                    cell.y[0].mean(axis=-1),
                                    cell.z[0].mean(axis=-1)])

    def test_cell_set_pos_06(self):
        '''test LFPy.Cell.set_pos'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         pt3d=True)
        np.testing.assert_allclose(cell.somapos,
                                   [cell.x[0].mean(axis=-1),
                                    cell.y[0].mean(axis=-1),
                                    cell.z[0].mean(axis=-1)])

    def test_cell_set_rotation_00(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))

        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation 180 deg around x-axis
        cell.set_rotation(x=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.y[:, 0], -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], -yends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], -zends, atol=1e-07)

    def test_cell_set_rotation_01(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation 180 deg around y-axis
        cell.set_rotation(y=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], -xends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], -zends, atol=1e-07)

    def test_cell_set_rotation_02(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        # test rotation 180 deg around z-axis
        cell.set_rotation(z=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], -xends, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, 0], -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], -yends, atol=1e-07)

    def test_cell_set_rotation_03(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         pt3d=True)

        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation 180 deg around x-axis
        cell.set_rotation(x=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.y[:, 0], -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], -yends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], -zends, atol=1e-07)

    def test_cell_set_rotation_04(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         pt3d=True)

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation 180 deg around y-axis
        cell.set_rotation(y=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], -xends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], -zends, atol=1e-07)

    def test_cell_set_rotation_05(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         pt3d=True)

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        # test rotation 180 deg around z-axis
        cell.set_rotation(z=np.pi)
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], -xends, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, 0], -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], -yends, atol=1e-07)

    def test_cell_set_rotation_06(self):
        '''test LFPy.Cell.set_rotation()'''
        cell = LFPy.Cell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'))

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation: 90 deg around x-axis, 90 deg around y-axis, 90 deg
        # around z-axis
        cell.set_rotation(x=np.pi / 2., y=np.pi, z=np.pi / 4.)
        # revert rotation: -90 deg around x-axis, -90 deg around y-axis, -90
        # deg around z-axis, rotation_order='zyx'
        cell.set_rotation(
            x=-np.pi / 2.,
            y=-np.pi,
            z=-np.pi / 4.,
            rotation_order='zyx')
        # assert that x-, y- and z-coordinates are same as beginning,
        # using absolute tolerances
        np.testing.assert_allclose(cell.x[:, 0], xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], xends, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, 0], ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], yends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], zends, atol=1e-07)

    def test_cell_chiral_morphology_00(self):
        '''test LFPy.Cell.chiral_morphology()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation 180 deg around x-axis
        cell.chiral_morphology(axis='x')
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], -xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), -xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], -xends, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, 0], ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], yends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], zends, atol=1e-07)

    def test_cell_chiral_morphology_01(self):
        '''test LFPy.Cell.chiral_morphology()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1]
        # test rotation 180 deg around y-axis
        cell.chiral_morphology(axis='y')
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], xends, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, 0], -ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), -ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], -yends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], zends, atol=1e-07)

    def test_cell_chiral_morphology_02(self):
        '''test LFPy.Cell.chiral_morphology()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))

        xstarts = cell.x[:, 0].copy()
        xmids = cell.x.mean(axis=-1).copy()
        xends = cell.x[:, -1].copy()
        ystarts = cell.y[:, 0].copy()
        ymids = cell.y.mean(axis=-1).copy()
        yends = cell.y[:, -1].copy()
        zstarts = cell.z[:, 0].copy()
        zmids = cell.z.mean(axis=-1).copy()
        zends = cell.z[:, -1].copy()
        # test rotation 180 deg around z-axis
        cell.chiral_morphology(axis='z')
        # assert that y- and z-coordinates are inverted, using absolute
        # tolerances
        np.testing.assert_allclose(cell.x[:, 0], xstarts, atol=1e-07)
        np.testing.assert_allclose(cell.x.mean(axis=-1), xmids, atol=1e-07)
        np.testing.assert_allclose(cell.x[:, -1], xends, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, 0], ystarts, atol=1e-07)
        np.testing.assert_allclose(cell.y.mean(axis=-1), ymids, atol=1e-07)
        np.testing.assert_allclose(cell.y[:, -1], yends, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, 0], -zstarts, atol=1e-07)
        np.testing.assert_allclose(cell.z.mean(axis=-1), -zmids, atol=1e-07)
        np.testing.assert_allclose(cell.z[:, -1], -zends, atol=1e-07)

    def test_cell_get_rand_prob_area_norm_00(self):
        '''test LFPy.Cell.get_rand_prob_area_norm()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        p = cell.get_rand_prob_area_norm()
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertTrue(p.min() >= 0.)
        self.assertTrue(p.max() <= 1.)

    def test_cell_get_rand_prob_area_norm_from_idx(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        p = cell.get_rand_prob_area_norm_from_idx(
            idx=cell.get_idx(section='allsec'))
        self.assertListEqual(
            cell.get_rand_prob_area_norm().tolist(),
            p.tolist())

    def test_cell_get_rand_prob_area_norm_from_idx_00(self):
        '''test LFPy.Cell.get_rand_prob_area_norm()'''
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        p = cell.get_rand_prob_area_norm_from_idx(idx=np.array([0]))
        np.testing.assert_equal(p, np.array([1.]))

    def test_cell_get_intersegment_vector_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        idx0 = 0
        idx1 = 1
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)

        self.assertListEqual(vector,
                             [cell.x[idx1].mean(axis=-1)
                              - cell.x[idx0].mean(axis=-1),
                              cell.y[idx1].mean(axis=-1)
                              - cell.y[idx0].mean(axis=-1),
                              cell.z[idx1].mean(axis=-1)
                              - cell.z[idx0].mean(axis=-1)])

    def test_cell_get_intersegment_distance_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        idx0 = 0
        idx1 = 1
        distance = cell.get_intersegment_distance(idx0=idx0, idx1=idx1)
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)

        self.assertEqual(np.sqrt(np.array(vector)**2).sum(), distance)

    def test_cell_get_idx_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
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
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'),
                         nsegs_method=None)
        self.assertEqual(cell.get_closest_idx(x=0, y=0, z=0),
                         cell.get_idx(section='soma')[0])

        self.assertEqual(cell.get_closest_idx(x=-25, y=0, z=175),
                         cell.get_idx(section='dend[1]')[0])

    def test_cell_get_closest_idx_01(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        x = -41.7
        z = 156.7
        sec_name = "dend"

        idx1 = cell.get_closest_idx(x=x, z=z)
        idx2 = cell.get_closest_idx(x=x, z=z, section=sec_name)
        self.assertEqual(idx1, idx2)

    def test_cell_get_idx_children_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))

        np.testing.assert_array_equal(cell.get_idx_children(parent='soma[0]'),
                                      cell.get_idx(section='dend[0]'))

    def test_cell_get_idx_parent_children_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        np.testing.assert_array_equal(
            cell.get_idx_parent_children(
                parent='soma[0]'), cell.get_idx(
                section=[
                    'soma[0]', 'dend[0]']))

    def test_cell_get_idx_name_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        np.testing.assert_array_equal(cell.get_idx_name(idx=np.array([0])),
                                      np.array([[0, 'soma[0]', 0.5]],
                                               dtype=object))

    def test_cell_get_rand_idx_area_norm_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        idx = cell.get_rand_idx_area_norm(nidx=1000000)

        # compute histogram and correlate with segment area
        bins = np.arange(cell.totnsegs + 1)
        hist, bin_edges = np.histogram(idx, bins=bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 4 decimal places
        self.assertAlmostEqual(
            np.corrcoef(
                cell.area, hist)[
                0, 1], 1., places=4)

        # check if min and max is in the range of segment indices
        self.assertEqual(idx.min(), 0)
        self.assertEqual(idx.max(), cell.totnsegs - 1)

    def test_cell_set_synapse_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        cell.set_synapse(idx=0, syntype='ExpSyn', record_curret=False,
                         record_potential=False, weight=1.,
                         **dict(e=10., tau=2.))

        self.assertTrue('ExpSyn' in cell._hoc_synlist[0].hname())
        self.assertEqual(len(cell._hoc_synlist), 1)
        self.assertEqual(len(cell._hoc_netconlist), 1)
        self.assertEqual(len(cell._hoc_netstimlist), 1)
        self.assertEqual(cell._hoc_synlist[0].e, 10.)
        self.assertEqual(cell._hoc_synlist[0].tau, 2.)
        self.assertEqual(cell._hoc_netconlist[0].weight[0], 1.)

    def test_cell_set_point_process_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        cell.set_point_process(idx=0, pptype='IClamp', record_current=False,
                               **dict(delay=1., amp=1.))
        self.assertEqual(cell._hoc_stimlist[0].hname(), 'IClamp[0]')
        self.assertEqual(len(cell._hoc_stimlist), 1)
        self.assertEqual(cell._hoc_stimlist[0].delay, 1.)
        self.assertEqual(cell._hoc_stimlist[0].amp, 1.)

    def test_cell_strip_hoc_objects_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        cell.strip_hoc_objects()
        for attribute in dir(cell):
            self.assertNotEqual(str(type(getattr(cell, attribute))),
                                'hoc.HocObject')

    def test_cell_cellpickler_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks.hoc'))
        cell_pickle = cell.cellpickler(filename=None, pickler=pickle.dumps)
        pickled_cell = pickle.loads(cell_pickle)

        for attribute in dir(cell):
            if attribute.startswith('__') or attribute.startswith('_'):
                pass
            else:
                self.assertEqual(type(getattr(cell, attribute)),
                                 type(getattr(pickled_cell, attribute)))

    def test_cell_get_axial_currents_from_vmem_00(self):
        '''
        Check Kirchhoff in single dend.
        '''
        neuron.h('forall delete_section()')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend2.connect(dend1(1.), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[0], imem[0], rtol=1E-5)
        np.testing.assert_almost_equal(iaxial[1], imem[1], decimal=9)
        np.testing.assert_allclose(iaxial[1], imem[1], rtol=1E-5)

    def test_cell_get_axial_currents_from_vmem_01(self):
        '''
        Check Kirchhoff in soma when single dend connected to soma mid.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')
        dend.connect(soma(0.5), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()

        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0], imem[0], decimal=9)
        np.testing.assert_almost_equal(-iaxial[1], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[0], imem[0], rtol=1E-4)
        np.testing.assert_allclose(-iaxial[1], imem[0], rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_02(self):
        '''
        Check Kirchhoff in soma when single dend connected to soma end.
        '''
        neuron.h.topology()
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')
        dend.connect(soma(1.0), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[0], imem[0], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[1], imem[1], decimal=9)
        np.testing.assert_allclose(iaxial[1], imem[1], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[0], iaxial[1], decimal=9)
        np.testing.assert_allclose(iaxial[0], iaxial[1], rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_03(self):
        '''
        Check Kirchhoff in soma when single dend connected to random
        soma point.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend = neuron.h.Section(name='dend')
        dend.connect(soma(random.uniform(1e-2, 1.)), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[0], imem[0], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[1], imem[1], decimal=9)
        np.testing.assert_allclose(iaxial[1], imem[1], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[0], iaxial[1], decimal=9)
        np.testing.assert_allclose(iaxial[0], iaxial[1], rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_04(self):
        '''
        Check Kirchhoff in soma when two dends connected to soma mid.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(0.5), 0)
        dend2.connect(soma(0.5), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[1] -
                                       iaxial[3], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[1] - iaxial[3], imem[0], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[0], iaxial[1], decimal=9)
        np.testing.assert_allclose(iaxial[0], iaxial[1], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[2], iaxial[3], decimal=9)
        np.testing.assert_allclose(iaxial[2], iaxial[3], rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_05(self):
        '''
        Check Kirchhoff in soma when two dends connected to soma end.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(1.), 0)
        dend2.connect(soma(1.), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()

        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0] -
                                       iaxial[2], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[0] - iaxial[2], imem[0], rtol=1E-4)
        np.testing.assert_almost_equal(
            iaxial[0] + iaxial[2],
            iaxial[1] + iaxial[3],
            decimal=9)
        np.testing.assert_allclose(
            iaxial[0] + iaxial[2],
            iaxial[1] + iaxial[3],
            rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_06(self):
        '''
        Check Kirchhoff in soma when two dends connected to diff soma points.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(1.0), 0)
        dend2.connect(soma(.5), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0] -
                                       iaxial[2], imem[0], decimal=9)
        np.testing.assert_allclose(-iaxial[0] - iaxial[2], imem[0], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[0], iaxial[1], decimal=9)
        np.testing.assert_allclose(iaxial[0], iaxial[1], rtol=1E-4)
        np.testing.assert_almost_equal(iaxial[2], iaxial[3], decimal=9)
        np.testing.assert_allclose(iaxial[2], iaxial[3], rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_07(self):
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
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(
            iaxial[0] + iaxial[4],
            iaxial[1] + iaxial[5],
            decimal=9)
        np.testing.assert_allclose(
            iaxial[0] + iaxial[4],
            iaxial[1] + iaxial[5],
            rtol=1E-4)
        np.testing.assert_almost_equal(-iaxial[1] +
                                       iaxial[2] + iaxial[4], -imem[1],
                                       decimal=9)
        np.testing.assert_allclose(-iaxial[1] +
                                   iaxial[2] + iaxial[4], -imem[1], rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_08(self):
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
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0] -
                                       iaxial[2] - iaxial[4], imem[0],
                                       decimal=9)
        np.testing.assert_allclose(-iaxial[0] -
                                   iaxial[2] - iaxial[4], imem[0], rtol=1E-3)

    def test_cell_get_axial_currents_from_vmem_09(self):
        '''
        Check Kirchhoff in 2-comp model where dend 0 is connected to soma 0.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend1.connect(soma(0.0), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()

        imem = cell.imem

        np.testing.assert_almost_equal(iaxial[0], -imem[0], decimal=9)
        np.testing.assert_almost_equal(iaxial[0], imem[1], decimal=9)
        np.testing.assert_allclose(iaxial[0], -imem[0], rtol=1E-3)
        np.testing.assert_allclose(iaxial[0], imem[1], rtol=1E-3)

    def test_cell_get_axial_currents_from_vmem_10(self):
        '''
        Check that len(iaxial) = (cell.totnsegs - 1)*2
        '''
        soma = neuron.h.Section(name='soma[0]')
        dend1 = neuron.h.Section(name='dend1[0]')
        dend2 = neuron.h.Section(name='dend2[0]')
        dend3 = neuron.h.Section(name='dend3[0]')
        dend1.connect(soma(1.0), 0)
        dend2.connect(soma(.5), 0)
        dend3.connect(soma(0.8), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        totnsegs = cell.totnsegs

        self.assertEqual(iaxial.shape[0], (totnsegs - 1) * 2)

    def test_cell_get_axial_currents_from_vmem_11(self):
        '''
        Check Kirchhoff in soma when three dends connected to soma mid.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend1.connect(soma(0.5), 0)
        dend2.connect(soma(0.5), 0)
        dend3.connect(soma(0.5), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-iaxial[0] -
                                       iaxial[2] - iaxial[4], imem[0],
                                       decimal=9)
        np.testing.assert_allclose(-iaxial[0] -
                                   iaxial[2] - iaxial[4], imem[0], rtol=1E-3)

    def test_cell_get_axial_currents_from_vmem_12(self):
        '''
        Check Kirchhoff in morph where secs are connected to arc length 0.5.
        '''
        morphology = os.path.join(
            LFPy.__path__[0],
            'test',
            'sticks_not_connected_head_to_toe.hoc')
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()

        # some cleanup of Python-created section references
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(
            iaxial[6] + iaxial[10] + imem[3], iaxial[5], decimal=9)
        np.testing.assert_allclose(
            iaxial[6] +
            iaxial[10] +
            imem[3],
            iaxial[5],
            rtol=1E-5)

    def test_cell_get_axial_currents_from_vmem_13(self):
        '''
        Check Kirchhoff in morph where secs are connected to arc length 0.7.
        '''
        morphology = os.path.join(
            LFPy.__path__[0],
            'test',
            'sticks_not_connected_head_to_toe.hoc')
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(
            iaxial[8] + iaxial[20] + imem[4], iaxial[7], decimal=9)
        np.testing.assert_allclose(
            iaxial[8] +
            iaxial[20] +
            imem[4],
            iaxial[7],
            rtol=1E-5)

    def test_cell_get_axial_currents_from_vmem_14(self):
        '''
        Check iaxial current mid positions in three-section stick.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(1), 0)
        dend2.connect(soma(1), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree(sec=soma)
        cell = cell_w_synapse_from_sections(morphology)
        # iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        new_x = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        new_y = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        new_z = [[-10, 0, 10], [10, 15, 20], [20, 30, 50]]
        new_d = [[20, 20, 20], [10, 10, 10], [5, 5, 5]]
        for j, sec in enumerate(neuron.h.allsec()):
            for n in range(3):
                neuron.h.pt3dchange(n,
                                    new_x[j][n],
                                    new_y[j][n],
                                    new_z[j][n],
                                    new_d[j][n],
                                    sec=sec)
                neuron.h.define_shape()
        cell._collect_geometry()
        cell2 = cell_w_synapse_from_sections(morphology)
        iaxial2, d_list2, pos_list2 = cell2.get_axial_currents_from_vmem()

        mid_current_positions = np.array(
            [[0., 0., 5], [0., 0., 20], [0., 0., 5.], [0., 0., 12.5]])
        np.testing.assert_almost_equal(
            mid_current_positions, pos_list2, decimal=9)
        np.testing.assert_allclose(mid_current_positions, pos_list2, rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_15(self):
        '''
        Check iaxial current mid positions in ball-n-y.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend1.connect(soma(1.0), 0)
        dend2.connect(dend1(1.), 0)
        dend3.connect(dend1(.5), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree(sec=soma)
        cell = cell_w_synapse_from_sections(morphology)
        iaxial1, d_list1, pos_list1 = cell.get_axial_currents_from_vmem()
        new_x = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 5, 10]]
        new_y = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        new_z = [[-10, 0, 10], [10, 15, 20], [20, 30, 40], [15, 15, 15]]
        new_d = [[20, 20, 20], [10, 10, 10], [5, 5, 5], [2, 2, 2]]
        for j, sec in enumerate(neuron.h.allsec()):
            for n in range(3):
                neuron.h.pt3dchange(n,
                                    new_x[j][n],
                                    new_y[j][n],
                                    new_z[j][n],
                                    new_d[j][n],
                                    sec=sec)
                neuron.h.define_shape()
        cell._collect_geometry()
        cell2 = cell_w_synapse_from_sections(morphology)
        iaxial2, d_list2, pos_list2 = cell2.get_axial_currents_from_vmem()
        mid_current_positions = np.array([[0., 0., 5.], [0., 0., 12.5],
                                          [0., 0., 15.], [2.5, 0., 15.],
                                          [0., 0., 17.5], [0, 0., 25.]])

        np.testing.assert_almost_equal(
            mid_current_positions, pos_list2, decimal=9)
        np.testing.assert_allclose(mid_current_positions, pos_list2, rtol=1E-4)

    def test_cell_get_axial_currents_from_vmem_16(self):
        '''
        Check Kirchhoff in soma when three dends connected to soma end.
        '''
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend1.connect(soma(1.0), 0)
        dend2.connect(soma(1.0), 0)
        dend3.connect(soma(1.0), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology)
        iaxial, d_list, pos_list = cell.get_axial_currents_from_vmem()
        imem = copy(cell.imem)

        np.testing.assert_almost_equal(-imem[0],
                                       iaxial[1] + iaxial[3] + iaxial[5],
                                       decimal=9)
        np.testing.assert_allclose(-imem[0],
                                   iaxial[1] + iaxial[3] + iaxial[5],
                                   rtol=1E-3)

    def test_cell_simulate_recorder_00(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stick = LFPy.Cell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_01(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.1,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.Cell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_02(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.2,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.Cell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_03(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stick = LFPy.Cell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_04(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 0.1,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.Cell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_05(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 0.2,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.Cell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_cvode_00(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 0.2,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': True,
            'record_potential': True
        }

        stick = LFPy.Cell(**stickParams)
        synapse = LFPy.StimIntElectrode(stick, idx=0,
                                        **stimParams)
        stick.simulate(rec_vmem=True, rec_imem=True, variable_dt=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1] ==
                        synapse.i.size == synapse.v.size)

    def test_cell_simulate_current_dipole_moment_00(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.1,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': False
        }
        for idx in range(31):  # 31 segments
            if idx != 15:  # no net dipole moment because of stick symmetry
                stick = LFPy.Cell(**stickParams)
                LFPy.StimIntElectrode(stick, idx=idx, **stimParams)
                probes = [LFPy.CurrentDipoleMoment(stick)]
                stick.simulate(probes=probes, rec_imem=True)
                p = np.c_[stick.x.mean(axis=-1),
                          stick.y.mean(axis=-1),
                          stick.z.mean(axis=-1)].T @ stick.imem
                np.testing.assert_allclose(p, probes[0].data)

    def test_cell_simulate_current_dipole_moment_01(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': -100,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': False
        }

        for idx in range(31):  # 31 segments
            if idx != 15:  # no net dipole moment because of stick symmetry
                stick = LFPy.Cell(**stickParams)
                LFPy.StimIntElectrode(stick, idx=idx, **stimParams)
                probes = [LFPy.CurrentDipoleMoment(stick)]
                stick.simulate(probes=probes, rec_imem=True)
                p = np.c_[stick.x.mean(axis=-1),
                          stick.y.mean(axis=-1),
                          stick.z.mean(axis=-1)].T @ stick.imem
                np.testing.assert_allclose(p, probes[0].data)

    def test_cell_simulate_current_dipole_moment_02(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': -100,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stimParams = {
            'e': 0,                                # reversal potential
            'syntype': 'Exp2Syn',                   # synapse type
            'tau1': 0.1,                              # syn. time constant
            'tau2': 2.,                              # syn. time constant
            'weight': 0.01,
        }

        for idx in range(31):  # 31 segments
            if idx != 15:  # no net dipole moment because of stick symmetry
                stick = LFPy.Cell(**stickParams)
                synapse = LFPy.Synapse(stick, idx=idx,
                                       **stimParams)
                synapse.set_spike_times(np.array([10., 20., 30., 40., 50.]))
                probes = [LFPy.CurrentDipoleMoment(stick)]
                stick.simulate(probes=probes, rec_imem=True)
                p = np.c_[stick.x.mean(axis=-1),
                          stick.y.mean(axis=-1),
                          stick.z.mean(axis=-1)].T @ stick.imem
                np.testing.assert_allclose(p, probes[0].data)

    def test_cell_tstart_00(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stimParams = {
            'pptype': 'SinSyn',
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'bias': 0.,
            'record_current': False
        }

        stick0 = LFPy.Cell(tstart=0, tstop=200, **stickParams)
        LFPy.StimIntElectrode(stick0,
                              stick0.get_closest_idx(0, 0, 1000),
                              delay=0, phase=0.,
                              **stimParams)
        stick0.simulate(rec_imem=True, rec_vmem=True)

        stick1 = LFPy.Cell(tstart=-100, tstop=100, **stickParams)
        LFPy.StimIntElectrode(stick1,
                              stick1.get_closest_idx(0, 0, 1000),
                              delay=-100, phase=0.,
                              **stimParams)
        stick1.simulate(rec_imem=True, rec_vmem=True)

        inds = stick0.tvec >= 100
        np.testing.assert_allclose(stick0.vmem[:, inds], stick1.vmem)
        np.testing.assert_allclose(stick0.imem[:, inds], stick1.imem)

    def test_cell_with_recextelectrode_00(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,

        }

        electrodeParams = {
            'sigma': 0.3,
            'x': np.ones(11) * 100.,
            'y': np.zeros(11),
            'z': np.linspace(1000, 0, 11),
            'method': 'pointsource'
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': False
        }

        stick = LFPy.Cell(**stickParams)
        LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                              **stimParams)
        electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
        stick.simulate(probes=[electrode], rec_imem=True)

        M = electrode.get_transformation_matrix()
        LFP = M @ stick.imem
        np.testing.assert_allclose(electrode.data, LFP)
        self.assertTrue(stick.tvec.size == stick.imem.shape[1] ==
                        electrode.data.shape[1] == LFP.shape[1] ==
                        int(stick.tstop / stick.dt) + 1)

    def test_cell_with_recextelectrode_01(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': -100,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,

        }

        electrodeParams = {
            'sigma': 0.3,
            'x': np.ones(11) * 100.,
            'y': np.zeros(11),
            'z': np.linspace(1000, 0, 11),
            'method': 'pointsource'
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': False
        }

        stick = LFPy.Cell(**stickParams)
        LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                              **stimParams)
        electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
        stick.simulate(probes=[electrode], rec_imem=True)

        M = electrode.get_transformation_matrix()
        LFP = M @ stick.imem
        np.testing.assert_allclose(electrode.data, LFP)
        self.assertTrue(stick.tvec.size == stick.imem.shape[1] ==
                        electrode.data.shape[1] == LFP.shape[1] ==
                        int(stick.tstop / stick.dt) + 1)

    def test_cell_with_recextelectrode_02(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.1,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,

        }

        electrodeParams = {
            'sigma': 0.3,
            'x': np.ones(11) * 100.,
            'y': np.zeros(11),
            'z': np.linspace(1000, 0, 11),
            'method': 'pointsource'
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': False
        }

        stick = LFPy.Cell(**stickParams)
        LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                              **stimParams)
        electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
        stick.simulate(probes=[electrode], rec_imem=True)

        M = electrode.get_transformation_matrix()
        LFP = M @ stick.imem
        np.testing.assert_allclose(electrode.data, LFP)
        self.assertTrue(stick.tvec.size == stick.imem.shape[1] ==
                        electrode.data.shape[1] == LFP.shape[1] ==
                        int(stick.tstop / stick.dt) + 1)

    def test_cell_with_recextelectrode_03(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.2,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,

        }

        electrodeParams = {
            'sigma': 0.3,
            'x': np.ones(11) * 100.,
            'y': np.zeros(11),
            'z': np.linspace(1000, 0, 11),
            'method': 'pointsource'
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': False
        }

        stick = LFPy.Cell(**stickParams)
        LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                              **stimParams)
        electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
        stick.simulate(probes=[electrode], rec_imem=True)

        M = electrode.get_transformation_matrix()
        LFP = M @ stick.imem
        np.testing.assert_allclose(electrode.data, LFP)
        self.assertTrue(stick.tvec.size == stick.imem.shape[1] ==
                        electrode.data.shape[1] == LFP.shape[1] ==
                        int(stick.tstop / stick.dt) + 1)

    def test_cell_with_recextelectrode_and_cvode_00(self):
        stickParams = {
            'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
            'tstart': -100,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,

        }

        electrodeParams = {
            'sigma': 0.3,
            'x': np.ones(11) * 100.,
            'y': np.zeros(11),
            'z': np.linspace(1000, 0, 11),
            'method': 'pointsource'
        }

        stimParams = {
            'pptype': 'SinSyn',
            'delay': 0.,
            'dur': 1000.,
            'pkamp': 1.,
            'freq': 100.,
            'phase': 0,
            'bias': 0.,
            'record_current': True,
            'record_potential': True
        }

        stick = LFPy.Cell(**stickParams)
        synapse = LFPy.StimIntElectrode(
            stick, stick.get_closest_idx(
                0, 0, 1000), **stimParams)
        electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
        stick.simulate(probes=[electrode],
                       rec_imem=True, rec_vmem=True,
                       variable_dt=True)

        self.assertTrue(stick.tvec.size
                        == stick.imem.shape[1]
                        == synapse.i.size
                        == synapse.v.size
                        == electrode.data.shape[1])

    def test_get_multi_current_dipole_moments_00(self):
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(0.5), 0)
        dend2.connect(dend1(1.0), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        dipoles, dipole_locs = cell.get_multi_current_dipole_moments()
        t_point = -1
        P_from_multi_dipoles = np.sum(dipoles[:, :, t_point], axis=0)
        P = cell.current_dipole_moment[:, t_point]

        np.testing.assert_almost_equal(P, P_from_multi_dipoles)
        np.testing.assert_allclose(P, P_from_multi_dipoles, rtol=1E-5)

    def test_get_multi_current_dipole_moments_01(self):
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend1.connect(soma(1.), 0)
        dend2.connect(soma(1.), 0)
        dend3.connect(soma(1.), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        dipoles, dipole_locs = cell.get_multi_current_dipole_moments()
        t_point = -1
        P_from_multi_dipoles = np.sum(dipoles[:, :, t_point], axis=0)
        P = cell.current_dipole_moment[:, t_point]

        np.testing.assert_almost_equal(P, P_from_multi_dipoles)

    def test_get_multi_current_dipole_moments_02(self):
        neuron.h('forall delete_section()')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend2.connect(dend1(1.), 0)
        dend3.connect(dend2(.5), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        dipoles, dipole_locs = cell.get_multi_current_dipole_moments()
        t_point = -1
        P_from_multi_dipoles = np.sum(dipoles[:, :, t_point], axis=0)
        P = cell.current_dipole_moment[:, t_point]

        np.testing.assert_almost_equal(P, P_from_multi_dipoles)

    def test_get_multi_current_dipole_moments_03(self):
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend4 = neuron.h.Section(name='dend4')
        dend5 = neuron.h.Section(name='dend5')
        dend1.connect(soma(1.), 0)
        dend2.connect(soma(0.), 0)
        dend3.connect(soma(0.), 0)
        dend4.connect(soma(0.), 0)
        dend5.connect(soma(0.432), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        dipoles, dipole_locs = cell.get_multi_current_dipole_moments()
        t_point = -1
        P_from_multi_dipoles = np.sum(dipoles[:, :, t_point], axis=0)
        P = cell.current_dipole_moment[:, t_point]

        np.testing.assert_almost_equal(P, P_from_multi_dipoles)

    def test_get_multi_current_dipole_moments_04(self):
        morphology = os.path.join(LFPy.__path__[0], 'test',
                                  'ball_and_sticks.hoc')
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        dipoles, dipole_locs = cell.get_multi_current_dipole_moments()
        t_point = -1
        P_from_multi_dipoles = np.sum(dipoles[:, :, t_point], axis=0)
        P = cell.current_dipole_moment[:, t_point]

        np.testing.assert_almost_equal(P, P_from_multi_dipoles)

    def test_cell_distort_geometry_00(self):
        cell0 = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks.hoc'))
        factors = [-0.2, 0.1, 0., 0.1, 0.2]
        nus = [-0.5, 0., 0.5]
        for factor in factors:
            for nu in nus:
                for axis in 'xyz':
                    cell1 = LFPy.Cell(
                        morphology=os.path.join(
                            LFPy.__path__[0],
                            'test',
                            'ball_and_sticks.hoc'))
                    cell1.distort_geometry(factor=factor, nu=nu, axis=axis)
                    for ax in 'xyz'.replace(axis, ''):
                        np.testing.assert_allclose(
                            getattr(cell0, ax) * (1 + factor * nu),
                            getattr(cell1, ax))
                    np.testing.assert_allclose(
                        getattr(cell0, axis) * (1 - factor),
                        getattr(cell1, axis))


# ####### Functions used by tests: ##########################################
def stickSimulationTesttvec(**kwargs):
    stick = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                              'stick.hoc'), verbose=False,
                      **kwargs)
    stick.simulate(rec_imem=False)
    return stick.tvec


def cell_w_synapse_from_sections(morphology, rec_current_dipole_moment=False):
    '''
    Make cell and synapse objects, set spike, simulate and return cell
    '''
    cellParams = {
        'morphology': morphology,
        'cm': 1,
        'Ra': 150,
        'v_init': -65,
        'passive': True,
        'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
        'dt': 2**-6,
        'tstart': -50,
        'tstop': 50,
        'delete_sections': False
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

    if rec_current_dipole_moment:
        cdm = LFPy.CurrentDipoleMoment(cell)
        probes = [cdm]
    else:
        probes = []

    cell.simulate(probes=probes, rec_imem=True, rec_vmem=True)

    if rec_current_dipole_moment:
        cell.current_dipole_moment = cdm.data

    return cell
