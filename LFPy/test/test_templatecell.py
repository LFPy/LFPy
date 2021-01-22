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
import pickle

# for nosetests to run load the SinSyn sinusoid synapse currrent mechanism
if "win32" in sys.platform:
    pth = os.path.join(LFPy.__path__[0], 'test', 'nrnmech.dll')
    pth = pth.replace(os.sep, posixpath.sep)
    if pth not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth)
        neuron.nrn_dll_loaded.append(pth)
else:
    neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))


class testTemplateCell(unittest.TestCase):
    """
    test class LFPy.TemplateCell
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
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        np.testing.assert_allclose(cell.somapos, [0, 0, 0])

    def test_cell_set_pos_01(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_02(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            pt3d=True)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_03(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_04(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            pt3d=True)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_05(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        np.testing.assert_allclose(cell.somapos,
                                   [cell.x[0].mean(axis=-1),
                                    cell.y[0].mean(axis=-1),
                                    cell.z[0].mean(axis=-1)])

    def test_cell_set_pos_06(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            pt3d=True)
        np.testing.assert_allclose(cell.somapos,
                                   [cell.x[0].mean(axis=-1),
                                    cell.y[0].mean(axis=-1),
                                    cell.z[0].mean(axis=-1)])

    def test_cell_set_rotation_00(self):
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

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
        '''test LFPy.TemplateCell.chiral_morphology()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

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
        '''test LFPy.TemplateCell.chiral_morphology()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

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
        '''test LFPy.TemplateCell.chiral_morphology()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

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
        '''test LFPy.TemplateCell.get_rand_prob_area_norm()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        p = cell.get_rand_prob_area_norm()
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertTrue(p.min() >= 0.)
        self.assertTrue(p.max() <= 1.)

    def test_cell_get_rand_prob_area_norm_from_idx(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        p = cell.get_rand_prob_area_norm_from_idx(
            idx=cell.get_idx(section='allsec'))
        self.assertListEqual(
            cell.get_rand_prob_area_norm().tolist(),
            p.tolist())

    def test_cell_get_rand_prob_area_norm_from_idx_00(self):
        '''test LFPy.TemplateCell.get_rand_prob_area_norm()'''
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        p = cell.get_rand_prob_area_norm_from_idx(idx=np.array([0]))
        np.testing.assert_equal(p, np.array([1.]))

    def test_cell_get_intersegment_vector_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
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
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        idx0 = 0
        idx1 = 1
        distance = cell.get_intersegment_distance(idx0=idx0, idx1=idx1)
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)

        self.assertEqual(np.sqrt(np.array(vector)**2).sum(), distance)

    def test_cell_get_idx_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
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
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            nsegs_method=None)
        self.assertEqual(cell.get_closest_idx(x=0, y=0, z=0),
                         cell.get_idx(section='soma')[0])

        self.assertEqual(cell.get_closest_idx(x=-25, y=0, z=175),
                         cell.get_idx(section='dend[1]')[0])

    def test_cell_get_idx_children_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )

        np.testing.assert_array_equal(
            cell.get_idx_children(
                parent='ball_and_stick_template[0].soma[0]'), cell.get_idx(
                section='ball_and_stick_template[0].dend[0]'))

    def test_cell_get_idx_parent_children_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        np.testing.assert_array_equal(
            cell.get_idx_parent_children(
                parent='ball_and_stick_template[0].soma[0]'),
            cell.get_idx(
                section=[
                    'ball_and_stick_template[0].soma[0]',
                    'ball_and_stick_template[0].dend[0]']))

    def test_cell_get_idx_name_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        target = np.array([[0, 'ball_and_stick_template[0].soma[0]', 0.5]],
                          dtype=object)
        np.testing.assert_array_equal(cell.get_idx_name(idx=np.array([0])),
                                      target)

    def test_cell_get_rand_idx_area_norm_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        idx = cell.get_rand_idx_area_norm(nidx=1000000)

        # compute histogram and correlate with segment area
        bins = np.arange(cell.totnsegs + 1)
        hist, bin_edges = np.histogram(idx, bins=bins)

        # compute Pearson correlation coefficients between area and histogram
        # reporting success if within 5 decimal places
        self.assertAlmostEqual(
            np.corrcoef(
                cell.area, hist)[
                0, 1], 1., places=4)

        # check if min and max is in the range of segment indices
        self.assertEqual(idx.min(), 0)
        self.assertEqual(idx.max(), cell.totnsegs - 1)

    def test_cell_set_synapse_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
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
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        cell.set_point_process(idx=0, pptype='IClamp', record_current=False,
                               **dict(delay=1., amp=1.))
        self.assertEqual(cell._hoc_stimlist[0].hname(), 'IClamp[0]')
        self.assertEqual(len(cell._hoc_stimlist), 1)
        self.assertEqual(cell._hoc_stimlist[0].delay, 1.)
        self.assertEqual(cell._hoc_stimlist[0].amp, 1.)

    def test_cell_strip_hoc_objects_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        cell.strip_hoc_objects()
        for attribute in dir(cell):
            self.assertNotEqual(str(type(getattr(cell, attribute))),
                                'hoc.HocObject')

    def test_cell_cellpickler_00(self):
        cell = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        cell_pickle = cell.cellpickler(filename=None, pickler=pickle.dumps)
        pickled_cell = pickle.loads(cell_pickle)

        for attribute in dir(cell):
            if attribute.startswith('__') or attribute.startswith('_'):
                pass
            else:
                self.assertEqual(type(getattr(cell, attribute)),
                                 type(getattr(pickled_cell, attribute)))

    def test_cell_simulate_recorder_00(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stick = LFPy.TemplateCell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_01(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.1,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.TemplateCell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_02(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
            'tstart': 0,
            'tstop': 100,
            'dt': 0.2,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.TemplateCell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_03(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 2**-4,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }

        stick = LFPy.TemplateCell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_04(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 0.1,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.TemplateCell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_recorder_05(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
            'tstart': 0,
            'tstop': 10000,
            'dt': 0.2,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
        }
        stick = LFPy.TemplateCell(**stickParams)
        stick.simulate(rec_vmem=True, rec_imem=True)
        self.assertTrue(stick.tvec.size ==
                        stick.vmem.shape[1] ==
                        stick.imem.shape[1])
        self.assertTrue(np.all(stick.vmem == stick.v_init))
        self.assertTrue(np.all(stick.imem == 0.))

    def test_cell_simulate_current_dipole_moment_00(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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
                stick = LFPy.TemplateCell(**stickParams)
                LFPy.StimIntElectrode(stick, idx=idx, **stimParams)
                probes = [LFPy.CurrentDipoleMoment(stick)]
                stick.simulate(probes=probes, rec_imem=True)
                p = np.c_[stick.x.mean(axis=-1),
                          stick.y.mean(axis=-1),
                          stick.z.mean(axis=-1)].T @ stick.imem
                np.testing.assert_allclose(p, probes[0].data)

    def test_cell_simulate_current_dipole_moment_01(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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
                # neuron.h('forall delete_section()')
                stick = LFPy.TemplateCell(**stickParams)
                LFPy.StimIntElectrode(stick, idx=idx, **stimParams)
                probes = [LFPy.CurrentDipoleMoment(stick)]
                stick.simulate(probes=probes, rec_imem=True)
                p = np.c_[stick.x.mean(axis=-1),
                          stick.y.mean(axis=-1),
                          stick.z.mean(axis=-1)].T @ stick.imem
                np.testing.assert_allclose(p, probes[0].data)

    def test_cell_simulate_current_dipole_moment_02(self):
        stickParams = {
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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
                stick = LFPy.TemplateCell(**stickParams)
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
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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

        stick0 = LFPy.TemplateCell(tstart=0, tstop=200, **stickParams)
        LFPy.StimIntElectrode(stick0,
                              stick0.get_closest_idx(0, 0, 1000),
                              delay=0, phase=0.,
                              **stimParams)
        stick0.simulate(rec_imem=True, rec_vmem=True)

        stick1 = LFPy.TemplateCell(tstart=-100, tstop=100, **stickParams)
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
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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

        stick = LFPy.TemplateCell(**stickParams)
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
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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

        stick = LFPy.TemplateCell(**stickParams)
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
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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

        stick = LFPy.TemplateCell(**stickParams)
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
            'morphology': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick.hoc'),
            'templatefile': os.path.join(
                LFPy.__path__[0],
                'test',
                'stick_template.hoc'),
            'templatename': 'stick_template',
            'templateargs': None,
            'cm': 1,
            'Ra': 150,
            'v_init': -65,
            'passive': True,
            'passive_parameters': {
                'g_pas': 1. / 30000,
                'e_pas': -65},
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

        stick = LFPy.TemplateCell(**stickParams)
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

    def test_cell_distort_geometry_00(self):
        cell0 = LFPy.TemplateCell(
            morphology=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(
                LFPy.__path__[0],
                'test',
                'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
        )
        factors = [-0.2, 0.1, 0., 0.1, 0.2]
        nus = [-0.5, 0., 0.5]
        for factor in factors:
            for nu in nus:
                for axis in 'xyz':
                    cell1 = LFPy.TemplateCell(
                        morphology=os.path.join(
                            LFPy.__path__[0],
                            'test',
                            'ball_and_sticks_w_lists.hoc'),
                        templatefile=os.path.join(
                            LFPy.__path__[0],
                            'test',
                            'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                    )
                    cell1.distort_geometry(factor=factor, nu=nu, axis=axis)
                    for ax in 'xyz'.replace(axis, ''):
                        np.testing.assert_allclose(
                            getattr(cell0, ax) * (1 + factor * nu),
                            getattr(cell1, ax))
                    np.testing.assert_allclose(
                        getattr(cell0, axis) * (1 - factor),
                        getattr(cell1, axis))

# ####### Functions used by tests: ########################################


def stickSimulationTesttvec(**kwargs):
    stick = LFPy.TemplateCell(
        morphology=os.path.join(
            LFPy.__path__[0],
            'test',
            'stick.hoc'),
        verbose=False,
        templatefile=os.path.join(
            LFPy.__path__[0],
            'test',
            'stick_template.hoc'),
        templatename='stick_template',
        templateargs=None,
        **kwargs)
    stick.simulate(rec_imem=False)
    return stick.tvec
