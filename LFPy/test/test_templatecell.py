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
    if not pth in neuron.nrn_dll_loaded:
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
            'dt' : 2**-3,
            'tstart' : 0.,
            'tstop' : 100.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_01(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : 0.,
            'tstop' : 100.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_02(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : 0.,
            'tstop' : 10000.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_03(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : 0.,
            'tstop' : 10000.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])


    def test_cell_tvec_04(self):
        stickParams = {
            'dt' : 0.1,
            'tstart' : 0,
            'tstop' : 100,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_05(self):
        stickParams = {
            'dt' : 0.1,
            'tstart' : 0.,
            'tstop' : 100.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        for i in range(tvec.size):
            self.assertAlmostEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_06(self):
        stickParams = {
            'dt' : 0.1,
            'tstart' : 0,
            'tstop' : 10000,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        self.assertEqual(tvec.size, tvec_numpy.size)

    def test_cell_tvec_07(self):
        stickParams = {
            'dt' : 0.1,
            'tstart' : 0.,
            'tstop' : 10000.,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_08(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : -100,
            'tstop' : 100,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        self.assertEqual(tvec.size, tvec_numpy.size)


    def test_cell_tvec_09(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : -100,
            'tstop' : 100,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_10(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : -100,
            'tstop' : 10000,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        self.assertEqual(tvec.size, tvec_numpy.size)


    def test_cell_tvec_11(self):
        stickParams = {
            'dt' : 2**-3,
            'tstart' : -100,
            'tstop' : 10000,
        }

        tvec = stickSimulationTesttvec(**stickParams)
        tvec_numpy = np.linspace(0, stickParams['tstop'],
                    stickParams['tstop']/stickParams['dt'] + 1)

        for i in range(tvec.size):
            self.assertEqual(tvec[i], tvec_numpy[i])

    def test_cell_tvec_12(self):
        stickParams = {
            'dt' : 0.1,
            'tstart' : -100,
            'tstop' : 10000,
        }

        try:
            stickSimulationTesttvec(**stickParams)
        except AssertionError:
            pass

    def test_cell_set_pos_00(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                         )
        np.testing.assert_allclose(cell.somapos, [0, 0, 0])

    def test_cell_set_pos_01(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_02(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                          pt3d=True)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_03(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                         )
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])

    def test_cell_set_pos_04(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc'),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        pt3d=True)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        cell.set_pos(10., 20., -30.)
        np.testing.assert_allclose(cell.somapos, [10., 20., -30.])


    def test_cell_set_pos_05(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        np.testing.assert_allclose(cell.somapos,
                                   [cell.xmid[0], cell.ymid[0], cell.zmid[0]])


    def test_cell_set_pos_06(self):
        '''test LFPy.TemplateCell.set_pos'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        pt3d=True)
        np.testing.assert_allclose(cell.somapos,
                                   [cell.xmid[0], cell.ymid[0], cell.zmid[0]])


    def test_cell_set_rotation_00(self):
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )

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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )

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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
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
        '''test LFPy.TemplateCell.set_rotation()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )

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
        '''test LFPy.TemplateCell.chiral_morphology()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                         )

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


    def test_cell_chiral_morphology_01(self):
        '''test LFPy.TemplateCell.chiral_morphology()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )

        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around y-axis
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

    def test_cell_chiral_morphology_02(self):
        '''test LFPy.TemplateCell.chiral_morphology()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )

        xstarts = cell.xstart.copy()
        xmids = cell.xmid.copy()
        xends = cell.xend.copy()
        ystarts = cell.ystart.copy()
        ymids = cell.ymid.copy()
        yends = cell.yend.copy()
        zstarts = cell.zstart.copy()
        zmids = cell.zmid.copy()
        zends = cell.zend.copy()
        # test rotation 180 deg around z-axis
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
        '''test LFPy.TemplateCell.get_rand_prob_area_norm()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        p = cell.get_rand_prob_area_norm()
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertTrue(p.min() >= 0.)
        self.assertTrue(p.max() <= 1.)


    def test_cell_get_rand_prob_area_norm_from_idx(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        p = cell.get_rand_prob_area_norm_from_idx(idx=cell.get_idx(section='allsec'))
        self.assertListEqual(cell.get_rand_prob_area_norm().tolist(), p.tolist())


    def test_cell_get_rand_prob_area_norm_from_idx_00(self):
        '''test LFPy.TemplateCell.get_rand_prob_area_norm()'''
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        p = cell.get_rand_prob_area_norm_from_idx(idx=np.array([0]))
        np.testing.assert_equal(p, np.array([1.]))


    def test_cell_get_intersegment_vector_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        idx0 = 0
        idx1 = 1
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)

        self.assertListEqual(vector,
                            [cell.xmid[idx1] - cell.xmid[idx0],
                             cell.ymid[idx1] - cell.ymid[idx0],
                             cell.zmid[idx1] - cell.zmid[idx0]])


    def test_cell_get_intersegment_distance_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        idx0 = 0
        idx1 = 1
        distance = cell.get_intersegment_distance(idx0=idx0, idx1=idx1)
        vector = cell.get_intersegment_vector(idx0=idx0, idx1=idx1)

        self.assertEqual(np.sqrt(np.array(vector)**2).sum(), distance)


    def test_cell_get_idx_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
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
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        nsegs_method=None)
        self.assertEqual(cell.get_closest_idx(x=0, y=0, z=0),
                             cell.get_idx(section='soma')[0])

        self.assertEqual(cell.get_closest_idx(x=-25, y=0, z=175),
                             cell.get_idx(section='dend[1]')[0])


    def test_cell_get_idx_children_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )

        np.testing.assert_array_equal(cell.get_idx_children(parent='soma[0]'),
                                      cell.get_idx(section='dend[0]'))


    def test_cell_get_idx_parent_children_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        np.testing.assert_array_equal(cell.get_idx_parent_children(parent='soma[0]'),
                                      cell.get_idx(section=['soma[0]',
                                                            'dend[0]']))


    def test_cell_get_idx_name_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        np.testing.assert_array_equal(cell.get_idx_name(idx=np.array([0])),
                                                np.array([[0, 'ball_and_stick_template[0].soma[0]', 0.5]],
                                                         dtype=object))


    def test_cell_get_rand_idx_area_norm_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
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
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
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
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        cell.set_point_process(idx=0, pptype='IClamp', record_current=False,
                               **dict(delay=1., amp=1.))
        self.assertEqual(cell.stimlist[0].hname(), 'IClamp[0]')
        self.assertEqual(len(cell.stimlist), 1)
        self.assertEqual(cell.stimlist[0].delay, 1.)
        self.assertEqual(cell.stimlist[0].amp, 1.)


    def test_cell_strip_hoc_objects_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        cell.strip_hoc_objects()
        for attribute in dir(cell):
            self.assertNotEqual(str(type(getattr(cell, attribute))),
                                'hoc.HocObject')

    def test_cell_cellpickler_00(self):
        cell = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
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


    def test_cell_simulate_current_dipole_moment_00(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'templatefile' : os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
            'templatename' : 'stick_template',
            'templateargs' : None,
            'cm' : 1,
            'Ra' : 150,
            'v_init' : -65,
            'passive' : True,
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'tstart' : 0,
            'tstop' : 100,
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
                stick = LFPy.TemplateCell(**stickParams)
                synapse = LFPy.StimIntElectrode(stick, idx=idx,
                                       **stimParams)
                stick.simulate(rec_imem=True, rec_current_dipole_moment=True)
                p = np.dot(stick.imem.T, np.c_[stick.xmid, stick.ymid, stick.zmid])
                np.testing.assert_allclose(p, stick.current_dipole_moment)


    def test_cell_simulate_current_dipole_moment_01(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'templatefile' : os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
            'templatename' : 'stick_template',
            'templateargs' : None,
            'cm' : 1,
            'Ra' : 150,
            'v_init' : -65,
            'passive' : True,
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'tstart' : -100,
            'tstop' : 100,
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
                stick = LFPy.TemplateCell(**stickParams)
                synapse = LFPy.StimIntElectrode(stick, idx=idx,
                                       **stimParams)
                stick.simulate(rec_imem=True, rec_current_dipole_moment=True)
                p = np.dot(stick.imem.T, np.c_[stick.xmid, stick.ymid, stick.zmid])
                np.testing.assert_allclose(p, stick.current_dipole_moment)

    def test_cell_simulate_current_dipole_moment_02(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'templatefile' : os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
            'templatename' : 'stick_template',
            'templateargs' : None,
            'cm' : 1,
            'Ra' : 150,
            'v_init' : -65,
            'passive' : True,
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'tstart' : -100,
            'tstop' : 100,
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
                stick = LFPy.TemplateCell(**stickParams)
                synapse = LFPy.Synapse(stick, idx=idx,
                                       **stimParams)
                synapse.set_spike_times(np.array([10., 20., 30., 40., 50.]))
                stick.simulate(rec_imem=True, rec_current_dipole_moment=True)
                p = np.dot(stick.imem.T, np.c_[stick.xmid, stick.ymid, stick.zmid])
                np.testing.assert_allclose(p, stick.current_dipole_moment)

    def test_cell_tstart_00(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'templatefile' : os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
            'templatename' : 'stick_template',
            'templateargs' : None,
            'cm' : 1,
            'Ra' : 150,
            'v_init' : -65,
            'passive' : True,
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
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

        stick0 = LFPy.TemplateCell(tstart=0, tstop=200, **stickParams)
        synapse0 = LFPy.StimIntElectrode(stick0,
                                         stick0.get_closest_idx(0, 0, 1000),
                                         delay=0, phase=0.,
                                         **stimParams)
        stick0.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)


        stick1 = LFPy.TemplateCell(tstart=-100, tstop=100, **stickParams)
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
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'templatefile' : os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
            'templatename' : 'stick_template',
            'templateargs' : None,
            'cm' : 1,
            'Ra' : 150,
            'v_init' : -65,
            'passive' : True,
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'tstart' : 0,
            'tstop' : 100,
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

        stick = LFPy.TemplateCell(**stickParams)
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        stick.simulate(electrode, rec_imem=True)

        electrode1 = LFPy.RecExtElectrode(cell=stick, **electrodeParams)
        electrode1.calc_lfp()

        np.testing.assert_allclose(electrode.LFP, electrode1.LFP)


    def test_cell_with_recextelectrode_01(self):
        stickParams = {
            'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
            'templatefile' : os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
            'templatename' : 'stick_template',
            'templateargs' : None,
            'cm' : 1,
            'Ra' : 150,
            'v_init' : -65,
            'passive' : True,
            'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
            'tstart' : -100,
            'tstop' : 100,
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

        stick = LFPy.TemplateCell(**stickParams)
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        stick.simulate(electrode, rec_imem=True)

        electrode1 = LFPy.RecExtElectrode(cell=stick, **electrodeParams)
        electrode1.calc_lfp()

        np.testing.assert_allclose(electrode.LFP, electrode1.LFP)


    def test_cell_distort_geometry_01(self):
        cell0 = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
        factors = [-0.2, 0.1, 0., 0.1, 0.2]
        nus = [-0.5, 0., 0.5]
        for factor in factors:
            for nu in nus:
                for axis in 'xyz':
                    cell1 = LFPy.TemplateCell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                  'ball_and_sticks_w_lists.hoc' ),
                        templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
                        templatename='ball_and_stick_template',
                        templateargs=None,
                        )
                    cell1.distort_geometry(factor=factor, nu=nu, axis=axis)
                    for attr in ['start', 'mid', 'end']:
                        for ax in 'xyz'.replace(axis, ''):
                            np.testing.assert_allclose(getattr(cell0, ax+attr)*(1+factor*nu),
                                                       getattr(cell1, ax+attr))
                        np.testing.assert_allclose(getattr(cell0, axis+attr)*(1-factor),
                                                   getattr(cell1, axis+attr))


    ######## Functions used by tests: ##########################################

def stickSimulationTesttvec(**kwargs):
    stick = LFPy.TemplateCell(morphology = os.path.join(LFPy.__path__[0], 'test',
                                                'stick.hoc'), verbose=False,
                              templatefile=os.path.join(LFPy.__path__[0], 'test', 'stick_template.hoc'),
                              templatename='stick_template',
                              templateargs=None,
                              **kwargs)
    stick.simulate(rec_imem=False)
    return stick.tvec
