#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2020 Computational Neuroscience Group, NMBU.

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
import shutil
import tempfile
import unittest
import numpy as np
import LFPy


def _test_file(filename):
    """Absolute path to a file in the LFPy/test directory"""
    return os.path.join(LFPy.__path__[0], 'test', filename)


class testMorphologyImport(unittest.TestCase):
    """
    test that importing a morphology in NEURON results in the correct
    morphology in LFPy.Cell
    """

    def test_morphology_import_00(self):
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'L23E_p23.hoc'),
                         nsegs_method=None)
        data = np.c_[cell.x[:, 0], cell.x.mean(axis=-1), cell.x[:, -1],
                     cell.y[:, 0], cell.y.mean(axis=-1), cell.y[:, -1],
                     cell.z[:, 0], cell.z.mean(axis=-1), cell.z[:, -1]]
        expected_data = np.array([
            [0., 0., 0., 0., 0., 0., -10., 0., 10.],
            [0., 50., 100., 0., -50., -100., 10., 55., 100.],
            [0., -50., -100., 0., 50., 100., 10., 55., 100.],
            [0., 50., 100., 0., 50., 100., -10., -55., -100.],
            [0., -50., -100., 0., -50., -100., -10., -55., -100.],
            [0., 0., 0., 0., 0., 0., 10., 167., 324.]])

        np.testing.assert_equal(data, expected_data)

    def test_morphology_import_swc_cell(self):
        '''LFPy.Cell can import a morphology in SWC format'''
        cell = LFPy.Cell(morphology=_test_file('ball_and_stick.swc'),
                         nsegs_method=None)
        # one soma section plus one dendritic section
        self.assertEqual(cell.totnsegs, 2)
        np.testing.assert_allclose(cell.d, [20., 2.])
        np.testing.assert_allclose(cell.length, [20., 1000.])

    def test_morphology_import_uppercase_hoc_cell(self):
        '''LFPy.Cell treats a .HOC file as a hoc morphology'''
        morphology = _test_file('ball_and_sticks.hoc')
        cell = LFPy.Cell(morphology=morphology, nsegs_method=None)
        tmpdir = tempfile.mkdtemp()
        try:
            uppercase = os.path.join(tmpdir, 'ball_and_sticks.HOC')
            shutil.copy(morphology, uppercase)
            cell_uppercase = LFPy.Cell(morphology=uppercase,
                                       nsegs_method=None)
        finally:
            shutil.rmtree(tmpdir)
        self.assertEqual(cell_uppercase.totnsegs, cell.totnsegs)
        np.testing.assert_allclose(cell_uppercase.z, cell.z)

    def test_morphology_import_uppercase_hoc_templatecell(self):
        '''LFPy.TemplateCell treats a .HOC file as a hoc morphology'''
        morphology = _test_file('ball_and_sticks_w_lists.hoc')
        templatefile = _test_file('ball_and_stick_template.hoc')
        cell = LFPy.TemplateCell(morphology=morphology,
                                 templatefile=templatefile,
                                 templatename='ball_and_stick_template',
                                 templateargs=None,
                                 nsegs_method=None)
        tmpdir = tempfile.mkdtemp()
        try:
            uppercase = os.path.join(tmpdir, 'ball_and_sticks_w_lists.HOC')
            shutil.copy(morphology, uppercase)
            cell_uppercase = LFPy.TemplateCell(
                morphology=uppercase,
                templatefile=templatefile,
                templatename='ball_and_stick_template',
                templateargs=None,
                nsegs_method=None)
        finally:
            shutil.rmtree(tmpdir)
        self.assertEqual(cell_uppercase.totnsegs, cell.totnsegs)
        np.testing.assert_allclose(cell_uppercase.z, cell.z)

    def test_morphology_import_unsupported_fileending_cell(self):
        '''LFPy.Cell raises ValueError on an unsupported morphology format'''
        tmpdir = tempfile.mkdtemp()
        try:
            morphology = os.path.join(tmpdir, 'ball_and_stick.txt')
            shutil.copy(_test_file('ball_and_stick.swc'), morphology)
            with self.assertRaises(ValueError):
                LFPy.Cell(morphology=morphology)
        finally:
            shutil.rmtree(tmpdir)

    def test_morphology_import_unsupported_fileending_templatecell(self):
        '''LFPy.TemplateCell raises ValueError on an unsupported format'''
        tmpdir = tempfile.mkdtemp()
        try:
            morphology = os.path.join(tmpdir, 'ball_and_stick.txt')
            shutil.copy(_test_file('ball_and_stick.swc'), morphology)
            with self.assertRaises(ValueError):
                LFPy.TemplateCell(
                    morphology=morphology,
                    templatefile=_test_file('ball_and_stick_template.hoc'),
                    templatename='ball_and_stick_template',
                    templateargs=None)
        finally:
            shutil.rmtree(tmpdir)
