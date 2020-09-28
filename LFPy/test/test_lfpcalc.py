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
from lfpykit import CellGeometry
from LFPy import lfpcalc


class testLfpCalc(unittest.TestCase):
    """
    test module lfpykit.lfpcalc
    """

    def test_lfpcalc_return_dist_from_segments_00(self):
        """test function lfpcalc.return_dist_from_segments"""
        cell = DummyCell()
        dist, clostest_point = lfpcalc.return_dist_from_segments(
            cell.x[:, 0], cell.y[:, 0],
            cell.z[:, 0], cell.x[:, -1],
            cell.y[:, -1], cell.z[:, -1],
            [1, 10, 0])
        np.testing.assert_equal([10], dist)
        np.testing.assert_equal([1, 0, 0], clostest_point.T[0])

        dist, clostest_point = lfpcalc.return_dist_from_segments(
            cell.x[:, 0], cell.y[:, 0],
            cell.z[:, 0], cell.x[:, -1],
            cell.y[:, -1], cell.z[:, -1],
            [-1, 10, 0])
        np.testing.assert_equal([np.sqrt(101)], dist)
        np.testing.assert_equal([0, 0, 0], clostest_point.T[0])

    def test_lfpcalc_calc_lfp_pointsource_00(self):
        """Test that function lfpcalc.calc_lfp_pointsource
        reproduces analytic formula"""
        sigma = 0.3
        cell = DummyCell()
        np.testing.assert_equal(1. / (4 * np.pi * sigma),
                                lfpcalc.calc_lfp_pointsource(cell,
                                                             x=0.5, y=0, z=1,
                                                             sigma=sigma,
                                                             r_limit=cell.d / 2
                                                             ))

    def test_lfpcalc_calc_lfp_pointsource_moi_00(self):
        """
        Test slice where all layers have same conductivity reproduces
        isotropic case.
        """
        sigma_T = 0.3
        sigma_G = 0.3
        sigma_S = 0.3
        h = 300
        steps = 20
        cell = DummyCell(np.array([[h / 2, h / 2]]))

        in_vivo = lfpcalc.calc_lfp_pointsource(cell,
                                               x=0.5, y=0, z=1, sigma=sigma_T,
                                               r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                    x=0.5, y=0, z=1,
                                                    sigma_T=sigma_T,
                                                    sigma_G=sigma_G,
                                                    sigma_S=sigma_S,
                                                    r_limit=cell.d / 2,
                                                    h=h,
                                                    steps=steps)

        np.testing.assert_equal(in_vivo, in_vitro)

    def test_lfpcalc_calc_lfp_pointsource_moi_01(self):
        """
        Test that ECoG scenario gives expected analytical result
        """

        sigma_T = 0.3
        sigma_G = 0.3
        sigma_S = 1.5
        h = 5000
        steps = 20
        cell = DummyCell(x=np.array([[0, 0]]),
                         z=np.array([[h - 50, h - 50]]))

        source_scaling = (sigma_T - sigma_S) / (sigma_S + sigma_T)

        z = h - 20  # Recording position z <= h, z != cell.z.mean(axis=-1)[0]

        analytic = cell.imem[0] / (4 * np.pi * sigma_T) * (
            1 / np.abs(z - cell.z.mean(axis=-1)[0]) +  # real source
            # image source
            source_scaling / np.abs(z - (2 * h - cell.z.mean(axis=-1)[0]))
        )

        moi_method_lfpy = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                           x=0., y=0, z=z,
                                                           sigma_T=sigma_T,
                                                           sigma_G=sigma_G,
                                                           sigma_S=sigma_S,
                                                           r_limit=cell.d / 2,
                                                           h=h, steps=steps)

        np.testing.assert_equal(analytic, moi_method_lfpy)

    def test_lfpcalc_calc_lfp_pointsource_moi_02(self):
        """
        Very close to point source, in vivo and in vitro have similar results,
        e.g., the positions should be adjusted similarly.
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 2000
        steps = 20
        cell = DummyCell(z=np.array([[h / 2, h / 2]]))

        in_vivo = lfpcalc.calc_lfp_pointsource(cell,
                                               x=0.5, y=0, z=h / 2,
                                               sigma=sigma_T,
                                               r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                    x=0.5, y=0, z=h / 2,
                                                    sigma_T=sigma_T,
                                                    sigma_G=sigma_G,
                                                    sigma_S=sigma_S,
                                                    r_limit=cell.d / 2,
                                                    h=h,
                                                    steps=steps)

        np.testing.assert_almost_equal(in_vivo, in_vitro, 4)

    def test_lfpcalc_calc_lfp_linesource_moi_00(self):
        """
        Very close to point source, in vivo and in vitro have similar results,
        e.g., the positions should be adjusted similarly.
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 0.3
        h = 200

        steps = 20
        cell = DummyCell()

        in_vivo = lfpcalc.calc_lfp_linesource(cell,
                                              x=0.5, y=0, z=0, sigma=sigma_T,
                                              r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_linesource_moi(cell,
                                                   x=0.5, y=0, z=0,
                                                   sigma_T=sigma_T,
                                                   sigma_G=sigma_G,
                                                   sigma_S=sigma_S,
                                                   r_limit=cell.d / 2,
                                                   h=h,
                                                   steps=steps)

        np.testing.assert_almost_equal(2 * in_vivo, in_vitro, 4)

    def test_lfpcalc_calc_lfp_pointsource_moi_03(self):
        """
        Very close to point source, in vivo and in vitro have similar results,
        e.g., the positions should be adjusted similarly.
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 2000
        steps = 20
        cell = DummyCell()

        in_vivo = lfpcalc.calc_lfp_root_as_point(cell,
                                                 x=0.0, y=0, z=0,
                                                 sigma=sigma_T,
                                                 r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                      x=0.0, y=0, z=0,
                                                      sigma_T=sigma_T,
                                                      sigma_G=sigma_G,
                                                      sigma_S=sigma_S,
                                                      r_limit=cell.d / 2,
                                                      h=h,
                                                      steps=steps)

        np.testing.assert_almost_equal(2 * in_vivo, in_vitro, 4)

    def test_lfpcalc_calc_lfp_linesource_00(self):
        """
        Test that calc_lfp_linesource method does not give infinite potential
        """
        sigma_T = 0.3
        cell = DummyCell()

        in_vivo = lfpcalc.calc_lfp_linesource(cell,
                                              x=0.5, y=0.0, z=0, sigma=sigma_T,
                                              r_limit=cell.d / 2)[0]
        np.testing.assert_array_less(in_vivo, 1e12)

    def test_lfpcalc_calc_lfp_pointsource_moi_04(self):
        """
        Test that slice with zero-conductivity MEA region (z<0) has twice
        the potential as in vivo case at MEA electrode plane
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 0.3
        h = 200
        steps = 3

        cell = DummyCell(z=np.array([[50, 50]]))

        in_vivo = lfpcalc.calc_lfp_pointsource(cell,
                                               x=50., y=0, z=0,
                                               sigma=sigma_T,
                                               r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                    x=50, y=0, z=0,
                                                    sigma_T=sigma_T,
                                                    sigma_G=sigma_G,
                                                    sigma_S=sigma_S,
                                                    r_limit=cell.d / 2,
                                                    h=h,
                                                    steps=steps)

        np.testing.assert_almost_equal(2 * in_vivo, in_vitro, decimal=9)

    def test_lfpcalc_calc_lfp_linesource_moi_01(self):
        """
        Test that slice with zero conductivity in MEA region (z<0) has twice
        the potential as in vivo case at MEA electrode plane
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 0.3
        h = 200
        steps = 3

        cell = DummyCell(z=np.array([[50, 50]]))

        in_vivo = lfpcalc.calc_lfp_linesource(cell,
                                              x=50., y=0, z=0,
                                              sigma=sigma_T,
                                              r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_linesource_moi(cell,
                                                   x=50, y=0, z=0,
                                                   sigma_T=sigma_T,
                                                   sigma_G=sigma_G,
                                                   sigma_S=sigma_S,
                                                   r_limit=cell.d / 2,
                                                   h=h,
                                                   steps=steps)

        np.testing.assert_almost_equal(2 * in_vivo, in_vitro, decimal=9)

    def test_lfpcalc_calc_lfp_root_as_point_moi_doubling(self):
        """
        Test that slice with zero conductivity in MEA region (z<0) has twice
        the potential as in vivo case at MEA electrode plane
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 0.3
        h = 200
        steps = 3

        cell = DummyCell(z=np.array([[50, 50]]))

        in_vivo = lfpcalc.calc_lfp_root_as_point(cell,
                                                 x=50., y=0, z=0,
                                                 sigma=sigma_T,
                                                 r_limit=cell.d / 2)
        in_vitro = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                      x=50, y=0, z=0,
                                                      sigma_T=sigma_T,
                                                      sigma_G=sigma_G,
                                                      sigma_S=sigma_S,
                                                      r_limit=cell.d / 2,
                                                      h=h,
                                                      steps=steps)

        np.testing.assert_almost_equal(2 * in_vivo, in_vitro, decimal=9)

    def test_lfpcalc_calc_lfp_pointsource_moi_saline_effect(self):
        """
        Test that the saline bath decreases signal as expected
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 200
        steps = 20

        cell = DummyCell(z=np.array([[100, 100]]))

        with_saline = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                       x=0, y=0, z=0,
                                                       sigma_T=sigma_T,
                                                       sigma_G=sigma_G,
                                                       sigma_S=sigma_S,
                                                       r_limit=cell.d / 2,
                                                       h=h,
                                                       steps=steps)

        without_saline = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                          x=0, y=0, z=0,
                                                          sigma_T=sigma_T,
                                                          sigma_G=sigma_G,
                                                          sigma_S=sigma_T,
                                                          r_limit=cell.d / 2,
                                                          h=h,
                                                          steps=steps)

        np.testing.assert_array_less(with_saline, without_saline)

    def test_lfpcalc_calc_lfp_linesource_moi_saline_effect(self):
        """
        Test that the saline bath decreases signal as expected
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 200
        steps = 20

        cell = DummyCell(z=np.array([[100, 100]]))

        with_saline = lfpcalc.calc_lfp_linesource_moi(cell,
                                                      x=0, y=0, z=0,
                                                      sigma_T=sigma_T,
                                                      sigma_G=sigma_G,
                                                      sigma_S=sigma_S,
                                                      r_limit=cell.d / 2,
                                                      h=h,
                                                      steps=steps)

        without_saline = lfpcalc.calc_lfp_linesource_moi(cell,
                                                         x=0, y=0, z=0,
                                                         sigma_T=sigma_T,
                                                         sigma_G=sigma_G,
                                                         sigma_S=sigma_T,
                                                         r_limit=cell.d / 2,
                                                         h=h,
                                                         steps=steps)

        np.testing.assert_array_less(with_saline, without_saline)

    def test_lfpcalc_calc_lfp_root_as_point_moi_saline_effect(self):
        """
        Test that the saline bath decreases signal as expected
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 200
        steps = 20

        cell = DummyCell(z=np.array([[100, 100]]))

        with_saline = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                         x=0, y=0, z=0,
                                                         sigma_T=sigma_T,
                                                         sigma_G=sigma_G,
                                                         sigma_S=sigma_S,
                                                         r_limit=cell.d / 2,
                                                         h=h,
                                                         steps=steps)

        without_saline = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                            x=0, y=0, z=0,
                                                            sigma_T=sigma_T,
                                                            sigma_G=sigma_G,
                                                            sigma_S=sigma_T,
                                                            r_limit=cell.d / 2,
                                                            h=h,
                                                            steps=steps)

        np.testing.assert_array_less(with_saline, without_saline)

    def test_lfpcalc_calc_lfp_pointsource_moi_20steps(self):
        """
        Test that the calc_lfp_pointsource_moi reproduces previously known
        nummerical value
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 200
        steps = 20

        correct = 0.00108189

        cell = DummyCell(x=np.array([[0., -200.]]), z=np.array(([[0., 220.]])))

        calculated = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                      x=100, y=0, z=0,
                                                      sigma_T=sigma_T,
                                                      sigma_G=sigma_G,
                                                      sigma_S=sigma_S,
                                                      r_limit=cell.d / 2,
                                                      h=h,
                                                      steps=steps)

        np.testing.assert_almost_equal(correct, calculated, 5)

    def test_lfpcalc_calc_lfp_linesource_moi_20steps(self):
        """
        Test that the calc_lfp_linesource_moi reproduces previously known
        nummerical value
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 200
        steps = 20

        correct = 0.00246539

        cell = DummyCell(x=np.array([[-100, 50]]),
                         z=np.array([[0, 110]]))

        calculated = lfpcalc.calc_lfp_linesource_moi(cell,
                                                     x=100, y=0, z=0,
                                                     sigma_T=sigma_T,
                                                     sigma_G=sigma_G,
                                                     sigma_S=sigma_S,
                                                     r_limit=cell.d / 2,
                                                     h=h,
                                                     steps=steps)

        np.testing.assert_almost_equal(correct, calculated, 5)

    def test_lfpcalc_calc_lfp_root_as_point_moi_20steps(self):
        """
        Test that the calc_lfp_root_as_point_moi reproduces previously known
        nummerical value
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 200
        steps = 20

        correct = 0.00108189

        cell = DummyCell(x=np.array([[0, -200]]),
                         y=np.array([[0., 0.]]),
                         z=np.array([[0, 220]]))

        calculated = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                        x=100, y=0, z=0,
                                                        sigma_T=sigma_T,
                                                        sigma_G=sigma_G,
                                                        sigma_S=sigma_S,
                                                        r_limit=cell.d / 2,
                                                        h=h,
                                                        steps=steps)

        np.testing.assert_almost_equal(correct, calculated, 5)

    def test_lfpcalc_calc_lfp_pointsource_moi_infinite_slice(self):
        """
        Test that infinitely thick slice does not affect potential.
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 1e10
        steps = 20

        cell = DummyCell(z=np.array([[100, 100]]))

        with_saline = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                       x=0, y=0, z=50,
                                                       sigma_T=sigma_T,
                                                       sigma_G=sigma_G,
                                                       sigma_S=sigma_S,
                                                       r_limit=cell.d / 2,
                                                       h=h,
                                                       steps=steps)

        without_saline = lfpcalc.calc_lfp_pointsource_moi(cell,
                                                          x=0, y=0, z=50,
                                                          sigma_T=sigma_T,
                                                          sigma_G=sigma_G,
                                                          sigma_S=sigma_T,
                                                          r_limit=cell.d / 2,
                                                          h=h,
                                                          steps=steps)

        np.testing.assert_almost_equal(with_saline, without_saline)

    def test_lfpcalc_calc_lfp_linesource_moi_infinite_slice(self):
        """
        Test that infinitely thick slice does not affect potential.
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 1e10
        steps = 20

        cell = DummyCell(z=np.array([[100, 100]]))

        with_saline = lfpcalc.calc_lfp_linesource_moi(cell,
                                                      x=0, y=0, z=0,
                                                      sigma_T=sigma_T,
                                                      sigma_G=sigma_G,
                                                      sigma_S=sigma_S,
                                                      r_limit=cell.d / 2,
                                                      h=h,
                                                      steps=steps)

        without_saline = lfpcalc.calc_lfp_linesource_moi(cell,
                                                         x=0, y=0, z=0,
                                                         sigma_T=sigma_T,
                                                         sigma_G=sigma_G,
                                                         sigma_S=sigma_T,
                                                         r_limit=cell.d / 2,
                                                         h=h,
                                                         steps=steps)

        np.testing.assert_almost_equal(with_saline, without_saline)

    def test_lfpcalc_calc_lfp_root_as_point_moi_infinite_slice(self):
        """
        Test that infinitely thick slice does not affect potential.
        """
        sigma_T = 0.3
        sigma_G = 0.0
        sigma_S = 1.5
        h = 1e10
        steps = 20

        cell = DummyCell(z=np.array([[100, 100]]))

        with_saline = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                         x=0, y=0, z=0,
                                                         sigma_T=sigma_T,
                                                         sigma_G=sigma_G,
                                                         sigma_S=sigma_S,
                                                         r_limit=cell.d / 2,
                                                         h=h,
                                                         steps=steps)

        without_saline = lfpcalc.calc_lfp_root_as_point_moi(cell,
                                                            x=0, y=0, z=0,
                                                            sigma_T=sigma_T,
                                                            sigma_G=sigma_G,
                                                            sigma_S=sigma_T,
                                                            r_limit=cell.d / 2,
                                                            h=h,
                                                            steps=steps)

        np.testing.assert_almost_equal(with_saline, without_saline)

    def test_lfpcalc_calc_lfp_pointsource_anisotropic(self):

        sigma = [0.6, 0.3, 0.45]
        cell = DummyCell(x=np.array([[0., 2.4]]),
                         y=np.array([[0., 2.4]]),
                         z=np.array([[0., 2.4]]),
                         d=np.array([1.]))

        sigma_r = np.sqrt(sigma[1] * sigma[2] * 1.2**2
                          + sigma[0] * sigma[2] * 1.2**2
                          + sigma[0] * sigma[1] * 1.2**2)

        phi_analytic = 1. / (4 * np.pi * sigma_r)
        np.testing.assert_equal(
            phi_analytic, lfpcalc.calc_lfp_pointsource_anisotropic(
                cell, x=0, y=0, z=0, sigma=sigma, r_limit=cell.d / 2))

    def test_deltaS_calc(self):
        cell = DummyCell(y=np.array([[0, 5]]))
        ds = lfpcalc._deltaS_calc(cell.x[:, 0], cell.x[:, -1],
                                  cell.y[:, 0], cell.y[:, -1],
                                  cell.z[:, 0], cell.z[:, -1])
        np.testing.assert_equal(ds, np.sqrt(26))


class DummyCell(CellGeometry):
    """CellGeometry like object with attributes for predicting extracellular
    potentials, but with:
        - 1 compartment
        - position in (0.5,0,0) [um]
        - length 1 [um]
        - diam 1 [um]
        - current amplitude 1 [nA]
        - 1 timestep
    """

    def __init__(self,
                 x=np.array([[0., 1.]]),
                 y=np.array([[0., 0.]]),
                 z=np.array([[0., 0.]]),
                 d=np.array([1.])):
        super().__init__(x=x, y=y, z=z, d=d)

        # set Attributes
        self.imem = np.array([[1.]])
