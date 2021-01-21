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
import os
import numpy as np
import LFPy
import neuron


class testMEG(unittest.TestCase):
    """
    test class eegmegcalc.MEG
    """

    def test_MEG_00(self):
        '''test eegmegcalc.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((3, 11))
        current_dipole_moment[0, :] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0], 3,
                       current_dipole_moment.shape[1]))
        gt[1, 2, :] = 1. / 4 / np.pi
        gt[2, 1, :] = -1. / 4 / np.pi
        gt[4, 2, :] = -1. / 4 / np.pi
        gt[5, 1, :] = 1. / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_01(self):
        '''test eegmegcalc.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((3, 11))
        current_dipole_moment[1, :] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0], 3,
                       current_dipole_moment.shape[1]))
        gt[0, 2, :] = -1. / 4 / np.pi
        gt[2, 0, :] = 1. / 4 / np.pi
        gt[3, 2, :] = 1. / 4 / np.pi
        gt[5, 0, :] = -1. / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_02(self):
        '''test eegmegcalc.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((3, 11))
        current_dipole_moment[2, :] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        # ground truth
        gt = np.zeros((sensor_locations.shape[0], 3,
                       current_dipole_moment.shape[1]))
        gt[0, 1, :] = 1. / 4 / np.pi
        gt[1, 0, :] = -1. / 4 / np.pi
        gt[3, 1, :] = -1. / 4 / np.pi
        gt[4, 0, :] = 1. / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_03(self):
        '''test eegmegcalc.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((3, 1))
        current_dipole_moment[0, :] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0], 3,
                       current_dipole_moment.shape[1]))
        gt[1, 2, :] = 1. / 4 / np.pi
        gt[2, 1, :] = -1. / 4 / np.pi
        gt[4, 2, :] = -1. / 4 / np.pi
        gt[5, 1, :] = 1. / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_04(self):
        '''test eegmegcalc.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((3, 1))
        current_dipole_moment[1, :] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0], 3,
                       current_dipole_moment.shape[1]))
        gt[0, 2, :] = -1. / 4 / np.pi
        gt[2, 0, :] = 1. / 4 / np.pi
        gt[3, 2, :] = 1. / 4 / np.pi
        gt[5, 0, :] = -1. / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_05(self):
        '''test eegmegcalc.MEG.calculate_H()'''
        current_dipole_moment = np.zeros((3, 1))
        current_dipole_moment[2, :] += 1.
        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.zeros((sensor_locations.shape[0], 3,
                       current_dipole_moment.shape[1]))
        gt[0, 1, :] = 1. / 4 / np.pi
        gt[1, 0, :] = -1. / 4 / np.pi
        gt[3, 1, :] = -1. / 4 / np.pi
        gt[4, 0, :] = 1. / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        np.testing.assert_equal(gt, meg.calculate_H(current_dipole_moment,
                                                    dipole_location))

    def test_MEG_06(self):
        '''test eegmegcalc.MEG.get_transformation_matrix()'''
        current_dipole_moment = np.c_[np.eye(3), -np.eye(3)]

        dipole_location = np.zeros(3)
        sensor_locations = np.r_[np.eye(3), -np.eye(3)]

        gt = np.array([[[0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., -1.],
                        [0., -1., 0., 0., 1., 0.]],
                       [[0., 0., -1., 0., 0., 1.],
                        [0., 0., 0., 0., 0., 0.],
                        [1., 0., 0., -1., 0., 0.]],
                       [[0., 1., 0., 0., -1., 0.],
                        [-1., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0.]],
                       [[0., 0., 0., 0., 0., 0.],
                        [0., 0., -1., 0., 0., 1.],
                        [0., 1., 0., 0., -1., 0.]],
                       [[0., 0., 1., 0., 0., -1.],
                        [0., 0., 0., 0., 0., 0.],
                        [-1., 0., 0., 1., 0., 0.]],
                       [[0., -1., 0., 0., 1., 0.],
                        [1., 0., 0., -1., 0., 0.],
                        [0., 0., 0., 0., 0., 0.]]]) / 4 / np.pi

        meg = LFPy.MEG(sensor_locations)
        M = meg.get_transformation_matrix(dipole_location)

        np.testing.assert_equal(gt, M @ current_dipole_moment)


class testFourSphereVolumeConductor(unittest.TestCase):
    """
    test class eegmegcalc.FourSphereVolumeConductor
    """

    def test_rz_params(self):
        radii = [1., 2., 4., 8.]
        sigmas = [1., 2., 4., 8.]
        r_el = np.array([[1., 0., 7.]])
        fs = LFPy.FourSphereVolumeConductor(r_electrodes=r_el,
                                            radii=radii,
                                            sigmas=sigmas)

        rz1 = np.array([0., 0., 0.])
        with np.testing.assert_raises(RuntimeError):
            fs._rz_params(rz1)
        rz2 = np.array([0., 0., 1.])
        with np.testing.assert_raises(RuntimeError):
            fs._rz_params(rz2)
        rz3 = np.array([0., 0., 1.2])
        with np.testing.assert_raises(RuntimeError):
            fs._rz_params(rz3)

    def test_check_params00(self):
        '''Test that invalid radius values raises RuntimeError'''
        radii1 = [-1., 2., 4., 8.]
        radii2 = [1., .5, 4., 8.]
        radii3 = [1., 2., 1.1, 8.]
        radii4 = [1., 2., 4., 1.]
        sigmas = [1., 2., 4., 8.]
        r_el = np.array([[0., 0., 1.5]])
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii1, sigmas)
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii2, sigmas)
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii3, sigmas)
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii4, sigmas)

    def test_check_params01(self):
        '''Test that Error is raised if invalid entries in sigmas'''
        radii = [1., 2., 4., 10.]
        sigmas1 = [1., 'str', 4., 8.]
        sigmas2 = [-1., 2., 4., 8.]
        sigmas3 = [1., 2., -4., 8.]
        sigmas4 = [1., 2., 4., -8.]
        r_el = np.array([[0., 0., 1.5]])
        with np.testing.assert_raises(ValueError):
            LFPy.FourSphereVolumeConductor(r_el, radii, sigmas1)
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii, sigmas2)
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii, sigmas3)
        with np.testing.assert_raises(RuntimeError):
            LFPy.FourSphereVolumeConductor(r_el, radii, sigmas4)

    def test_check_params02(self):
        '''Test that ValueError is raised if electrode outside head'''
        radii = [1., 2., 4., 10.]
        sigmas = [1., 2., 4., 8.]
        r_el1 = np.array([[0., 0., 15.]])
        r_el2 = np.array([[0., 0., 1.5], [12., 0., 0.]])
        with np.testing.assert_raises(ValueError):
            LFPy.FourSphereVolumeConductor(r_el1, radii, sigmas)
        with np.testing.assert_raises(ValueError):
            LFPy.FourSphereVolumeConductor(r_el2, radii, sigmas)

    def test_decompose_dipole01(self):
        '''Test radial and tangential parts of dipole sums to dipole'''
        P1 = np.array([[1., 1., 1.]]).T
        p_rad, p_tan = decompose_dipole(P1)
        np.testing.assert_equal(p_rad + p_tan, P1)

    def test_decompose_dipole02(self):
        '''Test radial and tangential parts of dipole sums to dipole'''
        radii = [88000, 90000, 95000, 100000]
        sigmas = [0.3, 1.5, 0.015, 0.3]
        ps = np.array([[1000., 0., 0.],
                       [-1000., 0., 0.],
                       [0., 1000., 0.],
                       [0., -1000., 0.],
                       [0., 0., 1000.],
                       [0., 0., -1000.],
                       [10., 20., 30.],
                       [-10., -20., -30.]]).T
        p_locs = np.array([[87000., 0., 0.],
                           [-87000., 0., 0.],
                           [0., 87000., 0.],
                           [0., -87000., 0.],
                           [0., 0., 87000.],
                           [0., 0., -87000.],
                           [80000., 2000., 3000.],
                           [-2000., -80000., -3000.]])
        el_locs = np.array([[90000., 5000., -5000.]])
        fs = LFPy.FourSphereVolumeConductor(
            el_locs, radii, sigmas)
        for p_loc in p_locs:
            fs._rz_params(p_loc)
            p_rads, p_tans = fs._decompose_dipole(ps)
            np.testing.assert_equal(p_rads + p_tans, ps)

    def test_rad_dipole(self):
        '''Test that radial part of decomposed dipole is correct'''
        P1 = np.array([[1., 1., 1.]]).T
        p_rad, p_tan = decompose_dipole(P1)
        np.testing.assert_equal(p_rad, np.array([[0., 0., 1.]]).T)

    def test_tan_dipole(self):
        '''Test that tangential part of decomposed dipole is correct'''
        P1 = np.array([[1., 1., 1.]]).T
        p_rad, p_tan = decompose_dipole(P1)
        np.testing.assert_equal(p_tan, np.array([[1., 1., 0.]]).T)

    def test_calc_theta(self):
        '''Test theta: angle between rz and r'''
        rz1 = np.array([0., 0., 10.])
        r_el = np.array([[0., 0., 90.], [0., 0., -90.],
                         [0., 70., 0.], [0., -70., 0.], [0., 10., 10.]])
        fs = make_class_object(rz1, r_el)
        theta = fs._calc_theta()
        np.testing.assert_almost_equal(theta, np.array(
            [0., np.pi, np.pi / 2, np.pi / 2, np.pi / 4]))

    def test_calc_phi01(self):
        '''Test phi: azimuthal angle between rx and rxy'''
        rz1 = np.array([0., 0., 0.5])
        r_el = np.array([[0., 1., 0], [-1., -1., 1.],
                         [1., 1., 4.], [0., 0., 89.], [0., 0., -80.]])
        fs = make_class_object(rz1, r_el)
        P_tan = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]).T
        phi = fs._calc_phi(P_tan)
        np.testing.assert_almost_equal(phi,
                                       np.array([[np.pi / 2, np.pi, 0.],
                                                 [-3 * np.pi / 4,
                                                  -np.pi / 4, 0.],
                                                 [np.pi / 4,
                                                  3 * np.pi / 4, 0.],
                                                 [0., 0., 0.],
                                                 [0., 0., 0.]]))

    def test_calc_phi02(self):
        '''Test phi: azimuthal angle between rx and rxy,
           check that theta is not NaN, due to round-off errors'''
        radii = [79000., 80000., 85000., 100000.]
        sigmas = [0.3, 0.015, 15, 0.3]
        rz = np.array([0., 0., 76500.])
        r_el = np.array([[1e-5, 0, 99999.],
                         [0, 0.000123, 99998.9],
                         [-5.59822325e3, -9.69640709e3, -9.93712111e4],
                         [99990., 0., 0.001]])

        fs = LFPy.FourSphereVolumeConductor(r_el, radii, sigmas)
        fs._rz_params(rz)

        P1 = np.array([[0., 0., 123456789.],
                       [0., 0., 0.05683939],
                       [89892340., 0., -123456789],
                       [0.00004, 0.002, .0987654321],
                       [0., 0., 0.05683939],
                       [0.0003, 0.001, 123456789.],
                       [1e-11, 1e-12, 1000.],
                       [1e-15, 0, 1000.]]).T
        p_rad, p_tan = fs._decompose_dipole(P1)
        phi = fs._calc_phi(p_tan)

        np.testing.assert_equal(np.isnan(phi).any(), False)

    def test_rad_sign(self):
        '''Test if radial dipole points inwards or outwards'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 0., 90.]])
        fs = make_class_object(rz1, r_el)
        P1 = np.array([[0., 0., 1.], [0., 0., -2.]]).T
        s_vector = fs._sign_rad_dipole(P1)
        np.testing.assert_almost_equal(s_vector, np.array([1., -1.]))

    def test_calc_vn(self):
        '''test that calc_vn gives correct values'''
        n = 1
        fs = make_simple_class_object()
        v1 = fs._calc_vn(n)
        np.testing.assert_almost_equal(v1, -4.75)

    def test_calc_yn(self):
        '''test that calc_yn gives correct values'''
        n = 1
        fs = make_simple_class_object()
        y1 = fs._calc_yn(n)
        np.testing.assert_almost_equal(y1, -2.3875)

    def test_calc_zn(self):
        '''test that calc_zn gives correct values'''
        n = 1
        fs = make_simple_class_object()
        z1 = fs._calc_zn(n)
        np.testing.assert_almost_equal(z1, -2.16574585635359)

    def test_get_dipole_potential(self):
        '''test comparison between four-sphere model and model for
        infinite homogeneous space
        when sigma is constant and r4 goes to infinity'''
        sigmas = [0.3, 0.3, 0.3 + 1e-16, 0.3]
        radii = [10., 20 * 1e6, 30. * 1e6, 40. * 1e6]
        rz = np.array([0., 0., 3.])
        p = np.array([[0., 0., 100.], [50., 50., 0.]]).T
        r_elec = np.array([[0., 0., 9.],
                           [0., 0., 15.],
                           [0., 0., 25.],
                           [0., 0., 40.],
                           [0., 9., 0.],
                           [0., 15., 0.],
                           [0., 25., 0.],
                           [0., 40., 0.]])
        four_s = LFPy.FourSphereVolumeConductor(
            r_elec, radii, sigmas)
        pots_4s = four_s.get_dipole_potential(p, rz)
        inf_s = LFPy.InfiniteVolumeConductor(0.3)
        pots_inf = inf_s.get_dipole_potential(p, r_elec - rz)

        np.testing.assert_allclose(pots_4s, pots_inf, rtol=1e-6)

    def test_get_dipole_potential01(self):
        '''test comparison between analytical 4S-model and FEM simulation'''
        # load data
        fem_sim = np.load(
            os.path.join(
                LFPy.__path__[0],
                'test',
                'fem_mix_dip.npz'))
        pot_fem = fem_sim['pot_fem']  # [µV]
        p = fem_sim['p'].T  # [nA µm]
        rz = fem_sim['rz']  # [µm]
        radii = fem_sim['radii']  # [µm]
        sigmas = fem_sim['sigmas']  # [S/cm]
        ele_coords = fem_sim['ele_coords']  # [µm]

        fs = LFPy.FourSphereVolumeConductor(
            ele_coords, radii, sigmas)
        k_mV_to_muV = 1e3
        pot_analytical = fs.get_dipole_potential(
            p, rz).reshape(
            (len(ele_coords),)).reshape(
            pot_fem.shape) * k_mV_to_muV
        global_error = np.abs(pot_analytical - pot_fem) / \
            (np.max(np.abs(pot_fem)))
        np.testing.assert_array_less(global_error, 0.01)

    def test_get_dipole_potential02(self):
        '''Test radial and tangential parts of dipole sums to dipole'''
        radii = [88000, 90000, 95000, 100000]
        sigmas = [0.3, 1.5, 0.015, 0.3]

        dips = np.array([[[1000., 0., 0.]],
                         [[-1000., 0., 0.]],
                         [[0., 1000., 0.]],
                         [[0., -1000., 0.]],
                         [[0., 0., 1000.]],
                         [[0., 0., -1000.]]])

        p_locs = np.array([[87000., 0., 0.],
                           [-87000., 0., 0.],
                           [0., 87000., 0.],
                           [0., -87000., 0.],
                           [0., 0., 87000],
                           [0., 0., -87000]])

        el_locs = np.array([[[99000., 0., 0.]],
                            [[-99000., 0., 0.]],
                            [[0., 99000., 0.]],
                            [[0., -99000., 0.]],
                            [[0., 0., 99000.]],
                            [[0., 0., -99000.]]])

        for i in range(len(p_locs)):
            fs = LFPy.FourSphereVolumeConductor(
                el_locs[i], radii, sigmas)
            phi = fs.get_dipole_potential(dips[i].T, p_locs[i])
            if i == 0:
                phi0 = phi[0][0]
            else:
                np.testing.assert_equal(phi0, phi[0][0])

    def test_get_transformation_matrix_00(self):
        '''Test radial and tangential parts of dipole sums to dipole'''
        radii = [88000, 90000, 95000, 100000]
        sigmas = [0.3, 1.5, 0.015, 0.3]

        dips = np.array([[[1000., 0., 0.]],
                         [[-1000., 0., 0.]],
                         [[0., 1000., 0.]],
                         [[0., -1000., 0.]],
                         [[0., 0., 1000.]],
                         [[0., 0., -1000.]]])

        p_locs = np.array([[87000., 0., 0.],
                           [-87000., 0., 0.],
                           [0., 87000., 0.],
                           [0., -87000., 0.],
                           [0., 0., 87000],
                           [0., 0., -87000]])

        el_locs = np.array([[[99000., 0., 0.]],
                            [[-99000., 0., 0.]],
                            [[0., 99000., 0.]],
                            [[0., -99000., 0.]],
                            [[0., 0., 99000.]],
                            [[0., 0., -99000.]]])

        for i in range(len(p_locs)):
            fs = LFPy.FourSphereVolumeConductor(
                el_locs[i], radii, sigmas)
            phi = fs.get_dipole_potential(dips[i].T, p_locs[i])

            M = fs.get_transformation_matrix(p_locs[i])
            np.testing.assert_allclose(M @ dips[i].T, phi)

    def test_get_dipole_potential_from_multi_dipoles00(self):
        """test comparison between multi-dipoles and single dipole approach"""
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend1.connect(soma(0.5), 0)
        dend2.connect(dend1(1.0), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        radii = [300, 400, 500, 600]
        sigmas = [0.3, 1.5, 0.015, 0.3]
        electrode_locs = np.array([[0., 0., 290.],
                                   [10., 90., 300.],
                                   [-90, 50., 400.],
                                   [110.3, -100., 500.]])
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        cell.set_pos(x=0, y=0, z=100)
        t_point = [1, 100, -1]

        MD_4s = LFPy.FourSphereVolumeConductor(
            electrode_locs, radii, sigmas)
        p, dipole_locs = cell.get_multi_current_dipole_moments(t_point)
        Np, Nt, Nd = p.shape
        Ne = electrode_locs.shape[0]
        pot_MD = MD_4s.get_dipole_potential_from_multi_dipoles(cell, t_point)

        pot_sum = np.zeros((Ne, Nt))
        for i in range(Np):
            dip = p[i]
            dip_loc = dipole_locs[i]
            fs = LFPy.FourSphereVolumeConductor(
                electrode_locs, radii, sigmas)
            pot = fs.get_dipole_potential(dip, dip_loc)
            pot_sum += pot

        np.testing.assert_almost_equal(pot_MD, pot_sum)
        np.testing.assert_allclose(pot_MD, pot_sum, rtol=1E-4)

    def test_get_dipole_potential_from_multi_dipoles01(self):
        """test comparison between multi-dipoles and single dipole approach"""
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
        radii = [300, 400, 500, 600]
        sigmas = [0.3, 1.5, 0.015, 0.3]
        electrode_locs = np.array([[0., 0., 290.],
                                   [10., 90., 300.],
                                   [-90, 50., 400.],
                                   [110.3, -100., 500.]])
        cell = cell_w_synapse_from_sections(morphology,
                                            rec_current_dipole_moment=True)
        cell.set_pos(x=0, y=0, z=100)
        t_point = [1, 100, -1]

        MD_4s = LFPy.FourSphereVolumeConductor(
            electrode_locs, radii, sigmas)
        p, dipole_locs = cell.get_multi_current_dipole_moments(t_point)
        Np, Nt, Nd = p.shape
        Ne = electrode_locs.shape[0]
        pot_MD = MD_4s.get_dipole_potential_from_multi_dipoles(cell, t_point)

        pot_sum = np.zeros((Ne, Nt))
        for i in range(Np):
            dip = p[i]
            dip_loc = dipole_locs[i]
            fs = LFPy.FourSphereVolumeConductor(
                electrode_locs, radii, sigmas)
            pot = fs.get_dipole_potential(dip, dip_loc)
            pot_sum += pot

        np.testing.assert_almost_equal(pot_MD, pot_sum)
        np.testing.assert_allclose(pot_MD, pot_sum, rtol=1E-4)


class testInfiniteVolumeConductor(unittest.TestCase):
    """
    test class InfiniteVolumeConductor
    """

    def test_get_dipole_potential_00(self):
        sigma = 0.3
        r = np.array([[0., 0., 1.], [0., 1., 0.]])
        p = np.array([[0., 0., 4 * np.pi * 0.3], [0., 4 * np.pi * 0.3, 0.]]).T
        inf_model = LFPy.InfiniteVolumeConductor(sigma)
        phi = inf_model.get_dipole_potential(p, r)
        np.testing.assert_allclose(phi, np.array([[1., 0.], [0., 1.]]))

    def test_get_transformation_matrix_00(self):
        sigma = 0.3
        r = np.array([[0., 0., 1.], [0., 1., 0.]])
        p = np.array([[0., 0., 4 * np.pi * 0.3], [0., 4 * np.pi * 0.3, 0.]]).T
        inf_model = LFPy.InfiniteVolumeConductor(sigma)

        M = inf_model.get_transformation_matrix(r)
        phi = M @ p
        np.testing.assert_allclose(phi, np.array([[1., 0.], [0., 1.]]))

    def test_get_multi_dipole_potential00(self):
        neuron.h('forall delete_section()')
        soma = neuron.h.Section(name='soma')
        dend1 = neuron.h.Section(name='dend1')
        dend2 = neuron.h.Section(name='dend2')
        dend3 = neuron.h.Section(name='dend3')
        dend4 = neuron.h.Section(name='dend4')
        dend5 = neuron.h.Section(name='dend5')
        dend1.connect(soma(0.5), 0)
        dend2.connect(dend1(1.0), 0)
        dend3.connect(dend2(1.0), 0)
        dend4.connect(dend3(1.0), 0)
        dend5.connect(dend4(1.0), 0)
        morphology = neuron.h.SectionList()
        morphology.wholetree()
        electrode_locs = np.array([[0., 0., 10000.]])
        cell, electrode, cdm = cell_w_synapse_from_sections_w_electrode(
            morphology, electrode_locs)
        sigma = 0.3

        MD_inf = LFPy.InfiniteVolumeConductor(sigma)
        pot_MD = MD_inf.get_multi_dipole_potential(cell, electrode_locs)
        pot_cb = electrode.data

        np.testing.assert_almost_equal(pot_MD, pot_cb)
        np.testing.assert_allclose(pot_MD, pot_cb, rtol=1E-4)

    def test_get_multi_dipole_potential01(self):
        morphology = os.path.join(
            LFPy.__path__[0], 'test', 'ball_and_sticks.hoc')
        electrode_locs = np.array([[0., 0., 10000.]])
        cell, electrode, cdm = cell_w_synapse_from_sections_w_electrode(
            morphology, electrode_locs)
        sigma = 0.3

        MD_inf = LFPy.InfiniteVolumeConductor(sigma)
        pot_MD = MD_inf.get_multi_dipole_potential(cell, electrode_locs)
        pot_cb = electrode.data

        np.testing.assert_almost_equal(pot_MD, pot_cb)
        np.testing.assert_allclose(pot_MD, pot_cb, rtol=1E-3)

    def test_get_multi_dipole_potential02(self):
        morphology = os.path.join(
            LFPy.__path__[0], 'test', 'ball_and_sticks.hoc')
        electrode_locs = np.array([[0., 0., 10000.]])
        cell, electrode, cdm = cell_w_synapse_from_sections_w_electrode(
            morphology, electrode_locs)
        sigma = 0.3
        t_point = [10, 100, 1000]

        MD_inf = LFPy.InfiniteVolumeConductor(sigma)
        pot_MD = MD_inf.get_multi_dipole_potential(
            cell, electrode_locs, t_point)
        pot_cb = electrode.data[:, t_point]

        np.testing.assert_almost_equal(pot_MD, pot_cb)
        np.testing.assert_allclose(pot_MD, pot_cb, rtol=1E-3)


class testOneSphereVolumeConductor(unittest.TestCase):
    """
    test class lfpykit.OneSphereVolumeConductor
    """

    def test_OneSphereVolumeConductor_00(self):
        """test case where sigma_i == sigma_o which
        should be identical to the standard point-source potential in
        infinite homogeneous media
        """
        # current magnitude
        current = 1.
        # conductivity
        sigma = 0.3
        # sphere radius
        R = 1000
        # source location (along x-axis)
        rs = 800
        # sphere coordinates of observation points
        radius = np.r_[np.arange(0, rs), np.arange(rs + 1, rs * 2)]
        theta = np.zeros(radius.shape)
        phi = np.zeros(radius.shape)
        r = np.array([radius, theta, phi])

        # predict potential
        sphere = LFPy.OneSphereVolumeConductor(
            cell=None,
            r=r, R=R, sigma_i=sigma, sigma_o=sigma)
        phi = sphere.calc_potential(rs=rs, current=current)

        # ground truth
        phi_gt = current / (4 * np.pi * sigma * abs(radius - rs))

        # test
        np.testing.assert_almost_equal(phi, phi_gt)

    def test_OneSphereVolumeConductor_01(self):
        """test case where sigma_i == sigma_o which
        should be identical to the standard point-source potential in
        infinite homogeneous media
        """
        # current magnitude
        current = np.ones(10)
        # conductivity
        sigma = 0.3
        # sphere radius
        R = 1000
        # source location (along x-axis)
        rs = 800
        # sphere coordinates of observation points
        radius = np.r_[np.arange(0, rs), np.arange(rs + 1, rs * 2)]
        theta = np.zeros(radius.shape)
        phi = np.zeros(radius.shape)
        r = np.array([radius, theta, phi])

        # predict potential
        sphere = LFPy.OneSphereVolumeConductor(
            cell=None,
            r=r, R=R, sigma_i=sigma, sigma_o=sigma)
        phi = sphere.calc_potential(rs=rs, current=current)

        # ground truth
        phi_gt = current[0] / (4 * np.pi * sigma * abs(radius - rs))

        # test
        np.testing.assert_almost_equal(phi,
                                       np.array([phi_gt] * current.size).T)

    def test_OneSphereVolumeConductor_02(self):
        """test case where sigma_i == sigma_o which
        should be identical to the standard point-source potential in
        infinite homogeneous media
        """
        # current magnitude
        current = 1.
        # conductivity
        sigma = 0.3
        # sphere radius
        R = 10000
        # cell body position
        xs = 8000.
        # sphere coordinates of observation points
        radius = np.r_[np.arange(0, xs), np.arange(xs + 1, xs * 2)][::10]
        theta = np.zeros(radius.shape) + np.pi / 2
        phi = np.zeros(radius.shape)
        r = np.array([radius, theta, phi])
        # set up cell
        cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
                                                 'stick.hoc'))
        cell.set_pos(x=xs, y=0, z=0)
        cell.set_rotation(y=np.pi / 2)

        # predict potential
        sphere = LFPy.OneSphereVolumeConductor(cell=cell, r=r, R=R,
                                               sigma_i=sigma, sigma_o=sigma)
        M = sphere.get_transformation_matrix(n_max=100)

        # ground truth and tests
        for i, x in enumerate(cell.x.mean(axis=-1)):
            dist = radius - x
            dist[abs(dist) < cell.d[i]] = cell.d[i]
            phi_gt = current / (4 * np.pi * sigma * abs(dist))
            np.testing.assert_almost_equal(M[:, i], phi_gt)


# Functions used by tests: ###################################################
def make_class_object(rz, r_el):
    '''Return class object fs'''
    radii = [79., 80., 85., 90.]
    sigmas = [0.3, 0.015, 15, 0.3]
    fs = LFPy.FourSphereVolumeConductor(r_el, radii, sigmas)
    fs._rz_params(rz)
    return fs


def make_simple_class_object():
    '''Return class object fs'''
    radii = [1., 2., 4., 8.]
    sigmas = [1., 2., 4., 8.]
    rz1 = np.array([0., 0., .9])
    r_el = np.array([[0., 0., 1.5]])
    fs = LFPy.FourSphereVolumeConductor(r_el, radii, sigmas)
    fs._rz_params(rz1)
    return fs


def decompose_dipole(P1):
    '''Return decomposed current dipole'''
    rz1 = np.array([0., 0., 70.])
    r_el = np.array([[0., 0., 90.]])
    fs = make_class_object(rz1, r_el)
    p_rad, p_tan = fs._decompose_dipole(P1)
    return p_rad, p_tan


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
        probes = [LFPy.CurrentDipoleMoment(cell)]
    else:
        probes = []
    cell.simulate(probes=probes, rec_vmem=True)
    if rec_current_dipole_moment:
        cell.current_dipole_moment = probes[0].data
    return cell


def cell_w_synapse_from_sections_w_electrode(morphology, electrode_locs):
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

    electrodeParams = {'sigma': 0.3,
                       'x': electrode_locs[:, 0],
                       'y': electrode_locs[:, 1],
                       'z': electrode_locs[:, 2],
                       }
    cell = LFPy.Cell(**cellParams)
    electrode = LFPy.RecExtElectrode(cell, **electrodeParams)
    cdm = LFPy.CurrentDipoleMoment(cell)

    synapse_parameters = {'e': 0.,
                          'syntype': 'ExpSyn',
                          'tau': 5.,
                          'weight': .1,
                          'record_current': True,
                          'idx': cell.totnsegs - 1}

    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))
    cell.simulate(probes=[electrode, cdm], rec_vmem=True)
    return cell, electrode, cdm
