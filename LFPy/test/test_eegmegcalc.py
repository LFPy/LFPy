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
import unittest
import numpy as np
import LFPy


class testMEG(unittest.TestCase):
    """
    test class LFPy.MEG
    """

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
        
        
class testFourSphereVolumeConductor(unittest.TestCase):
    """
    test class LFPy.FourSphereVolumeConductor
    """

    def test_decompose_dipole(self):
        '''Test radial and tangential parts of dipole sums to dipole'''
        P1 = np.array([[1., 1., 1.]])
        p_rad, p_tan = decompose_dipole(P1)
        np.testing.assert_equal(p_rad + p_tan, P1)

    def test_rad_dipole(self):
        '''Test that radial part of decomposed dipole is correct'''
        P1 = np.array([[1., 1., 1.]])
        p_rad, p_tan = decompose_dipole(P1)
        np.testing.assert_equal(p_rad, np.array([[0., 0., 1.]]))

    def test_tan_dipole(self):
        '''Test that tangential part of decomposed dipole is correct'''
        P1 = np.array([[1., 1., 1.]])
        p_rad, p_tan = decompose_dipole(P1)
        np.testing.assert_equal(p_tan, np.array([[1., 1., 0.]]))

    def test_calc_theta(self):
        '''Test theta: angle between rz and r'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 0., 90.], [0., 0., -90.],[0., 70., 0.], [0., -70., 0.], [0., 10., 10.]])
        fs = make_class_object(rz1, r_el)
        theta = fs.calc_theta()
        np.testing.assert_almost_equal(theta, np.array([0., np.pi, np.pi/2, np.pi/2, np.pi/4]))

    def test_calc_phi(self):
        '''Test phi: azimuthal angle between rx and rxy'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 1., 0], [-1., -1., 1.],[1., 1., 4.]])
        fs = make_class_object(rz1, r_el)
        P1 = np.array([[0., 1., 0.], [1., 0., 1.]])
        phi = fs.calc_phi(P1)
        np.testing.assert_almost_equal(phi, np.array([[np.pi/2, np.pi], [5*np.pi/4, 7*np.pi/4], [np.pi/4, 3*np.pi/4]]))

    def test_rad_sign(self):
        '''Test if radial dipole points inwards or outwards'''
        rz1 = np.array([0., 0., 70.])
        r_el = np.array([[0., 0., 90.]])
        fs = make_class_object(rz1, r_el)
        P1 = np.array([[0., 0., 1.], [0., 0., -2.]])
        s_vector = fs._sign_rad_dipole(P1)
        np.testing.assert_almost_equal(s_vector, np.array([1., -1.]))


class testInfiniteVolumeConductor(unittest.TestCase):
    """
    test class InfiniteVolumeConductor
    """
    def test_get_dipole_potential(self):
        sigma = 0.3
        r = np.array([[0., 0., 1.], [0., 1., 0.]])
        p = np.array([[0., 0., 4*np.pi*0.3], [0., 4*np.pi*0.3, 0.]])
        inf_model = LFPy.InfiniteVolumeConductor(sigma)
        phi = inf_model.get_dipole_potential(p, r)
        np.testing.assert_allclose(phi, np.array([[1., 0.], [0., 1.]]))


######## Functions used by tests: ##############################################

def make_class_object(rz1, r_el):
    '''Return class object fs'''
    radii = [79., 80., 85., 90.]
    sigmas = [0.3, 0.015, 15, 0.3]
    fs = LFPy.FourSphereVolumeConductor(radii, sigmas, r_el, rz1)
    return fs

def decompose_dipole(P1):
    '''Return decomposed current dipole'''
    rz1 = np.array([0., 0., 70.])
    r_el = np.array([[0., 0., 90.]])
    fs = make_class_object(rz1, r_el)
    p_rad, p_tan = fs._decompose_dipole(P1)
    return p_rad, p_tan

