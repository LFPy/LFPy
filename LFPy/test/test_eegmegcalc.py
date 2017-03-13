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
neuron.load_mechanisms(os.path.join(LFPy.__path__[0], 'test'))

class testLFPy(unittest.TestCase):
    """

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


    def test_get_dipole_potential(self):
        sigma = 0.3
        r = np.array([[0., 0., 1.], [0., 1., 0.]])
        p = np.array([[0., 0., 4*np.pi*0.3], [0., 4*np.pi*0.3, 0.]])
        inf_model = LFPy.InfiniteVolumeConductor(sigma)
        phi = inf_model.get_dipole_potential(p, r)
        np.testing.assert_allclose(phi, np.array([[1., 0.], [0., 1.]]))

######## Functions used by tests: ##############################################
def cell_w_synapse_from_sections(sections=None):
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

