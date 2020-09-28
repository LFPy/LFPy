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
import numpy as np
from scipy.integrate import quad
from numpy import real, imag
import LFPy


# ####### Functions used by tests: ########################################
def stickSimulation(method):
    stickParams = {
        'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
        'cm': 1,
        'Ra': 150,
        'v_init': -65,
        'passive': True,
        'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
        'tstart': -100,
        'tstop': 100,
        'dt': 2**-6,
        'nsegs_method': 'lambda_f',
        'lambda_f': 1000,
    }

    electrodeParams = {
        'sigma': 0.3,
        'x': np.ones(11) * 100.,
        'y': np.zeros(11),
        'z': np.linspace(1000, 0, 11),
        'method': method
    }

    stimParams = {
        'pptype': 'SinSyn',
        'delay': -100.,
        'dur': 1000.,
        'pkamp': 1.,
        'freq': 100.,
        'phase': -np.pi / 2,
        'bias': 0.,
        'record_current': True
    }

    stick = LFPy.Cell(**stickParams)
    stick.set_pos(z=-stick.z[0, 0])

    LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                          **stimParams)

    electrode = LFPy.RecExtElectrode(stick, **electrodeParams)

    stick.simulate(probes=[electrode])

    return electrode.data


def stickSimulationAveragingElectrode(contactRadius, contactNPoints, method):
    stickParams = {
        'morphology': os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
        'cm': 1,
        'Ra': 150,
        'v_init': -65,
        'passive': True,
        'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
        'tstart': -100,
        'tstop': 100,
        'dt': 2**-6,
        'nsegs_method': 'lambda_f',
        'lambda_f': 1000,
    }

    N = np.empty((11, 3))
    for i in range(N.shape[0]):
        N[i, ] = [1, 0, 0]  # normal unit vec. to contacts
    electrodeParams = {
        'sigma': 0.3,
        'x': np.ones(11) * 100.,
        'y': np.zeros(11),
        'z': np.linspace(1000, 0, 11),
        'r': contactRadius,
        'n': 10,
        'N': N,
        'method': method
    }

    stimParams = {
        'pptype': 'SinSyn',
        'delay': -100.,
        'dur': 1000.,
        'pkamp': 1.,
        'freq': 100.,
        'phase': -np.pi / 2,
        'bias': 0.,
        'record_current': True
    }

    stick = LFPy.Cell(**stickParams)
    stick.set_pos(z=-stick.z[0, 0])

    LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                          **stimParams)

    electrode = LFPy.RecExtElectrode(stick, **electrodeParams)

    stick.simulate(probes=[electrode])

    return electrode.data


'''def stickSimulationDotprodcoeffs(method):
    stickParams = {
        'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
        'cm' : 1,
        'Ra' : 150,
        'v_init' : -65,
        'passive' : True,
        'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
        'tstart' : -100,
        'tstop' : 100,
        'dt' : 2**-6,
        'nsegs_method' : 'lambda_f',
        'lambda_f' : 1000,
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
    stick.set_pos(z=-stick.zstart[0])

    #dummy variables for mapping
    stick.imem = np.eye(stick.totnsegs)
    stick.tvec = np.arange(stick.totnsegs)*stick.dt

    electrode = LFPy.RecExtElectrode(stick, **electrodeParams)
    electrode.calc_lfp()
    # not needed anymore:
    del stick.imem, stick.tvec

    synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                           **stimParams)
    stick.simulate(dotprodcoeffs=electrode.LFP,
                   rec_imem=True, rec_vmem=True)

    return stick.dotprodresults[0]'''


def analytical_LFP(time=np.linspace(0, 100, 1001),
                   stickLength=1000.,
                   stickDiam=2.,
                   Rm=30000.,
                   Cm=1.,
                   Ri=150.,
                   stimFrequency=100.,
                   stimAmplitude=1.,
                   sigma=0.3,
                   electrodeR=100.,
                   electrodeZ=0.,
                   ):
    """
    Will calculate the analytical LFP from a dendrite stick aligned
    with z-axis.
    The synaptic current is always assumed to be at the end of the stick, i.e.
    Zin = stickLength.

    Parameters
    ----------
    time : ndarray
        The LFP is calculated for values in this np.array (ms)
    stickLength : float
        length of stick (mum)
    stickDiam : float
        diameter of stick (mum)
    Rm : float
        Membrane resistivity (Ohm * cm2)
    Cm : float
        Membrane capacitance (muF/cm2)
    Ri : float
        Intracellular resistivity (Ohm*cm)
    stimFrequency : float
        Frequency of cosine synapse current (Hz)
    stimAmplitude : float
        Amplitude of cosine synapse current (nA)
    # stimPos : float in [0, 1]
    #     Relative stimulus current position from start (0) to end (1) of stick
    sigma : float
        Extracellular conductivity (muS/mum)
    electrodeR : float
        Radial distance from stick (mum)
    electrodeZ : float
        Longitudal distance along stick(mum)
    """
    # absolute membrane conductance (muS / mum)
    gm = 1E2 * np.pi * stickDiam / Rm
    # intracellular resistance  (Mohm/mum)
    ri = 1E-2 * 4. * Ri / (np.pi * stickDiam**2)

    # Electrotonic length constant of stick (mum)
    Lambda = 1E2 / np.sqrt(gm * ri)
    Ginf = 10 / (ri * Lambda)   # infinite stick input cond (10*muS)?

    tau_m = Rm * Cm / 1000        # membrane time constant (ms)
    Omega = 2 * np.pi * stimFrequency * tau_m / 1000  # impedance
    Zel = electrodeZ / Lambda    # relative z-position of extracellular point
    L = stickLength / Lambda      # Length of stick in units of Lambda
    # extracellular, location along x-axis, or radius, in units of Lambda
    Rel = electrodeR / Lambda
    q = np.sqrt(1 + 1j * Omega)	    # Note: j is sqrt(-1)
    Yin = q * Ginf * np.tanh(q * L)	    # Admittance
    Zin = stickLength / Lambda  # unitless location of input current
    # Zin = stickLength / Lambda * stimPos  # unitless location of input
    # current

    PhiExImem = np.empty(time.size)
    PhiExInput = np.empty(time.size)

    def i_mem(z):  # z is location at stick
        return gm * q**2 * np.cosh(q * z) / \
            np.cosh(q * L) * stimAmplitude / Yin

    def f_to_integrate(z):
        return 1E-3 / (4 * np.pi * sigma) * i_mem(z) \
            / np.sqrt(Rel**2 + (z - Zel)**2)

    # calculate contrib from membrane currents
    Vex_imem = -complex_quadrature(f_to_integrate, 0, L, epsabs=1E-20)

    # adding contrib from input current to Vex
    Vex_input = stimAmplitude / \
        (4 * np.pi * sigma * Lambda * np.sqrt(Rel**2 + (Zin - Zel)**2))

    PhiExImemComplex = Vex_imem * np.exp(1j * 2 * np.pi * stimFrequency *
                                         time / 1000)
    PhiExInputComplex = Vex_input * np.exp(1j * 2 * np.pi * stimFrequency *
                                           time / 1000)

    # Using only real component
    PhiExImem = PhiExImemComplex.real
    PhiExInput = PhiExInputComplex.real

    PhiEx = PhiExImem + PhiExInput
    return PhiEx


def complex_quadrature(func, a, b, **kwargs):
    """
    Will return the complex integral value.
    """
    def real_func(x):
        return real(func(x))

    def imag_func(x):
        return imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]
