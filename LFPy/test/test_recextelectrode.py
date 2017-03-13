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
        LFP_LFPy = stickSimulation(method='pointsource')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_linesource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulation(method='linesource')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_som_as_point(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulation(method='som_as_point')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    
    def test_method_pointsource_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationDotprodcoeffs(method='pointsource')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_linesource_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationDotprodcoeffs(method='linesource')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_som_as_point_dotprodcoeffs(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationDotprodcoeffs(method='som_as_point')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    
    def test_method_pointsource_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='som_as_point')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_linesource_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='linesource')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_som_as_point_contact_average_r10n100(self):
        #create LFPs using LFPy-model
        LFP_LFPy = stickSimulationAveragingElectrode(
            contactRadius=10, contactNPoints=100, method='som_as_point')
    
        #create LFPs using the analytical approach
        time = np.linspace(0, 100, 100*2**6+1)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
    
        LFP_analytic = np.empty((R.size, time.size))
        for i in range(R.size):
            LFP_analytic[i, ] = analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in range(a):
            for j in range(b):
                self.assertAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)



######## Functions used by tests: ##############################################


def stickSimulation(method):
    stickParams = {
        'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
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

def stickSimulationAveragingElectrode(contactRadius, contactNPoints, method):
    stickParams = {
        'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
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

def stickSimulationDotprodcoeffs(method):
    stickParams = {
        'morphology' : os.path.join(LFPy.__path__[0], 'test', 'stick.hoc'),
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
                   electrodeZ=0.):
    """
    Will calculate the analytical LFP from a dendrite stick aligned with z-axis.
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
    sigma : float
        Extracellular conductivity (muS/mum)
    electrodeR : float
        Radial distance from stick (mum)
    electrodeZ : float
        Longitudal distance along stick(mum)
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
    Vex_imem = -complex_quadrature(f_to_integrate, 0, L, epsabs=1E-20)

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
    return real_integral[0] + 1j*imag_integral[0]


