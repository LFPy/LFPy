#!/usr/bin/env python
import unittest
import numpy as np
from scipy.integrate import quad
from scipy import real, imag
import LFPy

class testLFPy(unittest.TestCase):
    '''
    A set of test functions for each method of calculating the LFP, where the
    model outcome from LFPy is compared with analytically obtained results for
    a stick neuron with sinusoid synaptic current input at the end, and LFP
    calculated alongside the neuron.
    
    The tests should pass with 3 significant numbers in the LFP, as effects
    of compartmentalising the stick is prominent.
    '''
    
    def test_method_pointsource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulation(method='pointsource')
        
        #create LFPs using the analytical approac
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in xrange(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in xrange(a):
            for j in xrange(b):
                self.failUnlessAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)

    def test_method_linesource(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulation(method='linesource')
        
        #create LFPs using the analytical approac
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in xrange(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in xrange(a):
            for j in xrange(b):
                self.failUnlessAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)
    
    def test_method_som_as_point(self):
        #create LFPs using LFPy-model
        LFP_LFPy = self.stickSimulation(method='som_as_point')
        
        #create LFPs using the analytical approac
        time = np.linspace(0, 100, 10001)
        R = np.ones(11)*100
        Z = np.linspace(1000, 0, 11)
        
        LFP_analytic = np.empty((R.size, time.size))
        for i in xrange(R.size):
            LFP_analytic[i, ] = self.analytical_LFP(time, electrodeR=R[i],
                                                    electrodeZ=Z[i])
        (a, b) = LFP_LFPy.shape
        for i in xrange(a):
            for j in xrange(b):
                self.failUnlessAlmostEqual(LFP_LFPy[i, j], LFP_analytic[i, j],
                                           places=3)

            
    def stickSimulation(self, method):
        stickParams = {
            'morphology' : 'stick.hoc',
            'rm' : 30000,
            'cm' : 1,
            'Ra' : 150,
            'tstartms' : -100,
            'tstopms' : 100,
            'timeres_python' : 0.01,
            'timeres_NEURON' : 0.01,
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
        
        
        electrode = LFPy.RecExtElectrode(**electrodeParams)
        
        stick = LFPy.Cell(**stickParams)
        
        synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                               **stimParams)
        stick.simulate(electrode, rec_imem=True, rec_istim=True, rec_vmem=True)
        
        return electrode.LFP

    
    def analytical_LFP(self, time=np.linspace(0, 100, 1001),
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
        '''
        Will calculate the analytical LFP from a dendrite stick aligned with z-axis.
        The synaptic current is always assumed to be at the end of the stick, i.e.
        Zin = stickLength.
        
        Arguments:
        ::
            
            time : The LFP is calculated for values in this np.array (ms)
            stickLength : length of stick (mum)
            stickDiam : diameter of stick (mum)
            Rm : Membrane resistivity (Ohm * cm2)
            Cm : Membrane capacitance (muF/cm2)
            Ri : Intracellular resistivity (Ohm*cm)
            stimFrequency : Frequency of cosine synapse current (Hz)
            stimAmplitude : Amplitude of cosine synapse current (nA)
            sigma : Extracellular conductivity (muS/mum)
            electrodeR : Radial distance from stick (mum)
            electrodeZ : Longitudal distance along stick(mum)
        '''    
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
        Vex_imem = -self.complex_quadrature(f_to_integrate, 0, L, epsabs=1E-20)
        
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
    
    def complex_quadrature(self, func, a, b, **kwargs):
        '''
        Will return the complex integral value.
        '''
        def real_func(x):
            return real(func(x))
        def imag_func(x):
            return imag(func(x))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return real_integral[0] + 1j*imag_integral[0]


suite = unittest.TestLoader().loadTestsFromTestCase(testLFPy)
unittest.TextTestRunner(verbosity=2).run(suite)

#if __name__ == '__main__':
#    unittest.main()