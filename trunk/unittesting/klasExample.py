#!/usr/bin/env python
#from numpy import pi, cos, tanh, cosh, sqrt, exp, arange, array, linspace, empty, zeros
import numpy as np
from scipy.integrate import quad
from scipy import real, imag
import matplotlib.pyplot as pl

pl.interactive(1)

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
    Vex_imem = -complex_quadrature(f_to_integrate, 0, L) #, epsabs=1E-20)
    
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
    
    return PhiEx, PhiExImem, PhiExInput

def complex_quadrature(func, a, b, **kwargs):
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




modelspec = {
    'stimAmplitude' : 1.,
    'stimFrequency' : 100.,
    'stickLength' : 1000.,
    'stickDiam' : 2.,
    'Rm' : 30000.,
    'Cm' : 1.,
    'Ri' : 150.,
    'sigma' : 0.3
}
time = np.linspace(0, 100, 1001)
R = np.ones(101)*100
Z = np.linspace(1000, 0, 101)


PhiEx = np.empty((R.size, time.size))
PhiExImem = np.empty((R.size, time.size))
PhiExInput = np.empty((R.size, time.size))
for i in xrange(R.size):
    PhiEx[i, ], PhiExImem[i, ],PhiExInput[i, ] = analytical_LFP(
              time = time, electrodeZ = Z[i], electrodeR = R[i], **modelspec)


pl.close('all')

pl.figure(figsize=(10, 10))
pl.subplot(421)
pl.plot(time, modelspec['stimAmplitude'] * np.exp(1j * 2 * np.pi * modelspec['stimFrequency'] *
                                              time / 1000).real)
pl.xlabel('time (ms)')
pl.ylabel('$i_\mathrm{electrode}$ (nA)')
pl.title('Synapse current')

pl.subplot(422)
pl.plot([0, 0], [0, 1000], 'k', lw=2)
pl.plot(0, modelspec['stickLength'], '.', marker='o', color='r')
pl.plot(R, Z, '.', color='b', marker='o')
pl.axis('equal')
pl.xlabel('x ($\mu$m)')
pl.ylabel('z ($\mu$m)')
pl.title('geometry')

pl.subplot(412)
pl.imshow(PhiEx, cmap='jet_r', interpolation='nearest')
pl.axis('tight')
cbar = pl.colorbar()
cbar.set_label('LFP (mV)')
pl.title('total LFP')

pl.subplot(413)
pl.imshow(PhiExInput, cmap='jet_r', interpolation='nearest')
pl.axis('tight')
cbar = pl.colorbar()
cbar.set_label('Stimulus LFP (mV)')
pl.title('Stimulus LFP')

pl.subplot(414)
pl.imshow(PhiEx - PhiExInput, cmap='jet_r', interpolation='nearest')
pl.axis('tight')
cbar = pl.colorbar()
cbar.set_label('residual LFP (mV)')
pl.title('Residual LFP')
