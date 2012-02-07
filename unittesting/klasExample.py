#!/usr/bin/env python
from numpy import pi, cos, tanh, cosh, sqrt, exp, arange, array, linspace, empty, zeros
from scipy.integrate import quad
from scipy import real, imag
from matplotlib.pyplot import plot, subplot, imshow, axis, show, xlabel, ylabel, title, close, colorbar, interactive

interactive(1)

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return real(func(x))
    def imag_func(x):
        return imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]#  , real_integral[1:], imag_integral[1:])



time = linspace(0, 100, 1001) #time vector (ms)

input_z_location = 1000     # input, end of stick (z = L)
electrode_x = 100    # x-coordinate of electrode (mum)
sigma = 0.3             # extracellular conductivity (muS/mum)
electrode_z = linspace(1000, 0, 21)


#current input
I0 = 1E-3               # input current amplitude (nA)
frequency = 100.        # input frequency (Hz)


# Electrotonic parameters
len_stick = 1000.        # length of stick (m)
diam = 2.             # diameter of stick (m)
Rm = 30000.                 # specific membrane resistivity (Ohm*cm2)
Gm = 1. / Rm            # specific membrane conductivity (S/cm2)
Cm = 1.               # specific membrane capacitance (muF/cm2)
Ri = 150                # intracellular resistance (Ohm*cm)

gm = 1E2 * pi * diam / Rm     # absolute membrane conductance (muS / mum)
ri = 1E-2 * 4. * Ri / (pi * diam**2) # intracellular resistance  (Mohm/mum)

Lambda = 1E2 / sqrt(gm * ri) # Electrotonic length constant of stick (mum)
Ginf = 10 / (ri * Lambda)   # iinfinite stick input cond (muS)?

tau_m = Rm * Cm / 1000        # membrane time constant (ms)
Omega = 2 * pi * frequency * tau_m / 1000 #impedance
#Zel = input_z_location / Lambda    # z-position of extracellular point, in units of Lambda
L = len_stick / Lambda      # Length of stick in units of Lambda
Rel = electrode_x / Lambda    # extracellular, location along x-axis, or radius, in units of Lambda
q = sqrt(1 + 1j*Omega)	    # Note: j is sqrt(-1)
Yin = q * Ginf * tanh(q * L)	    # Admittance, Zin is input position?
Zin = input_z_location / Lambda

V_extracellular = empty((electrode_z.size, time.size))
j = 0
for Zel in electrode_z / Lambda:
    print 'calculate Vex for Zel=%.3f' % Zel
    def i_mem(z): #z is location at stick
        return 1E-1 * gm * q**2 * cosh(q * L - q * z) / cosh(q * L) * I0 / Yin
        
        
    def f_to_integrate(z):
        #print z - Zel
        return Lambda / (4 * pi * sigma) * i_mem(z) / sqrt(Rel**2 + (z - Zel)**2)
    
    print 'int(i_mem(Zel), 0, L) = %6f' % complex_quadrature(i_mem, 0, L).real
    
    #calculate contrib from membrane currents
    Vex_imem = complex_quadrature(f_to_integrate, 0, L, epsabs=1E-20)
    
    #adding contrib from input current to Vex
    Vex_input = 1000*I0 / (4 * pi * sigma * Lambda * sqrt(Rel**2 + (Zin-Zel)**2))
    
    Vex = Vex_imem + Vex_input
    #Vex = Vex_input
    #Vex = Vex_imem
    Vcomplex = []
    for i in xrange(time.size):
        Vcomplex.append(Vex * exp(1j*2*pi*frequency * time[i] / 1000))
    
    Vcomplex = array(Vcomplex)

    V_extracellular[j, ] = Vcomplex.real
    j += 1


close('all')

subplot(221)
plot(time, I0 * cos(2 * pi * frequency * time / 1000))

subplot(222)
plot([0, 0], [0, 1000], 'k', lw=2)
plot(0, 1000, '.', marker='o', color='r')
plot(zeros(21) + electrode_x, electrode_z, '.', color='b', marker='o')
axis('equal')

subplot(212)
imshow(V_extracellular, cmap='jet_r', interpolation='nearest')
axis('tight')
colorbar()


