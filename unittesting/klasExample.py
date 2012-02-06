#!/usr/bin/env python
from numpy import pi, cos, tanh, cosh, sqrt, exp, arange, array, linspace, empty
from scipy.integrate import quad
from scipy import real, imag
from matplotlib.pyplot import plot, subplot, imshow, axis, show, xlabel, ylabel, title

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return real(func(x))
    def imag_func(x):
        return imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]#  , real_integral[1:], imag_integral[1:])



time = arange(0, 0.1, 0.0001) #time vector (s)

input_z_location = 1000E-6     # z-coordinate of electrode (m)
electrode_x = 1000E-6 #linspace(0, 1000E-6, 1001) #1000E-6      # x-coordinate of electrode (m)
sigma = 0.3             # extracellular conductivity (ohm/m)


#current input
I0 = 1E-6               # input current amplitude (A)
frequency = 100.        # input frequency (Hz)


# Electrotonic parameters
len_stick = 1E-3        # length of stick (m)
diam = 2E-6             # diameter of stick (m)
Rm = 3.                 # specific membrane resistivity (Ohm/m2)
Gm = 1. / Rm            # specific membrane conductivity (S)
Cm = 1E-2               # specific membrane capacitance (F/m2)
Ri = 1.5                # intracellular resistance (Ohm/m)

gm = pi * diam / Rm     # absolute membrane capacititance
ri = 4. * Ri / (pi * diam**2) # intracellular resistance

Lambda = 1. / sqrt(gm * ri) # Electrotonic length constant of stick
Ginf = 1. / (ri * Lambda)   # input conductance?
tau_m = Rm * Cm         # membrane time constant
Omega = 2 * pi * frequency * tau_m
Z = input_z_location / Lambda    # position of input current, end of stick
L = len_stick / Lambda      # Unitless length of stick
R = electrode_x / Lambda    # extracellular, location along x-axis
q = sqrt(1+1j*Omega)	    # Note: j is sqrt(-1)

Yin = q*Ginf*tanh(q*L)	    # Zin is input position


#V_extracellular = empty((electrode_x.size, time.size))
#j = 0
#for R in electrode_x / Lambda:
def i_mem(Z):
    return gm*q**2*cosh(q*L-q*Z)/cosh(q*L)*I0/Yin

def f_to_integrate(Z):
    return Lambda / (4 * pi * sigma) * i_mem(Z) / sqrt(R**2 + Z**2)


Vex = complex_quadrature(f_to_integrate, 0, L, epsabs=1.49e-20) + \
    I0 / (4 * pi * sigma * sqrt(electrode_x**2 + (input_z_location - L)**2))

Vcomplex = []
for i in xrange(time.size):
    Vcomplex.append(Vex * exp(1j*2*pi*frequency*time[i]))

Vcomplex = array(Vcomplex)

V_extracellular = Vcomplex.real
#j += 1

subplot(211)
plot(time, I0 * cos(2 * pi *frequency * time))

subplot(212)
plot(time, V_extracellular)
axis('tight')

show()
