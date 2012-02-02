#!/usr/bin/env python
from pylab import *
interactive(1)
from scipy.integrate import quad
import scipy

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]#  , real_integral[1:], imag_integral[1:])

frequency = 100.
time = arange(0, 2/frequency, 0.0001) #seconds
z =  1E-6
lat_dist = 1.

len_stick = 1E-3

I0 = 1E-6 #current

d = 2E-6
Rm = 3.
Gm = 1. / Rm
Cm = 1E-2
Ri = 1.5
sigma = 0.3 # extracellular conductivity

gm = pi * d / Rm
ri = 4.*Ri / (pi * d**2)

Lambda = 1/sqrt(gm * ri)
Ginf = 1/(ri * Lambda)
tau_m = Rm * Cm
Omega = 2 * pi * frequency * tau_m
Z = z / Lambda
L = len_stick / Lambda
R = lat_dist / Lambda # vertical, extracellular distance
q = sqrt(1+1j*Omega)	# Note: j is sqrt(-1)

Yin = q*Ginf*tanh(q*L)	# Xin is input position

#H = gm*q**2*cosh(q*L-q*Z)/cosh(q*L)

def i_mem(Z):
    return gm*q**2*cosh(q*L-q*Z)/cosh(q*L)*I0/Yin
#i_mem = gm*q**2*cosh(q*L-q*Z)/cosh(q*L)*I0*Yin # I0 is input current

def f_to_integrate(Z):
    return Lambda/(4*pi*sigma) * i_mem(Z) / sqrt(R**2+Z**2)
    
Vex = complex_quadrature(f_to_integrate, 0, L, epsabs=1.49e-20) + I0/(4*pi*sigma*sqrt(lat_dist**2 + (z-L)**2))

Vcomplex = np.empty(time.size)
for i in xrange(time.size):
    Vcomplex[i] = Vex * exp(1j*2*pi*frequency*time[i])

Vreal = Vcomplex.real

plot(time, Vreal)

show()
