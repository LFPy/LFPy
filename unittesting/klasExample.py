#!/usr/bin/env python
from numpy import pi, cos, tanh, cosh, sqrt, exp, arange, array, linspace, empty, zeros
from scipy.integrate import quad
from scipy import real, imag
from matplotlib.pyplot import plot, subplot, imshow, axis, show, xlabel, ylabel, title, close, colorbar

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return real(func(x))
    def imag_func(x):
        return imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]#  , real_integral[1:], imag_integral[1:])



time = arange(0, 0.1, 0.0001) #time vector (s)

input_z_location = 1000E-6     # input, end of stick (z = L)
electrode_x = 100E-6    # x-coordinate of electrode (m)
sigma = 0.3             # extracellular conductivity (ohm/m)
electrode_z = linspace(1000E-6, 0, 101)


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
Omega = 2 * pi * frequency * tau_m  #impedance
#Z = input_z_location / Lambda    # z-position of extracellular point, in units of Lambda
L = len_stick / Lambda      # Length of stick in units of Lambda
R = electrode_x / Lambda    # extracellular, location along x-axis, or radius, in units of Lambda
q = sqrt(1+1j*Omega)	    # Note: j is sqrt(-1)
Yin = q*Ginf*tanh(q*L)	    # Zin is input position


V_extracellular = empty((electrode_z.size, time.size))
j = 0
for Z in electrode_z / Lambda:
    print 'calculate Vex for Z=%.3f' % Z
    def i_mem(Z):
        return gm * q**2*cosh(q * L - q * Z)/cosh(q * L) * I0 / Yin / 1000.
        
        
    def f_to_integrate(Z):
        return Lambda / (4 * pi * sigma) * i_mem(Z) / sqrt(R**2 + Z**2)
    
    #calculate contrib from membrane currents
    Vex_imem = complex_quadrature(f_to_integrate, 0, L, epsabs=1E-20)
    
    #adding contrib from input current to Vex
    Vex_input = I0 / (4 * pi * sigma * sqrt(R**2 + (Z-L)**2))
    
    Vex = Vex_imem + Vex_input
    Vcomplex = []
    for i in xrange(time.size):
        Vcomplex.append(Vex * exp(1j*2*pi*frequency*time[i]))
    
    Vcomplex = array(Vcomplex)

    V_extracellular[j, ] = Vcomplex.real
    j += 1

close('all')

subplot(221)
plot(time, I0 * cos(2 * pi *frequency * time))

subplot(222)
plot([0, 0], [0, 1E-3], 'k', lw=2)
plot(0, 1E-3, '.', marker='o', color='r')
plot(zeros(101) + electrode_x, electrode_z, '.', color='b', marker='o')
axis('equal')

subplot(212)
imshow(V_extracellular, cmap='jet_r', interpolation='nearest')
axis('tight')
colorbar()

show()
