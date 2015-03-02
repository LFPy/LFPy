#!/usr/bin/env python
'''
Example plot for LFPy: Single-synapse contribution to the LFP
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################

# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology' : 'morphologies/L5_Mainen96_LFPy.hoc', # Mainen&Sejnowski, 1996
    'rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150.,        # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : True,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100.,
    'timeres_NEURON' : 2.**-3,   # [ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2.**-3,   # need binary representation
    'tstartms' : 0.,    # start time of simulation, recorders start at t=0
    'tstopms' : 100.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments in cell.simulation()
}

# Create cell
cell = LFPy.Cell(**cell_parameters)

# Define synapse parameters
synapse_parameters = {
    'idx' : cell.get_closest_idx(x=0., y=0., z=900.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 10.,                # syn. time constant
    'weight' : .001,            # syn. weight
    'record_current' : True,
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([20.]))


# Create a grid of measurement locations, in (mum)
X, Z = np.mgrid[-500:501:20, -400:1201:40]
Y = np.zeros(X.shape)

# Define electrode parameters
electrode_parameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()
}

# Create electrode object
electrode = LFPy.RecExtElectrode(**electrode_parameters)

# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(electrode=electrode, rec_isyn=True)
print("done")

#create a plot
from example_suppl import plot_ex1
fig = plot_ex1(cell, electrode, X, Y, Z)
# Optionally save figure (uncomment the line below)
fig.savefig('example4.pdf', dpi=300)
plt.show()
