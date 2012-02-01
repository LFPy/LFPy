import LFPy
import numpy as np
import matplotlib.pylab as plt

plt.ion()

# Define parameters

cell_params = {          #various cell parameters,
    'morphology' : 'morphologies/L5_Mainen96_LFPy.hoc', #Mainen&Sejnowski, Nature, 1996
    'rm' : 30000.,       #membrane resistance
    'cm' : 1.0,        #membrane capacitance
    'Ra' : 150.,         #axial resistance
    'v_init' : -65.,     #initial crossmembrane potential
    'e_pas' : -65.,      #reversal potential passive mechs
    'passive' : True,   #switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100.,
    'timeres_NEURON' : 2**-3,   #[ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2**-3,   #need binary representation
    'tstartms' : -100.,  #start time of simulation, recorders start at t=0
    'tstopms' : 200.,   #stop simulation at 1000 ms. these can be overridden
                        #by setting these arguments in cell.simulation()
}

# Create cell

cell = LFPy.Cell(**cell_params)

# Define synapse parameters

synapse_params = {
    'idx' : cell.get_closest_idx(x=0., y=0., z=800.),
    'e' : 0.,                                # reversal potential
    'syntype' : 'ExpSyn',                   # synapse type
    'tau' : 2.,                              # syn. time constant
    'weight' : 1.,                       # syn. weight
    'record_current' : True                 # syn. current record
}

synapse = LFPy.Synapse(cell,**synapse_params)
synapse.set_spike_times(cell, np.array([50.]))

cell.simulate(rec_imem=True)
