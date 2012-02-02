import LFPy
import numpy as np
import matplotlib.pylab as plt

plt.ion()

# Define parameters

cell_parameters = {          #various cell parameters,
    'morphology' : 'morphologies/L5_Mainen96_LFPy.hoc', #Mainen&Sejnowski, Nature, 1996
#     'morphology' : 'morphologies/CablesIncluded.xml',
    'rm' : 30000.,       #membrane resistance
    'cm' : 1.0,        #membrane capacitance
    'Ra' : 150.,         #axial resistance
    'v_init' : -65.,     #initial crossmembrane potential
    'e_pas' : -65.,      #reversal potential passive mechs
    'passive' : True,   #switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100.,
    'timeres_NEURON' : 2.**-3,   #[ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2.**-3,   #need binary representation
    'tstartms' : -100.,  #start time of simulation, recorders start at t=0
    'tstopms' : 200.,   #stop simulation at 1000 ms. these can be overridden
                        #by setting these arguments in cell.simulation()
}

# Create cell

cell = LFPy.Cell(**cell_parameters)

# Define synapse parameters

synapse_parameters = {
    'idx' : cell.get_closest_idx(x=-50., y=0., z=900.),
    'e' : 0.,                                # reversal potential
    'syntype' : 'ExpSyn',                   # synapse type
    'tau' : 2.,                              # syn. time constant
    'weight' : .1,                       # syn. weight
    'record_current':True,
}

# Create synapse and input spike time

synapse = LFPy.Synapse(cell,**synapse_parameters)
synapse.set_spike_times(np.array([50.]))

# Define electrode parameters

# But first create a grid of measurement locations

x_elec = np.tile(np.arange(-400,401,40),10)
y_elec = np.zeros(len(x_elec))
z_elec = np.repeat(np.arange(-400,1201,1600./20),10)

electrode_parameters = {
    'sigma' : 0.3,   # extracellular conductivity
    'x' : x_elec,
    'y' : y_elec,
    'z' : z_elec
}

# Run simulation

cell.simulate(rec_imem=True, rec_isyn=True)

# Set up electrode and calculate extracellular potential

electrode = LFPy.RecExtElectrode(cell,**electrode_parameters)
electrode.calc_lfp()

# Find max(abs(LFP))

max_abs_LFP = np.max(np.abs(electrode.LFP),1)
sign_max_abs_LFP = np.zeros(len(max_abs_LFP))
for i in xrange(len(sign_max_abs_LFP)):
    sign_max_abs_LFP[i] = np.sign(electrode.LFP[i]\
        [np.where(np.abs(electrode.LFP[i])==max_abs_LFP[i])])

# Plot results
def plot_morphology():
    # Plot dendrites
    for i in xrange(cell.xend.size):     
        plt.plot([cell.xstart[i],cell.xend[i]],\
            [cell.zstart[i],cell.zend[i]],color='k')
    # Plot synapses
    for i in xrange(len(cell.synapses)):
        plt.plot([cell.synapses[i].x],[cell.synapses[i].z],\
            color=cell.synapses[i].color,marker=cell.synapses[i].marker,\
            markersize=10)

plt.figure()

plot_morphology()
n_contours = 20
X = np.reshape(x_elec,(10,21))
Y = np.reshape(z_elec,(10,21))
Z = np.reshape(sign_max_abs_LFP*max_abs_LFP,(10,21))
plt.contourf(X,Y,Z,n_contours)


# Figure formatting
plt.axis('equal')