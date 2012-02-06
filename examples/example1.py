import LFPy
import numpy as np
import matplotlib.pyplot as plt

# Example plot for LFPy: Single-synapse contribution to the LFP

# Define cell parameters

cell_parameters = {          # various cell parameters,
    'morphology' : 'morphologies/L5_Mainen96_LFPy.hoc', # Mainen&Sejnowski, Nature, 1996
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
    'tstartms' : -100., # start time of simulation, recorders start at t=0
    'tstopms' : 200.,   # stop simulation at 200 ms. These can be overridden
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
synapse.set_spike_times(np.array([50.]))

# Define electrode parameters

# But first create a grid of measurement locations

x = np.arange(-500, 501, 50) #last parameter sets grid spacing
z = np.arange(-400, 1201, 50)
X, Z = np.meshgrid(x, z)
Y = np.zeros(np.shape(X))

electrode_parameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()
}

# Create electrode

electrode = LFPy.RecExtElectrode(**electrode_parameters)

# Run simulation

print "running simulation..."
cell.simulate(electrode=electrode, rec_isyn=True)
print "done"

# Plotting #####################################################################

# function for plotting the morphology of the cell 

def plot_morphology(plot_synapses=False, color='k'):
    for i in xrange(cell.xend.size):     
        if len(color) > 1:
            col=str(color[i])
        else:
            col = color
        plt.plot([cell.xstart[i], cell.xend[i]],
            [cell.zstart[i], cell.zend[i]], color=col)
    if plot_synapses:
        for i in xrange(len(cell.synapses)):
            plt.plot([cell.synapses[i].x], [cell.synapses[i].z],
                color=cell.synapses[i].color, marker=cell.synapses[i].marker, 
                markersize=10)

# Create figure

# plt.ion() # Turn on interactive plotting in matplotlib (optional)

plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
    wspace=0.4, hspace=0.4)

# Plot LFP around the cell with in color and with equipotential lines

plt.subplot(121, aspect='equal')

t_show = 60 #time point to show LFP
tidx = np.where(cell.tvec == t_show)

plot_morphology(plot_synapses=True)
n_contours = 200
n_contours_black = 40

LFP = np.reshape(np.arcsinh(electrode.LFP[:, tidx]), (len(z), len(x)))
plt.contourf(X, Z, LFP, n_contours)
plt.clim((-0.00007, 0.00002))
plt.contour(X, Z, LFP, n_contours_black, colors='k')

# Plot synaptic input current

plt.subplot(222)
plt.plot(cell.tvec, cell.synapses[0].i)

# Plot soma potential

plt.subplot(224)
plt.plot(cell.tvec, cell.somav)

# Figure formatting

plt.suptitle('example 1', fontsize=14)

plt.subplot(121, aspect='equal')
plt.axis('off')
plt.title('LFP at t=' + str(t_show) + ' ms', fontsize=12)

plt.subplot(222)
plt.title('synaptic input current', fontsize=12)
plt.ylabel('(nA)')
plt.xlabel('time (ms)')

plt.subplot(224)
plt.title('somatic membrane potential', fontsize=12)
plt.ylabel('(mV)')
plt.xlabel('time (ms)')

plt.show()

# Optionally save figure (uncomment the line below)
# plt.savefig('example1.pdf')