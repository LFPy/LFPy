#!/usr/bin/env python
'''
###############################################################################
#
# This is an example scripts using LFPy with a passive cell model adapted from
# Mainen and Sejnowski, Nature 1996, for the original files, see
# http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=2488
#
# Here, excitatory and inhibitory neurons are distributed on different parts of
# the morphology, with stochastic spike times produced by the
# LFPy.inputgenerators.stationary_gamma() function.
#
# Same as "example6.py", just without the active conductances
###############################################################################
'''

# importing some modules, setting some matplotlib values for pl.plot.
import matplotlib.pyplot as plt
import LFPy
import numpy as np
import matplotlib
matplotlib.use('Agg')
plt.rcParams.update({'font.size': 12,
                     'figure.facecolor': '1',
                     'figure.subplot.wspace': 0.5,
                     'figure.subplot.hspace': 0.5})

# seed for random generation
np.random.seed(1234)

##########################################################################
# A couple of function declarations
##########################################################################


def insert_synapses(synparams, section, n, spTimesFun, args):
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)

    # Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx': int(i)})

        # Some input spike train using the function call
        spiketimes = spTimesFun(args[0], args[1], args[2], args[3], args[4])

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(spiketimes)


def plot_nsg_example(cell, electrode):
    '''plotting function used by example3/4'''
    fig = plt.figure(figsize=[12, 8])

    # plot the somatic trace
    ax = fig.add_axes([0.1, 0.7, 0.5, 0.2])
    ax.plot(cell.tvec, cell.somav)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Soma pot. [mV]')

    # plot the synaptic current
    ax = fig.add_axes([0.1, 0.4, 0.5, 0.2])
    for i in range(len(cell.synapses)):
        ax.plot(cell.tvec, cell.synapses[i].i, color=cell.synapses[i].color)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Syn. i [nA]')

    # plot the LFP as image plot
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.2])
    absmaxLFP = abs(np.array([electrode.LFP.max(), electrode.LFP.min()])).max()
    im = ax.pcolormesh(cell.tvec, electrode.z, electrode.LFP,
                       vmax=absmaxLFP, vmin=-absmaxLFP,
                       cmap='spectral_r')

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.02
    cax = fig.add_axes(rect)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('LFP (mV)')
    ax.axis(ax.axis('tight'))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'z [$\mu$m]')

    # plot the morphology, electrode contacts and synapses
    ax = fig.add_axes([0.65, 0.1, 0.25, 0.8], frameon=False)
    for secname in cell.allsecnames:
        idx = cell.get_idx(secname)
        ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color='k')
    for i in range(len(cell.synapses)):
        ax.plot([cell.synapses[i].x], [cell.synapses[i].z],
                color=cell.synapses[i].color, marker=cell.synapses[i].marker)
    for i in range(electrode.x.size):
        ax.plot(electrode.x[i], electrode.z[i], color='g', marker='o')
    plt.axis('equal')
    plt.axis(np.array(plt.axis()) * 0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


##########################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
##########################################################################

# define cell parameters used as input to cell-class
cellParameters = {
    'morphology': 'L5_Mainen96_wAxon_LFPy.hoc',
    'rm': 30000,               # membrane resistance
    'cm': 1.0,                 # membrane capacitance
    'Ra': 150,                 # axial resistance
    'v_init': -65,             # initial crossmembrane potential
    'e_pas': -65,              # reversal potential passive mechs
    'passive': True,           # switch on passive mechs
    'nsegs_method': 'lambda_f',  # method for setting number of segments,
    'lambda_f': 100,           # segments are isopotential at this frequency
    'dt': 2**-4,               # dt of LFP and NEURON simulation.
    'tstart': -100,  # start time, recorders start at t=0
    'tstop': 200,  # stop time of simulation
}

# Synaptic parameters taken from Hendrickson et al 2011
# Excitatory synapse parameters:
synapseParameters_AMPA = {
    'e': 0,  # reversal potential
    'syntype': 'Exp2Syn',  # conductance based exponential synapse
    'tau1': 1.,  # Time constant, rise
    'tau2': 3.,  # Time constant, decay
    'weight': 0.005,  # Synaptic weight
    'color': 'r',  # for plt.plot
    'marker': '.',  # for plt.plot
    'record_current': True,  # record synaptic currents
}
# Excitatory synapse parameters
synapseParameters_NMDA = {
    'e': 0,
    'syntype': 'Exp2Syn',
    'tau1': 10.,
    'tau2': 30.,
    'weight': 0.005,
    'color': 'm',
    'marker': '.',
    'record_current': True,
}
# Inhibitory synapse parameters
synapseParameters_GABA_A = {
    'e': -80,
    'syntype': 'Exp2Syn',
    'tau1': 1.,
    'tau2': 12.,
    'weight': 0.005,
    'color': 'b',
    'marker': '.',
    'record_current': True
}
# where to insert, how many, and which input statistics
insert_synapses_AMPA_args = {
    'section': 'apic',
    'n': 100,
    'spTimesFun': LFPy.inputgenerators.stationary_gamma,
    'args': [cellParameters['tstart'], cellParameters['tstop'], 0.5, 40,
             cellParameters['tstart']]
}
insert_synapses_NMDA_args = {
    'section': ['dend', 'apic'],
    'n': 15,
    'spTimesFun': LFPy.inputgenerators.stationary_gamma,
    'args': [cellParameters['tstart'], cellParameters['tstop'], 2, 50,
             cellParameters['tstart']]
}
insert_synapses_GABA_A_args = {
    'section': 'dend',
    'n': 100,
    'spTimesFun': LFPy.inputgenerators.stationary_gamma,
    'args': [cellParameters['tstart'], cellParameters['tstop'], 0.5, 40,
             cellParameters['tstart']]
}

# Define electrode geometry corresponding to a laminar electrode, where contact
# points have a radius r, surface normal vectors N, and LFP calculated as the
# average LFP in n random points on each contact:
N = np.empty((16, 3))
for i in range(N.shape[0]):
    N[i, ] = [1, 0, 0]  # normal unit vec. to contacts
# put parameters in dictionary
electrodeParameters = {
    'sigma': 0.3,              # Extracellular potential
    'x': np.zeros(16) + 25,      # x,y,z-coordinates of electrode contacts
    'y': np.zeros(16),
    'z': np.linspace(-500, 1000, 16),
    'n': 20,
    'r': 10,
    'N': N,
}

# Parameters for the cell.simulate() call, recording membrane- and
# syn.-currents
simulationParameters = {
    'rec_imem': True,  # Record Membrane currents during simulation
    'rec_isyn': True,  # Record synaptic currents
}

##########################################################################
# Main simulation procedure
##########################################################################

# Initialize cell instance, using the LFPy.Cell class
cell = LFPy.Cell(**cellParameters)

# set cell alignment
cell.set_rotation(x=4.98919, y=-4.33261, z=0)
cell.set_pos = (0, 0, 0)

# Insert synapses using the function defined earlier
insert_synapses(synapseParameters_AMPA, **insert_synapses_AMPA_args)
insert_synapses(synapseParameters_NMDA, **insert_synapses_NMDA_args)
insert_synapses(synapseParameters_GABA_A, **insert_synapses_GABA_A_args)

# perform NEURON simulation, results saved as attributes in the cell instance
cell.simulate(**simulationParameters)

# Initialize electrode geometry, then calculate the LFP, using the
# LFPy.RecExtElectrode class. Note that now cell is given as input to electrode
# and created after the NEURON simulations are finished
electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
print('simulating LFPs....')
electrode.calc_lfp()
print('done')

# plotting some variables and geometry, saving output to .pdf.
fig = plot_nsg_example(cell, electrode)
fig.savefig('nsg_example.pdf', dpi=300)
