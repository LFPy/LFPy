#!/usr/bin/env python
'''
Example plot for LFPy: Single-synapse contribution to the LFP
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt

'''
Function declaration
'''
def plotstuff():
    '''
    plot the morphology and LFP contours, synaptic current and soma trace
    '''
    #some plot parameters
    t_show = 30 #time point to show LFP
    tidx = np.where(cell.tvec == t_show)
    #contour lines:
    n_contours = 200
    n_contours_black = 40
    
    #This is the extracellular potential, reshaped to the X, Z mesh
    LFP = np.arcsinh(electrode.LFP[:, tidx]).reshape(len(z), len(x))
    
    #figure object
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
        wspace=0.4, hspace=0.4)
    
    # Plot LFP around the cell with in color and with equipotential lines
    ax1 = fig.add_subplot(121, aspect='equal', frameon=False)
    
    #plot_morphology(plot_synapses=True)
    for sec in LFPy.cell.neuron.h.allsec():
        idx = cell.get_idx_section(sec.name())
        ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color='k')
    for i in xrange(len(cell.synapses)):
        ax1.plot([cell.synapses[i].x], [cell.synapses[i].z],
            color=cell.synapses[i].color, marker=cell.synapses[i].marker, 
            markersize=10)
    
    #contour lines
    ct1 = ax1.contourf(X, Z, LFP, n_contours)
    ct1.set_clim((-0.00007, 0.00002))
    ct2 = ax1.contour(X, Z, LFP, n_contours_black, colors='k')
    
    # Plot synaptic input current
    ax2 = fig.add_subplot(222)
    ax2.plot(cell.tvec, cell.synapses[0].i)
    
    # Plot soma potential
    ax3 = fig.add_subplot(224)
    ax3.plot(cell.tvec, cell.somav)
    
    # Figure formatting and labels
    fig.suptitle('example 1', fontsize=14)
    
    ax1.set_title('LFP at t=' + str(t_show) + ' ms', fontsize=12)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    
    ax2.set_title('synaptic input current', fontsize=12)
    ax2.set_ylabel('(nA)')
    ax2.set_xlabel('time (ms)')
    
    ax3.set_title('somatic membrane potential', fontsize=12)
    ax3.set_ylabel('(mV)')
    ax3.set_xlabel('time (ms)')
    
    return fig

if __name__ == '__main__':
    '''
    Main script, set parameters and create cell, synapse and electrode objects
    '''
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
    
    
    # Create a grid of measurement locations
    x = np.arange(-500, 501, 50) #last parameter sets grid spacing
    z = np.arange(-400, 1201, 50)
    X, Z = np.meshgrid(x, z)
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
    print "running simulation..."
    cell.simulate(electrode=electrode, rec_isyn=True)
    print "done"
    
    #create a plot
    fig = plotstuff()
    # Optionally save figure (uncomment the line below)
    # fig.savefig('example1.pdf')
    plt.show()
