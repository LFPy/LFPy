#!/usr/bin/env python
'''
LFPs from a population of cells relying on MPI
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import os
from os.path import join
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile
import LFPy
import neuron
from mpi4py import MPI

#initialize the MPI interface
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def stationary_poisson(nsyn,lambd,tstart,tstop):
    ''' Generates nsyn stationary possion processes with rate lambda between tstart and tstop'''
    interval_s = (tstop-tstart)*.001
    spiketimes = []
    for i in range(nsyn):
        spikecount = np.random.poisson(interval_s*lambd)
        spikevec = np.empty(spikecount)
        if spikecount==0:
            spiketimes.append(spikevec)
        else:
            spikevec = tstart + (tstop-tstart)*np.random.random(spikecount)
            spiketimes.append(np.sort(spikevec)) #sort them too!

    return spiketimes


#Fetch Mainen&Sejnowski 1996 model files
if not os.path.isfile(join('cells', 'cells', 'j4a.hoc')) and RANK==0:
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=2488&a=23&mime=application/zip')
    localFile = open('patdemo.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('patdemo.zip', 'r')
    myzip.extractall('.')
    myzip.close()

#resync MPI threads
COMM.Barrier()

# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology' : join('cells', 'cells', 'j4a.hoc'), # Mainen&Sejnowski, 1996
    'rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,         # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : True,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100.,
    'timeres_NEURON' : 2.**-3,   # [ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2.**-3,   # need binary representation
    'tstartms' :  0.,   # start time of simulation, recorders start at t=0
    'tstopms' : 300.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
}

# Define synapse parameters
synapse_parameters = {
    'idx' : 0, # to be set later
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 5.,                 # syn. time constant
    'weight' : .001,            # syn. weight
    'record_current' : True,
}

# Define electrode parameters
point_electrode_parameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : 0.,  # electrode requires 1d vector of positions
    'y' : 0.,
    'z' : 0.,
}


#number of units
n_cells = SIZE
cell_id = RANK

#set the numpy random seeds
global_seed = 1234
np.random.seed(global_seed)


#assign cell positions
x_cell_pos = np.linspace(-250., 250., n_cells)

z_rotation = np.random.permutation(np.arange(0., np.pi, np.pi / n_cells))

#synaptic spike times
n_pre_syn = 1000
pre_syn_sptimes = stationary_poisson(nsyn=n_pre_syn, lambd=5., tstart=0, tstop=300)

#re-seed the random number generator
cell_seed = global_seed + cell_id
np.random.seed(cell_seed)

# Create cell
cell = LFPy.Cell(**cell_parameters)

#Have to position and rotate the cells!
cell.set_rotation(x=4.99, y=-4.33, z=z_rotation[RANK])
cell.set_pos(xpos=x_cell_pos[RANK])

#assign spike times to different units
n_synapses = 100

# Create synapse and set time of synaptic input
pre_syn_pick = np.random.permutation(np.arange(n_pre_syn))[0:n_synapses]

for i_syn in range(n_synapses):
    syn_idx = int(cell.get_rand_idx_area_norm())
    synapse_parameters.update({'idx' : syn_idx})
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(pre_syn_sptimes[pre_syn_pick[i_syn]])

#run the cell simulation
cell.simulate(rec_imem=True,rec_isyn=True)

#set up the extracellular device
point_electrode = LFPy.RecExtElectrode(cell, **point_electrode_parameters)
point_electrode.calc_lfp()

if RANK==0:
    single_LFPs = [point_electrode.LFP[0]]
    for i_proc in range(1, SIZE):
        single_LFPs = np.r_['0,2', single_LFPs, COMM.recv(source=i_proc)]
else:
    COMM.send(point_electrode.LFP[0], dest=0)

# we can also use MPI to sum arrays directly:
summed_LFP = COMM.reduce(point_electrode.LFP[0])


if RANK==0:
    #assign color to each unit
    color_vec = [plt.cm.rainbow(int(x*256./n_cells)) for x in range(n_cells)]

    #figure
    fig = plt.figure(figsize=(12, 8))
    
    # Morphologies axes:
    plt.axes([.175, .0, .65, 1], aspect='equal')
    plt.axis('off')

    for i_cell in range(n_cells):
        cell = LFPy.Cell(join('cells', 'cells', 'j4a.hoc'),
                         nsegs_method='lambda_f',
                         lambda_f=5)
        cell.set_rotation(x=4.99, y=-4.33, z=z_rotation[i_cell])
        cell.set_pos(xpos=x_cell_pos[i_cell])

        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))
        linecol = LineCollection(zips,
                    edgecolor = 'none',
                    facecolor = color_vec[i_cell],
                    rasterized=False,
                    )            

        ax = plt.gca()
        ax.add_collection(linecol)
    
    axis = ax.axis(ax.axis('equal'))
    ax.axis(np.array(axis) / 1.15)


    #adding a blue dot:
    ax.plot(point_electrode.x, point_electrode.z, 'o',
            markeredgecolor='none', markerfacecolor='b', markersize=3,
            zorder=10, clip_on=False)
    plt.annotate("Electrode",
            xy=(0., 0.), xycoords='data',
            xytext=(-100., 1000.),
            arrowprops=dict(arrowstyle='wedge',
                            shrinkA=1,
                            shrinkB=1,
                            #lw=0.5,
                            mutation_scale=20,
                            fc="0.6", ec="none",
                            edgecolor='k', facecolor='w'))

    plt.xlim([-700., 700.])

    ax.plot([100, 200], [-250, -250], 'k', lw=1, clip_on=False)
    ax.text(150, -300, r'100$\mu$m', va='center', ha='center')

    #presynaptic spike trains axes
    plt.axes([.05, .35, .25, .55])

    pop_sptimes = []
    for i_pre in range(n_pre_syn):
        sp = pre_syn_sptimes[i_pre]
        for i_sp in range(len(sp)):
            pop_sptimes.append(sp[i_sp])
               
    for i_pre in range(n_pre_syn):
        plt.scatter(pre_syn_sptimes[i_pre],
                    i_pre*np.ones(len(pre_syn_sptimes[i_pre])),
                    s=1, edgecolors='none', facecolors='k')

    plt.ylim([0,n_pre_syn])
    plt.xlim([0,cell_parameters['tstopms']])
    plt.ylabel('train #', ha='left', labelpad=0)
    plt.title('Presynaptic spike times')
    
    ax = plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.set_xticklabels([])

    #spike rate axes
    plt.axes([.05,.12,.25,.2])

    binsize = 5
    bins=np.arange(0, cell_parameters['tstopms']+1., binsize)
    count,b = np.histogram(pop_sptimes, bins=bins)
    rate = count*(1000./binsize)*(1./n_pre_syn)
    plt.plot(b[0:-1],rate,color='black',lw=1)

    plt.xlim([0,cell_parameters['tstopms']])
    plt.ylim([0,10.])
    
    tvec = np.arange(point_electrode.LFP.shape[1])*cell.timeres_python 

    plt.xlabel('$t$ (ms)')
    plt.ylabel('Rate (spike/s)')
    
    ax = plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    #single neuron EPs axes
    plt.axes([.7,.35,.25,.55])

    plt.title('Single neuron extracellular potentials')
    plt.axis('off')

    for i_cell in range(n_cells):
        plt.plot(tvec,
                        i_cell+2.e3*single_LFPs[i_cell],
                        color=color_vec[i_cell], lw=1,
                        )

    plt.ylim([-1,n_cells-.5])

    #Summed LFPs axes
    plt.axes([.7,.12,.25,.2])
    plt.plot(tvec, 1E3*summed_LFP, color='black', lw=1)
    plt.ylim([-5.e-1,5e-1])

    plt.title('Summed extracellular potentials')
    plt.xlabel(r'$t$ (ms)')
    plt.ylabel(r'$\mu$V',ha='left',rotation='horizontal')

    ax = plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


    fig.savefig('example3.pdf', dpi=300)

