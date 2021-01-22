#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LFPs from a population of cells relying on MPI (Message Passing Interface)

Execution:

    <mpiexec> -n <processes> python example_mpi_2.py

Notes:
- on certain platforms and with mpirun, the --oversubscribe argument is needed
  to get more processes than the number of physical CPU cores.

Copyright (C) 2017 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from os.path import join
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
import zipfile
import ssl
import LFPy
from mpi4py import MPI

# initialize the MPI interface
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# set the numpy random seeds
global_seed = 1234
np.random.seed(global_seed)


def stationary_poisson(nsyn, lambd, tstart, tstop):
    ''' Generates nsyn stationary possion processes with rate
    lambda between tstart and tstop'''
    interval_s = (tstop - tstart) * .001
    spiketimes = []
    for i in range(nsyn):
        spikecount = np.random.poisson(interval_s * lambd)
        spikevec = np.empty(spikecount)
        if spikecount == 0:
            spiketimes.append(spikevec)
        else:
            spikevec = tstart + (tstop - tstart) * np.random.random(spikecount)
            spiketimes.append(np.sort(spikevec))  # sort them too!

    return spiketimes


# Fetch Mainen&Sejnowski 1996 model files
if not os.path.isfile(join('cells', 'cells', 'j4a.hoc')) and RANK == 0:
    # get the model files:
    url = '{}{}'.format('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp',
                        '?o=2488&a=23&mime=application/zip')
    u = urlopen(url, context=ssl._create_unverified_context())
    localFile = open('patdemo.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    # unzip:
    myzip = zipfile.ZipFile('patdemo.zip', 'r')
    myzip.extractall('.')
    myzip.close()

# resync MPI threads
COMM.Barrier()

# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology': join('cells', 'cells', 'j4a.hoc'),  # Mainen&Sejnowski, 1996
    'cm': 1.0,         # membrane capacitance
    'Ra': 150,         # axial resistance
    'v_init': -65.,    # initial crossmembrane potential
    'passive': True,   # turn on passive mechanism for all sections
    # passive params
    'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
    'nsegs_method': 'lambda_f',
    'lambda_f': 100.,
    'dt': 2.**-3,      # simulation time step size
    'tstart': 0.,     # start time of simulation, recorders start at t=0
    'tstop': 300.,     # stop simulation at 200 ms. These can be overridden
    # by setting these arguments i cell.simulation()
}

# Define synapse parameters
synapse_parameters = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'ExpSyn',       # synapse type
    'tau': 5.,                 # syn. time constant
    'weight': .001,            # syn. weight
    'record_current': True,
}

# Define electrode parameters
point_el_parameters = {
    'sigma': 0.3,      # extracellular conductivity
    'x': 0.,  # electrode requires 1d vector of positions
    'y': 0.,
    'z': 0.,
}

# number of units
n_cells = 6

# assign cell positions
x_cell_pos = np.linspace(-250., 250., n_cells)

# default rotation around x and y axis
xy_rotations = dict(x=4.99, y=-4.33)

# rotations around z-axis
if RANK == 0:
    z_rotation = COMM.bcast(np.random.permutation(np.arange(0., np.pi,
                                                            np.pi / n_cells)),
                            root=0)
else:
    z_rotation = COMM.bcast(None, root=0)


# synaptic spike times drawn on RANK 0 distributed to all processes
n_pre_syn = 1000
if RANK == 0:
    pre_syn_sptimes = COMM.bcast(stationary_poisson(nsyn=n_pre_syn, lambd=5.,
                                                    tstart=0, tstop=300),
                                 root=0)
else:
    pre_syn_sptimes = COMM.bcast(None, root=0)

# number of synapses on each cell
n_synapses = 100

# indices for presynaptic spike trains for each neuron also picked on RANK 0
# and scattered (for illustrating purposes, not efficiency)
if RANK == 0:
    # set up len SIZE nested list for spike train IDs.
    pre_syn_ids = [[]] * SIZE
    for cell_id in range(n_cells):
        pre_syn_ids[cell_id % SIZE] += [np.random.permutation(np.arange(
            n_pre_syn))[0:n_synapses]]
else:
    pre_syn_ids = None
pre_syn_ids = COMM.scatter(pre_syn_ids, root=0)

# containers for per-cell LFP and summed LFPs
single_LFPs = []
summed_LFP = np.zeros(
    int(cell_parameters['tstop'] / cell_parameters['dt'] + 1))

# get state of random seed generator
state = np.random.get_state()

# iterate over cells in populations
for cell_id in range(n_cells):
    if cell_id % SIZE == RANK:
        # get set seed per cell in order to synapse locations
        np.random.seed(global_seed + cell_id)

        # Create cell
        cell = LFPy.Cell(**cell_parameters)

        # Have to position and rotate the cells!
        cell.set_rotation(z=z_rotation[cell_id], **xy_rotations)
        cell.set_pos(x=x_cell_pos[cell_id])

        for i_syn in range(n_synapses):
            syn_idx = cell.get_rand_idx_area_norm()
            synapse_parameters.update({'idx': syn_idx})
            synapse = LFPy.Synapse(cell, **synapse_parameters)
            synapse.set_spike_times(pre_syn_sptimes[pre_syn_ids[
                cell_id % SIZE][i_syn]])

        # run the cell simulation
        cell.simulate(rec_imem=True)

        # set up the extracellular device
        point_el = LFPy.RecExtElectrode(cell, **point_el_parameters)
        point_el.data = point_el.get_transformation_matrix() @ cell.imem

        # sum LFP on this RANK
        summed_LFP += point_el.data[0]

        # send LFP of this cell to RANK 0
        if RANK != 0:
            COMM.send(point_el.data[0], dest=0)
        else:
            single_LFPs += [point_el.data[0]]

    # collect single LFP contributions on RANK 0
    if RANK == 0:
        if cell_id % SIZE != RANK:
            single_LFPs += [COMM.recv(source=cell_id % SIZE)]

# we can also use MPI to sum arrays directly:
summed_LFP = COMM.reduce(summed_LFP)

# reset state of random number generator
np.random.set_state(state)

# plot output on RANK 0.
if RANK == 0:
    # assign color to each unit
    color_vec = [plt.cm.rainbow(int(x * 256. / n_cells))
                 for x in range(n_cells)]

    # figure
    fig = plt.figure(figsize=(12, 8))

    # Morphologies axes:
    plt.axes([.175, .0, .65, 1], aspect='equal')
    plt.axis('off')

    for i_cell in range(n_cells):
        cell = LFPy.Cell(join('cells', 'cells', 'j4a.hoc'),
                         nsegs_method='lambda_f',
                         lambda_f=5)
        cell.set_rotation(z=z_rotation[i_cell], **xy_rotations)
        cell.set_pos(x=x_cell_pos[i_cell])

        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))
        linecol = LineCollection(zips,
                                 edgecolor='none',
                                 facecolor=color_vec[i_cell],
                                 rasterized=False,
                                 )

        ax = plt.gca()
        ax.add_collection(linecol)

    axis = ax.axis(ax.axis('equal'))
    ax.axis(np.array(axis) / 1.15)

    # adding a blue dot:
    ax.plot(point_el.x, point_el.z, 'o',
            markeredgecolor='none', markerfacecolor='b', markersize=3,
            zorder=10, clip_on=False)
    plt.annotate("Electrode",
                 xy=(0., 0.), xycoords='data',
                 xytext=(-100., 1000.),
                 arrowprops=dict(arrowstyle='wedge',
                                 shrinkA=1,
                                 shrinkB=1,
                                 # lw=0.5,
                                 mutation_scale=20,
                                 fc="0.6", ec="none",
                                 edgecolor='k', facecolor='w'))

    plt.xlim([-700., 700.])

    ax.plot([100, 200], [-250, -250], 'k', lw=1, clip_on=False)
    ax.text(150, -300, r'100$\mu$m', va='center', ha='center')

    # presynaptic spike trains axes
    plt.axes([.05, .35, .25, .55])

    pop_sptimes = []
    for i_pre in range(n_pre_syn):
        sp = pre_syn_sptimes[i_pre]
        for i_sp in range(len(sp)):
            pop_sptimes.append(sp[i_sp])

    for i_pre in range(n_pre_syn):
        plt.scatter(pre_syn_sptimes[i_pre],
                    i_pre * np.ones(len(pre_syn_sptimes[i_pre])),
                    s=1, edgecolors='none', facecolors='k')

    plt.ylim([0, n_pre_syn])
    plt.xlim([0, cell_parameters['tstop']])
    plt.ylabel('train #', ha='left', labelpad=0)
    plt.title('Presynaptic spike times')

    ax = plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xticklabels([])

    # spike rate axes
    plt.axes([.05, .12, .25, .2])

    binsize = 5
    bins = np.arange(0, cell_parameters['tstop'] + 1., binsize)
    count, b = np.histogram(pop_sptimes, bins=bins)
    rate = count * (1000. / binsize) * (1. / n_pre_syn)
    plt.plot(b[0:-1], rate, color='black', lw=1)

    plt.xlim([0, cell_parameters['tstop']])
    plt.ylim([0, 10.])

    tvec = np.arange(point_el.data.shape[1]) * cell.dt

    plt.xlabel('$t$ (ms)')
    plt.ylabel('Rate (spike/s)')

    ax = plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # single neuron EPs axes
    plt.axes([.7, .35, .25, .55])

    plt.title('Single neuron extracellular potentials')
    plt.axis('off')

    for cell_id in range(n_cells):
        plt.plot(tvec,
                 cell_id + 2.e3 * single_LFPs[cell_id],
                 color=color_vec[cell_id], lw=1,
                 )

    plt.ylim([-1, n_cells - .5])

    # Summed LFPs axes
    plt.axes([.7, .12, .25, .2])
    plt.plot(tvec, 1E3 * summed_LFP, color='black', lw=1)
    plt.ylim([-5.e-1, 5e-1])

    plt.title('Summed extracellular potentials')
    plt.xlabel(r'$t$ (ms)')
    plt.ylabel(r'$\mu$V', ha='left', rotation='horizontal')

    ax = plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    fig.savefig('example_mpi_2.pdf', dpi=300)
    plt.show()
