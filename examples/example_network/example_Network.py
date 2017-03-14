#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate usage of LFPy.Network with network of ball-and-stick type
morphologies with active HH channels inserted in the somas and passive-leak
channels distributed throughout the apical dendrite. The corresponding
morphology and template specifications are in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ss
import scipy.stats as st
import os
from mpi4py import MPI
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode
import neuron

# MPI variables
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell locations and random synapse activation
# times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

################################################################################
# Function declarations
################################################################################

def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.


    Parameters
    ----------
    x : ndarray
        Array to be downsampled along last axis.
    q : int
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    ndarray
        Downsampled signal.

    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]

def remove_axis_junk(ax, lines=['right', 'top']):
    for loc, spine in ax.spines.iteritems():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    ''' draw some nice lines'''

    tvec = np.arange(data.shape[1])*dt
    try:
        tinds = (tvec >= T[0]) & (tvec <= T[1])
    except TypeError:
        print data.shape, T
        raise Exception

    # apply temporal filter
    if filter:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    #subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels=[]
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[-1], tvec[-1]],
                [-1, -2], lw=2, color='k', clip_on=False)
        ax.text(tvec[-1]+np.diff(T)*0.02, -1.5,
                '$2^{' + '{}'.format(np.log2(vlimround)) + '}$ ' + '{0}'.format(unit),
                color='k', rotation='vertical',
                va='center')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r't (ms)', labelpad=0.1)

    return vlimround


def ReduceStructArray(sendbuf, op=MPI.SUM):
    """
    simplify MPI Reduce for structured ndarrays with floating point numbers

    Parameters
    ----------
    sendbuf : structured ndarray
        Array data to be reduced (default: summed)
    op : mpi4py.MPI.Op object
        MPI_Reduce function. Default is mpi4py.MPI.SUM
    """
    if RANK == 0:
        shape = sendbuf.shape
        dtype_names = sendbuf.dtype.names
    else:
        shape = None
        dtype_names = None
    shape = COMM.bcast(shape)
    dtype_names = COMM.bcast(dtype_names)

    if RANK == 0:
        reduced = np.zeros(shape,
                           dtype=zip(dtype_names,
                                     ['f8' for i in range(len(dtype_names))]))
    else:
        reduced = None
    for name in dtype_names:
        if RANK == 0:
            recvbuf = np.zeros(shape)
        else:
            recvbuf = None
        COMM.Reduce(np.array(sendbuf[name]), recvbuf, op=op, root=0)
        if RANK == 0:
            reduced[name] = recvbuf
    return reduced


################################################################################
# Set up shared and population-specific parameters
################################################################################
OUTPUTPATH='example_network_output'

cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    passive=False,
    dt=2**-4,
    tstopms=1200,
    delete_sections=False,
)

populationParameters = dict(
    CWD=None,
    CELLPATH=None,
    Cell=NetworkCell,
    cell_args = cellParameters,
    pop_args = dict(
        radius=100,
        loc=0.,
        scale=20.),
    rotation_args = dict(x=0, y=0),
)

networkParameters = dict(
    dt = 2**-4,
    tstop = 1200.,
    v_init = -65.,
    celsius = 6.5,
    OUTPUTPATH = OUTPUTPATH
)

electrodeParameters = dict(
    x = np.zeros(13),
    y = np.zeros(13),
    z = np.linspace(1000, -200, 13),
    N = np.array([[0, 1, 0]]*13),
    r = 5,
    n = 50,
    sigma = 0.3,
)

networkSimulationArguments = dict(
    rec_current_dipole_moment = True,
    rec_pop_contributions = True,
)

# population and connection specific parameters
population_names = ['E', 'I']
population_sizes = [80, 20]

connectionProbability = 0.05
synapseModel = neuron.h.Exp2Syn
synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.)]*2,
                     [dict(tau1=0.1, tau2=9.0, e=-80.)]*2]
weightFunction = np.random.normal
weightArguments = [[dict(loc=0.002, scale=0.0002)]*2,
                   [dict(loc=0.01,  scale=0.001)]*2]
delayFunction = np.random.normal
delayArguments = [[dict(loc=1.5, scale=0.3)]*2]*2
multapseFunction = np.random.normal
multapseArguments = [[dict(loc=2., scale=.5)]*2, [dict(loc=5., scale=1.)]*2]
synapsePositionArguments = [[dict(section=['soma', 'apic'],
                                  fun=[st.norm]*2,
                                  funargs=[dict(loc=500., scale=100.)]*2,
                                  funweights=[0.5, 1.])]*2,
                            [dict(section=['soma', 'apic'],
                                  fun=[st.norm]*2,
                                  funargs=[dict(loc=0., scale=100.)]*2,
                                  funweights=[1., 0.5])]*2]

if __name__ == '__main__':
    ############################################################################
    # Main simulation
    ############################################################################

    if not os.path.isdir(OUTPUTPATH):
        if RANK == 0:
            os.mkdir(OUTPUTPATH)
    COMM.Barrier()

    # instantiate network class
    network = Network(**networkParameters)

    # instantiate populations
    for name, size in zip(population_names, population_sizes):
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters)


        # create some background synaptic activity onto the cells with Poisson
        # activation statistics
        for cell in network.populations[name].cells:
            # if cell.gid == 0:
            idx = cell.get_rand_idx_area_norm(section='allsec', nidx=64)
            for i in idx:
                syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn',
                              weight=0.002,
                              **dict(tau1=0.2, tau2=1.8, e=0.))
                syn.set_spike_times_w_netstim(interval=1000. / 5.)



    # create connectivity and connect populations
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic neurons
            # in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(pre=pre, post=post,
                                                connprob=connectionProbability)

            # connect network
            (conncount, syncount) = network.connect(
                            pre=pre, post=post,
                            connectivity=connectivity,
                            syntype=synapseModel,
                            synparams=synapseParameters[i][j],
                            weightfun=np.random.normal,
                            weightargs=weightArguments[i][j],
                            delayfun=delayFunction,
                            delayargs=delayArguments[i][j],
                            multapsefun=multapseFunction,
                            multapseargs=multapseArguments[i][j],
                            syn_pos_args=synapsePositionArguments[i][j],
                            )

    # set up extracellular recording device
    electrode = RecExtElectrode(**electrodeParameters)

    # run simulation
    OUTPUT, DIPOLEMOMENT = network.simulate(
        electrode=electrode,
        **networkSimulationArguments
    )

    # Sum output across RANKs to RANK 0
    OUTPUT = ReduceStructArray(OUTPUT[0])
    DIPOLEMOMENT = ReduceStructArray(DIPOLEMOMENT)

    # Collect spike trains across all RANKs to RANK 0
    for name in population_names:
        population = network.populations[name]
        for i in range(len(population.spike_vectors)):
            population.spike_vectors[i] = np.array(population.spike_vectors[i])
    if RANK == 0:
        spikes = []
        neurons = []
        for i, name in enumerate(population_names):
            spikes.append([])
            neurons.append([])
            spikes[i] += [x for x in network.populations[name].spike_vectors]
            neurons[i] += [x for x in network.populations[name].gids]
            for j in range(1, SIZE):
                spikes[i] += COMM.recv(source=j, tag=13)
                neurons[i] += COMM.recv(source=j, tag=14)
    else:
        spikes = None
        neurons = None
        for name in population_names:
            COMM.send([x for x in network.populations[name].spike_vectors],
                dest=0, tag=13)
            COMM.send([x for x in network.populations[name].gids],
                dest=0, tag=14)

    # collect somatic potentials across all RANKs to RANK 0
    if RANK == 0:
        somavs = []
        for i, name in enumerate(population_names):
            somavs.append([])
            somavs[i] += [cell.somav
                          for cell in network.populations[name].cells]
            for j in range(1, SIZE):
                somavs[i] += COMM.recv(source=j, tag=15)
    else:
        somavs = None
        for name in population_names:
            COMM.send([cell.somav for cell in network.populations[name].cells],
                dest=0, tag=15)

    ############################################################################
    # Plot some output on RANK 0
    ############################################################################

    if RANK == 0:
        # spike raster
        fig, ax = plt.subplots(1,1)
        for name, spts, gids in zip(population_names, spikes, neurons):
            t = []
            g = []
            for spt, gid in zip(spts, gids):
                t = np.r_[t, spt]
                g = np.r_[g, np.zeros(spt.size)+gid]
            ax.plot(t[t >= 200], g[t >= 200], '.', label=name)
        ax.legend(loc=1)
        remove_axis_junk(ax, lines=['right', 'top'])
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('gid')
        ax.set_title('spike raster')
        fig.savefig(os.path.join(OUTPUTPATH, 'spike_raster.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


        # somatic potentials
        fig = plt.figure()
        gs = GridSpec(5, 1)
        ax = fig.add_subplot(gs[:4])
        draw_lineplot(ax, decimate(np.array(somavs)[0], q=16), dt=network.dt*16,
                    T=(200, 1200),
                    scaling_factor=1.,
                    vlimround=16,
                    label='E',
                    scalebar=True,
                    unit='mV',
                    ylabels=False,
                    color='C0',
        ztransform=True)
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_ylabel('E')
        ax.set_title('somatic potentials')
        ax.set_xlabel('')

        ax = fig.add_subplot(gs[4])
        draw_lineplot(ax, decimate(np.array(somavs)[1], q=16), dt=network.dt*16,
                    T=(200, 1200),
                    scaling_factor=1.,
                    vlimround=16,
                    label='I',
                    scalebar=True,
                    unit='mV',
                    ylabels=False,
                    color='C1',
                    ztransform=True)
        ax.set_yticks([])
        ax.set_ylabel('I')

        fig.savefig(os.path.join(OUTPUTPATH, 'soma_potentials.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


        # extracellular potentials, E and I contributions, sum
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8))
        fig.suptitle('extracellular potentials')
        for i, (ax, name, label) in enumerate(zip(axes, ['E', 'I', 'imem'],
                                                  ['E', 'I', 'sum'])):
            draw_lineplot(ax, decimate(OUTPUT[name], q=16), dt=network.dt*16,
                        T=(200, 1200),
                        scaling_factor=1.,
                        vlimround=None,
                        label=label,
                        scalebar=True,
                        unit='mV',
                        ylabels=True if i == 0 else False,
                        color='C{}'.format(i),
                        ztransform=True)
            ax.set_title(label)
        fig.savefig(os.path.join(OUTPUTPATH, 'extracellular_potential.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


        # current-dipole moments, E and I contributions, sum
        fig, axes = plt.subplots(3, 3, figsize=(6.4, 4.8))
        fig.subplots_adjust(wspace=0.45)
        fig.suptitle('current-dipole moments')
        for i, u in enumerate(['x', 'y', 'z']):
            for j, label in enumerate(['E', 'I', 'sum']):
                t = np.arange(DIPOLEMOMENT.shape[0])*network.dt
                inds = (t >= 200) & (t <= 1200)
                if label != 'sum':
                    axes[i, j].plot(t[inds][::16],
                                    decimate(DIPOLEMOMENT[label][inds, i],
                                             q=16),
                                    'C{}'.format(j))
                else:
                    axes[i, j].plot(t[inds][::16],
                                    decimate(DIPOLEMOMENT['E'][inds, i] +
                                             DIPOLEMOMENT['I'][inds, i], q=16),
                                    'C{}'.format(j))

                if j == 0:
                    axes[i, j].set_ylabel(r'$\mathbf{p}\cdot\mathbf{e}_{' +
                                          '{}'.format(u) +'}$ (nA$\mu$m)')
                if i == 0:
                    axes[i, j].set_title(label)
                if i != 2:
                    axes[i, j].set_xticklabels([])
                else:
                    axes[i, j].set_xlabel('t (ms)')
        fig.savefig(os.path.join(OUTPUTPATH, 'current_dipole_moment.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


    # population illustration (per RANK)
    fig = plt.figure(figsize=(6.4, 4.8*2))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=5)
    ax.plot(electrode.x, electrode.y, electrode.z, 'ko', zorder=0)
    for i, (name, pop) in enumerate(network.populations.items()):
        for cell in pop.cells:
            ax.plot([cell.xstart[0], cell.xend[0]],
                    [cell.ystart[0], cell.yend[0]],
                    [cell.zstart[0], cell.zend[0]], 'C{}'.format(i),
                    lw=5, zorder=-cell.xmid[0]-cell.ymid[0])
            ax.plot([cell.xstart[1], cell.xend[-1]],
                    [cell.ystart[1], cell.yend[-1]],
                    [cell.zstart[1], cell.zend[-1]], 'C{}'.format(i),
                    lw=0.5, zorder=-cell.xmid[0]-cell.ymid[0])
    ax.set_xlabel('$x$ ($\mu$m)')
    ax.set_ylabel('$y$ ($\mu$m)')
    ax.set_zlabel('$z$ ($\mu$m)')
    ax.set_title('network populations')
    fig.savefig(os.path.join(OUTPUTPATH, 'population_RANK_{}.pdf'.format(RANK)),
                bbox_inches='tight')
    plt.close(fig)


    ############################################################################
    # customary cleanup of object references - the psection() function may not
    # write correct information if NEURON still has object references in memory,
    # even if Python references has been deleted. It will also allow the script
    # to be run in successive fashion.
    ############################################################################
    network.pc.gid_clear() # allows assigning new gids to threads
    syn = None
    for population in network.populations.values():
        for cell in population.cells:
            cell = None
        population.cells = None
        population = None
    network = None
    neuron.h('forall delete_section()')
