#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate usage of LFPy.Network with network of ball-and-stick type
morphologies with active HH channels inserted in the somas and passive-leak
channels distributed throughout the apical dendrite. The corresponding
morphology and template specifications are in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

Execution (w. MPI):

    mpirun -np 2 python example_network.py

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
# import modules:
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ss
import scipy.stats as st
from mpi4py import MPI
import neuron
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode

# set up MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
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
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
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
        filter_data=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    """helper function to draw line plots"""
    tvec = np.arange(data.shape[1])*dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    # apply temporal filter
    if filter_data:
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

    yticklabels = []
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
                '$2^{' + '{}'.format(np.log2(vlimround)
                                    ) + '}$ ' + '{0}'.format(unit),
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


################################################################################
# Set up shared and population-specific parameters
################################################################################
# relative path for simulation output:
OUTPUTPATH = 'example_network_output'

# class NetworkCell parameters:
cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    delete_sections=False,
)

# class NetworkPopulation parameters:
populationParameters = dict(
    Cell=NetworkCell,
    cell_args=cellParameters,
    pop_args=dict(
        radius=100.,
        loc=0.,
        scale=20.),
    rotation_args=dict(x=0., y=0.),
)

# class Network parameters:
networkParameters = dict(
    dt=2**-4,
    tstop=1200.,
    v_init=-65.,
    celsius=6.5,
    OUTPUTPATH=OUTPUTPATH
)

# class RecExtElectrode parameters:
electrodeParameters = dict(
    x=np.zeros(13),
    y=np.zeros(13),
    z=np.linspace(1000., -200., 13),
    N=np.array([[0., 1., 0.] for _ in range(13)]),
    r=5.,
    n=50,
    sigma=0.3,
    method="soma_as_point"
)

# method Network.simulate() parameters:
networkSimulationArguments = dict(
    rec_current_dipole_moment=True,
    rec_pop_contributions=True,
    to_memory=True,
    to_file=False
)

# population names, sizes and connection probability:
population_names = ['E', 'I']
population_sizes = [80, 20]
connectionProbability = [[0.1, 0.1], [0.1, 0.1]]

# synapse model. All corresponding parameters for weights,
# connection delays, multapses and layerwise positions are
# set up as shape (2, 2) nested lists for each possible
# connection on the form:
# [["E:E", "E:I"],
#  ["I:E", "I:I"]].
synapseModel = neuron.h.Exp2Syn
# synapse parameters
synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.),
                      dict(tau1=0.2, tau2=1.8, e=0.)],
                     [dict(tau1=0.1, tau2=9.0, e=-80.),
                      dict(tau1=0.1, tau2=9.0, e=-80.)]]
# synapse max. conductance (function, mean, st.dev., min.):
weightFunction = np.random.normal
weightArguments = [[dict(loc=0.002, scale=0.0002),
                    dict(loc=0.002, scale=0.0002)],
                   [dict(loc=0.02, scale=0.002),
                    dict(loc=0.02, scale=0.002)]]
minweight = 0.
# conduction delay (function, mean, st.dev., min.):
delayFunction = np.random.normal
delayArguments = [[dict(loc=1.5, scale=0.3),
                   dict(loc=1.5, scale=0.3)],
                  [dict(loc=1.5, scale=0.3),
                   dict(loc=1.5, scale=0.3)]]
mindelay = 0.3
multapseFunction = np.random.normal
multapseArguments = [[dict(loc=2., scale=.5), dict(loc=2., scale=.5)],
                     [dict(loc=5., scale=1.), dict(loc=5., scale=1.)]]
# method NetworkCell.get_rand_idx_area_and_distribution_norm
# parameters for layerwise synapse positions:
synapsePositionArguments = [[dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=500., scale=100.),
                                           dict(loc=500., scale=100.)],
                                  funweights=[0.5, 1.]
                                 ) for _ in range(2)],
                            [dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=0., scale=100.)],
                                  funweights=[1., 0.5]
                                 ) for _ in range(2)]]

if __name__ == '__main__':
    ############################################################################
    # Main simulation
    ############################################################################
    # create directory for output:
    if not os.path.isdir(OUTPUTPATH):
        if RANK == 0:
            os.mkdir(OUTPUTPATH)
    COMM.Barrier()

    # instantiate Network:
    network = Network(**networkParameters)

    # create E and I populations:
    for name, size in zip(population_names, population_sizes):
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters)


        # create excitatory background synaptic activity for each cell
        # with Poisson statistics
        for cell in network.populations[name].cells:
            idx = cell.get_rand_idx_area_norm(section='allsec', nidx=64)
            for i in idx:
                syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn',
                              weight=0.002,
                              **dict(tau1=0.2, tau2=1.8, e=0.))
                syn.set_spike_times_w_netstim(interval=100.)


    # create connectivity matrices and connect populations:
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic neurons
            # in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
                )

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=synapseModel,
                synparams=synapseParameters[i][j],
                weightfun=np.random.normal,
                weightargs=weightArguments[i][j],
                minweight=minweight,
                delayfun=delayFunction,
                delayargs=delayArguments[i][j],
                mindelay=mindelay,
                multapsefun=multapseFunction,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j],
                )

    # set up extracellular recording device:
    electrode = RecExtElectrode(**electrodeParameters)

    # run simulation:
    SPIKES, OUTPUT, DIPOLEMOMENT = network.simulate(
        electrode=electrode,
        **networkSimulationArguments
    )

    # collect somatic potentials across all RANKs to RANK 0:
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
        fig, ax = plt.subplots(1, 1)
        for name, spts, gids in zip(population_names, SPIKES['times'], SPIKES['gids']):
            t = []
            g = []
            for spt, gid in zip(spts, gids):
                t = np.r_[t, spt]
                g = np.r_[g, np.zeros(spt.size)+gid]
            ax.plot(t[t >= 200], g[t >= 200], '|', label=name)
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
                      ztransform=True
                     )
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
                      ztransform=True
                     )
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
            draw_lineplot(ax, decimate(OUTPUT[0][name], q=16), dt=network.dt*16,
                          T=(200, 1200),
                          scaling_factor=1.,
                          vlimround=None,
                          label=label,
                          scalebar=True,
                          unit='mV',
                          ylabels=True if i == 0 else False,
                          color='C{}'.format(i),
                          ztransform=True
                         )
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
            c = 'C0' if name == 'E' else 'C1'
            ax.plot([cell.xstart[0], cell.xend[0]],
                    [cell.ystart[0], cell.yend[0]],
                    [cell.zstart[0], cell.zend[0]], c,
                    lw=5, zorder=-cell.xmid[0]-cell.ymid[0])
            ax.plot([cell.xstart[1], cell.xend[-1]],
                    [cell.ystart[1], cell.yend[-1]],
                    [cell.zstart[1], cell.zend[-1]], c,
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
    electrode = None
    syn = None
    synapseModel = None
    for population in network.populations.values():
        for cell in population.cells:
            cell = None
        population.cells = None
        population = None
    pop = None
    network = None
    neuron.h('forall delete_section()')
