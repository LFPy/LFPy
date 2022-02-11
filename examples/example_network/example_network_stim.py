#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate extracellular electric stimulation of a LFPy.Network instance,
using the LFPy.Network.enable_extracellular_stimulation() method.

Here a simple model of neuron ball-and-stick is used to populate two
morphologies with active HH channels inserted in the somas and passive-leak
channels distributed throughout the apical dendrite. The corresponding
morphology and template specifications are in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

Same as example_network_to_file.py, except that a stimulating current is
applied instead of background Poisson activity.

Execution (w. MPI):

    mpirun python example_network_stim.py

Copyright (C) 2021 Computational Neuroscience Group, NMBU.

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
import numpy as np
import scipy.signal as ss
import scipy.stats as st
import h5py
from mpi4py import MPI
import neuron
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, \
    CurrentDipoleMoment

# set up MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

##########################################################################
# Function declarations
##########################################################################


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
    tvec = np.arange(data.shape[1]) * dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    # apply temporal filter
    if filter_data:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    # subtract mean in each channel
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
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=0.75,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=0.75,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i + 1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[-1], tvec[-1]],
                [-1, -2], lw=2, color='k', clip_on=False)
        ax.text(tvec[-1] + np.diff(T) * 0.02, -1.5,
                '$2^{' + f'{np.log2(vlimround)}' + '}$ ' + f'{unit}',
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


##########################################################################
# Set up shared and population-specific parameters
##########################################################################
# relative path for simulation output:
OUTPUTPATH = 'example_network_stim_output'

# class NetworkCell parameters:
cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    delete_sections=False,
    dt=2**-4,
    tstop=2000,
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
    celsius=36.5,
    OUTPUTPATH=OUTPUTPATH
)

# class RecExtElectrode parameters:
electrodeParameters = dict(
    x=np.array([-100, 0, 100, -100, 0, 100, -100, 0, 100]),
    y=np.array([100, 100, 100, 0, 0, 0, -100, -100, -100]),
    z=np.zeros(9),
    N=np.array([[0., 0., 1.] for _ in range(9)]),
    r=20.,  # 5um radius
    n=50,  # nb of discrete point used to compute the potential
    sigma=1,  # conductivity S/m
    method="linesource"
)

# method Network.simulate() parameters:
networkSimulationArguments = dict(
    rec_pop_contributions=True,
    to_memory=False,
    to_file=True
)

# population names, sizes and connection probability:
population_names = ['E', 'I']
population_sizes = [32, 8]
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
weightArguments = [[dict(loc=0.001, scale=0.0001),
                    dict(loc=0.001, scale=0.0001)],
                   [dict(loc=0.01, scale=0.001),
                    dict(loc=0.01, scale=0.001)]]
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
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=500., scale=100.)],
                                  funweights=[0.5, 1.]
                                  ) for _ in range(2)],
                            [dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=100., scale=100.)],
                                  funweights=[1., 0.5]
                                  ) for _ in range(2)]]

if __name__ == '__main__':
    ##########################################################################
    # Main simulation
    ##########################################################################
    # create directory for output:
    if RANK == 0:
        if not os.path.isdir(OUTPUTPATH):
            os.mkdir(OUTPUTPATH)
        # remove old simulation output if directory exist
        else:
            for fname in os.listdir(OUTPUTPATH):
                os.unlink(os.path.join(OUTPUTPATH, fname))
    COMM.Barrier()

    # instantiate Network:
    network = Network(**networkParameters)

    # create E and I populations:
    for name, size in zip(population_names, population_sizes):
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters)

    # create connectivity matrices and connect populations:
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic
            # neurons in each population (postsynaptic on this RANK)
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
                weightfun=weightFunction,
                weightargs=weightArguments[i][j],
                minweight=minweight,
                delayfun=delayFunction,
                delayargs=delayArguments[i][j],
                mindelay=mindelay,
                multapsefun=multapseFunction,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j],
                save_connections=False,
            )

    # set up extracellular recording device.
    # Here `cell` is set to None as handles to cell geometry is handled
    # internally
    electrode = RecExtElectrode(cell=None, **electrodeParameters)
    stim_elec = 4
    I_stim, t_ext = electrode.probe.set_current_pulses(
        n_pulses=20,
        biphasic=True,  # width2=width1, amp2=-amp1
        width1=5,
        amp1=3000,  # nA
        dt=network.dt,
        t_stop=network.tstop,
        interpulse=200,
        el_id=stim_elec,
        t_start=200)
    network.enable_extracellular_stimulation(electrode, t_ext, n=5)
    # set up recording of current dipole moments. Ditto with regards to
    # `cell` being set to None
    current_dipole_moment = CurrentDipoleMoment(cell=None)

    # run simulation:
    SPIKES = network.simulate(
        probes=[electrode, current_dipole_moment],
        **networkSimulationArguments
    )

    # collect somatic potentials across all RANKs to RANK 0:
    if RANK == 0:
        somavs = []
    for i, name in enumerate(population_names):
        somavs_pop = None  # avoid undeclared variable
        for j, cell in enumerate(network.populations[name].cells):
            if j == 0:
                somavs_pop = cell.somav
            else:
                somavs_pop = np.vstack((somavs_pop, cell.somav))
        if RANK == 0:
            for j in range(1, SIZE):
                recv = COMM.recv(source=j, tag=15)
                if somavs_pop is None:
                    if recv is not None:
                        somavs_pop = recv
                    else:
                        continue
                else:
                    if recv is not None:
                        somavs_pop = np.vstack((somavs_pop, recv))
            if somavs_pop.ndim == 1:
                somavs_pop = somavs_pop.reshape((1, -1))
            somavs.append(somavs_pop)
        else:
            COMM.send(somavs_pop, dest=0, tag=15)
        del somavs_pop

    ##########################################################################
    # Plot some output on RANK 0
    ##########################################################################
    if RANK == 0:
        # spike raster
        fig, ax = plt.subplots(1, 1)
        for name, spts, gids in zip(
                population_names, SPIKES['times'], SPIKES['gids']):
            t = []
            g = []
            for spt, gid in zip(spts, gids):
                t = np.r_[t, spt]
                g = np.r_[g, np.zeros(spt.size) + gid]
            ax.plot(t[t >= 200], g[t >= 200], '.', ms=3, label=name)
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
        gs = GridSpec(4, 1)
        ax = fig.add_subplot(gs[:2])
        if somavs[0].shape[0] > 10:
            somavs_pop = ss.decimate(somavs[0][:10], q=16, axis=-1,
                                     zero_phase=True)
        else:
            somavs_pop = ss.decimate(somavs[0], q=16, axis=-1,
                                     zero_phase=True)
        draw_lineplot(ax,
                      somavs_pop,
                      dt=network.dt * 16,
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

        ax = fig.add_subplot(gs[2:])
        if somavs[1].shape[0] > 10:
            somavs_pop = ss.decimate(somavs[1][:10], q=16, axis=-1,
                                     zero_phase=True)
        else:
            somavs_pop = ss.decimate(somavs[1], q=16, axis=-1,
                                     zero_phase=True)
        draw_lineplot(ax,
                      somavs_pop,
                      dt=network.dt * 16,
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
            with h5py.File(os.path.join(OUTPUTPATH, 'OUTPUT.h5'), 'r') as f:
                draw_lineplot(ax,
                              ss.decimate(f['RecExtElectrode0'][name], q=16,
                                          zero_phase=True),
                              dt=network.dt * 16,
                              T=(200, 1200),
                              scaling_factor=1.,
                              vlimround=None,
                              label=label,
                              scalebar=True,
                              unit='mV',
                              ylabels=True if i == 0 else False,
                              color=f'C{i}',
                              ztransform=True
                              )
            ax.set_title(label)
        fig.savefig(os.path.join(OUTPUTPATH, 'extracellular_potential.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

    ##########################################################################
    # customary cleanup of object references - the psection() function may not
    # write correct information if NEURON still has object references in memory
    # even if Python references has been deleted. It will also allow the script
    # to be run in successive fashion.
    ##########################################################################
    network.pc.gid_clear()  # allows assigning new gids to threads
    electrode = None
    syn = None
    synapseModel = None
    for population in network.populations.values():
        for cell in population.cells:
            cell.__del__()
            cell = None
        population.cells = None
        population = None
    pop = None
    network = None
    neuron.h('forall delete_section()')
