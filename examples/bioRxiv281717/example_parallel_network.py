#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Main simulation script for setting up the model producing the data shown in
figure 4, 5 and 6 in:

    Multimodal modeling of neural network activity: computing LFP, ECoG, EEG
    and MEG signals with LFPy2.0
    Espen Hagen, Solveig Næss, Torbjørn V Ness, Gaute T Einevoll
    bioRxiv 281717; doi: https://doi.org/10.1101/281717

This example file is not suited for execution on laptops or desktop computers.
In the corresponding parameter file example_parallel_network_parameters.py
there is an option to set

    TESTING = True

which will set the number of neurons in each network population to one, which
facilitates testing for missing files or data. Otherwise, the full number of
neurons in each population will be created.

Various biophysically detailed neuron models required for these simulations
can be obtained from The Neocortical Microcircuit Collaboration Portal
(https://bbp.epfl.ch/nmc-portal/welcome) providing the cell models used in a
reconstruction of a rat somatosensory cortex column described in:

Markram H, Muller E, Ramaswamy S†, Reimann MW,  Abdellah M, Sanchez CA,
Ailamaki A, Alonso-Nanclares L, Antille N, Arsever S et al. (2015).
Reconstruction and Simulation of Neocortical Microcircuitry.
Cell 163:2, 456 - 492. doi: 10.1016/j.cell.2015.09.029

A tar file with all single-cell models zipped can be downloaded and unpacked
by issuing:
$ wget https://bbp.epfl.ch/nmc-portal/assets/documents/static/Download/\
hoc_combos_syn.1_0_10.allzips.tar
$ tar -xvf hoc_combos_syn.1_0_10.allzips.tar
$ cd hoc_combos_syn.1_0_10.allzips
$ unzip 'L4_PC_*.zip'
$ unzip 'L4_LBC_*.zip'
$ unzip 'L5_TTPC1_*.zip'
$ unzip 'L5_MC_*.zip'
$ cd -

.json files with anatomy and physiology data of the microcircuit are also
needed:
$ wget https://bbp.epfl.ch/nmc-portal/assets/documents/static/Download/\
pathways_anatomy_factsheets_simplified.json
$ wget https://bbp.epfl.ch/nmc-portal/assets/documents/static/Download/\
pathways_physiology_factsheets_simplified.json

Some preparatory steps has to be made in order to compile NMODL language
files used by the neuron models:

Set working dir
>>> import os
>>> import sys
>>> import neuron
>>> from glob import glob
>>> CWD = os.getcwd()
>>> NMODL = 'hoc_combos_syn.1_0_10.allmods'

Load some required neuron-interface files
>>> neuron.h.load_file("stdrun.hoc")
>>> neuron.h.load_file("import3d.hoc")

Load only some layer 5 pyramidal cell types
Define a list of the neuron models (defined in the parameter file)
>>> neurons = [os.path.join('hoc_combos_syn.1_0_10.allzips',
                            'L4_PC_cADpyr230_1'),
               os.path.join('hoc_combos_syn.1_0_10.allzips',
                            'L4_LBC_dNAC222_1'),
               os.path.join('hoc_combos_syn.1_0_10.allzips',
                            'L5_TTPC1_cADpyr232_1'),
               os.path.join('hoc_combos_syn.1_0_10.allzips',
                            'L5_MC_bAC217_1')]

Attempt to set up a folder with all unique mechanism mod files, compile, and
load them all. One synapse mechanism file is faulty and must be patched.
>>> if not os.path.isdir(NMODL):
>>>     os.mkdir(NMODL)
>>> for NRN in neurons:
>>>     for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
>>>         while not os.path.isfile(os.path.join(NMODL,
>>>                                               os.path.split(nmodl)[-1])):
>>>             if "win32" in sys.platform:
>>>                 os.system("copy {} {}".format(nmodl, NMODL))
>>>             else:
>>>                 os.system('cp {} {}'.format(nmodl,
>>>                                             os.path.join(NMODL,
>>>                                                      '.')))
>>> os.chdir(NMODL)
>>> diff = """319c319
<                 urand = scop_random(1)
---
>                 value = scop_random(1)
"""
>>> f = open('ProbGABAAB_EMS.patch', 'w')
>>> f.writelines(diff)
>>> f.close()
>>> os.system('patch ProbGABAAB_EMS.mod ProbGABAAB_EMS.patch')
>>> if "win32" in sys.platform:
>>>     warn("no autompile of NMODL (.mod) files on Windows. "
>>>          + "Run mknrndll from NEURON bash in the folder "
>>>          + "%s and rerun example script" % NMODL)
>>> else:
>>>     os.system('nrnivmodl')
>>> os.chdir(CWD)

An example job script set up using the SLURM workload management software
(https://slurm.schedmd.com/) on a compute cluster is provided in the file
example_parallel_network.job. This job script asks for exclusive access to
24 nodes with 24 physical CPU cores each. Adjust accordingly for other
clusters.

The job can be submitted issuing:

    $ sbatch example_parallel_network.job


Execution example:

    $ mpirun -np 1152 python example_parallel_network.py

Output is stored in the folder ./example_parallel_network_output, which is set
in the parameter file. Adjust accordingly to use the work area on your cluster.
Some pdf figures are saved as example_parallel_network*.pdf,
showing somatic responses of different cells on some MPI processes

The script uses some plotting routines from the file
example_parallel_network_plotting.py


If the simulation finished successfully, the figures 4-6 from the manuscript
can be generated by issuing:

    $ python figure_4.py
    $ python figure_5.py
    $ python figure_6.py


Copyright (C) 2018 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''
from mpi4py import MPI
import LFPy
from time import time
import os
import h5py
from packaging.version import Version
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from scipy.signal import decimate
import neuron
import matplotlib
matplotlib.use('agg')
if Version(h5py.version.hdf5_version) < Version('1.8.16'):
    m = f'h5py uses HDF5 {h5py.version.hdf5_version}: 1.8.16 or newer required'
    raise ImportError(m)


# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# set default plotting parameters
fontsize = 8
plt.rcParams.update({
    'axes.xmargin': 0.0,
    'axes.ymargin': 0.0,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'axes.titleweight': fontsize,
    'figure.titlesize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': fontsize,
})


# tic toc
tic = time()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell locations and random synapse activation
# times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)


##########################################################################
# Main simulation procedure
##########################################################################

if __name__ == '__main__':
    # Remove cells from previous script executions
    neuron.h('forall delete_section()')

    ##########################################################################
    # Simulation control and parameters
    ##########################################################################
    # import LFPy.NetworkCell and Network classes
    from LFPy import NetworkCell, Network

    # tic-toc
    if RANK == 0:
        initialization_time = time() - tic
        print(f'Initialization in {initialization_time} seconds')
    tic = time()

    # import main parameters dictionary for simulation
    from example_parallel_network_parameters import PSET

    # set up file destination
    if RANK == 0:
        # create directory for output:
        if not os.path.isdir(PSET.OUTPUTPATH):
            os.mkdir(PSET.OUTPUTPATH)
        # remove old simulation output if directory exist
        else:
            for fname in os.listdir(PSET.OUTPUTPATH):
                os.unlink(os.path.join(PSET.OUTPUTPATH, fname))
    COMM.Barrier()

    # Modify release probabilities of excitatory synapses in order to
    # stabilize circuit.
    # This change is incorporated in https://github.com/LFPy/LFPy/pull/320
    # which modifies slightly the way multapse counts are generated
    # (and hence affected the network state compared to the old behaviour)
    for i, pre in enumerate(PSET.populationParameters['m_type']):
        for j, post in enumerate(PSET.populationParameters['m_type']):
            if (pre in ['L4_PC', 'L5_TTPC1']) & (pre == post):
                PSET.connParams['synparams'][i][j]['Use'] = \
                    PSET.connParams['synparams'][i][j]['Use'] * 0.8

    if RANK == 0:
        parameters_time = time() - tic
        print(f'Parameters in {parameters_time} seconds')
    tic = time()

    ##########################################################################
    # Create population, provide noisy input to each cell, connect network
    ##########################################################################
    # create network object instance
    network = Network(dt=PSET.dt, tstop=PSET.tstop, v_init=PSET.v_init,
                      celsius=PSET.celsius,
                      OUTPUTPATH=PSET.OUTPUTPATH)
    # create populations iteratively
    for name, pop_args, rotation_args, POP_SIZE in zip(
            PSET.populationParameters['me_type'],
            PSET.populationParameters['pop_args'],
            PSET.populationParameters['rotation_args'],
            (PSET.populationParameters['POP_SIZE']
             * PSET.POPSCALING).astype(int)):
        network.create_population(CWD=PSET.CWD, CELLPATH=PSET.CELLPATH,
                                  Cell=NetworkCell, POP_SIZE=POP_SIZE,
                                  name=name,
                                  cell_args=PSET.cellParameters[name],
                                  pop_args=pop_args,
                                  rotation_args=rotation_args)

    # tic-toc
    if RANK == 0:
        create_population_time = time() - tic
        print(f'Populations initialized in {create_population_time} seconds')
    tic = time()

    # Sync MPI threads as populations may take a different amount of
    # time across RANKs. All neurons must have been created before connections
    # are made
    COMM.Barrier()

    #
    # # Attach current stimulus to the soma of the cell with gid 0.
    # if False:
    #     for name in PSET.populationParameters['me_type']:
    #         for cell in network.populations[name].cells:
    #             if cell.gid == 0:
    #                 LFPy.StimIntElectrode(cell,
    #                                       amp = 0.4,
    #                                       **PSET.PointProcParams)

    # create for each cell in each population some external input with Poisson
    # statistics using NEURON's NetStim device (controlled using LFPy.Synapse)
    for m_type, me_type, section, rho, f, synparams, weightfun, weightargs \
        in zip(PSET.populationParameters['m_type'],
               PSET.populationParameters['me_type'],
               PSET.populationParameters['extrinsic_input_section'],
               PSET.populationParameters['extrinsic_input_density'],
               PSET.populationParameters['extrinsic_input_frequency'],
               PSET.connParamsExtrinsic['synparams'],
               PSET.connParamsExtrinsic['weightfuns'],
               PSET.connParamsExtrinsic['weightargs']):
        for cell in network.populations[me_type].cells:
            idx = cell.get_rand_idx_area_norm(
                section=section,
                nidx=int(cell.area[cell.get_idx(section)].sum() * rho))
            for i in idx:
                syn = LFPy.Synapse(cell=cell, idx=i,
                                   syntype=PSET.connParamsExtrinsic['syntype'],
                                   weight=weightfun(**weightargs),
                                   **synparams)
                syn.set_spike_times_w_netstim(interval=1000. / f,
                                              seed=np.random.rand() * 2**32 - 1
                                              )

    # connect pre and post-synaptic populations with some connectivity and
    # weight of connections and other connection parameters:
    total_conncount = 0
    total_syncount = 0
    for i, pre in enumerate(PSET.populationParameters['me_type']):
        for j, post in enumerate(PSET.populationParameters['me_type']):
            # boolean connectivity matrix between pre- and post-synaptic
            # neurons in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=PSET.connParams['connprob'][i][j])

            # connect network
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=PSET.connParams['syntypes'][i][j],
                synparams=PSET.connParams['synparams'][i][j],
                weightfun=PSET.connParams['weightfuns'][i][j],
                weightargs=PSET.connParams['weightargs'][i][j],
                delayfun=PSET.connParams['delayfuns'][i][j],
                delayargs=PSET.connParams['delayargs'][i][j],
                mindelay=PSET.connParams['mindelay'],
                multapsefun=PSET.connParams['multapsefuns'][i][j],
                multapseargs=PSET.connParams['multapseargs'][i][j],
                syn_pos_args=PSET.connParams['syn_pos_args'][i][j],
                save_connections=PSET.save_connections,
            )
            total_conncount += conncount
            total_syncount += syncount

    # tic-toc
    if RANK == 0:
        create_connections_time = time() - tic
        print(
            'Network build finished with ' +
            f'{total_conncount} connections and ' +
            f'{total_syncount} synapses in {create_connections_time} seconds')
    tic = time()

    ##########################################################################
    # Set up extracellular electrodes and devices
    ##########################################################################
    probes = []

    if PSET.COMPUTE_LFP:
        electrode = LFPy.RecExtElectrode(cell=None, **PSET.electrodeParams)
        probes.append(electrode)

    if PSET.COMPUTE_ECOG:
        ecog_electrode = LFPy.RecMEAElectrode(cell=None, **PSET.ecogParameters)
        probes.append(ecog_electrode)

    if PSET.COMPUTE_P:
        current_dipole_moment = LFPy.CurrentDipoleMoment(cell=None)
        probes.append(current_dipole_moment)

    ##########################################################################
    # Recording of additional variables
    ##########################################################################
    if RANK == 0:
        network.t = neuron.h.Vector()
        network.t.record(neuron.h._ref_t)
    else:
        network.t = None

    ##########################################################################
    # run simulation, gather results across all RANKs
    ##########################################################################
    # Assert that connect routines has finished across RANKS before starting
    # main simulation procedure
    COMM.Barrier()
    if RANK == 0:
        print('running simulation....')
    SPIKES = network.simulate(probes=probes,
                              rec_pop_contributions=PSET.rec_pop_contributions,
                              **PSET.NetworkSimulateArgs)

    if RANK == 0:
        run_simulation_time = time() - tic
        print(f'Simulations finished in {run_simulation_time} seconds')
    tic = time()

    ##########################################################################
    # save simulated output to file to allow for offline plotting
    ##########################################################################

    if RANK == 0:
        f = h5py.File(os.path.join(PSET.OUTPUTPATH,
                                   'example_parallel_network_output.h5'),
                      'w')

        if PSET.COMPUTE_P:
            # save current dipole moment
            f['CURRENT_DIPOLE_MOMENT'] = current_dipole_moment.data

        if PSET.COMPUTE_LFP:
            # save all extracellular traces
            f['SUMMED_OUTPUT'] = electrode.data
        if PSET.COMPUTE_ECOG:
            # save extracellular potential on top of cortex
            f['SUMMED_ECOG'] = ecog_electrode.data

        # all spikes
        grp = f.create_group('SPIKES')
        # variable length datatype for spike time arrays
        dtype = h5py.special_dtype(vlen=np.dtype('float'))
        for i, name in enumerate(PSET.populationParameters['me_type']):
            subgrp = grp.create_group(name)
            if len(SPIKES['gids'][i]) > 0:
                subgrp['gids'] = np.array(SPIKES['gids'][i]).flatten()
                dset = subgrp.create_dataset('times',
                                             (len(SPIKES['gids'][i]),),
                                             dtype=dtype)
                for j, spt in enumerate(SPIKES['times'][i]):
                    dset[j] = spt
            else:
                subgrp['gids'] = []
                subgrp['times'] = []
        f.close()

    COMM.Barrier()

    # clean up namespace
    del electrode, ecog_electrode, current_dipole_moment, probes

    # tic toc
    if RANK == 0:
        saving_data_time = time() - tic
        print(f'Wrote output files in {saving_data_time} seconds')
    tic = time()

    # create logfile recording time in seconds for different simulation steps
    # (initialization, parameters, simulation etc.)
    if RANK == 0:
        logfile = open(os.path.join(PSET.OUTPUTPATH, 'log.txt'), 'w')
        logfile.write(f'initialization {initialization_time}\n')
        logfile.write(f'parameters {parameters_time}\n')
        logfile.write(f'population {create_population_time}\n')
        logfile.write(f'connections {create_connections_time}\n')
        logfile.write(f'simulation {run_simulation_time}\n')
        logfile.write(f'save {saving_data_time}\n')
        logfile.close()


if __name__ == '__main__':
    ##########################################################################
    # Plot simulated output (relies on Network class instance)
    ##########################################################################

    T = (PSET.TRANSIENT, PSET.tstop)

    colors = [plt.get_cmap('Set1', PSET.populationParameters.size)(i)
              for i in range(PSET.populationParameters.size)]

    # don't want thousands of figure files:
    PLOTRANKS = np.arange(0, SIZE, 16) if SIZE >= 48 else np.arange(SIZE)

    if RANK in PLOTRANKS:
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(left=0.2)

        nrows = np.sum([len(population.gids)
                        for population in network.populations.values()])
        ncols = 1

        gs = GridSpec(nrows=nrows, ncols=ncols)

        fig.suptitle(f'RANK {RANK}')
        counter = 0
        # somatic traces
        tvec = np.arange(PSET.tstop / PSET.dt + 1) * PSET.dt
        tinds = tvec >= PSET.TRANSIENT
        for i, name in enumerate(PSET.populationParameters['me_type']):
            population = network.populations[name]
            for j, cell in enumerate(population.cells):
                ax = fig.add_subplot(gs[counter, 0])
                if counter == 0:
                    ax.set_title('somatic voltages')
                ax.plot(tvec[tinds][::PSET.decimate_q],
                        decimate(cell.somav[tinds],
                                 q=PSET.decimate_q),
                        color=colors[i], lw=1.5, label=name)
                ax.set_ylabel(f'gid {population.gids[j]}',
                              rotation='horizontal', labelpad=30)
                ax.axis(ax.axis('tight'))
                ax.set_ylim(-90, -20)
                ax.legend(loc='best')
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                counter += 1
                if counter == nrows:
                    ax.set_xlabel('time (ms)')
                else:
                    ax.set_xticklabels([])

        # save figure output
        fig.savefig(os.path.join(PSET.OUTPUTPATH,
                                 f'example_parallel_network_RANK_{RANK}.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

    # make an illustration of the different populations on each RANK.
    if RANK in PLOTRANKS:  # don't want thousands of figure files

        # figure out which populations has at least one cell on this RANK
        local_me_types = []
        for i, name in enumerate(PSET.populationParameters['me_type']):
            if len(network.populations[name].gids) >= 1:
                local_me_types += [name]

        fig, axes = plt.subplots(1, len(local_me_types) + 1,
                                 figsize=((len(local_me_types) + 1) * 5, 10),
                                 sharey=True, sharex=True)
        ax = axes[0]
        # plot electrode contact points
        ax.plot(PSET.electrodeParams['x'], PSET.electrodeParams['z'],
                'ko', markersize=5)

        # plot cell geometries
        for i, name in enumerate(PSET.populationParameters['me_type']):
            population = network.populations[name]
            zips = []
            for cell in population.cells:
                for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                    zips.append(list(zip(x, z)))
            polycol = PolyCollection(zips,
                                     edgecolors=colors[i],
                                     linewidths=0.01,
                                     facecolors=colors[i],
                                     label=name,
                                     )
            ax.add_collection(polycol)
        ax.set_xlim(-400, 400)
        axis = ax.axis()
        ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()],
                  axis[0], axis[1], 'k', lw=0.5)
        ax.set_xticks([-400, 0, 400])
        ax.set_xlabel(r'x ($\mu$m)')
        ax.set_ylabel(r'z ($\mu$m)')
        ax.set_title('network populations')

        # for i, (name, population) in enumerate(network.populations.items()):
        j = 1  # counter
        for i, name in enumerate(PSET.populationParameters['me_type']):
            if name in local_me_types:
                population = network.populations[name]
                ax = axes[j]
                # plot electrode contact points
                ax.plot(PSET.electrodeParams['x'], PSET.electrodeParams['z'],
                        'ko', markersize=5)

                # plot cell geometries and synapse locations
                zips = []
                synpos_e = []
                synpos_i = []
                for cell in population.cells:
                    for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                        zips.append(list(zip(x, z)))
                    for idx, syn in zip(cell.synidx, cell.netconsynapses):
                        if hasattr(syn, 'e') and syn.e > -50:
                            synpos_e += [[cell.x[idx].mean(),
                                          cell.z[idx].mean()]]
                        else:
                            synpos_i += [[cell.x[idx].mean(),
                                          cell.z[idx].mean()]]

                polycol = PolyCollection(zips,
                                         edgecolors=colors[i],
                                         linewidths=0.01,
                                         facecolors=colors[i],
                                         label=name,
                                         )
                ax.add_collection(polycol)
                synpos_e = np.array(synpos_e)
                synpos_i = np.array(synpos_i)
                if synpos_e.size > 0:
                    ax.plot(synpos_e[:, 0], synpos_e[:, 1], 'r.',
                            markersize=3, zorder=1)
                if synpos_i.size > 0:
                    ax.plot(synpos_i[:, 0], synpos_i[:, 1], 'b.',
                            markersize=1.5, zorder=2)
                ax.set_xlim(-400, 400)
                axis = ax.axis()
                ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()],
                          axis[0], axis[1], 'k', lw=0.5)
                ax.set_xticks([-400, 0, 400])
                ax.set_xlabel(r'x ($\mu$m)')
                ax.set_title(name)
                j += 1  # counter

        # save figure output
        fig.savefig(os.path.join(
            PSET.OUTPUTPATH,
            f'example_parallel_network_populations_RANK_{RANK}.pdf'),
            bbox_inches='tight')
        plt.close(fig)

    ##########################################################################
    # customary cleanup of object references - the psection() function may not
    # write correct information if NEURON still has object references in memory
    # even if Python references has been deleted. It will also allow the script
    # to be run in successive fashion.
    ##########################################################################
    network.pc.gid_clear()  # allows assigning new gids to threads
    for population in network.populations.values():
        for cell in population.cells:
            cell = None
        population.cells = None
        population = None
    network = None
    neuron.h('forall delete_section()')
