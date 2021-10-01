#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An LFPy example file showing how cells can be run in parallel using MPI.
To run using MPI with 4 cpu cores, issue in terminal
openmpirun -np 4 python example_mpi.py

The example uses mpi4py with openmpi, and do not rely on NEURON's MPI
implementation.

Execution:

    <mpiexec> -n <processes> python example_mpi.py

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

import os
from os.path import join
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import LFPy
from LFPy.inputgenerators import get_activation_times_from_distribution
import neuron
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
import zipfile
import ssl
from warnings import warn
from mpi4py import MPI

# MPI stuff we're using
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

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

# compile mod files every time, because of incompatibility with Hay2011 files:
if "win32" in sys.platform:
    pth = "cells"
    warn("no autompile of NMODL (.mod) files on Windows. "
         + "Run mknrndll from NEURON bash in the folder cells and "
         + "rerun example script")
    if pth not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth + "/nrnmech.dll")
    neuron.nrn_dll_loaded.append(pth)
else:
    if RANK == 0:
        os.system('''
                  cd cells
                  nrnivmodl
                  ''')
    COMM.Barrier()
    neuron.load_mechanisms('cells')


# set one global seed, ensure all randomizations are set on RANK 0 in script!
np.random.seed(12345)


class Population:
    '''prototype population class'''

    def __init__(self, POPULATION_SIZE,
                 cellParameters,
                 populationParameters,
                 electrodeParameters,
                 synapseParameters,
                 stationaryGammaArgs,
                 ):
        '''
        class initialization

        POPULATION_SIZE:       int, number of cells
        cellParameters:        dict
        populationParameters:  dict
        electrodeParameters:   dict
        synapseParameters:     dict
        stationaryGammaArgs:   dict

        '''
        self.POPULATION_SIZE = POPULATION_SIZE
        self.cellParameters = cellParameters
        self.populationParameters = populationParameters
        self.electrodeParameters = electrodeParameters
        self.synapseParameters = synapseParameters
        self.stationaryGammaArgs = stationaryGammaArgs

        # get synaptic times and cell positions, rotations, store in
        # self-object
        self.synapseTimes = self.drawRandSpikeTimes()
        self.cellPositions = self.drawRandCellPositions()
        self.cellRotations = self.drawRandCellRotations()

    def run(self):
        '''execute the proper simulation and collect simulation results'''
        # produce simulation results on each RANK
        self.results = self.distribute_cellsims()

        # superimpose local LFPs on every RANK, then sum using MPI to RANK 0
        self.LFP = []
        for key, value in list(self.results.items()):
            self.LFP.append(value['LFP'])
        self.LFP = np.array(self.LFP).sum(axis=0)
        self.LFP = COMM.reduce(self.LFP)  # LFP is None on all but RANK 0

        # collect all simulation results on RANK 0, including single cell LFP
        if RANK == 0:
            for i in range(1, SIZE):
                result = COMM.recv(
                    source=MPI.ANY_SOURCE)  # receive from ANY rank
                self.results.update(result)  # collect
        else:
            COMM.send(self.results, dest=0)  # send to RANK 0
            self.results = None  # results only exist on RANK 0

        COMM.Barrier()  # sync MPI threads

    def distribute_cellsims(self):
        '''Will distribute and run cell simulations across ranks'''
        # start unique cell simulation on every RANK,
        # and store the electrode and cell objects in dicts indexed by
        # cellindex
        results = {}
        for cellindex in range(self.POPULATION_SIZE):
            if divmod(cellindex, SIZE)[1] == RANK:
                results.update({cellindex: self.cellsim(cellindex)})
        return results

    def drawRandSpikeTimes(self):
        '''draw and distribute some spike times for each cell in population'''
        if RANK == 0:
            RandSpikeTimes = []
            for cellindex in range(self.POPULATION_SIZE):
                sptimes = get_activation_times_from_distribution(
                    n=1, tstart=0, tstop=cellParameters['tstop'],
                    distribution=st.gamma,
                    rvs_args=self.stationaryGammaArgs
                )
                RandSpikeTimes.append(sptimes[0])
        else:
            RandSpikeTimes = None
        return COMM.bcast(RandSpikeTimes, root=0)

    def drawRandCellPositions(self):
        '''draw and distribute some random cell positions within a
        cylinder constraints'''
        if RANK == 0:
            cellPositions = []
            for cellindex in range(self.POPULATION_SIZE):
                r = np.sqrt(np.random.rand()) * \
                    self.populationParameters['radius']
                theta = np.random.rand() * 2 * np.pi
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                z = np.random.rand() * (self.populationParameters['zmax'] -
                                        self.populationParameters['zmin']) + \
                    self.populationParameters['zmin']
                cellPositions.append([x, y, z])
            cellPositions = np.array(cellPositions)
        else:
            cellPositions = None
        return COMM.bcast(cellPositions, root=0)

    def drawRandCellRotations(self):
        '''draw and distribute random cell rotations for all cells in population
        '''
        if RANK == 0:
            cellRotations = np.random.rand(self.POPULATION_SIZE) * np.pi * 2
        else:
            cellRotations = None
        return COMM.bcast(cellRotations, root=0)

    def cellsim(self, cellindex):
        '''main cell- and LFP simulation procedure'''

        # Initialize cell instance, using the LFPy.Cell class
        cell = LFPy.Cell(**self.cellParameters)
        # set the position of midpoint in soma
        cell.set_pos(x=self.cellPositions[cellindex, 0],
                     y=self.cellPositions[cellindex, 1],
                     z=self.cellPositions[cellindex, 2])
        # rotate the morphology
        cell.set_rotation(z=self.cellRotations[cellindex])

        # attach synapse with parameters and set spike time
        synapse = LFPy.Synapse(cell, **self.synapseParameters)
        synapse.set_spike_times(self.synapseTimes[cellindex])

        # create extracellular electrode object
        electrode = LFPy.RecExtElectrode(cell, **self.electrodeParameters)

        # perform NEURON simulation, results saved as attributes in cell
        cell.simulate(probes=[electrode])

        # return dict with primary results from simulation
        return {'LFP': electrode.data, 'somav': cell.somav}

    def plotstuff(self):
        '''plot LFPs and somatraces'''

        if RANK == 0:
            fig = plt.figure(figsize=(12, 8))

            ax = fig.add_axes([0.05, 0.0, 0.45, 1.0],
                              aspect='equal', frameon=False,
                              xticks=[], xticklabels=[],
                              yticks=[], yticklabels=[])
            for cellindex in range(self.POPULATION_SIZE):
                cell = LFPy.Cell(**self.cellParameters)
                cell.set_pos(x=self.cellPositions[cellindex, 0],
                             y=self.cellPositions[cellindex, 1],
                             z=self.cellPositions[cellindex, 2])
                cell.set_rotation(z=self.cellRotations[cellindex])

                zips = []
                for x, z in cell.get_idx_polygons():
                    zips.append(list(zip(x, z)))

                polycol = PolyCollection(zips,
                                         edgecolors='none',
                                         facecolors='bgrcmykbgrcmykbgrcmyk'[
                                             cellindex],
                                         zorder=self.cellPositions[cellindex,
                                                                   1])

                ax.add_collection(polycol)

            ax.plot(self.electrodeParameters['x'],
                    self.electrodeParameters['z'],
                    marker='o', color='g', clip_on=False, zorder=0)

            ax = fig.add_axes([0.5, 0.55, 0.40, 0.4])
            for key, value in list(self.results.items()):
                tvec = np.arange(value['somav'].size) * \
                    self.cellParameters['dt']
                ax.plot(tvec, value['somav'],
                        label='cell %i' % key)
            ax.legend()
            ax.set_ylabel('$V_{soma}$ (mV)')
            ax.set_title('somatic potentials')

            ax = fig.add_axes([0.5, 0.075, 0.40, 0.4])
            cax = fig.add_axes([0.91, 0.075, 0.02, 0.40])
            im = ax.pcolormesh(tvec, self.electrodeParameters['z'], self.LFP,
                               cmap='PRGn',
                               vmin=-self.LFP.std() * 3,
                               vmax=self.LFP.std() * 3,
                               shading='auto')
            ax.axis(ax.axis('tight'))
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('extracellular potential (mV)')
            ax.set_title('superimposed extracellular potential')
            ax.set_xlabel('time (ms)')
            ax.set_ylabel(r'$z$ ($\mu$m)')

            fig.savefig('example_mpi.pdf', dpi=300)
            plt.show()


if __name__ == '__main__':
    # ########## PARAMETERS ##################
    # define cell parameters used as input to cell-class
    cellParameters = {
        'morphology': join('morphologies', 'L5_Mainen96_wAxon_LFPy.hoc'),
        'cm': 1.0,                 # membrane capacitance
        'Ra': 150,                 # axial resistance
        'v_init': -65,             # initial crossmembrane potential
        'passive': True,           # switch on passive mechs
        # passive params
        'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 10,            # segments are isopotential at frequency
        'dt': 2**-3,               # dt of LFP and NEURON simulation.
        'tstart': -100,  # start time, recorders start at t=0
        'tstop': 200,  # stop time of simulation
        'custom_code': ['active_declarations_example3.hoc'],  # active decl.
    }

    # Synaptic parameters, corresponding to a NetCon synapse built into NEURON
    synapseParameters = {
        'idx': 0,               # insert synapse on index "0", the soma
        'e': 0.,                # reversal potential of synapse
        'syntype': 'Exp2Syn',   # conductance based double-exponential synapse
        'tau1': 1.0,            # Time constant, rise
        'tau2': 1.0,            # Time constant, decay
        'weight': 0.002,        # Synaptic weight
        'record_current': False,  # disable synapse current recording
    }

    # parameter args for LFPy.inputgenerators.stationary_gamma()
    stationaryGammaArgs = {
        'a': 0.25,  # shape parameter
        'scale': 12,  # "rate" parameter
    }

    # Define electrode geometry corresponding to a laminar electrode, where
    # contact points have a radius r, surface normal vectors N, and LFP
    # calculated as the average LFP in n random points on each contact:
    X, Y, Z = np.mgrid[0:1, 0:1, -500:1001:50]
    N = np.zeros((X.size, 3))
    for i in range(N.shape[0]):
        N[i, ] = [1, 0, 0]  # normal unit vec. to contacts
    # put parameters in dictionary
    electrodeParameters = {
        'sigma': 0.3,              # Extracellular potential
        'x': X.flatten(),      # x,y,z-coordinates of electrode contacts
        'y': Y.flatten(),
        'z': Z.flatten(),
        'n': 50,
        'r': 5,
        'N': N,
    }

    # the number of cells in the population
    POPULATION_SIZE = 4

    # will draw random cell locations within cylinder constraints:
    populationParameters = {
        'radius': 20,
        'zmin': -200,
        'zmax': 200,
    }

    # ######### INITIALIZE POPULATION #########################################
    population = Population(POPULATION_SIZE,
                            cellParameters,
                            populationParameters,
                            electrodeParameters,
                            synapseParameters,
                            stationaryGammaArgs,)
    population.run()
    population.plotstuff()
