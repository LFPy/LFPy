#!/usr/bin/env python
'''
################################################################################
# An LFPy example file showing how cells can be run in parallel using MPI.
# To run using MPI with 4 cpu cores, issue in terminal
# openmpirun -np 4 python example_mpi.py
#
# The example uses mpi4py with openmpi, and do not rely on NEURON's MPI.
################################################################################
'''

import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import LFPy
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile
from mpi4py import MPI

#MPI stuff we're using
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

#Fetch Mainen&Sejnowski 1996 model files
if not os.path.isfile(join('cells', 'cells', 'j4a.hoc')) and RANK == 0:

    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=2488&a=23&mime=application/zip')
    localFile = open('patdemo.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('patdemo.zip', 'r')
    myzip.extractall('.')
    myzip.close()

if RANK == 0:
    #compile mod files
    os.system('''
              cd cells
              nrnivmodl
              ''')

    #sync threads
COMM.Barrier()
    
#os.system('nrnivmodl')
LFPy.cell.neuron.load_mechanisms('cells')

#set one global seed, ensure all randomizations are set on RANK 0 in script!
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
        
        POPULATION_SIZE :       int, number of cells
        cellParameters :        dict
        populationParameters :  dict
        electrodeParameters :   dict
        synapseParameters :     dict
        stationaryGammaArgs :   dict
        
        '''
        self.POPULATION_SIZE = POPULATION_SIZE
        self.cellParameters = cellParameters
        self.populationParameters = populationParameters
        self.electrodeParameters = electrodeParameters
        self.synapseParameters = synapseParameters
        self.stationaryGammaArgs = stationaryGammaArgs
        
        #get synaptic times and cell positions, rotations, store in self-object
        self.synapseTimes = self.drawRandSpikeTimes()
        self.cellPositions = self.drawRandCellPositions()
        self.cellRotations = self.drawRandCellRotations()
        
    def run(self):
        '''execute the proper simulation and collect simulation results'''
        #produce simulation results on each RANK
        self.results = self.distribute_cellsims()
        
        #superimpose local LFPs on every RANK, then sum using MPI to RANK 0
        self.LFP = []
        for key, value in list(self.results.items()):
            self.LFP.append(value['LFP'])
        self.LFP = np.array(self.LFP).sum(axis=0)
        self.LFP = COMM.reduce(self.LFP)        #LFP is None on all but RANK 0
        
        #collect all simulation results on RANK 0, including single cell LFP
        if RANK == 0:
            for i in range(1, SIZE):
                result = COMM.recv(source=MPI.ANY_SOURCE) #receive from ANY rank
                self.results.update(result)     #collect
        else:
            COMM.send(self.results, dest=0)     #send to RANK 0
            self.results = None                 #results only exist on RANK 0
            
        COMM.Barrier()  #sync MPI threads
    
    def distribute_cellsims(self):
        '''Will distribute and run cell simulations across ranks'''
        #start unique cell simulation on every RANK,
        #and store the electrode and cell objects in dicts indexed by cellindex
        results = {}
        for cellindex in range(self.POPULATION_SIZE):
            if divmod(cellindex, SIZE)[1] == RANK:
                results.update({cellindex : self.cellsim(cellindex)})
        return results

    def drawRandSpikeTimes(self):
        '''draw and distribute some spike times for each cell in population'''
        if RANK == 0:
            RandSpikeTimes = []
            for cellindex in range(self.POPULATION_SIZE):
                sptimes = LFPy.inputgenerators.stationary_gamma(
                    self.cellParameters['tstartms'],
                    self.cellParameters['tstopms'],
                    tmin=self.cellParameters['tstartms'],
                    **self.stationaryGammaArgs)
                RandSpikeTimes.append(sptimes)
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
        #create extracellular electrode object
        electrode = LFPy.RecExtElectrode(**self.electrodeParameters)
        
        #Initialize cell instance, using the LFPy.Cell class
        cell = LFPy.Cell(**self.cellParameters)
        #set the position of midpoint in soma
        cell.set_pos(xpos = self.cellPositions[cellindex, 0],
                     ypos = self.cellPositions[cellindex, 1],
                     zpos = self.cellPositions[cellindex, 2])
        #rotate the morphology
        cell.set_rotation(z = self.cellRotations[cellindex])
        
        #attach synapse with parameters and set spike time
        synapse = LFPy.Synapse(cell, **self.synapseParameters)
        synapse.set_spike_times(self.synapseTimes[cellindex])
        
        #perform NEURON simulation, results saved as attributes in cell
        cell.simulate(electrode = electrode)
        
        #return dict with primary results from simulation
        return {'LFP' : electrode.LFP, 'somav' : cell.somav}

    def plotstuff(self):
        '''plot LFPs and somatraces'''
                
        if RANK == 0:
            fig = plt.figure(figsize=(12, 8))
            
            ax = fig.add_axes([0.05, 0.0, 0.45, 1.0],
                        aspect='equal', frameon=False,
                        xticks=[], xticklabels=[], yticks=[], yticklabels=[])
            for cellindex in range(self.POPULATION_SIZE):
                cell = LFPy.Cell(**self.cellParameters)
                cell.set_pos(xpos = self.cellPositions[cellindex, 0],
                     ypos = self.cellPositions[cellindex, 1],
                     zpos = self.cellPositions[cellindex, 2])
                cell.set_rotation(z = self.cellRotations[cellindex])

                zips = []
                for x, z in cell.get_idx_polygons():
                    zips.append(list(zip(x, z)))
                
                polycol = PolyCollection(zips,
                                edgecolors='none',
                                facecolors='bgrcmykbgrcmykbgrcmyk'[cellindex],
                                zorder = self.cellPositions[cellindex, 1])
       
                ax.add_collection(polycol)

            ax.plot(self.electrodeParameters['x'],
                    self.electrodeParameters['z'],
                    marker='o', color='g', clip_on=False, zorder=0)
            
            ax = fig.add_axes([0.5, 0.55, 0.40, 0.4])
            for key, value in list(self.results.items()):
                tvec = np.arange(value['somav'].size) * \
                                        self.cellParameters['timeres_python']
                ax.plot(tvec, value['somav'],
                        label = 'cell %i' % key)
            leg = ax.legend()
            #ax.set_xlabel('time (ms)')
            ax.set_ylabel('$V_{soma}$ (mV)')
            ax.set_title('somatic potentials')
            
            ax = fig.add_axes([0.5, 0.075, 0.40, 0.4])
            cax = fig.add_axes([0.91, 0.075, 0.02, 0.40])
            im = ax.pcolormesh(tvec, self.electrodeParameters['z'], self.LFP,
                           cmap='spectral_r',
                           vmin = -abs(self.LFP).max(),
                           vmax = abs(self.LFP).max())
            ax.axis(ax.axis('tight'))
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('LFP (mV)')
            ax.set_title('superimposed LFP')
            ax.set_xlabel('time (ms)')
            ax.set_ylabel('$z$ ($\mu$m)')
            
            fig.savefig('example_mpi.pdf', dpi=300)
        

if __name__ == '__main__':
    ########### PARAMETERS ##################
    #define cell parameters used as input to cell-class
    cellParameters = {
        'morphology' : join('morphologies', 'L5_Mainen96_wAxon_LFPy.hoc'),
        'rm' : 30000,               # membrane resistance
        'cm' : 1.0,                 # membrane capacitance
        'Ra' : 150,                 # axial resistance
        'v_init' : -65,             # initial crossmembrane potential
        'e_pas' : -65,              # reversal potential passive mechs
        'passive' : True,           # switch on passive mechs
        'nsegs_method' : 'lambda_f',# method for setting number of segments,
        'lambda_f' : 10,            # segments are isopotential at frequency
        'timeres_NEURON' : 2**-3,   # dt of LFP and NEURON simulation.
        'timeres_python' : 2**-3,
        'tstartms' : -100,          #start time, recorders start at t=0
        'tstopms' : 200,            #stop time of simulation
        'custom_code'  : ['active_declarations_example3.hoc'], #active decl.
    }
    
    #Synaptic parameters, corresponding to a NetCon synapse built into NEURON
    synapseParameters = {
        'idx' : 0,               # insert synapse on index "0", the soma
        'e' : 0.,                # reversal potential of synapse
        'syntype' : 'Exp2Syn',   # conductance based double-exponential synapse
        'tau1' : 1.0,            # Time constant, rise
        'tau2' : 1.0,            # Time constant, decay
        'weight' : 0.002,        # Synaptic weight
        'record_current' : False,# disable synapse current recording
    }
    
    #parameter args for LFPy.inputgenerators.stationary_gamma()
    stationaryGammaArgs = {
        'k' : 0.25,               #shape parameter
        'theta' : 12,             #"rate" parameter
    }
    
    # Define electrode geometry corresponding to a laminar electrode, where
    # contact points have a radius r, surface normal vectors N, and LFP
    # calculated as the average LFP in n random points on each contact:
    X, Y, Z = np.mgrid[0:1, 0:1, -500:1001:50]
    N = np.zeros((X.size, 3))
    for i in range(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
    # put parameters in dictionary
    electrodeParameters = {
        'sigma' : 0.3,              # Extracellular potential
        'x' : X.flatten(),      # x,y,z-coordinates of electrode contacts
        'y' : Y.flatten(),
        'z' : Z.flatten(),
        'n' : 50,
        'r' : 5,
        'N' : N,
    }
    
    #the number of cells in the population
    POPULATION_SIZE = 4
    
    #will draw random cell locations within cylinder constraints:
    populationParameters = {
        'radius' : 20,
        'zmin' : -200,
        'zmax' : 200,
    }
    
    ########## INITIALIZE POPULATION ###########################################
    population = Population(POPULATION_SIZE,
                     cellParameters,
                     populationParameters,
                     electrodeParameters,
                     synapseParameters,
                     stationaryGammaArgs,)
    population.run()
    population.plotstuff()
    plt.show()

