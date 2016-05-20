#!/usr/bin/env python
'''
################################################################################
#
# This is an example scripts using LFPy with an active cell model adapted from
# Mainen and Sejnowski, Nature 1996, for the original files, see
# http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=2488
#
# This scripts is set up to use the model, where the active conductances are set
# in the file "active_declarations_example2.hoc", and uses the mechanisms from
# the .mod-files provided here. For this example to work, run "nrnivmodl" in
# this folder to compile these mechanisms
# (i.e. /$PATHTONEURON/nrn/x86_64/bin/nrnivmodl).
#
# A single excitatory synapse drive the neuron into producing a single action-
# potential, and the local field potential are calculated on a dense 2D-grid
# on around the soma.
#
################################################################################
'''


#import some plotting stuff and the LFPy-module
import os
from os.path import join
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile
import LFPy
import numpy as np
import matplotlib.pyplot as plt

#Fetch Mainen&Sejnowski 1996 model files
if not os.path.isfile(join('cells', 'cells', 'j4a.hoc')):
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=2488&a=23&mime=application/zip')
    localFile = open('patdemo.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('patdemo.zip', 'r')
    myzip.extractall('.')
    myzip.close()

#compile mod files every time, because of incompatibility with Hay2011 files:
os.system('''
          cd cells
          nrnivmodl
          ''')
#os.system('nrnivmodl')
LFPy.cell.neuron.load_mechanisms('cells')

################################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
################################################################################

#define cell parameters used as input to cell-class
cellParameters = {
    'morphology' : 'morphologies/L5_Mainen96_wAxon_LFPy.hoc',
    'rm' : 30000,               # membrane resistance
    'cm' : 1.0,                 # membrane capacitance
    'Ra' : 150,                 # axial resistance
    'v_init' : -65,             # initial crossmembrane potential
    'e_pas' : -65,              # reversal potential passive mechs
    'passive' : True,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 500,           # segments are isopotential at this frequency
    'timeres_NEURON' : 2**-5,   # dt of LFP and NEURON simulation.
    'timeres_python' : 2**-5,
    'tstartms' : -10,           #start time, recorders start at t=0
    'tstopms' : 10,             #stop time of simulation
    'custom_code'  : ['active_declarations_example2.hoc'], # will run this file
}

#Synaptic parameters, corresponding to a NetCon synapse built into NEURON
synapseParameters = {
    'idx' : 0,               # insert synapse on index "0", the soma
    'e' : 0.,                # reversal potential of synapse
    'syntype' : 'Exp2Syn',   # conductance based double-exponential synapse
    'tau1' : 1.0,            # Time constant, rise
    'tau2' : 1.0,            # Time constant, decay
    'weight' : 0.03,         # Synaptic weight
    'record_current' : True, # Will enable synapse current recording
}

#Generate the grid in xz-plane over which we calculate local field potentials
X, Z = np.mgrid[-5:6, -5:6] * 10
Y = np.zeros(X.shape)

#define parameters for extracellular recording electrode, using optional method
electrodeParameters = {
    'sigma' : 0.3,              # extracellular conductivity
    'x' : X.flatten(),        # x,y,z-coordinates of contact points
    'y' : Y.flatten(),
    'z' : Z.flatten(),
    'method' : 'som_as_point',  #treat soma segment as sphere source
}

################################################################################
# Main simulation procedure, setting up extracellular electrode, cell, synapse
################################################################################

#create extracellular electrode object
electrode = LFPy.RecExtElectrode(**electrodeParameters)

#Initialize cell instance, using the LFPy.Cell class
cell = LFPy.Cell(**cellParameters)
#set the position of midpoint in soma to Origo (not needed, this is the default)
cell.set_pos(xpos = 0, ypos = 0, zpos = 0)
#rotate the morphology 90 degrees around z-axis
cell.set_rotation(z = np.pi/2)

#attach synapse with parameters and set spike time
synapse = LFPy.Synapse(cell, **synapseParameters)
synapse.set_spike_times(np.array([1]))

#perform NEURON simulation, results saved as attributes in the cell instance
cell.simulate(electrode = electrode, rec_isyn=True)

# Plotting of simulation results:
from example_suppl import plot_ex2
fig = plot_ex2(cell, electrode)
#Optional: saving the figure
fig.savefig('example5.pdf', dpi=300)

plt.show()


