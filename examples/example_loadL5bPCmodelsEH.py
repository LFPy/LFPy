#!/usr/bin/env python
'''Example script loading and execiting a simulation with the
Hay et al. 2011 L5b-pyramidal cell model, which is implemented by default
using templates.

This script assume that the model files is downloaded and unzipped inside
this folder from ModelDB:
http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=139653

The mod-files inside /L5bPCmodelsEH/mod/ must be compiled using nrnivmodl.

Note that LFPy can only deal with one cell at the time, creating several
cell objects will slow everything down, but each cell *should* get the correct
cell responses.
'''

import LFPy
import neuron
import matplotlib.pyplot as plt

#remove cells from previous script executions
for sec in neuron.h.allsec():
    neuron.h.delete_section()

#load the corresponding neuron mech's, but run nrnivmodl here first!
neuron.load_mechanisms('L5bPCmodelsEH/mod/')

#cell parameters with additional arguments for the TemplateCell-class.
#Note that 'morphology' is required, even though it is loaded through
#'templateargs'!
#Reason is LFPy looks for a default rotation .rot-file.
cellParams = {
    'morphology'    : 'L5bPCmodelsEH/morphologies/cell1.asc',
    'templatefile'  : ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
                       'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
    'templatename'  : 'L5PCtemplate',
    'templateargs'  : 'L5bPCmodelsEH/morphologies/cell1.asc',
    'passive' :     False,
    'nsegs_method' : None,
    'v_init' : -80,
    'tstartms' : 0,
    'tstopms' : 3000,
    'timeres_NEURON' : 2**-3,
    'timeres_python' : 2**-3,
    'verbose' : True,
    'extracellular' : False,
}

#Use the TemplateCell-class to create the cell
cell = LFPy.TemplateCell(**cellParams)

#some stimuli
PointProcParams = {
    'idx' : 0,
    'record_current' : False,
    'pptype' : 'IClamp',
    'amp' : 0.793,
    'dur' : 2000,
    'delay' : 700,
}

pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

#run simulation
cell.simulate(rec_variables = [])

#plot response
plt.plot(cell.tvec, cell.somav)
plt.show()