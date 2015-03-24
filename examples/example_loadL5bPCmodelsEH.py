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
import os
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile



#Fetch Hay et al. 2011 model files
if not os.path.isfile('L5bPCmodelsEH/morphologies/cell1.asc'):
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=139653&a=23&mime=application/zip')
    localFile = open('L5bPCmodelsEH.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('L5bPCmodelsEH.zip', 'r')
    myzip.extractall('.')
    myzip.close()

#compile mod files every time, because of incompatibility with Mainen96 files:
os.system('''
          cd L5bPCmodelsEH/mod/
          nrnivmodl
          ''')
#os.system('nrnivmodl')
LFPy.cell.neuron.load_mechanisms('L5bPCmodelsEH/mod/')

#remove cells from previous script executions
neuron.h('forall delete_section()')

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