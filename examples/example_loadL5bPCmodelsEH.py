#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script loading and execiting a simulation with the
Hay et al. 2011 L5b-pyramidal cell model, which is implemented by default
using templates.

This script assume that the model files is downloaded and unzipped inside
this folder from ModelDB:
http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=139653

The mod-files inside /139653/mod/ must be compiled using nrnivmodl.

Note that LFPy can only deal with one cell at the time, creating several
cell objects will slow everything down, but each cell *should* get the correct
cell responses.

Execution:

    python example_loadL5bPCmodelsEH.py

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

import LFPy
import neuron
import matplotlib.pyplot as plt
import os
import sys
from warnings import warn


# Fetch Hay et al. 2011 model files
if not os.path.isfile('139653/morphologies/cell1.asc'):
    # get the model files:
    os.system('git clone https://github.com/ModelDBRepository/139653.git')

# compile mod files every time, because of incompatibility with Mainen96 files:
if "win32" in sys.platform:
    pth = "139653/mod/"
    warn("no autompile of NMODL (.mod) files on Windows."
         + "Run mknrndll from NEURON bash in the folder "
         + "139653/mod and rerun example script")
    if pth not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth + "nrnmech.dll")
    neuron.nrn_dll_loaded.append(pth)
else:
    os.system('''
              cd 139653/mod/
              nrnivmodl
              ''')
    neuron.load_mechanisms('139653/mod/')

# remove cells from previous script executions
neuron.h('forall delete_section()')

# cell parameters with additional arguments for the TemplateCell-class.
# Note that 'morphology' is required, even though it is loaded through
# 'templateargs'!
# Reason is LFPy looks for a default rotation .rot-file.
cellParams = {
    'morphology': '139653/morphologies/cell1.asc',
    'templatefile': ['139653/models/L5PCbiophys3.hoc',
                     '139653/models/L5PCtemplate.hoc'],
    'templatename': 'L5PCtemplate',
    'templateargs': '139653/morphologies/cell1.asc',
    'nsegs_method': None,
    'v_init': -80,
    'tstart': 0,
    'tstop': 3000,
    'dt': 2**-3,
    'verbose': True,
    'extracellular': False,
}

# Use the TemplateCell-class to create the cell
cell = LFPy.TemplateCell(**cellParams)

# some stimuli
PointProcParams = {
    'idx': 0,
    'record_current': False,
    'pptype': 'IClamp',
    'amp': 0.793,
    'dur': 2000,
    'delay': 700,
}

pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

# run simulation
cell.simulate(rec_variables=[])

# plot response
plt.plot(cell.tvec, cell.somav)
plt.show()
