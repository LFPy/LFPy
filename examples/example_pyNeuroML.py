#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example plot for LFPy: Usage of LEMS/NeuroML2 files with LFPy

This example requires the pyNeuroML module 
(https://github.com/NeuroML/pyNeuroML) from the NeuroML Initiative
(https://www.neuroml.org). For most, it should suffice to issue

    pip install pyNeuroML --user

to install pyNeuroML. 

Execution:

    python example_pyNeuroML.py

Copyright (C) 2018 Computational Neuroscience Group, NMBU.

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
from matplotlib.collections import PolyCollection
import os
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import ssl
import numpy as np
import LFPy
import neuron

# define different PATHs
NEUROMLPATH = 'NeuroML2'
FPATH = os.path.join(NEUROMLPATH, 'LEMSexamples')
MODEL = 'LEMS_NML2_Ex25_MultiComp.xml'

# clone into NeuroML2 repo to get LEMS/NeuroML format example files at
# branch 'development'
if not os.path.isdir(NEUROMLPATH):
    os.system('git clone --depth=1 -b development --single-branch https://github.com/NeuroML/NeuroML2.git')

# convert LEMS/NeuroML file to NEURON format
cwd = os.getcwd()
os.chdir(FPATH)
try:
    status = os.system('pynml {} -neuron'.format(MODEL))    
    if status != 0:
        raise Exception
except Exception:
    raise Exception('Something went wrong converting from LEMS. Is pyNeuroML correctly installed?')

# compile NMODL file output (on linux/unix/macos)
if "win32" in sys.platform:
    warn("no autompile of NMODL (.mod) files on Windows.\n" 
         + "Run mknrndll from NEURON bash in the folder {} and rerun example script".format(FPATH))
    if not os.getcwd() in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll("nrnmech.dll")
    neuron.nrn_dll_loaded.append(os.getcwd())
else:
    os.system('nrnivmodl')
    neuron.load_mechanisms('.')

# return to working directory    
os.chdir(cwd)

# create LFPy.TemplateCell object
cell = LFPy.TemplateCell(morphology=os.path.join(FPATH, 'MultiCompCell.hoc'), 
                         templatefile=os.path.join(FPATH, 'MultiCompCell.hoc'), 
                         templatename='MultiCompCell',
                         tstop=100, v_init=-65)

# connect synapse, set activation times
syn = LFPy.Synapse(cell, idx=57, syntype='ExpSyn', tau=1., weight=.0001,
                   record_current=True)
syn.set_spike_times(np.arange(1, 5)*20.)

# run simulation
cell.simulate()

# plot cell geometry, synapse location, synapse current, somatic response
fig = plt.figure()
ax = fig.add_subplot(121, aspect='equal')
zips = []
for x, y in cell.get_idx_polygons(projection='xy'):
    zips.append(list(zip(x, y)))
polycol = PolyCollection(zips,
                         edgecolors='k',
                         facecolors='0.5')
ax.add_collection(polycol)
ax.plot(cell.xmid[syn.idx], cell.ymid[syn.idx], 'ro')
ax.set_xlabel('$x$ ($\mu$m)')
ax.set_ylabel('$y$ ($\mu$m)')

ax = fig.add_subplot(222)
ax.plot(cell.tvec, syn.i, 'r')
ax.set_ylabel(r'$I_\mathrm{syn}(t)$ (nA)')

ax = fig.add_subplot(224)
ax.plot(cell.tvec, cell.somav, 'k')
ax.set_ylabel('$V_\mathrm{soma}(t)$ (mV)')
ax.set_xlabel('time (ms)')

plt.show()
