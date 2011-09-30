#!/usr/bin/env python
'''Initialization of LFPy, a module for simulating extracellular potentials.

Developed by: 
Henrik Linden, Espen Hagen, Eivind Norheim, Szymon Leski

Group of Computational Neuroscience (compneuro.umb.no),
Department of Mathematical Sciences and Technology,
Norwegian University of Life Sciences.

Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.

:Classes:
  * Cell - The pythonic neuron object itself laying on top of NEURON
  * CellWithElectrode - Subclass of Cell that calculate LFP at every fadvance
  * PointProcessSynapse - Convenience class for inserting synapses onto Cell objects
  * PointProcessElectrode - Convenience class for inserting electrodes onto Cell objects
  * Electrode - Class for performing simulations of extracellular potentials

:Modules:
  * lfpcalc - functions used by Electrode class
  * tools - some convenient functions
  * inputgenerators - functions for synaptic input time generation
'''

from cell import Cell
from pointprocess import Synapse, PointProcess, \
    PointProcessSynapse, PointProcessElectrode
from electrode import Electrode, ElectrodeThreaded, ElectrodeSetup
from cellwithelectrode import CellWithElectrode

import tools
import inputgenerators
