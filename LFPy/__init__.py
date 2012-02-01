#!/usr/bin/env python
'''Initialization of LFPy, a module for simulating extracellular potentials.

Developed by: 
Henrik Linden, Espen Hagen, Eivind Norheim, Szymon Leski

Group of Computational Neuroscience (compneuro.umb.no),
Department of Mathematical Sciences and Technology,
Norwegian University of Life Sciences.

Copyright (C) 2012 Computational Neuroscience Group, UMB.
All rights reserved.

:Classes:
  * Cell - The pythonic neuron object itself laying on top of NEURON
  * Synapse - Convenience class for inserting synapses onto Cell objects
  * StimIntraElectrode - Convenience class for inserting electrodes onto Cell objects
  * RecExtElectrode - Class for performing simulations of extracellular potentials
  * RecExtElectrdoeThreaded - same as above using multiprocessing to distribute contact points across cores

:Modules:
  * lfpcalc - functions used by RecExtElectrode class
  * tools - some convenient functions
  * inputgenerators - functions for synaptic input time generation
'''

from pointprocess import Synapse, PointProcess, StimIntraElectrode
from recextelectrode import RecExtElectrode, RecExtElectrodeSetup
from cell import Cell
from recextelectrodethreaded import RecExtElectrodeThreaded

import lfpcalc
import tools
import inputgenerators
import run_simulation