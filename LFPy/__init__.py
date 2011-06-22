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
  * Cell - The neuron object itself laying on top of NEURON
  * Synapse - Convenience class for inserting synapses onto Cell objects
  * PointProcess - Convenience class for inserting electrodes onto Cell objects
  * Electrode - Class for performing simulations of extracellular potentials

:Modules:
  * lfpcalc - functions used by Electrode class
  * tools - some convenient functions
  * inputgenerators - functions for synaptic input time generation
'''

from cell import Cell
from synapse import Synapse, PointProcess
from electrode import Electrode

import tools
import inputgenerators
