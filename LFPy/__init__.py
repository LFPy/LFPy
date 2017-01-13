#!/usr/bin/env python
'''Initialization of LFPy, a module for simulating extracellular potentials.

Group of Computational Neuroscience,
Department of Mathematical Sciences and Technology,
Norwegian University of Life Sciences.

Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

:Classes:
  * Cell - The pythonic neuron object itself laying on top of NEURON
  * Synapse - Convenience class for inserting synapses onto Cell objects
  * StimIntraElectrode - Convenience class for inserting electrodes onto Cell objects
  * RecExtElectrode - Class for performing simulations of extracellular potentials

:Modules:
  * lfpcalc - functions used by RecExtElectrode class
  * tools - some convenient functions
  * inputgenerators - functions for synaptic input time generation
'''

__version__ = "1.1.3"

from .pointprocess import Synapse, PointProcess, StimIntElectrode
from .recextelectrode import RecExtElectrode, RecExtElectrodeSetup
from .cell import Cell
from .templatecell import TemplateCell
from .testing import test

from . import lfpcalc
from . import tools
from . import inputgenerators
from . import run_simulation

