#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialization of LFPy, a Python module for simulating extracellular potentials.

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
  * Cell - The pythonic neuron object itself laying on top of NEURON representing cells
  * TemplateCell - Similar to Cell, but for models using cell templates
  * Synapse - Convenience class for inserting synapses onto Cell objects
  * StimIntElectrode - Convenience class for inserting electrodes onto Cell objects
  * PointProcess - Parent class of Synapse and StimIntElectrode
  * RecExtElectrode - Class for performing simulations of extracellular potentials
  * RecMEAElectrode - Class for performing simulations of in vitro (slice) extracellular potentials
  * Network - Class for creating distributed populations of cells and handling connections between cells in populations
  * NetworkCell - Similar to `TemplateCell` with some attributes and methods for spike communication between parallel RANKs
  * NetworkPopulation - Class representing group of Cell objects distributed across MPI RANKs
  * OneSphereVolumeConductor - For computing extracellular potentials within and outside a homogeneous sphere
  * FourSphereVolumeConductor - For computing extracellular potentials in 4-sphere model (brain, CSF, skull, scalp)
  * InfiniteVolumeConductor - To compute extracellular potentials with current dipoles in infinite volume conductor
  * MEG - Class for computing magnetic field from current dipole moment
:Modules:
  * lfpcalc - Functions used by RecExtElectrode class
  * tools - Some convenient functions
  * inputgenerators - Functions for synaptic input time generation
  * eegcalc - Functions for calculating current dipole moment vector P and P_tot from currents and distances.
  * run_simulations - Functions to run NEURON simulations
"""

from .version import version as __version__

from .pointprocess import Synapse, PointProcess, StimIntElectrode
from .recextelectrode import RecExtElectrode, RecMEAElectrode
from .cell import Cell
from .templatecell import TemplateCell
from .network import NetworkCell, NetworkPopulation, Network
from .test import _test as run_tests
from .eegmegcalc import OneSphereVolumeConductor, FourSphereVolumeConductor, InfiniteVolumeConductor, get_current_dipole_moment, MEG
from . import lfpcalc
from . import tools
from . import inputgenerators
from . import run_simulation
