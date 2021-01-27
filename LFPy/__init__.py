#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialization of LFPy, a Python module for simulating
extracellular potentials.

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
  * ``Cell`` - object built on top of NEURON representing biological neuron
  * ``TemplateCell`` - Similar to ``Cell``, but for models using cell templates
  * ``NetworkCell`` - Similar to ``TemplateCell`` with some attributes and
    methods for spike communication between parallel RANKs
  * ``PointProcess`` - Parent class of ``Synapse`` and ``StimIntElectrode``
  * ``Synapse`` - Convenience class for inserting synapses onto ``Cell``
    objects
  * ``StimIntElectrode`` - Convenience class for inserting stimulating
    electrodes into ``Cell`` objects
  * ``Network`` - Class for creating distributed populations of cells and
    handling connections between cells in populations
  * ``NetworkPopulation`` - Class representing group of ``Cell`` objects
    distributed across MPI RANKs
  * ``RecExtElectrode`` - Class for setup of simulations of extracellular
    potentials
  * ``RecMEAElectrode`` - Class for setup of simulations of in vitro (slice)
    extracellular potentials
  * ``PointSourcePotential`` - Base forward-model for extracellular potentials
    assuming point current sources in conductive media
  * ``LineSourcePotential`` - Base forward-model for extracellular potentials
    assuming line current sources in conductive media
  * ``OneSphereVolumeConductor`` - For computing extracellular potentials
    within and outside a homogeneous sphere
  * ``CurrentDipoleMoment`` - For computing the current dipole moment,
  * ``FourSphereVolumeConductor`` - For computing extracellular potentials in
    four-sphere head model (brain, CSF, skull, scalp)
  * ``InfiniteVolumeConductor`` - To compute extracellular potentials with
    current dipoles in infinite volume conductor
  * ``MEG`` - Class for computing magnetic field from current dipole moment
:Modules:
  * ``lfpcalc`` - Misc. functions used by RecExtElectrode class
  * ``tools`` - Some convenient functions
  * ``inputgenerators`` - Functions for synaptic input time generation
  * ``eegmegcalc`` - Classes for calculating current dipole moment vector
    P and P_tot from currents and distances.
  * ``run_simulations`` - Functions to run NEURON simulations
"""

from .version import version as __version__
from .pointprocess import Synapse, PointProcess, StimIntElectrode
from lfpykit import RecExtElectrode, RecMEAElectrode, CurrentDipoleMoment, \
    PointSourcePotential, LineSourcePotential, OneSphereVolumeConductor, \
    LaminarCurrentSourceDensity, VolumetricCurrentSourceDensity
from .cell import Cell
from .templatecell import TemplateCell
from .network import NetworkCell, NetworkPopulation, Network
from .test import _test as run_tests
from .eegmegcalc import FourSphereVolumeConductor, InfiniteVolumeConductor, \
    MEG, NYHeadModel
from lfpykit import lfpcalc
from . import tools
from . import inputgenerators
from . import run_simulation
