#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate usage of LFPy.NetworkPopulation with a ball-and-stick type
morphology with active HH channels inserted in the soma and passive-leak
channels distributed throughout the apical stick. The corresponding morphology
and template-specification is in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

Execution:

    python example_network_population.py

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
# import modules
from mpi4py.MPI import COMM_WORLD as COMM
from LFPy import NetworkPopulation, NetworkCell

# class NetworkCell parameters:
cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    delete_sections=False,
)
# class NetworkPopulation parameters:
populationParameters = dict(
    Cell=NetworkCell,
    cell_args=cellParameters,
    pop_args=dict(
        radius=100.,
        loc=0.,
        scale=20.),
    rotation_args=dict(x=0., y=0.),
)
# create population:
population = NetworkPopulation(
    first_gid=0, name='E',
    **populationParameters
)
# print out some info:
for cell in population.cells:
    print('RANK {}; pop {}; gid {}; cell {}'.format(
        COMM.Get_rank(), population.name,
        cell.gid, cell))
