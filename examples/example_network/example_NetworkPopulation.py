#!/usr/bin/env python
from mpi4py.MPI import COMM_WORLD as COMM
from LFPy import NetworkPopulation, NetworkCell
# class NetworkCell parameters
cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    passive=False,
    delete_sections=False,
)
# class NetworkPopulation parameters
populationParameters = dict(
    Cell=NetworkCell,
    cell_args = cellParameters,
    pop_args = dict(
        radius=100,
        loc=0.,
        scale=20.),
    rotation_args = dict(x=0, y=0),
)
# create population instance
population = NetworkPopulation(
    first_gid=0, name='E',
    **populationParameters
    )
for cell in population.cells:
    print('RANK {}; pop {}; gid {}; cell {}'.format(
        COMM.Get_rank(), population.name,
        cell.gid, cell))