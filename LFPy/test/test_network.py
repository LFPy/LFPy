#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

from __future__ import division
import os
import unittest
import numpy as np
import LFPy
import neuron

class testNetworkPopulation(unittest.TestCase):
    """

    """
    def test_NetworkPopulation_00(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstopms=100,
            delete_sections=False,
        )
        
        populationParameters = dict(
            CWD=None,
            CELLPATH=None,
            Cell=LFPy.NetworkCell,
            cell_args = cellParameters,
            pop_args = dict(
                radius=100,
                loc=0.,
                scale=20.),
            rotation_args = dict(x=0, y=0),
            POP_SIZE = 4,
            name = 'ball_and_sticks',
            OUTPUTPATH='tmp_testNetworkPopulation'
        )
        
        population = LFPy.NetworkPopulation(**populationParameters)
    
        self.assertTrue(len(population.cells) == population.POP_SIZE)
        for cell, soma_pos, gid in zip(population.cells, population.soma_pos, population.gids):
            self.assertTrue(type(cell) is LFPy.NetworkCell)
            self.assertTrue((cell.somapos[0] == soma_pos['xpos']) &
                            (cell.somapos[1] == soma_pos['ypos']) &
                            (cell.somapos[2] == soma_pos['zpos']))
            self.assertEqual(cell.gid, gid)
            self.assertTrue(np.sqrt(soma_pos['xpos']**2 + soma_pos['ypos']**2 + soma_pos['zpos']**2) <= 100.)
        np.testing.assert_equal(population.gids, np.arange(4))
        
        
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')



class testNetwork(unittest.TestCase):
    """

    """
    pass
