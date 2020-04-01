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


import os
import unittest
import numpy as np
import LFPy
import neuron
import h5py
import scipy.signal as ss

class testNetworkPopulation(unittest.TestCase):
    """
    class LFPy.NetworkPopulation test suite
    """
    def test_NetworkPopulation_00(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstop=100,
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
            self.assertTrue((cell.somapos[0] == soma_pos['x']) &
                            (cell.somapos[1] == soma_pos['y']) &
                            (cell.somapos[2] == soma_pos['z']))
            self.assertEqual(cell.gid, gid)
            self.assertTrue(np.sqrt(soma_pos['x']**2 + soma_pos['y']**2) <= 100.)
        np.testing.assert_equal(population.gids, np.arange(4))


        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')


class testNetwork(unittest.TestCase):
    """
    class LFPy.Network test suite
    """
    def test_Network_00(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstop=100,
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
            name = 'test',
        )
        networkParameters = dict(
            dt=2**-3,
            tstart=0.,
            tstop=100.,
            v_init=-65.,
            celsius=6.3,
            OUTPUTPATH='tmp_testNetworkPopulation'
            )
        # set up
        network = LFPy.Network(**networkParameters)
        network.create_population(**populationParameters)
        connectivity = network.get_connectivity_rand(pre='test', post='test', connprob=0.5)

        # test set up
        for population in network.populations.values():
            self.assertTrue(len(population.cells) == population.POP_SIZE)
            for cell, soma_pos, gid in zip(population.cells, population.soma_pos, population.gids):
                self.assertTrue(type(cell) is LFPy.NetworkCell)
                self.assertTrue((cell.somapos[0] == soma_pos['x']) &
                                (cell.somapos[1] == soma_pos['y']) &
                                (cell.somapos[2] == soma_pos['z']))
                self.assertEqual(cell.gid, gid)
                self.assertTrue(np.sqrt(soma_pos['x']**2 + soma_pos['y']**2) <= 100.)
            np.testing.assert_equal(population.gids, np.arange(4))

        np.testing.assert_equal(connectivity.shape, (population.POP_SIZE, population.POP_SIZE))
        np.testing.assert_equal(connectivity.diagonal(), np.zeros(population.POP_SIZE))

        # connect and run sim
        network.connect(pre='test', post='test', connectivity=connectivity)
        network.simulate()

        # test output
        for population in network.populations.values():
            for cell in population.cells:
                self.assertTrue(np.all(cell.somav == network.v_init))

        network.pc.gid_clear()
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')


    def test_Network_01(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstop=100,
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
            name = 'test',
        )
        networkParameters = dict(
            dt=2**-3,
            tstart=0.,
            tstop=100.,
            v_init=-65.,
            celsius=6.3,
            OUTPUTPATH='tmp_testNetworkPopulation'
            )
        # set up
        network = LFPy.Network(**networkParameters)
        network.create_population(**populationParameters)
        connectivity = network.get_connectivity_rand(pre='test', post='test', connprob=0.5)

        # connect and run sim
        network.connect(pre='test', post='test', connectivity=connectivity)
        SPIKES, LFP, P = network.simulate(rec_current_dipole_moment=True)

        # test output
        for population in network.populations.values():
            for cell in population.cells:
                self.assertTrue(np.all(cell.somav == network.v_init))

        self.assertTrue(np.all(P['test'] == 0.))
        self.assertTrue(P.shape[0] == cell.somav.size)
        self.assertTrue(len(LFP) == 0)

        network.pc.gid_clear()
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')


    def test_Network_02(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstop=100,
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
            name = 'test',
        )
        networkParameters = dict(
            dt=2**-3,
            tstart=0.,
            tstop=100.,
            v_init=-65.,
            celsius=6.3,
            OUTPUTPATH='tmp_testNetworkPopulation'
            )
        clampParams = {
            'idx' : 0,
            'pptype' : 'VClamp',
            'amp[0]' : -65,
            'dur[0]' : 10,
            'amp[1]' : 0,
            'dur[1]' : 1,
            'amp[2]' : -65,
            'dur[2]' : 1E8,
        }

        # set up
        network = LFPy.Network(**networkParameters)
        network.create_population(**populationParameters)
        connectivity = network.get_connectivity_rand(pre='test', post='test', connprob=1)

        # test connectivity
        self.assertTrue(np.all(connectivity == (np.eye(populationParameters['POP_SIZE']) == 0)))

        # connect
        network.connect(pre='test', post='test', connectivity=connectivity,
                        multapseargs=dict(loc=1, scale=1E-9))

        # create synthetic AP in cell with gid == 0
        for population in network.populations.values():
            for cell in population.cells:
                if cell.gid == 0:
                    vclamp = LFPy.StimIntElectrode(cell=cell, **clampParams)

        # simulate
        network.simulate()

        # test output
        for population in network.populations.values():
            for cell in population.cells:
                self.assertFalse(np.all(cell.somav == network.v_init))

        network.pc.gid_clear()
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')


    def test_Network_03(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstop=100,
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
            name = 'test',
        )
        networkParameters = dict(
            dt=2**-3,
            tstart=0.,
            tstop=100.,
            v_init=-65.,
            celsius=6.3,
            OUTPUTPATH='tmp_testNetworkPopulation'
            )
        electrodeParameters = dict(
            sigma=0.3,
            x = np.arange(10)*100,
            y = np.arange(10)*100,
            z = np.arange(10)*100
        )
        # set up
        network = LFPy.Network(**networkParameters)
        network.create_population(**populationParameters)
        connectivity = network.get_connectivity_rand(pre='test', post='test', connprob=0.5)

        # test set up
        for population in network.populations.values():
            self.assertTrue(len(population.cells) == population.POP_SIZE)
            for cell, soma_pos, gid in zip(population.cells, population.soma_pos, population.gids):
                self.assertTrue(type(cell) is LFPy.NetworkCell)
                self.assertTrue((cell.somapos[0] == soma_pos['x']) &
                                (cell.somapos[1] == soma_pos['y']) &
                                (cell.somapos[2] == soma_pos['z']))
                self.assertEqual(cell.gid, gid)
                self.assertTrue(np.sqrt(soma_pos['x']**2 + soma_pos['y']**2) <= 100.)
            np.testing.assert_equal(population.gids, np.arange(4))

        np.testing.assert_equal(connectivity.shape, (population.POP_SIZE, population.POP_SIZE))
        np.testing.assert_equal(connectivity.diagonal(), np.zeros(population.POP_SIZE))

        # set up electrode
        electrode = LFPy.RecExtElectrode(**electrodeParameters)

        # connect and run sim
        network.connect(pre='test', post='test', connectivity=connectivity)
        network.simulate(electrode=electrode, to_file=True, to_memory=False,
                         file_name='OUTPUT.h5')

        # test output
        for population in network.populations.values():
            for cell in population.cells:
                self.assertTrue(np.all(cell.somav == network.v_init))
            
        f = h5py.File(os.path.join(network.OUTPUTPATH, 'OUTPUT.h5'), 'r')        
        np.testing.assert_equal(f['OUTPUT[0]'][()], np.zeros_like(f['OUTPUT[0]'][()]))
        f.close()


        network.pc.gid_clear()
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')


    def test_Network_04(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=True,
            dt=2**-3,
            tstop=100,
            delete_sections=False,
        )
        
        synapseParameters = dict(idx=0, syntype='Exp2Syn', weight=0.002,
                                 tau1=0.1, tau2=0.1, e=0)
        
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
            POP_SIZE = 1,
            name = 'test',
        )
        networkParameters = dict(
            dt=2**-3,
            tstart=0.,
            tstop=100.,
            v_init=-70.,
            celsius=6.3,
            OUTPUTPATH='tmp_testNetworkPopulation'
            )
        # set up
        network = LFPy.Network(**networkParameters)
        network.create_population(**populationParameters)
        
        cell = network.populations['test'].cells[0]
        
        # create synapses
        synlist = []
        numsynapses = 2
        for i in range(numsynapses):
            synlist.append(LFPy.Synapse(cell=cell, **synapseParameters))
            synlist[-1].set_spike_times(np.array([10+(i*10)]))
        
        network.simulate()
        
        # test that the input results in the correct amount of PSPs
        np.testing.assert_equal(ss.argrelextrema(cell.somav, np.greater)[0].size, numsynapses)

        # clean up
        network.pc.gid_clear()
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')


    def test_Network_05(self):
        cellParameters = dict(
            morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks_w_lists.hoc'),
            templatefile=os.path.join(LFPy.__path__[0], 'test', 'ball_and_stick_template.hoc'),
            templatename='ball_and_stick_template',
            templateargs=None,
            passive=False,
            dt=2**-3,
            tstop=100,
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
            name = 'test',
        )
        networkParameters = dict(
            dt=2**-3,
            tstart=0.,
            tstop=100.,
            v_init=-65.,
            celsius=6.3,
            OUTPUTPATH='tmp_testNetworkPopulation'
            )
        electrodeParameters = dict(
            sigma=0.3,
            x = np.arange(10)*100,
            y = np.arange(10)*100,
            z = np.arange(10)*100
        )
        # set up
        network = LFPy.Network(**networkParameters)
        network.create_population(**populationParameters)
        connectivity = network.get_connectivity_rand(pre='test', post='test',
                                                     connprob=0.5)

        # test set up
        for population in network.populations.values():
            self.assertTrue(len(population.cells) == population.POP_SIZE)
            for cell, soma_pos, gid in zip(population.cells,
                                           population.soma_pos,
                                           population.gids):
                self.assertTrue(type(cell) is LFPy.NetworkCell)
                self.assertTrue((cell.somapos[0] == soma_pos['x']) &
                                (cell.somapos[1] == soma_pos['y']) &
                                (cell.somapos[2] == soma_pos['z']))
                self.assertEqual(cell.gid, gid)
                self.assertTrue(np.sqrt(soma_pos['x']**2 + soma_pos['y']**2) <= 100.)
            np.testing.assert_equal(population.gids, np.arange(4))

        np.testing.assert_equal(connectivity.shape,
                                (population.POP_SIZE, population.POP_SIZE))
        np.testing.assert_equal(connectivity.diagonal(),
                                np.zeros(population.POP_SIZE))

        # set up electrode
        electrode = LFPy.RecExtElectrode(**electrodeParameters)

        # connect and run sim
        network.connect(pre='test', post='test', connectivity=connectivity)
        SPIKES, RESULTS, P = network.simulate(electrode=electrode, to_file=True,
                                      to_memory=True, file_name='OUTPUT.h5')

        # test output
        for population in network.populations.values():
            for cell in population.cells:
                self.assertTrue(np.all(cell.somav == network.v_init))
            
        f = h5py.File(os.path.join(network.OUTPUTPATH, 'OUTPUT.h5'), 'r')        
        np.testing.assert_equal(f['OUTPUT[0]'][()], RESULTS[0])
        f.close()


        network.pc.gid_clear()
        os.system('rm -r tmp_testNetworkPopulation')
        neuron.h('forall delete_section()')

