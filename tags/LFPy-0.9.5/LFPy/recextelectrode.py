#!/usr/bin/env python
'''Copyright (C) 2012 Computational Neuroscience Group, UMB.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.'''

import numpy as np
import warnings
from LFPy import lfpcalc, tools

class RecExtElectrodeSetup(object):
    '''
    RecExtElectrode superclass. 
    If (optional) cell argument is given then the it is imported,
    otherwise the cell argument has to be passed later on to calc_lfp.
    The argument cell can be an LFPy.cell.Cell object 
    or either a list or a dictionary containing such Cell objects. 
    Keyword arguments determine properties of later LFP-calculations
    
    Arguments:
    ::
        sigma : float,
            extracellular conductivity
        x, y, z : np.ndarray,
            coordinates or arrays of coordinates. Must be same length
        N : np.ndarray,
            Normal vector [x, y, z] of contact surface, default None
        r : float,
            radius of contact surface, default None
        n : int,
            if N != None and r > 0, the number of points to use for each
                  contact point in order to calculate average
        color : str,
            color of electrode contact points in plots
        marker : str,
            marker of electrode contact points in plots
        from_file : Bool,
            if True, load cell object from file
        cellfile : str,
            path to cell pickle
        verbose : Bool,
            Flag for verbose output
        seedvalue : int,
            rand seed when finding random position on contact with r >0
    '''
    def __init__(self, cell=None, sigma=0.3,
                 x=np.array([0]), y=np.array([0]), z=np.array([0]),
                 N=None, r=None, n=None, r_z=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None, verbose=False,
                 seedvalue=None):
        '''Initialize class RecExtElectrodeSetup'''
        self.sigma = sigma
        if type(x) == float or type(x) == int:
            self.x = np.array([x])
        else:
            self.x = np.array(x).flatten()
        if type(y) == float or type(y) == int:
            self.y = np.array([y])
        else:
            self.y = np.array(y).flatten()
        if type(z) == float or type(z) == int:
            self.z = np.array([z])
        else:
            self.z = np.array(z).flatten()
        self.color = color
        self.marker = marker
        if N != None:
            if type(N) == list:
                N = np.array(N)
            if N.shape[-1] == 3:
                self.N = N
            else:
                self.N = N.T
                if N.shape[-1] != 3:
                    raise Exception('N.shape must be (n_contacts, 1, 3)!')
        else:
            self.N = N
            
        self.r = r
        self.n = n
        self.r_z = r_z
        self.perCellLFP = perCellLFP
        
        self.method = method
        self.verbose = verbose
        self.seedvalue = seedvalue
        
        #None-type some attributes created by the Cell class
        self.electrodecoeff = None
        #self.tvec = np.array([])
        self.cells = {}
        self.nCells = None
        self.CellLFP = None
        self.circle = None
        #self.LFP = None
        self.offsets = None

        if from_file:
            if type(cellfile) == type(str()):
                cell = tools.load(cellfile)
            elif type(cellfile) == type([]):
                cell = []
                for fil in cellfile:
                    cell.append(tools.load(fil))
            else:
                raise ValueError('cell either string or list of strings')

        if cell is not None:
            self._import_cell(cell)


    class cell():
        '''Empty object that cell-specific variables are stored in'''
        def __init__(self):
            '''Just a container'''
            pass

    def _import_cell(self, cell):
        '''Keeps the relevant variables for LFP-calculation from cell'''
        #keeping these variables:
        variables = [
            #'somaidx',
            'timeres_python',
            'tstopms',
            'tvec',
            'imem',
            'diam',
            'xstart',
            'xmid',
            'xend',
            'ystart',
            'ymid',
            'yend',
            'zstart',
            'zmid',
            'zend',
            'totnsegs',
            #'synapses',
        ]
        
        #redefine list of cells as dict of cells
        if type(cell) == list:
            cellkey = 0
            celldict = {}
            for c in cell:
                celldict[cellkey] = c
                cellkey += 1
            cell = celldict
        
        
        if type(cell) == dict:
            for cellkey in cell:
                self.cells[cellkey] = self.cell()
                for v in variables:
                    try:
                        setattr(self.cells[cellkey], v, getattr(cell[cellkey], v))
                    except:
                        raise ValueError('cell[%s].%s missing' % (str(cellkey), v))
        else:
            self.cells[0] = self.cell()
            for v in variables:
                try:
                    setattr(self.cells[0], v, getattr(cell, v))
                except:
                    raise ValueError('cell.%s missing' % v)
        #
        #self.dt = self.cells[0].timeres_python
        #self.tvec = np.arange(self.cells[0].tstopms/self.dt + 1) * self.dt
        setattr(self, 'tvec', self.cells[0].tvec)
        #setattr(self, 'dt', self.cells[self.cells.keys()[0]].timeres_python)
        
        self.nCells = np.array(list(self.cells.keys())).size 
        
        #test that currents sum towards zero        
        self._test_imem_sum()
            
    
    def _test_imem_sum(self, tolerance=1E-12):
        '''Test that the membrane currents sum to zero'''
        for cellkey in self.cells.keys():
            sum_imem = self.cells[cellkey].imem.sum(axis=0)
            if np.any(sum_imem == np.ones(self.cells[cellkey].totnsegs)):
                pass
            else:
                if abs(sum_imem).max() >= tolerance:
                    warnings.warn('Membrane currents do not sum to zero')
                    [inds] = np.where((abs(sum_imem) >= tolerance))
                    for i in inds:
                        print('membrane current sum cell %i, timestep %i: %.3e'\
                            % (cellkey, i, sum_imem[i]))
                else:
                    pass


class RecExtElectrode(RecExtElectrodeSetup):
    '''
    RecExtElectrode class with inheritance from LFPy.RecExtElectrodeSetup 
    able to actually calculate local field potentials from LFPy.Cell objects. 
    **kwargs are passed on to LFPy.RecExtElectrodeSetup
    
    Usage:
    ::
        import numpy as np
        import import matplotlib.pyplot as plt
        import LFPy
        
        N = np.empty((16, 3))
        for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal vec. of contacts
        electrodeParameters = {         #parameters for RecExtElectrode class
            'sigma' : 0.3,              #Extracellular potential
            'x' : np.zeros(16)+25,      #Coordinates of electrode contacts
            'y' : np.zeros(16),
            'z' : np.linspace(-500,1000,16),
            'n' : 20,
            'r' : 10,
            'N' : N,
        }
        
        cellParameters = {                          
            'morphology' : 'L5_Mainen96_LFPy.hoc',  # morphology file
            'rm' : 30000,                           # membrane resistivity
            'cm' : 1.0,                             # membrane capacitance
            'Ra' : 150,                             # axial resistivity
            'timeres_NEURON' : 2**-4,                # dt for NEURON sim.
            'timeres_python' : 2**-4,                 # dt for python output
            'tstartms' : -50,                         # start t of simulation
            'tstopms' : 50,                        # end t of simulation
        }
        
        cell = LFPy.Cell(**cellParameters)
        
        synapseParameters = {
            'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
            'e' : 0,                                # reversal potential
            'syntype' : 'ExpSyn',                   # synapse type
            'tau' : 2,                              # syn. time constant
            'weight' : 0.01,                       # syn. weight
            'record_current' : True                 # syn. current record
        }
        
        synapse = LFPy.PointProcessSynapse(cell, **synapseParameters)
        synapse.set_spike_times(cell, np.array([10, 15, 20, 25]))
        
        cell.simulate()
        
        electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
        electrode.calc_lfp()
        plt.matshow(electrode.LFP)
    '''

    def __init__(self, cell=None, sigma=0.3,
                 x=np.array([0]), y=np.array([0]), z=np.array([0]),
                 N=None, r=None, n=0, r_z=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None, verbose=False,
                 seedvalue=None):
        '''This is the regular implementation of the RecExtElectrode class
        that calculates the LFP serially using a single core'''
        RecExtElectrodeSetup.__init__(self, cell, sigma, x, y, z,
                                N, r, n, r_z, perCellLFP,
                                method, color, marker, from_file,
                                cellfile, verbose, seedvalue)
        
        
    def calc_lfp(self, t_indices=None, cell=None):
        '''Calculate LFP on electrode geometry from all cell instances.
        Will chose distributed calculated if electrode contain 'n', 'N', and 'r'
        '''
        if not hasattr(self, 'cells') or len(list(self.cells.keys())) == 0:
            self._import_cell(cell)
       
        if not hasattr(self,  'LFP'):
            if t_indices != None:
                self.LFP = np.zeros((self.x.size, t_indices.size))
            else:
                self.LFP = np.zeros((self.x.size, self.tvec.size))
        if t_indices != None:
            LFP_temp = np.zeros((self.nCells, self.x.size, t_indices.size))
        else:
            LFP_temp = np.zeros((self.nCells, self.x.size, self.tvec.size))
            
        variables = {
            'sigma' : self.sigma,
            't_indices' : t_indices,
            'method' : self.method,
        }
        
        if self.n != None and self.N != None and self.r != None:
            if self.n <= 1:
                raise ValueError("n = %i must be larger that 1" % self.n)
            else:
                variables.update({
                    'radius' : self.r,
                    'n' : self.n,
                    'N' : self.N,
                    't_indices' : t_indices,
                    })


            for cellkey in self.cells.keys():                    
                variables.update({
                    'r_limit' : self.cells[cellkey].diam/2,
                    })
                [self.circle, self.offsets, LFP_temp[cellkey, :, :]] = \
                    self._lfp_el_pos_calc_dist(cellkey, **variables)
                if self.verbose:
                    print('Calculated potential contribution, cell %i.' % cellkey)
        else:
            for cellkey in self.cells.keys():
                variables.update({
                    'r_limit' : self.cells[cellkey].diam/2
                })
                LFP_temp[cellkey, :, :] = self._loop_over_contacts(**variables)
                if self.verbose:
                    print('Calculated potential contribution, cell %i.' % cellkey)
        if self.perCellLFP:
            self.CellLFP = []
            for LFPtrace in LFP_temp:
                self.CellLFP.append(LFPtrace)
        
        self.LFP = LFP_temp.sum(axis=0)

    def _loop_over_contacts(self, cellkey=0, sigma=0.3,
                    r_limit=None,
                    timestep=None, t_indices=None, method='linesource'):
        '''Loop over electrode contacts, and will return LFP_temp filled'''
        if t_indices != None:
            LFP_temp = np.zeros((self.x.size, t_indices.size))
        else:
            LFP_temp = np.zeros((self.x.size, self.tvec.size))
            
        for i in range(self.x.size):
            LFP_temp[i, :] = LFP_temp[i, :] + \
                    lfpcalc.calc_lfp_choose(self.cells[cellkey], x = self.x[i],
                                            y = self.y[i], z = self.z[i],
                                            sigma=sigma, r_limit=r_limit,
                                            timestep=timestep,
                                            t_indices=t_indices, method=method)
            
        return LFP_temp

    
    def _lfp_el_pos_calc_dist(self, cellkey=0, r_limit=None, sigma=0.3, radius=10, n=10,
                             m=50, N=None, t_indices=None, 
                             method='linesource'):
        '''
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface with radius r. The locations of these n points on
        the electrode surface are random,  within the given radius. The '''
        lfp_el_pos = np.zeros(self.LFP.shape)
        offsets = {}
        circle = {}
        for i in range(len(self.x)):
            if n > 1:
                lfp_e = np.zeros((n, self.LFP.shape[1]))

                offs = np.zeros((n, 3))
                r2 = np.zeros(n)

                crcl = np.zeros((m, 3))
                
                #assert the same random numbers are drawn every time
                if self.seedvalue != None:
                    np.random.seed(self.seedvalue)
                for j in range(n):
                    A = [(np.random.rand()-0.5)*radius*2,
                        (np.random.rand()-0.5)*radius*2,
                        (np.random.rand()-0.5)*radius*2]
                    offs[j, ] = np.cross(N[i, ], A)
                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
                    while r2[j] > radius**2:
                        A = [(np.random.rand()-0.5)*radius*2,
                            (np.random.rand()-0.5)*radius*2,
                            (np.random.rand()-0.5)*radius*2]
                        offs[j, ] = np.cross(N[i, ], A)
                        r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2

                x_n = offs[:, 0] + self.x[i]
                y_n = offs[:, 1] + self.y[i]
                z_n = offs[:, 2] + self.z[i]

                for j in range(m):
                    B = [(np.random.rand()-0.5),
                        (np.random.rand()-0.5),
                        (np.random.rand()-0.5)]
                    crcl[j, ] = np.cross(N[i, ], B)
                    crcl[j, ] = crcl[j, ]/np.sqrt(crcl[j, 0]**2 +
                                               crcl[j, 1]**2 + \
                                               crcl[j, 2]**2)*radius

                crclx = crcl[:, 0] + self.x[i]
                crcly = crcl[:, 1] + self.y[i]
                crclz = crcl[:, 2] + self.z[i]

                for j in range(n):
                    lfp_e[j, ] = lfpcalc.calc_lfp_choose(self.cells[cellkey],
                                                        x = x_n[j],
                                                        y = y_n[j],
                                                        z = z_n[j],
                                                        r_limit = r_limit,
                                                        sigma = sigma,
                                                        t_indices = t_indices,
                                                        method = method)

                lfp_el_pos[i] = lfp_e.mean(axis=0)

            else:
                lfp_el_pos[i] = lfpcalc.calc_lfp_choose(self.cells[cellkey], \
                    x=self.x[i], y=self.y[i], z=self.z[i], r_limit = r_limit, \
                    sigma=sigma, t_indices=t_indices)
            offsets[i] = {
                'x_n' : x_n,
                'y_n' : y_n,
                'z_n' : z_n
            }
            circle[i] = {
                'x' : crclx,
                'y' : crcly,
                'z' : crclz
            }
        return circle,  offsets,  lfp_el_pos



