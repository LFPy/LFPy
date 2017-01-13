#!/usr/bin/env python
'''Copyright (C) 2012 Computational Neuroscience Group, NMBU.

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
    The argument cell can be an LFPy.cell.Cell or LFPy.templatecell.TemplateCell
    object
    Keyword arguments determine properties of later LFP-calculations
    
    Arguments:
    ::
        
        cell : object,
            LFPy.cell.Cell or LFPy.templatecell.TemplateCell
        sigma : float,
            extracellular conductivity
        x, y, z : np.ndarray,
            coordinates or arrays of coordinates. Must be same length
        N : np.ndarray,
            Normal vector [x, y, z] of contact surface, default None
        r : float,
            radius of contact surface, default None
        n : int,
            if N is not None and r > 0, the number of points to use for each
                  contact point in order to calculate average
        shape : str,
            'circle'/'square' (default 'circle') defines the contact point shape
                  If 'circle' r is the radius, if 'square' r is the side length
        method : str,
            ['linesource']/'pointsource'/'som_as_point' switch
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
                 N=None, r=None, n=None, shape=None, r_z=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None, verbose=False,
                 seedvalue=None,
                 **kwargs):
        '''Initialize class RecExtElectrodeSetup'''
        self.cell = cell
        self.sigma = sigma
        if type(x) is float or type(x) is int:
            self.x = np.array([x])
        else:
            self.x = np.array(x).flatten()
        if type(y) is float or type(y) is int:
            self.y = np.array([y])
        else:
            self.y = np.array(y).flatten()
        if type(z) is float or type(z) is int:
            self.z = np.array([z])
        else:
            self.z = np.array(z).flatten()
        try:
            assert((self.x.size==self.y.size) and (self.x.size==self.z.size))
        except AssertionError as ae:
            raise ae, "The number of elements in [x, y, z] must be identical"
        
        self.color = color
        self.marker = marker
        if N is not None:
            if type(N) != np.array:
                try:
                    N = np.array(N)
                except:
                    print('Keyword argument N could not be converted to a '
                          'numpy.ndarray of shape (n_contacts, 3)')
                    print(sys.exc_info()[0])
                    raise
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

        if shape is None:
            self.shape = 'circle'
        elif shape in ['circle', 'square']:
            self.shape = shape
        else:
            raise ValueError('The shape argument must be either: None, \'circle\', \'square\'')

        self.r_z = r_z
        self.perCellLFP = perCellLFP
        
        self.method = method
        self.verbose = verbose
        self.seedvalue = seedvalue
        
        self.kwargs = kwargs
        
        #None-type some attributes created by the Cell class
        self.electrodecoeff = None
        self.circle = None
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

        if self.cell is not None:
            self._test_imem_sum()
            
        
    def _test_imem_sum(self, tolerance=1E-8):
        '''Test that the membrane currents sum to zero'''
        if type(self.cell) == dict or type(self.cell) == list:
            raise DeprecationWarning('no support for more than one cell-object')
        
        
        if self.cell is not None:
            sum_imem = self.cell.imem.sum(axis=0)
            #check if eye matrix is supplied:
            if np.any(sum_imem == np.ones(self.cell.totnsegs)):
                pass
            else:
                if abs(sum_imem).max() >= tolerance:
                    warnings.warn('Membrane currents do not sum to zero')
                    [inds] = np.where((abs(sum_imem) >= tolerance))
                    if self.cell.verbose:
                        for i in inds:
                            print('membrane current sum of celltimestep %i: %.3e'
                                % (i, sum_imem[i]))
                else:
                    pass
        else:
            pass


class RecExtElectrode(RecExtElectrodeSetup):
    '''
    RecExtElectrode class with inheritance from LFPy.RecExtElectrodeSetup 
    able to actually calculate local field potentials from LFPy.Cell objects. 
     
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
                 N=None, r=None, n=0, shape=None, r_z=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None, verbose=False,
                 seedvalue=None, **kwargs):
        '''This is the regular implementation of the RecExtElectrode class
        that calculates the LFP serially using a single core
        
        Arguments:
        ::
            
            cell : LFPy.Cell like object
            sigma : 
        '''
        RecExtElectrodeSetup.__init__(self, cell, sigma, x, y, z,
                                N, r, n, shape, r_z, perCellLFP,
                                method, color, marker, from_file,
                                cellfile, verbose, seedvalue, **kwargs)
        
        
    def calc_lfp(self, t_indices=None, cell=None):
        '''Calculate LFP on electrode geometry from all cell instances.
        Will chose distributed calculated if electrode contain 'n', 'N', and 'r'
        '''

        if cell is not None:
            self.cell = cell
            self._test_imem_sum()
       
        if not hasattr(self,  'LFP'):
            if t_indices is not None:
                self.LFP = np.zeros((self.x.size, t_indices.size))
            else:
                self.LFP = np.zeros((self.x.size, self.cell.imem.shape[1]))
                    
        
        if self.n is not None and self.N is not None and self.r is not None:
            if self.n <= 1:
                raise ValueError("n = %i must be larger that 1" % self.n)
            else:
                pass

            [self.circle_circum, self.offsets, LFP_temp] = \
                self._lfp_el_pos_calc_dist(t_indices=t_indices,
                                           r_limit=self.cell.diam/2)
            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        else:
            LFP_temp = self._loop_over_contacts(t_indices=t_indices,
                                                r_limit=self.cell.diam/2)
            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        
        #dump results:
        self.LFP = LFP_temp


    def _loop_over_contacts(self,
                    r_limit=None,
                    timestep=None,
                    t_indices=None):
        '''Loop over electrode contacts, and will return LFPs across channels'''
        if t_indices is not None:
            LFP_temp = np.zeros((self.x.size, t_indices.size))
        else:
            LFP_temp = np.zeros((self.x.size, self.cell.imem.shape[1]))
            
        for i in range(self.x.size):
            LFP_temp[i, :] = LFP_temp[i, :] + \
                    lfpcalc.calc_lfp_choose(self.cell,
                                            x = self.x[i],
                                            y = self.y[i],
                                            z = self.z[i],
                                            sigma = self.sigma,
                                            r_limit = r_limit,
                                            timestep = timestep,
                                            t_indices = t_indices,
                                            method = self.method,
                                            **self.kwargs)
            
        return LFP_temp

    
    def _lfp_el_pos_calc_dist(self,
                              r_limit=None,
                             m=50,
                             t_indices=None,
                             ):
        '''
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface: circle of radius r or square of side r. The
        locations of these n points on the electrode surface are random,
        within the given surface. '''
        lfp_el_pos = np.zeros(self.LFP.shape)
        offsets = {}
        circle_circ = {}

        def create_crcl(m, i):
            '''make circumsize of contact point'''
            crcl = np.zeros((m, 3))
            for j in range(m):
                B = [(np.random.rand()-0.5),
                    (np.random.rand()-0.5),
                    (np.random.rand()-0.5)]
                crcl[j, ] = np.cross(self.N[i, ], B)
                crcl[j, ] = crcl[j, ]/np.sqrt(crcl[j, 0]**2 +
                                           crcl[j, 1]**2 +
                                           crcl[j, 2]**2)*self.r

            crclx = crcl[:, 0] + self.x[i]
            crcly = crcl[:, 1] + self.y[i]
            crclz = crcl[:, 2] + self.z[i]
            
            return crclx, crcly, crclz

        def create_sqr(m, i):
            '''make circle in which square contact is circumscribed'''
            sqr = np.zeros((m, 3))
            for j in range(m):
                B = [(np.random.rand() - 0.5),
                     (np.random.rand() - 0.5),
                     (np.random.rand() - 0.5)]
                sqr[j,] = np.cross(self.N[i,], B)/np.linalg.norm(np.cross(self.N[i,], B)) * self.r * np.sqrt(2)/2

            sqrx = sqr[:, 0] + self.x[i]
            sqry = sqr[:, 1] + self.y[i]
            sqrz = sqr[:, 2] + self.z[i]

            return sqrx, sqry, sqrz

        def calc_xyz_n(i):
            '''calculate some offsets'''
            #offsets and radii init
            offs = np.zeros((self.n, 3))
            r2 = np.zeros(self.n)
            
            #assert the same random numbers are drawn every time
            if self.seedvalue is not None:
                np.random.seed(self.seedvalue)

            if self.shape is 'circle':
                for j in range(self.n):
                    A = [(np.random.rand()-0.5)*self.r*2,
                        (np.random.rand()-0.5)*self.r*2,
                        (np.random.rand()-0.5)*self.r*2]
                    offs[j, ] = np.cross(self.N[i, ], A)
                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
                    while r2[j] > self.r**2:
                        A = [(np.random.rand()-0.5)*self.r*2,
                            (np.random.rand()-0.5)*self.r*2,
                            (np.random.rand()-0.5)*self.r*2]
                        offs[j, ] = np.cross(self.N[i, ], A)
                        r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
            elif self.shape is 'square':
                for j in range(self.n):
                    A = [(np.random.rand()-0.5),
                        (np.random.rand()-0.5),
                        (np.random.rand()-0.5)]
                    offs[j, ] = np.cross(self.N[i, ], A)*self.r
                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2

            x_n = offs[:, 0] + self.x[i]
            y_n = offs[:, 1] + self.y[i]
            z_n = offs[:, 2] + self.z[i]
            
            return x_n, y_n, z_n

        def loop_over_points(x_n, y_n, z_n):

            #loop over points on contact
            for j in range(self.n):
                tmp = lfpcalc.calc_lfp_choose(self.cell,
                                              x = x_n[j],
                                              y = y_n[j],
                                              z = z_n[j],
                                              r_limit = r_limit,
                                              sigma = self.sigma,
                                              t_indices = t_indices,
                                              method = self.method,
                                              **self.kwargs)

                
                if j == 0:
                    lfp_e = tmp
                else:
                    lfp_e = np.r_['0,2', lfp_e, tmp]
                
                #no longer needed
                del tmp
            
            return lfp_e.mean(axis=0)

        #loop over contacts
        for i in range(len(self.x)):
            if self.n > 1:
            
                #fetch offsets:
                x_n, y_n, z_n = calc_xyz_n(i)
                
                #fill in with contact average
                lfp_el_pos[i] = loop_over_points(x_n, y_n, z_n) #lfp_e.mean(axis=0)
                
                ##no longer needed
                #del lfp_e
                
            else:
                lfp_el_pos[i] = lfpcalc.calc_lfp_choose(self.cell, 
                                                        x=self.x[i],
                                                        y=self.y[i],
                                                        z=self.z[i],
                                                        r_limit = r_limit, 
                                                        sigma=self.sigma,
                                                        t_indices=t_indices,
                                                        **self.kwargs)
                
            offsets[i] = {
                'x_n' : x_n,
                'y_n' : y_n,
                'z_n' : z_n
            }

            
            #fetch circumscribed circle around contact
            if self.shape is 'circle':
                crcl = create_crcl(m, i)
                circle_circ[i] = {
                    'x' : crcl[0],
                    'y' : crcl[1],
                    'z' : crcl[2],
                }
            elif self.shape  is 'square':
                sqr = create_sqr(m, i)
                circle_circ[i] = {
                    'x': sqr[0],
                    'y': sqr[1],
                    'z': sqr[2],
                }

        
        return circle_circ,  offsets,  lfp_el_pos



