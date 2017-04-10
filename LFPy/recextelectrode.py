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
import sys
import warnings
import numpy as np
from . import lfpcalc, tools

class RecExtElectrode:
    """class RecExtElectrode
    
    Main class that represents an extracellular electric recording devices such
    as a laminar probe.
    
    Parameters
    ----------
    cell : None or object
        If not None, instantiation of LFPy.Cell, LFPy.TemplateCell or similar. 
    sigma : float
        extracellular conductivity in unit (S/m)
    x, y, z : np.ndarray
        coordinates or arrays of coordinates in units of (um). Must be same length
    N : None or list of lists
        Normal vectors [x, y, z] of each circular electrode contact surface,
        default None
    r : float
        radius of each contact surface, default None
    n : int
        if N is not None and r > 0, the number of discrete points used to
        compute the n-point average potential on each circular contact point. 
    contact_shape : str
        'circle'/'square' (default 'circle') defines the contact point shape
        If 'circle' r is the radius, if 'square' r is the side length
    method : str
        switch between the assumption of 'linesource', 'pointsource',
        'soma_as_point' to represent each compartment when computing
        extracellular potentials
    from_file : bool
        if True, load cell object from file
    cellfile : str
        path to cell pickle
    verbose : bool
        Flag for verbose output, i.e., print more information
    seedvalue : int
        random seed when finding random position on contact with r > 0

    Examples
    --------
    
    Compute extracellular potentials after simulating and storage of
    transmembrane currents with the LFPy.Cell class:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import LFPy

    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    >>>     'v_init' : -65                          # initial voltage
    >>>     'rm' : 30000,                           # membrane resistivity
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True                        # insert passive channels
    >>>     'passive_parameters' : {g_pas=1./3E4, e_pas=-65} # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)

    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                       # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))

    >>> cell.simulate(rec_imem=True)

    >>> N = np.empty((16, 3))
    >>> for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal vec. of contacts
    >>> electrodeParameters = {         #parameters for RecExtElectrode class
    >>>     'sigma' : 0.3,              #Extracellular potential
    >>>     'x' : np.zeros(16)+25,      #Coordinates of electrode contacts
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.linspace(-500,1000,16),
    >>>     'n' : 20,
    >>>     'r' : 10,
    >>>     'N' : N,
    >>> }
    >>> electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
    >>> electrode.calc_lfp()
    >>> plt.matshow(electrode.LFP)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()


    Compute extracellular potentials during simulation (recommended):
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import LFPy

    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    >>>     'v_init' : -65                          # initial voltage
    >>>     'rm' : 30000,                           # membrane resistivity
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True                        # insert passive channels
    >>>     'passive_parameters' : {g_pas=1./3E4, e_pas=-65} # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)

    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                       # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))

    >>> N = np.empty((16, 3))
    >>> for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal vec. of contacts
    >>> electrodeParameters = {         #parameters for RecExtElectrode class
    >>>     'sigma' : 0.3,              #Extracellular potential
    >>>     'x' : np.zeros(16)+25,      #Coordinates of electrode contacts
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.linspace(-500,1000,16),
    >>>     'n' : 20,
    >>>     'r' : 10,
    >>>     'N' : N,
    >>> }
    >>> electrode = LFPy.RecExtElectrode(**electrodeParameters)

    >>> cell.simulate(electrode=electrode)

    >>> plt.matshow(electrode.LFP)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()

    """

    def __init__(self, cell=None, sigma=0.3,
                 x=np.array([0]), y=np.array([0]), z=np.array([0]),
                 N=None, r=None, n=None, contact_shape='circle', r_z=None,
                 perCellLFP=False, method='linesource',
                 from_file=False, cellfile=None, verbose=False,
                 seedvalue=None, **kwargs):
        """Initialize RecExtElectrode class"""

        self.sigma = sigma
        if type(sigma) in [list, np.ndarray]:
            self.sigma = np.array(sigma)
            if not self.sigma.shape == (3,):
                raise ValueError("Conductivity, sigma, should be float "
                                 "or array of length 3: "
                                 "[sigma_x, sigma_y, sigma_z]")

            self.anisotropic = True
        else:
            self.sigma = sigma
            self.anisotropic = False


        if type(x) in [float, int]:
            self.x = np.array([x])
        else:
            self.x = np.array(x).flatten()
        if type(y) in [float, int]:
            self.y = np.array([y])
        else:
            self.y = np.array(y).flatten()
        if type(z) in [float, int]:
            self.z = np.array([z])
        else:
            self.z = np.array(z).flatten()
        try:
            assert((self.x.size==self.y.size) and (self.x.size==self.z.size))
        except AssertionError:
            raise AssertionError("The number of elements in [x, y, z] must be identical")

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

        if contact_shape is None:
            self.contact_shape = 'circle'
        elif contact_shape in ['circle', 'square']:
            self.contact_shape = contact_shape
        else:
            raise ValueError('The contact_shape argument must be either: '
                             'None, \'circle\', \'square\'')

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

        if cell is not None:
            self.set_cell(cell)


        if method == 'soma_as_point':
            if self.anisotropic:
                self.lfp_method = lfpcalc.calc_lfp_soma_as_point_anisotropic
            else:
                self.lfp_method = lfpcalc.calc_lfp_soma_as_point
        elif method == 'som_as_point':
            raise RuntimeError('The method "som_as_point" is deprecated.'
                                     'Use "soma_as_point" instead')
        elif method == 'linesource':
            if self.anisotropic:
                self.lfp_method = lfpcalc.calc_lfp_linesource_anisotropic
            else:
                self.lfp_method = lfpcalc.calc_lfp_linesource
        elif method == 'pointsource':
            if self.anisotropic:
                self.lfp_method = lfpcalc.calc_lfp_pointsource_anisotropic
            else:
                self.lfp_method = lfpcalc.calc_lfp_pointsource
        else:
            raise ValueError("LFP method not recognized. "
                             "Should be 'soma_as_point', 'linesource' "
                             "or 'pointsource'")

    def set_cell(self, cell):
        self.cell = cell
        if self.cell is not None:
            self._test_imem_sum()
            # Handling the r_limits. If a r_limit is a single value, an array r_limit
            # of shape cell.diam is returned.
            # if type(r_limit) == int or type(r_limit) == float:
            #     r_limit = np.ones(np.shape(cell.diam))*abs(r_limit)
            # elif np.shape(r_limit) != np.shape(cell.diam):
            #     raise Exception('r_limit is neither a float- or int- value, nor is \
            #         r_limit.shape() equal to cell.diam.shape()')

            self.r_limit = self.cell.diam/2
            self.mapping = np.zeros((self.x.size, len(cell.xmid)))


    def _test_imem_sum(self, tolerance=1E-8):
        """Test that the membrane currents sum to zero"""
        if type(self.cell) == dict or type(self.cell) == list:
            raise DeprecationWarning('no support for more than one cell-object')

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



    def calc_lfp(self, t_indices=None, cell=None):
        """Calculate LFP on electrode geometry from all cell instances.
        Will chose distributed calculated if electrode contain 'n', 'N', and 'r'

        Parameters
        ----------
        cell : obj, optional
            `LFPy.Cell` or `LFPy.TemplateCell` instance. Must be specified here
            if it was not specified at the initiation of the `RecExtElectrode`
            class
        t_indices : np.ndarray
            Array of timestep indexes where extracellular potential should
            be calculated.
        """

        if cell is not None:
            self.set_cell(cell)

        if self.n is not None and self.N is not None and self.r is not None:
            if self.n <= 1:
                raise ValueError("n = %i must be larger that 1" % self.n)
            else:
                pass

            self._lfp_el_pos_calc_dist()

            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        else:
            self._loop_over_contacts()
            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        if t_indices is not None:
            currmem = self.cell.imem[:, t_indices]
        else:
            currmem = self.cell.imem

        self.LFP = np.dot(self.mapping, currmem)
        # del self.mapping


    def _loop_over_contacts(self):
        """Loop over electrode contacts, and return LFPs across channels"""

        for i in range(self.x.size):
            self.mapping[i, :] = self.lfp_method(self.cell,
                                             x = self.x[i],
                                             y = self.y[i],
                                             z = self.z[i],
                                             sigma = self.sigma,
                                             r_limit = self.r_limit)

    
    def _lfp_el_pos_calc_dist(self, m=50):

        """
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface: circle of radius r or square of side r. The
        locations of these n points on the electrode surface are random,
        within the given surface. """
        # lfp_el_pos = np.zeros(self.LFP.shape)
        self.offsets = {}
        self.circle_circ = {}

        def create_crcl(m, i):
            """make circumsize of contact point"""
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
            """make circle in which square contact is circumscribed"""
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
            """calculate some offsets"""
            #offsets and radii init
            offs = np.zeros((self.n, 3))
            r2 = np.zeros(self.n)
            
            #assert the same random numbers are drawn every time
            if self.seedvalue is not None:
                np.random.seed(self.seedvalue)

            if self.contact_shape is 'circle':
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
            elif self.contact_shape is 'square':
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
                tmp = self.lfp_method(self.cell,
                                              x = x_n[j],
                                              y = y_n[j],
                                              z = z_n[j],
                                              r_limit = self.r_limit,
                                              sigma = self.sigma)

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
                self.mapping[i] = loop_over_points(x_n, y_n, z_n) #lfp_e.mean(axis=0)

            else:
                self.mapping[i] = self.lfp_method(self.cell,
                                              x=self.x[i],
                                              y=self.y[i],
                                              z=self.z[i],
                                              r_limit = self.r_limit,
                                              sigma=self.sigma)

            self.offsets[i] = {'x_n' : x_n,
                               'y_n' : y_n,
                               'z_n' : z_n}

            #fetch circumscribed circle around contact
            if self.contact_shape is 'circle':
                crcl = create_crcl(m, i)
                self.circle_circ[i] = {
                    'x' : crcl[0],
                    'y' : crcl[1],
                    'z' : crcl[2],
                }
            elif self.contact_shape is 'square':
                sqr = create_sqr(m, i)
                self.circle_circ[i] = {
                    'x': sqr[0],
                    'y': sqr[1],
                    'z': sqr[2],
                }

