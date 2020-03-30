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


import sys
import warnings
import numpy as np
import MEAutility as mu
from copy import deepcopy
from . import lfpcalc, tools


class RecExtElectrode(object):
    """class RecExtElectrode

    Main class that represents an extracellular electric recording devices such
    as a laminar probe.

    Parameters
    ----------
    cell : None or object
        If not None, instantiation of LFPy.Cell, LFPy.TemplateCell or similar.
    sigma : float or list/ndarray of floats
        extracellular conductivity in units of (S/m). A scalar value implies an
        isotropic extracellular conductivity. If a length 3 list or array of
        floats is provided, these values corresponds to an anisotropic
        conductor with conductivities [sigma_x, sigma_y, sigma_z] accordingly.
    probe : MEAutility MEA object or None
        MEAutility probe object
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
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    >>>     'v_init' : -65,                          # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                        # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4, "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>>
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
    >>>
    >>> cell.simulate(rec_imem=True)
    >>>
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
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    >>>     'v_init' : -65,                          # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                        # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4, "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>>
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
    >>>
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
    >>>
    >>> cell.simulate(electrode=electrode)
    >>>
    >>> plt.matshow(electrode.LFP)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()

    Use MEAutility to to handle probes

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import MEAutility as mu
    >>> import LFPy
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    >>>     'v_init' : -65,                          # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                        # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4, "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>>
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
    >>>
    >>> cell.simulate(rec_imem=True)
    >>>
    >>> probe = mu.return_mea('Neuropixels-128')
    >>> electrode = LFPy.RecExtElectrode(cell, probe=probe)
    >>> electrode.calc_lfp()
    >>> mu.plot_mea_recording(electrode.LFP, probe)
    >>> plt.axis('tight')
    >>> plt.show()

    """

    def __init__(self, cell=None, sigma=0.3, probe=None,
                 x=None, y=None, z=None,
                 N=None, r=None, n=None, contact_shape='circle',
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

        if probe is None:
            assert x is not None and y is not None and z is not None, "Provide either a 'probe' or the " \
                                                                      "'x', 'y', and 'z'"
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
                assert ((self.x.size == self.y.size) and (self.x.size == self.z.size))
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
            elif contact_shape in ['circle', 'square', 'rect']:
                self.contact_shape = contact_shape
            else:
                raise ValueError('The contact_shape argument must be either: '
                                 'None, \'circle\', \'square\', \'rect\'')
            if self.contact_shape == 'rect':
                assert len(np.array(self.r)) == 2, "For 'rect' shape, 'r' must indicate the 2 dimensions " \
                                                   "of the rectangle"

            positions = np.array([self.x, self.y, self.z]).T
            probe_info = {'pos': positions, 'description': 'custom', 'size': self.r, 'shape': self.contact_shape,
                          'type': 'wire', 'center': False}  # add mea type
            self.probe = mu.MEA(positions=positions, info=probe_info, normal=self.N, sigma=self.sigma)
        else:
            assert isinstance(probe, mu.core.MEA), "'probe' should be a MEAutility MEA object"
            self.probe = deepcopy(probe)
            self.x = probe.positions[:, 0]
            self.y = probe.positions[:, 1]
            self.z = probe.positions[:, 2]
            self.N = np.array([el.normal for el in self.probe.electrodes])
            self.r = self.probe.size
            self.contact_shape = self.probe.shape
            self.n = n

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
        """Set the supplied cell object as attribute "cell" of the
        RecExtElectrode object

        Parameters
        ----------
        cell : obj
            `LFPy.Cell` or `LFPy.TemplateCell` instance.

        Returns
        -------
        None
        """
        self.cell = cell
        if self.cell is not None:
            self.r_limit = self.cell.diam/2
            self.mapping = np.zeros((self.x.size, len(cell.xmid)))


    def _test_imem_sum(self, tolerance=1E-8):
        """Test that the membrane currents sum to zero"""
        if type(self.cell) == dict or type(self.cell) == list:
            raise DeprecationWarning('no support for more than one cell-object')

        sum_imem = self.cell.imem.sum(axis=0)
        #check if eye matrix is supplied:
        if ((self.cell.imem.shape == (self.cell.totnsegs, self.cell.totnsegs))
            and (np.all(self.cell.imem == np.eye(self.cell.totnsegs)))):
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


    def calc_mapping(self, cell):
        """Creates a linear mapping of transmembrane currents of each segment
        of the supplied cell object to contribution to extracellular potential
        at each electrode contact point of the RexExtElectrode object. Sets
        the class attribute "mapping", which is a shape (n_contact, n_segs)
        ndarray, such that the extracellular potential at the contacts
        phi = np.dot(mapping, I_mem)
        where I_mem is a shape (n_segs, n_tsteps) ndarray with transmembrane
        currents for each time step of the simulation.

        Parameters
        ----------
        cell : obj
            `LFPy.Cell` or `LFPy.TemplateCell` instance.

        Returns
        -------
        mapping : ndarray
            The attribute RecExtElectrode.mapping is returned (optional)
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
        # return mapping
        return self.mapping


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

        self.calc_mapping(cell)

        if t_indices is not None:
            currmem = self.cell.imem[:, t_indices]
        else:
            currmem = self.cell.imem

        self._test_imem_sum()
        self.LFP = np.dot(self.mapping, currmem)
        # del self.mapping

    def _loop_over_contacts(self, **kwargs):
        """Loop over electrode contacts, and return LFPs across channels"""

        for i in range(self.x.size):
            self.mapping[i, :] = self.lfp_method(self.cell,
                                             x = self.x[i],
                                             y = self.y[i],
                                             z = self.z[i],
                                             sigma = self.sigma,
                                             r_limit = self.r_limit,
                                             **kwargs)

    def _lfp_el_pos_calc_dist(self, **kwargs):

        """
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface: circle of radius r or square of side r. The
        locations of these n points on the electrode surface are random,
        within the given surface. """

        def loop_over_points(points):

            # loop over points on contact
            lfp_e = 0
            for j in range(self.n):
                tmp = self.lfp_method(self.cell,
                                      x=points[j, 0],
                                      y=points[j, 1],
                                      z=points[j, 2],
                                      r_limit=self.r_limit,
                                      sigma=self.sigma,
                                      **kwargs
                                      )

                lfp_e += tmp
                # no longer needed
                del tmp

            return lfp_e / self.n

        # extract random points for each electrode
        if self.n > 1:
            points = self.probe.get_random_points_inside(self.n)
            for i, p in enumerate(points):
                #fill in with contact average
                self.mapping[i] = loop_over_points(p)
            self.recorded_points = points
        else:
            for i, (x, y, z) in enumerate(zip(self.x, self.y, self.z)):
                self.mapping[i] = self.lfp_method(self.cell,
                                                  x=x,
                                                  y=y,
                                                  z=z,
                                                  r_limit = self.r_limit,
                                                  sigma=self.sigma,
                                                  **kwargs)
            self.recorded_points = np.array([self.x, self.y, self.z]).T


class RecMEAElectrode(RecExtElectrode):
    r"""class RecMEAElectrode

    Electrode class that represents an extracellular in vitro slice recording
    as a Microelectrode Array (MEA). Inherits RecExtElectrode class

    Set-up:

              Above neural tissue (Saline) -> sigma_S
    <----------------------------------------------------> z = z_shift + h

              Neural Tissue -> sigma_T

                   o -> source_pos = [x',y',z']

    <-----------*----------------------------------------> z = z_shift + 0
                 \-> elec_pos = [x,y,z]

              Below neural tissue (MEA Glass plate) -> sigma_G

    Parameters
    ----------
    cell : None or object
        If not None, instantiation of LFPy.Cell, LFPy.TemplateCell or similar.
    sigma_T : float
        extracellular conductivity of neural tissue in unit (S/m)
    sigma_S : float
        conductivity of saline bath that the neural slice is
        immersed in [1.5] (S/m)
    sigma_G : float
        conductivity of MEA glass electrode plate. Most commonly
        assumed non-conducting [0.0] (S/m)
    h : float, int
        Thickness in um of neural tissue layer containing current
        the current sources (i.e., in vitro slice or cortex)
    z_shift : float, int
        Height in um of neural tissue layer bottom. If e.g., top of neural
        tissue layer should be z=0, use z_shift=-h. Defaults to z_shift = 0, so
        that the neural tissue layer extends from z=0 to z=h.
    squeeze_cell_factor : float or None
        Factor to squeeze the cell in the z-direction. This is
        needed for large cells that are thicker than the slice, since no part
        of the cell is allowed to be outside the slice. The squeeze is done
        after the neural simulation, and therefore does not affect neuronal
        simulation, only calculation of extracellular potentials.
    probe : MEAutility MEA object or None
        MEAutility probe object
    x, y, z : np.ndarray
        coordinates or arrays of coordinates in units of (um).
        Must be same length
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
    See also examples/example_MEA.py

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import LFPy
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    >>>     'v_init' : -65,                          # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                        # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4, "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>> cell.set_rotation(x=np.pi/2, z=np.pi/2)
    >>> cell.set_pos(z=100)
    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=800, y=0, z=100), # compartment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                       # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))
    >>>
    >>> MEA_electrode_parameters = {
    >>>     'sigma_T' : 0.3,      # extracellular conductivity
    >>>     'sigma_G' : 0.0,      # MEA glass electrode plate conductivity
    >>>     'sigma_S' : 1.5,      # Saline bath conductivity
    >>>     'x' : np.linspace(0, 1200, 16),  # electrode requires 1d vector of positions
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.zeros(16),
    >>>     "method": "pointsource",
    >>>     "h": 300,
    >>>     "squeeze_cell_factor": 0.3,
    >>> }
    >>> MEA = LFPy.RecMEAElectrode(cell, **MEA_electrode_parameters)
    >>>
    >>> cell.simulate(electrode=MEA)
    >>>
    >>> plt.matshow(MEA.LFP)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()
    """
    def __init__(self, cell=None, sigma_T=0.3, sigma_S=1.5, sigma_G=0.0,
                 h=300., z_shift=0., steps=20, probe=None,
                 x=np.array([0]), y=np.array([0]), z=np.array([0]),
                 N=None, r=None, n=None,
                 perCellLFP=False, method='linesource',
                 from_file=False, cellfile=None, verbose=False,
                 seedvalue=None, squeeze_cell_factor=None, **kwargs):

        RecExtElectrode.__init__(self, cell=cell,
                     x=x, y=y, z=z, probe=probe,
                     N=N, r=r, n=n,
                     perCellLFP=perCellLFP, method=method,
                     from_file=from_file, cellfile=cellfile, verbose=verbose,
                     seedvalue=seedvalue, **kwargs)

        self.sigma_G = sigma_G
        self.sigma_T = sigma_T
        self.sigma_S = sigma_S
        self.sigma = None
        self.h = h
        self.z_shift = z_shift
        self.steps = steps
        self.squeeze_cell_factor = squeeze_cell_factor
        self.moi_param_kwargs = {"h": self.h,
                                 "steps": self.steps,
                                 "sigma_G": self.sigma_G,
                                 "sigma_T": self.sigma_T,
                                 "sigma_S": self.sigma_S,
                                 }

        if cell is not None:
            self.set_cell(cell)

        if method == 'pointsource':
            self.lfp_method = lfpcalc.calc_lfp_pointsource_moi
        elif method == "linesource":
            if (np.abs(z - self.z_shift) > 1e-9).any():
                raise NotImplementedError("The method 'linesource' is only "
                                          "supported for electrodes at the "
                                          "z=0 plane. Use z=0 or method "
                                          "'pointsource'.")
            if np.abs(self.sigma_G) > 1e-9:
                raise NotImplementedError("The method 'linesource' is only "
                                          "supported for sigma_G=0. Use "
                                          "sigma_G=0 or method "
                                          "'pointsource'.")
            self.lfp_method = lfpcalc.calc_lfp_linesource_moi
        elif method == "soma_as_point":
            if (np.abs(z - self.z_shift) > 1e-9).any():
                raise NotImplementedError("The method 'soma_as_point' is only "
                                          "supported for electrodes at the "
                                          "z=0 plane. Use z=0 or method "
                                          "'pointsource'.")
            if np.abs(self.sigma_G) > 1e-9:
                raise NotImplementedError("The method 'soma_as_point' is only "
                                          "supported for sigma_G=0. Use "
                                          "sigma_G=0 or method "
                                          "'pointsource'.")
            self.lfp_method = lfpcalc.calc_lfp_soma_as_point_moi
        else:
            raise ValueError("LFP method not recognized. "
                             "Should be 'soma_as_point', 'linesource' "
                             "or 'pointsource'")

    def _squeeze_cell_in_depth_direction(self):
        """Will squeeze self.cell centered around the soma by a scaling factor,
        so that it fits inside the slice. If scaling factor is not big enough,
        a RuntimeError is raised. """

        self.cell.distort_geometry(factor=self.squeeze_cell_factor)

        if (np.max([self.cell.zstart, self.cell.zend]) > self.h + self.z_shift or
            np.min([self.cell.zstart, self.cell.zend]) < self.z_shift):
            bad_comps, reason = self._return_comp_outside_slice()
            msg = ("Compartments {} of cell ({}) has cell.{} slice. "
                   "Increase squeeze_cell_factor, move or rotate cell."
                   ).format(bad_comps, self.cell.morphology, reason)

            raise RuntimeError(msg)

    def _return_comp_outside_slice(self):
        """
        Assuming part of the cell is outside the valid region,
        i.e, not in the slice (self.z_shift < z < self.z_shift + self.h)
        this function check what array (cell.zstart or cell.zend) that is
        outside, and if it is above or below the valid region.

        Raises: RuntimeError
            If no compartment is outside valid region.

        Returns: array, str
            Numpy array with the compartments that are outside the slice,
            and a string with additional information on the problem.
        """
        zstart_above = np.where(self.cell.zstart > self.z_shift + self.h)[0]
        zend_above = np.where(self.cell.zend > self.z_shift + self.h)[0]
        zend_below = np.where(self.cell.zend < self.z_shift)[0]
        zstart_below = np.where(self.cell.zstart < self.z_shift)[0]

        if len(zstart_above) > 0:
            return zstart_above, "zstart above"
        if len(zstart_below) > 0:
            return zstart_below, "zstart below"
        if len(zend_above) > 0:
            return zend_above, "zend above"
        if len(zend_below) > 0:
            return zend_below, "zend below"
        raise RuntimeError("This function should only be called if cell"
                           "extends outside slice")

    def test_cell_extent(self):
        """
        Test if the cell is confined within the slice.
        If class argument "squeeze_cell" is True, cell is squeezed to to
        fit inside slice.

        """
        if self.cell is None:
            raise RuntimeError("Does not have cell instance.")

        if (np.max([self.cell.zstart, self.cell.zend]) > self.z_shift + self.h or
                np.min([self.cell.zstart, self.cell.zend]) < self.z_shift):

            if self.verbose:
                print("Cell extends outside slice.")

            if self.squeeze_cell_factor is not None:
                if not self.z_shift < self.cell.zmid[0] < self.z_shift + self.h:
                    raise RuntimeError("Soma position is not in slice.")
                self._squeeze_cell_in_depth_direction()
            else:
                bad_comps, reason = self._return_comp_outside_slice()
                msg = ("Compartments {} of cell ({}) has cell.{} slice "
                       "and argument squeeze_cell_factor is None."
                       ).format(bad_comps, self.cell.morphology, reason)
                raise RuntimeError(msg)
        else:
            if self.verbose:
                print("Cell position is good.")
            if self.squeeze_cell_factor is not None:
                if self.verbose:
                    print("Squeezing cell anyway.")
                self._squeeze_cell_in_depth_direction()

    def calc_mapping(self, cell):
        """Creates a linear mapping of transmembrane currents of each segment
        of the supplied cell object to contribution to extracellular potential
        at each electrode contact point of the RexExtElectrode object. Sets
        the class attribute "mapping", which is a shape (n_contact, n_segs)
        ndarray, such that the extracellular potential at the contacts
        phi = np.dot(mapping, I_mem)
        where I_mem is a shape (n_segs, n_tsteps) ndarray with transmembrane
        currents for each time step of the simulation.

        Parameters
        ----------
        cell : obj
            `LFPy.Cell` or `LFPy.TemplateCell` instance.

        Returns
        -------
        None
        """
        if cell is not None:
            self.set_cell(cell)
        self.test_cell_extent()

        # Temporarily shift coordinate system so middle layer extends
        # from z=0 to z=h
        self.z = self.z - self.z_shift
        self.cell.zstart = self.cell.zstart - self.z_shift
        self.cell.zmid = self.cell.zmid - self.z_shift
        self.cell.zend = self.cell.zend - self.z_shift

        if self.n is not None and self.N is not None and self.r is not None:
            if self.n <= 1:
                raise ValueError("n = %i must be larger that 1" % self.n)
            else:
                pass

            self._lfp_el_pos_calc_dist(**self.moi_param_kwargs)

            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        else:
            self._loop_over_contacts(**self.moi_param_kwargs)
            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))

        # Shift coordinate system back so middle layer extends
        # from z=z_shift to z=z_shift + h
        self.z = self.z + self.z_shift
        self.cell.zstart = self.cell.zstart + self.z_shift
        self.cell.zmid = self.cell.zmid + self.z_shift
        self.cell.zend = self.cell.zend + self.z_shift


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

        self.calc_mapping(cell)

        if t_indices is not None:
            currmem = self.cell.imem[:, t_indices]
        else:
            currmem = self.cell.imem

        self._test_imem_sum()
        self.LFP = np.dot(self.mapping, currmem)
        # del self.mapping
