#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines classes and methods used for recurrent neuronal networks.

Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import numpy as np
import os
import scipy.stats as stats
import h5py
from mpi4py import MPI
import neuron
from neuron import units
from .templatecell import TemplateCell
import scipy.sparse as ss

# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def flattenlist(lst):
    return [item for sublist in lst for item in sublist]


##########################################################################
# NetworkCell class that has a create_synapse method that
# creates a synapse on the target cell, and a create_spike_detector method that
# allows for connecting to a synapse on a target cell. All other methods and
# attributes are inherited from the standard LFPy.TemplateCell class
##########################################################################
class NetworkCell(TemplateCell):
    """
    Similar to `LFPy.TemplateCell` with the addition of some attributes and
    methods allowing for spike communication between parallel RANKs.

    This class allow using NEURON templates with some limitations.

    This takes all the same parameters as the Cell class, but requires three
    more template related parameters

    Parameters
    ----------
    morphology: str
        path to morphology file
    templatefile: str
        File with cell template definition(s)
    templatename: str
        Cell template-name used for this cell object
    templateargs: str
        Parameters provided to template-definition
    v_init: float
        Initial membrane potential. Default to -65.
    Ra: float
        axial resistance. Defaults to 150.
    cm: float
        membrane capacitance. Defaults to 1.0
    passive: bool
        Passive mechanisms are initialized if True. Defaults to True
    passive_parameters: dict
        parameter dictionary with values for the passive membrane mechanism in
        NEURON ('pas'). The dictionary must contain keys 'g_pas' and 'e_pas',
        like the default: passive_parameters=dict(g_pas=0.001, e_pas=-70)
    extracellular: bool
        switch for NEURON's extracellular mechanism. Defaults to False
    dt: float
        Simulation time step. Defaults to 2**-4
    tstart: float
        initialization time for simulation <= 0 ms. Defaults to 0.
    tstop: float
        stop time for simulation > 0 ms. Defaults to 100.
    nsegs_method: 'lambda100' or 'lambda_f' or 'fixed_length' or None
        nseg rule, used by NEURON to determine number of compartments.
        Defaults to 'lambda100'
    max_nsegs_length: float or None
        max segment length for method 'fixed_length'. Defaults to None
    lambda_f: int
        AC frequency for method 'lambda_f'. Defaults to 100
    d_lambda: float
        parameter for d_lambda rule. Defaults to 0.1
    delete_sections: bool
        delete pre-existing section-references. Defaults to True
    custom_code: list or None
        list of model-specific code files ([.py/.hoc]). Defaults to None
    custom_fun: list or None
        list of model-specific functions with args. Defaults to None
    custom_fun_args: list or None
        list of args passed to custom_fun functions. Defaults to None
    pt3d: bool
        use pt3d-info of the cell geometries switch. Defaults to False
    celsius: float or None
        Temperature in celsius. If nothing is specified here
        or in custom code it is 6.3 celcius
    verbose: bool
        verbose output switch. Defaults to False

    Examples
    --------

    >>> import LFPy
    >>> cellParameters = {
    >>>     'morphology': '<path to morphology.hoc>',
    >>>     'templatefile':  '<path to template_file.hoc>',
    >>>     'templatename':  'templatename',
    >>>     'templateargs':  None,
    >>>     'v_init': -65,
    >>>     'cm': 1.0,
    >>>     'Ra': 150,
    >>>     'passive': True,
    >>>     'passive_parameters': {'g_pas': 0.001, 'e_pas': -65.},
    >>>     'dt': 2**-3,
    >>>     'tstart': 0,
    >>>     'tstop': 50,
    >>> }
    >>> cell = LFPy.NetworkCell(**cellParameters)
    >>> cell.simulate()

    See also
    --------
    Cell
    TemplateCell
    """

    def __init__(self, **args):
        super().__init__(**args)

        # create list netconlist for spike detecting NetCon object(s)
        self._hoc_sd_netconlist = neuron.h.List()
        # create list of recording device for action potentials
        self.spikes = []
        # create list of random number generators used with synapse model
        self.rng_list = []

        # create separate list for networked synapses
        self.netconsynapses = []

        # create recording device for membrane voltage
        self.somav = neuron.h.Vector()
        for sec in self.somalist:
            self.somav.record(sec(0.5)._ref_v)

    def create_synapse(self, cell, sec, x=0.5, syntype=neuron.h.ExpSyn,
                       synparams=dict(tau=2., e=0.),
                       assert_syn_values=False):
        """
        Create synapse object of type syntype on sec(x) of cell and
        append to list cell.netconsynapses

        TODO: Use LFPy.Synapse class if possible.

        Parameters
        ----------
        cell: object
            instantiation of class NetworkCell or similar
        sec: neuron.h.Section object,
            section reference on cell
        x: float in [0, 1],
            relative position along section
        syntype: hoc.HocObject
            NEURON synapse model reference, e.g., neuron.h.ExpSyn
        synparams: dict
            parameters for syntype, e.g., for neuron.h.ExpSyn we have:
                tau: float, synapse time constant
                e: float, synapse reversal potential
        assert_syn_values: bool
            if True, raise AssertionError if synapse attribute values do not
            match the values in the synparams dictionary

        Raises
        ------
        AssertionError
        """
        # create a synapse object on the target cell
        syn = syntype(x, sec=sec)
        if hasattr(syn, 'setRNG'):
            # Create the random number generator for the synapse
            rng = neuron.h.Random()
            # not sure if this is how it is supposed to be set up...
            rng.MCellRan4(
                np.random.randint(
                    0,
                    2**32 - 1),
                np.random.randint(
                    0,
                    2**32 - 1))
            rng.uniform(0, 1)
            # used for e.g., stochastic synapse mechanisms (cf. BBP
            # microcircuit portal files)
            syn.setRNG(rng)
            cell.rng_list.append(rng)  # must store ref to rng object
        cell.netconsynapses.append(syntype(x, sec=sec))

        for key, value in synparams.items():
            setattr(cell.netconsynapses[-1], key, value)
            # check that synapses are parameterized correctly
            if assert_syn_values:
                try:
                    np.testing.assert_almost_equal(
                        getattr(cell.netconsynapses[-1], key), value)
                except AssertionError:
                    raise AssertionError('{} = {} != {}'.format(
                        key, getattr(cell.netconsynapses[-1], key), value))

    def create_spike_detector(self, target=None, threshold=-10.,
                              weight=0.0, delay=0.0):
        """
        Create spike-detecting NetCon object attached to the cell's soma
        midpoint, but this could be extended to having multiple spike-detection
        sites. The NetCon object created is attached to the cell's
        `_hoc_sd_netconlist` attribute, and will be used by the Network class
        when creating connections between all presynaptic cells and
        postsynaptic cells on each local RANK.

        Parameters
        ----------
        target: None (default) or a NEURON point process
        threshold: float
            spike detection threshold
        weight: float
            connection weight (not used unless target is a point process)
        delay: float
            connection delay (not used unless target is a point process)
        """
        # create new NetCon objects for the connections. Activation times will
        # be triggered on the somatic voltage with a given threshold.
        for sec in self.somalist:
            self._hoc_sd_netconlist.append(neuron.h.NetCon(sec(0.5)._ref_v,
                                                           target,
                                                           sec=sec))
            self._hoc_sd_netconlist[-1].threshold = threshold
            self._hoc_sd_netconlist[-1].weight[0] = weight
            self._hoc_sd_netconlist[-1].delay = delay


class DummyCell(object):
    def __init__(self, totnsegs=0,
                 x=None,
                 y=None,
                 z=None,
                 d=None,
                 area=None,
                 length=None,
                 somainds=None):
        """
        Dummy Cell object initialized with all attributes needed for LFP
        calculations using the LFPy.RecExtElectrode class and methods.
        This cell can be imagined as one "super" cell containing transmembrane
        currents generated by all NetworkCell segments on this RANK at once.

        Parameters
        ----------
        totnsegs: int
            total number of segments
        x, y, z: ndarray
            arrays of shape (totnsegs, 2) with (x,y,z) coordinates of start
            and end points of segments in units of (um)
        d: ndarray
            array of length totnsegs with segment diameters
        area: ndarray
            array of segment surface areas
        length: ndarray
            array of segment lengths
        """
        # set attributes
        self.totnsegs = totnsegs
        self.x = x if x is not None else np.array([])
        self.y = y if y is not None else np.array([])
        self.z = z if z is not None else np.array([])
        self.d = d if d is not None else np.array([])
        self.area = area if area is not None else np.array([])
        self.length = length if area is not None else np.array([])
        self.somainds = somainds if somainds is not None else np.array([])

    def get_idx(self, section="soma"):
        if section == "soma":
            return self.somainds
        else:
            raise ValueError('section argument must be "soma"')


class NetworkPopulation(object):
    """
    NetworkPopulation class representing a group of Cell objects
    distributed across RANKs.

    Parameters
    ----------
    CWD: path or None
        Current working directory
    CELLPATH: path or None
        Relative path from CWD to source files for cell model
        (morphology, hoc routines etc.)
    first_gid: int
        The global identifier of the first cell created in this population
        instance. The first_gid in the first population created should be 0
        and cannot exist in previously created NetworkPopulation instances
    Cell: class
        class defining a Cell object, see class NetworkCell above
    POP_SIZE: int
        number of cells in population
    name: str
        population name reference
    cell_args: dict
        keys and values for Cell object
    pop_args: dict
        keys and values for Network.draw_rand_pos assigning cell positions
    rotation_arg: dict
        default cell rotations around x and y axis on the form
        { 'x': np.pi/2, 'y': 0 }. Can only have the keys 'x' and 'y'.
        Cells are randomly rotated around z-axis using the
        Cell.set_rotation() method.
    OUTPUTPATH: str
        path to output file destination
    """
    def __init__(self, CWD=None, CELLPATH=None, first_gid=0, Cell=NetworkCell,
                 POP_SIZE=4, name='L5PC',
                 cell_args=None, pop_args=None,
                 rotation_args=None,
                 OUTPUTPATH='example_parallel_network'):
        # set class attributes
        self.CWD = CWD
        self.CELLPATH = CELLPATH
        self.first_gid = first_gid
        self.Cell = Cell
        self.POP_SIZE = POP_SIZE
        self.name = name
        self.cell_args = cell_args if cell_args is not None else dict()
        self.pop_args = pop_args if pop_args is not None else dict()
        self.rotation_args = rotation_args if rotation_args is not None \
            else dict()
        self.OUTPUTPATH = OUTPUTPATH

        # create folder for output if it does not exist
        if RANK == 0:
            if not os.path.isdir(OUTPUTPATH):
                os.mkdir(OUTPUTPATH)
        COMM.Barrier()

        # container of Vector objects used to record times of action potentials
        self.spike_vectors = []

        # set up population of cells on this RANK
        self.gids = [
            (i +
             first_gid) for i in range(POP_SIZE) if (
                i +
                first_gid) %
            SIZE == RANK]

        # we have to enter the cell's corresponding file directory to
        # create cell because how EPFL set their code up
        if CWD is not None:
            os.chdir(os.path.join(CWD, CELLPATH, self.name))
            self.cells = [Cell(**cell_args) for gid in self.gids]
            os.chdir(CWD)
        else:
            self.cells = [Cell(**cell_args) for gid in self.gids]
        # position each cell's soma in space
        self.soma_pos = self.draw_rand_pos(POP_SIZE=len(self.gids), **pop_args)
        for i, cell in enumerate(self.cells):
            cell.set_pos(**self.soma_pos[i])

        # assign a random rotation around the z-axis of each cell
        self.rotations = np.random.uniform(0, np.pi * 2, len(self.gids))
        assert 'z' not in self.rotation_args.keys()
        for i, cell in enumerate(self.cells):
            cell.set_rotation(z=self.rotations[i], **self.rotation_args)

        # assign gid to each cell
        for gid, cell in zip(self.gids, self.cells):
            cell.gid = gid

        # gather gids, soma positions and cell rotations to RANK 0, and write
        # as structured array.
        if RANK == 0:
            populationData = flattenlist(COMM.gather(
                zip(self.gids, self.soma_pos, self.rotations)))

            # create structured array for storing data
            dtype = [('gid', 'i8'), ('x', float), ('y', float), ('z', float),
                     ('x_rot', float), ('y_rot', float), ('z_rot', float)]
            popDataArray = np.empty((len(populationData, )), dtype=dtype)
            for i, (gid, pos, z_rot) in enumerate(populationData):
                popDataArray[i]['gid'] = gid
                popDataArray[i]['x'] = pos['x']
                popDataArray[i]['y'] = pos['y']
                popDataArray[i]['z'] = pos['z']
                popDataArray[i]['x_rot'] = np.pi / 2
                popDataArray[i]['y_rot'] = 0.
                popDataArray[i]['z_rot'] = z_rot

            # Dump to hdf5 file, append to file if it exists
            f = h5py.File(os.path.join(self.OUTPUTPATH,
                                       'cell_positions_and_rotations.h5'), 'a')
            # delete old entry if it exist
            if self.name in f.keys():
                del f[self.name]
                assert self.name not in f.keys()
            f[self.name] = popDataArray
            f.close()
        else:
            COMM.gather(zip(self.gids, self.soma_pos, self.rotations))

        # sync
        COMM.Barrier()

    def draw_rand_pos(self, POP_SIZE, radius, loc, scale, cap=None):
        """
        Draw some random location for POP_SIZE cells within radius radius,
        at mean depth loc and standard deviation scale.

        Returned argument is a list of dicts [{'x', 'y', 'z'},].


        Parameters
        ----------
        POP_SIZE: int
            Population size
        radius: float
            Radius of population.
        loc: float
            expected mean depth of somas of population.
        scale: float
            expected standard deviation of depth of somas of population.
        cap: None, float or length to list of floats
            if float, cap distribution between [loc-cap, loc+cap),
            if list, cap distribution between [loc-cap[0], loc+cap[1]]


        Returns
        -------
        soma_pos: list
            List of dicts of len POP_SIZE
            where dict have keys x, y, z specifying
            xyz-coordinates of cell at list entry `i`.


        """

        x = np.empty(POP_SIZE)
        y = np.empty(POP_SIZE)
        z = np.empty(POP_SIZE)
        for i in range(POP_SIZE):
            x[i] = (np.random.rand() - 0.5) * radius * 2
            y[i] = (np.random.rand() - 0.5) * radius * 2
            while np.sqrt(x[i]**2 + y[i]**2) >= radius:
                x[i] = (np.random.rand() - 0.5) * radius * 2
                y[i] = (np.random.rand() - 0.5) * radius * 2
        z = np.random.normal(loc=loc, scale=scale, size=POP_SIZE)
        if cap is not None:
            if type(cap) in [float, np.float32, np.float64]:
                while not np.all((z >= loc - cap) & (z < loc + cap)):
                    inds = (z < loc - cap) ^ (z > loc + cap)
                    z[inds] = np.random.normal(loc=loc, scale=scale,
                                               size=inds.sum())
            elif isinstance(cap, list):
                assert len(cap) == 2, \
                    'cap = {} is not a length 2 list'.format(float)
                while not np.all((z >= loc - cap[0]) & (z < loc + cap[1])):
                    inds = (z < loc - cap[0]) ^ (z > loc + cap[1])
                    z[inds] = np.random.normal(loc=loc, scale=scale,
                                               size=inds.sum())
            else:
                raise Exception('cap = {} is not None'.format(float),
                                'a float or length 2 list of floats')

        soma_pos = []
        for i in range(POP_SIZE):
            soma_pos.append({'x': x[i], 'y': y[i], 'z': z[i]})

        return soma_pos


class Network(object):
    """
    Network class, creating distributed populations of cells of
    type Cell and handling connections between cells in the respective
    populations.

    Parameters
    ----------
    dt: float
        Simulation timestep size
    tstart: float
        Start time of simulation
    tstop: float
        End time of simulation
    v_init: float
        Membrane potential set at first timestep across all cells
    celsius: float
        Global control of temperature, affect channel kinetics.
        It will also be forced when creating the different Cell objects, as
        LFPy.Cell and LFPy.TemplateCell also accept the same keyword
        argument.
    verbose: bool
        if True, print out misc. messages
    """
    def __init__(
            self,
            dt=0.1,
            tstart=0.,
            tstop=1000.,
            v_init=-65.,
            celsius=6.3,
            OUTPUTPATH='example_parallel_network',
            verbose=False):
        # set attributes
        self.dt = dt
        self.tstart = tstart
        self.tstop = tstop
        self.v_init = v_init
        self.celsius = celsius
        self.OUTPUTPATH = OUTPUTPATH
        self.verbose = verbose

        # we need NEURON's ParallelContext for communicating NetCon events
        self.pc = neuron.h.ParallelContext()

        # create empty list for connections between cells (not to be confused
        # with each cell's list of netcons _hoc_netconlist)
        self._hoc_netconlist = neuron.h.List()

        # The different populations in the Network will be collected in
        # a dictionary of NetworkPopulation object, where the keys represent
        # population names. The names are also put in a list ordered according
        # to the order populations are created in (as some operations rely on
        # this particular order)
        self.populations = dict()
        self.population_names = []

    def create_population(self, CWD=None, CELLPATH=None, Cell=NetworkCell,
                          POP_SIZE=4, name='L5PC',
                          cell_args=None, pop_args=None,
                          rotation_args=None):
        """
        Create and append a distributed POP_SIZE-sized population of cells of
        type Cell with the corresponding name. Cell-object references, gids on
        this RANK, population size POP_SIZE and names will be added to the
        lists Network.gids, Network.cells, Network.sizes and Network.names,
        respectively

        Parameters
        ----------
        CWD: path
            Current working directory
        CELLPATH: path
            Relative path from CWD to source files for cell model
            (morphology, hoc routines etc.)
        Cell: class
            class defining a Cell-like object, see class NetworkCell
        POP_SIZE: int
            number of cells in population
        name: str
            population name reference
        cell_args: dict
            keys and values for Cell object
        pop_args: dict
            keys and values for Network.draw_rand_pos assigning cell positions
        rotation_arg: dict
            default cell rotations around x and y axis on the form
            { 'x': np.pi/2, 'y': 0 }. Can only have the keys 'x' and 'y'.
            Cells are randomly rotated around z-axis using the
            Cell.set_rotation method.

        """
        assert name not in self.populations.keys(), \
            'population name {} already taken'.format(name)

        # compute the first global id of this new population, based
        # on population sizes of existing populations
        first_gid = 0
        for p in self.populations.values():
            first_gid += p.POP_SIZE

        # create NetworkPopulation object
        population = NetworkPopulation(
            CWD=CWD,
            CELLPATH=CELLPATH,
            first_gid=first_gid,
            Cell=Cell,
            POP_SIZE=POP_SIZE,
            name=name,
            cell_args=cell_args,
            pop_args=pop_args,
            rotation_args=rotation_args,
            OUTPUTPATH=self.OUTPUTPATH)

        # associate gids of cells on this RANK such that NEURON can look up
        # at which RANK different cells are created when connecting the network
        for gid in population.gids:
            self.pc.set_gid2node(gid, RANK)

        # Prepare connection targets by iterating over local neurons in pop.
        for gid, cell in zip(population.gids, population.cells):
            # attach NetCon source (spike detektor) to each cell's soma with no
            # target to cell gid
            cell.create_spike_detector(None)
            # assosiate cell gid with the NetCon source
            self.pc.cell(gid, cell._hoc_sd_netconlist[-1])

            # record spike events
            population.spike_vectors.append(neuron.h.Vector())
            cell._hoc_sd_netconlist[-1].record(population.spike_vectors[-1])

        # add population object to dictionary of populations
        self.populations[name] = population

        # append population name to list (Network.populations.keys() not
        # unique)
        self.population_names.append(name)

    def get_connectivity_rand(self, pre='L5PC', post='L5PC', connprob=0.2):
        """
        Dummy function creating a (boolean) cell to cell connectivity matrix
        between pre and postsynaptic populations.

        Connections are drawn randomly between presynaptic cell gids in
        population 'pre' and postsynaptic cell gids in 'post' on this RANK with
        a fixed connection probability. self-connections are disabled if
        presynaptic and postsynaptic populations are the same.

        Parameters
        ----------
        pre: str
            presynaptic population name
        post: str
            postsynaptic population name
        connprob: float in [0, 1]
            connection probability, connections are drawn on random

        Returns
        -------
        ndarray, dtype bool
            n_pre x n_post array of connections between n_pre presynaptic
            neurons and n_post postsynaptic neurons on this RANK. Entries
            with True denotes a connection.
        """
        n_pre = self.populations[pre].POP_SIZE
        gids = np.array(self.populations[post].gids).astype(int)

        # first check if there are any postsyn cells on this RANK
        if gids.size > 0:
            # define incoming connections for cells on this RANK
            C = np.random.binomial(n=1, p=connprob,
                                   size=(n_pre, gids.size)
                                   ).astype(bool)
            if pre == post:
                # avoid self connections.
                gids_pre, gids_post = np.where(C)
                gids_pre += self.populations[pre].first_gid
                gids_post *= SIZE  # asssume round-robin distribution of gids
                gids_post += self.populations[post].gids[0]
                inds = gids_pre != gids_post
                gids_pre = gids_pre[inds]
                gids_pre -= self.populations[pre].first_gid
                gids_post = gids_post[inds]
                gids_post -= self.populations[post].gids[0]
                gids_post //= SIZE
                c = np.c_[gids_pre, gids_post]
                # create boolean matrix
                C = ss.csr_matrix((np.ones(gids_pre.shape[0], dtype=bool),
                                   (c[:, 0], c[:, 1])),
                                  shape=(n_pre, gids.size), dtype=bool)
                return C.toarray()
            else:
                return C
        else:
            return np.zeros((n_pre, 0), dtype=bool)

    def connect(self, pre, post, connectivity,
                syntype=neuron.h.ExpSyn,
                synparams=dict(tau=2., e=0.),
                weightfun=np.random.normal,
                weightargs=dict(loc=0.1, scale=0.01),
                minweight=0,
                delayfun=np.random.normal,
                delayargs=dict(loc=2, scale=0.2),
                mindelay=0.3,
                multapsefun=np.random.normal,
                multapseargs=dict(loc=4, scale=1),
                syn_pos_args=dict(section=['soma', 'dend', 'apic'],
                                  fun=[stats.norm] * 2,
                                  funargs=[dict(loc=0, scale=100)] * 2,
                                  funweights=[0.5] * 2,
                                  z_min=-1E6, z_max=1E6,
                                  ),
                save_connections=False,
                ):
        """
        Connect presynaptic cells to postsynaptic cells. Connections are
        drawn from presynaptic cells to postsynaptic cells, hence connectivity
        array must only be specified for postsynaptic units existing on this
        RANK.

        Parameters
        ----------
        pre: str
            presynaptic population name
        post: str
            postsynaptic population name
        connectivity: ndarray / (scipy.sparse array)
            boolean connectivity matrix between pre and post.
        syntype: hoc.HocObject
            reference to NEURON synapse mechanism, e.g., ``neuron.h.ExpSyn``
        synparams: dict
            dictionary of parameters for synapse mechanism, keys 'e', 'tau'
            etc.
        weightfun: function
            function used to draw weights from a numpy.random distribution
        weightargs: dict
            parameters passed to weightfun
        minweight: float,
            minimum weight in units of nS
        delayfun: function
            function used to draw delays from a numpy.random distribution
        delayargs: dict
            parameters passed to delayfun
        mindelay: float,
            minimum delay in multiples of dt
        multapsefun: function or None
            function reference, e.g., numpy.random.normal used to draw a number
            of synapses for a cell-to-cell connection. If None, draw only one
            connection
        multapseargs: dict
            arguments passed to multapsefun
        syn_pos_args: dict
            arguments passed to inherited ``LFPy.Cell`` method
            ``NetworkCell.get_rand_idx_area_and_distribution_norm`` to find
            synapse locations.
        save_connections: bool
            if True (default False), save instantiated connections to HDF5 file
            ``Network.OUTPUTPATH/synapse_connections.h5`` as dataset
            ``<pre>:<post>`` using a structured ndarray with dtype
            ::

                [('gid_pre'), ('gid', 'i8'), ('weight', 'f8'), ('delay', 'f8'),
                ('sec', 'U64'), ('sec.x', 'f8'),
                ('x', 'f8'), ('y', 'f8'), ('z', 'f8')],

            where ``gid_pre`` is presynapic cell id,
            ``gid`` is postsynaptic cell id,
            ``weight`` connection weight, ``delay`` connection delay,
            ``sec`` section name, ``sec.x`` relative location on section,
            and ``x``, ``y``, ``z`` the corresponding
            midpoint coordinates of the target compartment.

        Returns
        -------
        list
            Length 2 list with ndarrays [conncount, syncount] with numbers of
            instantiated connections and synapses.
        """
        # set up connections from all cells in presynaptic to post across RANKs
        n0 = self.populations[pre].first_gid
        # gids of presynaptic neurons:
        gids_pre = np.arange(n0, n0 + self.populations[pre].POP_SIZE)

        # count connections and synapses made on this RANK
        conncount = connectivity.astype(int).sum()
        syncount = 0

        # keep track of synapse positions for this connect
        # call on this rank such that these can be communicated and stored
        syn_idx_pos = []

        # iterate over gids on this RANK and create connections
        for i, (gid_post, cell) in enumerate(zip(self.populations[post].gids,
                                                 self.populations[post].cells)
                                             ):
            # do NOT iterate over all possible presynaptic neurons
            for gid_pre in gids_pre[connectivity[:, i]]:
                # throw a warning if sender neuron is identical to receiving
                # neuron
                if gid_post == gid_pre:
                    print(
                        'connecting cell w. gid {} to itself (RANK {})'.format(
                            gid_post, RANK))

                # assess number of synapses
                if multapsefun is None:
                    nidx = 1
                else:
                    nidx = 0
                    j = 0
                    while nidx <= 0 and j < 1000:
                        nidx = int(round(multapsefun(**multapseargs)))
                        j += 1
                    if j == 1000:
                        raise Exception('change multapseargs as no positive '
                                        'synapse # was found in 1000 trials')

                # find synapse locations and corresponding section names
                idxs = cell.get_rand_idx_area_and_distribution_norm(
                    nidx=nidx, **syn_pos_args)
                secs = cell.get_idx_name(idxs)

                # draw weights
                weights = weightfun(size=nidx, **weightargs)
                # redraw weights less that minweight
                while np.any(weights < minweight):
                    j = weights < minweight
                    weights[j] = weightfun(size=j.sum(), **weightargs)

                # draw delays
                delays = delayfun(size=nidx, **delayargs)
                # redraw delays shorter than mindelay
                while np.any(delays < mindelay):
                    j = delays < mindelay
                    delays[j] = delayfun(size=j.sum(), **delayargs)

                for i, ((idx, secname, secx), weight, delay) in enumerate(
                        zip(secs, weights, delays)):
                    cell.create_synapse(
                        cell,
                        # TODO: Find neater way of accessing
                        # Section reference, this looks slow
                        sec=list(
                            cell.allseclist)[
                            np.where(
                                np.array(
                                    cell.allsecnames) == secname)[0][0]],
                        x=secx,
                        syntype=syntype,
                        synparams=synparams)
                    # connect up NetCon object
                    nc = self.pc.gid_connect(gid_pre, cell.netconsynapses[-1])
                    nc.weight[0] = weight
                    nc.delay = delays[i]
                    self._hoc_netconlist.append(nc)

                    # store also synapse indices allowing for computing LFPs
                    # from syn.i
                    cell.synidx.append(idx)

                    # store gid and xyz-coordinate of synapse positions
                    syn_idx_pos.append((gid_pre,
                                        cell.gid,
                                        weight,
                                        delays[i],
                                        secname,
                                        secx,
                                        cell.x[idx].mean(axis=-1),
                                        cell.y[idx].mean(axis=-1),
                                        cell.z[idx].mean(axis=-1)))

                syncount += nidx

        conncount = COMM.reduce(conncount, op=MPI.SUM, root=0)
        syncount = COMM.reduce(syncount, op=MPI.SUM, root=0)

        if RANK == 0:
            print('Connected population {} to {}'.format(pre, post),
                  'by {} connections and {} synapses'.format(conncount,
                                                             syncount))
        else:
            conncount = None
            syncount = None

        # gather and write syn_idx_pos data
        if save_connections:
            if RANK == 0:
                synData = flattenlist(COMM.gather(syn_idx_pos))

                # convert to structured array
                dtype = [('gid_pre', 'i8'),
                         ('gid', 'i8'),
                         ('weight', 'f8'),
                         ('delay', 'f8'),
                         ('sec', 'S64'),
                         ('sec.x', 'f8'),
                         ('x', 'f8'),
                         ('y', 'f8'),
                         ('z', 'f8')]
                synDataArray = np.empty((len(synData), ), dtype=dtype)
                for i, (gid_pre, gid, weight, delay, secname, secx, x, y, z
                        ) in enumerate(synData):
                    synDataArray[i]['gid_pre'] = gid_pre
                    synDataArray[i]['gid'] = gid
                    synDataArray[i]['weight'] = weight
                    synDataArray[i]['delay'] = delay
                    synDataArray[i]['sec'] = secname
                    synDataArray[i]['sec.x'] = secx
                    synDataArray[i]['x'] = x
                    synDataArray[i]['y'] = y
                    synDataArray[i]['z'] = z
                # Dump to hdf5 file, append to file if entry exists
                with h5py.File(os.path.join(self.OUTPUTPATH,
                                            'synapse_connections.h5'),
                               'a') as f:
                    key = '{}:{}'.format(pre, post)
                    if key in f.keys():
                        del f[key]
                        assert key not in f.keys()
                    f[key] = synDataArray
                    # save global connection data (synapse type/parameters)
                    # equal for all synapses
                    try:
                        grp = f.create_group('synparams')
                    except ValueError:
                        grp = f['synparams']
                    try:
                        subgrp = grp.create_group(key)
                    except ValueError:
                        subgrp = grp[key]
                    subgrp['mechanism'] = syntype.__str__().strip('()')
                    for key, value in synparams.items():
                        subgrp[key] = value
            else:
                COMM.gather(syn_idx_pos)

        return COMM.bcast([conncount, syncount])

    def enable_extracellular_stimulation(self, electrode, t_ext=None, n=1,
                                         seed=None):
        raise NotImplementedError()

    def simulate(self, probes=None,
                 rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
                 rec_pop_contributions=False,
                 rec_variables=[], variable_dt=False, atol=0.001,
                 to_memory=True, to_file=False,
                 file_name='OUTPUT.h5',
                 **kwargs):
        """
        This is the main function running the simulation of the network model.

        Parameters
        ----------
        probes: list of :obj:, optional
            None or list of LFPykit.RecExtElectrode like object instances that
            each have a public method `get_transformation_matrix` returning
            a matrix that linearly maps each compartments' transmembrane
            current to corresponding measurement as

            .. math:: \\mathbf{P} = \\mathbf{M} \\mathbf{I}

        rec_imem: bool
            If true, segment membrane currents will be recorded
            If no electrode argument is given, it is necessary to
            set rec_imem=True in order to calculate LFP later on.
            Units of (nA).
        rec_vmem: bool
            record segment membrane voltages (mV)
        rec_ipas: bool
            record passive segment membrane currents (nA)
        rec_icap: bool
            record capacitive segment membrane currents (nA)
        rec_isyn: bool
            record synaptic currents of from Synapse class (nA)
        rec_vmemsyn: bool
            record membrane voltage of segments with Synapse (mV)
        rec_istim: bool
            record currents of StimIntraElectrode (nA)
        rec_pop_contributions: bool
            If True, compute and return single-population contributions to
            the extracellular potential during simulation time
        rec_variables: list of str
            variables to record, i.e arg=['cai', ]
        variable_dt: boolean
            use variable timestep in NEURON. Can not be combimed with `to_file`
        atol: float
            absolute tolerance used with NEURON variable timestep
        to_memory: bool
            Simulate to memory. Only valid with `probes=[<probe>, ...]`, which
            store measurements to -> <probe>.data
        to_file: bool
            only valid with `probes=[<probe>, ...]`, saves measurement in
            hdf5 file format.
        file_name: str
            If to_file is True, file which measurements will be
            written to. The file format is HDF5, default is "OUTPUT.h5", put
            in folder Network.OUTPUTPATH
        **kwargs:  keyword argument dict values passed along to function
                    `__run_simulation_with_probes()`, containing some or all of
                    the boolean flags: `use_ipas`, `use_icap`, `use_isyn`
                    (defaulting to `False`).

        Returns
        -------
        events
            Dictionary with keys `times` and `gids`, where values are
            ndarrays with detected spikes and global neuron identifiers

        Raises
        ------
        Exception
            if `CVode().use_fast_imem()` method not found
        AssertionError
            if rec_pop_contributions==True and probes==None
        """
        # set up integrator, use the CVode().fast_imem method by default
        # as it doesn't hurt sim speeds much if at all.
        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError:
            raise Exception('neuron.h.CVode().use_fast_imem() not found. '
                            'Please update NEURON to v.7.4 or newer')

        # test some of the inputs
        if probes is None:
            assert rec_pop_contributions is False, \
                'rec_pop_contributions can not be True when probes is None'

        if not variable_dt:
            dt = self.dt
        else:
            dt = None

        for name in self.population_names:
            for cell in self.populations[name].cells:
                cell._set_soma_volt_recorder(dt)
                if rec_imem:
                    cell._set_imem_recorders(dt)
                if rec_vmem:
                    cell._set_voltage_recorders(dt)
                if rec_ipas:
                    cell._set_ipas_recorders(dt)
                if rec_icap:
                    cell._set_icap_recorders(dt)
                if len(rec_variables) > 0:
                    cell._set_variable_recorders(rec_variables)

        # run fadvance until t >= tstop, and calculate LFP if asked for
        if probes is None and not rec_pop_contributions and not to_file:
            if not rec_imem:
                if self.verbose:
                    print("rec_imem==False, not recording membrane currents!")
            self.__run_simulation(cvode, variable_dt, atol)
        else:
            self.__run_simulation_with_probes(
                cvode=cvode,
                probes=probes,
                variable_dt=variable_dt,
                atol=atol,
                to_memory=to_memory,
                to_file=to_file,
                file_name='tmp_output_RANK_{:03d}.h5',
                rec_pop_contributions=rec_pop_contributions,
                **kwargs)

        for name in self.population_names:
            for cell in self.populations[name].cells:
                # somatic trace
                cell.somav = np.array(cell.somav)
                if rec_imem:
                    cell._calc_imem()
                if rec_ipas:
                    cell._calc_ipas()
                if rec_icap:
                    cell._calc_icap()
                if rec_vmem:
                    cell._collect_vmem()
                if rec_isyn:
                    cell._collect_isyn()
                if rec_vmemsyn:
                    cell._collect_vsyn()
                if rec_istim:
                    cell._collect_istim()
                if len(rec_variables) > 0:
                    cell._collect_rec_variables(rec_variables)
                if hasattr(cell, '_hoc_netstimlist'):
                    del cell._hoc_netstimlist

        # Collect spike trains across all RANKs to RANK 0
        for name in self.population_names:
            population = self.populations[name]
            for i in range(len(population.spike_vectors)):
                population.spike_vectors[i] = \
                    np.array(population.spike_vectors[i])
        if RANK == 0:
            times = []
            gids = []
            for i, name in enumerate(self.population_names):
                times.append([])
                gids.append([])
                times[i] += [x for x in self.populations[name].spike_vectors]
                gids[i] += [x for x in self.populations[name].gids]
                for j in range(1, SIZE):
                    times[i] += COMM.recv(source=j, tag=13)
                    gids[i] += COMM.recv(source=j, tag=14)
        else:
            times = None
            gids = None
            for name in self.population_names:
                COMM.send([x for x in self.populations[name].spike_vectors],
                          dest=0, tag=13)
                COMM.send([x for x in self.populations[name].gids],
                          dest=0, tag=14)

        # create final output file, summing up single RANK output from
        # temporary files
        if to_file and probes is not None:
            op = MPI.SUM
            fname = os.path.join(self.OUTPUTPATH,
                                 'tmp_output_RANK_{:03d}.h5'.format(RANK))
            f0 = h5py.File(fname, 'r')
            if RANK == 0:
                f1 = h5py.File(os.path.join(self.OUTPUTPATH, file_name), 'w')
            dtype = []
            for key, value in f0[list(f0.keys())[0]].items():
                dtype.append((str(key), float))
            for grp in f0.keys():
                if RANK == 0:
                    # get shape from the first dataset
                    # (they should all be equal):
                    for value in f0[grp].values():
                        shape = value.shape
                        continue
                    f1[grp] = np.zeros(shape, dtype=dtype)
                for key, value in f0[grp].items():
                    if RANK == 0:
                        recvbuf = np.zeros(shape, dtype=float)
                    else:
                        recvbuf = None
                    COMM.Reduce(value[()].astype(float), recvbuf,
                                op=op, root=0)
                    if RANK == 0:
                        f1[grp][key] = recvbuf
            f0.close()
            if RANK == 0:
                f1.close()
            os.remove(fname)

        if probes is not None:
            if to_memory:
                # communicate and sum up measurements on each probe before
                # returing spike times and corresponding gids:
                for probe in probes:
                    probe.data = ReduceStructArray(probe.data)

        return dict(times=times, gids=gids)

    def __create_network_dummycell(self):
        """
        set up parameters for a DummyCell object, allowing for computing
        the sum of all single-cell LFPs at each timestep, essentially
        creating one supercell with all segments of all cell objects
        present on this RANK.
        """
        # compute the total number of segments per population on this RANK
        nsegs = [[cell.totnsegs for cell in self.populations[name].cells]
                 for name in self.population_names]
        for i, nseg in enumerate(nsegs):
            if nseg == []:
                nsegs[i] = [0]
        for i, y in enumerate(nsegs):
            nsegs[i] = np.sum(y)
        nsegs = np.array(nsegs, dtype=int)

        totnsegs = nsegs.sum()
        x = np.empty((0, 2))
        y = np.empty((0, 2))
        z = np.empty((0, 2))
        d = np.array([])
        area = np.array([])
        length = np.array([])

        somainds = np.array([], dtype=int)
        nseg = 0

        for name in self.population_names:
            for cell in self.populations[name].cells:
                x = np.r_[x, cell.x]
                y = np.r_[y, cell.y]
                z = np.r_[z, cell.z]
                d = np.r_[d, cell.d]
                area = np.r_[area, cell.area]
                length = np.r_[length, cell.length]

                somainds = np.r_[somainds, cell.get_idx("soma") + nseg]
                nseg += cell.totnsegs

        # return number of segments per population and DummyCell object
        return nsegs, DummyCell(totnsegs, x, y, z, d, area, length, somainds)

    def __run_simulation(self, cvode, variable_dt=False, atol=0.001):
        """
        Running the actual simulation in NEURON, simulations in NEURON
        are now interruptable.

        Parameters
        ----------
        cvode: neuron.h.CVode() object
        variable_dt: bool
            switch for variable-timestep method
        atol: float
            absolute tolerance with CVode for variable time-step method
        """
        # set maximum integration step, it is necessary for communication of
        # spikes across RANKs to occur.
        self.pc.set_maxstep(10)

        # time resolution
        neuron.h.dt = self.dt

        # needed for variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
        else:
            cvode.active(0)

        # initialize state
        neuron.h.finitialize(self.v_init * units.mV)

        # initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()

        # Starting simulation at tstart
        neuron.h.t = self.tstart

        # only needed if LFPy.Synapse classes are used.
        for name in self.population_names:
            for cell in self.populations[name].cells:
                cell._load_spikes()

        # advance simulation until tstop
        neuron.h.continuerun(self.tstop * units.ms)

    def __run_simulation_with_probes(self, cvode,
                                     probes=None,
                                     variable_dt=False,
                                     atol=0.001,
                                     rtol=0.,
                                     to_memory=True,
                                     to_file=False,
                                     file_name=None,
                                     use_ipas=False, use_icap=False,
                                     use_isyn=False,
                                     rec_pop_contributions=False
                                     ):
        """
        Running the actual simulation in NEURON with list of probes.
        Each object in `probes` must have a public method
        `get_transformation_matrix` which returns a linear mapping of
        transmembrane currents to corresponding measurement.

        Parameters
        ----------
        cvode: neuron.h.CVode() object
        probes: list of :obj:, optional
            None or list of LFPykit.RecExtElectrode like object instances that
            each have a public method `get_transformation_matrix` returning
            a matrix that linearly maps each compartments' transmembrane
            current to corresponding measurement as

            .. math:: \\mathbf{P} = \\mathbf{M} \\mathbf{I}

        variable_dt: bool
            switch for variable-timestep method
        atol: float
            absolute tolerance with CVode for variable time-step method
        rtol: float
            relative tolerance with CVode for variable time-step method
        to_memory: bool
            Boolean flag for computing extracellular potentials,
            default is True.
            If True, the corresponding <probe>.data attribute will be set.
        to_file: bool or None
            Boolean flag for computing extracellular potentials to file
            <OUTPUTPATH/file_name>, default is False. Raises an Exception if
            `to_memory` is True.
        file_name: formattable str
            If to_file is True, file which extracellular potentials will be
            written to. The file format is HDF5, default is
            "output_RANK_{:03d}.h5". The output is written per RANK, and the
            RANK # will be inserted into the corresponding file name.
        use_ipas: bool
            if True, compute the contribution to extracellular potentials
            across the passive leak channels embedded in the cells membranes
            summed over populations
        use_icap: bool
            if True, compute the contribution to extracellular potentials
            across the membrane capacitance embedded in the cells membranes
            summed over populations
        use_isyn: bool
            if True, compute the contribution to extracellular potentials
            across the excitatory and inhibitory synapses embedded in the cells
            membranes summed over populations
        rec_pop_contributions: bool
            if True, compute and return single-population contributions to the
            extracellular potential during each time step of the simulation

        Returns
        -------

        Raises
        ------
        Exception:
        - `if to_memory == to_file == True`
        - `if to_file == True and file_name is None`
        - `if to_file == variable_dt == True`
        - `if <probe>.cell is not None`
        """
        if to_memory and to_file:
            raise Exception('to_memory and to_file can not both be True')
        if to_file and file_name is None:
            raise Exception
        # create a dummycell object lumping together needed attributes
        # for calculation of extracellular potentials etc. The population_nsegs
        # array is used to slice indices such that single-population
        # contributions to the potential can be calculated.
        population_nsegs, network_dummycell = self.__create_network_dummycell()

        # set cell attribute on each probe, assuming that each probe was
        # instantiated with argument cell=None
        for probe in probes:
            if probe.cell is None:
                probe.cell = network_dummycell
            else:
                raise Exception('{}.cell!=None'.format(probe.__class__))

        # create list of transformation matrices; one for each probe
        transforms = []
        if probes is not None:
            for probe in probes:
                transforms.append(probe.get_transformation_matrix())

        # reset probe.cell to None, as it is no longer needed
        for probe in probes:
            probe.cell = None

        # set maximum integration step, it is necessary for communication of
        # spikes across RANKs to occur.
        # NOTE: Should this depend on the minimum delay in the network?
        self.pc.set_maxstep(10)

        # Initialize NEURON simulations of cell object
        neuron.h.dt = self.dt

        # needed for variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
        else:
            cvode.active(0)

        # initialize state
        neuron.h.finitialize(self.v_init * units.mV)

        # use fast calculation of transmembrane currents
        cvode.use_fast_imem(1)

        # initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()

        # Starting simulation at tstart
        neuron.h.t = self.tstart

        # create list of cells across all populations to simplify loops
        cells = []
        for name in self.population_names:
            cells += self.populations[name].cells

        # load spike times from NetCon, only needed if LFPy.Synapse class
        # is used
        for cell in cells:
            cell._load_spikes()

        # define data type for structured arrays dependent on the boolean
        # arguments
        dtype = [('imem', float)]
        if use_ipas:
            dtype += [('ipas', float)]
        if use_icap:
            dtype += [('icap', float)]
        if use_isyn:
            dtype += [('isyn_e', float), ('isyn_i', float)]
        if rec_pop_contributions:
            dtype += list(zip(self.population_names,
                              [float] * len(self.population_names)))

        # setup list of structured arrays for all extracellular potentials
        # at each contact from different source terms and subpopulations
        if to_memory:
            for probe, M in zip(probes, transforms):
                probe.data = np.zeros((M.shape[0],
                                       int(self.tstop / self.dt) + 1),
                                      dtype=dtype)

        # signals for each probe will be stored here during simulations
        if to_file:
            # ensure right ending:
            if file_name.split('.')[-1] != 'h5':
                file_name += '.h5'
            outputfile = h5py.File(os.path.join(self.OUTPUTPATH,
                                                file_name.format(RANK)), 'w')

            # define unique group names for each probe
            names = []
            for probe, M in zip(probes, transforms):
                name = probe.__class__.__name__
                i = 0
                while True:
                    if name + '{}'.format(i) not in names:
                        names.append(name + '{}'.format(i))
                        break
                    i += 1

            # create groups
            for i, (name, probe, M) in enumerate(zip(names, probes,
                                                     transforms)):
                # can't do it this way until h5py issue #740
                # (https://github.com/h5py/h5py/issues/740) is fixed:
                # outputfile['{}'.format(name)] = np.zeros((M.shape[0],
                #     int(network.tstop / network.dt) + 1), dtype=dtype)
                probe.data = outputfile.create_group('{}'.format(name))
                for key, val in dtype:
                    probe.data[key] = np.zeros((M.shape[0],
                                                int(self.tstop / self.dt)
                                                + 1),
                                               dtype=val)

        # temporary vector to store membrane currents at each timestep:
        imem = np.zeros(network_dummycell.totnsegs, dtype=dtype)

        def get_imem(imem):
            '''helper function to gather currents across all cells
            on this RANK'''
            i = 0
            totnsegs = 0
            if use_isyn:
                imem['isyn_e'] = 0.  # must reset these for every iteration
                imem['isyn_i'] = 0.  # because we sum over synapses
            for cell in cells:
                for sec in cell.allseclist:
                    for seg in sec:
                        imem['imem'][i] = seg.i_membrane_
                        if use_ipas:
                            imem['ipas'][i] = seg.i_pas
                        if use_icap:
                            imem['icap'][i] = seg.i_cap
                        i += 1

                if use_isyn:
                    for idx, syn in zip(cell.synidx, cell.netconsynapses):
                        if hasattr(syn, 'e') and syn.e > -50:
                            imem['isyn_e'][idx + totnsegs] += syn.i
                        else:
                            imem['isyn_i'][idx + totnsegs] += syn.i

                totnsegs += cell.totnsegs
            return imem

        # run fadvance until time limit, and calculate LFPs for each timestep
        tstep = 0
        while neuron.h.t < self.tstop:
            if neuron.h.t >= 0:
                imem = get_imem(imem)

                for j, (probe, M) in enumerate(zip(probes, transforms)):
                    probe.data['imem'][:, tstep] = M @ imem['imem']
                    if use_ipas:
                        probe.data['ipas'][:, tstep] = \
                            M @ (imem['ipas'] * network_dummycell.area * 1E-2)
                    if use_icap:
                        probe.data['icap'][:, tstep] = \
                            M @ (imem['icap'] * network_dummycell.area * 1E-2)
                    if use_isyn:
                        probe.data['isyn_e'][:, tstep] = M @ imem['isyn_e']
                        probe.data['isyn_i'][:, tstep] = M @ imem['isyn_i']

                if rec_pop_contributions:
                    for j, (probe, M) in enumerate(zip(probes, transforms)):
                        k = 0  # counter
                        for nsegs, pop_name in zip(population_nsegs,
                                                   self.population_names):
                            cellinds = np.arange(k, k + nsegs)
                            probe.data[pop_name][:, tstep] = \
                                M[:, cellinds] @ imem['imem'][cellinds, ]
                            k += nsegs

                tstep += 1
            neuron.h.fadvance()
            if neuron.h.t % 100. == 0.:
                if RANK == 0:
                    print('t = {} ms'.format(neuron.h.t))

        try:
            # calculate LFP after final fadvance(), skipped if IndexError is
            # encountered
            imem = get_imem(imem)

            for j, (probe, M) in enumerate(zip(probes, transforms)):
                probe.data['imem'][:, tstep] = M @ imem['imem']
                if use_ipas:
                    probe.data['ipas'][:, tstep] = \
                        M @ (imem['ipas'] * network_dummycell.area * 1E-2)
                if use_icap:
                    probe.data['icap'][:, tstep] = \
                        M @ (imem['icap'] * network_dummycell.area * 1E-2)
                if use_isyn:
                    probe.data['isyn_e'][:, tstep] = M @ imem['isyn_e']
                    probe.data['isyn_i'][:, tstep] = M @ imem['isyn_i']

            if rec_pop_contributions:
                for j, (probe, M) in enumerate(zip(probes, transforms)):
                    k = 0  # counter
                    for nsegs, pop_name in zip(population_nsegs,
                                               self.population_names):
                        cellinds = np.arange(k, k + nsegs)
                        probe.data[pop_name][:, tstep] = \
                            M[:, cellinds] @ imem['imem'][cellinds, ]
                        k += nsegs
        except IndexError:
            pass

        if to_file:
            outputfile.close()


def ReduceStructArray(sendbuf, op=MPI.SUM):
    """
    simplify MPI Reduce for structured ndarrays with floating point numbers

    Parameters
    ----------
    sendbuf: structured ndarray
        Array data to be reduced (default: summed)
    op: mpi4py.MPI.Op object
        MPI_Reduce function. Default is mpi4py.MPI.SUM

    Returns
    -------
    recvbuf: structured ndarray or None
        Reduced array on RANK 0, None on all other RANKs
    """
    if RANK == 0:
        shape = sendbuf.shape
        dtype_names = sendbuf.dtype.names
    else:
        shape = None
        dtype_names = None
    shape = COMM.bcast(shape)
    dtype_names = COMM.bcast(dtype_names)

    if RANK == 0:
        reduced = np.zeros(shape,
                           dtype=list(zip(dtype_names,
                                          ['f8' for i in range(len(dtype_names)
                                                               )])))
    else:
        reduced = None
    for name in dtype_names:
        if RANK == 0:
            recvbuf = np.zeros(shape)
        else:
            recvbuf = None
        COMM.Reduce(np.array(sendbuf[name]), recvbuf, op=op, root=0)
        if RANK == 0:
            reduced[name] = recvbuf
    return reduced
