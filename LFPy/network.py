#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines classes and methods used by example_parallel_network.py script

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
from .templatecell import TemplateCell
import scipy.sparse as ss

# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


flattenlist = lambda lst: [item for sublist in lst for item in sublist]


################################################################################
# NetworkCell class that has a create_synapse method that
# creates a synapse on the target cell, and a create_spike_detector method that
# allows for connecting to a synapse on a target cell. All other methods and
# attributes are inherited from the standard LFPy.TemplateCell class
################################################################################
class NetworkCell(TemplateCell):
    """
    class NetworkCell

    Similar to `LFPy.TemplateCell` with the addition of some attributes and
    methods allowing for spike communication between parallel RANKs.

    This class allow using NEURON templates with some limitations.

    This takes all the same parameters as the Cell class, but requires three
    more template related parameters

    Parameters
    ----------
    morphology : str
        path to morphology file
    templatefile : str
        File with cell template definition(s)
    templatename : str
        Cell template-name used for this cell object
    templateargs : str
        Parameters provided to template-definition
    v_init : float
        Initial membrane potential. Default to -65.
    Ra : float
        axial resistance. Defaults to 150.
    cm : float
        membrane capacitance. Defaults to 1.0
    passive : bool
        Passive mechanisms are initialized if True. Defaults to True
    passive_parameters : dict
        parameter dictionary with values for the passive membrane mechanism in
        NEURON ('pas'). The dictionary must contain keys 'g_pas' and 'e_pas',
        like the default: passive_parameters=dict(g_pas=0.001, e_pas=-70)
    extracellular : bool
        switch for NEURON's extracellular mechanism. Defaults to False
    dt: float
        Simulation time step. Defaults to 2**-4
    tstart : float
        initialization time for simulation <= 0 ms. Defaults to 0.
    tstop : float
        stop time for simulation > 0 ms. Defaults to 100.
    nsegs_method : 'lambda100' or 'lambda_f' or 'fixed_length' or None
        nseg rule, used by NEURON to determine number of compartments.
        Defaults to 'lambda100'
    max_nsegs_length : float or None
        max segment length for method 'fixed_length'. Defaults to None
    lambda_f : int
        AC frequency for method 'lambda_f'. Defaults to 100
    d_lambda : float
        parameter for d_lambda rule. Defaults to 0.1
    delete_sections : bool
        delete pre-existing section-references. Defaults to True
    custom_code : list or None
        list of model-specific code files ([.py/.hoc]). Defaults to None
    custom_fun : list or None
        list of model-specific functions with args. Defaults to None
    custom_fun_args : list or None
        list of args passed to custom_fun functions. Defaults to None
    pt3d : bool
        use pt3d-info of the cell geometries switch. Defaults to False
    celsius : float or None
        Temperature in celsius. If nothing is specified here
        or in custom code it is 6.3 celcius
    verbose : bool
        verbose output switch. Defaults to False

    Examples
    --------

    >>> import LFPy
    >>> cellParameters = {
    >>>     'morphology' : '<path to morphology.hoc>',
    >>>     'templatefile' :  '<path to template_file.hoc>',
    >>>     'templatename' :  'templatename',
    >>>     'templateargs' :  None,
    >>>     'v_init' : -65,
    >>>     'cm' : 1.0,
    >>>     'Ra' : 150,
    >>>     'passive' : True,
    >>>     'passive_parameters' : {'g_pas' : 0.001, 'e_pas' : -65.},
    >>>     'dt' : 2**-3,
    >>>     'tstart' : 0,
    >>>     'tstop' : 50,
    >>> }
    >>> cell = LFPy.NetworkCell(**cellParameters)
    >>> cell.simulate()


    """
    def __init__(self, **args):
        """
        Initialization of class LFPy.NetworkCell.

        """
        TemplateCell.__init__(self, **args)

        # create list netconlist for spike detecting NetCon object(s)
        self.sd_netconlist = neuron.h.List()
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
        cell : object
            instantiation of class NetworkCell or similar
        sec : neuron.h.Section object,
            section reference on cell
        x : float in [0, 1],
            relative position along section
        syntype : hoc.HocObject
            NEURON synapse model reference, e.g., neuron.h.ExpSyn
        synparams : dict
            parameters for syntype, e.g., for neuron.h.ExpSyn we have:
                tau : float, synapse time constant
                e : float, synapse reversal potential
        assert_syn_values : bool
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
            rng.MCellRan4(np.random.randint(0, 2**32-1), np.random.randint(0, 2**32-1))
            rng.uniform(0, 1)
            syn.setRNG(rng) # used for e.g., stochastic synapse mechanisms (cf. BBP microcircuit portal files)
            cell.rng_list.append(rng) # must store ref to rng object
        cell.netconsynapses.append(syntype(x, sec=sec))

        for key, value in synparams.items():
            exec("cell.netconsynapses[-1].{} = {}".format(key, value))
            # check that synapses are parameterized correctly
            if assert_syn_values:
                try:
                    np.testing.assert_almost_equal(getattr(cell.netconsynapses[-1], key), value)
                except AssertionError:
                    raise AssertionError('{} = {} != {}'.format(key,
                                                                getattr(cell.netconsynapses[-1], key),
                                                                value))


    def create_spike_detector(self, target=None, threshold=-10.,
                     weight=0.0, delay=0.0):
        """
        Create spike-detecting NetCon object attached to the cell's soma
        midpoint, but this could be extended to having multiple spike-detection
        sites. The NetCon object created is attached to the cell's sd_netconlist
        attribute, and will be used by the Network class when creating
        connections between all presynaptic cells and postsynaptic cells on
        each local RANK.

        Parameters
        ----------
        target : None (default) or a NEURON point process
        threshold : float
            spike detection threshold
        weight : float
            connection weight (not used unless target is a point process)
        delay : float
            connection delay (not used unless target is a point process)
        """
        # create new NetCon objects for the connections. Activation times will
        # be triggered on the somatic voltage with a given threshold.
        for sec in self.somalist:
            self.sd_netconlist.append(neuron.h.NetCon(sec(0.5)._ref_v,
                                                target,
                                                sec=sec))
            self.sd_netconlist[-1].threshold = threshold
            self.sd_netconlist[-1].weight[0] = weight
            self.sd_netconlist[-1].delay = delay


class DummyCell(object):
    def __init__(self, totnsegs=0,
                 imem=np.array([[]]),
                 xstart=np.array([]), xmid=np.array([]), xend=np.array([]),
                 ystart=np.array([]), ymid=np.array([]), yend=np.array([]),
                 zstart=np.array([]), zmid=np.array([]), zend=np.array([]),
                 diam=np.array([]), area=np.array([]), somainds=np.array([])):
        """
        Dummy Cell object initialized with all attributes needed for LFP
        calculations using the LFPy.RecExtElectrode class and methods. This cell
        can be imagined as one "super" cell containing transmembrane currents
        generated by all NetworkCell segments on this RANK at once.


        Parameters
        ----------
        totnsegs : int
            total number of segments
        imem : ndarray
            totnsegs x ntimesteps array with transmembrane currents in nA
        xstart, ystart, zstart : ndarray
            arrays of length totnsegs with start (x,y,z) coordinate of segments
            in units of um
        xmid, ymid, zmid : ndarray
            midpoint coordinates of segments
        xend, yend, zend : ndarray
            endpoint coordinateso of segments
        diam : ndarray
            array of length totnsegs with segment diameters
        area : ndarray
            array of segment surface areas
        """
        # set attributes
        self.totnsegs = totnsegs
        self.imem = imem
        self.xstart = xstart
        self.xmid = xmid
        self.xend = xend
        self.ystart = ystart
        self.ymid = ymid
        self.yend = yend
        self.zstart = zstart
        self.zmid = zmid
        self.zend = zend
        self.diam = diam
        self.area = area
        self.somainds = somainds

    def get_idx(self, section="soma"):
        if section=="soma":
            return self.somainds
        else:
            raise ValueError('section argument must be "soma"')


class NetworkPopulation(object):
    def __init__(self, CWD=None, CELLPATH=None, first_gid=0, Cell=NetworkCell,
                 POP_SIZE=4, name='L5PC',
                 cell_args=dict(), pop_args=dict(),
                 rotation_args=dict(),
                 OUTPUTPATH='example_parallel_network'):
        """
        NetworkPopulation class representing a group of Cell objects distributed
        across RANKs.

        Parameters
        ----------
        CWD : path or None
            Current working directory
        CELLPATH: path or None
            Relative path from CWD to source files for cell model (morphology, hoc routines etc.)
        first_gid : int
            The global identifier of the first cell created in this population
            instance. The first_gid in the first population created should be 0
            and cannot exist in previously created NetworkPopulation instances
        Cell : class
            class defining a Cell object, see class NetworkCell above
        POP_SIZE : int
            number of cells in population
        name : str
            population name reference
        cell_args : dict
            keys and values for Cell object
        pop_args : dict
            keys and values for Network.draw_rand_pos assigning cell positions
        rotation_arg : dict
            default cell rotations around x and y axis on the form
            { 'x' : np.pi/2, 'y' : 0 }. Can only have the keys 'x' and 'y'.
            Cells are randomly rotated around z-axis using the Cell.set_rotation
            method.
        OUTPUTPATH : str
            path to output file destination
        """
        # set class attributes
        self.CWD = CWD
        self.CELLPATH = CELLPATH
        self.first_gid = first_gid
        self.Cell = Cell
        self.POP_SIZE = POP_SIZE
        self.name = name
        self.cell_args = cell_args
        self.pop_args = pop_args
        self.rotation_args = rotation_args
        self.OUTPUTPATH = OUTPUTPATH

        # create folder for output if it does not exist
        if RANK == 0:
            if not os.path.isdir(OUTPUTPATH):
                os.mkdir(OUTPUTPATH)
        COMM.Barrier()

        # container of Vector objects used to record times of action potentials
        self.spike_vectors = []

        # set up population of cells on this RANK
        self.gids = [(i+first_gid) for i in range(POP_SIZE) if (i+first_gid) % SIZE == RANK]

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
        self.rotations = np.random.uniform(0, np.pi*2, len(self.gids))
        assert('z' not in self.rotation_args.keys())
        for i, cell in enumerate(self.cells):
            cell.set_rotation(z=self.rotations[i], **self.rotation_args)

        # assign gid to each cell
        for gid, cell in zip(self.gids, self.cells):
            cell.gid = gid


        # gather gids, soma positions and cell rotations to RANK 0, and write
        # as structured array.
        if RANK == 0:
            populationData = flattenlist(COMM.gather(zip(self.gids, self.soma_pos, self.rotations)))

            # create structured array for storing data
            dtype = [('gid', 'i8'), ('x', float), ('y', float), ('z', float),
                     ('x_rot', float), ('y_rot', float), ('z_rot', float)]
            popDataArray = np.empty((len(populationData, )), dtype=dtype)
            for i, (gid, pos, z_rot) in enumerate(populationData):
                popDataArray[i]['gid'] = gid
                popDataArray[i]['x'] = pos['x']
                popDataArray[i]['y'] = pos['y']
                popDataArray[i]['z'] = pos['z']
                popDataArray[i]['x_rot'] = np.pi/2
                popDataArray[i]['y_rot'] = 0.
                popDataArray[i]['z_rot'] = z_rot

            # Dump to hdf5 file, append to file if it exists
            f = h5py.File(os.path.join(self.OUTPUTPATH,
                                       'cell_positions_and_rotations.h5'), 'a')
            # delete old entry if it exist
            if self.name in f.keys():
                del f[self.name]
                try:
                    assert self.name not in f.keys()
                except AssertionError:
                    raise AssertionError
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
        POP_SIZE : int
            Population size
        radius : float
            Radius of population.
        loc : float
            expected mean depth of somas of population.
        scale : float
            expected standard deviation of depth of somas of population.
        cap : None, float or length to list of floats
            if float, cap distribution between [loc-cap, loc+cap),
            if list, cap distribution between [loc-cap[0], loc+cap[1]]


        Returns
        -------
        soma_pos : list
            List of dicts of len POP_SIZE
            where dict have keys x, y, z specifying
            xyz-coordinates of cell at list entry `i`.


        """

        x = np.empty(POP_SIZE)
        y = np.empty(POP_SIZE)
        z = np.empty(POP_SIZE)
        for i in range(POP_SIZE):
            x[i] = (np.random.rand()-0.5) * radius*2
            y[i] = (np.random.rand()-0.5) * radius*2
            while np.sqrt(x[i]**2 + y[i]**2) >= radius:
                x[i] = (np.random.rand()-0.5)*radius*2
                y[i] = (np.random.rand()-0.5)*radius*2
        z = np.random.normal(loc=loc, scale=scale, size=POP_SIZE)
        if cap is not None:
            if type(cap) in [float, np.float, np.float32, np.float64]:
                while not np.all((z >= loc-cap) & (z < loc+cap)):
                    inds = (z < loc-cap) ^ (z > loc+cap)
                    z[inds] = np.random.normal(loc=loc, scale=scale,
                                               size=inds.sum())
            elif type(cap) is list:
                try:
                    assert(len(cap) == 2)
                except AssertionError:
                    raise AssertionError('cap = {} is not a length 2 list'.format(float))
                while not np.all((z >= loc-cap[0]) & (z < loc+cap[1])):
                    inds = (z < loc-cap[0]) ^ (z > loc+cap[1])
                    z[inds] = np.random.normal(loc=loc, scale=scale,
                                               size=inds.sum())
            else:
                raise Exception('cap = {} is not None, a float or length 2 list of floats'.format(float))

        soma_pos = []
        for i in range(POP_SIZE):
            soma_pos.append({'x' : x[i], 'y' : y[i], 'z' : z[i]})

        return soma_pos


class Network(object):
    def __init__(self, dt=0.1, tstart=0., tstop=1000., v_init=-65., celsius=6.3,
                 OUTPUTPATH='example_parallel_network',
                 verbose=False):
        """
        Network class, creating distributed populations of cells of
        type Cell and handling connections between cells in the respective
        populations.

        Parameters
        ----------
        dt : float
            Simulation timestep size
        tstart : float
            Start time of simulation
        tstop : float
            End time of simulation
        v_init : float
            Membrane potential set at first timestep across all cells
        celsius : float
            Global control of temperature, affect channel kinetics.
            It will also be forced when creating the different Cell objects, as
            LFPy.Cell and LFPy.TemplateCell also accept the same keyword
            argument.
        verbose : bool
            if True, print out misc. messages


        """
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
        # with each cell's list of netcons)
        self.netconlist = neuron.h.List()


        # The different populations in the Network will be collected in
        # a dictionary of NetworkPopulation object, where the keys represent the
        # population name. The names are also put in a list ordered according to
        # order populations are created in (as some operations rely on this)
        self.populations = dict()
        self.population_names = []



    def create_population(self, CWD=None, CELLPATH=None, Cell=NetworkCell,
                          POP_SIZE=4, name='L5PC',
                          cell_args=dict(), pop_args=dict(),
                          rotation_args=dict()):
        """
        Create and append a distributed POP_SIZE-sized population of cells of
        type Cell with the corresponding name. Cell-object references, gids on
        this RANK, population size POP_SIZE and names will be added to the lists
        Network.gids, Network.cells, Network.sizes and Network.names,
        respectively

        Parameters
        ----------
        CWD : path
            Current working directory
        CELLPATH: path
            Relative path from CWD to source files for cell model (morphology, hoc routines etc.)
        Cell : class
            class defining a Cell-like object, see class NetworkCell
        POP_SIZE : int
            number of cells in population
        name : str
            population name reference
        cell_args : dict
            keys and values for Cell object
        pop_args : dict
            keys and values for Network.draw_rand_pos assigning cell positions
        rotation_arg : dict
            default cell rotations around x and y axis on the form
            { 'x' : np.pi/2, 'y' : 0 }. Can only have the keys 'x' and 'y'.
            Cells are randomly rotated around z-axis using the Cell.set_rotation
            method.

        """
        try:
            assert name not in self.populations.keys()
        except AssertionError:
            raise AssertionError('population name {} already taken'.format(name))

        # compute the first global id of this new population, based
        # on population sizes of existing populations
        first_gid = 0
        for p in self.populations.values():
            first_gid += p.POP_SIZE

        # create NetworkPopulation object
        population = NetworkPopulation(CWD=CWD, CELLPATH=CELLPATH, first_gid=first_gid,
                                Cell=Cell,
                                POP_SIZE=POP_SIZE, name=name,
                                cell_args=cell_args, pop_args=pop_args,
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
            self.pc.cell(gid, cell.sd_netconlist[-1])

            # record spike events
            population.spike_vectors.append(neuron.h.Vector())
            cell.sd_netconlist[-1].record(population.spike_vectors[-1])

        # add population object to dictionary of populations
        self.populations[name] = population

        # append population name to list (Network.populations.keys() not unique)
        self.population_names.append(name)


    def get_connectivity_rand(self, pre='L5PC', post='L5PC', connprob = 0.2):
        """
        Dummy function creating a (boolean) cell to cell connectivity matrix
        between pre and postsynaptic populations.

        Connections are drawn randomly between presynaptic cell gids in
        population 'pre' and postsynaptic cell gids in 'post' on this RANK with
        a fixed connection probability. self-connections are disabled if
        presynaptic and postsynaptic populations are the same.

        Parameters
        ----------
        pre : str
            presynaptic population name
        post : str
            postsynaptic population name
        connprob : float in [0, 1]
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
            C = np.random.rand(n_pre, gids.size) < connprob
            if pre == post:
                # avoid self connections.
                gids_pre, gids_post = np.where(C)
                gids_pre += self.populations[pre].first_gid
                gids_post *= SIZE # asssume round-robin distribution of gids
                gids_post += self.populations[post].gids[0]
                inds = gids_pre == gids_post
                gids_pre = gids_pre[inds == False]
                gids_pre -= self.populations[pre].first_gid
                gids_post = gids_post[inds == False]
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
                                  fun=[stats.norm]*2,
                                  funargs=[dict(loc=0, scale=100)]*2,
                                  funweights=[0.5]*2,
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
        pre : str
            presynaptic population name
        post : str
            postsynaptic population name
        connectivity : ndarray / (scipy.sparse array)
            boolean connectivity matrix between pre and post.
        syntype : hoc.HocObject
            reference to NEURON synapse mechanism, e.g., neuron.h.ExpSyn
        synparams : dict
            dictionary of parameters for synapse mechanism, keys 'e', 'tau' etc.
        weightfun : function
            function used to draw weights from a numpy.random distribution
        weightargs : dict
            parameters passed to weightfun
        minweight : float,
            minimum weight in units of nS
        delayfun : function
            function used to draw delays from a numpy.random distribution
        delayargs : dict
            parameters passed to delayfun
        mindelay : float,
            minimum delay in multiples of dt
        multapsefun : function or None
            function reference, e.g., numpy.random.normal used to draw a number
            of synapses for a cell-to-cell connection. If None, draw only one
            connection
        multapseargs : dict
            arguments passed to multapsefun
        syn_pos_args : dict
            arguments passed to inherited LFPy.Cell method
            NetworkCell.get_rand_idx_area_and_distribution_norm to find
            synapse locations.
        save_connections : bool
            if True (default False), save instantiated connections to HDF5 file
            "Network.OUTPUTPATH/synapse_positions.h5" as dataset "<pre>:<post>"
            using a structured ndarray with dtype
            [('gid', 'i8'), ('x', float), ('y', float), ('z', float)]
            where gid is postsynaptic cell id, and x,y,z the corresponding
            midpoint coordinates of the target compartment.
        """
        # set up connections from all cells in presynaptic to post across RANKs
        n0 = self.populations[pre].first_gid
        # gids of presynaptic neurons:
        pre_gids = np.arange(n0, n0 + self.populations[pre].POP_SIZE)

        # count connections and synapses made on this RANK
        conncount = connectivity.astype(int).sum()
        syncount = 0

        # keep track of synapse positions for this connect
        # call on this rank such that these can be communicated and stored
        syn_idx_pos = []

        # iterate over gids on this RANK and create connections
        for i, (post_gid, cell) in enumerate(zip(self.populations[post].gids, self.populations[post].cells)):
            # do NOT iterate over all possible presynaptic neurons
            for pre_gid in pre_gids[connectivity[:, i]]:
                # throw a warning if sender neuron is identical to receiving neuron
                if post_gid == pre_gid:
                    print('connecting cell w. gid {} to itself (RANK {})'.format(post_gid, RANK))

                # assess number of synapses
                if multapsefun is None:
                    nidx = 1
                else:
                    nidx = 0
                    j = 0
                    while nidx <= 0 and j < 1000:
                        nidx = int(multapsefun(**multapseargs))
                        j += 1
                    if j == 1000:
                        raise Exception('change multapseargs as no positive synapse count was found in 1000 trials')

                # find synapse locations and corresponding section names
                idxs = cell.get_rand_idx_area_and_distribution_norm(nidx=nidx, **syn_pos_args)
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

                for i, ((idx, secname, x), weight, delay) in enumerate(zip(secs, weights, delays)):
                    cell.create_synapse(cell,
                                        # TODO: Find neater way of accessing Section reference, this seems slow
                                        sec=list(cell.allseclist)[np.where(np.array(cell.allsecnames)==secname)[0][0]],
                                        x=x, syntype=syntype,
                                        synparams=synparams)
                    # connect up NetCon object
                    nc = self.pc.gid_connect(pre_gid, cell.netconsynapses[-1])
                    nc.weight[0] = weight
                    nc.delay = delays[i]
                    self.netconlist.append(nc)

                    # store also synapse indices allowing for computing LFPs from syn.i
                    cell.synidx.append(idx)

                    # store gid and xyz-coordinate of synapse positions
                    syn_idx_pos.append((cell.gid, cell.xmid[idx], cell.ymid[idx], cell.zmid[idx]))

                syncount += nidx

        conncount = COMM.reduce(conncount, op=MPI.SUM, root=0)
        syncount = COMM.reduce(syncount, op=MPI.SUM, root=0)

        if RANK == 0:
            print('Connected population {} to {} by {} connections and {} synapses'.format(pre, post, conncount, syncount))

        else:
            conncount = None
            syncount = None


        # gather and write syn_idx_pos data
        if save_connections:
            if RANK == 0:
                synData = flattenlist(COMM.gather(syn_idx_pos))

                # convert to structured array
                dtype = [('gid', 'i8'), ('x', float), ('y', float), ('z', float)]
                synDataArray = np.empty((len(synData), ), dtype=dtype)
                for i, (gid, x, y, z) in enumerate(synData):
                    synDataArray[i]['gid'] = gid
                    synDataArray[i]['x'] = x
                    synDataArray[i]['y'] = y
                    synDataArray[i]['z'] = z
                # Dump to hdf5 file, append to file if entry exists
                f = h5py.File(os.path.join(self.OUTPUTPATH,
                                           'synapse_positions.h5'), 'a')
                key = '{}:{}'.format(pre, post)
                if key in f.keys():
                    del f[key]
                    try:
                        assert key not in f.keys()
                    except AssertionError:
                        raise AssertionError
                f[key] = synDataArray
                f.close()
            else:
                COMM.gather(syn_idx_pos)

        return COMM.bcast([conncount, syncount])

    def enable_extracellular_stimulation(self, electrode, t_ext=None, n=1, seed=None):
        raise NotImplementedError()

    def simulate(self, electrode=None, rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
                 rec_current_dipole_moment=False,
                 rec_pop_contributions=False,
                 rec_variables=[], variable_dt=False, atol=0.001,
                 to_memory=True, to_file=False,
                 file_name='OUTPUT.h5',
                 dotprodcoeffs=None, **kwargs):
        """
        This is the main function running the simulation of the network model.

        Parameters
        ----------
        electrode:
            Either an LFPy.RecExtElectrode object or a list of such.
                    If supplied, LFPs will be calculated at every time step
                    and accessible as electrode.LFP. If a list of objects
                    is given, accessible as electrode[0].LFP etc.
        rec_imem:   If true, segment membrane currents will be recorded
                    If no electrode argument is given, it is necessary to
                    set rec_imem=True in order to calculate LFP later on.
                    Units of (nA).
        rec_vmem:   record segment membrane voltages (mV)
        rec_ipas:   record passive segment membrane currents (nA)
        rec_icap:   record capacitive segment membrane currents (nA)
        rec_isyn:   record synaptic currents of from Synapse class (nA)
        rec_vmemsyn:    record membrane voltage of segments with Synapse(mV)
        rec_istim:  record currents of StimIntraElectrode (nA)
        rec_current_dipole_moment : bool
            If True, compute and record current-dipole moment from
            transmembrane currents as in Linden et al. (2010) J Comput Neurosci,
            DOI: 10.1007/s10827-010-0245-4. Will set the `LFPy.Cell` attribute
            `current_dipole_moment` as n_timesteps x 3 `ndarray` where the
            last dimension contains the x,y,z components of the dipole moment.
        rec_pop_contributions : bool
            If True, compute and return single-population contributions to
            the extracellular potential during simulation time
        rec_variables: list of variables to record, i.e arg=['cai', ]
        variable_dt: boolean, using variable timestep in NEURON
        atol:       absolute tolerance used with NEURON variable timestep
        to_memory:  only valid with electrode, store lfp in -> electrode.LFP
        to_file:    only valid with electrode, save LFPs in hdf5 file format
        file_name : str
            If to_file is True, file which extracellular potentials will be
            written to. The file format is HDF5, default is "OUTPUT.h5", put
            in folder Network.OUTPUTPATH
        dotprodcoeffs :  list of N x Nseg ndarray. These arrays will at
                    every timestep be multiplied by the membrane currents.
                    Presumably useful for memory efficient csd or lfp calcs
        **kwargs :  keyword argument dict values passed along to function
                    _run_simulation_with_electrode(), containing some or all of
                    the boolean flags: use_ipas, use_icap, use_isyn
                    (defaulting to 'False').

        Returns
        -------
        SPIKES : dict
            the first returned argument is a dictionary with keys 'gids' and
            'times'. Each item is a nested list of len(Npop) times N_X where N_X
            is the corresponding population size. Each entry is a np.ndarray
            containing the spike times of each cell in the nested list in item
            'gids'
        OUTPUT : list of ndarray
            if parameters electrode is not None and/or dotprodcoeffs is not
            None, contains the
            [electrode.LFP, ...., (dotprodcoeffs[0] dot I)(t), ...]
            The first output is a structured array, so OUTPUT[0]['imem']
            corresponds to the total LFP and the other the per-population
            contributions.
        P : ndarray
            if rec_current_dipole_moment==True, contains the x,y,z-components of
            current-dipole moment from transmembrane currents summed up over
            all populations

        """
        # set up integrator, use the CVode().fast_imem method by default
        # as it doesn't hurt sim speeds much if at all.
        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError as ae:
            raise Exception('neuron.h.CVode().use_fast_imem() not found. Please update NEURON to v.7.4 or newer')

        # test some of the inputs
        try:
            if electrode is None:
                assert(rec_pop_contributions is False)
        except AssertionError:
            raise AssertionError('rec_pop_contributions can not be True when electrode is None')

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
                # if rec_current_dipole_moment:
                #     self._set_current_dipole_moment_array()
                if len(rec_variables) > 0:
                    cell._set_variable_recorders(rec_variables)

        #run fadvance until t >= tstop, and calculate LFP if asked for
        if electrode is None and dotprodcoeffs is None and not rec_current_dipole_moment and not rec_pop_contributions and not to_file:
            if not rec_imem:
                if self.verbose:
                    print("rec_imem = {}, not recording membrane currents!".format(rec_imem))
            _run_simulation(self, cvode, variable_dt, atol)
        else:
            if dotprodcoeffs is not None:
                raise NotImplementedError
            LFP, P = _run_simulation_with_electrode(self, cvode=cvode,
                            electrode=electrode,
                            variable_dt=variable_dt,
                            atol=atol,
                            to_memory=to_memory,
                            to_file=to_file,
                            file_name='tmp_output_RANK_{:03d}.h5',
                            dotprodcoeffs=dotprodcoeffs,
                            rec_current_dipole_moment=rec_current_dipole_moment,
                            rec_pop_contributions=rec_pop_contributions,
                            **kwargs)

        for name in self.population_names:
            for cell in self.populations[name].cells:
                #somatic trace
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
                if hasattr(cell, 'netstimlist'):
                    del cell.netstimlist

        # Collect spike trains across all RANKs to RANK 0
        for name in self.population_names:
            population = self.populations[name]
            for i in range(len(population.spike_vectors)):
                population.spike_vectors[i] = np.array(population.spike_vectors[i])
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

        # create final output file, summing up single RANK output from temp files
        if to_file and electrode is not None:
            op=MPI.SUM
            fname = os.path.join(self.OUTPUTPATH, 'tmp_output_RANK_{:03d}.h5'.format(RANK))
            f0 = h5py.File(fname, 'r')
            if RANK == 0:
                f1 = h5py.File(os.path.join(self.OUTPUTPATH, file_name), 'w')
            dtype = []
            for key, value in f0[list(f0.keys())[0]].items():
                dtype.append((str(key), np.float))
            shape = value.shape
            for grp in f0.keys():
                if RANK == 0:
                    f1[grp] = np.zeros(shape, dtype=dtype)
                for key, value in f0[grp].items():
                    if RANK == 0:
                        recvbuf = np.zeros(shape, dtype=np.float)
                    else:
                        recvbuf = None
                    COMM.Reduce(value[()].astype(np.float), recvbuf, op=op, root=0)
                    if RANK == 0:
                        f1[grp][key] = recvbuf
            f0.close()
            if RANK == 0:
                f1.close()
            os.remove(fname)


        if electrode is None and dotprodcoeffs is None and not rec_current_dipole_moment and not rec_pop_contributions:
            return dict(times=times, gids=gids)
        else:
            # communicate and sum up LFPs and dipole moments:
            if LFP is not None:
                for i in range(len(LFP)):
                    LFP[i] = ReduceStructArray(LFP[i])
            if P is not None:
                P = ReduceStructArray(P)
            return dict(times=times, gids=gids), LFP, P


    def _create_network_dummycell(self):
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
        for i, y in enumerate(nsegs): nsegs[i] = np.sum(y)
        nsegs = np.array(nsegs, dtype=int)

        totnsegs = nsegs.sum()
        imem = np.eye(totnsegs)
        xstart = np.array([])
        xmid = np.array([])
        xend = np.array([])
        ystart = np.array([])
        ymid = np.array([])
        yend = np.array([])
        zstart = np.array([])
        zmid = np.array([])
        zend = np.array([])
        diam = np.array([])
        area = np.array([])

        somainds = np.array([], dtype=int)
        nseg = 0

        for name in self.population_names:
            for cell in self.populations[name].cells:
                xstart = np.r_[xstart, cell.xstart]
                ystart = np.r_[ystart, cell.ystart]
                zstart = np.r_[zstart, cell.zstart]
                xmid = np.r_[xmid, cell.xmid]
                ymid = np.r_[ymid, cell.ymid]
                zmid = np.r_[zmid, cell.zmid]
                xend = np.r_[xend, cell.xend]
                yend = np.r_[yend, cell.yend]
                zend = np.r_[zend, cell.zend]
                diam = np.r_[diam, cell.diam]
                area = np.r_[area, cell.area]

                somainds = np.r_[somainds, cell.get_idx("soma")+nseg]
                nseg += cell.totnsegs


        # return number of segments per population and DummyCell object
        return nsegs, DummyCell(totnsegs,
                         imem,
                         xstart, xmid, xend,
                         ystart, ymid, yend,
                         zstart, zmid, zend,
                         diam, area, somainds)


def _run_simulation(network, cvode, variable_dt=False, atol=0.001):
    """
    Running the actual simulation in NEURON, simulations in NEURON
    are now interruptable.

    Parameters
    ----------
    network : LFPy.Network object
        instantiation of class LFPy.Network
    cvode : neuron.h.CVode() object
    variable_dt : bool
        switch for variable-timestep method
    atol : float
        absolute tolerance with CVode for variable time-step method
    """
    # set maximum integration step, it is necessary for communication of
    # spikes across RANKs to occur.
    network.pc.set_maxstep(10)

    # time resolution
    neuron.h.dt = network.dt

    # needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
    else:
        cvode.active(0)

    # initialize state
    neuron.h.finitialize(network.v_init)

    # initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()

    # Starting simulation at tstart
    neuron.h.t = network.tstart

    # only needed if LFPy.Synapse classes are used.
    for name in network.population_names:
        for cell in network.populations[name].cells:
            cell._loadspikes()

    while neuron.h.t < network.tstop:
        neuron.h.fadvance()
        if neuron.h.t % 100 == 0:
            if RANK == 0:
                print('t = {} ms'.format(neuron.h.t))

    return


def _run_simulation_with_electrode(network, cvode,
                                   electrode=None,
                                   variable_dt=False,
                                   atol=0.001,
                                   to_memory=True,
                                   to_file=False,
                                   file_name=None,
                                   dotprodcoeffs=None,
                                   rec_current_dipole_moment=False,
                                   use_ipas=False, use_icap=False,
                                   use_isyn=False,
                                   rec_pop_contributions=False
                                   ):
    """
    Running the actual simulation in NEURON.
    electrode argument used to determine coefficient
    matrix, and calculate the LFP on every time step.

    Parameters
    ----------
    network : LFPy.Network object
        instantiation of class LFPy.Network
    cvode : neuron.h.CVode() object
    electrode : LFPy.RecExtElectrode object or None
        instantiation of class LFPy.RecExtElectrode for which extracellular
        potentials will be computed.
    variable_dt : bool
        switch for variable-timestep method
    atol : float
        absolute tolerance with CVode for variable time-step method
    to_memory : bool
        Boolean flag for computing extracellular potentials, default is True
    to_file : bool or None
        Boolean flag for computing extracellular potentials to file
        <OUTPUTPATH/file_name>, default is False
    file_name : formattable str
        If to_file is True, file which extracellular potentials will be
        written to. The file format is HDF5, default is
        "output_RANK_{:03d}.h5". The output is written per RANK, and the
        RANK # will be inserted into the corresponding file name.
    dotprodcoeffs : None or list of ndarrays
        Each element in list is a mapping of transmembrane currents to a measure
        on the form :math:`V = \\mathbf{C} \\cdot \\mathbf{I}`
    rec_current_dipole_moment : bool
        if True, compute and store the total current-dipole moment per time
        step as the sum over each individual population
    use_ipas : bool
        if True, compute the contribution to extracellular potentials across
        the passive leak channels embedded in the cells membranes summed over
        populations
    use_icap : bool
        if True, compute the contribution to extracellular potentials across
        the membrane capacitance embedded in the cells membranes summed over
        populations
    use_isyn : bool
        if True, compute the contribution to extracellular potentials across
        the excitatory and inhibitory synapses embedded in the cells membranes
        summed over populations
    rec_pop_contributions : bool
        if True, compute and return single-population contributions to the
        extracellular potential during each time step of the simulation

    Returns
    -------
    RESULTS : list
        ordered according to [dotprodcoeffs, ..., electrode, ...], each element
        being the superimposed contribution to i.e., the extracellular potential
        at each timestep from all cell objects on this particular RANK.
        Thus, no single-cell contributions to the LFP
        are returned.
    DIPOLE_MOMENT : ndarray
        Shape (n_timesteps, 3) array containing the x,y,z-components of the
        current-dipole moment summed up over contributions from cells across
        all populations on this MPI RANK.
    """
    # create a dummycell object lumping together needed attributes
    # for calculation of extracellular potentials etc. The population_nsegs
    # array is used to slice indices such that single-population
    # contributions to the potential can be calculated.
    population_nsegs, network_dummycell = network._create_network_dummycell()

    # Use electrode object(s) to calculate coefficient matrices for LFP
    # calculations. If electrode is a list, then
    # put electrodecoeff in a list, if it isn't already
    if dotprodcoeffs is not None:
        if type(dotprodcoeffs) != list:
            dotprodcoeffs = [dotprodcoeffs]
    else:
        #create empty list if no dotprodcoeffs are supplied
        dotprodcoeffs = []

    #access electrode object and append dotprodcoeffs
    if electrode is not None:
        #put electrode argument in list if needed
        if type(electrode) == list:
            electrodes = electrode
        else:
            electrodes = [electrode]

        # At each timestep we will later construct a single vector I of all
        # transmembrane currents. With that, and a corresponding matrix G
        # mapping a current contribution to an electrode contact, we can here
        # compute the extracellular potentials V_r in all contacts r at
        # timestep t_i as
        # V_r(r, t_i) = G x I(r, t_i)


        # # create a dummycell object lumping together needed attributes
        # # for calculation of extracellular potentials. The population_nsegs
        # # array is used to slice indices such that single-population
        # # contributions to the potential can be calculated.
        # population_nsegs, network_dummycell = network._create_network_dummycell()

        # We can have a number of separate electrode objects in a list, create
        # mappings for each
        for el in electrodes:
            # el.calc_lfp(cell=network_dummycell)
            el.calc_mapping(cell=network_dummycell)
            dotprodcoeffs += [el.mapping]
            # del el.LFP
            del el.mapping

    elif electrode is None:
        electrodes = None
        # if rec_current_dipole_moment:
        #     population_nsegs, network_dummycell = network._create_network_dummycell()




    # set maximum integration step, it is necessary for communication of
    # spikes across RANKs to occur.
    # NOTE: Should this depend on the minimum delay in the network?
    network.pc.set_maxstep(10)

    # Initialize NEURON simulations of cell object
    neuron.h.dt = network.dt

    # needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
    else:
        cvode.active(0)

    #initialize state
    neuron.h.finitialize(network.v_init)

    # use fast calculation of transmembrane currents
    cvode.use_fast_imem(1)

    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()

    #Starting simulation at tstart
    neuron.h.t = network.tstart

    # create list of cells across all populations to simplify loops
    cells = []
    for name in network.population_names:
        cells += network.populations[name].cells

    #load spike times from NetCon, only needed if LFPy.Synapse class is used
    for cell in cells:
        cell._loadspikes()


    # define data type for structured arrays dependent on the boolean arguments
    dtype = [('imem', np.float)]
    if use_ipas: dtype += [('ipas', np.float)]
    if use_icap: dtype += [('icap', np.float)]
    if use_isyn: dtype += [('isyn_e', np.float), ('isyn_i', np.float)]
    if rec_pop_contributions: dtype += list(zip(network.population_names,
                                                [np.float]*len(network.population_names)))

    # setup list of structured arrays for all extracellular potentials
    # at each contact from different source terms and subpopulations
    if to_memory:
        RESULTS = []
        for coeffs in dotprodcoeffs:
            RESULTS.append(np.zeros((coeffs.shape[0],
                                     int(network.tstop / network.dt) + 1),
                                    dtype=dtype)
                           )
    else:
        RESULTS = None

    # container for electric current dipole moment for the individual
    # populations captured inside the DummyCell instance
    if rec_current_dipole_moment:
        DIPOLE_MOMENT = np.zeros((int(network.tstop / network.dt) + 1, 3),
            dtype=list(zip(network.population_names,
                           [np.float]*len(network.population_names))))
    else:
        DIPOLE_MOMENT = None

    #LFPs for each electrode will be put here during simulations
    if to_file:
        #ensure right ending:
        if file_name.split('.')[-1] != 'h5':
            file_name += '.h5'
        outputfile = h5py.File(os.path.join(network.OUTPUTPATH,
                                            file_name.format(RANK)), 'w')
        for i, coeffs in enumerate(dotprodcoeffs):
            # can't do it this way until h5py issue #770
            # (https://github.com/h5py/h5py/issues/770) is fixed:
            # outputfile['OUTPUT[{}]'.format(i)] = np.zeros((coeffs.shape[0],
            #                     int(network.tstop / network.dt) + 1), dtype=dtype)
            grp = outputfile.create_group('OUTPUT[{}]'.format(i))
            for key, val in dtype:
                grp[key] = np.zeros((coeffs.shape[0], int(network.tstop / network.dt) + 1), dtype=val)

    # temp vector to store membrane currents at each timestep:
    imem = np.zeros(network_dummycell.totnsegs, dtype=dtype)

    # create a 2D array representation of segment midpoints for dot product
    # with transmembrane currents when computing dipole moment
    if rec_current_dipole_moment:
        midpoints = np.c_[network_dummycell.xmid,
                          network_dummycell.ymid,
                          network_dummycell.zmid]

    #run fadvance until time limit, and calculate LFPs for each timestep
    tstep = 0
    while neuron.h.t < network.tstop:
        if neuron.h.t >= 0:
            i = 0
            totnsegs = 0
            if use_isyn:
                imem['isyn_e'] = 0. # need to reset these for every iteration
                imem['isyn_i'] = 0. # because we sum over synapses
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
                            imem['isyn_e'][idx+totnsegs] += syn.i
                        else:
                            imem['isyn_i'][idx+totnsegs] += syn.i

                totnsegs += cell.totnsegs

            if rec_current_dipole_moment:
                k = 0 # counter
                for nsegs, name in zip(population_nsegs, network.population_names):
                    cellinds = np.arange(k, k+nsegs)
                    DIPOLE_MOMENT[name][tstep, ] = np.dot(imem['imem'][cellinds, ],
                                                          midpoints[cellinds, ])
                    k += nsegs

            if to_memory:
                for j, coeffs in enumerate(dotprodcoeffs):
                    RESULTS[j]['imem'][:, tstep] = np.dot(coeffs, imem['imem'])
                    if use_ipas:
                        RESULTS[j]['ipas'][:, tstep] = np.dot(coeffs, imem['ipas'] * network_dummycell.area * 1E-2)
                    if use_icap:
                        RESULTS[j]['icap'][:, tstep] = np.dot(coeffs, imem['icap'] * network_dummycell.area * 1E-2)
                    if use_isyn:
                        RESULTS[j]['isyn_e'][:, tstep] = np.dot(coeffs, imem['isyn_e'])
                        RESULTS[j]['isyn_i'][:, tstep] = np.dot(coeffs, imem['isyn_i'])

                if rec_pop_contributions:
                    for j, coeffs in enumerate(dotprodcoeffs):
                        k = 0 # counter
                        for nsegs, name in zip(population_nsegs, network.population_names):
                            cellinds = np.arange(k, k+nsegs)
                            RESULTS[j][name][:, tstep] = np.dot(coeffs[:, cellinds],
                                                                imem['imem'][cellinds, ])
                            k += nsegs

            if to_file:
                for j, coeffs in enumerate(dotprodcoeffs):
                    outputfile['OUTPUT[{}]'.format(j)
                               ]['imem'][:, tstep] = np.dot(coeffs, imem['imem'])
                    if use_ipas:
                        outputfile['OUTPUT[{}]'.format(j)
                                   ]['ipas'][:, tstep] = np.dot(coeffs, imem['ipas'] * network_dummycell.area * 1E-2)
                    if use_icap:
                        outputfile['OUTPUT[{}]'.format(j)
                                   ]['icap'][:, tstep] = np.dot(coeffs, imem['icap'] * network_dummycell.area * 1E-2)
                    if use_isyn:
                        outputfile['OUTPUT[{}]'.format(j)
                                   ]['isyn_e'][:, tstep] = np.dot(coeffs, imem['isyn_e'])
                        outputfile['OUTPUT[{}]'.format(j)
                                   ]['isyn_i'][:, tstep] = np.dot(coeffs, imem['isyn_i'])

                if rec_pop_contributions:
                    for j, coeffs in enumerate(dotprodcoeffs):
                        k = 0 # counter
                        for nsegs, name in zip(population_nsegs, network.population_names):
                            cellinds = np.arange(k, k+nsegs)
                            outputfile['OUTPUT[{}]'.format(j)
                                       ][name][:, tstep] = np.dot(coeffs[:, cellinds], imem['imem'][cellinds, ])
                            k += nsegs

            tstep += 1
        neuron.h.fadvance()
        if neuron.h.t % 100. == 0.:
            if RANK == 0:
                print('t = {} ms'.format(neuron.h.t))


    try:
        #calculate LFP after final fadvance()
        i = 0
        totnsegs = 0
        if use_isyn:
            imem['isyn_e'] = 0. # need to reset these for every iteration because we sum over synapses
            imem['isyn_i'] = 0.
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
                        imem['isyn_e'][idx+totnsegs] += syn.i
                    else:
                        imem['isyn_i'][idx+totnsegs] += syn.i

            totnsegs += cell.totnsegs

        if rec_current_dipole_moment:
            k = 0 # counter
            for nsegs, name in zip(population_nsegs, network.population_names):
                cellinds = np.arange(k, k+nsegs)
                DIPOLE_MOMENT[name][tstep, ] = np.dot(imem['imem'][cellinds, ], midpoints[cellinds, ])
                k += nsegs

        if to_memory:
            for j, coeffs in enumerate(dotprodcoeffs):
                RESULTS[j]['imem'][:, tstep] = np.dot(coeffs, imem['imem'])
                if use_ipas:
                    RESULTS[j]['ipas'][:, tstep] = np.dot(coeffs, imem['ipas'] * network_dummycell.area * 1E-2)
                if use_icap:
                    RESULTS[j]['icap'][:, tstep] = np.dot(coeffs, imem['icap'] * network_dummycell.area * 1E-2)
                if use_isyn:
                    RESULTS[j]['isyn_e'][:, tstep] = np.dot(coeffs, imem['isyn_e'])
                    RESULTS[j]['isyn_i'][:, tstep] = np.dot(coeffs, imem['isyn_i'])

            if rec_pop_contributions:
                for j, coeffs in enumerate(dotprodcoeffs):
                    k = 0 # counter
                    for nsegs, name in zip(population_nsegs, network.population_names):
                        cellinds = np.arange(k, k+nsegs)
                        RESULTS[j][name][:, tstep] = np.dot(coeffs[:, cellinds], imem['imem'][cellinds, ])
                        k += nsegs


        if to_file:
            for j, coeffs in enumerate(dotprodcoeffs):
                outputfile['OUTPUT[{}]'.format(j)
                           ]['imem'][:, tstep] = np.dot(coeffs, imem['imem'])
                if use_ipas:
                    outputfile['OUTPUT[{}]'.format(j)
                               ]['ipas'][:, tstep] = np.dot(coeffs, imem['ipas'] * network_dummycell.area * 1E-2)
                if use_icap:
                    outputfile['OUTPUT[{}]'.format(j)
                               ]['icap'][:, tstep] = np.dot(coeffs, imem['icap'] * network_dummycell.area * 1E-2)
                if use_isyn:
                    outputfile['OUTPUT[{}]'.format(j)
                               ]['isyn_e'][:, tstep] = np.dot(coeffs, imem['isyn_e'])
                    outputfile['OUTPUT[{}]'.format(j)
                               ]['isyn_i'][:, tstep] = np.dot(coeffs, imem['isyn_i'])

            if rec_pop_contributions:
                for j, coeffs in enumerate(dotprodcoeffs):
                    k = 0 # counter
                    for nsegs, name in zip(population_nsegs, network.population_names):
                        cellinds = np.arange(k, k+nsegs)
                        outputfile['OUTPUT[{}]'.format(j)
                                   ][name][:, tstep] = np.dot(coeffs[:, cellinds],
                                                              imem['imem'][cellinds, ])
                        k += nsegs

    except IndexError:
        pass

    if to_memory:
        return RESULTS, DIPOLE_MOMENT

    if to_file:
        outputfile.close()
        return RESULTS, DIPOLE_MOMENT


def ReduceStructArray(sendbuf, op=MPI.SUM):
    """
    simplify MPI Reduce for structured ndarrays with floating point numbers

    Parameters
    ----------
    sendbuf : structured ndarray
        Array data to be reduced (default: summed)
    op : mpi4py.MPI.Op object
        MPI_Reduce function. Default is mpi4py.MPI.SUM
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
                                     ['f8' for i in range(len(dtype_names))])))
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
