#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
import numpy as np
import neuron
from neuron import units
from pathlib import Path
import h5py


def _run_simulation_with_probes(cell, cvode, probes=[],
                                variable_dt=False, atol=0.001, rtol=0.,
                                to_memory=True,
                                to_file=False, file_name=None):
    '''Initialize and run simulation in NEURON,
    repeatedly calling neuron.h.fadvance() until cell.tstop is reached.

    Parameters
    ----------
    cell: LFPy.Cell like object
    cvode: neuron.h.CVode object
    probes: list of objects
    variable_dt: bool
    atol: float
    rtol: float
    to_memory: bool
    to_file: bool
    file_name: str

    Returns
    -------

    '''
    if variable_dt and to_file:
        raise NotImplementedError('to_file=True with variable_dt=True '
                                  'not supported')
    # Initialize NEURON simulations of cell object
    neuron.h.dt = cell.dt

    # needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
        cvode.rtol(rtol)
    else:
        cvode.active(0)

    # re-initialize state
    neuron.h.finitialize(cell.v_init * units.mV)

    # initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()

    neuron.h.frecord_init()  # wrong voltages t=0 for tstart < 0 otherwise

    # Start simulation at tstart (which may be < 0)
    neuron.h.t = cell.tstart

    # load spike times from NetCon
    cell._load_spikes()

    # temporary vector to store membrane currents at each timestep
    imem = np.zeros(cell.totnsegs)

    # precompute linear transformation matrices for each probe
    transforms = []  # container
    for probe in probes:
        M = probe.get_transformation_matrix()
        assert M.shape[-1] == cell.totnsegs, \
            'Linear transformation shape mismatch'
        transforms.append(M)
        if not variable_dt:
            probe.data = np.zeros((M.shape[0], int(cell.tstop / cell.dt) + 1))
        else:
            # for variable_dt, data will be added to last axis each time step
            probe.data = np.zeros((M.shape[0], 0))

    if to_file:
        # ensure right file extension:
        file_name = Path(file_name)
        if file_name.suffix != '.h5':
            file_name = file_name.parent / (file_name.name + '.h5')
        if not cvode.active():
            print('creating output file {}'.format(file_name))
            f = h5py.File(file_name, 'w')
            # create empty data arrays for data storage of output
            # corresponding to each probe. The naming scheme is
            # probe.__class__.__name__+'0', probe.__class__.__name__+'1' etc.
            names = []
            for probe, M in zip(probes, transforms):
                name = probe.__class__.__name__
                i = 0
                while True:
                    if name + '{}'.format(i) not in names:
                        names.append(name + '{}'.format(i))
                        break
                    i += 1
                #
                probe.data = f.create_dataset(
                    name=name,
                    shape=(M.shape[0],
                           int(cell.tstop / cell.dt) + 1),
                    dtype=float)

    def get_imem(imem):
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane_
                i += 1
        return imem

    tstep = 0
    # run fadvance until time limit, and calculate LFPs for each timestep
    while neuron.h.t < cell.tstop:
        if neuron.h.t >= 0:
            imem = get_imem(imem)
            for j, (probe, transform) in enumerate(zip(probes, transforms)):
                if not variable_dt:
                    probe.data[:, tstep] = transform @ imem
                else:
                    probe.data = np.c_[probes[j].data, transform @ imem]

            tstep += 1
        neuron.h.fadvance()

    # calculate LFP after final fadvance() if needed
    # (may occur for certain values for dt)
    if tstep < len(cell._neuron_tvec):
        imem = get_imem(imem)
        for j, (probe, transform) in enumerate(zip(probes, transforms)):
            if not variable_dt:
                probe.data[:, tstep] = transform @ imem
            else:
                probe.data = np.c_[probes[j].data, transform @ imem]

    if to_file:
        f.close()


def _collect_geometry_neuron(cell):
    '''Loop over allseclist to determine area, diam, xyz-start- and
    endpoints, embed geometry to cell object'''
    areavec = np.zeros(cell.totnsegs)
    diamvec = np.zeros(cell.totnsegs)
    lengthvec = np.zeros(cell.totnsegs)
    xstartvec = np.zeros(cell.totnsegs)
    xendvec = np.zeros(cell.totnsegs)
    ystartvec = np.zeros(cell.totnsegs)
    yendvec = np.zeros(cell.totnsegs)
    zstartvec = np.zeros(cell.totnsegs)
    zendvec = np.zeros(cell.totnsegs)

    counter = 0

    # loop over all segments
    for sec in cell.allseclist:
        n3d = int(neuron.h.n3d(sec=sec))
        nseg = sec.nseg
        gsen2 = 1. / 2 / nseg
        if n3d > 0:
            # create interpolation objects for the xyz pt3d info:
            L = np.zeros(n3d)
            x = np.zeros(n3d)
            y = np.zeros(n3d)
            z = np.zeros(n3d)
            for i in range(n3d):
                L[i] = neuron.h.arc3d(i, sec=sec)
                x[i] = neuron.h.x3d(i, sec=sec)
                y[i] = neuron.h.y3d(i, sec=sec)
                z[i] = neuron.h.z3d(i, sec=sec)

            # normalize as seg.x [0, 1]
            L /= sec.L

            # temporary store position of segment midpoints
            segx = np.zeros(nseg)
            for i, seg in enumerate(sec):
                segx[i] = seg.x

            # can't be >0 which may happen due to NEURON->Python float
            # transfer:
            segx0 = (segx - gsen2).round(decimals=6)
            segx1 = (segx + gsen2).round(decimals=6)

            # fill vectors with interpolated coordinates of start and end
            # points
            xstartvec[counter:counter + nseg] = np.interp(segx0, L, x)
            xendvec[counter:counter + nseg] = np.interp(segx1, L, x)

            ystartvec[counter:counter + nseg] = np.interp(segx0, L, y)
            yendvec[counter:counter + nseg] = np.interp(segx1, L, y)

            zstartvec[counter:counter + nseg] = np.interp(segx0, L, z)
            zendvec[counter:counter + nseg] = np.interp(segx1, L, z)

            # fill in values area, diam, length
            for seg in sec:
                areavec[counter] = neuron.h.area(seg.x, sec=sec)
                diamvec[counter] = seg.diam
                lengthvec[counter] = sec.L / nseg

                counter += 1

    # set cell attributes
    cell.x = np.c_[xstartvec, xendvec]
    cell.y = np.c_[ystartvec, yendvec]
    cell.z = np.c_[zstartvec, zendvec]

    cell.area = areavec
    cell.d = diamvec
    cell.length = lengthvec
