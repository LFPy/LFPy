#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: language_level=2
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


from time import time
import numpy as np
cimport numpy as np
import neuron
from pathlib import Path

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t   LTYPE_t


def _run_simulation(cell, cvode, variable_dt=False, atol=0.001, rtol=0.):
    """
    Running the actual simulation in NEURON, simulations in NEURON
    is now interruptable.
    """
    neuron.h.dt = cell.dt

    # variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
        cvode.rtol(rtol)
    else:
        cvode.active(0)

    #re-initialize state
    neuron.h.finitialize(cell.v_init)

    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()

    #Starting simulation at t != 0
    neuron.h.t = cell.tstart

    cell._loadspikes()

    #print sim.time at intervals
    cdef int counter = 0
    cdef double interval
    cdef double tstop = cell.tstop
    cdef double t0 = time()
    cdef double ti = neuron.h.t
    cdef double rtfactor
    if tstop >= 10000:
        interval = 1000. / cell.dt
    else:
        interval = 100. / cell.dt

    while neuron.h.t < tstop:
        neuron.h.fadvance()
        counter += 1
        if counter % interval == 0:
            rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0 + 1E-9)
            if cell.verbose:
                print('t = {:.0f}, realtime factor: {:.3f}'.format(neuron.h.t,
                                                                   rtfactor))
            t0 = time()
            ti = neuron.h.t


def _run_simulation_with_electrode(cell, cvode, electrode=None,
                                   variable_dt=False, atol=0.001, rtol=0.,
                                   to_memory=True, to_file=False,
                                   file_name=None, dotprodcoeffs=None,
                                   rec_current_dipole_moment=False):
    """
    Running the actual simulation in NEURON.
    electrode argument used to determine coefficient
    matrix, and calculate the LFP on every time step.
    """

    #c-declare some variables
    cdef int i, j, tstep#, ncoeffs
    #cdef int totnsegs = cell.totnsegs
    cdef double tstop = cell.tstop
    cdef int counter
    cdef int lendotprodcoeffs0
    cdef double interval
    cdef double t0
    cdef double ti
    cdef double rtfactor
    cdef double dt = cell.dt
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] coeffs
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] current_dipole_moment
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] midpoints

    #check if h5py exist and saving is possible
    try:
        import h5py
    except:
        print('h5py not found, LFP to file not possible')
        to_file = False
        file_name = None

    if variable_dt and to_file:
        print('LFP to file with variable_dt is not supported')
        to_file = False
        file_name = None

    # Use electrode object(s) to calculate coefficient matrices for LFP
    # calculations. If electrode is a list, then
    if cell.verbose:
        print('precalculating geometry - LFP mapping')

    #put electrodecoeff in a list, if it isn't already
    if dotprodcoeffs is not None:
        if type(dotprodcoeffs) != list:
            dotprodcoeffs = [dotprodcoeffs]
        electrodes = []
    else:
        #create empty list if no dotprodcoeffs are supplied
        dotprodcoeffs = []

    #just for safekeeping
    lendotprodcoeffs0 = len(dotprodcoeffs)

    #access electrode object and append mapping
    if electrode is not None:
        #put electrode argument in list if needed
        if type(electrode) == list:
            electrodes = electrode
        else:
            electrodes = [electrode]

        for el in electrodes:
            el.calc_mapping(cell)
            dotprodcoeffs.append(el.mapping)
    elif electrode is None:
        electrodes = None


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
    neuron.h.finitialize(cell.v_init)

    # initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()

    neuron.h.frecord_init()  # wrong voltages t=0 for tstart < 0 otherwise

    #Starting simulation at t != 0
    neuron.h.t = cell.tstart

    #load spike times from NetCon
    cell._loadspikes()

    #print sim.time at intervals
    counter = 0
    tstep = 0
    t0 = time()
    ti = neuron.h.t
    if tstop >= 10000:
        interval = 1000. / dt
    else:
        interval = 100. / dt

    #temp vector to store membrane currents at each timestep
    imem = np.zeros(cell.totnsegs)
    #LFPs for each electrode will be put here during simulation
    if to_memory:
        electrodesLFP = []
        for coeffs in dotprodcoeffs:
            if not cvode.active():
                # if cvode is on lists are created on-the-fly
                electrodesLFP.append(np.zeros((coeffs.shape[0],
                                     int(cell.tstop / cell.dt) + 1)))
    #LFPs for each electrode will be put here during simulations
    if to_file:
        #ensure right ending:
        file_name = Path(file_name)
        if file_name.suffix != '.h5':
            file_name = file_name.parent / (file_name.name + '.h5')
        if not cvode.active():
            el_LFP_file = h5py.File(file_name, 'w')
            i = 0
            for coeffs in dotprodcoeffs:
                el_LFP_file['electrode{:03d}'.format(i)] = np.zeros((coeffs.shape[0], int(cell.tstop / cell.dt + 1)))
                i += 1

    # create a 2D array representation of segment midpoints for dot product
    # with transmembrane currents when computing dipole moment
    if rec_current_dipole_moment:
        if not cvode.active():
            current_dipole_moment = cell.current_dipole_moment.copy()
            cell.current_dipole_moment = np.array([[]])
        midpoints = np.c_[cell.xmid, cell.ymid, cell.zmid]

    #run fadvance until time limit, and calculate LFPs for each timestep
    while neuron.h.t < tstop:
        if neuron.h.t >= 0:
            i = 0
            for sec in cell.allseclist:
                for seg in sec:
                    imem[i] = seg.i_membrane_
                    i += 1

            if rec_current_dipole_moment:
                if not cvode.active():
                    current_dipole_moment[tstep, ] = np.dot(imem, midpoints)
                else:
                    cell.current_dipole_moment.append(np.dot(imem, midpoints))

            if to_memory:
                for j, coeffs in enumerate(dotprodcoeffs):
                    if not cvode.active():
                        electrodesLFP[j][:, tstep] = np.dot(coeffs, imem)
                    else:
                        if tstep == 0:
                            electrodesLFP.append(np.dot(coeffs, imem)[:, np.newaxis])
                        else:
                            electrodesLFP[j] = np.hstack((electrodesLFP[j], np.dot(coeffs, imem)[:, np.newaxis]))
            if to_file:
                for j, coeffs in enumerate(dotprodcoeffs):
                    el_LFP_file['electrode{:03d}'.format(j)
                                ][:, tstep] = np.dot(coeffs, imem)

            tstep += 1
        neuron.h.fadvance()
        counter += 1
        if counter % interval == 0:
            rtfactor = (neuron.h.t - ti) * 1E-3 / (time() - t0)
            if cell.verbose:
                print('t = {:.0f}, realtime factor: {:.3f}'.format(neuron.h.t,
                                                                   rtfactor))
            t0 = time()
            ti = neuron.h.t

    if tstep < len(cell._neuron_tvec):
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane_
                i += 1

        if rec_current_dipole_moment:
            if not cvode.active():
                current_dipole_moment[tstep, ] = np.dot(imem, midpoints)
            else:
                cell.current_dipole_moment.append(np.dot(imem, midpoints))


        if to_memory:
            for j, coeffs in enumerate(dotprodcoeffs):
                if not cvode.active():
                    electrodesLFP[j][:, tstep] = np.dot(coeffs, imem)
                else:
                    electrodesLFP[j] = np.hstack((electrodesLFP[j], np.dot(coeffs, imem)[:, np.newaxis]))

        if to_file:
            for j, coeffs in enumerate(dotprodcoeffs):
                el_LFP_file['electrode{:03d}'.format(j)
                            ][:, tstep] = np.dot(coeffs, imem)

    # update current dipole moment values
    if rec_current_dipole_moment:
        if not cvode.active():
            cell.current_dipole_moment = np.array(current_dipole_moment)
        else:
            cell.current_dipole_moment = np.array(cell.current_dipole_moment)

    # Final step, put LFPs in the electrode object, superimpose if necessary
    # If electrode.perCellLFP, store individual LFPs
    if to_memory:
        #the first few belong to input dotprodcoeffs
        cell.dotprodresults = electrodesLFP[:lendotprodcoeffs0]
        #the remaining belong to input electrode arguments
        if electrodes is not None:
            for j, LFP in enumerate(electrodesLFP):
                if not j < lendotprodcoeffs0:
                    if hasattr(electrodes[j-lendotprodcoeffs0], 'LFP'):
                        electrodes[j-lendotprodcoeffs0].LFP += LFP
                    else:
                        electrodes[j-lendotprodcoeffs0].LFP = LFP
                    #will save each cell contribution separately
                    if electrodes[j-lendotprodcoeffs0].perCellLFP:
                        if not hasattr(electrodes[j], 'CellLFP'):
                            electrodes[j-lendotprodcoeffs0].CellLFP = []
                        electrodes[j-lendotprodcoeffs0].CellLFP.append(LFP)
                    electrodes[j-lendotprodcoeffs0].electrodecoeff = dotprodcoeffs[j]

    if to_file:
        el_LFP_file.close()


cpdef _collect_geometry_neuron(cell):
    """Loop over allseclist to determine area, diam, xyz-start- and
    endpoints, embed geometry to cell object"""


    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] areavec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] diamvec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] lengthvec = np.zeros(cell.totnsegs)

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstartvec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xendvec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ystartvec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yendvec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zstartvec = np.zeros(cell.totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zendvec = np.zeros(cell.totnsegs)

    cdef DTYPE_t gsen2, secL
    cdef LTYPE_t counter, nseg, n3d, i


    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] L, x, y, z, segx, segx0, segx1


    counter = 0

    #loop over all segments
    for sec in cell.allseclist:
        n3d = int(neuron.h.n3d(sec=sec))
        nseg = sec.nseg
        gsen2 = 1./2/nseg
        secL = sec.L
        if n3d > 0:
            #create interpolation objects for the xyz pt3d info:
            L = np.zeros(n3d)
            x = np.zeros(n3d)
            y = np.zeros(n3d)
            z = np.zeros(n3d)
            for i in range(n3d):
                L[i] = neuron.h.arc3d(i, sec=sec)
                x[i] = neuron.h.x3d(i, sec=sec)
                y[i] = neuron.h.y3d(i, sec=sec)
                z[i] = neuron.h.z3d(i, sec=sec)

            #normalize as seg.x [0, 1]
            L /= secL

            #temporary store position of segment midpoints
            segx = np.zeros(nseg)
            i = 0
            for seg in sec:
                segx[i] = seg.x
                i += 1

            #can't be >0 which may happen due to NEURON->Python float transfer:
            #segx0 = (segx - gsen2).round(decimals=6)
            #segx1 = (segx + gsen2).round(decimals=6)
            segx0 = segx - gsen2
            segx1 = segx + gsen2

            #fill vectors with interpolated coordinates of start and end points
            xstartvec[counter:counter+nseg] = np.interp(segx0, L, x)
            xendvec[counter:counter+nseg] = np.interp(segx1, L, x)

            ystartvec[counter:counter+nseg] = np.interp(segx0, L, y)
            yendvec[counter:counter+nseg] = np.interp(segx1, L, y)

            zstartvec[counter:counter+nseg] = np.interp(segx0, L, z)
            zendvec[counter:counter+nseg] = np.interp(segx1, L, z)

            #fill in values area, diam, length
            for seg in sec:
                areavec[counter] = neuron.h.area(seg.x, sec=sec)
                diamvec[counter] = seg.diam
                lengthvec[counter] = secL/nseg

                counter += 1


    #set cell attributes
    cell.xstart = xstartvec
    cell.ystart = ystartvec
    cell.zstart = zstartvec

    cell.xend = xendvec
    cell.yend = yendvec
    cell.zend = zendvec

    cell.area = areavec
    cell.diam = diamvec
    cell.length = lengthvec
