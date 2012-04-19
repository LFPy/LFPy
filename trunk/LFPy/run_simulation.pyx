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
cimport numpy as np
import neuron
from time import time

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def _run_simulation(cell):
    '''
    Running the actual simulation in NEURON, simulations in NEURON
    is now interruptable.
    '''
    neuron.h.dt = cell.timeres_NEURON
    
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if neuron.h.dt <= 1E-8:
        cvode.active(1)
        cvode.atol(0.001)
    else:
        cvode.active(0)
    
    #initialize state
    neuron.h.finitialize(cell.v_init)
    
    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()
    
    #Starting simulation at t != 0
    neuron.h.t = cell.tstartms
    
    cell.loadspikes()
        
    #print sim.time at intervals
    cdef double counter = 0.
    cdef double interval
    cdef double tstopms = cell.tstopms
    cdef double t0 = time()
    cdef double ti = neuron.h.t
    cdef double rtfactor
    if tstopms > 1000:
        interval = 1 / cell.timeres_NEURON * 100
    else:
        interval = 1 / cell.timeres_NEURON * 10
    
    while neuron.h.t < tstopms:
        neuron.h.fadvance()
        counter += 1.
        if divmod(counter, interval)[1] == 0:
            rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0)
            print 't = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor)
            t0 = time()
            ti = neuron.h.t

def _run_simulation_with_electrode(cell, electrode):
    '''
    Running the actual simulation in NEURON.
    electrode argument used to determine coefficient
    matrix, and calculate the LFP on every time step.
    '''
    
    #c-declare some variables
    cdef int i, j, tstep, ncoeffs
    cdef int totnsegs = cell.totnsegs
    cdef double tstopms = cell.tstopms
    cdef double counter, interval
    cdef double t0
    cdef double ti
    cdef double rtfactor
    cdef double timeres_NEURON = cell.timeres_NEURON
    cdef double timeres_python = cell.timeres_python
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] coeffs
    #cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] LFP
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] imem = \
        np.empty(totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] area = \
        cell.area
    
    # Use electrode object(s) to calculate coefficient matrices for LFP
    # calculations. If electrode is a list, then
    
    #put electrode argument in list if needed
    if type(electrode) == type([]):
        electrodes = electrode
    else:
        electrodes = [electrode]
    
    #calculate list of electrodecoeffs, will try temp storage of imem, tvec, LFP
    cellTvec = cell.tvec
    try:
        cellImem = cell.imem.copy()
    except:
        pass
    
    cell.imem = np.eye(totnsegs)
    cell.tvec = np.arange(totnsegs) * timeres_python
    electrodecoeffs = []
    electrodeLFP = []   #list of electrode.LFP objects if they exist
    restoreLFP = False
    restoreCellLFP = False
    ncoeffs = 0
    for el in electrodes:
        if hasattr(el, 'LFP'):
            LFPcopy = el.LFP
            del el.LFP
            restoreLFP = True
        if hasattr(el, 'CellLFP'):
            CellLFP = el.CellLFP
            restoreCellLFP = True
        el.calc_lfp(cell=cell)
        electrodecoeffs.append(el.LFP.copy())
        if restoreLFP:
            del el.LFP
            el.LFP = LFPcopy
        else:
            del el.LFP
        if restoreCellLFP:
            el.CellLFP = CellLFP
        else:
            if hasattr(el, 'CellLFP'):
                del el.CellLFP
        ncoeffs += 1
    
    #putting back variables
    cell.tvec = cellTvec        
    try:
        cell.imem = cellImem
    except:
        del cell.imem
    
    # Initialize NEURON simulations of cell object    
    neuron.h.dt = timeres_NEURON
    
    #integrator
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if neuron.h.dt <= 1E-8:
        cvode.active(1)
        cvode.atol(0.001)
    else:
        cvode.active(0)
    
    #initialize state
    neuron.h.finitialize(cell.v_init)
    
    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()
    
    #Starting simulation at t != 0
    neuron.h.t = cell.tstartms
    
    #load spike times from NetCon
    cell.loadspikes()
    
    #print sim.time at intervals
    counter = 0.
    tstep = 0
    t0 = time()
    ti = neuron.h.t
    if tstopms > 1000:
        interval = 1 / timeres_NEURON * 100
    else:
        interval = 1 / timeres_NEURON * 10
        
    #temp vector to store membrane currents at each timestep
    imem = np.empty(cell.totnsegs)
    #LFPs for each electrode will be put here during simulation
    electrodesLFP = []
    for coeffs in electrodecoeffs:
        electrodesLFP.append(np.empty((coeffs.shape[0],
                                       cell.tstopms / cell.timeres_NEURON + 1)))
    
    #run fadvance until time limit, and calculate LFPs for each timestep
    while neuron.h.t < tstopms:
        if neuron.h.t >= 0:
            i = 0
            for sec in cell.allseclist:
                for seg in sec:
                    imem[i] = seg.i_membrane * area[i] * 1E-2
                    i += 1
            for j in xrange(ncoeffs):
                electrodesLFP[j][:, tstep] = np.dot(electrodecoeffs[j], imem)
            tstep += 1
        neuron.h.fadvance()
        counter += 1.
        if divmod(counter, interval)[1] == 0:
            rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0)
            print 't = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor)
            t0 = time()
            ti = neuron.h.t
    
    try:
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane * area[i] * 1E-2
                i += 1
        j = 0
        for coeffs in electrodecoeffs:
            electrodesLFP[j][:, tstep] = np.dot(coeffs, imem)
            j += 1
    except:
        pass
    
    # Final step, put LFPs in the electrode object, superimpose if necessary
    # If electrode.perCellLFP, store individual LFPs
    j = 0
    for el in electrodes:
        if hasattr(el, 'LFP'):
            el.LFP += electrodesLFP[j]
        else:
            el.LFP = electrodesLFP[j]
        #will save each cell contribution separately
        if el.perCellLFP:
            if not hasattr(el, 'CellLFP'):
                el.CellLFP = []
            el.CellLFP.append(electrodesLFP[j])
        el.electrodecoeff = electrodecoeffs[j]
        j += 1
  
