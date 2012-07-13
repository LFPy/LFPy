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
import neuron
from time import time

def _run_simulation(cell, variable_dt=False, atol=0.001):
    '''
    Running the actual simulation in NEURON, simulations in NEURON
    is now interruptable.
    '''
    neuron.h.dt = cell.timeres_NEURON
    
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
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
    
    ##Starting simulation at tstart
    neuron.h.t = cell.tstartms
    
    cell._loadspikes()
    
    #print sim.time and realtime factor at intervals
    counter = 0.
    t0 = time()
    ti = neuron.h.t
    if cell.tstopms > 1000:
        interval = 1 / cell.timeres_NEURON * 100
    else:
        interval = 1 / cell.timeres_NEURON * 10
    
    while neuron.h.t < cell.tstopms:
        neuron.h.fadvance()
        counter += 1.
        if np.mod(counter, interval) == 0:
            rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0)
            print 't = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor)
            t0 = time()
            ti = neuron.h.t

def _run_simulation_with_electrode(cell, electrode, variable_dt=False, atol=0.001,
                                   to_memory=True, to_file=False, file_name=None):
    '''
    Running the actual simulation in NEURON.
    electrode argument used to determine coefficient
    matrix, and calculate the LFP on every time step.
    '''
    try:
        import h5py
    except:
        print 'h5py not found, LFP to file not possible'
        to_file = False
        file_name = None

    # Use electrode object(s) to calculate coefficient matrices for LFP
    # calculations. If electrode is a list, then
    
    #put electrode argument in list if needed
    if type(electrode) == type([]):
        electrodes = electrode
    else:
        electrodes = [electrode]
    
    #calculate list of electrodecoeffs, will try temp store of imem, tvec, LFP
    cellTvec = cell.tvec
    try:
        cellImem = cell.imem.copy()
    except:
        pass
    
    cell.imem = np.eye(cell.totnsegs)
    cell.tvec = np.arange(cell.totnsegs) * cell.timeres_python
    electrodecoeffs = []
    electrodeLFP = []   #list of electrode.LFP objects if they exist
    restoreLFP = False
    restoreCellLFP = False
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
        
    #putting back variables
    cell.tvec = cellTvec        
    try:
        cell.imem = cellImem
    except:
        del cell.imem
    
    # Initialize NEURON simulations of cell object    
    neuron.h.dt = cell.timeres_NEURON
    
    #integrator
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if cell.timeres_NEURON<=1E-8:
        cvode.active(1)
        cvode.atol(atol)
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
    
    #Starting simulation at tstartms
    neuron.h.t = cell.tstartms
    
    #load spike times from NetCon
    cell._loadspikes()
    
    #print sim.time at intervals
    counter = 0.
    tstep = 0
    t0 = time()
    ti = neuron.h.t
    if cell.tstopms > 1000:
        interval = 1 / cell.timeres_NEURON * 100
    else:
        interval = 1 / cell.timeres_NEURON * 10
    
    #temp vector to store membrane currents at each timestep
    imem = np.empty(cell.totnsegs)
    #LFPs for each electrode will be put here during simulation
    if to_memory:
        electrodesLFP = []
        for coeffs in electrodecoeffs:
            electrodesLFP.append(np.empty((coeffs.shape[0],
                                    cell.tstopms / cell.timeres_NEURON + 1)))
    #LFPs for each electrode will be put here during simulations
    if to_file:
        #ensure right ending:
        if file_name.split('.')[-1] != 'h5':
            file_name += '.h5'
        el_LFP_file = h5py.File(file_name, 'w')
        i = 0
        for coeffs in electrodecoeffs:
            el_LFP_file['electrode%.3i' % i] = np.empty((coeffs.shape[0],
                                    cell.tstopms / cell.timeres_NEURON + 1))
            i += 1
    
    #run fadvance until time limit, and calculate LFPs for each timestep
    area = cell.area
    while neuron.h.t < cell.tstopms:
        if neuron.h.t >= 0:
            i = 0
            for sec in cell.allseclist:
                for seg in sec:
                    imem[i] = seg.i_membrane * area[i] * 1E-2
                    i += 1
            #this is the lfp at this timestep
            lfp = np.dot(coeffs, imem)
            
            if to_memory:
                j = 0
                for coeffs in electrodecoeffs:
                    electrodesLFP[j][:, tstep] = lfp
                    j += 1
                    
            if to_file:
                j = 0
                for coeffs in electrodecoeffs:
                    el_LFP_file['electrode%.3i' % j][:, tstep] = lfp
            
            tstep += 1
        neuron.h.fadvance()
        counter += 1.
        if np.mod(counter, interval) == 0:
            rtfactor = (neuron.h.t - ti) * 1E-3 / (time() - t0)
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
        if to_memory:
            j = 0
            for coeffs in electrodecoeffs:
                lfp = np.dot(coeffs, imem)
                electrodesLFP[j][:, tstep] = lfp
                j += 1
        if to_file:
            j = 0
            for coeffs in electrodecoeffs:
                el_LFP_file['electrode%.3i' % j][:, tstep] = lfp

    except:
        pass
    
    # Final step, put LFPs in the electrode object, superimpose if necessary
    # If electrode.perCellLFP, store individual LFPs
    if to_memory:
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


    if to_file:
        el_LFP_file.close()

