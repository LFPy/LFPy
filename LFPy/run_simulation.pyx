#!/usr/bin/env python

import numpy as np
cimport numpy as np
import neuron

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
    if cell.tstartms != None:
        neuron.h.t = cell.tstartms
    
    cell.loadspikes()
    
    
    #print sim.time at intervals
    cdef double counter = 0.
    cdef double interval
    if cell.tstopms > 1000:
        interval = 1 / cell.timeres_NEURON * 100
    else:
        interval = 1 / cell.timeres_NEURON * 10
    
    while neuron.h.t < cell.tstopms:
        neuron.h.fadvance()
        counter += 1.
        if np.mod(counter, interval) == 0:
            print 't = %.0f' % neuron.h.t

def _run_simulation_with_electrode(cell, electrode):
    '''
    Running the actual simulation in NEURON.
    electrode argument used to determine coefficient
    matrix, and calculate the LFP on every time step.
    '''
    
    # Use electrode object(s) to calculate coefficient matrices for LFP
    # calculations. If electrode is a list, then
    
    #c-declare some variables
    cdef int i, j
    cdef int totnsegs = cell.totnsegs
    cdef double tstopms = cell.tstopms
    cdef double counter, interval,
    cdef double timeres_NEURON = cell.timeres_NEURON
    cdef double timeres_python = cell.timeres_python
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] electrodecoeffs
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] LFP
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] imem = \
        np.empty(totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] area = \
        cell.area
    
    #calculate coefficient matrices by using eye-matrix as imem
    cell.imem = np.eye(totnsegs)
    tvec = cell.tvec            
    cell.tvec = np.arange(totnsegs) * timeres_python
    if type(electrode) == type([]):
        for el in electrode:
            el.calc_lfp(cell=cell)
            electrodecoeffs = el.LFP
            el.electrodecoeffs = electrodecoeffs
            el.LFP = []
    else:
        electrode.calc_lfp(cell=cell)
        electrodecoeffs = electrode.LFP
        electrode.electrodecoeffs = electrodecoeffs
        electrode.LFP = []
    cell.tvec = tvec
        
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
    if cell.tstartms != None:
        neuron.h.t = cell.tstartms
    
    cell.loadspikes()
    
    #print sim.time at intervals
    counter = 0.
    if cell.tstopms > 1000:
        interval = 1 / timeres_NEURON * 100
    else:
        interval = 1 / timeres_NEURON * 10

    #separate while loops for electrode cases
    if type(electrode) == type([]):
        while neuron.h.t < cell.tstopms:
            if neuron.h.t >= 0:
                i = 0
                for sec in cell.allseclist:
                    for seg in sec:
                        imem[i] = seg.i_membrane * area[i] * 1E-2
                        i += 1
                #calculate LFP for each electrode in list
                for el in electrode:
                    LFP = np.dot(el.electrodecoeffs, imem)
                    el.LFP.append(LFP)
            
            neuron.h.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                print 't = %.0f' % neuron.h.t
        
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane * area[i] * 1E-2
                i += 1
        
        #calculate LFP for each electrode in list
        for el in electrode:
            LFP = np.dot(el.electrodecoeffs, imem)
            el.LFP.append(LFP)
            el.LFP = np.array(el.LFP).T
    else:
        while neuron.h.t < cell.tstopms:
            if neuron.h.t >= 0:
                i = 0
                for sec in cell.allseclist:
                    for seg in sec:
                        imem[i] = seg.i_membrane * area[i] * 1E-2
                        i += 1
                #calculate LFP for electrode
                LFP = np.dot(electrode.electrodecoeffs, imem)
                electrode.LFP.append(LFP)
            
            neuron.h.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                print 't = %.0f' % neuron.h.t
        
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane * area[i] * 1E-2
                i += 1
        #calculate LFP for electrode
        LFP = np.dot(electrode.electrodecoeffs, imem)
        electrode.LFP.append(LFP)
        
        electrode.LFP = np.array(electrode.LFP).T
