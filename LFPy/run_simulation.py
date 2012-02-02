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
    counter = 0.
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
    cell.imem = np.eye(cell.totnsegs)
    tvec = cell.tvec            
    cell.tvec = np.arange(cell.totnsegs) * cell.timeres_python
    if type(electrode) == type([]):
        for el in electrode:
            el.calc_lfp(cell=cell)
            el.electrodecoeffs = el.LFP
            el.LFP = []
    else:
        electrode.calc_lfp(cell=cell)
        electrode.electrodecoeffs = electrode.LFP
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
        interval = 1 / cell.timeres_NEURON * 100
    else:
        interval = 1 / cell.timeres_NEURON * 10
    
    #temp vector to store membrane currents at each timestep
    imem = np.empty(cell.totnsegs)
    
    if type(electrode) == type([]):
        while neuron.h.t < cell.tstopms:
            if neuron.h.t >= 0:
                i = 0
                for sec in cell.allseclist:
                    for seg in sec:
                        imem[i] = seg.i_membrane * cell.area[i] * 1E-2
                        i += 1
                for el in electrode:
                    el.LFP.append(np.dot(el.electrodecoeffs, imem))
            
            neuron.h.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                print 't = %.0f' % neuron.h.t
        
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane * cell.area[i] * 1E-2
                i += 1
        
        for el in electrode:
            
            el.LFP.append(np.dot(el.electrodecoeffs, imem))
            el.LFP = np.array(el.LFP).T
    else:
        while neuron.h.t < cell.tstopms:
            if neuron.h.t >= 0:
                i = 0
                for sec in cell.allseclist:
                    for seg in sec:
                        imem[i] = seg.i_membrane * cell.area[i] * 1E-2
                        i += 1
                electrode.LFP.append(np.dot(electrode.electrodecoeffs, imem))
            
            neuron.h.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                print 't = %.0f' % neuron.h.t
        
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane * cell.area[i] * 1E-2
                i += 1
        
        electrode.LFP.append(np.dot(electrode.electrodecoeffs, imem))
        
        electrode.LFP = np.array(electrode.LFP).T
