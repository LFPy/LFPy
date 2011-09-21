#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import numpy

class PointProcess:
    '''Superclass on top of PointProcessSynapse, PointProcessElectrode, 
    just to set some common variables'''
    def __init__(self, cell, idx, color='k', marker='o', 
                 record_current=False, **kwargs):
        '''cell is an LFPy.Cell object, idx index of segment. This class
        set some variables and extracts carthesian coordinates of segment'''
        self.idx = idx
        self.color = color
        self.marker = marker
        self.record_current = record_current
        self.kwargs = kwargs
        self.update_pos(cell)

    def update_pos(self, cell):
        '''Extract coordinate of point-process to geometry'''
        self.x = cell.xmid[self.idx]
        self.y = cell.ymid[self.idx]
        self.z = cell.zmid[self.idx]
        
class PointProcessSynapse(PointProcess):
    '''The synapse class, pointprocesses that spawn membrane currents'''
    def __init__(self, cell, idx, syntype, color='r', marker='o',
                 record_current=False, **kwargs):
        '''cell - cell instance, idx - index of compartment where synapse is
        inserted, syntype - NetCon-enabled synapse mech, color - color in plot,
        marker - marker for plot, record_current - switch for enabling current
        recording, **kwargs - syntype specific arguments passed onto
        cell.set_synapse
        
        Usage:
        synparams = {
            'idx' : 0,                  #index number of compartment
            'color' : 'p',              #color, for plotting
            'marker' : 'o'              #marker for plotting
            'record_current' : True     #record synapse currents
            'pptype' : 'Exp2Syn',       #type of synapse
            'tau1' : 0.5,               #time-constant, rise, Exp2Syn specific
            'tau2' : 1,                 #ditto, decay
            'e' : 10,                   #reversal potential +10 mV
            'weight' : 0.001,           #NetCon weight aka max conductance
            }
        LFPy.PointProcessSynapse(cell,**synparams)
        '''
        PointProcess.__init__(self, cell, idx, color, marker, record_current, 
                              **kwargs)
            
        self.syntype = syntype
        self.hocidx = int(cell.set_synapse(idx, syntype,
                                           record_current, **kwargs))
        cell.synapses.append(self)
        cell.synidx.append(idx)

    def set_spike_times(self, cell , sptimes=numpy.zeros(0)):
        '''Set the spike times'''
        self.sptimes = sptimes
        cell.sptimeslist.append(sptimes)
        
    def collect_current(self, cell):
        '''Collect synapse current'''
        self.i = numpy.array(cell.synireclist.o(self.hocidx))
    
    def collect_potential(self, cell):
        '''Collect membrane potential of segment with synapse'''
        self.v = numpy.array(cell.synvreclist.o(self.hocidx))

class Synapse(PointProcessSynapse):
    def __init__(self, cell, idx, syntype, color='r', marker='o',
                 record_current=False, **kwargs):
        PointProcessSynapse.__init__(self, cell, idx, syntype, color, marker, 
                                     record_current, **kwargs)
        print 'LFPy.Synapse pending Deprecation,'
        print 'use LFPy.PointProcessSynapse instead'
        
class PointProcessElectrode(PointProcess):
    '''Class for NEURON point processes, ie VClamp, SEClamp and ICLamp,
    SinIClamp, ChirpIClamp with arguments.
    Electrode currents go here, whics make membrane currents not sum to zero'''
    def __init__(self, cell, idx, pptype='SEClamp',
                 color='p', marker='*', record_current=False, **kwargs):
        ''' Will insert pptype on
        cell-instance, pass the corresponding kwargs onto
        cell.set_point_process.
        
        Refer to NEURON documentation @ neuron.yale.edu for kwargs
        
        'cell' is the cell instance.
        
        Usage:
        pointprocparams = {
            'idx' : 0,                  #index number of compartment
            'color' : 'p',              #color, for plotting
            'marker' : '*'              #marker for plotting
            'record_current' : True     #record electrode currents
            'pptype' : 'IClamp',        #type of pointprocess
            'amp' : 1,                  #the rest is kwargs
            'dur' : 10,
            'delay' : 10,
            'rs' : 1,
            }
        LFPy.PointProcessElectrode(cell,**pointprocparams)
        '''
        PointProcess.__init__(self, cell, idx, color, marker, record_current)
        self.pptype = pptype
        self.hocidx = int(cell.set_point_process(idx, pptype,
                                                 record_current, **kwargs))
        cell.pointprocesses.append(self)
        cell.pointprocess_idx.append(idx)

    def collect_current(self, cell):
        '''Fetch electrode current from recorder list'''
        self.i = numpy.array(cell.stimireclist.o(self.hocidx))
    
    def collect_potential(self, cell):
        '''Collect membrane potential of segment with PointProcess'''
        self.v = numpy.array(cell.synvreclist.o(self.hocidx))
    