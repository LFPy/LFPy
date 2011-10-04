#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import numpy
import neuron

class PointProcess:
    '''
    Superclass on top of PointProcessSynapse, PointProcessElectrode, 
    just to import and set some common variables.
    
    Arguments:
    ::
        cell    : LFPy.Cell object
        idx     : index of segment
        color   : opt. color in plot
        marker  : opt. marker in plot
        record_current : Must be set True for recording of pointprocess currents
        kwargs  : pointprocess specific variables passed on to cell/neuron
    '''
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
    '''
    The synapse class, pointprocesses that spawn membrane currents.
    See http://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#pointprocesses
    for details, or corresponding mod-files.
    
    Usage:
    ::
        ...
        cell = LFPy.Cell(**cellParameters)
        
        synapseParameters = {
            'idx' : cell.get_closest_idx(x=0, y=0, z=800),
            'e' : 0,                                # reversal potential
            'syntype' : 'ExpSyn',                   # synapse type
            'tau' : 2,                              # syn. time constant
            'weight' : 0.01,                       # syn. weight
            'record_current' : True                 # syn. current record
        }
        
        synapse = LFPy.PointProcessSynapse(cell, **synapseParameters)
        synapse.set_spike_times(cell, pl.array([10, 15, 20, 25]))
        cell.simulate(rec_isyn=True)
    '''
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
    Electrode currents go here, whics make membrane currents not sum to zero.
    
    Refer to NEURON documentation @ neuron.yale.edu for kwargs
        
    'cell' is the cell instance.
        
    Usage:
    ::
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
    def __init__(self, cell, idx, pptype='SEClamp',
                 color='p', marker='*', record_current=False, **kwargs):
        '''
        Will insert pptype on
        cell-instance, pass the corresponding kwargs onto
        cell.set_point_process.
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

class PointProcessPlayInSoma:
    '''class implementation of Eivind's playback alghorithm'''
    def __init__(self, soma_trace):
        '''Class for playing back somatic trace at specific time points
        into soma as boundary condition for the membrane voltage'''
        self.soma_trace = soma_trace
    
    def set_play_in_soma(self, cell, t_on=numpy.array([0])):
        '''Set mechanisms for playing soma trace at time(s) t_on,
        where t_on is a numpy.array'''
        if type(t_on) != numpy.ndarray:
            t_on = numpy.array(t_on)
        
        f = file(self.soma_trace)
        x = []
        for line in f.readlines():
            x.append(map(float, line.split()))
        x = numpy.array(x)
        X = x.T
        f.close()
        
        #time and values for trace, shifting
        tTrace = X[0, ]
        tTrace -= tTrace[0]
        
        trace = X[1, ]
        trace -= trace[0]
        trace += cell.e_pas
        
        #creating trace
        somaTvec0 = tTrace
        somaTvec0 += t_on[0]
        somaTvec = somaTvec0
        somaTrace = trace
        
        for i in xrange(1, t_on.size):
            numpy.concatenate((somaTvec, somaTvec0 + t_on[i]))
            numpy.concatenate((somaTrace, trace))
        
        somaTvec1 = numpy.interp(numpy.arange(somaTvec[0], somaTvec[-1], 
                                cell.timeres_NEURON),
                                somaTvec, somaTvec)
        somaTrace1 = numpy.interp(numpy.arange(somaTvec[0], somaTvec[-1],
                                cell.timeres_NEURON),
                                somaTvec, somaTrace)
        
        somaTvecVec = neuron.h.Vector(somaTvec1)
        somaTraceVec = neuron.h.Vector(somaTrace1)
        
        for sec in neuron.h.somalist:
            #ensure that soma is perfect capacitor
            sec.cm = 1E9
            #Why the fuck doesnt this work:
            #for seg in sec:
            #    somaTraceVec.play(seg._ref_v, somaTvecVec)
        
        #call function that insert trace on soma
        self._play_in_soma(somaTvecVec, somaTraceVec)
            
    def _play_in_soma(self, somaTvecVec, somaTraceVec):
        '''Replacement of LFPy.hoc "proc play_in_soma()",
        seems necessary that this function lives in hoc'''
        neuron.h('objref soma_tvec, soma_trace')
        
        neuron.h('soma_tvec = new Vector()')
        neuron.h('soma_trace = new Vector()')
        
        neuron.h.soma_tvec.from_python(somaTvecVec)
        neuron.h.soma_trace.from_python(somaTraceVec)
        
        neuron.h('soma_trace.play(&soma.v(0.5), soma_tvec)')
