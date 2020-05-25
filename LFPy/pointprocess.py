#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


import numpy as np
import neuron

class PointProcess(object):
    """
    Superclass on top of Synapse, StimIntElectrode, just to import and set
    some shared variables and extracts Cartesian coordinates of a segment
    
    Parameters
    ----------
    cell : obj
        LFPy.Cell object
    idx : int
        index of segment
    record_current : bool
        Must be set to True for recording of pointprocess currents
    record_potential : bool
        Must be set to True for recording potential of pointprocess target idx
    kwargs : pointprocess specific variables passed on to cell/neuron

    """
    def __init__(self, cell, idx, record_current=False, record_potential=False, **kwargs):
        """
        Initializes the PointProcess class
        """
        for key in ['color', 'marker']:
            if key in kwargs.keys():
                raise DeprecationWarning('Parameter {} has been deprecated'.format(key))
        self.idx = idx
        self.record_current = record_current
        self.record_potential = record_potential
        self.kwargs = kwargs
        self.update_pos(cell)

    def update_pos(self, cell):
        """
        Extract coordinates of point-process 
        """
        self.x = cell.xmid[self.idx]
        self.y = cell.ymid[self.idx]
        self.z = cell.zmid[self.idx]


class Synapse(PointProcess):
    """
    The synapse class, pointprocesses that spawn membrane currents.
    See http://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#pointprocesses
    for details, or corresponding mod-files.
    
    This class is meant to be used with synaptic mechanisms, giving rise to
    currents that will be part of the membrane currents. 

    Parameters
    ----------
    cell : obj
        `LFPy.Cell` or `LFPy.TemplateCell` instance to receive synapptic
        input
    idx : int
        Cell index where the synaptic input arrives
    syntype : str
        Type of synapse. Built-in examples: ExpSyn, Exp2Syn
    record_current : bool
        Decides if current is recorded
    **kwargs
        Additional arguments to be passed on to
        NEURON in `cell.set_synapse`

    Examples
    --------
    >>> import pylab as pl
    >>> pl.interactive(1)
    >>> import LFPy
    >>> import os
    >>> cellParameters = {
    >>>     'morphology' :  os.path.join('examples', 'morphologies', 'L5_Mainen96_LFPy.hoc'),
    >>>     'passive' : True,
    >>>     'tstop' :     50,
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)

    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=0, y=0, z=800),
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                        # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(pl.array([10, 15, 20, 25]))
    >>> cell.simulate()

    >>> pl.subplot(211)
    >>> pl.plot(cell.tvec, synapse.i)
    >>> pl.title('Synapse current (nA)')
    >>> pl.subplot(212)
    >>> pl.plot(cell.tvec, cell.somav)
    >>> pl.title('Somatic potential (mV)')

    """
    def __init__(self, cell, idx, syntype, record_current=False, record_potential=False, **kwargs):
        """
        Initialization of class Synapse
        """
        PointProcess.__init__(self, cell, idx, record_current, record_potential, **kwargs)
            
        self.syntype = syntype
        self.cell = cell
        self.hocidx = int(cell.set_synapse(idx=idx, syntype=syntype,
                                           record_current=record_current,
                                           record_potential=record_potential,
                                           **kwargs))
        self._ns_index = int(cell.netstimlist.count()) - 1
        cell.synapses.append(self)
        cell.synidx.append(idx)
        self.cell.sptimeslist.append(np.array([]))

    def set_spike_times(self, sptimes=np.zeros(0)):
        """Set the spike times explicitly using numpy arrays"""
        try:
            assert type(sptimes) is np.ndarray
        except AssertionError:
            raise AssertionError('synapse activation times must be a np.ndarray, not type({})'.format(type(sptimes)))
        self.cell.sptimeslist.insrt(self._ns_index, sptimes)
        self.cell.sptimeslist.remove(self._ns_index + 1)
    
    def set_spike_times_w_netstim(self, noise=1., start=0., number=1E3,
                                  interval=10., seed=1234.):
        """
        Generate a train of pre-synaptic stimulus times by setting up the
        neuron NetStim object associated with this synapse
        
        Parameters
        ----------
        noise : float in range [0, 1]
            Fractional randomness, from deterministic to intervals that drawn
            from negexp distribution (Poisson spiketimes).
        start : float
            ms, (most likely) start time of first spike
        number : int
            (average) number of spikes
        interval : float
            ms, (mean) time between spikes
        seed : float
            Random seed value
        """
        self.cell.netstimlist[self._ns_index].noise = noise
        self.cell.netstimlist[self._ns_index].start = start
        self.cell.netstimlist[self._ns_index].number = number
        self.cell.netstimlist[self._ns_index].interval = interval        
        self.cell.netstimlist[self._ns_index].seed(seed)

    def collect_current(self, cell):
        """Collect synapse current"""
        try:
            self.i = np.array(cell.synireclist.o(self.hocidx))
        except:
            raise Exception('cell.synireclist deleted from consequtive runs')
    
    def collect_potential(self, cell):
        """Collect membrane potential of segment with synapse"""
        try:
            self.v = np.array(cell.synvreclist.o(self.hocidx))
        except:
            raise Exception('cell.synvreclist deleted from consequtive runs')


class StimIntElectrode(PointProcess):
    """Class for NEURON point processes representing electrode currents,
    such as VClamp, SEClamp and ICLamp.
    
    Membrane currents will no longer sum to zero if these mechanisms are used,
    as the equivalent circuit is akin to a current input to the compartment
    from a far away extracellular location ("ground"), not immediately from
    the surface to the inside of the compartment as with transmembrane currents.
    
    Refer to NEURON documentation @ neuron.yale.edu for keyword arguments or 
    class documentation in Python issuing e.g.
        
        help(neuron.h.VClamp)
    
    Will insert pptype on cell-instance, pass the corresponding kwargs onto
    cell.set_point_process.

    Parameters
    ----------
    cell : obj
        `LFPy.Cell` or `LFPy.TemplateCell` instance to receive Stimulation
         electrode input
    idx : int
        Cell segment index where the stimulation electrode is placed
    pptype : str
        Type of point process. Built-in examples: VClamp, SEClamp and ICLamp.
        Defaults to 'SEClamp'.
    record_current : bool
        Decides if current is recorded
    record_potential : bool
        switch for recording the potential on postsynaptic segment index
    **kwargs
        Additional arguments to be passed on to
        NEURON in `cell.set_point_process`

    Examples
    --------
    >>> import pylab as pl
    >>> pl.ion()
    >>> import os
    >>> import LFPy
    >>> #define a list of different electrode implementations from NEURON
    >>> pointprocesses = [
    >>>     {
    >>>         'idx' : 0,
    >>>         'record_current' : True,
    >>>         'pptype' : 'IClamp',
    >>>         'amp' : 1,
    >>>         'dur' : 20,
    >>>         'delay' : 10,
    >>>     },
    >>>     {
    >>>         'idx' : 0,
    >>>         'record_current' : True,
    >>>         'pptype' : 'VClamp',
    >>>         'amp[0]' : -70,
    >>>         'dur[0]' : 10,
    >>>         'amp[1]' : 0,
    >>>         'dur[1]' : 20,
    >>>         'amp[2]' : -70,
    >>>         'dur[2]' : 10,
    >>>    },
    >>>    {
    >>>        'idx' : 0,
    >>>        'record_current' : True,
    >>>        'pptype' : 'SEClamp',
    >>>        'dur1' : 10,
    >>>        'amp1' : -70,
    >>>        'dur2' : 20,
    >>>        'amp2' : 0,
    >>>        'dur3' : 10,
    >>>        'amp3' : -70,
    >>>     },
    >>>  ]
    >>>  #create a cell instance for each electrode
    >>> for pointprocess in pointprocesses:
    >>>      cell = LFPy.Cell(morphology=os.path.join('examples', 'morphologies', 'L5_Mainen96_LFPy.hoc'),
    >>>                      passive=True)
    >>>      stimulus = LFPy.StimIntElectrode(cell, **pointprocess)
    >>>      cell.simulate()
    >>>      pl.subplot(211)
    >>>      pl.plot(cell.tvec, stimulus.i, label=pointprocess['pptype'])
    >>>      pl.legend(loc='best')
    >>>      pl.title('Stimulus currents (nA)')
    >>>      pl.subplot(212)
    >>>      pl.plot(cell.tvec, cell.somav, label=pointprocess['pptype'])
    >>>      pl.legend(loc='best')
    >>>      pl.title('Somatic potential (mV)')

    """    
    def __init__(self, cell, idx, pptype='SEClamp',
                 record_current=False,
                 record_potential=False, **kwargs):
        """Initialize StimIntElectrode class"""
        PointProcess.__init__(self, cell=cell, idx=idx,
                              record_current=record_current,
                              record_potential=record_potential)
        self.pptype = pptype
        self.hocidx = int(cell.set_point_process(idx, pptype,
                                                 record_current=record_current,
                                                 record_potential=record_potential,
                                                 **kwargs))
        cell.pointprocesses.append(self)
        cell.pointprocess_idx.append(idx)

    def collect_current(self, cell):
        """Fetch electrode current from recorder list"""
        self.i = np.array(cell.stimireclist.o(self.hocidx))

    def collect_potential(self, cell):
        """Collect membrane potential of segment with PointProcess"""
        self.v = np.array(cell.stimvreclist.o(self.hocidx))

