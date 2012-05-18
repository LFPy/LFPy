#!/usr/bin/env python
'''
Copyright (C) 2012 Computational Neuroscience Group, UMB.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''

import os
import neuron
import numpy as np
import cPickle
from LFPy import RecExtElectrode
from LFPy.run_simulation import _run_simulation, _run_simulation_with_electrode

class Cell(object):
    '''
    The main cell class used in LFPy.
    
    Arguments:
    ::
        morphology : path to morphology file;
    
        v_init: initial potential;
        passive: passive mechs are initialized if True;
        Ra: axial resistance;
        rm: membrane resistivity;
        cm: membrane capacitance;
        e_pas: passive mechanism reversal potential;
    
        timeres_NEURON: internal dt for NEURON simulation;
        timeres_python: overall dt for python simulation;
    
        tstartms: initialization time for simulation <= 0 ms
        tstopms: stop time for simulation > 0 ms
    
        nsegs_method: method for setting the number of segments;
        max_nsegs_length: max segment length for method 'fixed_length';
        lambda_f: AC frequency for method 'lambda_f';
    
        custom_code: list of model-specific code files ([.py/.hoc]);
        custom_fun: list of model-specific functions to be called with args:
        custom_fun_args: list of arguments passed to custom_fun functions
        verbose: switching verbose output on/off
    
    Usage of cell class:
    ::
        import LFPy
        cellParameters = {                          
            'morphology' : 'path/to/morphology',
            'rm' : 30000,
            'cm' : 1.0,
            'Ra' : 150,
            'timeres_NEURON' : 0.1,
            'timeres_python' : 0.1,
            'tstartms' : -50,
            'tstopms' : 50,
        }
        cell = LFPy.Cell(**cellParameters)
        cell.simulate()
    '''
    def __init__(self, morphology,
                    v_init=-65.,
                    Ra=150,
                    rm=30000,
                    cm=1.0,
                    e_pas=-65.,
                    passive = True,
                    timeres_NEURON=2**-3,
                    timeres_python=2**-3,
                    tstartms=0,
                    tstopms=100,
                    nsegs_method='lambda100',
                    max_nsegs_length=30,
                    lambda_f = 100,
                    custom_code=None,
                    custom_fun=None,
                    custom_fun_args=None,
                    verbose=False):
        '''
        Initialization of the Cell object.
        '''
        self.verbose = verbose
        
        neuron.h.load_file('stdlib.hoc')    #NEURON std. library
        
        #load morphology
        if os.path.isfile(morphology):
            self.morphology = morphology
            self._load_geometry()
        else:
            raise Exception, "File %s does not exist!" % morphology
        
        #Some parameters and lists initialised
        if timeres_python not in 2.**np.arange(-16, -1) or timeres_NEURON \
                not in 2.**np.arange(-16, -1):
            if self.verbose:
                print 'timeres_python and or timeres_NEURON not a power of 2,',
                print 'cell.tvec errors may occur.',
                print 'Initialization will continue.'
            else:
                pass
        if timeres_python < timeres_NEURON:
            raise ValueError, 'timeres_python = %.3e < timeres_NEURON = %.3e' \
                                        % (timeres_python, timeres_NEURON)
        self.timeres_python = timeres_python
        self.timeres_NEURON = timeres_NEURON
        
        self.tstartms = tstartms
        self.tstopms = tstopms
        
        self.synapses = []
        self.synidx = []
        self.pointprocesses = []
        self.pointprocess_idx = []
        
        self.v_init = v_init
        
        self.default_rotation = self._get_rotation()
        
        if passive:
            #Set passive properties, insert passive on all segments
            self.Ra = Ra
            self.rm = rm
            self.cm = cm
            self.e_pas = e_pas
            self._set_passive()
        else:
            if self.verbose:
                print 'No passive properties added'
        
        #run user specified code and functions if argument given
        if custom_code != None or custom_fun != None:
            self._run_custom_codes(custom_code, custom_fun, custom_fun_args)
        
        #Insert extracellular mech on all segments
        self._set_extracellular()
        
        #set number of segments accd to rule, and calculate the number
        self._set_nsegs(nsegs_method, lambda_f, max_nsegs_length)
        self.totnsegs = self._calc_totnsegs()
        if self.verbose:
            print "Total number of segments = ", self.totnsegs
        
        #Gather geometry, set position and rotation of morphology
        self._collect_geometry()
        if hasattr(self, 'somapos'):
            self.set_pos()
        else:
            if self.verbose:
                print 'will not set position, no soma section.'
        self.set_rotation(**self.default_rotation)
        
    def _load_geometry(self):
        '''Load the morphology-file in NEURON''' 
        try: 
            neuron.h.sec_counted = 0
        except LookupError:
            neuron.h('sec_counted = 0')
        
        #Not sure if all of these are needed, just precautions
        neuron.h('objref axonlist, dendlist, apicdendlist')
        neuron.h('objref somalist, allseclist, alldendlist')
        neuron.h.somalist = None
        neuron.h.dendlist = None
        neuron.h.axonlist = None
        neuron.h.apicdendlist = None
        neuron.h('forall delete_section()')
        
        #import the morphology, try and determine format
        fileEnding = self.morphology.split('.')[-1]
        if fileEnding == 'hoc' or fileEnding == 'HOC':
            neuron.h.load_file(1, self.morphology)
        else:
            neuron.h.load_file('import3d.hoc')
            neuron.h('objref this')
            if fileEnding == 'asc' or fileEnding == 'ASC':
                Import = neuron.h.Import3d_Neurolucida3()
                if not self.verbose:
                    Import.quiet = 1
            elif fileEnding == 'swc' or fileEnding ==  'SWC':
                Import = neuron.h.Import3d_SWC_read()
            elif fileEnding == 'xml' or fileEnding ==  'XML':
                Import = neuron.h.Import3d_MorphML()
            else:
                raise ValueError, \
                    '%s is not a recognised morphology file format! ',\
                    'Should be either .hoc, .asc, .swc, .xml!' \
                     % self.morphology
            
            #assuming now that morphologies file is the correct format
            try:
                Import.input(self.morphology)
            except:
                if not hasattr(neuron, 'neuroml'):
                    raise Exception, 'Can not import, try and copy the ' + \
                    'nrn/share/lib/python/neuron/neuroml ' + \
                    'folder into %s' % neuron.__path__[0]
                else:
                    raise Exception, 'something wrong with file, see output'
            try:
                imprt = neuron.h.Import3d_GUI(Import, 0)
            except:
                raise Exception, 'See output, try to correct the file'
            imprt.instantiate(neuron.h.this)
            
        neuron.h.define_shape()
        self._create_sectionlists()
        
    def _run_custom_codes(self, custom_code, custom_fun, custom_fun_args):
        '''Execute custom model code and functions with arguments'''
        # load custom codes
        if custom_code != None:
            for code in custom_code:
                if code.split('.')[-1] == 'hoc':
                    try:
                        neuron.h.xopen(code)
                    except RuntimeError:
                        ERRMSG = '\n'.join(['', 
                            'Could not load custom model code (%s)' %code, 
                            'while creating a Cell object.',
                            'One possible cause is the NEURON mechanisms have not been compiled, ',
                            'try running nrnivmodl. ',])
                        raise Exception, ERRMSG
                elif code.split('.')[-1] == 'py':
                    exec(code)
                else:
                    raise Exception, '%s not a .hoc- nor .py-file' % code
        
        # run custom functions with arguments
        i = 0
        if custom_fun != None:
            for fun in custom_fun:
                fun(**custom_fun_args[i])
                i +=  1
        
        #recreate sectionlists in case something changed
        neuron.h.define_shape()
        self._create_sectionlists()
    
    def _set_nsegs(self, nsegs_method, lambda_f, max_nsegs_length):
        '''Set number of segments per section according to the lambda-rule,
        or according to maximum length of segments'''
        if nsegs_method == 'lambda100':
            self._set_nsegs_lambda100()
        elif nsegs_method == 'lambda_f':
            self._set_nsegs_lambda_f(lambda_f)
        elif nsegs_method == 'fixed_length':
            self._set_nsegs_fixed_length(max_nsegs_length)
        else:
            if self.verbose:
                print 'No nsegs_method applied (%s)' % nsegs_method
    
    def _get_rotation(self):
        '''Check if there exists a corresponding file
        with rotation angles'''
        base = os.path.splitext(self.morphology)[0]
        if os.path.isfile(base+'.rot'):
            rotation_file = base+'.rot'
            rotation_data = open(rotation_file)
            rotation = {}
            for line in rotation_data:
                var, val = line.split('=')
                val = val.strip()
                val = float(str(val))
                rotation[var] = val
        else:
            rotation = {
                }
        return rotation

    def _create_sectionlists(self):
        '''Create section lists for different kinds of sections'''
        self.somalist = neuron.h.SectionList()
        self.axonlist = neuron.h.SectionList()
        self.dendlist = neuron.h.SectionList()
        self.apiclist = neuron.h.SectionList()
        
        if neuron.h.sec_counted == 0:
            self.nsomasec = 0
            #Place sections in lists
            for sec in neuron.h.allsec():
                if sec.name()[:4] == 'soma':
                    self.nsomasec += 1
                    self.somalist.append(sec)
                elif sec.name()[:4] == 'axon':
                    self.axonlist.append(sec)
                elif sec.name()[:4] == 'dend':
                    self.dendlist.append(sec)
                elif sec.name()[:4] == 'apic':
                    self.apiclist.append(sec)
        elif neuron.h.sec_counted == 1:
            self.nsomasec = neuron.h.nsomasec
            #Place sections in lists
            for sec in neuron.h.soma:
                self.somalist.append(sec)
            for sec in neuron.h.dendlist:
                self.dendlist.append(sec)
            for sec in neuron.h.allsec():
                if sec.name()[:4] == 'apic':
                    self.apiclist.append(sec)
            try:
                for sec in neuron.h.axonlist:
                    self.axonlist.append(sec)
            except:
                pass
        
        #list with all sections
        self.allseclist = neuron.h.SectionList()
        for sec in self.somalist:
            self.allseclist.append(sec)
        for sec in self.dendlist:
            self.allseclist.append(sec)
        for sec in self.apiclist:
            self.allseclist.append(sec)
        for sec in self.axonlist:
            self.allseclist.append(sec)
        
        #list with all dendritic sections
        self.alldendlist = neuron.h.SectionList()
        for sec in self.dendlist:
            self.alldendlist.append(sec)
        for sec in self.apiclist:
            self.alldendlist.append(sec)
    
    def _get_idx(self, seclist):
        '''Return boolean vector which indexes where segments in seclist 
        matches segments in self.allseclist, rewritten from 
        LFPy.hoc function get_idx()'''
        if self.allseclist == seclist:
            return np.ones(self.totnsegs)
        else:
            idxvec = np.zeros(self.totnsegs)
            i = 0
            for sec in self.allseclist:
                for seg in sec:
                    for secl in seclist:
                        if sec.name() == secl.name():
                            idxvec[i] = 1
                    i += 1
            return idxvec
    
    def _set_nsegs_lambda_f(self, frequency):
        '''Set the number of segments for section according to the 
        d_lambda-rule for a given input frequency'''
        for sec in self.allseclist:
            sec.nseg = int((sec.L / (0.1 * neuron.h.lambda_f(frequency)) + .9)
                / 2 )*2 + 1
        if self.verbose:
            print "set nsegs using lambda-rule with frequency %i." % frequency
   
    def _set_nsegs_lambda100(self):
        '''Set the numbers of segments using d_lambda(100)'''
        self._set_nsegs_lambda_f(100)
    
    def _set_nsegs_fixed_length(self, maxlength):
        '''Set nseg for sections so that every segment L < maxlength'''
        for sec in self.allseclist:
            sec.nseg = int(sec.L / maxlength) + 1
    
    def _calc_totnsegs(self):
        '''Calculate the number of segments in the allseclist'''
        i = 0
        for sec in self.allseclist:
            i += sec.nseg
        
        return i
    
    def _check_currents(self):
        '''Check that the sum of all membrane and electrode currents over all
        segments is sufficiently close to zero'''
        raise NotImplementedError, 'this function need to be written'
    
    def _set_passive(self):
        '''Insert passive mechanism on all segments'''
        for sec in self.allseclist:
            sec.insert('pas')
            sec.Ra = self.Ra
            sec.cm = self.cm
            sec.g_pas = 1. / self.rm
            sec.e_pas = self.e_pas
    
    def _set_extracellular(self):
        '''Insert extracellular mechanism on all sections
        to access i_membrane'''
        for sec in self.allseclist:
            sec.insert('extracellular')
            
    def set_synapse(self, idx, syntype,
                    record_current=False, record_potential=False,
                    weight=None, **kwargs):
        '''
        Insert syntype (e.g. ExpSyn) synapse on segment with index idx, 
        **kwargs passed on from class PointProcessSynapse.
        '''

        if not hasattr(self, 'synlist'):
            self.synlist = neuron.h.List()
        if not hasattr(self, 'synireclist'):
            self.synireclist = neuron.h.List()
        if not hasattr(self, 'synvreclist'):
            self.synvreclist = neuron.h.List()
        if not hasattr(self, 'netconlist'):
            self.netconlist = neuron.h.List()
        if not hasattr(self, 'sptimeslist'):
            self.sptimeslist = neuron.h.List()

        i = 0 
        cmd1 = 'syn = neuron.h.'
        cmd2 = '(seg.x, sec=sec)'
        for sec in self.allseclist:
            for seg in sec:
                if i == idx:
                    command = cmd1 + syntype + cmd2  
                    exec(command)
                    for param in kwargs.keys():
                        try:
                            exec('syn.' + param + '=' + str(kwargs[param]))
                        except:
                            pass
                    self.synlist.append(syn)  

                    #create NetCon
                    netstim = neuron.h.NetStim()
                    nc = neuron.h.NetCon(netstim, syn)
                    nc.weight[0] = weight
                    self.netconlist.append(nc)

                    #record currents
                    if record_current:
                        synirec = neuron.h.Vector(int(self.tstopms /
                                                      self.timeres_python+1))
                        synirec.record(syn._ref_i, self.timeres_python)
                        self.synireclist.append(synirec)

                    #record potential
                    if record_potential:
                        synvrec = neuron.h.Vector(int(self.tstopms /
                                                      self.timeres_python+1))
                        synvrec.record(seg._ref_v, self.timeres_python)
                        self.synvreclist.append(synvrec)

                i += 1

        return self.synlist.count() - 1

    def set_point_process(self, idx, pptype, record_current=False, **kwargs):
        '''
        Insert pptype-electrode type pointprocess on segment numbered
        idx on cell object, with keyword arguments according to types:
        SEClamp, VClamp, IClamp, SinIClamp, ChirpIClamp.
        idx, pptype, **kwargs is passed on from PointProcessElectrode class.
        '''
        
        if not hasattr(self, 'stimlist'):
            self.stimlist = neuron.h.List()
        if not hasattr(self, 'stimireclist'):
            self.stimireclist = neuron.h.List()
        
        i = 0
        cmd1 = 'stim = neuron.h.'
        cmd2 = '(seg.x, sec=sec)'
        for sec in self.allseclist:
            for seg in sec:
                if i == idx:
                    command = cmd1 + pptype + cmd2  
                    exec(command)
                    for param in kwargs.keys():
                        exec('stim.' + param + '=' + str(kwargs[param]))
                    self.stimlist.append(stim)
                    
                    #record current
                    if record_current:
                        stimirec = neuron.h.Vector(int(self.tstopms /
                                                       self.timeres_python+1))
                        stimirec.record(stim._ref_i, self.timeres_python)
                        self.stimireclist.append(stimirec)
                
                i += 1
        
        return self.stimlist.count() - 1
    
    def _collect_geometry(self):
        '''Collects x, y, z-coordinates from NEURON'''
        self._collect_geometry_neuron()
        self._calc_midpoints()
        
        self.somaidx = self.get_idx(section='soma')
                
        if self.somaidx.size == 0:
            if self.verbose:
                print 'There is no soma!'
                print 'using first segment as root point'
            self.somaidx = np.array([0])
            self.somapos = np.zeros(3)
            self.somapos[0] = self.xmid[self.somaidx]
            self.somapos[1] = self.ymid[self.somaidx]
            self.somapos[2] = self.zmid[self.somaidx]
        elif self.somaidx.size == 1:
            self.somapos = np.zeros(3)
            self.somapos[0] = self.xmid[self.somaidx]
            self.somapos[1] = self.ymid[self.somaidx]
            self.somapos[2] = self.zmid[self.somaidx]
        elif self.somaidx.size > 1:
            xmids = self.xmid[self.somaidx]
            ymids = self.ymid[self.somaidx]
            zmids = self.zmid[self.somaidx]
            
            self.somapos = np.zeros(3)
            self.somapos[0] = xmids.mean()
            self.somapos[1] = ymids.mean()
            self.somapos[2] = zmids.mean()
    
    def _collect_geometry_neuron(self):
        '''Loop over allseclist to determine area, diam, xyz-start- and
        endpoints, embed geometry to cell object'''
        
        areavec = neuron.h.Vector(self.totnsegs)
        diamvec = neuron.h.Vector(self.totnsegs)
        lengthvec = neuron.h.Vector(self.totnsegs)
        
        xstartvec = neuron.h.Vector(self.totnsegs)
        xendvec = neuron.h.Vector(self.totnsegs)
        ystartvec = neuron.h.Vector(self.totnsegs)
        yendvec = neuron.h.Vector(self.totnsegs)
        zstartvec = neuron.h.Vector(self.totnsegs)
        zendvec = neuron.h.Vector(self.totnsegs)
        
        counter = 0
        
        #loop over all segments
        for sec in self.allseclist:
            if neuron.h.n3d() > 0:
                #length of sections
                xlength = neuron.h.x3d(neuron.h.n3d() - 1) - neuron.h.x3d(0)
                ylength = neuron.h.y3d(neuron.h.n3d() - 1) - neuron.h.y3d(0)
                zlength = neuron.h.z3d(neuron.h.n3d() - 1) - neuron.h.z3d(0)
                
                for seg in sec:
                    areavec.x[counter] = neuron.h.area(seg.x)
                    diamvec.x[counter] = seg.diam
                    lengthvec.x[counter] = sec.L/sec.nseg
                    
                    xstartvec.x[counter] = neuron.h.x3d(0) + \
                        xlength * (seg.x - 1./2./sec.nseg)
                    xendvec.x[counter] = neuron.h.x3d(0) + \
                        xlength * (seg.x + 1./2./sec.nseg)
                    
                    ystartvec.x[counter] = neuron.h.y3d(0) + \
                        ylength * (seg.x - 1./2./sec.nseg)
                    yendvec.x[counter] = neuron.h.y3d(0) + \
                        ylength * (seg.x + 1./2./sec.nseg)
                    
                    zstartvec.x[counter] = neuron.h.z3d(0) + \
                        zlength * (seg.x - 1./2./sec.nseg)
                    zendvec.x[counter] = neuron.h.z3d(0) + \
                        zlength * (seg.x + 1./2./sec.nseg)
                    
                    counter += 1
        
        self.xstart = np.array(xstartvec)
        self.ystart = np.array(ystartvec)
        self.zstart = np.array(zstartvec)
        
        self.xend = np.array(xendvec)
        self.yend = np.array(yendvec)
        self.zend = np.array(zendvec)
        
        self.area = np.array(areavec)
        self.diam = np.array(diamvec)
        self.length = np.array(lengthvec)
    
    def _calc_midpoints(self):
        '''Calculate midpoints of each segment'''
        self.xmid = .5*(self.xstart+self.xend)
        self.ymid = .5*(self.ystart+self.yend)
        self.zmid = .5*(self.zstart+self.zend)

    def get_idx(self, section='allsec', z_min=-10000, z_max=10000):
        '''Returns neuron idx of segments on interval [z_min, z_max]
        '''
        if section == 'allsec': 
            seclist = self.allseclist
        elif section == 'soma': 
            seclist = self.somalist
        elif section == 'dend': 
            seclist = self.dendlist
        elif section == 'apic': 
            seclist = self.apiclist
        elif section == 'alldend': 
            seclist = self.alldendlist
        elif section == 'axon': 
            seclist = self.axonlist
        else:
            sections = ['allsec', 'soma', 'alldend', 'dend', 'apic', 'axon']
            raise Exception, "section %s is not any of %s" % (section, 
                                                              str(sections))
        
        idx = np.where(self._get_idx(seclist))[0]
        
        sel_z_idx = np.where(np.logical_and(self.zmid[idx] > z_min,
                                                self.zmid[idx] < z_max))
        return idx[sel_z_idx]
                
    def get_closest_idx(self, x=0, y=0, z=0, section='allsec'):
        '''Get the index number of a segment in specified section which 
        midpoint is closest to the coordinates defined by the user
        '''
        idx = self.get_idx(section)
        dist = np.sqrt((self.xmid[idx] - x)**2 + \
            (self.ymid[idx] - y)**2 + (self.zmid[idx] - z)**2)
        
        mindist = np.where(dist == np.min(dist))
        
        return int(idx[mindist])
    
    def get_rand_idx_area_norm(self, section='allsec', nidx=1,
                               z_min=-10000, z_max=10000):
        '''Return nidx segment indices in section with random probability
        normalized to the membrane area of segment on 
        interval [z_min, z_max]
        '''
        poss_idx = self.get_idx(section=section, z_min=z_min, z_max = z_max)
        idx = np.empty(nidx, dtype=int)
        
        for i in xrange(nidx):
            idx[i] = np.min(np.nonzero(np.random.rand() < \
                np.cumsum(self.area[poss_idx]/sum(self.area[poss_idx])))[0])
        
        return poss_idx[idx]
    
    def simulate(self, electrode=None, rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
                 rec_variables=[]):
        '''
        This is the main function running the simulation of the NEURON model.
        Start NEURON simulation and record variables specified by arguments.
        
        Arguments:
        ::
            electrode:  Either an LFPy.RecExtElectrode object or a list of such.
                        If supplied, LFPs will be calculated at every time step
                        and accessible as electrode.LFP. If a list of objects
                        is given, accessible as electrode[0].LFP etc.
            rec_imem:   If true, segment membrane currents will be recorded
                        If no electrode argument is given, it is necessary to
                        set rec_imem=True in order to calculate LFP later on.
            rec_vmem:   record segment membrane voltages
            rec_ipas:   record passive segment membrane currents
            rec_icap:   record capacitive segment membrane currents
            rec_isyn:   record synaptic currents of from Synapse class instances
            rec_vmemsyn:    record membrane voltage of segments with Synapse
            rec_istim:  record currents of StimIntraElectrode
            rec_variables: list of variables to record, i.e arg=['cai', ]
        '''
        self._set_soma_volt_recorder()
        self._collect_tvec()
        
        if rec_imem:
            self._set_imem_recorders()
        if rec_vmem:
            self._set_voltage_recorders()
        if rec_ipas:
            self._set_ipas_recorders()
        if rec_icap:
            self._set_icap_recorders()
        if len(rec_variables) > 0:
            self._set_variable_recorders(rec_variables)
        
        #run fadvance until t >= tstopms, and calculate LFP if asked for
        if electrode == None:
            if not rec_imem:
                print "rec_imem = %s, membrane currents will not be recorded!" \
                                  % str(rec_imem)
            _run_simulation(self)
        else:
            if self.timeres_NEURON != self.timeres_python:
                raise ValueError, 'timeres_NEURON != timeres_python'
            _run_simulation_with_electrode(self, electrode)
        
        #somatic trace
        self.somav = np.array(self.somav)
        
        if rec_imem:
            self._calc_imem()        
        if rec_ipas:
            self._calc_ipas()        
        if rec_icap:
            self._calc_icap()        
        if rec_vmem:
            self._collect_vmem()        
        if rec_isyn:
            self._collect_isyn()        
        if rec_vmemsyn:
            self._collect_vsyn()        
        if rec_istim:
            self._collect_istim()
        if len(rec_variables) > 0:
            self._collect_rec_variables(rec_variables)

    def _collect_tvec(self):
        '''
        Set the tvec to be a monotonically increasing numpy array after sim.
        '''
        self.tvec = np.arange(self.tstopms / self.timeres_python + 1) \
                            * self.timeres_python
        
    def _calc_imem(self):
        '''
        Fetch the vectors from the memireclist and calculate self.imem
        containing all the membrane currents.
        '''
        self.imem = np.array(self.memireclist)
        for i in xrange(self.imem.shape[0]):
            self.imem[i, ] *= self.area[i] * 1E-2
        self.memireclist = None
        del self.memireclist
    
    def _calc_ipas(self):
        '''
        Get the passive currents
        '''
        self.ipas = np.array(self.memipasreclist)
        for i in xrange(self.ipas.shape[0]):
            self.ipas[i, ] *= self.area[i] * 1E-2
        self.memipasreclist = None
        del self.memipasreclist
    
    def _calc_icap(self):
        '''
        Get the capacitive currents
        '''
        self.icap = np.array(self.memicapreclist)
        for i in xrange(self.icap.shape[0]):
            self.icap[i, ] *= self.area[i] * 1E-2
        self.memicapreclist = None
        del self.memicapreclist
    
    def _collect_vmem(self):
        '''
        Get the membrane currents
        '''
        self.vmem = np.array(self.memvreclist)
        self.memvreclist = None
        del self.memvreclist
    
    def _collect_isyn(self):
        '''
        Get the synaptic currents
        '''
        for syn in self.synapses:
            if syn.record_current:
                syn.collect_current(self)
            else:
                raise Exception, 'must set record_current=True in Synapse class'
        self.synireclist = None
        del self.synireclist
    
    def _collect_vsyn(self):
        '''
        Collect the membrane voltage of segments with synapses
        '''
        for i in xrange(len(self.synapses)):
            self.synapses[i].collect_potential(self)
        self.synvreclist = None
        del self.synvreclist
    
    def _collect_istim(self):
        '''
        Get the pointprocess currents
        '''
        for i in xrange(len(self.pointprocesses)):
            if self.pointprocesses[i].record_current:
                self.pointprocesses[i].collect_current(self)
            else:
                raise Exception, 'must set record_current=True for pointp.'
        self.stimireclist = None
        del self.stimireclist
        
    def _collect_rec_variables(self, rec_variables):
        '''
        Create dict of np.arrays from recorded variables, each dictionary
        element named as the corresponding recorded variable name, i.e 'cai'
        '''
        self.rec_variables = {}
        i = 0
        for values in self.recvariablesreclist:
            self.rec_variables.update({rec_variables[i] : np.array(values)})
            if self.verbose:
                print 'collected recorded variable %s' % rec_variables[i] 
            i += 1
        del self.recvariablesreclist
    
    def loadspikes(self):
        '''
        Initialize spiketimes from netcon if they exist
        '''
        if hasattr(self, 'synlist'):
            for i in xrange(int(self.synlist.count())):
                for ii in xrange(int(self.sptimeslist.o(i).size)):
                    self.netconlist.o(i).event(float(self.sptimeslist.o(i)[ii]))

    
    def _set_soma_volt_recorder(self):
        '''
        Record somatic membrane potential
        '''
        self.somav = neuron.h.Vector(int(self.tstopms / 
                                         self.timeres_python+1))
        if self.nsomasec == 0:
            pass
        elif self.nsomasec == 1:
            self.somav.record(neuron.h.soma[0](0.5)._ref_v, 
                              self.timeres_python)
        elif self.nsomasec > 1:
            i, j = divmod(self.nsomasec, 2)
            self.somav.record(neuron.h.soma[int(i)](j/2)._ref_v, 
                              self.timeres_python)
    
    def _set_imem_recorders(self):
        '''
        Record membrane currents for all segments
        '''
        self.memireclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                memirec = neuron.h.Vector(int(self.tstopms / 
                                              self.timeres_python+1))
                memirec.record(seg._ref_i_membrane, self.timeres_python)
                self.memireclist.append(memirec)
    
    def _set_ipas_recorders(self):
        '''
        Record passive membrane currents for all segments
        '''
        self.memipasreclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                memipasrec = neuron.h.Vector(int(self.tstopms / 
                                                 self.timeres_python+1))
                memipasrec.record(seg._ref_i_pas, self.timeres_python)
                self.memipasreclist.append(memipasrec)
    
    def _set_icap_recorders(self):
        '''
        Record capacitive membrane currents for all segments
        '''
        self.memicapreclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                memicaprec = neuron.h.Vector(int(self.tstopms / 
                                                 self.timeres_python+1))
                memicaprec.record(seg._ref_i_cap, self.timeres_python)
                self.memicapreclist.append(memicaprec)
    
    def _set_voltage_recorders(self):
        '''
        Record membrane potentials for all segments
        '''
        self.memvreclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                memvrec = neuron.h.Vector(int(self.tstopms / 
                                              self.timeres_python+1))
                memvrec.record(seg._ref_v, self.timeres_python)
                self.memvreclist.append(memvrec)

    
    def _set_variable_recorders(self, rec_variables):
        '''
        Create a recorder for each variable name in list
        rec_variables
        
        Variables is stored in nested list self.recvariablesreclist
        '''
        self.recvariablesreclist = neuron.h.List()        
        for variable in rec_variables:
            variablereclist = neuron.h.List()
            self.recvariablesreclist.append(variablereclist)
            for sec in self.allseclist:
                for seg in sec:
                    recvector = neuron.h.Vector(int(self.tstopms /
                                                    self.timeres_python + 1))
                    if hasattr(seg, variable):
                        recvector.record(getattr(seg, '_ref_%s' % variable),
                                         self.timeres_python)
                    else:
                        print 'non-existing variable %s, section %s.%f' % \
                                (variable, sec.name(), seg.x)
                    variablereclist.append(recvector)
        
    
    def set_pos(self, xpos=0, ypos=0, zpos=0):
        '''
        Move the cell geometry so that midpoint of soma section is
        in (xpos, ypos, zpos). If no soma pos, use the first segment
        '''
        diffx = self.somapos[0]-xpos
        diffy = self.somapos[1]-ypos
        diffz = self.somapos[2]-zpos
        
        self.somapos[0] = xpos
        self.somapos[1] = ypos
        self.somapos[2] = zpos
        
        self.xstart = self.xstart - diffx
        self.ystart = self.ystart - diffy
        self.zstart = self.zstart - diffz
        
        self.xend = self.xend - diffx
        self.yend = self.yend - diffy
        self.zend = self.zend - diffz
        
        self._calc_midpoints()
        self._update_synapse_positions()
    
    def strip_hoc_objects(self):
        '''
        Destroy any NEURON hoc objects in the cell object
        '''
        for varname in dir(self):
            if type(getattr(self, varname)) == type(neuron.h.List()):
                setattr(self, varname, None)
                if self.verbose:
                    print 'None-typed %s in cell instance' % varname
        
    def cellpickler(self, filename):
        '''Save data in cell to filename, using cPickle. It will however destroy
        any neuron.h objects upon saving, as they cannot be pickled
        
        Usage:
        ::
            cell.cellpickler('cell.cpickle')
        
        To load this cell again in another session:
        ::
            import cPickle
            f = file('cell.cpickle', 'rb')
            cell = cPickle.load(f)
            f.close()
        or
        ::
            import LFPy
            cell = LFPy.tools.load('cell.cpickle')
        '''
        self.strip_hoc_objects()
        filen = open(filename, 'wb')
        cPickle.dump(self, filen, protocol=2)
        filen.close()
    
    def _update_synapse_positions(self):
        '''
        Update synapse positions after rotation of morphology
        '''
        for i in xrange(len(self.synapses)):
            self.synapses[i].update_pos(self)
    
    def set_rotation(self, x=None, y=None, z=None):
        '''
        Rotate geometry of cell object around the x-, y-, z-axis in that order.
        Input should be angles in radians.
        
        using rotation matrices, takes dict with rot. angles,
        where x, y, z are the rotation angles around respective axes.
        All rotation angles are optional.
        
        Usage:
        ::
            cell = LFPy.Cell(**kwargs)
            rotation = {'x' : 1.233, 'y' : 0.236, 'z' : np.pi}
            cell.set_rotation(**rotation)
        '''
        if x != None:
            theta = -x
            rotation_x = np.matrix([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])
            
            rel_start, rel_end = self._rel_positions()
            
            rel_start = rel_start * rotation_x
            rel_end = rel_end * rotation_x
            
            self._real_positions(rel_start, rel_end)
            if self.verbose:
                print 'Rotated geometry %g radians around x-axis' % (-theta)
        else:
            if self.verbose:
                print 'Geometry not rotated around x-axis'
        
        if y != None:
            phi = -y
            rotation_y = np.matrix([[np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]])
            
            rel_start, rel_end = self._rel_positions()
            
            rel_start = rel_start * rotation_y
            rel_end = rel_end * rotation_y
            
            self._real_positions(rel_start, rel_end)
            if self.verbose:
                print 'Rotated geometry %g radians around y-axis' % (-phi)
        else:
            if self.verbose:
                print 'Geometry not rotated around y-axis'
        
        if z != None:
            gamma = -z
            rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
            
            rel_start, rel_end = self._rel_positions()
            
            rel_start = rel_start * rotation_z
            rel_end = rel_end * rotation_z
            
            self._real_positions(rel_start, rel_end)
            if self.verbose:
                print 'Rotated geometry %g radians around z-axis' % (-gamma)
        else:
            if self.verbose:
                print 'Geometry not rotated around z-axis'
    
    
    def chiral_morphology(self, axis='x'):
        '''
        Mirror the morphology around given axis, (default x-axis),
        useful to introduce more heterogeneouties in morphology shapes
        '''
        #morphology relative to soma-position
        rel_start, rel_end = self._rel_positions()
        if axis == 'x':
            rel_start[:, 0] = -rel_start[:, 0]
            rel_end[:, 0] = -rel_end[:, 0]
        elif axis == 'y':
            rel_start[:, 1] = -rel_start[:, 1]
            rel_end[:, 1] = -rel_end[:, 1]
        elif axis == 'z':
            rel_start[:, 2] = -rel_start[:, 2]
            rel_end[:, 2] = -rel_end[:, 2]
        else:
            raise Exception, "axis must be either 'x', 'y' or 'z'"
        
        if self.verbose:
            print 'morphology mirrored across %s-axis' % axis
        
        #set the proper 3D positions
        self._real_positions(rel_start, rel_end)
        
    def _squeeze_me_macaroni(self):
        '''
        Reducing the dimensions of the morphology matrices from 3D->2D
        '''
        self.xstart = np.squeeze(np.array(self.xstart))
        self.xend = np.squeeze(np.array(self.xend))
        
        self.ystart = np.squeeze(np.array(self.ystart))
        self.yend = np.squeeze(np.array(self.yend))
        
        self.zstart = np.squeeze(np.array(self.zstart))
        self.zend = np.squeeze(np.array(self.zend))
    
    def _rel_positions(self):
        '''
        Morphology relative to soma position
        '''
        rel_start = np.transpose(np.array([self.xstart-self.somapos[0],
                                                self.ystart-self.somapos[1],
                                                self.zstart-self.somapos[2]]))
        rel_end = np.transpose(np.array([self.xend-self.somapos[0],
                                                self.yend-self.somapos[1],
                                                self.zend-self.somapos[2]]))
        return rel_start, rel_end
    
    def _real_positions(self, rel_start, rel_end):
        '''
        Morphology coordinates relative to Origo
        '''
        self.xstart = rel_start[:, 0] + self.somapos[0]
        self.ystart = rel_start[:, 1] + self.somapos[1]
        self.zstart = rel_start[:, 2] + self.somapos[2]
        
        self.xend = rel_end[:, 0] + self.somapos[0]
        self.yend = rel_end[:, 1] + self.somapos[1]
        self.zend = rel_end[:, 2] + self.somapos[2]
        
        self._squeeze_me_macaroni()
        self._calc_midpoints()
        self._update_synapse_positions()
    
    def get_rand_prob_area_norm(self, section='allsec', 
                                z_min=-10000, z_max=10000):
        '''
        Return the probability (0-1) for synaptic coupling on segments
        in section sum(prob)=1 over all segments in section.
        Prob. determined by area.
        '''
        idx = self.get_idx(section=section, z_min=z_min, z_max = z_max)
        prob = self.area[idx] / sum(self.area[idx])
        return prob

    def get_rand_prob_area_norm_from_idx(self, idx=np.array([0]), 
                                z_min=-10000, z_max=10000):
        '''
        Return the normalized probability (0-1) for synaptic coupling on
        segments in idx-array.
        Normalised probability determined by area of segments.
        '''
        prob = self.area[idx] / sum(self.area[idx])
        return prob
    
    def get_intersegment_vector(self, idx0=0, idx1=0):
        '''
        Return the distance between midpoints of two segments with index
        idx0 and idx1. The argument returned is a vector [x, y, z], where
        x = self.xmid[idx1] - self.xmid[idx0] etc.
        '''
        vector = []
        try:
            if idx1 < 0 or idx0 < 0:
                raise Exception
            vector.append(self.xmid[idx1] - self.xmid[idx0])
            vector.append(self.ymid[idx1] - self.ymid[idx0])
            vector.append(self.zmid[idx1] - self.zmid[idx0])
            return vector
        except:
            ERRMSG = 'idx0 and idx1 must be ints on [0, %i]' % self.totnsegs
            raise ValueError, ERRMSG
        
    def get_intersegment_distance(self, idx0=0, idx1=0):
        '''
        Return the Euclidean distance between midpoints of two segments 
        with indices idx0 and idx1. Will return a float in unit of micrometers.
        '''
        try:
            vector = np.array(self.get_intersegment_vector(idx0, idx1))
            return np.sqrt((vector**2).sum())
        except:
            ERRMSG = 'idx0 and idx1 must be ints on [0, %i]' % self.totnsegs
            raise ValueError, ERRMSG
    
    
    def get_idx_children(self, parent="soma[0]"):
        '''
        Get the idx of parent's children sections, i.e. compartments ids
        of sections connected to parent-argument'''
        idxvec = np.zeros(self.totnsegs)
        secnamelist = []
        childseclist = []
        #filling list of sectionnames for all sections, one entry per segment
        for sec in self.allseclist:
            for seg in sec:
                secnamelist.append(sec.name())
        #filling list of children section-names
        sref = neuron.h.SectionRef(parent)
        for sec in sref.child:
            childseclist.append(sec.name())
        #idxvec=1 where both coincide
        i = 0
        for sec in secnamelist:
            for childsec in childseclist:
                if sec == childsec:
                    idxvec[i] += 1
            i += 1
            
        [idx] = np.where(idxvec)
        return idx

    def get_idx_parent_children(self, parent="soma[0]"):
        '''
        Get all idx of segments of parent and children sections, i.e. segment
        idx of sections connected to parent-argument, and also of the parent
        segments
        '''
        idxvec = np.zeros(self.totnsegs)
        secnamelist = []
        childseclist = [parent]
        #filling list of sectionnames for all sections, one entry per segment
        for sec in self.allseclist:
            for seg in sec:
                secnamelist.append(sec.name())
        #filling list of children section-names
        sref = neuron.h.SectionRef(parent)
        for sec in sref.child:
            childseclist.append(sec.name())
        #idxvec=1 where both coincide
        i = 0
        for sec in secnamelist:
            for childsec in childseclist:
                if sec == childsec:
                    idxvec[i] += 1
            i += 1
            
        [idx] = np.where(idxvec)
        return idx
    
    def get_idx_section(self, section="soma[0]"):
        '''
        Get the idx of segments in any section-argument, uses hoc naming
        instead of general terms as in self.get_idx().
        
        For example:
        ::
            cell.get_idx_section(section='soma[0]')
        will output something like
        ::
            array([0])
            
        To list all sections in the cell object, type
        ::
            for sec in cell.allseclist:
                print sec.name()
        '''
        idxvec = np.zeros(self.totnsegs)
        secnamelist = []
        i = 0
        for sec in self.allseclist:
            for seg in sec:
                if sec.name() == section:
                    idxvec[i] += 1
                i += 1
        [idx] = np.where(idxvec)
        return idx
