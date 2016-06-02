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
import pickle
from LFPy import RecExtElectrode
from LFPy.run_simulation import _run_simulation, _run_simulation_with_electrode
from LFPy.run_simulation import _collect_geometry_neuron
from LFPy.alias_method import alias_method
import sys
from warnings import warn

class Cell(object):
    '''
    The main cell class used in LFPy.
    
    Arguments:
    ::
        
        morphology : [str]: path/to/morphology/file
    
        v_init: [-65.]: initial potential
        passive: [True]/False: passive mechs are initialized if True
        Ra: [150.]: axial resistance
        rm: [30000]: membrane resistivity
        cm: [1.0]: membrane capacitance
        e_pas: [-65.]: passive mechanism reversal potential
        extracellular: [True]/False: switch for NEURON's extracellular mechanism
    
        timeres_NEURON: [0.1]: internal dt for NEURON simulation
        timeres_python: [0.1]: overall dt for python simulation
    
        tstartms: [0.]:  initialization time for simulation <= 0 ms
        tstopms: [100.]: stop time for simulation > 0 ms
    
        nsegs_method: ['lambda100']/'lambda_f'/'fixed_length': nseg rule
        max_nsegs_length: [None]: max segment length for method 'fixed_length'
        lambda_f: [100]: AC frequency for method 'lambda_f'
        d_lambda: [0.1]: parameter for d_lambda rule
        
        delete_sections: [True]: delete pre-existing section-references
        
        custom_code: [None]: list of model-specific code files ([.py/.hoc])
        custom_fun: [None]: list of model-specific functions with args
        custom_fun_args: [None]: list of args passed to custom_fun functions
        pt3d: True/[False]: use pt3d-info of the cell geometries switch
        celsius: [None]: Temperature in celsius. If nothing is specified here or in custom code it is 6.3 C
        verbose: True/[False]: verbose output switch
    
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
                    passive = True,
                    Ra=150,
                    rm=30000,
                    cm=1.0,
                    e_pas=-65.,
                    extracellular = True,
                    timeres_NEURON=2**-3,
                    timeres_python=2**-3,
                    tstartms=0,
                    tstopms=100,
                    nsegs_method='lambda100',
                    lambda_f = 100,
                    d_lambda = 0.1,
                    max_nsegs_length=None,
                    delete_sections = True,
                    custom_code=None,
                    custom_fun=None,
                    custom_fun_args=None,
                    pt3d=False,
                    celsius=None,
                    verbose=False):
        '''
        Initialization of the Cell object.
        '''
        self.verbose = verbose
        self.pt3d = pt3d
        
        if not hasattr(neuron.h, 'd_lambda'):
            neuron.h.load_file('stdlib.hoc')    #NEURON std. library
            neuron.h.load_file('import3d.hoc')  #import 3D morphology lib

        if delete_sections:
            numsec = 0
            for numsec, sec in enumerate(neuron.h.allsec()):
                pass
            if numsec > 0 and self.verbose:
                print(('%s existing sections deleted from memory' % numsec))
            neuron.h('forall delete_section()')

        #print a warning if neuron have existing sections
        numsec = 0
        for numsec, sec in enumerate(neuron.h.allsec()):
            pass
        if numsec > 0 and self.verbose:
            mssg = "%s sections detected! " % numsec + \
                   "Consider setting 'delete_sections=True'"
            warn(mssg)
                
        #load morphology
        self.morphology = morphology
        if self.morphology is not None:
            if os.path.isfile(self.morphology):
                self._load_geometry()
            else:
                raise Exception('non-existent file %s' % self.morphology)
        else:
            try:
                #will try to import top level cell and create sectionlist,
                #in case there were no morphology file loaded
                neuron.h.define_shape()
                self._create_sectionlists()
            except:
                raise Exception("Could not load existent top-level cell")
        
        #Some parameters and lists initialised
        if timeres_python not in 2.**np.arange(-16, -1) or timeres_NEURON \
                not in 2.**np.arange(-16, -1):
            if self.verbose:
                print('timeres_python and or timeres_NEURON not a power of 2,')
                print('cell.tvec errors may occur.')
                print('Initialization will continue.')
            else:
                pass
        if timeres_python < timeres_NEURON:
            raise ValueError('timeres_python = %.3e < timeres_NEURON = %.3e' \
                                        % (timeres_python, timeres_NEURON))
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
                print('No passive properties added')
        
        #run user specified code and functions if argument given
        if custom_code is not None or custom_fun is not None:
            self._run_custom_codes(custom_code, custom_fun, custom_fun_args)
        
        #Insert extracellular mech on all segments
        self.extracellular = extracellular
        if self.extracellular:
            self._set_extracellular()
        else:
            if self.verbose:
                print("no extracellular mechanism inserted, can't access imem!")
        
        #set number of segments accd to rule, and calculate the number
        self._set_nsegs(nsegs_method, lambda_f, d_lambda, max_nsegs_length)
        self.totnsegs = self._calc_totnsegs()
        if self.verbose:
            print("Total number of segments: %i" % self.totnsegs)
        
        #extract pt3d info from NEURON, and set these with the same rotation
        #and position in space as in our simulations, assuming RH rule, which
        #NEURON do NOT use in shape plot
        if self.pt3d:
            self.x3d, self.y3d, self.z3d, self.diam3d = self._collect_pt3d()

        #Gather geometry, set position and rotation of morphology
        self._collect_geometry()
        if hasattr(self, 'somapos'):
            self.set_pos()
        else:
            if self.verbose:
                print('no soma, using the midpoint if initial segment.')
        self.set_rotation(**self.default_rotation)

        if celsius is not None:
            if neuron.h.celsius != 6.3:
                print("Overwriting custom temperature of %1.2f. New temperature is %1.2f"
                      % (neuron.h.celsius, celsius))
            neuron.h.celsius = celsius


    def _load_geometry(self):
        '''Load the morphology-file in NEURON''' 
        try: 
            neuron.h.sec_counted = 0
        except LookupError:
            neuron.h('sec_counted = 0')
        
        #import the morphology, try and determine format
        fileEnding = self.morphology.split('.')[-1]
        if fileEnding == 'hoc' or fileEnding == 'HOC':
            neuron.h.load_file(1, self.morphology)
        else:
            neuron.h('objref this')
            if fileEnding == 'asc' or fileEnding == 'ASC':
                Import = neuron.h.Import3d_Neurolucida3()
                if not self.verbose:
                    Import.quiet = 1
            elif fileEnding == 'swc' or fileEnding == 'SWC':
                Import = neuron.h.Import3d_SWC_read()
            elif fileEnding == 'xml' or fileEnding == 'XML':
                Import = neuron.h.Import3d_MorphML()
            else:
                raise ValueError('%s is not a recognised morphology file format!'
                                 ).with_traceback(
                    'Should be either .hoc, .asc, .swc, .xml!' % self.morphology)
            
            #assuming now that morphologies file is the correct format
            try:
                Import.input(self.morphology)
            except:
                if not hasattr(neuron, 'neuroml'):
                    raise Exception('Can not import, try and copy the ' + \
                    'nrn/share/lib/python/neuron/neuroml ' + \
                    'folder into %s' % neuron.__path__[0])
                else:
                    raise Exception('something wrong with file, see output')
            try:
                imprt = neuron.h.Import3d_GUI(Import, 0)
            except:
                raise Exception('See output, try to correct the file')
            imprt.instantiate(neuron.h.this)
            
        neuron.h.define_shape()
        self._create_sectionlists()

        
    def _run_custom_codes(self, custom_code, custom_fun, custom_fun_args):
        '''Execute custom model code and functions with arguments'''
        # load custom codes
        if custom_code is not None:
            for code in custom_code:
                if code.split('.')[-1] == 'hoc':
                    try:
                        neuron.h.xopen(code)
                    except RuntimeError:
                        ERRMSG = '\n'.join(['', 
                            'Could not load custom model code (%s)' %code, 
                            'while creating a Cell object.',
                            'One possible cause is the NEURON mechanisms have',
                            'not been compiled, ',
                            'try running nrnivmodl. ',])
                        raise Exception(ERRMSG)
                elif code.split('.')[-1] == 'py':
                    exec(code)
                else:
                    raise Exception('%s not a .hoc- nor .py-file' % code)
        
        # run custom functions with arguments
        i = 0
        if custom_fun is not None:
            for fun in custom_fun:
                fun(**custom_fun_args[i])
                i +=  1
        
        #recreate sectionlists in case something changed
        neuron.h.define_shape()
        self._create_sectionlists()

    
    def _set_nsegs(self, nsegs_method, lambda_f, d_lambda, max_nsegs_length):
        '''Set number of segments per section according to the lambda-rule,
        or according to maximum length of segments'''
        if nsegs_method == 'lambda100':
            self._set_nsegs_lambda100(d_lambda)
        elif nsegs_method == 'lambda_f':
            self._set_nsegs_lambda_f(lambda_f, d_lambda)
        elif nsegs_method == 'fixed_length':
            self._set_nsegs_fixed_length(max_nsegs_length)
        else:
            if self.verbose:
                print(('No nsegs_method applied (%s)' % nsegs_method))
    
    def _get_rotation(self):
        '''Check if there exists a corresponding file
        with rotation angles'''
        if self.morphology is not None:
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
                rotation = {}
        else:
            rotation = {}
        return rotation

    def _create_sectionlists(self):
        '''Create section lists for different kinds of sections'''
        #list with all sections
        self.allsecnames = []
        self.allseclist = neuron.h.SectionList()
        for sec in neuron.h.allsec():
            self.allsecnames.append(sec.name())
            self.allseclist.append(sec=sec)
        
        #list of soma sections, assuming it is named on the format "soma*"
        self.nsomasec = 0
        self.somalist = neuron.h.SectionList()
        for sec in neuron.h.allsec():
            if sec.name().find('soma') >= 0:
                self.somalist.append(sec=sec)
                self.nsomasec += 1
            
    def _get_idx(self, seclist):
        '''Return boolean vector which indexes where segments in seclist 
        matches segments in neuron.h.allsec(), rewritten from 
        LFPy.hoc function get_idx()'''
        if neuron.h.allsec() == seclist:
            return np.ones(self.totnsegs, dtype=bool)
        else:
            idxvec = np.zeros(self.totnsegs, dtype=bool)
            #get sectionnames from seclist
            seclistnames = []
            for sec in seclist:
                seclistnames.append(sec.name())
            seclistnames = np.array(seclistnames, dtype='|S128')
            segnames = np.empty(self.totnsegs, dtype='|S128')
            i = 0
            for sec in self.allseclist:
                secname = sec.name()
                for seg in sec:
                    segnames[i] = secname
                    i += 1
            for name in seclistnames:
                idxvec[segnames == name] = True

            return idxvec
    
    def _set_nsegs_lambda_f(self, frequency=100, d_lambda=0.1):
        '''Set the number of segments for section according to the 
        d_lambda-rule for a given input frequency
        
        kwargs:
        ::
            
            frequency: float, frequency at whihc AC length constant is computed
            d_lambda: float, 
        '''
        for sec in self.allseclist:
            sec.nseg = int((sec.L / (d_lambda*neuron.h.lambda_f(frequency,
                                                           sec=sec)) + .9)
                / 2 )*2 + 1
        if self.verbose:
            print(("set nsegs using lambda-rule with frequency %i." % frequency))
   
    def _set_nsegs_lambda100(self, d_lambda=0.1):
        '''Set the numbers of segments using d_lambda(100)'''
        self._set_nsegs_lambda_f(frequency=100, d_lambda=d_lambda)
    
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
        raise NotImplementedError('this function need to be written')
    
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
        
        Arguments:
        ::
            
            idx : int
            syntype : str
            record_current : bool
            record_potential : bool
            weight : float            
            kwargs : arguments passed on from class Synapse
        '''
        if not hasattr(self, 'synlist'):
            self.synlist = neuron.h.List()
        if not hasattr(self, 'synireclist'):
            self.synireclist = neuron.h.List()
        if not hasattr(self, 'synvreclist'):
            self.synvreclist = neuron.h.List()
        if not hasattr(self, 'netstimlist'):
           self.netstimlist = neuron.h.List()
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
                    for param in list(kwargs.keys()):
                        try:
                            exec('syn.' + param + '=' + str(kwargs[param]))
                        except:
                            pass
                    self.synlist.append(syn)  

                    #create NetStim (generator) and NetCon (connection) objects
                    self.netstimlist.append(neuron.h.NetStim(0.5))
                    self.netstimlist[-1].number = 0
                    
                    nc = neuron.h.NetCon(self.netstimlist[-1], syn)
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
        idx on cell object, with keyword arguments according to types 
        SEClamp, VClamp, IClamp, SinIClamp, ChirpIClamp.
        
        Arguments:
        ::
            
            idx : int
            pptype : str
            record_current : bool
            kwargs : arguments passed on from class StimIntElectrode
            
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
                    for param in list(kwargs.keys()):
                        try:
                            exec('stim.' + param + '=' + str(kwargs[param]))
                        except SyntaxError:
                            ERRMSG = ''.join(['',
                                'Point process type "{0}" might not '.format(
                                    pptype),
                                'recognize attribute "{0}". '.format(param),
                                'Check for misspellings'])
                            raise Exception(ERRMSG)                            
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
        #None-type some attributes if they do not exis:
        if not hasattr(self, 'xstart'):
            self.xstart = None
            self.ystart = None
            self.zstart = None
            self.xend = None
            self.yend = None
            self.zend = None
            self.area = None
            self.diam = None
            self.length = None
        
        _collect_geometry_neuron(self)
        self._calc_midpoints()
        
        self.somaidx = self.get_idx(section='soma')
                
        if self.somaidx.size > 1:
            xmids = self.xmid[self.somaidx]
            ymids = self.ymid[self.somaidx]
            zmids = self.zmid[self.somaidx]
            self.somapos = np.zeros(3)
            self.somapos[0] = xmids.mean()
            self.somapos[1] = ymids.mean()
            self.somapos[2] = zmids.mean()
        elif self.somaidx.size == 1:
            self.somapos = np.zeros(3)
            self.somapos[0] = self.xmid[self.somaidx]
            self.somapos[1] = self.ymid[self.somaidx]
            self.somapos[2] = self.zmid[self.somaidx]
        elif self.somaidx.size == 0:
            if self.verbose:
                print('There is no soma!')
                print('using first segment as root point')
            self.somaidx = np.array([0])
            self.somapos = np.zeros(3)
            self.somapos[0] = self.xmid[self.somaidx]
            self.somapos[1] = self.ymid[self.somaidx]
            self.somapos[2] = self.zmid[self.somaidx]
        else:
            raise Exception('Huh?!')
    
    def _calc_midpoints(self):
        '''Calculate midpoints of each segment'''
        self.xmid = .5*(self.xstart+self.xend).flatten()
        self.ymid = .5*(self.ystart+self.yend).flatten()
        self.zmid = .5*(self.zstart+self.zend).flatten()


    def get_idx(self, section='allsec', z_min=-10000, z_max=10000):
        '''
        Returns neuron idx of segments from sections with names that match
        the pattern defined in input section on interval [z_min, z_max].
        
        kwargs:
        ::
            
            section: str, any entry in cell.allsecnames or just 'allsec'.
            z_min: float, depth filter
            z_max: float depth filter
        
        Usage:
        ::
            
            idx = cell.get_idx(section='allsec')
            print idx
            idx = cell.get_idx(section=['soma', 'dend', 'apic'])
            print idx
            
        '''
        if section == 'allsec': 
            seclist = neuron.h.allsec()
        else:
            seclist = neuron.h.SectionList()
            if type(section) == str:
                for sec in self.allseclist:
                    if sec.name().find(section) >= 0:
                        seclist.append(sec=sec)
            elif type(section) == list:
                for secname in section:
                    for sec in self.allseclist:
                        if sec.name().find(secname) >= 0:
                            seclist.append(sec=sec)
            else:
                if self.verbose:
                    print(('%s did not match any section name' % str(section)))

        idx = self._get_idx(seclist)
        sel_z_idx = (self.zmid[idx] > z_min) & (self.zmid[idx] < z_max)
        return np.arange(self.totnsegs)[idx][sel_z_idx]
                            
                
    def get_closest_idx(self, x=0, y=0, z=0, section='allsec'):
        '''Get the index number of a segment in specified section which 
        midpoint is closest to the coordinates defined by the user
        kwargs:
        ::
            
            x: float, coordinate
            y: float, coordinate
            z: float, coordinate
            section: str, string matching a section-name
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
        
        kwargs:
        ::
            
            section: str, string matching a section-name
            nidx: int, number of random indices
            z_min: float, depth filter
            z_max: float depth filter            
        '''
        poss_idx = self.get_idx(section=section, z_min=z_min, z_max=z_max)
        if nidx < 1:
            print('nidx < 1, returning empty array')
            return np.array([])
        elif poss_idx.size == 0:
            print('No possible segment idx match enquire! returning empty array')
            return np.array([])
        else:
            area = self.area[poss_idx]
            area /= area.sum()
            idx = alias_method(poss_idx, area, nidx)

            return idx
    
    def simulate(self, electrode=None, rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
                 rec_variables=[], variable_dt=False, atol=0.001,
                 to_memory=True, to_file=False, file_name=None,
                 dotprodcoeffs=None):
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
                        Units of (nA).
            rec_vmem:   record segment membrane voltages (mV)
            rec_ipas:   record passive segment membrane currents (nA)
            rec_icap:   record capacitive segment membrane currents (nA)
            rec_isyn:   record synaptic currents of from Synapse class (nA)
            rec_vmemsyn:    record membrane voltage of segments with Synapse(mV)
            rec_istim:  record currents of StimIntraElectrode (nA)
            rec_variables: list of variables to record, i.e arg=['cai', ]
            variable_dt: boolean, using variable timestep in NEURON
            atol:       absolute tolerance used with NEURON variable timestep 
            to_memory:  only valid with electrode, store lfp in -> electrode.LFP 
            to_file:    only valid with electrode, save LFPs in hdf5 file format 
            file_name:  name of hdf5 file, '.h5' is appended if it doesnt exist
            dotprodcoeffs :  list of N x Nseg np.ndarray. These arrays will at
                        every timestep be multiplied by the membrane currents.
                        Presumably useful for memory efficient csd or lfp calcs
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
        if electrode is None and dotprodcoeffs is None:
            if not rec_imem:
                print(("rec_imem = %s, membrane currents will not be recorded!" \
                                  % str(rec_imem)))
            _run_simulation(self, variable_dt, atol)
        else:
            #allow using both electrode and additional coefficients:
            _run_simulation_with_electrode(self, electrode, variable_dt, atol,
                                               to_memory, to_file, file_name,
                                               dotprodcoeffs)
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
        if hasattr(self, 'netstimlist'):
            del self.netstimlist

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
        for i in range(self.imem.shape[0]):
            self.imem[i, ] *= self.area[i] * 1E-2
        self.memireclist = None
        del self.memireclist
    
    def _calc_ipas(self):
        '''
        Get the passive currents
        '''
        self.ipas = np.array(self.memipasreclist)
        for i in range(self.ipas.shape[0]):
            self.ipas[i, ] *= self.area[i] * 1E-2
        self.memipasreclist = None
        del self.memipasreclist
    
    def _calc_icap(self):
        '''
        Get the capacitive currents
        '''
        self.icap = np.array(self.memicapreclist)
        for i in range(self.icap.shape[0]):
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
                raise Exception('must set record_current=True in Synapse class')
        self.synireclist = None
        del self.synireclist
    
    def _collect_vsyn(self):
        '''
        Collect the membrane voltage of segments with synapses
        '''
        for i in range(len(self.synapses)):
            self.synapses[i].collect_potential(self)
        self.synvreclist = None
        del self.synvreclist
    
    def _collect_istim(self):
        '''
        Get the pointprocess currents
        '''
        for i in range(len(self.pointprocesses)):
            if self.pointprocesses[i].record_current:
                self.pointprocesses[i].collect_current(self)
            else:
                raise Exception('must set record_current=True for pointp.')
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
                print(('collected recorded variable %s' % rec_variables[i])) 
            i += 1
        del self.recvariablesreclist
    
    def _loadspikes(self):
        '''
        Initialize spiketimes from netcon if they exist
        '''
        if hasattr(self, 'synlist'):
            if len(self.synlist) == len(self.sptimeslist):
                for i in range(int(self.synlist.count())):
                    for ii in range(int(self.sptimeslist.o(i).size)):
                        self.netconlist.o(i).event(float(self.sptimeslist.o(i)[ii]))
            # elif len(self.synlist) > 0 and len(self.sptimeslist) == 0:
            #     errmsg = 'please run method "set_spike_times() for every' + \
            #             '\n' + 'instance of LFPy.pointprocess.Synapse'
            #     raise Exception(errmsg)
            # else:
            #     pass
            

    
    def _set_soma_volt_recorder(self):
        '''
        Record somatic membrane potential
        '''
        self.somav = neuron.h.Vector(int(self.tstopms / 
                                         self.timeres_python+1))
        if self.nsomasec == 0:
            pass
        elif self.nsomasec == 1:
            for sec in self.somalist:
                self.somav.record(sec(0.5)._ref_v, 
                              self.timeres_python)
        elif self.nsomasec > 1:
            nseg = self.get_idx('soma').size
            i, j = divmod(nseg, 2)
            k = 1
            for sec in self.somalist:
                for seg in sec:
                    if nseg==2 and k == 1:
                        #if 2 segments, record from the first one: 
                        self.somav.record(seg._ref_v, self.timeres_python)
                    else:
                        if k == i*2:
                            #record from one of the middle segments:
                            self.somav.record(seg._ref_v,
                                              self.timeres_python)
                    k += 1
    
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
                        print(('non-existing variable %s, section %s.%f' % \
                                (variable, sec.name(), seg.x)))
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

        #also update the pt3d_pos:
        if self.pt3d and hasattr(self, 'x3d'):
                self._set_pt3d_pos()
        else:
            self.xstart -= diffx
            self.ystart -= diffy
            self.zstart -= diffz
            
            self.xend -= diffx
            self.yend -= diffy
            self.zend -= diffz
        
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
                    print(('None-typed %s in cell instance' % varname))
        
    def cellpickler(self, filename):
        '''
        Save data in cell to filename, using cPickle. It will however destroy
        any neuron.h objects upon saving, as c-objects cannot be pickled
        
        Usage:
        ::
            
            cell.cellpickler('cell.cpickle')
        
        To load this cell again in another session:
        ::
            
            import cPickle
            f = file('cell.cpickle', 'rb')
            cell = cPickle.load(f)
            f.close()
        
        alternatively:
        ::
            
            import LFPy
            cell = LFPy.tools.load('cell.cpickle')
            
        '''
        self.strip_hoc_objects()
        filen = open(filename, 'wb')
        pickle.dump(self, filen, protocol=2)
        filen.close()
    
    def _update_synapse_positions(self):
        '''
        Update synapse positions after rotation of morphology
        '''
        for i in range(len(self.synapses)):
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
        if x is not None:
            theta = -x
            rotation_x = np.matrix([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])
            
            rel_start, rel_end = self._rel_positions()
            
            rel_start = rel_start * rotation_x
            rel_end = rel_end * rotation_x
            
            self._real_positions(rel_start, rel_end)
            if self.verbose:
                print(('Rotated geometry %g radians around x-axis' % (-theta)))
        else:
            if self.verbose:
                print('Geometry not rotated around x-axis')
        
        if y is not None:
            phi = -y
            rotation_y = np.matrix([[np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]])
            
            rel_start, rel_end = self._rel_positions()
            
            rel_start = rel_start * rotation_y
            rel_end = rel_end * rotation_y
            
            self._real_positions(rel_start, rel_end)
            if self.verbose:
                print(('Rotated geometry %g radians around y-axis' % (-phi)))
        else:
            if self.verbose:
                print('Geometry not rotated around y-axis')
        
        if z is not None:
            gamma = -z
            rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
            
            rel_start, rel_end = self._rel_positions()
            
            rel_start = rel_start * rotation_z
            rel_end = rel_end * rotation_z
            
            self._real_positions(rel_start, rel_end)
            if self.verbose:
                print(('Rotated geometry %g radians around z-axis' % (-gamma)))
        else:
            if self.verbose:
                print('Geometry not rotated around z-axis')

        #rotate the pt3d geometry accordingly
        if self.pt3d and hasattr(self, 'x3d'):
            self._set_pt3d_rotation(x, y, z)

    
    def chiral_morphology(self, axis='x'):
        '''
        Mirror the morphology around given axis, (default x-axis),
        useful to introduce more heterogeneouties in morphology shapes
        
        kwargs:
        ::
            
            axis : str
                'x' or 'y' or 'z'
        
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
            raise Exception("axis must be either 'x', 'y' or 'z'")
        
        if self.verbose:
            print(('morphology mirrored across %s-axis' % axis))
        
        #set the proper 3D positions
        self._real_positions(rel_start, rel_end)
        
    def _squeeze_me_macaroni(self):
        '''
        Reducing the dimensions of the morphology matrices from 3D->1D
        '''
        self.xstart = np.array(self.xstart).flatten()
        self.xend = np.array(self.xend).flatten()
        
        self.ystart = np.array(self.ystart).flatten()
        self.yend = np.array(self.yend).flatten()
        
        self.zstart = np.array(self.zstart).flatten()
        self.zend = np.array(self.zend).flatten()
        
    
    def _rel_positions(self):
        '''
        Morphology relative to soma position
        '''
        rel_start = np.array([self.xstart-self.somapos[0],
                              self.ystart-self.somapos[1],
                              self.zstart-self.somapos[2]]).T    
        rel_end = np.array([self.xend-self.somapos[0],
                            self.yend-self.somapos[1],
                            self.zend-self.somapos[2]]).T

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

        kwargs:
        ::
           
            section: str, string matching a section-name
            z_min: float, depth filter
            z_max: float depth filter            

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

        kwargs:
        ::
            
            idx : np.ndarray, dtype=int.
                array of segment indices
            z_min: float, depth filter
            z_max: float depth filter            

        '''
        prob = self.area[idx] / sum(self.area[idx])
        return prob
    
    def get_intersegment_vector(self, idx0=0, idx1=0):
        '''
        Return the distance between midpoints of two segments with index
        idx0 and idx1. The argument returned is a vector [x, y, z], where
        x = self.xmid[idx1] - self.xmid[idx0] etc.
        
        kwargs:
        ::
            
            idx0 : int
            idx1 : int
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
            raise ValueError(ERRMSG)
        
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
            raise ValueError(ERRMSG)
    
    
    def get_idx_children(self, parent="soma[0]"):
        '''
        Get the idx of parent's children sections, i.e. compartments ids
        of sections connected to parent-argument
        
        kwargs:
        ::
            
            parent: str
                name-pattern matching a sectionname
        '''
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
        
        kwargs:
        ::
            
            parent: str
                name-pattern matching a sectionname
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
        return np.r_[self.get_idx(parent), idx]


    def get_idx_name(self, idx=np.array([0])):
        '''
        Return NEURON convention name of segments with index idx.
        The returned argument is a list of tuples with corresponding
        segment idx, section name, and position along the section, like;
        [(0, 'neuron.h.soma[0]', 0.5),]
        
        kwargs:
        ::
            
            idx : np.ndarray, dtype int
                segment indices, must be between 0 and cell.totnsegs        
        '''
        #ensure idx is array-like, or convert
        if type(idx) == int:
            idx = np.array([idx])
        elif len(idx) == 0:
            return
        else:
            idx = np.array(idx).astype(int)

        #ensure all idx are valid
        if np.any(idx >= self.totnsegs):
            wrongidx = idx[np.where(idx >= self.totnsegs)]
            raise Exception('idx %s >= number of compartments' % str(wrongidx))
        
        #create list of seg names:
        allsegnames = []
        segidx = 0
        for sec in self.allseclist:
            for seg in sec:
                allsegnames.append((segidx, '%s'  % sec.name(), seg.x))
                segidx += 1
        
        return allsegnames[idx]

    def _collect_pt3d(self):
        '''collect the pt3d info, for each section'''
        x = []
        y = []
        z = []
        d = []

        for sec in self.allseclist:
            n3d = int(neuron.h.n3d())
            x_i, y_i, z_i = np.zeros(n3d), np.zeros(n3d), np.zeros(n3d),
            d_i = np.zeros(n3d)
            for i in range(n3d):
                x_i[i] = neuron.h.x3d(i)
                y_i[i] = neuron.h.y3d(i)
                z_i[i] = neuron.h.z3d(i)
                d_i[i] = neuron.h.diam3d(i)

            
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
            d.append(d_i)
        
        #remove offsets which may be present if soma is centred in Origo
        if len(x) > 1:
            xoff = x[0].mean()
            yoff = y[0].mean()
            zoff = z[0].mean()
            for i in range(len(x)):
                x[i] -= xoff
                y[i] -= yoff
                z[i] -= zoff

        return x, y, z, d

            
    def _update_pt3d(self):           
        '''
        update the locations in neuron.hoc.space using neuron.h.pt3dchange()
        '''
        i = 0
        for sec in self.allseclist:
            n3d = int(neuron.h.n3d())
            for n in range(n3d):
                neuron.h.pt3dchange(n,
                                self.x3d[i][n],
                                self.y3d[i][n],
                                self.z3d[i][n],
                                self.diam3d[i][n])
            i += 1
            #let NEURON know about the changes we just did:
            neuron.h.define_shape()
        #must recollect the geometry, otherwise we get roundoff errors!
        self._collect_geometry()


    def _set_pt3d_pos(self):
        '''
        Offset pt3d geometry with cell.somapos
        '''
        for i in range(len(self.x3d)):
            self.x3d[i] += self.somapos[0]
            self.y3d[i] += self.somapos[1]
            self.z3d[i] += self.somapos[2]
        self._update_pt3d()


    def _set_pt3d_rotation(self, x=None, y=None, z=None):
        '''
        Rotate pt3d geometry of cell object around the x-, y-, z-axis
        in that order.
        Input should be angles in radians.
        
        using rotation matrices, takes dict with rot. angles,
        where x, y, z are the rotation angles around respective axes.
        All rotation angles are optional.
        
        Usage:
        ::
            
            cell = LFPy.Cell(**kwargs)
            rotation = {'x' : 1.233, 'y' : 0.236, 'z' : np.pi}
            cell.set_pt3d_rotation(**rotation)
        '''
        if x is not None:
            theta = -x
            rotation_x = np.matrix([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])
            for i in range(len(self.x3d)):
                rel_pos = self._rel_pt3d_positions(self.x3d[i],
                                                   self.y3d[i], self.z3d[i])
                
                rel_pos = rel_pos * rotation_x
                
                self.x3d[i], self.y3d[i], self.z3d[i] = \
                                            self._real_pt3d_positions(rel_pos)
            if self.verbose:
                print(('Rotated geometry %g radians around x-axis' % (-theta)))
        else:
            if self.verbose:
                print('Geometry not rotated around x-axis')
        
        if y is not None:
            phi = -y
            rotation_y = np.matrix([[np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]])
            for i in range(len(self.x3d)):
                rel_pos = self._rel_pt3d_positions(self.x3d[i],
                                                   self.y3d[i], self.z3d[i])
                
                rel_pos = rel_pos * rotation_y
                                
                self.x3d[i], self.y3d[i], self.z3d[i] = \
                                            self._real_pt3d_positions(rel_pos)
            if self.verbose:
                print(('Rotated geometry %g radians around y-axis' % (-phi)))
        else:
            if self.verbose:
                print('Geometry not rotated around y-axis')
        
        if z is not None:
            gamma = -z
            rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
            for i in range(len(self.x3d)):
                rel_pos = self._rel_pt3d_positions(self.x3d[i],
                                                   self.y3d[i], self.z3d[i])
                
                rel_pos = rel_pos * rotation_z
                
                self.x3d[i], self.y3d[i], self.z3d[i] = \
                                            self._real_pt3d_positions(rel_pos)
            if self.verbose:
                print(('Rotated geometry %g radians around z-axis' % (-gamma)))
        else:
            if self.verbose:
                print('Geometry not rotated around z-axis')
        
        self._update_pt3d()

    def _rel_pt3d_positions(self, x, y, z):
        '''
        Morphology relative to soma position
        '''
        rel_pos = np.transpose(np.array([x - self.somapos[0],
                                         y - self.somapos[1],
                                         z - self.somapos[2]]))

        return rel_pos
    
    def _real_pt3d_positions(self, rel_pos):
        '''
        Morphology coordinates relative to Origo
        '''
        x = rel_pos[:, 0] + self.somapos[0]
        y = rel_pos[:, 1] + self.somapos[1]
        z = rel_pos[:, 2] + self.somapos[2]
        
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        z = np.array(z).flatten()
        
        return x, y, z
    
    def _create_polygon(self, i, projection=('x', 'z')):
        '''create a polygon to fill for each section'''        
        x = getattr(self, projection[0]+'3d')[i]
        y = getattr(self, projection[1]+'3d')[i]
        #x = self.x3d[i]
        #z = self.z3d[i]
        d = self.diam3d[i]
        
        #calculate angles        
        dx = np.diff(x)
        dy = np.diff(y)
        theta = np.arctan2(dy, dx)
        
        x = np.r_[x, x[::-1]]
        y = np.r_[y, y[::-1]]
        
        theta = np.r_[theta, theta[::-1]]
        d = np.r_[d, d[::-1]]
        
        #1st corner:
        x[0] -= 0.5 * d[0] * np.sin(theta[0])
        y[0] += 0.5 * d[0] * np.cos(theta[0])
        
        ##pt3d points between start and end of section, first side
        x[1:dx.size] -= 0.25 * d[1:dx.size] * (
            np.sin(theta[:dx.size-1]) + np.sin(theta[1:dx.size]))
        y[1:dy.size] += 0.25 * d[1:dy.size] * (
            np.cos(theta[:dy.size-1]) + np.cos(theta[1:dx.size]))
        
        #end of section, first side
        x[dx.size] -= 0.5 * d[dx.size] * np.sin(theta[dx.size])
        y[dy.size] += 0.5 * d[dy.size] * np.cos(theta[dy.size])
        
        #other side
        #end of section, second side
        x[dx.size+1] += 0.5 * d[dx.size+1] * np.sin(theta[dx.size])
        y[dy.size+1] -= 0.5 * d[dy.size+1] * np.cos(theta[dy.size])
        
        ##pt3d points between start and end of section, second side
        x[::-1][1:dx.size] += 0.25 * d[::-1][1:dx.size] * (
            np.sin(theta[::-1][:dx.size-1]) + np.sin(theta[::-1][1:dx.size]))
        y[::-1][1:dy.size] -= 0.25 * d[::-1][1:dy.size] * (
            np.cos(theta[::-1][:dy.size-1]) + np.cos(theta[::-1][1:dx.size]))

        #last corner:
        x[-1] += 0.5 * d[-1] * np.sin(theta[-1])
        y[-1] -= 0.5 * d[-1] * np.cos(theta[-1])
        
        return x, y
    
    def get_pt3d_polygons(self, projection=('x', 'z')):
        '''for each section create a polygon in the plane determined by keyword
        argument projection=('x', 'z'), that can be
        visualized using e.g., plt.fill()
        
        Returned argument is a list of (x, z) tuples giving the trajectory
        of each section that can be plotted using PolyCollection
        ::
            
            from matplotlib.collections import PolyCollection
            import matplotlib.pyplot as plt
            
            cell = LFPy.Cell(morphology='PATH/TO/MORPHOLOGY')
            
            zips = []
            for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                zips.append(zip(x, z))
            
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors='gray')
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.add_collection(polycol)
            ax.axis(ax.axis('equal'))
            
            plt.show()
        '''
        if len(projection) != 2:
            raise ValueError("projection arg be a tuple like ('x', 'y')")
        if 'x' in projection and 'y' in projection:
            pass
        elif 'x' in projection and 'z' in projection:
            pass
        elif 'y' in projection and 'z' in projection:
            pass
        else:
            mssg = "projection must be a length 2 tuple of 'x', 'y' or 'z'!"
            raise ValueError(messg)

        polygons = []
        for i in range(len(self.x3d)):
            polygons.append(self._create_polygon(i, projection))
        
        return polygons


    def _create_segment_polygon(self, i, projection=('x', 'z')):
        '''create a polygon to fill for segment i, in the plane
        determined by kwarg projection'''        
        x = [getattr(self, projection[0]+'start')[i],
             getattr(self, projection[0]+'end')[i]]
        z = [getattr(self, projection[1]+'start')[i],
             getattr(self, projection[1]+'end')[i]]        
        #x = [self.xstart[i], self.xend[i]]
        #z = [self.zstart[i], self.zend[i]]
        d = self.diam[i]
        
        #calculate angles        
        dx = np.diff(x)
        dz = np.diff(z)
        theta = np.arctan2(dz, dx)
        
        x = np.r_[x, x[::-1]]
        z = np.r_[z, z[::-1]]
                
        #1st corner:
        x[0] -= 0.5 * d * np.sin(theta)
        z[0] += 0.5 * d * np.cos(theta)
                
        #end of section, first side
        x[1] -= 0.5 * d * np.sin(theta)
        z[1] += 0.5 * d * np.cos(theta)
        
        #other side
        #end of section, second side
        x[2] += 0.5 * d * np.sin(theta)
        z[2] -= 0.5 * d * np.cos(theta)
        
        #last corner:
        x[3] += 0.5 * d * np.sin(theta)
        z[3] -= 0.5 * d * np.cos(theta)
        
        return x, z


    def get_idx_polygons(self, projection=('x', 'z')):
        '''for each segment idx in celll create a polygon in the plane
        determined by the projection kwarg (default ('x', 'z')),
        that can be visualized using plt.fill() or
        mpl.collections.PolyCollection
        
        Returned argument is a list of (np.ndarray, np.ndarray) tuples
        giving the trajectory of each section
        
        kwargs:
        ::
            
            projection : ('x', 'z') tuple of two strings determining projection 
        
        The most efficient way of using this would be something like
        ::
            
            from matplotlib.collections import PolyCollection
            import matplotlib.pyplot as plt
            
            cell = LFPy.Cell(morphology='PATH/TO/MORPHOLOGY')
            
            zips = []
            for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                zips.append(zip(x, z))
            
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors='gray')
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.add_collection(polycol)
            ax.axis(ax.axis('equal'))
            
            plt.show()
        '''
        if len(projection) != 2:
            raise ValueError("projection arg be a tuple like ('x', 'y')")
        if 'x' in projection and 'y' in projection:
            pass
        elif 'x' in projection and 'z' in projection:
            pass
        elif 'y' in projection and 'z' in projection:
            pass
        else:
            mssg = "projection must be a length 2 tuple of 'x', 'y' or 'z'!"
            raise ValueError(messg)

        polygons = []
        for i in np.arange(self.totnsegs):
            polygons.append(self._create_segment_polygon(i, projection))
        
        return polygons



    def insert_v_ext(self, v_ext, t_ext):        
        '''
        playback of some extracellular potential v_ext on each cell.totnseg
        compartments. Assumes that the "extracellular"-mechanism is inserted
        on each compartment.
        
        Can be used to study ephaptic effects and similar
        
        The inputs will be copied and attached to the cell object as
        cell.v_ext, cell.t_ext, and converted
        to (list of) neuron.h.Vector types, to allow playback into each
        compartment e_extracellular reference.
        
        Can not be deleted prior to running cell.simulate() 
        
        Args:
        ::
            
            v_ext : cell.totnsegs x t_ext.size np.array, unit mV
            t_ext : np.array, time vector of v_ext
        
        Simple usage:
        ::
            
            import LFPy
            import numpy as np
            import matplotlib.pyplot as plt
            
            #create cell
            cell = LFPy.Cell(morphology='morphologies/example_morphology.hoc')
            
            #time vector and extracellular field for every segment:
            t_ext = np.arange(cell.tstopms / cell.timeres_python+ 1) * \
                    cell.timeres_python
            v_ext = np.random.rand(cell.totnsegs, t_ext.size)-0.5
        
            #insert potentials and record response:
            cell.insert_v_ext(v_ext, t_ext)
            cell.simulate(rec_imem=True, rec_vmem=True)
        
            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            eim = ax1.matshow(np.array(cell.v_ext), cmap='spectral')
            cb1 = fig.colorbar(eim, ax=ax1)
            cb1.set_label('v_ext')
            ax1.axis(ax1.axis('tight'))
            iim = ax2.matshow(cell.imem, cmap='spectral')
            cb2 = fig.colorbar(iim, ax=ax2)
            cb2.set_label('imem')
            ax2.axis(ax2.axis('tight'))
            vim = ax3.matshow(cell.vmem, cmap='spectral')
            ax3.axis(ax3.axis('tight'))
            cb3 = fig.colorbar(vim, ax=ax3)
            cb3.set_label('vmem')
            ax3.set_xlabel('tstep')
            plt.show()
        
        
        '''
        #test dimensions of input
        try:
            if v_ext.shape[0] != self.totnsegs:
                raise ValueError("v_ext.shape[0] != cell.totnsegs")
            if v_ext.shape[1] != t_ext.size:
                raise ValueError('v_ext.shape[1] != t_ext.size')
        except:
            raise ValueError('v_ext, t_ext must both be np.array types')
        
        if not self.extracellular:
            raise Exception('LFPy.Cell arg extracellular != True')
        
        #create list of extracellular potentials on each segment, time vector
        self.t_ext = neuron.h.Vector(t_ext)
        self.v_ext = []
        for v in v_ext:
            self.v_ext.append(neuron.h.Vector(v))
        #play v_ext into e_extracellular reference
        i = 0
        for sec in self.allseclist:
            for seg in sec:
                self.v_ext[i].play(seg._ref_e_extracellular, self.t_ext)
                i += 1
        
        return 
