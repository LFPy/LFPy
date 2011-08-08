#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''

import os
import neuron
import numpy as np
import copy
import cPickle

#LFPy-specific mechanisms
installpath = os.getenv('LFPYPATH')
morpho_path = os.path.join(installpath, 'morphologies')     

_known_architectures = ['i386', 'i686', 'x86_64', 'umac', 'ppc']     
architecture = None
for arch in _known_architectures:
    if os.path.isdir(os.path.join(installpath, 'neuron', arch)): 
        architecture = arch
        break
if architecture is None:     
    errmsg = '\n'.join(['LFPy cannot find compiled neuron mechanisms!',
    'Try running nrnivmodl in %s' %os.path.join(installpath, 'neuron'),
    'LFPy knows about the following architectures:',
    str(_known_architectures)])
    raise Exception, errmsg

#loading the default LFPy mechanisms
dll_filename = os.path.join(installpath,
    'neuron', architecture, '.libs', 'libnrnmech.so')
neuron.h.nrn_load_dll(dll_filename)

class Cell(object):
    ''' The main cell class used in LFPy
    
    Inputs:
    ::
        morphology : morphology file;
        default_dir : if True will look for morphology in the default folder,
        otherwise full path to morphology must be provided;
        
        v_init : initial potential;
        passive : passive mechs are initialized if True;
        Ra : axial resistance;
        rm : memnbrane resistivity;
        cm : membrane capacitance;
        e_pas : passive mechanism reversal potential;
        
        timeres_NEURON : internal dt for NEURON simulation;
        timeres_python : overall dt for python simulation;
        
        tstartms : initialization time for simulation <= 0 ms
        tstopms : stop time for simulation > 0 ms
        
        nsegs_method : method for setting the number of segments;
        max_nsegs_length : max segment length for method 'fixed_length';
        lambda_f : AC frequency for method 'lambda_f';
        
        custom_code : list of model specific code files goes here ([*.py/.hoc]);
        verbose : switching verbose output on/off
        play_in_soma : if True, play somatrace in soma
        soma_trace :  filename somatrace, two columns, space-sep: t & v
    '''
    def __init__(self, morphology, 
                    default_dir=True, 
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
                    custom_code=[],
                    custom_fun = [],
                    custom_fun_args = [],
                    play_in_soma=False, 
                    soma_trace='', verbose=False):
        '''initialization of the Cell object.
        
        Inputs:
        ::
            morphology : morphology file;
            default_dir : if True will look for morphology in the default folder,
            otherwise full path to morphology must be provided;
            
            v_init : initial potential;
            passive : passive mechs are initialized if True;
            Ra : axial resistance;
            rm : memnbrane resistivity;
            cm : membrane capacitance;
            e_pas : passive mechanism reversal potential;
            
            timeres_NEURON : internal dt for NEURON simulation;
            timeres_python : overall dt for python simulation;
            
            tstartms : initialization time for simulation <= 0 ms
            tstopms : stop time for simulation > 0 ms
            
            nsegs_method : method for setting the number of segments;
            max_nsegs_length : max segment length for method 'fixed_length';
            lambda_f : AC frequency for method 'lambda_f';
            
            custom_code : list of model specific code files goes here ([*.py/.hoc]);
            verbose : switching verbose output on/off
            play_in_soma : if True, play somatrace in soma
            soma_trace : filename somatrace, two columns, space-sep: t & v
        '''

        #Loading NEURON standard library and custom LFPy codes
        neuron.h.load_file('stdlib.hoc')
        neuron.h.xopen(os.path.join(installpath, 'neuron', 'LFPy.hoc'))

        if default_dir:
            morpho_file = os.path.join(morpho_path, morphology)
        else:
            morpho_file = morphology

        if os.path.isfile(morpho_file):
            self.morphology_file = morpho_file
            neuron.h.load_geometry(morpho_file)
        else:
            raise Exception, "%s does not exist!" % morpho_file

        if os.path.isfile(morpho_file[0:-4]+'.rot'):
            rotation_file = morpho_file[0:-4]+'.rot'
            rotation_data = open(rotation_file)
            rotation={}
            for line in rotation_data:
                var, val=line.split('=')
                val=val.strip()
                val=float(str(val))
                rotation[var]=val
        else:
            rotation = {
                'x' : 0,
                'y' : 0,
            }
        
        self.timeres_python = timeres_python
        self.timeres_NEURON = timeres_NEURON
        
        self.tstartms = tstartms
        self.tstopms = tstopms
        
        self.synapses = []
        self.synidx = []
        self.pointprocesses = []
        self.pointprocess_idx = []

        self.v_init = v_init
        
        self.default_rotation = rotation
        
        if passive:
            #Set passive properties, insert passive on all compartments
            self.Ra = Ra
            self.rm = rm
            self.cm = cm
            self.e_pas = e_pas
            self.set_passive()
                
        # load custom codes
        for code in custom_code:
            if code.split('.')[-1] == 'hoc':
                neuron.h.xopen(code)
                #neuron.h.xopen(os.path.join(installpath, 'examples', codefile))
            elif code.split('.')[-1] == 'py':
                exec(code)
            else:
                raise Exception, '%s not a .hoc- nor .py-file' % codefile
        
        i = 0
        for fun in custom_fun:
            fun(**custom_fun_args[i])
            i += 1
        
        # Make NEURON calculate i_membrane
        self.set_extracellular()

        if nsegs_method == 'lambda100':
            self.set_nsegs_lambda100()
            print "Settings nsegs using lambda100"
        elif nsegs_method == 'lambda_f':
            self.set_nsegs_lambda_f(lambda_f)
        elif nsegs_method == 'fixed_length':
            self.set_nsegs_fixed_length(max_nsegs_length)
        else:
            print 'No nsegs_method applied (%s)' % nsegs_method

        self.calc_totnsegs()
        print "Total number of segments = ", self.totnsegs

        self.collect_geometry()
        self.set_pos()

        self.rotate_xyz(self.default_rotation)
        
        # Optional part for cases with play in soma
        # MOVE!!!!!!!!
        self.play_in_soma = play_in_soma
        if play_in_soma:
            self.soma_trace=soma_trace

    def set_nsegs_lambda_f(self, frequency):
        '''set the number of segments for section according to the d_lambda-rule
        for a given input frequency'''
        for sec in neuron.h.allseclist:
            sec.nseg = int((sec.L / (0.1 * neuron.h.lambda_f(frequency)) + .9)
                / 2 )*2 +1
        #if we want to have back nseg=1 for somalist, uncomment the following
        #for sec in neuron.h.somalist:
        #    sec.nseg = 1
    
    def set_nsegs_lambda100(self):
        '''set the numbers of segments using d_lambda(100)'''
        self.set_nsegs_lambda_f(100)

    def set_nsegs_fixed_length(self, maxlength):
        '''set nseg for sections so that not any compartment L >= maxlength'''
        for sec in neuron.h.allseclist:
            sec.nseg = int(sec.L / maxlength) + 1
        #if we want to have back nseg=1 for somalist, uncomment the following
        #for sec in neuron.h.somalist:
        #    sec.nseg = 1
    
    def calc_totnsegs(self):
        i = 0
        for sec in neuron.h.allseclist:
            i += sec.nseg
        
        self.totnsegs = i
    
    def check_currents(self):
        '''Check that the sum of all membrane and electrode currents over all
        compartments is sufficiently close to zero'''
        raise NotImplementedError, 'this function need to be written'

    def set_passive(self):
        '''insert passive mechanism on all compartments'''
        for sec in neuron.h.allseclist:
            sec.insert('pas')
            sec.Ra = self.Ra
            sec.cm = self.cm
            sec.g_pas = 1. / self.rm
            sec.e_pas = self.e_pas
    
    def set_extracellular(self):
        '''insert extracellular mechanism on all sections
        to access i_membrane'''
        for sec in neuron.h.allseclist:
            sec.insert('extracellular')
    
    def set_synapse(self, idx, syntype,
                    record_current=False, record_potential=False,
                    tau=None, tau1=None, tau2=None, e=None, weight=None,
                    g=None, pkamp=None, freq=None, delay=None, dur=None):
        '''Insert syntype synapse on compartment with index idx, **kwargs
        passed on from class Synapse.'''
        
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
        for sec in neuron.h.allseclist:
            for seg in sec:
                if i == idx:
                    if syntype == 'ExpSyn':
                        syn = neuron.h.ExpSyn(seg.x, sec=sec)
                        syn.tau = tau
                        syn.e = e
                        self.synlist.append(syn)
                    elif syntype == 'Exp2Syn':
                        syn = neuron.h.Exp2Syn(seg.x, sec=sec)
                        syn.tau1 = tau1
                        syn.tau2 = tau2
                        syn.e = e
                        self.synlist.append(syn)
                    elif syntype == 'AlphaSyn':
                        syn = neuron.h.AlphaSyn(seg.x, sec=sec)
                        syn.tau = tau
                        syn.g = g
                        syn.e = e
                        self.synlist.append(syn)
                    elif syntype == 'AlphaISyn':
                        syn = neuron.h.AlphaISyn(seg.x, sec=sec)
                        syn.tau = tau
                        self.synlist.append(syn)
                    elif syntype == 'SinSyn':
                        syn = neuron.h.SinSyn(seg.x, sec=sec)
                        syn.pkamp = pkamp
                        syn.freq = freq
                        syn.phase = phase
                        syn.delay = delay
                        syn.dur = dur
                        self.synlist.append(syn)
                    else:
                        raise Exception, '%s not valid synapse type' % syntype
                    
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

    def set_point_process(self, idx, pptype, record_current=False,
                          amp=None, dur=None, delay=None,
                          amp1=None, dur1=None,
                          amp2=None, dur2=None,
                          amp3=None, dur3=None,
                          rs=None,
                          pkamp=None, freq=None, phase=None,
                          chirp_rate=None):
        '''Insert pptype-electrode type pointprocess on compartment numbered
        idx on cell object, with keyword arguments according to types:
        SEClamp, VClamp, IClamp, SinIClamp, ChirpIClamp.
        idx, pptype, **kwargs is passed on from PointProcess class.'''
        
        if not hasattr(self, 'stimlist'):
            self.stimlist = neuron.h.List()
        if not hasattr(self, 'stimireclist'):
            self.stimireclist = neuron.h.List()
        
        i = 0
        for sec in neuron.h.allseclist:
            for seg in sec:
                if i == idx:
                    if pptype == 'SEClamp':
                        stim = neuron.h.SEClamp(seg.x, sec=sec)
                        stim.amp1 = amp1
                        stim.dur1 = dur1
                        stim.amp2 = amp2
                        stim.dur2 = dur2
                        stim.amp3 = amp3
                        stim.dur3 = dur3
                        stim.rs = rs
                        self.stimlist.append(stim)
                    elif pptype == 'VClamp':
                        stim = neuron.h.VClamp(seg.x, sec=sec)
                        stim.dur[0] = dur[0]
                        stim.dur[1] = dur[1]
                        stim.dur[2] = dur[2]
                        stim.amp[0] = amp[0]
                        stim.amp[1] = amp[1]
                        stim.amp[2] = amp[2]
                        self.stimlist.append(stim)
                    elif pptype == 'IClamp':
                        stim = neuron.h.IClamp(seg.x, sec=sec)
                        stim.amp = amp
                        stim.dur = dur
                        stim.delay = delay
                        self.stimlist.append(stim)
                    elif pptype == 'SinIClamp':
                        stim = neuron.h.SinIClamp(seg.x, sec=sec)
                        stim.pkamp = pkamp
                        stim.dur = dur
                        stim.delay = delay
                        stim.freq = freq
                        stim.phase = phase
                        self.stimlist.append(stim)
                    elif pptype == 'ChirpIClamp':
                        stim = neuron.h.ChirpIClamp(seg.x, sec=sec)
                        stim.pkamp = pkamp
                        stim.dur = dur
                        stim.delay = delay
                        stim.freq = freq
                        stim.chirp_rate = chirp_rate
                        self.stimlist.append(stim)
                    else:
                        raise Exception, '%s not valid pointprocess' % pptype
                    
                    #record current
                    if record_current:
                        stimirec = neuron.h.Vector(int(self.tstopms /
                                                       self.timeres_python+1))
                        stimirec.record(stim._ref_i, self.timeres_python)
                        self.stimireclist.append(stimirec)
                    
                i += 1
        
        return self.stimlist.count() - 1
    
    def collect_geometry(self):
        '''Collects x, y, z-coordinates from NEURON'''
        self.collect_geometry_neuron()
        
        self.calc_midpoints()

        self.somaidx = self.get_idx(section='som')

        if self.somaidx.size == 0:
            pass
            #print 'There is no soma!'
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
    
    def collect_geometry_neuron(self):
        '''looping over allseclist to determine area, diam, xyz-start- and
        endpoints, embed geometry to cell object.'''
        
        areavec = neuron.h.Vector(self.totnsegs)
        diamvec = neuron.h.Vector(self.totnsegs)
        
        xstartvec = neuron.h.Vector(self.totnsegs)
        xendvec = neuron.h.Vector(self.totnsegs)
        ystartvec = neuron.h.Vector(self.totnsegs)
        yendvec = neuron.h.Vector(self.totnsegs)
        zstartvec = neuron.h.Vector(self.totnsegs)
        zendvec = neuron.h.Vector(self.totnsegs)
        
        counter = 0
        
        #loop over all compartments
        for sec in neuron.h.allseclist:
            if neuron.h.n3d() > 0:
                #length of sections
                xlength = neuron.h.x3d(neuron.h.n3d() - 1) - neuron.h.x3d(0)
                ylength = neuron.h.y3d(neuron.h.n3d() - 1) - neuron.h.y3d(0)
                zlength = neuron.h.z3d(neuron.h.n3d() - 1) - neuron.h.z3d(0)
                
                for seg in sec:
                    areavec.x[counter] = neuron.h.area(seg.x)
                    diamvec.x[counter] = seg.diam
                    
                    xstartvec.x[counter] = neuron.h.x3d(0) + xlength * (seg.x - 1./2./sec.nseg)
                    xendvec.x[counter] = neuron.h.x3d(0) + xlength * (seg.x + 1./2./sec.nseg)
    
                    ystartvec.x[counter] = neuron.h.y3d(0) + ylength * (seg.x - 1./2./sec.nseg)
                    yendvec.x[counter] = neuron.h.y3d(0) + ylength * (seg.x + 1./2./sec.nseg)

                    zstartvec.x[counter] = neuron.h.z3d(0) + zlength * (seg.x - 1./2./sec.nseg)
                    zendvec.x[counter] = neuron.h.z3d(0) + zlength * (seg.x + 1./2./sec.nseg)
                    
                    counter += 1
        
        self.xstart = np.array(xstartvec)
        self.ystart = np.array(ystartvec)
        self.zstart = np.array(zstartvec)

        self.xend = np.array(xendvec)
        self.yend = np.array(yendvec)
        self.zend = np.array(zendvec)

        self.area = np.array(areavec)
        self.diam = np.array(diamvec)

    def calc_midpoints(self):
        '''calculate midpoints of each compartment'''
        self.xmid = .5*(self.xstart+self.xend)
        self.ymid = .5*(self.ystart+self.yend)
        self.zmid = .5*(self.zstart+self.zend)

    def get_idx(self, section='allsec', z_min=-10000, z_max=10000):
        '''Returns neuron idx of segments'''
        if section=='allsec':
            neuron.h.get_idx(neuron.h.allseclist, self.totnsegs)
        elif section=='som' or section=='soma':
            neuron.h.get_idx(neuron.h.somalist, self.totnsegs)
        elif section=='dend':
            neuron.h.get_idx(neuron.h.dendlist, self.totnsegs)
        elif section=='apicdend':
            neuron.h.get_idx(neuron.h.apicdendlist, self.totnsegs)
        elif section=='alldend':
            neuron.h.get_idx(neuron.h.alldendlist, self.totnsegs)
        elif section=='ax' or section=='axon':
            neuron.h.get_idx(neuron.h.axonlist, self.totnsegs)
        #check if these is needed;
        elif section=='dendritic':
            neuron.h.get_idx(neuron.h.dendritic, self.totnsegs)
        elif section=='dendritic_no_prim_dend':
            neuron.h.get_idx(neuron.h.dendritic_no_prim_dend, self.totnsegs)
        else:
            raise Exception, "%s is not a valid section list" % section
            
        idx = np.where(np.array(neuron.h.idxvec))[0]

        sel_z_idx = np.where(np.logical_and(self.zmid[idx] > z_min,
                                                self.zmid[idx]<z_max))
        return idx[sel_z_idx]

    #move to tools.py
    def get_closest_idx(self, x=0, y=0, z=0, section='allsec'):
        idx = self.get_idx(section)
        dist = np.sqrt((self.xmid[idx] - x)**2 + \
            (self.ymid[idx] - y)**2 + (self.zmid[idx] - z)**2)

        mindist = np.where(dist == np.min(dist))

        return int(idx[mindist])


    def get_rand_idx_area_norm(self, section='allsec', nidx=1,
                               z_min=-10000, z_max=10000):
        poss_idx = self.get_idx(section=section, z_min=z_min, z_max = z_max)
        idx = np.empty(nidx, dtype=int)

        for i in xrange(nidx):
            idx[i] = np.min(np.nonzero(np.random.rand() < \
                np.cumsum(self.area[poss_idx]/sum(self.area[poss_idx])))[0])

        return poss_idx[idx]

    def simulate(self, rec_i=True, rec_v=False, rec_ipas=False, rec_icap=False, 
                 rec_isyn=False, rec_vsyn=False, rec_istim=False,
                 tstartms=None, tstopms=None):
        '''Start NEURON simulation and record variables.'''
        if tstartms != None:
            self.tstartms = tstartms
        if tstopms != None:
            self.tstopms = tstopms
        
        self.set_soma_volt_recorder()
        self.set_time_recorder()
        
        if rec_i:
            self.set_imem_recorders()        
        if rec_v:
            self.set_voltage_recorders()
        if rec_ipas:
            self.set_ipas_recorders()
        if rec_icap:
            self.set_icap_recorders()

        #run fadvance until t >= tstopms
        self.run_simulation()

        #fixing tvec
        if self.tstartms != None:
            self.tvec = np.array(self.tvec) + self.timeres_NEURON
        else:
            self.tvec = np.array(self.tvec)
            self.tvec[1:] = self.tvec[1:] + self.timeres_NEURON

        self.somav = np.array(self.somav)
        
        if rec_i:
            self.calc_imem()
                
        if rec_ipas:
            self.calc_ipas()
            
        if rec_icap:
            self.calc_icap()
        
        if rec_v:
            self.collect_vmem()
        
        if rec_isyn:
            self.collect_isyn()
        
        if rec_vsyn:
            self.collect_vsyn()
            
        if rec_istim:
            self.collect_istim()
            

    def calc_imem(self):
        '''fetching the vectors from the memireclist and calculate self.imem
        containing all the membrane currents.'''
        self.imem = np.array(self.memireclist)
        for i in xrange(self.imem.shape[0]):
            self.imem[i, ] *= self.area[i] * 1E-2
        self.memireclist = None
        del self.memireclist
    
    def calc_ipas(self):
        self.ipas = np.array(self.memipasreclist)
        for i in xrange(self.ipas.shape[0]):
            self.ipas[i, ] *= self.area[i] * 1E-2
        self.memipasreclist = None
        del self.memipasreclist
    
    def calc_icap(self):
        self.icap = np.array(self.memicapreclist)
        for i in xrange(self.icap.shape[0]):
            self.icap[i, ] *= self.area[i] * 1E-2
        self.memicapreclist = None
        del self.memicapreclist
    
    def collect_vmem(self):
        self.vmem = np.array(self.memvreclist)
        self.memvreclist = None
        del self.memvreclist

    def collect_isyn(self):
        for i in xrange(len(self.synapses)):
            if self.synapses[i].record_current:
                self.synapses[i].collect_current(self)
            else:
                raise Exception, 'must set record_current=True for synapse'
        self.synireclist = None
        del self.synireclist
    
    def collect_vsyn(self):
        for i in xrange(len(self.synapses)):
            self.synapses[i].collect_potential(self)
        self.synvreclist = None
        del self.synvreclist
        
    def collect_istim(self):
        for i in xrange(len(self.pointprocesses)):
            if self.pointprocesses[i].record_current:
                self.pointprocesses[i].collect_current(self)
            else:
                raise Exception, 'must set record_current=True for pointp.'
        self.stimireclist = None
        del self.stimireclist
    
    def run_simulation(self):
        '''Running the actual simulation in NEURON, simulations in NEURON
        is now interruptable.'''
        neuron.h.dt = self.timeres_NEURON
        
        cvode = neuron.h.CVode()
        
        if neuron.h.dt <= 1E-8:
            cvode.active(1)
            cvode.atol(0.001)
        else:
            cvode.active(0)
        
        
        #initialize state
        neuron.h.finitialize(self.v_init)
            
        #initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()
        
        
        #Starting simulation at t != 0
        if self.tstartms != None:
            neuron.h.t = self.tstartms

        self.loadspikes()

        #print sim.time at intervals
        counter = 0.
        if self.tstopms > 1000:
            interval = 1 / self.timeres_NEURON * 100
        else:
            interval = 1 / self.timeres_NEURON * 10
        
        while neuron.h.t < self.tstopms:
            neuron.h.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                print 't = %.0f' % neuron.h.t
        
        
    def loadspikes(self):
        '''initialize spiketimes from netcon if they exist'''
        if hasattr(self, 'synlist'):
            for i in xrange(int(self.synlist.count())):
                for ii in xrange(int(self.sptimeslist.o(i).size)):
                    self.netconlist.o(i).event(self.sptimeslist.o(i)[ii])
    
    def set_time_recorder(self):
        '''Initialize time-vector recorder in NEURON'''
        self.tvec = neuron.h.Vector(int(self.tstopms/self.timeres_python+1))
        self.tvec.record(neuron.h._ref_t, self.timeres_python)
    
    def set_soma_volt_recorder(self):
        '''record somatic crossmembrane potential'''
        self.somav = neuron.h.Vector(int(self.tstopms/self.timeres_python+1))
        if neuron.h.nsomasec == 0:
            pass
        elif neuron.h.nsomasec == 1:
            self.somav.record(neuron.h.soma[0](0.5)._ref_v, self.timeres_python)
        elif neuron.h.nsomasec > 1:
            i, j = divmod(neuron.h.nsomasec, 2)
            self.somav.record(neuron.h.soma[int(i)](j/2)._ref_v, self.timeres_python)    
    
    def set_imem_recorders(self):
        '''record membrane currents for all compartments'''
        self.memireclist = neuron.h.List()
        for sec in neuron.h.allseclist:
            for seg in sec:
                memirec = neuron.h.Vector(int(self.tstopms/self.timeres_python+1))
                memirec.record(seg._ref_i_membrane, self.timeres_python)
                self.memireclist.append(memirec)

    def set_ipas_recorders(self):
        '''record passive (ohmic) membrane currents for all compartments'''
        self.memipasreclist = neuron.h.List()
        for sec in neuron.h.allseclist:
            for seg in sec:
                memipasrec = neuron.h.Vector(int(self.tstopms/self.timeres_python+1))
                memipasrec.record(seg._ref_i_pas, self.timeres_python)
                self.memipasreclist.append(memipasrec)

    def set_icap_recorders(self):
        '''record passive (ohmic) membrane currents for all compartments'''
        self.memicapreclist = neuron.h.List()
        for sec in neuron.h.allseclist:
            for seg in sec:
                memicaprec = neuron.h.Vector(int(self.tstopms/self.timeres_python+1))
                memicaprec.record(seg._ref_i_cap, self.timeres_python)
                self.memicapreclist.append(memicaprec)
                
    def set_voltage_recorders(self):
        '''record membrane potentials for all compartments'''
        self.memvreclist = neuron.h.List()
        for sec in neuron.h.allseclist:
            for seg in sec:
                memvrec = neuron.h.Vector(int(self.tstopms/self.timeres_python+1))
                memvrec.record(seg._ref_v, self.timeres_python)
                self.memvreclist.append(memvrec)

    #MOVE!
    def calc_ireturn(self):
        self.ireturn = copy.copy(self.imem)

        for i in xrange(len(self.synapses)):
            idx = self.synapses[i].idx
            self.ireturn[idx] -= self.synapses[i].i
            #print 'Calc ireturn, idx', str(idx)

    def set_pos(self, xpos=0, ypos=0, zpos=0):
        self.orig_pos = False

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

        self.calc_midpoints()
        self.update_synapse_positions()

    def cellpickler(self, filename):
        '''Save data in cell to file, using cPickle.'''
        filen = open(filename, 'wb')
        cPickle.dump(self, filen, protocol=2)
        filen.close()

    def update_synapse_positions(self):
        '''update synapse positions after rotation of morphology'''
        for i in xrange(len(self.synapses)):
            self.synapses[i].update_pos(self)

    def rotate_xyz(self, rotation):
        '''Rotate geometry using rotation matrices, takes dict with rot. angles, 
        where {'x' : ... } etc. are the rotation angles around respective axes.
        All rotation angles are optional.

        Usage:
        rotation = {'x' : 1.233, 'y : 0.236, 'z' : np.pi}
        c.rotate_xyz(rotation)
        '''
        if np.isscalar(rotation.get('x'))==True:
            rot_x = rotation.get('x')
            theta = -rot_x
            rotation_x = np.matrix([[1, 0, 0], 
                [0, np.cos(theta), -np.sin(theta)], 
                [0, np.sin(theta), np.cos(theta)]])

            rel_start, rel_end = self.rel_positions()

            rel_start = rel_start * rotation_x
            rel_end = rel_end * rotation_x

            self.real_positions(rel_start, rel_end)
            #print 'Rotated geometry %g radians around x-axis' % (-theta)
        else:
            pass
            #print 'Geometry not rotated around x-axis'

        if np.isscalar(rotation.get('y'))==True:
            rot_y = rotation.get('y')
            phi = -rot_y
            rotation_y = np.matrix([[np.cos(phi), 0, np.sin(phi)], 
                [0, 1, 0], 
                [-np.sin(phi), 0, np.cos(phi)]])

            rel_start, rel_end = self.rel_positions()

            rel_start = rel_start * rotation_y
            rel_end = rel_end * rotation_y

            self.real_positions(rel_start, rel_end)

            #print 'Rotated geometry %g radians around y-axis' % (-phi)
        else:
            pass
            #print 'Geometry not rotated around y-axis'

        if np.isscalar(rotation.get('z'))==True:
            rot_z = rotation.get('z')
            gamma = -rot_z
            rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], 
                    [np.sin(gamma), np.cos(gamma), 0], 
                    [0, 0, 1]])

            rel_start, rel_end = self.rel_positions()

            rel_start = rel_start * rotation_z
            rel_end = rel_end * rotation_z

            self.real_positions(rel_start, rel_end)

            #print 'Rotated geometry %g radians around z-axis' % (-gamma)
        else:
            pass
            #print 'Geometry not rotated around z-axis'

    def squeeze_me_macaroni(self):
        '''Reducing the dimensions of the morphology matrices'''
        self.xstart = np.squeeze(np.array(self.xstart))
        self.xend = np.squeeze(np.array(self.xend))

        self.ystart = np.squeeze(np.array(self.ystart))
        self.yend = np.squeeze(np.array(self.yend))

        self.zstart = np.squeeze(np.array(self.zstart))
        self.zend = np.squeeze(np.array(self.zend))

    def rel_positions(self):
        '''morphology relative to soma position'''
        rel_start = np.transpose(np.array([self.xstart-self.somapos[0], \
                                                self.ystart-self.somapos[1], \
                                                self.zstart-self.somapos[2]]))
        rel_end = np.transpose(np.array([self.xend-self.somapos[0], \
                                                self.yend-self.somapos[1], \
                                                self.zend-self.somapos[2]]))
        return rel_start, rel_end

    def real_positions(self, rel_start, rel_end):
        '''Morphology coordinates relative to Origo'''
        self.xstart = rel_start[:, 0] + self.somapos[0]
        self.ystart = rel_start[:, 1] + self.somapos[1]
        self.zstart = rel_start[:, 2] + self.somapos[2]

        self.xend = rel_end[:, 0] + self.somapos[0]
        self.yend = rel_end[:, 1] + self.somapos[1]
        self.zend = rel_end[:, 2] + self.somapos[2]

        self.squeeze_me_macaroni()
        self.calc_midpoints()
        self.update_synapse_positions()

    def get_rand_prob_area_norm(self, section='allsec', z_min=-10000, z_max=10000):
        '''Return the probability (0-1) for synaptic coupling on compartments in section
        sum(prob)=1 over all compartments in section. Prob. determined by area.'''
        idx = self.get_idx(section=section, z_min=z_min, z_max = z_max)
        prob = self.area[idx]/sum(self.area[idx])
        return prob

    #MOVE
    def set_play_in_soma(self, t_on=np.array([0])):
        '''Set mechanisms for playing soma trace at time(s) t_on. t_on is a np.array'''
        if type(t_on) != np.ndarray:
            t_on = np.array(t_on)
        
        f = file(self.soma_trace)
        x = []
        for line in f.readlines():
            x.append(map(float, line.split()))
        x = np.array(x)
        X = x.T
        f.close()
        
        #time and values for trace, shifting
        tTrace = X[0, ]
        tTrace -= tTrace[0]
        
        trace = X[1, ]
        trace -= trace[0]
        trace += self.e_pas
        
        #creating trace
        somaTvec0 = tTrace
        somaTvec0 += t_on[0]
        somaTvec = somaTvec0
        somaTrace = trace
        
        for i in xrange(1, t_on.size):
            np.concatenate((somaTvec, somaTvec0 + t_on[i]))
            np.concatenate((somaTrace, trace))
        
        somaTvec1 = np.interp(np.arange(somaTvec[0], somaTvec[-1], self.timeres_NEURON), somaTvec, somaTvec)
        somaTrace1 = np.interp(np.arange(somaTvec[0], somaTvec[-1], self.timeres_NEURON),
                              somaTvec, somaTrace)
        
        somaTvecVec = neuron.h.Vector(somaTvec1)
        somaTraceVec = neuron.h.Vector(somaTrace1)
        
        for sec in neuron.h.somalist:
            #ensure that soma is perfect capacitor
            sec.cm = 1E9
            #Why the fuck doesnt this work:
            #for seg in sec:
            #    somaTraceVec.play(seg._ref_v, somaTvecVec)
        
        #call hoc function that insert trace on soma
        neuron.h.play_in_soma(somaTvecVec, somaTraceVec)
        
            