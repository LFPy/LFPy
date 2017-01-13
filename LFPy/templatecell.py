#!/usr/bin/env python
'''
Copyright (C) 2012 Computational Neuroscience Group, NMBU.

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
from LFPy import Cell, RecExtElectrode
from LFPy.run_simulation import _run_simulation, _run_simulation_with_electrode
import sys

class TemplateCell(Cell):
    '''
    This class allow using cell templates with some limitations
    
    Arguments:
    ::
        
        morphology : [str]: path to morphology file;
        
        templatefile :  [str]: Cell template definition(s)
        templatename :  [str]: Cell template-name used for this cell object
        templateargs :  [str]: Arguments provided to template-definition
    
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
        
        delete_sections: [True]: delete pre-existing section-references
        
        custom_code: [None]: list of model-specific code files ([.py/.hoc])
        custom_fun: [None]: list of model-specific functions with args
        custom_fun_args: [None]: list of args passed to custom_fun functions
        pt3d: True/[False]: use pt3d-info of the cell geometries switch
        verbose: True/[False]: verbose output switch
    
    Usage of TemplateCell class:
    ::
        
        import LFPy
        cellParameters = {                      
            'morphology' : 'path/to/morphology',
            'templatefile' :  'path/to/template-file (.hoc)
            'templatename' :  'templatename'
            'templateargs' :  None

            'rm' : 30000,
            'cm' : 1.0,
            'Ra' : 150,
            'timeres_NEURON' : 0.1,
            'timeres_python' : 0.1,
            'tstartms' : -50,
            'tstopms' : 50,
        }
        cell = LFPy.TemplateCell(**cellParameters)
        cell.simulate()
    '''
    def __init__(self,
                 templatefile='LFPyCellTemplate.hoc',
                 templatename='LFPyCellTemplate',
                 templateargs=None,
                 **kwargs):
        '''
        Initialization of the Cell object.
        
        Arguments:
        ::
            
            templatefile :  Cell template definition(s)
            templatename :  Cell template-name used for this cell object
            templateargs :  Arguments provided to template-definition
            **kwargs :      See docstring of LFPy.Cell
        '''
        self.templatefile = templatefile
        self.templatename = templatename
        self.templateargs = templateargs
        
        if not hasattr(neuron.h, 'd_lambda'):
            neuron.h.load_file('stdlib.hoc', 'String')    #NEURON std. library
            neuron.h.load_file('import3d.hoc')  #import 3D morphology lib
                
        #load the cell template specification
        #check if templatename exist in neuron.h namespace:
        if hasattr(neuron.h, self.templatename):
            print('template %s exist already' % self.templatename)
        else:
            if type(self.templatefile) == str:
                neuron.h.load_file(self.templatefile)
            elif type(self.templatefile) == list:
                for template in self.templatefile:
                    neuron.h.load_file(template)
        
        #initialize the cell object
        Cell.__init__(self, **kwargs)
        
        
    def _load_geometry(self):
        '''Load the morphology-file in NEURON''' 
        try: 
            neuron.h.sec_counted = 0
        except LookupError:
            neuron.h('sec_counted = 0')
                
        #the python cell object we are loading the morphology into:
        celltemplate = getattr(neuron.h, self.templatename)
        self.cell = celltemplate(self.templateargs)
        #self.cell = getattr(neuron.h, self.templatename)(self.templateargs)
        
        #perform a test if the morphology is already loaded:
        seccount = 0
        for sec in self.cell.all:
            seccount += 1
        if seccount == 0:
            #import the morphology, try and determine format
            fileEnding = self.morphology.split('.')[-1]
        
            if not fileEnding == 'hoc' or fileEnding == 'HOC':            
                #create objects for importing morphologies of different formats
                if fileEnding == 'asc' or fileEnding == 'ASC':
                    Import = neuron.h.Import3d_Neurolucida3()
                    if not self.verbose:
                        Import.quiet = 1
                elif fileEnding == 'swc' or fileEnding ==  'SWC':
                    Import = neuron.h.Import3d_SWC_read()
                elif fileEnding == 'xml' or fileEnding ==  'XML':
                    Import = neuron.h.Import3d_MorphML()
                else:
                    raise ValueError('%s is not a recognised morphology file format! ').with_traceback('Should be either .hoc, .asc, .swc, .xml!' \
                         % self.morphology)
                
                #assuming now that morphology file is the correct format
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
                
                #instantiate the cell object
                if fileEnding == 'xml' or fileEnding ==  'XML':
                    #can not currently assign xml to cell template
                    try:
                        imprt.instantiate(self.cell)
                    except:
                        raise Exception("this xml file is not supported")
                else:
                    imprt.instantiate(self.cell)
                
            else:
                neuron.h.execute("xopen(\"%s\")" % self.morphology, self.cell)
                #neuron.h.load_file(1, self.morphology)
        
        #set shapes and create sectionlists
        neuron.h.define_shape()
        self._create_sectionlists()


    def _create_sectionlists(self):
        '''Create section lists for different kinds of sections'''
        #list with all sections
        
        #test if list self.cell.all is not empty
        numsec = 0
        for sec in self.cell.all:
            numsec += 1
        
        if numsec > 0:
            self.allsecnames = []
            for sec in self.cell.all:
                self.allsecnames.append(sec.name())
            
            #hotpatching the allseclist!!!
            self.allseclist = self.cell.all
            
            #list of soma sections, assuming it is named on the format "soma*"
            self.nsomasec = 0
            self.somalist = neuron.h.SectionList()
            for sec in self.cell.all:
                if sec.name().find('soma') >= 0:
                    self.somalist.append(sec=sec)
                    self.nsomasec += 1
        else:
            self.allsecnames = []
            for sec in self.cell.allsec():
                self.allsecnames.append(sec.name())
            
            self.allseclist = neuron.h.SectionList()
            for sec in self.cell.allsec():
                self.allseclist.append(sec=sec)
            
            
            #list of soma sections, assuming it is named on the format "soma*"
            self.nsomasec = 0
            self.somalist = neuron.h.SectionList()
            for sec in self.cell.allsec():
                if sec.name().find('soma') >= 0:
                    self.somalist.append(sec=sec)
                    self.nsomasec += 1
 

    def _update_pt3d(self):           
        '''
        update the locations in neuron.hoc.space using neuron.h.pt3dchange()
        '''
        for i, sec in enumerate(self.allseclist):
            n3d = int(neuron.h.n3d())
            for n in range(n3d):
                neuron.h.pt3dchange(n,
                                self.x3d[i][n],
                                self.y3d[i][n],
                                self.z3d[i][n],
                                self.diam3d[i][n])
            #let NEURON know about the changes we just did:
            neuron.h.define_shape()
        #must recollect the geometry, otherwise we get roundoff errors!
        self._collect_geometry()
