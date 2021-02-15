#!/usr/bin/env python
"""
Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import os
import posixpath
import sys
import neuron
from LFPy import Cell


class TemplateCell(Cell):

    """
    ``LFPy.Cell`` like class allowing use of NEURON templates with some
    limitations.

    This takes all the same parameters as the Cell class, but requires three
    more template related parameters ``templatefile``, ``templatename`` and
    ``templateargs``

    Parameters
    ----------
    morphology : str
        path to morphology file
    templatefile : str
        File with cell template definition(s)
    templatename : str
        Cell template-name used for this cell object
    templateargs : str
        Parameters provided to template-definition
    v_init : float
        Initial membrane potential. Default to -65.
    Ra : float
        axial resistance. Defaults to 150.
    cm : float
        membrane capacitance. Defaults to 1.0
    passive : bool
        Passive mechanisms are initialized if True. Defaults to True
    passive_parameters : dict
        parameter dictionary with values for the passive membrane mechanism in
        NEURON ('pas'). The dictionary must contain keys 'g_pas' and 'e_pas',
        like the default: passive_parameters=dict(g_pas=0.001, e_pas=-70)
    extracellular : bool
        switch for NEURON's extracellular mechanism. Defaults to False
    dt: float
        Simulation time step. Defaults to 2**-4
    tstart : float
        initialization time for simulation <= 0 ms. Defaults to 0.
    tstop : float
        stop time for simulation > 0 ms. Defaults to 100.
    nsegs_method : 'lambda100' or 'lambda_f' or 'fixed_length' or None
        nseg rule, used by NEURON to determine number of compartments.
        Defaults to 'lambda100'
    max_nsegs_length : float or None
        max segment length for method 'fixed_length'. Defaults to None
    lambda_f : int
        AC frequency for method 'lambda_f'. Defaults to 100
    d_lambda : float
        parameter for d_lambda rule. Defaults to 0.1
    delete_sections : bool
        delete pre-existing section-references. Defaults to True
    custom_code : list or None
        list of model-specific code files ([.py/.hoc]). Defaults to None
    custom_fun : list or None
        list of model-specific functions with args. Defaults to None
    custom_fun_args : list or None
        list of args passed to custom_fun functions. Defaults to None
    pt3d : bool
        use pt3d-info of the cell geometries switch. Defaults to False
    celsius : float or None
        Temperature in celsius. If nothing is specified here
        or in custom code it is 6.3 celcius
    verbose : bool
        verbose output switch. Defaults to False

    Examples
    --------

    >>> import LFPy
    >>> cellParameters = {
    >>>     'morphology' : '<path to morphology.hoc>',
    >>>     'templatefile' :  '<path to template_file.hoc>'
    >>>     'templatename' :  'templatename'
    >>>     'templateargs' :  None
    >>>     'v_init' : -65,
    >>>     'cm' : 1.0,
    >>>     'Ra' : 150,
    >>>     'passive' : True,
    >>>     'passive_parameters' : {'g_pas' : 0.001, 'e_pas' : -65.},
    >>>     'dt' : 2**-3,
    >>>     'tstart' : 0,
    >>>     'tstop' : 50,
    >>> }
    >>> cell = LFPy.TemplateCell(**cellParameters)
    >>> cell.simulate()

    See also
    --------
    Cell
    NetworkCell
    """

    def __init__(self,
                 templatefile='LFPyCellTemplate.hoc',
                 templatename='LFPyCellTemplate',
                 templateargs=None,
                 verbose=False,
                 **kwargs):
        if "win32" in sys.platform and isinstance(templatefile, str):
            templatefile = templatefile.replace(os.sep, posixpath.sep)
        self.templatefile = templatefile
        self.templatename = templatename
        self.templateargs = templateargs
        self.verbose = verbose

        if not hasattr(neuron.h, 'd_lambda'):
            neuron.h.load_file('stdlib.hoc', 'String')  # NEURON std. library
            neuron.h.load_file('import3d.hoc')  # import 3D morphology lib

        # load the cell template specification
        # check if templatename exist in neuron.h namespace:
        if hasattr(neuron.h, self.templatename):
            if self.verbose:
                print('template %s exist already' % self.templatename)
        else:
            if isinstance(self.templatefile, str):
                neuron.h.load_file(self.templatefile)
            elif isinstance(self.templatefile, list):
                for template in self.templatefile:
                    if "win32" in sys.platform:
                        template = template.replace(os.sep, posixpath.sep)
                    neuron.h.load_file(template)

        # initialize the parent Cell object
        super().__init__(**kwargs)

    def _load_geometry(self):
        """Load the morphology-file in NEURON"""
        # the python cell object we are loading the morphology into:
        self.template = getattr(neuron.h, self.templatename)(self.templateargs)

        # perform a test if the morphology is already loaded:
        seccount = 0
        for sec in self.template.all:
            seccount += 1
        if seccount == 0:
            # import the morphology, try and determine format
            fileEnding = self.morphology.split('.')[-1]

            if not fileEnding == 'hoc' or fileEnding == 'HOC':
                # create objects for importing morphologies of different
                # formats
                if fileEnding == 'asc' or fileEnding == 'ASC':
                    Import = neuron.h.Import3d_Neurolucida3()
                    if not self.verbose:
                        Import.quiet = 1
                elif fileEnding == 'swc' or fileEnding == 'SWC':
                    Import = neuron.h.Import3d_SWC_read()
                elif fileEnding == 'xml' or fileEnding == 'XML':
                    Import = neuron.h.Import3d_MorphML()
                else:
                    raise ValueError(
                        '%s not a recognised morphology format'
                        % self.morphology).with_traceback(
                        'Should be either .hoc, .asc, .swc, .xml')

                # assuming now that morphology file is the correct format
                try:
                    Import.input(self.morphology)
                except BaseException:
                    if not hasattr(neuron, 'neuroml'):
                        raise Exception(
                            'Can not import, try and copy the ' +
                            'nrn/share/lib/python/neuron/neuroml ' +
                            'folder into %s' %
                            neuron.__path__[0])
                    else:
                        raise Exception(
                            'something wrong with file, see output')
                try:
                    imprt = neuron.h.Import3d_GUI(Import, 0)
                except BaseException:
                    raise Exception('See output, try to correct the file')

                # instantiate the cell object
                if fileEnding == 'xml' or fileEnding == 'XML':
                    # can not currently assign xml to cell template
                    try:
                        imprt.instantiate(self.template)
                    except BaseException:
                        raise Exception("this xml file is not supported")
                else:
                    imprt.instantiate(self.template)

            else:
                neuron.h.execute(
                    "xopen(\"%s\")" %
                    self.morphology, self.template)

        # set shapes and create sectionlists
        neuron.h.define_shape()
        self._create_sectionlists()

    def _create_sectionlists(self):
        """Create section lists for different kinds of sections"""

        self.allsecnames = []
        for sec in self.template.all:
            self.allsecnames.append(sec.name())

        self.allseclist = self.template.all

        # list of soma sections, assuming it is named on the format "soma*"
        self.nsomasec = 0
        self.somalist = neuron.h.SectionList()
        for sec in self.allseclist:
            if 'soma' in sec.name():
                self.somalist.append(sec=sec)
                self.nsomasec += 1
