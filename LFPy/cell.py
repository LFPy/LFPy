#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import neuron
from neuron import units
import numpy as np
import scipy.stats
import sys
import posixpath
from warnings import warn
import pickle
from .run_simulation import _run_simulation_with_probes
from .run_simulation import _collect_geometry_neuron
from .alias_method import alias_method

# check neuron version:
try:
    try:
        assert neuron.version >= '7.7.2'
    except AttributeError:
        warn('Could not read NEURON version info. v7.7.2 or newer required')
    except TypeError:
        # workaround for doc build neuron Mock module
        pass
except AssertionError:
    warn('LFPy requires NEURON v7.7.2 or newer. Found v{}'.format(
        neuron.version))


class Cell(object):
    """
    The main cell class used in LFPy.

    Parameters
    ----------
    morphology: str or neuron.h.SectionList
        File path of morphology on format that NEURON can understand (w. file
        ending .hoc, .asc, .swc or .xml), or neuron.h.SectionList instance
        filled with references to neuron.h.Section instances.
    v_init: float
        Initial membrane potential. Defaults to -70 mV.
    Ra: float or None
        Axial resistance. Defaults to None (unit Ohm*cm)
    cm: float
        Membrane capacitance. Defaults to None (unit uF/cm2)
    passive: bool
        Passive mechanisms are initialized if True. Defaults to False
    passive_parameters: dict
        parameter dictionary with values for the passive membrane mechanism in
        NEURON ('pas'). The dictionary must contain keys 'g_pas' [S/cm^2] and
        'e_pas' [mV], like the default:
        passive_parameters=dict(g_pas=0.001, e_pas=-70)
    extracellular: bool
        Switch for NEURON's extracellular mechanism. Defaults to False
    dt: float
        simulation timestep. Defaults to 2^-4 ms
    tstart: float
        Initialization time for simulation <= 0 ms. Defaults to 0.
    tstop: float
        Stop time for simulation > 0 ms. Defaults to 100 ms.
    nsegs_method: 'lambda100' or 'lambda_f' or 'fixed_length' or None
        nseg rule, used by NEURON to determine number of compartments.
        Defaults to 'lambda100'
    max_nsegs_length: float or None
        Maximum segment length for method 'fixed_length'. Defaults to None
    lambda_f: int
        AC frequency for method 'lambda_f'. Defaults to 100
    d_lambda: float
        Parameter for d_lambda rule. Defaults to 0.1
    delete_sections: bool
        Delete pre-existing section-references. Defaults to True
    custom_code: list or None
        List of model-specific code files ([.py/.hoc]). Defaults to None
    custom_fun: list or None
        List of model-specific functions with args. Defaults to None
    custom_fun_args: list or None
        List of args passed to custom_fun functions. Defaults to None
    pt3d: bool
        Use pt3d-info of the cell geometries switch. Defaults to False
    celsius: float or None
        Temperature in celsius. If nothing is specified here
        or in custom code it is 6.3 celcius
    verbose: bool
        Verbose output switch. Defaults to False

    Examples
    --------
    Simple example of how to use the Cell class with a passive-circuit
    morphology (modify morphology path accordingly):

    >>> import os
    >>> import LFPy
    >>> cellParameters = {
    >>>     'morphology': os.path.join('examples', 'morphologies',
    >>>                                'L5_Mainen96_LFPy.hoc'),
    >>>     'v_init': -65.,
    >>>     'cm': 1.0,
    >>>     'Ra': 150,
    >>>     'passive': True,
    >>>     'passive_parameters': {'g_pas': 1./30000, 'e_pas': -65},
    >>>     'dt': 2**-3,
    >>>     'tstart': 0,
    >>>     'tstop': 50,
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>> cell.simulate()
    >>> print(cell.somav)

    See also
    --------
    TemplateCell
    NetworkCell
    """
    def __init__(self, morphology,
                 v_init=-70.,
                 Ra=None,
                 cm=None,
                 passive=False,
                 passive_parameters=None,
                 extracellular=False,
                 tstart=0.,
                 tstop=100.,
                 dt=2**-4,
                 nsegs_method='lambda100',
                 lambda_f=100,
                 d_lambda=0.1,
                 max_nsegs_length=None,
                 delete_sections=True,
                 custom_code=None,
                 custom_fun=None,
                 custom_fun_args=None,
                 pt3d=False,
                 celsius=None,
                 verbose=False,
                 **kwargs):
        self.verbose = verbose
        self.pt3d = pt3d

        if passive_parameters is None:
            passive_parameters = dict(g_pas=0.001, e_pas=-70.)

        # check if there are un-used keyword arguments present in kwargs
        for key, value in kwargs.items():
            raise ValueError('keyword/argument {}={}'.format(key, value),
                             'is invalid input to class LFPy.Cell')

        if passive:
            assert isinstance(passive_parameters, dict), \
                'passive_parameters must be a dictionary'
            for key in ['g_pas', 'e_pas']:
                assert key in passive_parameters.keys(), \
                    'key {} not found in passive_parameters'.format(key)

        if not hasattr(neuron.h, 'd_lambda'):
            neuron.h.load_file('stdlib.hoc')  # NEURON std. library
            neuron.h.load_file('import3d.hoc')  # import 3D morphology lib
        if not hasattr(neuron.h, 'continuerun'):
            neuron.h.load_file('stdrun.hoc')  # NEURON stdrun library

        if delete_sections:
            if not isinstance(morphology, type(neuron.h.SectionList)):
                if self.verbose:
                    print('%s existing sections deleted from memory' %
                          sum(1 for sec in neuron.h.allsec()))
                neuron.h('forall delete_section()')
        else:
            if not isinstance(morphology, type(neuron.h.SectionList)):
                mssg = "%s sections detected! " % sum(
                    1 for sec in neuron.h.allsec()) \
                    + "Consider setting 'delete_sections=True'"
                warn(mssg)

        # load morphology
        assert morphology is not None, \
            ('deprecated keyword argument morphology==None, value must be ' +
             'a file path or neuron.h.SectionList instance with ' +
             'neuron.h.Section instances')
        if "win32" in sys.platform and isinstance(morphology, str):
            # fix Path on windows
            morphology = morphology.replace(os.sep, posixpath.sep)
        self.morphology = morphology
        if isinstance(self.morphology, str):
            if os.path.isfile(self.morphology):
                self._load_geometry()
            else:
                raise Exception('non-existent file %s' % self.morphology)
        else:
            assert isinstance(self.morphology, type(neuron.h.SectionList)), \
                ("Could not recognize Cell keyword argument morphology as " +
                 "neuron.h.SectionList instance")

            # instantiate 3D geometry of all sections
            neuron.h.define_shape()
            # set some additional attributes
            self._create_sectionlists()

        # Some parameters and lists initialised
        assert tstart <= 0, 'tstart must be <= 0.'

        try:
            assert dt in 2.**np.arange(-16, -1)
        except AssertionError:
            if tstart == 0.:
                if self.verbose:
                    print('int(1./dt) not factorizable in base 2. cell.tvec '
                          'errors may occur, continuing initialization.')
            elif tstart < 0:
                raise AssertionError(
                    'int(1./dt) must be factorizable in base 2 if tstart < 0.')

        self.dt = dt

        self.tstart = tstart
        self.tstop = tstop

        self.synapses = []
        self.synidx = []
        self.pointprocesses = []
        self.pointprocess_idx = []

        self.v_init = v_init

        self.default_rotation = self.__get_rotation()

        # Set axial resistance and membrane capacitance
        self.Ra = Ra
        self.cm = cm
        self.__set_ra_and_cm()

        # Set passive properties, insert passive on all segments
        self.passive_parameters = passive_parameters
        if passive:
            self.__set_passive()
        else:
            if self.verbose:
                print('No passive properties added')

        # run user specified code and functions if argument given
        if custom_code is not None or custom_fun is not None:
            self.__run_custom_codes(custom_code, custom_fun, custom_fun_args)

        # Insert extracellular mech on all segments
        self.extracellular = extracellular
        if self.extracellular:
            self.__set_extracellular()
        else:
            if self.verbose:
                print("no extracellular mechanism inserted")

        # set number of segments accd to rule, and calculate the number
        self.__set_negs(nsegs_method, lambda_f, d_lambda, max_nsegs_length)
        self.totnsegs = self.__calc_totnsegs()
        if self.verbose:
            print("Total number of segments: %i" % self.totnsegs)

        # extract pt3d info from NEURON, and set these with the same rotation
        # and position in space as in our simulations, assuming RH rule, which
        # NEURON do NOT use in shape plot
        if self.pt3d:
            self.x3d, self.y3d, self.z3d, self.diam3d = self._collect_pt3d()

        # Gather geometry, set position and rotation of morphology
        if self.pt3d:
            self._update_pt3d()
        else:  # self._update_pt3d makes a call to self._collect_geometry()
            self._collect_geometry()
        if hasattr(self, 'somapos'):
            self.set_pos()
        else:
            if self.verbose:
                print('no soma, using the midpoint if initial segment.')
        self.set_rotation(**self.default_rotation)

        if celsius is not None:
            if neuron.h.celsius != 6.3:
                print("Changing temperature %1.2f to %1.2f"
                      % (neuron.h.celsius, celsius))
            neuron.h.celsius = celsius

        # initialize membrane voltage in all segments.
        neuron.h.finitialize(self.v_init * units.mV)
        self._neuron_tvec = None

    def __del__(self):
        """Cell finalizer"""
        self.strip_hoc_objects()

    def strip_hoc_objects(self):
        """Destroy any NEURON hoc objects in the cell object"""
        if not (isinstance(neuron, type(None)) or
                isinstance(neuron.nrn, type(None))):
            nrntypes = (neuron.nrn.Segment, neuron.nrn.Section,
                        neuron.nrn.Mechanism, type(neuron.h.List()))
            for key in self.__dict__.keys():
                if isinstance(getattr(self, key), nrntypes):
                    setattr(self, key, None)
                    if self.verbose:
                        print('{}.{} = None'.format(self.__name__, key))

    def _load_geometry(self):
        """Load the morphology-file in NEURON"""
        # import the morphology, try and determine format
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
                raise ValueError('%s not a recognised morphology file format'
                                 % self.morphology).with_traceback(
                    'Should be either .hoc, .asc, .swc, .xml')

            # assuming now that morphologies file is the correct format
            try:
                Import.input(self.morphology)
            except BaseException:
                if not hasattr(neuron, 'neuroml'):
                    raise Exception('Can not import, try and copy the '
                                    'nrn/share/lib/python/neuron/neuroml '
                                    'folder into %s' % neuron.__path__[0])
                else:
                    raise Exception('something wrong with file, see output')
            try:
                imprt = neuron.h.Import3d_GUI(Import, 0)
            except BaseException:
                raise Exception('See output, try to correct the file')
            imprt.instantiate(neuron.h.this)

        neuron.h.define_shape()
        self._create_sectionlists()

    def __run_custom_codes(self, custom_code, custom_fun, custom_fun_args):
        """Execute custom model code and functions with arguments"""
        # load custom codes
        if custom_code is not None:
            for code in custom_code:
                if "win32" in sys.platform:
                    code = code.replace(os.sep, posixpath.sep)
                if code.split('.')[-1] == 'hoc':
                    try:
                        neuron.h.xopen(code)
                    except RuntimeError:
                        ERRMSG = '\n'.join(
                            [
                                '',
                                'Could not load custom model code (%s)' % code,
                                'while creating a Cell object.',
                                'One possible cause is NEURON mechanisms have',
                                'not been compiled, ',
                                'try running nrnivmodl or mknrndll (Windows) ',
                                'in the .mod-file containing folder. ',
                            ])
                        raise Exception(ERRMSG)
                elif code.split('.')[-1] == 'py':
                    exec(code)
                else:
                    raise Exception('%s not a .hoc- nor .py-file' % code)

        # run custom functions with arguments
        i = 0
        if custom_fun is not None:
            for fun in custom_fun:
                fun(self, **custom_fun_args[i])
                i += 1

        # recreate sectionlists in case something changed
        neuron.h.define_shape()
        self._create_sectionlists()

    def __set_negs(self, nsegs_method, lambda_f, d_lambda, max_nsegs_length):
        """Set number of segments per section according to the lambda-rule,
        or according to maximum length of segments"""
        if nsegs_method == 'lambda100':
            self.__set_nsegs_lambda100(d_lambda)
        elif nsegs_method == 'lambda_f':
            self.__set_nsegs_lambda_f(lambda_f, d_lambda)
        elif nsegs_method == 'fixed_length':
            self.__set_nsegs_fixed_length(max_nsegs_length)
        else:
            if self.verbose:
                print('No nsegs_method applied (%s)' % nsegs_method)

    def __get_rotation(self):
        """Check if there exists a corresponding file
        with rotation angles"""
        if isinstance(self.morphology, str):
            base = os.path.splitext(self.morphology)[0]
            if os.path.isfile(base + '.rot'):
                rotation_file = base + '.rot'
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
        """Create section lists for different kinds of sections"""
        # list with all sections
        self.allsecnames = []
        if not isinstance(self.morphology, type(neuron.h.SectionList)):
            self.allseclist = neuron.h.SectionList()
            for sec in neuron.h.allsec():
                self.allsecnames.append(sec.name())
                self.allseclist.append(sec=sec)
        else:
            self.allseclist = self.morphology
            for sec in neuron.h.allsec():
                self.allsecnames.append(sec.name())

        # list of soma sections, assuming it is named on the format "soma*"
        self.nsomasec = 0
        self.somalist = neuron.h.SectionList()
        for sec in neuron.h.allsec():
            if sec.name().find('soma') >= 0:
                self.somalist.append(sec=sec)
                self.nsomasec += 1

    def __get_idx(self, seclist):
        """Return boolean vector which indexes where segments in seclist
        matches segments in neuron.h.allsec(), rewritten from
        LFPy.hoc function get_idx()"""
        if neuron.h.allsec() == seclist:
            return np.ones(self.totnsegs, dtype=bool)
        else:
            idxvec = np.zeros(self.totnsegs, dtype=bool)
            # get sectionnames from seclist
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

    def __set_nsegs_lambda_f(self, frequency=100, d_lambda=0.1):
        """Set the number of segments for section according to the
        d_lambda-rule for a given input frequency

        Parameters
        ----------
        frequency: float
            frequency at which AC length constant is computed
        d_lambda: float
        """
        neuron.h.pop_section()  # dirty fix, see NEURON doc
        for sec in self.allseclist:
            sec.nseg = int(
                (sec.L / (d_lambda * neuron.h.lambda_f(frequency, sec=sec))
                 + .9) / 2) * 2 + 1
        if self.verbose:
            print("set nsegs using lambda-rule with frequency %i." % frequency)

    def __set_nsegs_lambda100(self, d_lambda=0.1):
        """Set the numbers of segments using d_lambda(100)"""
        self.__set_nsegs_lambda_f(frequency=100, d_lambda=d_lambda)

    def __set_nsegs_fixed_length(self, maxlength):
        """Set nseg for sections so that every segment L < maxlength"""
        for sec in self.allseclist:
            sec.nseg = int(sec.L / maxlength) + 1

    def __calc_totnsegs(self):
        """Calculate the number of segments in the allseclist"""
        i = 0
        for sec in self.allseclist:
            i += sec.nseg

        return i

    def __check_currents(self):
        """Check that the sum of all membrane and electrode currents over all
        segments is sufficiently close to zero"""
        raise NotImplementedError('this function need to be written')

    def __set_ra_and_cm(self):
        """Insert ra and cm on all segments"""
        for sec in self.allseclist:
            if self.Ra is not None:
                sec.Ra = self.Ra
            if self.cm is not None:
                sec.cm = self.cm

    def __set_passive(self):
        """Insert passive mechanism on all segments"""
        for sec in self.allseclist:
            sec.insert('pas')
            sec.g_pas = self.passive_parameters['g_pas']
            sec.e_pas = self.passive_parameters['e_pas']

    def __set_extracellular(self):
        """Insert extracellular mechanism on all sections
        to set an external potential V_ext as boundary condition.
        """

        for sec in self.allseclist:
            sec.insert('extracellular')
        self.extracellular = True

    def set_synapse(self, idx, syntype,
                    record_current=False,
                    record_potential=False,
                    weight=None, **kwargs):
        """Insert synapse on cell segment

        Parameters
        ----------
        idx: int
            Index of compartment where synapse is inserted
        syntype: str
            Type of synapse. Built-in types in NEURON: ExpSyn, Exp2Syn
        record_current: bool
            If True, record synapse current
        record_potential: bool
            If True, record postsynaptic potential seen by the synapse
        weight: float
            Strength of synapse
        kwargs
            arguments passed on from class Synapse

        Returns
        -------
        int
            index of synapse object on cell
        """
        if not hasattr(self, '_hoc_synlist'):
            self._hoc_synlist = neuron.h.List()
        if not hasattr(self, '_synitorecord'):
            self._synitorecord = []
        if not hasattr(self, '_synvtorecord'):
            self._synvtorecord = []
        if not hasattr(self, '_hoc_netstimlist'):
            self._hoc_netstimlist = neuron.h.List()
        if not hasattr(self, '_hoc_netconlist'):
            self._hoc_netconlist = neuron.h.List()
        if not hasattr(self, '_sptimeslist'):
            self._sptimeslist = []

        # need to append w. one empty array per synapse
        self._sptimeslist.append(np.array([]))

        i = 0
        cmd = 'neuron.h.{}(seg.x, sec=sec)'
        for sec in self.allseclist:
            for seg in sec:
                if i == idx:
                    command = cmd.format(syntype)
                    syn = eval(command, locals(), globals())
                    for param in list(kwargs.keys()):
                        try:
                            setattr(syn, param, kwargs[param])
                        except BaseException:
                            pass
                    self._hoc_synlist.append(syn)

                    # create NetStim (generator) and NetCon (connection)
                    # objects
                    self._hoc_netstimlist.append(neuron.h.NetStim(0.5))
                    self._hoc_netstimlist[-1].number = 0

                    nc = neuron.h.NetCon(self._hoc_netstimlist[-1], syn)
                    nc.weight[0] = weight
                    self._hoc_netconlist.append(nc)

                    # record current
                    if record_current:
                        self._synitorecord.append(
                            self._hoc_synlist.count() - 1)

                    # record potential
                    if record_potential:
                        self._synvtorecord.append(
                            self._hoc_synlist.count() - 1)

                i += 1

        return self._hoc_synlist.count() - 1

    def set_point_process(self, idx, pptype, record_current=False,
                          record_potential=False, **kwargs):
        """Insert pptype-electrode type pointprocess on segment numbered
        idx on cell object

        Parameters
        ----------
        idx: int
            Index of compartment where point process is inserted
        pptype: str
            Type of pointprocess. Examples: SEClamp, VClamp,
            IClamp, SinIClamp, ChirpIClamp
        record_current: bool
            Decides if current is stored
        kwargs
            Parameters passed on from class StimIntElectrode

        Returns
        -------
        int
            index of point process on cell
        """

        if not hasattr(self, '_hoc_stimlist'):
            self._hoc_stimlist = neuron.h.List()
        if not hasattr(self, '_stimitorecord'):
            self._stimitorecord = []
        if not hasattr(self, '_stimvtorecord'):
            self._stimvtorecord = []

        i = 0
        cmd1 = 'neuron.h.'
        cmd2 = '(seg.x, sec=sec)'
        ppset = False
        for sec in self.allseclist:
            for seg in sec:
                if i == idx:
                    command = cmd1 + pptype + cmd2
                    stim = eval(command, locals(), globals())
                    for key, value in kwargs.items():
                        try:
                            itr = enumerate(iter(value))
                        except TypeError:
                            setattr(stim, key, value)
                        else:
                            for i, v in itr:
                                getattr(stim, key)[i] = v
                    self._hoc_stimlist.append(stim)

                    # record current
                    if record_current:
                        self._stimitorecord.append(
                            self._hoc_stimlist.count() - 1)

                    # record potential
                    if record_potential:
                        self._stimvtorecord.append(
                            self._hoc_stimlist.count() - 1)

                    ppset = True
                    break
                i += 1
            if ppset:
                break

        return self._hoc_stimlist.count() - 1

    def _collect_geometry(self):
        """Collects x, y, z-coordinates from NEURON"""
        # None-type some attributes if they do not exist:
        if not hasattr(self, 'x'):
            self.x = None
            self.y = None
            self.z = None
            self.area = None
            self.d = None
            self.length = None

        _collect_geometry_neuron(self)

        self.somaidx = self.get_idx(section='soma')

        if self.somaidx.size > 1:
            xmids = self.x[self.somaidx].mean(axis=-1)
            ymids = self.y[self.somaidx].mean(axis=-1)
            zmids = self.z[self.somaidx].mean(axis=-1)
            self.somapos = np.zeros(3)
            self.somapos[0] = xmids.mean()
            self.somapos[1] = ymids.mean()
            self.somapos[2] = zmids.mean()
        elif self.somaidx.size == 1:
            self.somapos = np.zeros(3)
            self.somapos[0] = self.x[self.somaidx].mean()
            self.somapos[1] = self.y[self.somaidx].mean()
            self.somapos[2] = self.z[self.somaidx].mean()
        elif self.somaidx.size == 0:
            if self.verbose:
                warn("There is no soma!" +
                     "Using first segment as root point")
            self.somaidx = np.array([0])
            self.somapos = np.zeros(3)
            self.somapos[0] = self.x[self.somaidx].mean()
            self.somapos[1] = self.y[self.somaidx].mean()
            self.somapos[2] = self.z[self.somaidx].mean()
        else:
            raise Exception('Huh?!')

    def get_idx(self, section='allsec', z_min=-np.inf, z_max=np.inf):
        """
        Returns compartment idx of segments from sections with names that match
        the pattern defined in input section on interval [z_min, z_max].

        Parameters
        ----------
        section: str
            Any entry in cell.allsecnames or just 'allsec'.
        z_min: float
            Depth filter. Specify minimum z-position
        z_max: float
            Depth filter. Specify maximum z-position

        Returns
        -------
        ndarray, dtype=int
            segment indices

        Examples
        --------
        >>> idx = cell.get_idx(section='allsec')
        >>> print(idx)
        >>> idx = cell.get_idx(section=['soma', 'dend', 'apic'])
        >>> print(idx)
        """

        if section == 'allsec':
            seclist = neuron.h.allsec()
        else:
            seclist = neuron.h.SectionList()
            if isinstance(section, str):
                for sec in self.allseclist:
                    if sec.name().find(section) >= 0:
                        seclist.append(sec=sec)
            elif isinstance(section, list):
                for secname in section:
                    for sec in self.allseclist:
                        if sec.name().find(secname) >= 0:
                            seclist.append(sec=sec)
            else:
                if self.verbose:
                    print('%s did not match any section name' % str(section))

        idx = self.__get_idx(seclist)
        sel_z_idx = (
            self.z[idx].mean(
                axis=-
                1) > z_min) & (
            self.z[idx].mean(
                axis=-
                1) < z_max)
        return np.arange(self.totnsegs)[idx][sel_z_idx]

    def get_closest_idx(self, x=0., y=0., z=0., section='allsec'):
        """Get the index number of a segment in specified section which
        midpoint is closest to the coordinates defined by the user

        Parameters
        ----------
        x: float
            x-coordinate
        y: float
            y-coordinate
        z: float
            z-coordinate
        section: str
            String matching a section-name. Defaults to 'allsec'.

        Returns
        -------
        int
            segment index
        """
        idx = self.get_idx(section)
        dist = ((self.x[idx].mean(axis=-1) - x)**2 +
                (self.y[idx].mean(axis=-1) - y)**2 +
                (self.z[idx].mean(axis=-1) - z)**2)
        return idx[np.argmin(dist)]

    def get_rand_idx_area_norm(self, section='allsec', nidx=1,
                               z_min=-1E6, z_max=1E6):
        """Return nidx segment indices in section with random probability
        normalized to the membrane area of segment on
        interval [z_min, z_max]

        Parameters
        ----------
        section: str
            String matching a section-name
        nidx: int
            Number of random indices
        z_min: float
            Depth filter
        z_max: float
            Depth filter

        Returns
        -------
        ndarray, dtype=int
            segment indices
        """
        poss_idx = self.get_idx(section=section, z_min=z_min, z_max=z_max)
        if nidx < 1:
            print('nidx < 1, returning empty array')
            return np.array([])
        elif poss_idx.size == 0:
            print('No possible segment idx match quiery - returning '
                  'empty array')
            return np.array([])
        else:
            area = self.area[poss_idx]
            area /= area.sum()
            return alias_method(poss_idx, area, nidx)

    def get_rand_idx_area_and_distribution_norm(self, section='allsec', nidx=1,
                                                z_min=-1E6, z_max=1E6,
                                                fun=scipy.stats.norm,
                                                funargs=dict(loc=0, scale=100),
                                                funweights=None):
        """
        Return nidx segment indices in section with random probability
        normalized to the membrane area of each segment multiplied by
        the value of the probability density function of "fun", a function
        in the scipy.stats module with corresponding function arguments
        in "funargs" on the interval [z_min, z_max]

        Parameters
        ----------
        section: str
            string matching a section name
        nidx: int
            number of random indices
        z_min: float
            lower depth interval
        z_max: float
            upper depth interval
        fun: function or str, or iterable of function or str
            if function a scipy.stats method, if str, must be method in
            scipy.stats module with the same name (like 'norm'),
            if iterable (list, tuple, numpy.array) of function or str some
            probability distribution in scipy.stats module
        funargs: dict or iterable
            iterable (list, tuple, numpy.array) of dict, arguments to fun.pdf
            method (e.g., w. keys 'loc' and 'scale')
        funweights: None or iterable
            iterable (list, tuple, numpy.array) of floats, scaling of each
            individual fun (i.e., introduces layer specificity)

        Examples
        --------
        >>> import LFPy
        >>> import numpy as np
        >>> import scipy.stats as ss
        >>> import matplotlib.pyplot as plt
        >>> from os.path import join
        >>> cell = LFPy.Cell(morphology=join('cells', 'cells', 'j4a.hoc'))
        >>> cell.set_rotation(x=4.99, y=-4.33, z=3.14)
        >>> idx = cell.get_rand_idx_area_and_distribution_norm(
            nidx=10000, fun=ss.norm, funargs=dict(loc=0, scale=200))
        >>> bins = np.arange(-30, 120)*10
        >>> plt.hist(cell.zmid[idx], bins=bins, alpha=0.5)
        >>> plt.show()
        """
        poss_idx = self.get_idx(section=section, z_min=z_min, z_max=z_max)
        if nidx < 1:
            print('nidx < 1, returning empty array')
            return np.array([])
        elif poss_idx.size == 0:
            print('No possible segment idx match query - returning '
                  'empty array')
            return np.array([])
        else:
            p = self.area[poss_idx]
            # scale with density function
            if type(fun) in [list, tuple, np.ndarray]:
                assert type(funargs) in [list, tuple, np.ndarray]
                assert type(funweights) in [list, tuple, np.ndarray]
                assert len(fun) == len(funargs) & len(fun) == len(funweights)
                mod = np.zeros(poss_idx.shape)
                for f, args, scl in zip(fun, funargs, funweights):
                    if isinstance(f, str) and f in dir(scipy.stats):
                        f = getattr(scipy.stats, f)
                    df = f(**args)
                    mod += df.pdf(x=self.z[poss_idx].mean(axis=-1)) * scl
                p *= mod
            else:
                if isinstance(fun, str) and fun in dir(scipy.stats):
                    fun = getattr(scipy.stats, fun)
                df = fun(**funargs)
                p *= df.pdf(x=self.z[poss_idx].mean(axis=-1))
            # normalize
            p /= p.sum()
            return alias_method(poss_idx, p, nidx)

    def enable_extracellular_stimulation(
            self, electrode, t_ext=None, n=1, model='inf'):
        r"""
        Enable extracellular stimulation with 'extracellular' mechanism.
        Extracellular potentials are computed from the electrode currents
        using the pointsource approximation.
        If 'model' is 'inf' (default), potentials are computed as
        (:math:`r_i` is the position of a comparment i,
        :math:`r_e` is the position of an elextrode e, :math:`\sigma` is the
        conductivity of the medium):

        .. math::
            V_e(r_i) = \sum_n \frac{I_n}{4 \pi \sigma |r_i - r_n|}

        If model is 'semi', the method of images is used:

        .. math::
            V_e(r_i) = \sum_n \frac{I_n}{2 \pi \sigma |r_i - r_n|}

        Parameters
        ----------
        electrode: RecExtElectrode
            Electrode object with stimulating currents
        t_ext: np.ndarray or list
            Time im ms corrisponding to step changes in the provided currents.
            If None, currents are assumed to have
            the same time steps as NEURON simulation.
        n: int
            Points per electrode to compute spatial averaging
        model: str
            'inf' or 'semi'. If 'inf' the medium is assumed to be infinite and
            homogeneous. If 'semi', the method of
            images is used.

        Returns
        -------
        v_ext: np.ndarray
            Computed extracellular potentials at cell mid points
        """
        # access electrode object and append mapping
        if electrode is not None:
            # put electrode argument in list if needed
            if isinstance(electrode, list):
                electrodes = electrode
            else:
                electrodes = [electrode]
        else:
            print("'electrode' is None")
            return

        assert model in ['inf', 'semi'], "'model' can be 'inf' or 'semi'"

        # extracellular stimulation
        if np.any([np.any(el.probe.currents != 0) for el in electrodes]):
            cell_mid_points = np.array([self.x.mean(axis=-1),
                                        self.y.mean(axis=-1),
                                        self.z.mean(axis=-1)]).T
            n_tsteps = int(self.tstop / self.dt + 1)
            t_cell = np.arange(n_tsteps) * self.dt

            if t_ext is None:
                print("Assuming t_ext is the same as simulation time")
                t_ext = t_cell
                for electrode in electrodes:
                    assert electrode.probe.currents.shape[1] == len(t_cell), \
                        ("Discrepancy between t_ext and cell simulation time" +
                         "steps. Provide the 't_ext' argument")
            else:
                assert len(t_ext) < len(t_cell), \
                    "Stimulation time steps greater than cell simulation steps"

            v_ext = np.zeros((self.totnsegs, len(t_ext)))
            for electrode in electrodes:
                if np.any(np.any(electrode.probe.currents != 0)):
                    electrode.probe.points_per_electrode = int(n)
                    electrode.probe.model = model
                    ve = electrode.probe.compute_field(cell_mid_points)
                    if len(electrode.probe.currents.shape) == 1:
                        ve = ve[:, np.newaxis]
                    v_ext += ve

            self.__set_extracellular()
            self.insert_v_ext(v_ext, np.array(t_ext))
        else:
            v_ext = None

        return v_ext

    def simulate(self, probes=None,
                 rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_variables=[],
                 variable_dt=False, atol=0.001, rtol=0.,
                 to_memory=True,
                 to_file=False, file_name=None,
                 **kwargs):
        """
        This is the main function running the simulation of the NEURON model.
        Start NEURON simulation and record variables specified by arguments.

        Parameters
        ----------
        probes: list of :obj:, optional
            None or list of LFPykit.RecExtElectrode like object instances that
            each have a public method `get_transformation_matrix` returning
            a matrix that linearly maps each compartments' transmembrane
            current to corresponding measurement as

            .. math:: \\mathbf{P} = \\mathbf{M} \\mathbf{I}

        rec_imem: bool
            If true, segment membrane currents will be recorded
            If no electrode argument is given, it is necessary to
            set rec_imem=True in order to make predictions later on.
            Units of (nA).
        rec_vmem: bool
            Record segment membrane voltages (mV)
        rec_ipas: bool
            Record passive segment membrane currents (nA)
        rec_icap: bool
            Record capacitive segment membrane currents (nA)
        rec_variables: list
            List of segment state variables to record, e.g. arg=['cai', ]
        variable_dt: bool
            Use NEURON's variable timestep method
        atol: float
            Absolute local error tolerance for NEURON variable timestep method
        rtol: float
            Relative local error tolerance for NEURON variable timestep method
        to_memory: bool
            Only valid with probes=[:obj:], store measurements as `:obj:.data`
        to_file: bool
            Only valid with probes, save simulated data in hdf5 file format
        file_name: str
            Name of hdf5 file, '.h5' is appended if it doesnt exist
        """
        for key in kwargs.keys():
            if key in ['electrode', 'rec_current_dipole_moment',
                       'dotprodcoeffs', 'rec_isyn', 'rec_vmemsyn',
                       'rec_istim', 'rec_vmemstim']:
                warn('Cell.simulate parameter {} is deprecated.'.format(key))

        # set up integrator, use the CVode().fast_imem method by default
        # as it doesn't hurt sim speeds much if at all.
        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError:
            raise Exception('neuron.h.CVode().use_fast_imem() method not '
                            'found. Update NEURON to v.7.4 or newer')

        if not variable_dt:
            dt = self.dt
        else:
            dt = None
        self._set_soma_volt_recorder(dt)

        if rec_imem:
            self._set_imem_recorders(dt)
        if rec_vmem:
            self._set_voltage_recorders(dt)
        if rec_ipas:
            self._set_ipas_recorders(dt)
        if rec_icap:
            self._set_icap_recorders(dt)
        if len(rec_variables) > 0:
            self._set_variable_recorders(rec_variables, dt)
        if hasattr(self, '_stimitorecord'):
            if len(self._stimitorecord) > 0:
                self.__set_ipointprocess_recorders(dt)
        if hasattr(self, '_stimvtorecord'):
            if len(self._stimvtorecord) > 0:
                self.__set_vpointprocess_recorders(dt)
        if hasattr(self, '_synitorecord'):
            if len(self._synitorecord) > 0:
                self.__set_isyn_recorders(dt)
        if hasattr(self, '_synvtorecord'):
            if len(self._synvtorecord) > 0:
                self.__set_vsyn_recorders(dt)

        # set time recorder from NEURON
        self.__set_time_recorders(dt)

        # run fadvance until t >= tstop, and calculate LFP if asked for
        if probes is None or len(probes) == 0:
            if not rec_imem and self.verbose:
                print("rec_imem = %s, membrane currents will not be recorded!"
                      % str(rec_imem))
            self.__run_simulation(cvode, variable_dt, atol, rtol)

        else:
            # simulate with probes saving to memory and/or file:
            _run_simulation_with_probes(self, cvode, probes,
                                        variable_dt, atol, rtol,
                                        to_memory, to_file, file_name)

        # somatic trace
        if self.nsomasec >= 1:
            self.somav = np.array(self.somav)

        self.__collect_tvec()

        if rec_imem:
            self._calc_imem()
        if rec_ipas:
            self._calc_ipas()
        if rec_icap:
            self._calc_icap()
        if rec_vmem:
            self._collect_vmem()

        if hasattr(self, '_hoc_stimireclist'):
            self._collect_istim()
        if hasattr(self, '_hoc_stimvreclist'):
            self.__collect_vstim()
        if hasattr(self, '_hoc_synireclist'):
            self._collect_isyn()
        if hasattr(self, '_hoc_synvreclist'):
            self._collect_vsyn()
        if len(rec_variables) > 0:
            self._collect_rec_variables(rec_variables)
        if hasattr(self, '_hoc_netstimlist'):
            self._hoc_netstimlist = None
            del self._hoc_netstimlist

        self.__purge_hoc_pointprocesses()

    def __purge_hoc_pointprocesses(self):
        """
        Empty lists which may store point process objects in `hoc` name space.
        This is needed to avoid "Segmentation Fault 11"
        """
        if hasattr(self, '_hoc_synlist'):
            self._hoc_synlist.remove_all()
        if hasattr(self, '__hoc_stimlist'):
            self._hoc_stimlist.remove_all()

    def __run_simulation(self, cvode, variable_dt=False, atol=0.001, rtol=0.):
        """
        Running the actual simulation in NEURON, simulations in NEURON
        is now interruptable.
        """
        neuron.h.dt = self.dt

        # variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
            cvode.rtol(rtol)
        else:
            cvode.active(0)

        # re-initialize state
        neuron.h.finitialize(self.v_init * units.mV)

        # initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()

        # Starting simulation at t != 0
        neuron.h.t = self.tstart

        self._load_spikes()

        # advance simulation until tstop
        neuron.h.continuerun(self.tstop * units.ms)

        # for consistency with 'old' behaviour where tstop is included in tvec:
        if neuron.h.t < self.tstop:
            neuron.h.fadvance()

    def __collect_tvec(self):
        """
        Set the tvec to be a monotonically increasing numpy array after sim.
        """
        self.tvec = np.array(self._neuron_tvec.to_python())
        self._neuron_tvec = None
        del self._neuron_tvec

    def _calc_imem(self):
        """
        Fetch the vectors from the memireclist and calculate self.imem
        containing all the membrane currents.
        """
        self.imem = np.array(self._hoc_memireclist)
        self._hoc_memireclist = None
        del self._hoc_memireclist

    def _calc_ipas(self):
        """
        Get the passive currents
        """
        self.ipas = np.array(self._hoc_memipasreclist)
        for i in range(self.ipas.shape[0]):
            self.ipas[i, ] *= self.area[i] * 1E-2
        self._hoc_memipasreclist = None
        del self._hoc_memipasreclist

    def _calc_icap(self):
        """
        Get the capacitive currents
        """
        self.icap = np.array(self._hoc_memicapreclist)
        for i in range(self.icap.shape[0]):
            self.icap[i, ] *= self.area[i] * 1E-2
        self._hoc_memicapreclist = None
        del self._hoc_memicapreclist

    def _collect_vmem(self):
        """
        Get the membrane currents
        """
        self.vmem = np.array(self._hoc_memvreclist)
        self._hoc_memvreclist = None
        del self._hoc_memvreclist

    def _collect_isyn(self):
        """
        Get the synaptic currents
        """
        for syn in self.synapses:
            if syn.record_current:
                syn.collect_current(self)
        self._hoc_synireclist = None
        del self._hoc_synireclist

    def _collect_vsyn(self):
        """
        Collect the membrane voltage of segments with synapses
        """
        for syn in self.synapses:
            if syn.record_potential:
                syn.collect_potential(self)
        self._hoc_synvreclist = None
        del self._hoc_synvreclist

    def _collect_istim(self):
        """
        Get the pointprocess currents
        """
        for pp in self.pointprocesses:
            if pp.record_current:
                pp.collect_current(self)
        self._hoc_stimireclist = None
        del self._hoc_stimireclist

    def __collect_vstim(self):
        """
        Collect the membrane voltage of segments with stimulus
        """
        for pp in self.pointprocesses:
            if pp.record_potential:
                pp.collect_potential(self)
        self._hoc_stimvreclist = None
        del self._hoc_stimvreclist

    def _collect_rec_variables(self, rec_variables):
        """
        Create dict of np.arrays from recorded variables, each dictionary
        element named as the corresponding recorded variable name, i.e 'cai'
        """
        self.rec_variables = {}
        i = 0
        for values in self._hoc_recvariablesreclist:
            self.rec_variables.update({rec_variables[i]: np.array(values)})
            if self.verbose:
                print('collected recorded variable %s' % rec_variables[i])
            i += 1
        del self._hoc_recvariablesreclist

    def _load_spikes(self):
        """
        Initialize spiketimes from netcon if they exist
        """
        if hasattr(self, '_hoc_synlist'):
            if len(self._hoc_synlist) == len(self._sptimeslist):
                for i in range(int(self._hoc_synlist.count())):
                    for spt in self._sptimeslist[i]:
                        self._hoc_netconlist.o(i).event(spt)

    def _set_soma_volt_recorder(self, dt):
        """Record somatic membrane potential"""

        if self.nsomasec == 0:
            if self.verbose:
                warn('Cell instance appears to have no somatic section. '
                     'No somav attribute will be set.')
        elif self.nsomasec == 1:
            if dt is not None:
                self.somav = neuron.h.Vector(int(self.tstop / self.dt + 1))
                for sec in self.somalist:
                    self.somav.record(sec(0.5)._ref_v, self.dt)
            else:
                self.somav = neuron.h.Vector()
                for sec in self.somalist:
                    self.somav.record(sec(0.5)._ref_v)
        elif self.nsomasec > 1:
            if dt is not None:
                self.somav = neuron.h.Vector(int(self.tstop / self.dt + 1))
                nseg = self.get_idx('soma').size
                i, j = divmod(nseg, 2)
                k = 1
                for sec in self.somalist:
                    for seg in sec:
                        if nseg == 2 and k == 1:
                            # if 2 segments, record from the first one:
                            self.somav.record(seg._ref_v, self.dt)
                        else:
                            if k == i * 2:
                                # record from one of the middle segments:
                                self.somav.record(seg._ref_v, self.dt)
                        k += 1
            else:
                self.somav = neuron.h.Vector()
                nseg = self.get_idx('soma').size
                i, j = divmod(nseg, 2)
                k = 1
                for sec in self.somalist:
                    for seg in sec:
                        if nseg == 2 and k == 1:
                            # if 2 segments, record from the first one:
                            self.somav.record(seg._ref_v)
                        else:
                            if k == i * 2:
                                # record from one of the middle segments:
                                self.somav.record(seg._ref_v)
                        k += 1

    def _set_imem_recorders(self, dt):
        """
        Record membrane currents for all segments
        """
        self._hoc_memireclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                if dt is not None:
                    memirec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    memirec.record(seg._ref_i_membrane_, self.dt)
                else:
                    memirec = neuron.h.Vector()
                    memirec.record(seg._ref_i_membrane_)
                self._hoc_memireclist.append(memirec)

    def __set_time_recorders(self, dt):
        """
        Record time of simulation
        """
        if dt is not None:
            self._neuron_tvec = neuron.h.Vector(int(self.tstop / self.dt + 1))
            self._neuron_tvec.record(neuron.h._ref_t, self.dt)
        else:
            self._neuron_tvec = neuron.h.Vector()
            self._neuron_tvec.record(neuron.h._ref_t)

    def _set_ipas_recorders(self, dt):
        """
        Record passive membrane currents for all segments
        """
        self._hoc_memipasreclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                if dt is not None:
                    memipasrec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    memipasrec.record(seg._ref_i_pas, self.dt)
                else:
                    memipasrec = neuron.h.Vector()
                    memipasrec.record(seg._ref_i_pas)
                self._hoc_memipasreclist.append(memipasrec)

    def _set_icap_recorders(self, dt):
        """
        Record capacitive membrane currents for all segments
        """
        self._hoc_memicapreclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                if dt is not None:
                    memicaprec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    memicaprec.record(seg._ref_i_cap, self.dt)
                else:
                    memicaprec = neuron.h.Vector()
                    memicaprec.record(seg._ref_i_cap)
                self._hoc_memicapreclist.append(memicaprec)

    def __set_ipointprocess_recorders(self, dt):
        """
        Record point process current
        """
        self._hoc_stimireclist = neuron.h.List()
        for idx, pp in enumerate(self.pointprocesses):
            if idx in self._stimitorecord:
                stim = self._hoc_stimlist[idx]
                if dt is not None:
                    stimirec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    stimirec.record(stim._ref_i, self.dt)
                else:
                    stimirec = neuron.h.Vector()
                    stimirec.record(stim._ref_i)
            else:
                stimirec = neuron.h.Vector(0)
            self._hoc_stimireclist.append(stimirec)

    def __set_vpointprocess_recorders(self, dt):
        """
        Record point process membrane
        """
        self._hoc_stimvreclist = neuron.h.List()
        for idx, pp in enumerate(self.pointprocesses):
            if idx in self._stimvtorecord:
                stim = self._hoc_stimlist[idx]
                seg = stim.get_segment()
                if dt is not None:
                    stimvrec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    stimvrec.record(seg._ref_v, self.dt)
                else:
                    stimvrec = neuron.h.Vector()
                    stimvrec.record(seg._ref_v)
            else:
                stimvrec = neuron.h.Vector(0)
            self._hoc_stimvreclist.append(stimvrec)

    def __set_isyn_recorders(self, dt):
        """
        Record point process current
        """
        self._hoc_synireclist = neuron.h.List()
        for idx, pp in enumerate(self.synapses):
            if idx in self._synitorecord:
                syn = self._hoc_synlist[idx]
                if dt is not None:
                    synirec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    synirec.record(syn._ref_i, self.dt)
                else:
                    synirec = neuron.h.Vector()
                    synirec.record(syn._ref_i)
            else:
                synirec = neuron.h.Vector(0)
            self._hoc_synireclist.append(synirec)

    def __set_vsyn_recorders(self, dt):
        """
        Record point process membrane
        """
        self._hoc_synvreclist = neuron.h.List()
        for idx, pp in enumerate(self.synapses):
            if idx in self._synvtorecord:
                syn = self._hoc_synlist[idx]
                seg = syn.get_segment()
                if dt is not None:
                    synvrec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    synvrec.record(seg._ref_v, self.dt)
                else:
                    synvrec = neuron.h.Vector()
                    synvrec.record(seg._ref_v)
            else:
                synvrec = neuron.h.Vector(0)
            self._hoc_synvreclist.append(synvrec)

    def _set_voltage_recorders(self, dt):
        """
        Record membrane potentials for all segments
        """
        self._hoc_memvreclist = neuron.h.List()
        for sec in self.allseclist:
            for seg in sec:
                if dt is not None:
                    memvrec = neuron.h.Vector(int(self.tstop / self.dt + 1))
                    memvrec.record(seg._ref_v, self.dt)
                else:
                    memvrec = neuron.h.Vector()
                    memvrec.record(seg._ref_v)
                self._hoc_memvreclist.append(memvrec)

    def __set_current_dipole_moment_array(self, dt):
        """
        Creates container for current dipole moment, an empty
        n_timesteps x 3 `numpy.ndarray` that will be filled with values during
        the course of each simulation
        """
        if dt is not None:
            self.current_dipole_moment = np.zeros(
                (int(self.tstop / self.dt + 1), 3))
        else:
            self.current_dipole_moment = []

    def _set_variable_recorders(self, rec_variables, dt):
        """
        Create a recorder for each variable name in list
        rec_variables
        Variables is stored in nested list self._hoc_recvariablesreclist
        """
        self._hoc_recvariablesreclist = neuron.h.List()
        for variable in rec_variables:
            variablereclist = neuron.h.List()
            self._hoc_recvariablesreclist.append(variablereclist)
            for sec in self.allseclist:
                for seg in sec:
                    if dt is not None:
                        recvector = neuron.h.Vector(
                            int(self.tstop / self.dt + 1))
                    else:
                        recvector = neuron.h.Vector()
                    try:
                        if dt is not None:
                            recvector.record(
                                getattr(
                                    seg, '_ref_%s' %
                                    variable), self.dt)
                        else:
                            recvector.record(
                                getattr(
                                    seg, '_ref_%s' %
                                    variable))
                    except(NameError, AttributeError):
                        print('non-existing variable %s, section %s.%f' %
                              (variable, sec.name(), seg.x))
                    variablereclist.append(recvector)

    def set_pos(self, x=0., y=0., z=0.):
        """Set the cell position.
        Move the cell geometry so that midpoint of soma section is
        in (x, y, z). If no soma pos, use the first segment

        Parameters
        ----------
        x: float
            x position defaults to 0.0
        y: float
            y position defaults to 0.0
        z: float
            z position defaults to 0.0
        """
        diffx = x - self.somapos[0]
        diffy = y - self.somapos[1]
        diffz = z - self.somapos[2]

        # also update the pt3d_pos:
        if self.pt3d and hasattr(self, 'x3d'):
            self._set_pt3d_pos(diffx, diffy, diffz)
        else:
            self.somapos[0] = x
            self.somapos[1] = y
            self.somapos[2] = z

            self.x[:, 0] += diffx
            self.y[:, 0] += diffy
            self.z[:, 0] += diffz

            self.x[:, -1] += diffx
            self.y[:, -1] += diffy
            self.z[:, -1] += diffz

        self.__update_synapse_positions()

    def cellpickler(self, filename, pickler=pickle.dump):
        """Save data in cell to filename, using cPickle. It will however
        destroy any ``neuron.h`` objects upon saving, as c-objects cannot be
        pickled

        Parameters
        ----------
        filename: str
            Where to save cell

        Examples
        --------
        >>> # To save a cell, issue:
        >>> cell.cellpickler('cell.cpickle')
        >>> # To load this cell again in another session:
        >>> import cPickle
        >>> with file('cell.cpickle', 'rb') as f:
        >>>     cell = cPickle.load(f)

        Returns
        -------
        None or pickle
        """
        self.strip_hoc_objects()
        if pickler == pickle.dump:
            filen = open(filename, 'wb')
            pickler(self, filen, protocol=2)
            filen.close()
            return None
        elif pickler == pickle.dumps:
            return pickle.dumps(self)

    def __update_synapse_positions(self):
        """
        Update synapse positions after rotation of morphology
        """
        for i in range(len(self.synapses)):
            self.synapses[i].update_pos(self)

    def set_rotation(self, x=None, y=None, z=None, rotation_order='xyz'):
        """
        Rotate geometry of cell object around the x-, y-, z-axis in the order
        described by rotation_order parameter.

        Parameters
        ----------
        x: float or None
            rotation angle in radians. Default: None
        y: float or None
            rotation angle in radians. Default: None
        z: float or None
            rotation angle in radians. Default: None
        rotation_order: str
            string with 3 elements containing x, y and z
            e.g. 'xyz', 'zyx'. Default: 'xyz'

        Examples
        --------
        >>> cell = LFPy.Cell(**kwargs)
        >>> rotation = {'x': 1.233, 'y': 0.236, 'z': np.pi}
        >>> cell.set_rotation(**rotation)
        """
        if not isinstance(rotation_order, str):
            raise AttributeError('rotation_order must be a string')
        elif not np.all([u in rotation_order for u in 'xyz']):
            raise AttributeError("'x', 'y', and 'z' must be in rotation_order")
        elif len(rotation_order) != 3:
            raise AttributeError(
                "rotation_order should have 3 elements (e.g. 'zyx')")

        for ax in rotation_order:
            if ax == 'x' and x is not None:
                theta = -x
                rotation_x = np.array([[1, 0, 0],
                                       [0, np.cos(theta), -np.sin(theta)],
                                       [0, np.sin(theta), np.cos(theta)]])

                rel_start, rel_end = self._rel_positions()

                rel_start = np.dot(rel_start, rotation_x)
                rel_end = np.dot(rel_end, rotation_x)

                self._real_positions(rel_start, rel_end)
                if self.verbose:
                    print(
                        'Rotated geometry %g radians around x-axis' %
                        (-theta))
            else:
                if self.verbose:
                    print('Geometry not rotated around x-axis')

            if ax == 'y' and y is not None:
                phi = -y
                rotation_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                                       [0, 1, 0],
                                       [-np.sin(phi), 0, np.cos(phi)]])

                rel_start, rel_end = self._rel_positions()

                rel_start = np.dot(rel_start, rotation_y)
                rel_end = np.dot(rel_end, rotation_y)

                self._real_positions(rel_start, rel_end)
                if self.verbose:
                    print('Rotated geometry %g radians around y-axis' % (-phi))
            else:
                if self.verbose:
                    print('Geometry not rotated around y-axis')

            if ax == 'z' and z is not None:
                gamma = -z
                rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                       [np.sin(gamma), np.cos(gamma), 0],
                                       [0, 0, 1]])

                rel_start, rel_end = self._rel_positions()

                rel_start = np.dot(rel_start, rotation_z)
                rel_end = np.dot(rel_end, rotation_z)

                self._real_positions(rel_start, rel_end)
                if self.verbose:
                    print(
                        'Rotated geometry %g radians around z-axis' %
                        (-gamma))
            else:
                if self.verbose:
                    print('Geometry not rotated around z-axis')

        # rotate the pt3d geometry accordingly
        if self.pt3d and hasattr(self, 'x3d'):
            self._set_pt3d_rotation(x, y, z, rotation_order)

    def chiral_morphology(self, axis='x'):
        """
        Mirror the morphology around given axis, (default x-axis),
        useful to introduce more heterogeneouties in morphology shapes

        Parameters
        ----------
        axis: str
            'x' or 'y' or 'z'
        """
        # morphology relative to soma-position
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
            print('morphology mirrored across %s-axis' % axis)

        # set the proper 3D positions
        self._real_positions(rel_start, rel_end)

    def _rel_positions(self):
        """
        Morphology relative to soma position
        """
        rel_start = np.array([self.x[:, 0] - self.somapos[0],
                              self.y[:, 0] - self.somapos[1],
                              self.z[:, 0] - self.somapos[2]]).T
        rel_end = np.array([self.x[:, -1] - self.somapos[0],
                            self.y[:, -1] - self.somapos[1],
                            self.z[:, -1] - self.somapos[2]]).T

        return rel_start, rel_end

    def _real_positions(self, rel_start, rel_end):
        """
        Morphology coordinates relative to Origo
        """
        self.x[:, 0] = rel_start[:, 0] + self.somapos[0]
        self.y[:, 0] = rel_start[:, 1] + self.somapos[1]
        self.z[:, 0] = rel_start[:, 2] + self.somapos[2]

        self.x[:, -1] = rel_end[:, 0] + self.somapos[0]
        self.y[:, -1] = rel_end[:, 1] + self.somapos[1]
        self.z[:, -1] = rel_end[:, 2] + self.somapos[2]

        self.__update_synapse_positions()

    def get_rand_prob_area_norm(self, section='allsec',
                                z_min=-10000, z_max=10000):
        """
        Return the probability (0-1) for synaptic coupling on segments
        in section sum(prob)=1 over all segments in section.
        Probability normalized by area.

        Parameters
        ----------
        section: str
            string matching a section-name. Defaults to 'allsec'
        z_min: float
            depth filter
        z_max: float
            depth filter

        Returns
        -------
        ndarray, dtype=float
        """
        idx = self.get_idx(section=section, z_min=z_min, z_max=z_max)
        prob = self.area[idx] / sum(self.area[idx])
        return prob

    def get_rand_prob_area_norm_from_idx(self, idx=np.array([0])):
        """
        Return the normalized probability (0-1) for synaptic coupling on
        segments in idx-array.
        Normalised probability determined by area of segments.

        Parameters
        ----------
        idx: ndarray, dtype=int.
            array of segment indices

        Returns
        -------
        ndarray, dtype=float
        """
        prob = self.area[idx] / sum(self.area[idx])
        return prob

    def get_intersegment_vector(self, idx0=0, idx1=0):
        """Return the distance between midpoints of two segments with index
        idx0 and idx1. The argument returned is a list [x, y, z], where
        x = self.x[idx1].mean(axis=-1) - self.x[idx0].mean(axis=-1) etc.

        Parameters
        ----------
        idx0: int
        idx1: int

        Returns
        -------
        list of floats
            distance between midpoints along x,y,z axis in µm
        """
        vector = []
        try:
            if idx1 < 0 or idx0 < 0:
                raise Exception('idx0 < 0 or idx1 < 0')
            vector.append(
                self.x[idx1].mean(axis=-1) -
                self.x[idx0].mean(axis=-1))
            vector.append(
                self.y[idx1].mean(axis=-1) -
                self.y[idx0].mean(axis=-1))
            vector.append(
                self.z[idx1].mean(axis=-1) -
                self.z[idx0].mean(axis=-1))
            return vector
        except BaseException:
            ERRMSG = 'idx0 and idx1 must be ints on [0, %i]' % self.totnsegs
            raise ValueError(ERRMSG)

    def get_intersegment_distance(self, idx0=0, idx1=0):
        """
        Return the Euclidean distance between midpoints of two segments.

        Parameters
        ----------
        idx0: int
        idx1: int

        Returns
        -------
        float
            distance (µm).
        """
        try:
            vector = np.array(self.get_intersegment_vector(idx0, idx1))
            return np.sqrt((vector**2).sum())
        except BaseException:
            ERRMSG = 'idx0 and idx1 must be ints on [0, %i]' % self.totnsegs
            raise ValueError(ERRMSG)

    def get_idx_children(self, parent="soma[0]"):
        """Get the idx of parent's children sections, i.e. compartments ids
        of sections connected to parent-argument

        Parameters
        ----------
        parent: str
            name-pattern matching a sectionname. Defaults to "soma[0]"

        Returns
        -------
        ndarray, dtype=int
        """
        idxvec = np.zeros(self.totnsegs)
        secnamelist = []
        childseclist = []
        # filling list of sectionnames for all sections, one entry per segment
        for sec in self.allseclist:
            for seg in sec:
                secnamelist.append(sec.name())
        if parent in secnamelist:
            # filling list of children section-names
            for sec in self.allseclist:
                if sec.name() == parent:
                    sref = neuron.h.SectionRef(sec=sec)
                    break
            assert sec.name() == parent == sref.sec.name()
            for sec in sref.child:
                childseclist.append(sec.name())
            # idxvec=1 where both coincide
            i = 0
            for sec in secnamelist:
                for childsec in childseclist:
                    if sec == childsec:
                        idxvec[i] += 1
                i += 1

            [idx] = np.where(idxvec)
            return idx
        else:
            return np.array([])

    def get_idx_parent_children(self, parent="soma[0]"):
        """
        Get all idx of segments of parent and children sections, i.e. segment
        idx of sections connected to parent-argument, and also of the parent
        segments

        Parameters
        ----------
        parent: str
            name-pattern matching a sectionname. Defaults to "soma[0]"

        Returns
        -------
        ndarray, dtype=int
        """
        seclist = [parent]
        for sec in self.allseclist:
            if sec.name() == parent:
                sref = neuron.h.SectionRef(sec=sec)
                break
        assert sref.sec.name() == parent
        for sec in sref.child:
            seclist.append(sec.name())

        return self.get_idx(section=seclist)

    def get_idx_name(self, idx=np.array([0], dtype=int)):
        '''
        Return NEURON convention name of segments with index idx.
        The returned argument is an array of tuples with corresponding
        segment idx, section name, and position along the section, like;
        [(0, 'neuron.h.soma[0]', 0.5),]

        Parameters
        ----------
        idx: ndarray, dtype int
            segment indices, must be between 0 and cell.totnsegs

        Returns
        -------
        ndarray, dtype=object
            tuples with section names of segments
        '''
        # ensure idx is array-like, or convert
        if isinstance(idx, int) or np.int64:
            idx = np.array([idx])
        elif len(idx) == 0:
            return
        else:
            idx = np.array(idx).astype(int)

        # ensure all idx are valid
        if np.any(idx >= self.totnsegs):
            wrongidx = idx[np.where(idx >= self.totnsegs)]
            raise Exception('idx %s >= number of compartments' % str(wrongidx))

        # create list of seg names:
        allsegnames = []
        segidx = 0
        for sec in self.allseclist:
            for seg in sec:
                allsegnames.append((segidx, '%s' % sec.name(), seg.x))
                segidx += 1

        return np.array(allsegnames, dtype=object)[idx][0]

    def _collect_pt3d(self):
        """collect the pt3d info, for each section"""
        x = []
        y = []
        z = []
        d = []

        for sec in self.allseclist:
            n3d = int(neuron.h.n3d(sec=sec))
            x_i, y_i, z_i = np.zeros(n3d), np.zeros(n3d), np.zeros(n3d),
            d_i = np.zeros(n3d)
            for i in range(n3d):
                x_i[i] = neuron.h.x3d(i, sec=sec)
                y_i[i] = neuron.h.y3d(i, sec=sec)
                z_i[i] = neuron.h.z3d(i, sec=sec)
                d_i[i] = neuron.h.diam3d(i, sec=sec)

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
            d.append(d_i)

        # remove offsets which may be present if soma is centred in Origo
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
        """
        update the locations in neuron.hoc.space using neuron.h.pt3dchange()
        """
        for i, sec in enumerate(self.allseclist):
            n3d = int(neuron.h.n3d(sec=sec))
            for n in range(n3d):
                neuron.h.pt3dchange(n,
                                    self.x3d[i][n],
                                    self.y3d[i][n],
                                    self.z3d[i][n],
                                    self.diam3d[i][n], sec=sec)
            # let NEURON know about the changes we just did:
            neuron.h.define_shape()
        # must recollect the geometry, otherwise we get roundoff errors!
        self._collect_geometry()

    def _set_pt3d_pos(self, diffx=0, diffy=0, diffz=0):
        """
        Offset pt3d geometry with differential displacement
        indicated in Cell.set_pos()
        """
        for i in range(len(self.x3d)):
            self.x3d[i] += diffx
            self.y3d[i] += diffy
            self.z3d[i] += diffz
        self._update_pt3d()

    def _set_pt3d_rotation(self, x=None, y=None, z=None, rotation_order='xyz'):
        """
        Rotate pt3d geometry of cell object around the x-, y-, z-axis
        in the order described by rotation_order parameter.
        rotation_order should be a string with 3 elements containing x, y and z
        e.g. 'xyz', 'zyx'
        Input should be angles in radians.
        using rotation matrices, takes dict with rot. angles,
        where x, y, z are the rotation angles around respective axes.
        All rotation angles are optional.

        Parameters
        ----------
        x: float
            rotation angle in radians
        y: float
            rotation angle in radians
        z: float
            rotation angle in radians
        rotation_order: str
            rotation order, default: 'xyz'

        Examples
        --------
        >>> cell = LFPy.Cell(**kwargs)
        >>> rotation = {'x': 1.233, 'y': 0.236, 'z': np.pi}
        >>> cell.set_pt3d_rotation(**rotation)
        """
        for ax in rotation_order:
            if ax == 'x' and x is not None:
                theta = -x
                rotation_x = np.array([[1, 0, 0],
                                       [0, np.cos(theta), -np.sin(theta)],
                                       [0, np.sin(theta), np.cos(theta)]])
                for i in range(len(self.x3d)):
                    rel_pos = self._rel_pt3d_positions(
                        self.x3d[i], self.y3d[i], self.z3d[i])

                    rel_pos = np.dot(rel_pos, rotation_x)

                    self.x3d[i], self.y3d[i], self.z3d[i] = \
                        self._real_pt3d_positions(rel_pos)
                if self.verbose:
                    print(('Rotated geometry %g radians around x-axis' %
                           (-theta)))
            else:
                if self.verbose:
                    print('Geometry not rotated around x-axis')

            if ax == 'y' and y is not None:
                phi = -y
                rotation_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                                       [0, 1, 0],
                                       [-np.sin(phi), 0, np.cos(phi)]])
                for i in range(len(self.x3d)):
                    rel_pos = self._rel_pt3d_positions(
                        self.x3d[i], self.y3d[i], self.z3d[i])

                    rel_pos = np.dot(rel_pos, rotation_y)

                    self.x3d[i], self.y3d[i], self.z3d[i] = \
                        self._real_pt3d_positions(rel_pos)
                if self.verbose:
                    print('Rotated geometry %g radians around y-axis' % (-phi))
            else:
                if self.verbose:
                    print('Geometry not rotated around y-axis')

            if ax == 'z' and z is not None:
                gamma = -z
                rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                       [np.sin(gamma), np.cos(gamma), 0],
                                       [0, 0, 1]])
                for i in range(len(self.x3d)):
                    rel_pos = self._rel_pt3d_positions(
                        self.x3d[i], self.y3d[i], self.z3d[i])

                    rel_pos = np.dot(rel_pos, rotation_z)

                    self.x3d[i], self.y3d[i], self.z3d[i] = \
                        self._real_pt3d_positions(rel_pos)
                if self.verbose:
                    print(
                        'Rotated geometry %g radians around z-axis' %
                        (-gamma))
            else:
                if self.verbose:
                    print('Geometry not rotated around z-axis')
        self._update_pt3d()

    def _rel_pt3d_positions(self, x, y, z):
        """Morphology relative to soma position """
        rel_pos = np.transpose(np.array([x - self.somapos[0],
                                         y - self.somapos[1],
                                         z - self.somapos[2]]))

        return rel_pos

    def _real_pt3d_positions(self, rel_pos):
        """Morphology coordinates relative to Origo """
        x = rel_pos[:, 0] + self.somapos[0]
        y = rel_pos[:, 1] + self.somapos[1]
        z = rel_pos[:, 2] + self.somapos[2]

        x = np.array(x).flatten()
        y = np.array(y).flatten()
        z = np.array(z).flatten()

        return x, y, z

    def _create_polygon(self, i, projection=('x', 'z')):
        """create a polygon to fill for each section"""
        x = getattr(self, projection[0] + '3d')[i]
        y = getattr(self, projection[1] + '3d')[i]
        # x = self.x3d[i]
        # z = self.z3d[i]
        d = self.diam3d[i]

        # calculate angles
        dx = np.diff(x)
        dy = np.diff(y)
        theta = np.arctan2(dy, dx)

        x = np.r_[x, x[::-1]]
        y = np.r_[y, y[::-1]]

        theta = np.r_[theta, theta[::-1]]
        d = np.r_[d, d[::-1]]

        # 1st corner:
        x[0] -= 0.5 * d[0] * np.sin(theta[0])
        y[0] += 0.5 * d[0] * np.cos(theta[0])

        # pt3d points between start and end of section, first side
        x[1:dx.size] -= 0.25 * d[1:dx.size] * (
            np.sin(theta[:dx.size - 1]) + np.sin(theta[1:dx.size]))
        y[1:dy.size] += 0.25 * d[1:dy.size] * (
            np.cos(theta[:dy.size - 1]) + np.cos(theta[1:dx.size]))

        # end of section, first side
        x[dx.size] -= 0.5 * d[dx.size] * np.sin(theta[dx.size])
        y[dy.size] += 0.5 * d[dy.size] * np.cos(theta[dy.size])

        # other side
        # end of section, second side
        x[dx.size + 1] += 0.5 * d[dx.size + 1] * np.sin(theta[dx.size])
        y[dy.size + 1] -= 0.5 * d[dy.size + 1] * np.cos(theta[dy.size])

        # pt3d points between start and end of section, second side
        x[::-1][1:dx.size] += 0.25 * d[::-1][1:dx.size] * (
            np.sin(theta[::-1][:dx.size - 1]) + np.sin(theta[::-1][1:dx.size]))
        y[::-1][1:dy.size] -= 0.25 * d[::-1][1:dy.size] * (
            np.cos(theta[::-1][:dy.size - 1]) + np.cos(theta[::-1][1:dx.size]))

        # last corner:
        x[-1] += 0.5 * d[-1] * np.sin(theta[-1])
        y[-1] -= 0.5 * d[-1] * np.cos(theta[-1])

        return x, y

    def get_pt3d_polygons(self, projection=('x', 'z')):
        """For each section create a polygon in the plane determined by keyword
        argument projection=('x', 'z'), that can be
        visualized using e.g., plt.fill()

        Parameters
        ----------
        projection: tuple of strings
            Determining projection. Defaults to ('x', 'z')

        Returns
        -------
        list
            list of (x, z) tuples giving the trajectory
            of each section that can be plotted using PolyCollection

        Examples
        --------
        >>> from matplotlib.collections import PolyCollection
        >>> import matplotlib.pyplot as plt
        >>> cell = LFPy.Cell(morphology='PATH/TO/MORPHOLOGY')
        >>> zips = []
        >>> for x, z in cell.get_pt3d_polygons(projection=('x', 'z')):
        >>>     zips.append(list(zip(x, z)))
        >>> polycol = PolyCollection(zips,
        >>>                          edgecolors='none',
        >>>                          facecolors='gray')
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.add_collection(polycol)
        >>> ax.axis(ax.axis('equal'))
        >>> plt.show()
        """
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
            raise ValueError(mssg)

        assert self.pt3d is True, 'Cell keyword argument pt3d != True'
        polygons = []
        for i in range(len(self.x3d)):
            polygons.append(self._create_polygon(i, projection))

        return polygons

    def _create_segment_polygon(self, i, projection=('x', 'z')):
        """Create a polygon to fill for segment i, in the plane
        determined by kwarg projection"""
        x = getattr(self, projection[0])[i]
        z = getattr(self, projection[1])[i]
        d = self.d[i]

        # calculate angles
        dx = np.diff(x)
        dz = np.diff(z)
        theta = np.arctan2(dz, dx)

        x = np.r_[x, x[::-1]]
        z = np.r_[z, z[::-1]]

        # 1st corner:
        x[0] -= 0.5 * d * np.sin(theta)
        z[0] += 0.5 * d * np.cos(theta)

        # end of section, first side
        x[1] -= 0.5 * d * np.sin(theta)
        z[1] += 0.5 * d * np.cos(theta)

        # other side
        # end of section, second side
        x[2] += 0.5 * d * np.sin(theta)
        z[2] -= 0.5 * d * np.cos(theta)

        # last corner:
        x[3] += 0.5 * d * np.sin(theta)
        z[3] -= 0.5 * d * np.cos(theta)

        return x, z

    def get_idx_polygons(self, projection=('x', 'z')):
        """For each segment idx in cell create a polygon in the plane
        determined by the projection kwarg (default ('x', 'z')),
        that can be visualized using plt.fill() or
        mpl.collections.PolyCollection

        Parameters
        ----------
        projection: tuple of strings
            Determining projection. Defaults to ('x', 'z')

        Returns
        -------
        polygons: list
            list of (ndarray, ndarray) tuples
            giving the trajectory of each section

        Examples
        --------
        >>> from matplotlib.collections import PolyCollection
        >>> import matplotlib.pyplot as plt
        >>> cell = LFPy.Cell(morphology='PATH/TO/MORPHOLOGY')
        >>> zips = []
        >>> for x, z in cell.get_idx_polygons(projection=('x', 'z')):
        >>>     zips.append(list(zip(x, z)))
        >>> polycol = PolyCollection(zips,
        >>>                          edgecolors='none',
        >>>                          facecolors='gray')
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.add_collection(polycol)
        >>> ax.axis(ax.axis('equal'))
        >>> plt.show()
        """
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
            raise ValueError(mssg)

        polygons = []
        for i in np.arange(self.totnsegs):
            polygons.append(self._create_segment_polygon(i, projection))

        return polygons

    def insert_v_ext(self, v_ext, t_ext):
        """Set external extracellular potential around cell.
        Playback of some extracellular potential v_ext on each cell.totnseg
        compartments. Assumes that the "extracellular"-mechanism is inserted
        on each compartment.
        Can be used to study ephaptic effects and similar
        The inputs will be copied and attached to the cell object as
        cell.v_ext, cell.t_ext, and converted
        to (list of) neuron.h.Vector types, to allow playback into each
        compartment e_extracellular reference.
        Can not be deleted prior to running cell.simulate()

        Parameters
        ----------
        v_ext: ndarray
            Numpy array of size cell.totnsegs x t_ext.size, unit mV
        t_ext: ndarray
            Time vector of v_ext in ms

        Examples
        --------
        >>> import LFPy
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> #create cell
        >>> cell = LFPy.Cell(morphology='morphologies/example_morphology.hoc',
        >>>                  passive=True)
        >>> #time vector and extracellular field for every segment:
        >>> t_ext = np.arange(cell.tstop / cell.dt+ 1) * cell.dt
        >>> v_ext = np.random.rand(cell.totnsegs, t_ext.size)-0.5
        >>> #insert potentials and record response:
        >>> cell.insert_v_ext(v_ext, t_ext)
        >>> cell.simulate(rec_imem=True, rec_vmem=True)
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(311)
        >>> ax2 = fig.add_subplot(312)
        >>> ax3 = fig.add_subplot(313)
        >>> eim = ax1.matshow(np.array(cell.v_ext), cmap='spectral')
        >>> cb1 = fig.colorbar(eim, ax=ax1)
        >>> cb1.set_label('v_ext')
        >>> ax1.axis(ax1.axis('tight'))
        >>> iim = ax2.matshow(cell.imem, cmap='spectral')
        >>> cb2 = fig.colorbar(iim, ax=ax2)
        >>> cb2.set_label('imem')
        >>> ax2.axis(ax2.axis('tight'))
        >>> vim = ax3.matshow(cell.vmem, cmap='spectral')
        >>> ax3.axis(ax3.axis('tight'))
        >>> cb3 = fig.colorbar(vim, ax=ax3)
        >>> cb3.set_label('vmem')
        >>> ax3.set_xlabel('tstep')
        >>> plt.show()
        """

        # test dimensions of input
        try:
            if v_ext.shape[0] != self.totnsegs:
                raise ValueError("v_ext.shape[0] != cell.totnsegs")
            if v_ext.shape[1] != t_ext.size:
                raise ValueError('v_ext.shape[1] != t_ext.size')
        except BaseException:
            raise ValueError('v_ext, t_ext must both be np.array types')

        if not self.extracellular:
            raise Exception('LFPy.Cell arg extracellular != True')

        # create list of extracellular potentials on each segment, time vector
        self.t_ext = neuron.h.Vector(t_ext)
        self.v_ext = []
        for v in v_ext:
            self.v_ext.append(neuron.h.Vector(v))
        # play v_ext into e_extracellular reference
        i = 0
        for sec in self.allseclist:
            for seg in sec:
                self.v_ext[i].play(seg._ref_e_extracellular, self.t_ext)
                i += 1

        return

    def get_axial_currents_from_vmem(self, timepoints=None):
        """Compute axial currents from cell sim: get current magnitude,
        distance vectors and position vectors.

        Parameters
        ----------
        timepoints: ndarray, dtype=int
            array of timepoints in simulation at which you want to compute
            the axial currents. Defaults to False. If not given,
            all simulation timesteps will be included.

        Returns
        -------
        i_axial: ndarray, dtype=float
            Shape ((cell.totnsegs-1)*2, len(timepoints)) array of axial current
            magnitudes I in units of (nA) in cell at all timesteps in
            timepoints, or at all timesteps of the simulation if
            timepoints=None.
            Contains two current magnitudes per segment,
            (except for the root segment): 1) the current from the mid point of
            the segment to the segment start point, and 2) the current from
            the segment start point to the mid point of the parent segment.
        d_vectors: ndarray, dtype=float
            Shape (3, (cell.totnsegs-1)*2) array of distance vectors traveled
            by each axial current in i_axial in units of (µm). The indices of
            the first axis, correspond to the first axis of i_axial and
            pos_vectors.
        pos_vectors: ndarray, dtype=float
            Shape ((cell.totnsegs-1)*2, 3) array of position vectors pointing
            to the mid point of each axial current in i_axial in units of (µm).
            The indices of the first axis, correspond to the first axis
            of i_axial and d_vectors.

        Raises
        ------
        AttributeError
            Raises an exeption if the cell.vmem attribute cannot be found
        """
        if not hasattr(self, 'vmem'):
            raise AttributeError('no vmem, run cell.simulate(rec_vmem=True)')
        self._ri_list = self.get_axial_resistance()
        i_axial = []
        d_vectors = []
        pos_vectors = []
        dseg = np.c_[self.x.mean(axis=-1) - self.x[:, 0],
                     self.y.mean(axis=-1) - self.y[:, 0],
                     self.z.mean(axis=-1) - self.z[:, 0]]
        dpar = np.c_[self.x[:, -1] - self.x.mean(axis=-1),
                     self.y[:, -1] - self.y.mean(axis=-1),
                     self.z[:, -1] - self.z.mean(axis=-1)]

        # children_dict = self.get_dict_of_children_idx()
        for sec in self.allseclist:
            if not neuron.h.SectionRef(sec=sec).has_parent():
                if sec.nseg == 1:
                    # skip soma, since soma is an orphan
                    continue
                else:
                    # the first segment has more than one segment,
                    # need to compute axial currents within this section.
                    seg_idx = 1
                    parent_idx = 0
                    bottom_seg = False
                    first_sec = True
                    branch = False
                    parentsec = None
                    children_dict = None
                    connection_dict = None
                    conn_point = 1
            else:
                # section has parent section
                first_sec = False
                bottom_seg = True
                secref = neuron.h.SectionRef(sec=sec)
                parentseg = secref.parent()
                parentsec = parentseg.sec
                children_dict = self.get_dict_of_children_idx()
                branch = len(children_dict[parentsec.name()]) > 1
                connection_dict = self.get_dict_parent_connections()
                conn_point = connection_dict[sec.name()]
                # find parent index
                if conn_point == 1 or parentsec.nseg == 1:
                    internal_parent_idx = -1  # last seg in sec
                elif conn_point == 0:
                    internal_parent_idx = 0  # first seg in sec
                else:
                    # if parentseg is not first or last seg in parentsec
                    segment_xlist = np.array(
                        [segment.x for segment in parentsec])
                    internal_parent_idx = np.abs(
                        segment_xlist - conn_point).argmin()
                parent_idx = self.get_idx(section=parentsec.name())[
                    internal_parent_idx]
                # find segment index
                seg_idx = self.get_idx(section=sec.name())[0]
            for _ in sec:
                if first_sec:
                    first_sec = False
                    continue
                iseg, ipar = self._parent_and_segment_current(seg_idx,
                                                              parent_idx,
                                                              bottom_seg,
                                                              branch,
                                                              parentsec,
                                                              children_dict,
                                                              connection_dict,
                                                              conn_point,
                                                              timepoints,
                                                              sec
                                                              )

                if bottom_seg:
                    # if a seg is connected to soma, it is
                    # connected to the middle of soma,
                    # and dpar needs to be altered.
                    par_dist = np.array([(self.x[seg_idx, 0] -
                                          self.x[parent_idx].mean(axis=-1)),
                                         (self.y[seg_idx, 0] -
                                          self.y[parent_idx].mean(axis=-1)),
                                         (self.z[seg_idx, 0] -
                                          self.z[parent_idx].mean(axis=-1))])

                else:
                    par_dist = dpar[parent_idx]
                d_vectors.append(par_dist)
                d_vectors.append(dseg[seg_idx])
                i_axial.append(ipar)
                i_axial.append(iseg)

                pos_par = np.array([self.x[seg_idx, 0],
                                    self.y[seg_idx, 0],
                                    self.z[seg_idx, 0]]) - 0.5 * par_dist

                pos_seg = np.array([self.x[seg_idx].mean(axis=-1),
                                    self.y[seg_idx].mean(axis=-1),
                                    self.z[seg_idx].mean(axis=-1)])
                pos_seg -= 0.5 * dseg[seg_idx]
                pos_vectors.append(pos_par)
                pos_vectors.append(pos_seg)

                parent_idx = seg_idx
                seg_idx += 1
                branch = False
                bottom_seg = False

        return np.array(i_axial), np.array(d_vectors).T, np.array(pos_vectors)

    def get_axial_resistance(self):
        """
        Return NEURON axial resistance for all cell compartments.

        Returns
        -------
        ri_list: ndarray, dtype=float
            Shape (cell.totnsegs, ) array containing neuron.h.ri(seg.x) in
            units of (MOhm) for all segments in cell calculated using the
            neuron.h.ri(seg.x) method. neuron.h.ri(seg.x) returns the
            axial resistance from the middle of the segment to the middle of
            the parent segment. Note: If seg is the first segment in a section,
            i.e. the parent segment belongs to a different section or there is
            no parent section, then neuron.h.ri(seg.x) returns the axial
            resistance from the middle of the segment to the node connecting
            the segment to the parent section (or a ghost node if there is no
            parent)
        """

        ri_list = np.zeros(self.totnsegs)
        comp = 0
        for sec in self.allseclist:
            for seg in sec:
                ri_list[comp] = neuron.h.ri(seg.x, sec=sec)
                comp += 1

        return ri_list

    def get_dict_of_children_idx(self):
        """
        Return dictionary with children segment indices for all sections.

        Returns
        -------
        children_dict: dictionary
            Dictionary containing a list for each section,
            with the segment index of all the section's children.
            The dictionary is needed to find the
            sibling of a segment.
        """
        children_dict = {}
        for sec in self.allseclist:
            children_dict[sec.name()] = []
            for child in neuron.h.SectionRef(sec=sec).child:
                # add index of first segment of each child
                children_dict[sec.name()].append(int(self.get_idx(
                    section=child.name())[0]))

        return children_dict

    def get_dict_parent_connections(self):
        """
        Return dictionary with parent connection point for all sections.

        Returns
        -------
        connection_dict: dictionary
            Dictionary containing a float in range [0, 1] for each section
            in cell. The float gives the location on the parent segment
            to which the section is connected.
            The dictionary is needed for computing axial currents.
        """
        connection_dict = {}
        for i, sec in enumerate(self.allseclist):
            connection_dict[sec.name()] = neuron.h.parent_connection(sec=sec)

        return connection_dict

    def _parent_and_segment_current(self, seg_idx, parent_idx, bottom_seg,
                                    branch=False, parentsec=None,
                                    children_dict=None, connection_dict=None,
                                    conn_point=1, timepoints=None, sec=None):
        """
        Return axial current from segment (seg_idx) mid to segment start,
        and current from parent segment (parent_idx) end to parent segment mid.

        Parameters
        ----------
        seg_idx: int
            Segment index
        parent_idx: int
            Parent index
        bottom_seg: boolean
        branch: boolean
        parentsec: neuron.Section object
            parent section
        children_dict: dict or None
            Default None
        connection_dict: dict or None
            Default None
        conn_point: float
            relative connection point on section in the interval [0, 1].
            Defaults to 1
        timepoints: ndarray, dtype=int
            array of timepoints in simulation at which you want to compute
            the axial currents. Defaults to None. If not given,
            the axial currents. Defaults to None. If not given,
            all simulation timesteps will be included.
        sec: neuron.Section object
            current section needed in new NEURON version

        Returns
        -------
        iseg: dtype=float
            Axial current in units of (nA)
            from segment mid point to segment start point.
        ipar: dtype=float
            Axial current in units of (nA)
            from parent segment end point to parent segment mid point.
        """
        # axial resistance between segment mid and parent node
        seg_ri = self._ri_list[seg_idx]
        vmem = self.vmem
        if timepoints is not None:
            vmem = self.vmem[:, timepoints]
        vpar = vmem[parent_idx]
        vseg = vmem[seg_idx]
        # if segment is the first in its section and it is connected to
        # top or bottom of parent section, we need to find parent_ri explicitly
        if bottom_seg and (conn_point == 0 or conn_point == 1):
            if conn_point == 0:
                parent_ri = self._ri_list[parent_idx]
            else:
                parent_ri = neuron.h.ri(0, sec=sec)
            if not branch:
                ri = parent_ri + seg_ri
                iseg = (vpar - vseg) / ri
                ipar = iseg
            else:
                # if branch, need to compute iseg and ipar separately
                [sib_idcs] = np.take(children_dict[parentsec.name()],
                                     np.where(children_dict[parentsec.name()]
                                              != seg_idx))
                sibs = [self.get_idx_name(sib_idcs)[i][1]
                        for i in range(len(sib_idcs))]
                # compute potential in branch point between parent and siblings
                v_branch_num = vpar / parent_ri + vseg / seg_ri
                v_branch_denom = 1. / parent_ri + 1. / seg_ri
                for sib_idx, sib in zip(sib_idcs, sibs):
                    sib_conn_point = connection_dict[sib]
                    if sib_conn_point == conn_point:
                        v_branch_num += vmem[sib_idx] / self._ri_list[sib_idx]
                        v_branch_denom += 1. / self._ri_list[sib_idx]
                v_branch = v_branch_num / v_branch_denom
                iseg = (v_branch - vseg) / seg_ri
                # set ipar=iseg
                # only fraction of total current into parent is added per
                # sibling
                ipar = iseg
        else:
            iseg = (vpar - vseg) / seg_ri
            ipar = iseg
        return iseg, ipar

    def distort_geometry(self, factor=0., axis='z', nu=0.0):
        """
        Distorts cellular morphology with a relative factor along a chosen axis
        preserving Poisson's ratio. A ratio nu=0.5 assumes uncompressible and
        isotropic media that embeds the cell. A ratio nu=0 will only affect
        geometry along the chosen axis. A ratio nu=-1 will isometrically scale
        the neuron geometry along each axis.
        This method does not affect the underlying cable properties of the
        cell, only predictions of extracellular measurements (by affecting the
        relative locations of sources representing the compartments).

        Parameters
        ----------
        factor: float
            relative compression/stretching factor of morphology. Default is 0
            (no compression/stretching). Positive values implies a compression
            along the chosen axis.
        axis: str
            which axis to apply compression/stretching. Default is "z".
        nu: float
            Poisson's ratio. Ratio between axial and transversal
            compression/stretching. Default is 0.
        """
        assert abs(factor) < 1., 'abs(factor) >= 1, factor must be in <-1, 1>'
        assert axis in ['x', 'y', 'z'], \
            'axis={} not "x", "y" or "z"'.format(axis)

        for pos, dir_ in zip(self.somapos, 'xyz'):
            geometry = np.c_[getattr(self, dir_)[:, 0],
                             getattr(self, dir_)[:, -1]]
            if dir_ == axis:
                geometry -= pos
                geometry *= (1. - factor)
                geometry += pos
            else:
                geometry -= pos
                geometry *= (1. + factor * nu)
                geometry += pos

            setattr(self, dir_, geometry)

        # recompute length of each segment
        self._set_length()

    def _set_length(self):
        '''callable method to (re)set length attribute'''
        self.length = np.sqrt(np.diff(self.x, axis=-1)**2 +
                              np.diff(self.y, axis=-1)**2 +
                              np.diff(self.z, axis=-1)**2).flatten()

    def _set_area(self):
        '''callable method to (re)set area attribute'''
        if self.d.ndim == 1:
            self.area = self.length * np.pi * self.d
        else:
            # Surface area of conical frusta
            # A = pi*(r1+r2)*sqrt((r1-r2)^2 + h^2)
            self.area = np.pi * self.d.sum(axis=-1) * \
                np.sqrt(np.diff(self.d, axis=-1)**2 + self.length**2)

    def get_multi_current_dipole_moments(self, timepoints=None):
        '''
        Return 3D current dipole moment vector and middle position vector
        from each axial current in space.

        Parameters
        ----------
        timepoints: ndarray, dtype=int or None
            array of timepoints at which you want to compute
            the current dipole moments. Defaults to None. If not given,
            all simulation timesteps will be included.

        Returns
        -------
        multi_dipoles: ndarray, dtype = float
            Shape (n_axial_currents, 3, n_timepoints) array
            containing the x-,y-,z-components of the current dipole moment
            from each axial current in cell, at all timepoints.
            The number of axial currents,
            n_axial_currents = (cell.totnsegs-1) * 2
            and the number of timepoints, n_timepoints = cell.tvec.size.
            The current dipole moments are given in units of (nA µm).
        pos_axial: ndarray, dtype = float
            Shape (n_axial_currents, 3) array containing the x-, y-, and
            z-components giving the mid position in space of each multi_dipole
            in units of (µm).

        Examples
        --------
        Get all current dipole moments and positions from all axial currents in
        a single neuron simulation:

        >>> import LFPy
        >>> import numpy as np
        >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=False)
        >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,1000),
        >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
        >>> syn.set_spike_times(np.mgrid[20:100:20])
        >>> cell.simulate(rec_vmem=True, rec_imem=False)
        >>> timepoints = np.array([1,2,3,4])
        >>> multi_dipoles, dipole_locs = cell.get_multi_current_dipole_moments(
        >>>     timepoints=timepoints)
        '''
        i_axial, d_axial, pos_axial = self.get_axial_currents_from_vmem(
            timepoints=timepoints)
        Ni, Nt = i_axial.shape

        multi_dipoles = np.zeros((Ni, 3, Nt))
        for i in range(Ni):
            multi_dipoles[i, ] = (i_axial[i][:, np.newaxis] * d_axial[:, i]).T

        return multi_dipoles, pos_axial
