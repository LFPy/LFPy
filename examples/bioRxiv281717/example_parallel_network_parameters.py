#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Defines parameters for simulation, used by example_parallel_network.py script

Copyright (C) 2018 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''
import matplotlib
import os
if 'DISPLAY' not in os.environ.keys():
    matplotlib.use('agg')
import os
import numpy as np
from scipy import stats
from glob import glob
import json
from parameters import ParameterSet
from mpi4py import MPI
import neuron
import sys
from urllib.request import urlopen
from example_parallel_network_methods import get_templatename, get_params, \
    get_syn_params
import LFPy

stringType = 'U'

# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# load some neuron-interface files needed for the EPFL cell types
neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")


#######################
# Functions
#######################
def get_pre_m_type(post):
    '''little helper function to return the most populuous excitatory m_type
    within the layer of m_type post, assuming this is representative for
    excitatory external connections onto postsynaptic cells '''
    if post.startswith('L23'):
        return 'L23_PC'
    elif post.startswith('L4'):
        return 'L4_PC'
    elif post.startswith('L5'):
        return 'L5_TTPC1'
    elif post.startswith('L6'):
        return 'L6_IPC'

#######################
# Parameters
#######################


# test mode (1 cell per pop, all-to-all connectivity)
TESTING = False

# Creating a NeuroTools.parameters.ParameterSet object for the main parameters
PSET = ParameterSet({})

# output file destination
if TESTING:
    PSET.OUTPUTPATH = 'example_parallel_network_output_testing'
else:
    PSET.OUTPUTPATH = 'example_parallel_network_output'

# input file paths
# PATHs to current cell-specific files and NMODL files
PSET.CWD = os.getcwd()
PSET.CELLPATH = 'hoc_combos_syn.1_0_10.allzips'
PSET.NMODL = 'hoc_combos_syn.1_0_10.allmods'


########################################################
# Simulation control
########################################################
PSET.dt = 2**-4  # simulation time step size (ms)
PSET.tstop = 1500.  # simulation duration (ms)
PSET.v_init = -77.  # membrane voltage(s) at t = 0 for all cells
PSET.celsius = 34.  # global temperature setting affecting active channels
PSET.TRANSIENT = 500.  # duration of startup transient

# population size scaling (multiplied with values in
# populationParams['POP_SIZE']):
PSET.POPSCALING = 1.

# global scaling of connection probabilities (to counteract POPSCALING)
PSET.CONNPROBSCALING = 1. / PSET.POPSCALING

# switch for fully connected network (do not use with large population sizes)
PSET.fully_connected = True if TESTING else False

# bool flag switching LFP calculations on or off (faster)
PSET.COMPUTE_LFP = True

# bool flag switching ECoG calculation on or off
PSET.COMPUTE_ECOG = PSET.COMPUTE_LFP

# bool flag switching on calculations of electric current dipole moments
# per population
PSET.COMPUTE_P = PSET.COMPUTE_LFP

# bool flag switching on calculations of contributions to the extracellular
# potential per population
PSET.rec_pop_contributions = PSET.COMPUTE_LFP

# downsample factor for timeseries plots
PSET.decimate_q = 10

# settings for filtered signals shown in plots (fc=100 Hz, lowpass)
PSET.filterargs = dict(
    N=2,
    Wn=100. *
    2. *
    PSET.dt /
    1000 *
    PSET.decimate_q,
    btype='lowpass')

# Base NetworkCell arguments, morphology and template specific args is
# defined below.
cellParams = {
    'passive': False,
    'nsegs_method': None,
    'v_init': PSET.v_init,
    'tstart': 0,
    'tstop': PSET.tstop,
    'dt': PSET.dt,
    'verbose': False,
    'extracellular': False,
    'delete_sections': False,
}


# some stimuli to activate the network
PSET.PointProcParams = {
    'idx': 0,
    'record_current': False,
    'pptype': 'IClamp',
    # 'amp' : 0.793, # amplitude parameter set later on
    'dur': 1E6,
    'delay': 0.,
}

# parameters for predicting extracellular potentials, specifying
# coordinates of electrodes and extracellular conductivity. So far only
# point contacts
PSET.electrodeParams = {
    'x': np.zeros(16),
    'y': np.zeros(16),
    'z': np.linspace(-1500, 0, 16)[::-1],
    'sigma': 0.3,
    'n': 50,
    'N': np.array([[0., 1., 0]] * 16),
    'r': 5.,
    'method': 'root_as_point',
}


# parameters for 4-sphere volume conductor model
# compute electrode positions on the outer radius for different angular offsets
_theta = np.linspace(-np.pi / 4, np.pi / 4, 9)
_x = 90000. * np.sin(_theta)
_y = np.zeros(_theta.size)
_z = 90000. * np.cos(_theta)
PSET.foursphereParams = {
    'radii': [79000., 80000., 85000., 90000.],  # shell radii
    'sigmas': [0.3, 1.5, 0.015, 0.3],  # shell conductivity
    'r_electrodes': np.c_[_x, _y, _z],  # contact coordinates
}


# Optional arguments to Network.simulate() for computing extracellular
# contribution from passive leak, membrane capactitance and synaptic currents
PSET.NetworkSimulateArgs = {
    'use_ipas': False,
    'use_icap': False,
    'use_isyn': False,
    'to_memory': True,
}

# layer thickness top to bottom L1-L6, Markram et al. 2015 Fig 3A.
PSET.layer_data = np.array([('L1', 165., -82.5),
                            ('L2', 149., -239.5),
                            ('L3', 353., -490.5),
                            ('L4', 190., -762.),
                            ('L5', 525, -1119.5),
                            ('L6', 700, -1732.)],
                           dtype=[('layer', '|{}2'.format(stringType)),
                                  ('thickness', float), ('center', float)])


# Define electrode geometry corresponding to an ECoG electrode, where contact
# points have a radius r, surface normal vectors N, and ECoG is calculated as
# the average LFP in n random points on each contact:
PSET.ecogParameters = {
    'sigma_S': 0.,        # CSF conductivity
    'sigma_T': 0.3,        # GM conductivity
    'sigma_G': 0.3,        # WM conductivity
    'h': PSET.layer_data['thickness'].sum(),
    'x': np.array([0.]),    # x,y,z-coordinates of electrode contacts
    'y': np.array([0.]),
    'z': np.array([0.]),
    # +PSET.layer_data[4]['thickness']/8,
    'z_shift': -PSET.layer_data['thickness'].sum(),
    'n': 500,
    'r': 250,  # ECoG radii are often 500-1000 um
    'N': np.array([[0., 0., 1.]]),
    'method': "pointsource",
}


# Main population parameters:
PSET.populationParameters = np.array([
    # Layer 4
    # Excitatory
    ('L4_PC', 'cAD', 'L4_PC_cADpyr230_1', 2674,
     dict(
         radius=210,
         loc=PSET.layer_data[3]['center'],
         scale=100.,
         cap=[
             1078.,
             97.]),
     dict(x=np.pi / 2, y=0.),
     ['dend', 'apic'],
     ['dend', 'apic'],
     0.125, 5.),
    # Inhibitory
    ('L4_LBC', 'dNAC', 'L4_LBC_dNAC222_1', 122,
     dict(
         radius=210,
         loc=PSET.layer_data[3]['center'],
         scale=100.,
         cap=[
             938.,
             670]),
     dict(x=np.pi / 2, y=0.),
     ['soma', 'dend', 'apic'],
     ['dend', 'apic'],
     0.125, 5.),

    # Layer 5
    # Excitatory
    ('L5_TTPC1', 'cAD', 'L5_TTPC1_cADpyr232_1', 2403,
     dict(
         radius=210,
         loc=PSET.layer_data[4]['center'],
         scale=125.,
         cap=[
                 719,
                 73.]),
     dict(x=np.pi / 2, y=0.),
     ['dend', 'apic'],
     ['dend', 'apic'],
     0.1, 5.),
    # Inhibitory
    ('L5_MC', 'bAC', 'L5_MC_bAC217_1', 395,
     dict(
         radius=210,
         loc=PSET.layer_data[4]['center'],
         scale=125.,
         cap=[
             378.,
             890]),
     dict(x=np.pi / 2, y=0.),
     ['soma', 'dend', 'apic'],
     ['dend', 'apic'],
     0.125, 5.),
],
    dtype=[('m_type', '|{}32'.format(stringType)),
           ('e_type', '|{}32'.format(stringType)),
           ('me_type', '|{}32'.format(stringType)
            ), ('POP_SIZE', 'i8'), ('pop_args', dict),
           ('rotation_args', dict), ('syn_section', list),
           ('extrinsic_input_section', list),
           ('extrinsic_input_density', 'f8'),
           ('extrinsic_input_frequency', 'f8')])
# column data:
# shortnames as used in pathway_*.json files
# names as used to denote individual cell types
# POP_SIZE :    number of neurons for each morphological type as given on
# https://bbp.epfl.ch/nmc-portal/microcircuit

# pop_args : dict,
#     radius, mean position (loc) and standard deviation (scale) of the soma
#     positions
# rotation_args : dict, default rotations around x and y axis applied to
# each cell in the population using LFPy.NetworkCell.set_rotation()
# method.

# syn_section : list
#     list of section names where outgoing connections from this population
#     are made onto postsynaptic neurons (i.e., no excitatory synapses on
#     somatic sections anywhere)
# extrinsic_input_density : density of extrinisc incoming connections in
#     units of [µm^-2]
# extrinsic_input_frequency : frequency of synapse activation in units of [Hz]

# TODO: Define only short names, pick random cell types or similar when
# creating populations. Column could be redone as
# [('m_type', '|U8'), ('e-type', '|U8')] and
# single cell objects picked from the glob('m+e type') on random

# # Override population sizes (for testing)
if TESTING:
    PSET.populationParameters['POP_SIZE'] = np.ones(
        PSET.populationParameters.size)


# Define a layer-specificity of connections L_YXL
# (see Hagen, Dahmen et al. (2016), Cereb Cortex) based on the anatomy of
# dendrites and axons. We here define this depth-dependence of synapse
# positioning as the product of total [soma + dendrite] length and
# total axon length in spatial bins corresponding to the thickness and
# boundaries of each layer. The products are normalized such that the sum of
# each column is 1, i.e., the sum of layer specificities of a connection
# between X and Y is 1.
PSET.L_YXL_m_types = {}
bins = np.r_[-PSET.layer_data['thickness'].cumsum()[::-1], 0]
for i, (y, Y, pop_args_Y, rotation_args_Y) in enumerate(zip(
        PSET.populationParameters['m_type'],
        PSET.populationParameters['me_type'],
        PSET.populationParameters['pop_args'],
        PSET.populationParameters['rotation_args'])):
    # create a container for the layer specificities of connections
    data = np.zeros((PSET.layer_data.size,
                     PSET.populationParameters.size))

    # find and load the corresponding morphology files into LFPy
    m_Y = glob(os.path.join(PSET.CELLPATH, Y, 'morphology', '*.asc'))[0]
    cell_Y = LFPy.Cell(morphology=m_Y)
    cell_Y.set_rotation(**rotation_args_Y)
    cell_Y.set_pos(z=pop_args_Y['loc'])

    # sum the total length of axon in each layer bin
    layerbounds = np.r_[0, -PSET.layer_data['thickness'].cumsum()]
    len_Y_sum = np.zeros(PSET.layer_data.size)
    for k in range(PSET.layer_data.size):
        len_Y_sum[k] = cell_Y.length[cell_Y.get_idx(
            ['soma', 'dend', 'apic'],
            z_min=layerbounds[k + 1],
            z_max=layerbounds[k])].sum()
    for j, (X, pop_args_X, rotation_args_X) in enumerate(zip(
            PSET.populationParameters['me_type'],
            PSET.populationParameters['pop_args'],
            PSET.populationParameters['rotation_args'])):
        m_X = glob(os.path.join(PSET.CELLPATH, X, 'morphology', '*.asc'))[0]
        cell_X = LFPy.Cell(morphology=m_X)
        cell_X.set_rotation(**rotation_args_X)
        cell_X.set_pos(z=pop_args_X['loc'])

        len_X_sum = np.zeros(PSET.layer_data.size)
        for k in range(PSET.layer_data.size):
            len_X_sum[k] = cell_X.length[cell_X.get_idx(
                'axon', z_min=layerbounds[k + 1], z_max=layerbounds[k])].sum()

        data[:, j] = np.sqrt(len_Y_sum * len_X_sum) / \
            np.sqrt(len_Y_sum * len_X_sum).sum()
    # fill in
    PSET.L_YXL_m_types[y] = data

# clean up namespace
del cell_X, cell_Y, len_X_sum, len_Y_sum, data


# Container for LFPy.NetworkCell class parameters (path to morphology file
# etc.)
PSET.cellParameters = dict()

##########################################################################
# Set up various files and folders such that single-cell models from BBP can
# be used, and extract some numbers from pathway .json files
##########################################################################

# TODO: Add automated download of cell models from EPFL microcircuit portal


# autodownload some json files with anatomical and pathway specific data
pathway_files = ['pathways_anatomy_factsheets_simplified.json',
                 'pathways_physiology_factsheets_simplified.json']
if RANK == 0:
    for fname in pathway_files:
        if not os.path.isfile(fname):
            u = urlopen(
                'https://bbp.epfl.ch/nmc-portal/documents/10184/7288948/' +
                fname)
            localFile = open(fname, 'w')
            localFile.write(u.read().decode('utf-8'))
            localFile.close()
            u.close()
COMM.Barrier()

# flag for cell template file to switch on (inactive) synapses
add_synapses = False

# load synapse file info for each cell type as structured arrays in dictionary
synapses_tsv_dtype = [
    ('synapse_id', int),
    ('pre_cell_id', int),
    ('pre_mtype', int),
    ('sectionlist_id', int),
    ('sectionlist_index', int),
    ('seg_x', float),
    ('synapse_type', int),
    ('dep', float),
    ('fac', float),
    ('use', float),
    ('tau_d', float),
    ('delay', float),
    ('weight', float)
]
synapses_tsv = {}


# attempt to set up a folder with all unique EPFL mechanism mod files,
# compile, and load them all in order to be able to load cells as
# LFPy.NetworkCell objects
if RANK == 0:
    if not os.path.isdir(PSET.NMODL):
        os.mkdir(PSET.NMODL)
        for NRN in PSET.populationParameters['me_type']:
            for nmodl in glob(os.path.join(
                    PSET.CELLPATH, NRN, 'mechanisms', '*.mod')):
                while not os.path.isfile(
                        os.path.join(PSET.NMODL, os.path.split(nmodl)[-1])):
                    os.system('cp {} {}'.format(nmodl,
                                                os.path.join(PSET.NMODL,
                                                             '.')))
        os.chdir(PSET.NMODL)
        # patch faulty ProbGABAAB_EMS.mod file (otherwise stochastic inhibitory
        # synapses will stay closed except at first activation)
        diff = '''319c319
<                 urand = scop_random(1)
---
>                 value = scop_random(1)
'''
        f = open('ProbGABAAB_EMS.patch', 'w')
        f.writelines(diff)
        f.close()
        os.system('patch ProbGABAAB_EMS.mod ProbGABAAB_EMS.patch')

        os.system('nrnivmodl')
        os.chdir(PSET.CWD)
COMM.Barrier()
neuron.load_mechanisms(PSET.NMODL)
os.chdir(PSET.CWD)


# Fill in dictionary of population-specific cell parameters
for NRN in PSET.populationParameters['me_type']:
    os.chdir(os.path.join(PSET.CWD, PSET.CELLPATH, NRN))

    # get the template name
    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    # get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    # get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    # get synapses template name
    f = open(os.path.join("synapses", "synapses.hoc"), 'r')
    synapses = get_templatename(f)
    f.close()

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, os.path.join('synapses', 'synapses.hoc'))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    # create parameter dictionaries specific for each cell type (population)
    PSET.cellParameters[NRN] = dict(list(dict(
        morphology=glob(os.path.join('morphology', '*'))[0],
        templatefile=os.path.join(NRN, 'template.hoc'),
        templatename=templatename,
        templateargs=1 if add_synapses else 0,
    ).items()) + list(cellParams.items()))


# load synapse and connectivity data. mtype_map is the same for all cell types
if sys.version < '3':
    with open(os.path.join('synapses', 'mtype_map.tsv')) as f:
        mtype_map = np.loadtxt(f,
                               dtype={'names': ('pre_mtype_id', 'pre_mtype'),
                                      'formats': ('i4', '{}9'.format(
                                          stringType))},
                               converters={1: lambda s: s.decode()})
else:
    with open(os.path.join('synapses', 'mtype_map.tsv'),
              encoding='us-ascii') as f:
        mtype_map = np.loadtxt(f,
                               dtype={'names': ('pre_mtype_id', 'pre_mtype'),
                                      'formats': ('i4', '{}9'.format(
                                          stringType))},
                               converters={1: lambda s: s.decode()})

os.chdir(PSET.CWD)


for name in PSET.populationParameters['m_type']:
    files = glob(
        os.path.join(
            PSET.CELLPATH,
            name + '*',
            'synapses',
            'synapses.tsv'))
    synapses_tsv[name] = np.array([], dtype=synapses_tsv_dtype)
    for f in files:
        synapses_tsv[name] = np.r_[
            synapses_tsv[name], np.loadtxt(
                f, dtype=synapses_tsv_dtype, skiprows=1)]


# Open pathway anatomy and physiology factsheet files and read out info
pathways_anatomy = dict()
pathways_physiology = dict()

f = open(pathway_files[0], 'r')
j = json.load(f)
for pre in PSET.populationParameters['m_type']:
    for post in PSET.populationParameters['m_type']:
        key = '{}:{}'.format(pre, post)
        try:
            pathways_anatomy[key] = j[key]
        except KeyError:
            # fill in dummy data, no synapses will be created
            print('no pathway anatomy data for connection {}'.format(key))
            if sys.version < '3':
                pathways_anatomy[key] = {
                    'common_neighbor_bias': 0,
                    'connection_probability': 0,
                    'mean_number_of_synapse_per_connection': 0,
                    'number_of_convergent_neuron_mean': 0,
                    'number_of_convergent_neuron_std': 0,
                    'number_of_divergent_neuron_mean': 0,
                    'number_of_divergent_neuron_std': 0,
                    'number_of_synapse_per_connection_std': 0,
                    'total_synapse_count': 0,
                }
            else:
                pathways_anatomy[key] = {
                    u'common_neighbor_bias': 0,
                    u'connection_probability': 0,
                    u'mean_number_of_synapse_per_connection': 0,
                    u'number_of_convergent_neuron_mean': 0,
                    u'number_of_convergent_neuron_std': 0,
                    u'number_of_divergent_neuron_mean': 0,
                    u'number_of_divergent_neuron_std': 0,
                    u'number_of_synapse_per_connection_std': 0,
                    u'total_synapse_count': 0,
                }
f.close()
j.clear()

f = open(pathway_files[1], 'r')
j = json.load(f)
for pre in PSET.populationParameters['m_type']:
    for post in PSET.populationParameters['m_type']:
        key = '{}:{}'.format(pre, post)
        try:
            pathways_physiology[key] = j[key]
        except KeyError:
            # fill in dummy data, no synapses will be created
            print('no pathway physiology data for connection {}'.format(key))
            if sys.version < '3':
                pathways_physiology[key] = {
                    'cv_psp_amplitude_mean': 3,
                    'cv_psp_amplitude_std': 0.95,
                    'd_mean': 360,
                    'd_std': 230,
                    'decay_mean': 9.8,
                    'decay_std': 6.7,
                    'epsp_mean': 1.6,
                    'epsp_std': 0.78,
                    'f_mean': 330,
                    'f_std': 240,
                    'failures_mean': 86,
                    'failures_std': 6.5,
                    'gsyn_mean': 0.3,
                    'gsyn_std': 0.11,
                    'latency_mean': 0.33,
                    'latency_std': 0.18,
                    'risetime_mean': 0.43,
                    'risetime_std': 0.47,
                    'space_clamp_correction_factor': 3.6,
                    'synapse_type': u'Excitatory, depressing',
                    'u_mean': 0.19,
                    'u_std': 0.23
                }
            else:
                pathways_physiology[key] = {
                    u'cv_psp_amplitude_mean': 3,
                    u'cv_psp_amplitude_std': 0.95,
                    u'd_mean': 360,
                    u'd_std': 230,
                    u'decay_mean': 9.8,
                    u'decay_std': 6.7,
                    u'epsp_mean': 1.6,
                    u'epsp_std': 0.78,
                    u'f_mean': 330,
                    u'f_std': 240,
                    u'failures_mean': 86,
                    u'failures_std': 6.5,
                    u'gsyn_mean': 0.3,
                    u'gsyn_std': 0.11,
                    u'latency_mean': 0.33,
                    u'latency_std': 0.18,
                    u'risetime_mean': 0.43,
                    u'risetime_std': 0.47,
                    u'space_clamp_correction_factor': 3.6,
                    u'synapse_type': u'Excitatory, depressing',
                    u'u_mean': 0.19,
                    u'u_std': 0.23
                }
f.close()
j.clear()


# get out stats for synapses and connections, temporary
syn_param_stats = get_syn_params(PSET.populationParameters['m_type'],
                                 PSET.populationParameters['me_type'],
                                 pathways_physiology, mtype_map, synapses_tsv)

del synapses_tsv  # not needed anymore.


###########################################################################
# Set up main connection parameters used by Network class instance methods
############################################################################

# Main connection parameters between pre and post-synaptic populations
# organized as dictionary of parameter lists between pre and postsynaptic
# populations:
if PSET.fully_connected:
    # fully connected network (no selfconnections)
    connprob = [[1] * PSET.populationParameters.size] * \
        PSET.populationParameters.size
else:
    connprob = get_params(PSET.populationParameters['m_type'],
                          pathways_anatomy,
                          'connection_probability',
                          # unit conversion % -> fraction
                          0.01 * PSET.CONNPROBSCALING)

PSET.connParams = dict(
    # connection probabilities between populations
    connprob=connprob,

    # synapse mechanisms
    syntypes=[[neuron.h.ProbAMPANMDA_EMS
               if syn_param_stats['{}:{}'.format(pre, post)
                                  ]['synapse_type'] >= 100 else
               neuron.h.ProbGABAAB_EMS
               for post in PSET.populationParameters['m_type']]
              for pre in PSET.populationParameters['m_type']],


    # synapse time constants and reversal potentials.
    # Use the mean/global EPFL synapse model parameters
    # (for now) as some connections appear to be missing in pathway files.
    synparams=[[dict(
        Use=syn_param_stats['{}:{}'.format(pre, post)]['Use_mean'],
        Dep=syn_param_stats['{}:{}'.format(pre, post)]['Dep_mean'],
        Fac=syn_param_stats['{}:{}'.format(pre, post)]['Fac_mean'],
        tau_r_AMPA=0.2,
        tau_d_AMPA=syn_param_stats['{}:{}'.format(pre, post)]['tau_d_mean'],
        tau_r_NMDA=0.29,
        tau_d_NMDA=43,
        e=0,
        mg=1,
        u0=0,
        synapseID=0,
        verboseLevel=0,
        NMDA_ratio=0.4  # this may take on several values in synconf.txt files,
                        # not accounted for here
    )
        if syn_param_stats['{}:{}'.format(pre, post)
                           ]['synapse_type'] >= 100 else
        dict(
        Use=syn_param_stats['{}:{}'.format(pre, post)]['Use_mean'],
        Dep=syn_param_stats['{}:{}'.format(pre, post)]['Dep_mean'],
        Fac=syn_param_stats['{}:{}'.format(pre, post)]['Fac_mean'],
        tau_r_GABAA=0.2,
        # from synapses.hoc: rng.lognormal(0.2, 0.1) (mean, variance)
        tau_d_GABAA=syn_param_stats['{}:{}'.format(pre, post)]['tau_d_mean'],
        tau_r_GABAB=3.5,
        tau_d_GABAB=260.9,
        e_GABAA=-80,
        e_GABAB=-75.8354,
        u0=0,
        synapseID=0,
        verboseLevel=0,
        GABAB_ratio=0.0,
        # this may take on several values, in synconf.txt files, not accounted
        # for here
    )
        for post in PSET.populationParameters['m_type']]
        for pre in PSET.populationParameters['m_type']],

    # maximum conductances
    weightfuns=[[np.random.normal] * PSET.populationParameters.size] * \
    PSET.populationParameters.size,
    weightargs=get_params(PSET.populationParameters['m_type'],
                          pathways_physiology,
                          ['gsyn_mean', 'gsyn_std'], 1.),
    # Correct??? (very small PSPs otherwise).
    # Also, weights in unknown units loaded from synapses_tsv is different
    # than the reported averaged gsyn.

    # connection delays
    delayfuns=[[np.random.normal] * PSET.populationParameters.size] * \
    PSET.populationParameters.size,
    delayargs=[[dict(
        loc=syn_param_stats['{}:{}'.format(pre, post)]['delay_mean'],
        scale=syn_param_stats['{}:{}'.format(pre, post)]['delay_std']
    ) for post in PSET.populationParameters['m_type']]
        for pre in PSET.populationParameters['m_type']],


    # numbers of synapses per connection
    multapsefuns=[[np.random.normal] \
                  * PSET.populationParameters.size] \
    * PSET.populationParameters.size,
    multapseargs=get_params(PSET.populationParameters['m_type'],
                            pathways_anatomy,
                            ['mean_number_of_synapse_per_connection',
                             'number_of_synapse_per_connection_std']),

    # parameters for finding random synapse locations using the method
    # LFPy.Cell.get_rand_idx_area_and_distribution_norm. The argument nidx is
    # default to 1
    syn_pos_args=[[dict(section=syn_section,
                        z_min=-1E6,
                        z_max=1E6,
                        fun=[stats.norm] * PSET.layer_data.size,
                        funargs=[dict(loc=loc, scale=scale / 2.)
                                 for loc, scale in PSET.layer_data[
                                 ['center', 'thickness']]],
                        funweights=PSET.L_YXL_m_types[post_m_type][:, i]
                        ) for i, pre_m_type in enumerate(
        PSET.populationParameters['m_type'])]
        for post_m_type, syn_section in PSET.populationParameters[
        ['m_type', 'syn_section']]],
)

# save connection data
PSET.save_connections = True

# connection parameters for synapses activated by putative external
# population(s)
PSET.connParamsExtrinsic = dict(
    # synapse type
    syntype='ProbAMPANMDA_EMS',
    # synapse parameters (assumes parameters of excitatory population in the
    # layer)
    synparams=[dict(
        Use=syn_param_stats['{}:{}'.format(
            get_pre_m_type(post), post)]['Use_mean'],
        Dep=syn_param_stats['{}:{}'.format(
            get_pre_m_type(post), post)]['Dep_mean'],
        Fac=syn_param_stats['{}:{}'.format(
            get_pre_m_type(post), post)]['Fac_mean'],
        tau_r_AMPA=0.2,
        tau_d_AMPA=syn_param_stats['{}:{}'.format(
            get_pre_m_type(post), post)]['tau_d_mean'],
        tau_r_NMDA=0.29,
        tau_d_NMDA=43,
        e=0,
        mg=1,
        u0=0,
        synapseID=0,
        verboseLevel=0,
        NMDA_ratio=0.4  # this may take on several values in synconf.txt files,
                        # not accounted for here
    ) for post in PSET.populationParameters['m_type']],

    # maximum conductances
    weightfuns=[np.random.normal] * PSET.populationParameters.size,
    weightargs=[get_params(np.array([m_type]), pathways_physiology,
                           ['gsyn_mean', 'gsyn_std'], 1.)[0][0]
                for m_type in PSET.populationParameters['m_type']],
)
