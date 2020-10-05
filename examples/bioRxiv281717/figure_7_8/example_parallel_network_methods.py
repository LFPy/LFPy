#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Defines classes and methods used by example_parallel_network.py script

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
import numpy as np
import json


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print('template {} found!'.format(templatename))
            continue

    return templatename


def get_params(names, dictionary, keys, scaling=1.):
    '''
    generic function to extract data from e.g., pathways_anatomy
    dictionary object
    '''
    out = []
    for i, pre in enumerate(names):
        out.append([])
        for j, post in enumerate(names):
            out[i].append([])
            if isinstance(keys, str):  # each entry just the value
                out[i][j] = dictionary['{}:{}'.format(
                    pre, post)][keys] * scaling
            elif isinstance(keys, list):  # each entry a dict w. keys loc scale
                assert len(keys) == 2
                out[i][j] = dict()
                for key, entry in zip(['loc', 'scale'], keys):
                    out[i][j][key] = dictionary['{}:{}'.format(
                        pre, post)][entry] * scaling
    return out


def get_syn_params(shortnames, names, pathways_physiology,
                   mtype_map, synapses_tsv):
    '''
    Extract from pathways_physiology and synapses_tsv connection stats
    between pre and postsynaptic populations for use with synapse model
    '''
    out = {}
    for i, (pre, pre_l) in enumerate(zip(shortnames, names)):
        for j, (post, post_l) in enumerate(zip(shortnames, names)):
            phys = pathways_physiology['{}:{}'.format(pre, post)]

            # use just the averaged values
            out['{}:{}'.format(pre, post)] = dict(Dep_mean=phys['d_mean'],
                                                  Dep_std=phys['d_std'],
                                                  Fac_mean=phys['f_mean'],
                                                  Fac_std=phys['f_std'],
                                                  )

            pre_mtype = mtype_map['pre_mtype_id'][mtype_map['pre_mtype']
                                                  == pre]
            data = synapses_tsv[post][synapses_tsv[post]
                                      ['pre_mtype'] == pre_mtype]

            if not isinstance(data, np.void):
                if data.size > 1:
                    out['{}:{}'.format(pre, post)].update(dict(
                        Use_mean=data['use'].mean(),
                        Use_std=data['use'].std(),
                        tau_d_mean=data['tau_d'].mean(),
                        tau_d_std=data['tau_d'].std(),
                        delay_mean=data['delay'].mean(),
                        delay_std=data['delay'].std(),
                        weight_mean=data['weight'].mean(),
                        weight_std=data['weight'].std(),
                        # I'm only gonna differentiate between excitatory
                        synapse_type=data['synapse_type'][0],
                        # (>= 100) and inhibitory (<100) connections.
                        # Connections can't be both.
                    ))
                elif data.size == 1:
                    out['{}:{}'.format(pre, post)].update(dict(
                        Use_mean=data['use'],
                        Use_std=data['use'],
                        tau_d_mean=data['tau_d'],
                        tau_d_std=data['tau_d'],
                        delay_mean=data['delay'],
                        delay_std=data['delay'],
                        weight_mean=data['weight'],
                        weight_std=data['weight'],
                        synapse_type=data['synapse_type']
                    ))
                # presumably no connections are gonna be made anyway (bc. conn.
                # prob. better be 0)
                else:
                    out['{}:{}'.format(pre, post)].update(dict(
                        Use_mean=0,
                        Use_std=0,
                        tau_d_mean=0,
                        tau_d_std=0,
                        delay_mean=0,
                        delay_std=0,
                        weight_mean=0,
                        weight_std=0,
                        synapse_type=0
                    ))
            else:
                if data.size == 1:
                    out['{}:{}'.format(pre, post)].update(dict(
                        Use_mean=data['use'],
                        Use_std=data['use'],
                        tau_d_mean=data['tau_d'],
                        tau_d_std=data['tau_d'],
                        delay_mean=data['delay'],
                        delay_std=data['delay'],
                        weight_mean=data['weight'],
                        weight_std=data['weight'],
                        synapse_type=data['synapse_type']
                    ))
                else:
                    out['{}:{}'.format(pre, post)].update(dict(
                        Use_mean=0,
                        Use_std=0.1,
                        tau_d_mean=0,
                        tau_d_std=0.1,
                        delay_mean=1.5,
                        delay_std=0.1,
                        weight_mean=0,
                        weight_std=0.1,
                        synapse_type=0
                    ))
    return out


def get_L_yXL(fname, y, x_in_X, L, fill_empty_columns=True):
    '''
    compute the layer specificity, defined as: L_yXL = k_yXL / k_yX where
    k_yXL is the mean number of connections from presynaptic population X onto
    a cell in population y.

    Parameters
    ----------
    fname : path
    y : str
        presynaptic cell type name
    x_in_X : list of list of str
        map postsynaptic cell type to postsynaptic population type
    L : list of str
        layer names
    fill_empty_columns : bool
        if True, empty columns is detected, and the value in the matrix
        corresponding to that column and "home" layer of the cell type wil be
        set to 1 (incoming connections may be assigned to that layer)
    '''
    def _get_L_yXL_per_yXL(fname, x_in_X, X_index,
                           y, layer):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()

        # Get number of synapses
        if layer in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
            # init variables
            k_yXL = 0
            k_yX = 0

            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][layer][x] / 100.
                k_yL = data['data'][y]['syn_dict'][layer][
                    'number of synapses per neuron']
                k_yXL += p_yxL * k_yL

            for ll in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
                for x in x_in_X[X_index]:
                    p_yxL = data['data'][y]['syn_dict'][ll][x] / 100.
                    k_yL = data['data'][y]['syn_dict'][ll][
                        'number of synapses per neuron']
                    k_yX += p_yxL * k_yL

            if k_yXL != 0.:
                return k_yXL / k_yX
            else:
                return 0.
        else:
            return 0.

    # init dict
    L_yXL = {}

    # iterate over postsynaptic cell types
    for y_value in y:
        # container
        data = np.zeros((len(L), len(x_in_X)))
        # iterate over lamina
        for i, Li in enumerate(L):
            # iterate over presynapse population inds
            for j in range(len(x_in_X)):
                data[i][j] = _get_L_yXL_per_yXL(fname, x_in_X,
                                                X_index=j,
                                                y=y_value,
                                                layer=Li)

        # check if values should be assigned in empty columns
        if fill_empty_columns:
            mapping = (data != 0)
            for j, col in enumerate(mapping.sum(axis=0)):
                if col == 0:
                    for i, Li in enumerate(L):
                        if Li in y_value[:4]:
                            data[i, j] = 1.
        L_yXL[y_value] = data

    return L_yXL
