#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''plotting script for figure 7 and 8 in manuscript preprint for parallel
performance of LFPy

Copyright (C) 2018 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.'''
import os
import numpy as np
import example_parallel_network_parameterspace as ps
from example_parallel_network_parameters import PSET
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

color_cycle = [
    u'#1f77b4',
    u'#1f77b4',
    u'#ff7f0e',
    u'#ff7f0e',
    u'#2ca02c',
    u'#2ca02c',
    u'#d62728',
    u'#d62728',
]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle)
plt.rcParams.update({
    'axes.xmargin': 0.0,
    'axes.ymargin': 0.0,
})
fontsize = 14
titlesize = 16
legendsize = 12
plt.rcParams.update({
    'axes.xmargin': 0.0,
    'axes.ymargin': 0.0,
    'axes.labelsize': fontsize,
    'axes.titlesize': titlesize,
    'figure.titlesize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': legendsize,
})


def get_plot_array(x, shape=(3, -1)):
    return np.array(x).reshape(shape)


plt.close('all')

for (PS0, PS1), figname in zip(
        [['MPI', 'POP'], ['MPI5', 'POP5']], ['figure_7', 'figure_8']):

    # fig, axes = plt.subplots(2, 1, sharey=True, figsize=(8, 8))
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 6))
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.15, top=0.9)

    # plot scaling with MPI pool size and fixed network size
    ax = axes[0]

    for ls, COMPUTE_LFP in zip(['-o', ':o'], [True, False]):
        # containers
        MPISIZE = []
        # init = []
        params = []
        pops = []
        conn = []
        sim = []
        # save = []
        ps_ids = []

        for pset in ps.PSPACES[PS0].iter_inner():
            if pset.COMPUTE_LFP == COMPUTE_LFP:
                # get identifier
                ps_id = ps.get_unique_id(pset)

                # set reference network size
                PSET.populationParameters['POP_SIZE'] = np.array(
                    pset.POP_SIZE_REF)

                # create x-axis
                MPISIZE.append(pset.MPISIZE)

                try:
                    # load log file
                    keys = np.loadtxt(
                        os.path.join(
                            ps.OUTPUTDIR,
                            ps_id,
                            'log.txt'),
                        usecols=[0],
                        dtype=str)
                    values = np.loadtxt(
                        os.path.join(
                            ps.OUTPUTDIR,
                            ps_id,
                            'log.txt'),
                        usecols=[1])

                    # fill in lists, try-excepts for possibly failed
                    # simulations
                    data = dict(zip(keys, values))
                    try:
                        params.append(data['parameters'])
                    except KeyError:
                        params.append(np.nan)
                        print('missing param. time id {}'.format(ps_id))
                    try:
                        pops.append(data['population'])
                    except KeyError:
                        pops.append(np.nan)
                        print('missing pop. time id {}'.format(ps_id))
                    try:
                        conn.append(data['connections'])
                    except KeyError:
                        conn.append(np.nan)
                        print('missing conn. time id {}'.format(ps_id))
                    try:
                        sim.append(data['simulation'])
                    except KeyError:
                        sim.append(np.nan)
                        print('missing sim. time id {}'.format(ps_id))

                except IOError:
                    print('fail to load param id {}'.format(ps_id))
                    params.append(np.nan)
                    pops.append(np.nan)
                    conn.append(np.nan)
                    sim.append(np.nan)
                ps_ids.append(ps_id)

        shape = (-1, len(ps.PSPACES[PS0]['GLOBALSEED']))
        x = get_plot_array(MPISIZE, shape)[:, 0]

        y = get_plot_array(params, shape)
        ax.loglog(x, y.mean(axis=1), ls,
                  label='par.' +
                  ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                  lw=2, ms=8,
                  basex=2, basey=2,
                  )
        ax.errorbar(
            x,
            y.mean(
                axis=1),
            yerr=y.std(
                axis=1),
            fmt='none',
            capsize=4,
            label='_nolegend_',
            lw=2,
            ms=8)

        y = get_plot_array(pops, shape)
        ax.loglog(x, y.mean(axis=1), ls,
                  label='pop.' +
                  ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                  lw=2, ms=8,
                  basex=2, basey=2,
                  )
        ax.errorbar(
            x,
            y.mean(
                axis=1),
            yerr=y.std(
                axis=1),
            fmt='none',
            capsize=4,
            label='_nolegend_',
            lw=2,
            ms=8)

        y = get_plot_array(conn, shape)
        ax.loglog(x, y.mean(axis=1), ls,
                  label='conn.' +
                  ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                  lw=2, ms=8,
                  basex=2, basey=2,
                  )
        ax.errorbar(
            x,
            y.mean(
                axis=1),
            yerr=y.std(
                axis=1),
            fmt='none',
            capsize=4,
            label='_nolegend_',
            lw=2,
            ms=8)

        y = get_plot_array(sim, shape)
        ax.loglog(x, y.mean(axis=1), ls,
                  label='sim.' +
                  ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                  lw=2, ms=8,
                  basex=2, basey=2,
                  )
        ax.errorbar(
            x,
            y.mean(
                axis=1),
            yerr=y.std(
                axis=1),
            fmt='none',
            capsize=4,
            label='_nolegend_',
            lw=2,
            ms=8)

    ax.minorticks_off()
    ax.axis(ax.axis('tight'))
    ax.grid('on', which='major')

    ax.set_xticks(np.unique(MPISIZE))
    ax.set_xticklabels(np.unique(MPISIZE), rotation=90)
    ax.set_yticks([2.**x for x in range(16)])
    ax.set_yticklabels([2**x for x in range(16)])
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r'$N_\mathrm{MPI}$', labelpad=0)
    ax.set_ylabel('time (s)', labelpad=0)
    ax.set_title(r'$N_\mathrm{%s}^{(1)}=%i, N_\mathrm{%s}^{(1)}=%i$' %
                 (PSET.populationParameters['m_type'][0].replace('_', '\\_'),
                  PSET.populationParameters['POP_SIZE'][0],
                  PSET.populationParameters['m_type'][1].replace(
                     '_', r'\_'),
                  PSET.populationParameters['POP_SIZE'][1]))
    ax.text(-0.05, 1.05, 'A',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)
    axis = ax.axis()

    # plot scaling with population size and fixed MPI pool size
    for ls, COMPUTE_LFP in zip(['-', ':'], [True, False]):
        for axindex, marker, PRESERVE in zip(
                [1, 2], ['o', 'o'], ['total', 'indegree']):
            ax = axes[axindex]
            # containers
            POPSCALING = []
            # init = []
            params = []
            pops = []
            conn = []
            sim = []
            # save = []
            ps_ids = []

            for pset in ps.PSPACES[PS1].iter_inner():
                if pset.COMPUTE_LFP == COMPUTE_LFP \
                        and pset.PRESERVE == PRESERVE:
                    # get identifier
                    ps_id = ps.get_unique_id(pset)

                    # create x-axis
                    POPSCALING.append(pset.POPSCALING)

                    try:
                        # load log file
                        keys = np.loadtxt(
                            os.path.join(
                                ps.OUTPUTDIR,
                                ps_id,
                                'log.txt'),
                            usecols=[0],
                            dtype=str)
                        values = np.loadtxt(
                            os.path.join(
                                ps.OUTPUTDIR,
                                ps_id,
                                'log.txt'),
                            usecols=[1])

                        # fill in lists, try-excepts for possibly failed
                        # simulations
                        data = dict(zip(keys, values))
                        try:
                            params.append(data['parameters'])
                        except KeyError:
                            params.append(np.nan)
                            print('missing param. time id {}'.format(ps_id))
                        try:
                            pops.append(data['population'])
                        except KeyError:
                            pops.append(np.nan)
                            print('missing pop. time id {}'.format(ps_id))
                        try:
                            conn.append(data['connections'])
                        except KeyError:
                            conn.append(np.nan)
                            print('missing conn. time id {}'.format(ps_id))
                        try:
                            sim.append(data['simulation'])
                        except KeyError:
                            sim.append(np.nan)
                            print('missing sim. time id {}'.format(ps_id))
                    except IOError:
                        print('fail to load param id {}'.format(ps_id))
                        params.append(np.nan)
                        pops.append(np.nan)
                        conn.append(np.nan)
                        sim.append(np.nan)
                    ps_ids.append(ps_id)

            shape = (-1, len(ps.PSPACES[PS1]['GLOBALSEED']))
            x = get_plot_array(POPSCALING, shape)[:, 0]

            y = get_plot_array(params, shape)
            ax.loglog(x, y.mean(axis=1), ls, marker=marker,
                      label='par.' +
                      ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                      lw=2, ms=8,
                      basex=2, basey=2
                      )
            ax.errorbar(
                x,
                y.mean(
                    axis=1),
                yerr=y.std(
                    axis=1),
                fmt='none',
                capsize=4,
                label='_nolegend_',
                lw=2,
                ms=8)

            y = get_plot_array(pops, shape)
            ax.loglog(x, y.mean(axis=1), ls, marker=marker,
                      label='pop.' +
                      ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                      lw=2, ms=8,
                      basex=2, basey=2
                      )
            ax.errorbar(
                x,
                y.mean(
                    axis=1),
                yerr=y.std(
                    axis=1),
                fmt='none',
                capsize=4,
                label='_nolegend_',
                lw=2,
                ms=8)

            y = get_plot_array(conn, shape)
            ax.loglog(x, y.mean(axis=1), ls, marker=marker,
                      label='conn.' +
                      ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                      lw=2, ms=8,
                      basex=2, basey=2
                      )
            ax.errorbar(
                x,
                y.mean(
                    axis=1),
                yerr=y.std(
                    axis=1),
                fmt='none',
                capsize=4,
                label='_nolegend_',
                lw=2,
                ms=8)

            y = get_plot_array(sim, shape)
            ax.loglog(x, y.mean(axis=1), ls, marker=marker,
                      label='sim.' +
                      ', {} E.P'.format('w.' if COMPUTE_LFP else 'w.o.'),
                      lw=2, ms=8,
                      basex=2, basey=2
                      )
            ax.errorbar(
                x,
                y.mean(
                    axis=1),
                yerr=y.std(
                    axis=1),
                fmt='none',
                capsize=4,
                label='_nolegend_',
                lw=2,
                ms=8)

            ax.axis(ax.axis('tight'))
            ax.grid('on', which='major')

            ax.set_xticks(np.unique(POPSCALING))
            ax.set_xticklabels(np.unique(POPSCALING), rotation=90)

            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())

            ax.set_xlabel(r'relative network size $b$', labelpad=10)
            ax.set_title(
                r'$N_\mathrm{%s}^{(1)}=%i, N_\mathrm{%s}^{(1)}=%i$,' %
                (PSET.populationParameters['m_type'][0].replace(
                    '_',
                    '\\_'),
                    PSET.populationParameters['POP_SIZE'][0],
                    PSET.populationParameters['m_type'][1].replace(
                    '_',
                    r'\_'),
                    PSET.populationParameters['POP_SIZE'][1],
                 ) +
                '\n' +
                r'$N_\mathrm{MPI}=%i$, ' % pset.MPISIZE
                + '$k_{YX}^{(b)}=%sK_{YX}^{(1)}/N_Y^{(1)}$' %
                'r' if PRESERVE == 'indegree' else '')
            ax.text(-0.05, 1.05, 'BC'[axindex - 1],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=ax.transAxes)
            axis = ax.axis()

        ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', borderaxespad=0.)

    fig.savefig(figname + '.pdf', bbox_inches='tight')
