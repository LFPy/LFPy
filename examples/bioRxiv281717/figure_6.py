#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''plotting script for figure 6 in manuscript preprint on output of
example_parallel_network.py

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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import h5py
import example_parallel_network_plotting as plotting
from mpi4py import MPI

# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

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

if __name__ == '__main__':
    # get simulation parameters
    from example_parallel_network_parameters import PSET

    # cell type colors
    colors = [
        plt.get_cmap(
            'Set1',
            PSET.populationParameters.size)(i) for i in range(
            PSET.populationParameters.size)]

    # time shown
    T = (PSET.TRANSIENT, PSET.TRANSIENT + 1000.)

    # Set up figure and subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(15, 5, left=0.075, right=0.975,
                  top=0.95, bottom=0.05, wspace=0.3, hspace=0.2)
    alphabet = 'ABCDEFGHIJKLMNOPQ'

    for j, (m_type, me_type) in enumerate(
            zip(PSET.populationParameters['m_type'],
                PSET.populationParameters['me_type'])):
        ax = fig.add_subplot(gs[:8, j])
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')
        for data, title, color in zip(
            [f['SUMMED_OUTPUT'][()][me_type]],
            [m_type],
                ['k']):
            ax.set_title(title)
            vlimround = plotting.draw_lineplot(
                ax=ax,
                data=plotting.decimate(data, q=PSET.decimate_q),
                dt=PSET.dt * PSET.decimate_q,
                T=T, color=color,
                scalebarbasis='log10')

        if j > 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')

        ax.text(-0.1, 1.05, alphabet[j],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16, fontweight='demibold',
                transform=ax.transAxes)

        p = f['CURRENT_DIPOLE_MOMENT'][me_type] * \
            1E-3  # nA um -> 1E-3 nA m unit conversion
        for i, (u, ls, lw, ylbl) in enumerate(zip(
            ['x', 'y', 'z'], ['-', '-', '-'], [1, 1, 1],
            [r'$\mathbf{p \cdot \hat{x}}$' + '\n' + r'($10^{-3}$ nA m)',
             r'$\mathbf{p \cdot \hat{y}}$' +
             '\n' + r'($10^{-3}$ nA m)',
             r'$\mathbf{p \cdot \hat{z}}$' + '\n' + r'($10^{-3}$ nA m)'])):
            ax = fig.add_subplot(gs[9 + i * 2:11 + i * 2, j])
            if i == 0:
                ax.text(-0.1, 1.2, alphabet[j + 5],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=16, fontweight='demibold',
                        transform=ax.transAxes)

            plotting.remove_axis_junk(ax)
            x = plotting.decimate(p[i, ], q=PSET.decimate_q)
            t = np.arange(x.size) * PSET.dt * PSET.decimate_q
            inds = (t >= T[0]) & (t <= T[1])
            ax.plot(t[inds], x[inds], ls=ls, lw=lw,
                    color='k')
            if i != 2:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ylbl, labelpad=0)

        f.close()

        ax.set_xlabel('time (ms)', labelpad=0)

    ax = fig.add_subplot(gs[:8, 4])
    ax.set_title('signal variance')
    y = PSET.electrodeParams['z']
    tind = int(PSET.TRANSIENT / PSET.dt)
    f = h5py.File(
        os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_output.h5'),
        'r')

    for m_type, me_type, color in zip(
        list(
            PSET.populationParameters['m_type']) + ['summed'], list(
            PSET.populationParameters['me_type']) + ['imem'], colors + ['k']):
        data = f['SUMMED_OUTPUT'][()][me_type]
        ax.semilogx(data[:, tind:].var(axis=1), y,
                    lw=2, label=m_type, color=color)
    f.close()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylabel('')
    plotting.remove_axis_junk(ax)
    ax.axis(ax.axis('tight'))
    ax.legend(loc='best')
    ax.set_xlabel(r'variance (mV$^2$)', labelpad=0)
    ax.text(-0.1, 1.05, alphabet[4],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    barhaxes = []
    barhmax = []
    barhmin = []
    for i, (u, ls, lw, ylbl) in enumerate(zip(['x', 'y', 'z'], [
            '-', '-', '-'], [1, 1, 1], [r'$p_x$', r'$p_y$', r'$p_z$'])):
        ax = fig.add_subplot(gs[9 + i * 2:11 + i * 2, 4])
        if i == 0:
            ax.text(-0.1, 1.2, alphabet[9],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=ax.transAxes)
        plotting.remove_axis_junk(ax)
        barhaxes.append(ax)
        f = h5py.File(os.path.join(PSET.OUTPUTPATH,
                                   'example_parallel_network_output.h5'), 'r')
        bars = []
        p_temp = np.zeros(f['CURRENT_DIPOLE_MOMENT'].shape)
        for me_type in PSET.populationParameters['me_type']:
            # nA um -> 1E-3 nA m unit conversion
            bars.append((f['CURRENT_DIPOLE_MOMENT']
                         [me_type][i, tind:] * 1E-3).var())
            p_temp += f['CURRENT_DIPOLE_MOMENT'][me_type]
        f.close()
        p_temp *= 1E-6  # nA um -> nA m unit conversion
        bars.append(p_temp[i, tind:].var())
        barhmax.append(np.array(bars).max())
        barhmin.append(np.array(bars).min())
        del p_temp
        rects = ax.barh(range(len(bars)), bars, log=True, color=colors + ['k'])
        if i != 2:
            ax.set_xticklabels([])
        ax.set_yticks([])

        if i == 0:
            for xpos, ypos, text in zip(bars, range(len(bars)), list(
                    PSET.populationParameters['m_type']) + ['summed']):
                ax.text(xpos, ypos, text, ha='left', va='center')

    ax.set_xlabel(r'variance (($10^{-3}$ nA m)$^2$)', labelpad=0)
    for axh in barhaxes:
        ax.axis(ax.axis('tight'))
        axh.set_xlim(left=np.min(barhmin), right=np.max(barhmax))

    fig.savefig(
        os.path.join(
            PSET.OUTPUTPATH,
            'figure_6.pdf'),
        bbox_inches='tight')

    plt.show()
