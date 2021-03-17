#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''plotting script for figure 4 in manuscript preprint on output of
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
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
import os
import numpy as np
from copy import copy
import h5py
from LFPy import NetworkCell
import neuron
import example_parallel_network_plotting as plotting
from mpi4py import MPI

# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# # set default plotting parameters
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


def plot_quantity_YXL(axes, PSET, quantity,
                      y=['p23', 'b23', 'nb23', 'p4', 'ss4(L23)', 'ss4(L4)',
                         'b4', 'nb4', 'p5(L23)', 'p5(L56)', 'b5', 'nb5',
                         'p6(L4)', 'p6(L56)', 'b6', 'nb6'],
                      label=r'$\mathcal{L}_{YXL}$',
                      layers=['L1', 'L2/3', 'L4', 'L5', 'L6'],
                      cmap=plt.get_cmap('inferno')):
    '''make a bunch of image plots, each showing the spatial normalized
    connectivity of synapses'''
    if np.array(axes).ndim == 1:
        nrows = 1
        ncols = np.array(axes).size
    else:
        (nrows, ncols) = np.array(axes).shape

    # assess vlims
    vmin = 0
    vmax = 0
    for yi in y:
        if quantity[yi].max() > vmax:
            vmax = quantity[yi].max()

    for i, yi in enumerate(y):
        ax = np.array(axes).flatten()[i]

        masked_array = np.ma.array(quantity[yi], mask=quantity[yi] == 0)

        im = ax.pcolormesh(masked_array,
                           vmin=vmin, vmax=vmax,
                           cmap=cmap,
                           )
        ax.invert_yaxis()

        ax.axis(ax.axis('tight'))
        ax.set_xticks(np.arange(len(y)) + 0.5)
        ax.set_yticks(np.arange(len(layers)) + 0.5)

        if i % ncols == 0:
            ax.set_yticklabels(layers, )
            ax.set_ylabel('$L$', labelpad=0.)
        else:
            ax.set_yticklabels([])
        if i < ncols:
            ax.set_xlabel(r'$X$', labelpad=-1)
            ax.set_xticklabels(y, rotation=90)
        else:
            ax.set_xticklabels([])

        ax.set_title(r'$Y=${}'.format(yi))

        # colorbar
        if (i // ncols == 0) and (i % ncols) == ncols - 1:
            rect = np.array(ax.get_position().bounds)
            rect[0] += rect[2] + 0.0025
            rect[2] = 0.005
            cax = fig.add_axes(rect)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(label, labelpad=0)


def plot_multapseargs(ax, PSET, cmap=plt.get_cmap('inferno'),
                      data='multapseargs',
                      cbarlabel=r'$\overline{n}_\mathrm{syn}$',
                      key='loc'):
    '''make an imshow of connectivity parameters'''
    item = PSET.connParams[data]
    array = [[i[key] for i in j] for j in item]

    masked_array = np.ma.array(array, mask=np.array(array) == 0.)
    cmap = copy(cmap)
    cmap.set_bad('k', 0.5)
    im = ax.pcolormesh(masked_array, cmap=cmap, vmin=0, )
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(PSET.populationParameters.size) + 0.5)
    ax.set_yticks(np.arange(PSET.populationParameters.size) + 0.5)
    ax.set_xticklabels(PSET.populationParameters['m_type'], rotation=270)
    ax.set_yticklabels(PSET.populationParameters['m_type'], )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'$Y$', labelpad=0)
    ax.set_ylabel(r'$X$', labelpad=0, rotation=0)

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.0025
    rect[2] = 0.005
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbarlabel, labelpad=0)


if __name__ == '__main__':
    # get simulation parameters
    from example_parallel_network_parameters import PSET

    # cell type colors
    colors = [
        plt.get_cmap(
            'Set1',
            PSET.populationParameters.size)(i) for i in range(
            PSET.populationParameters.size)]

    # Set up figure and subplots
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(left=0.075, right=0.95, hspace=0.2, wspace=0.7,
                        bottom=0.05, top=0.95)
    gs = GridSpec(16, 40, hspace=0.2, wspace=0.5)
    alphabet = 'ABCDEFG'
    dz = 50  # spatial resolution of histograms

    # PANEL A. Morphology overview
    ax = fig.add_subplot(gs[:12, :12])
    plotting.remove_axis_junk(ax, lines=['top', 'bottom', 'right'])
    n_segs, areas = plotting.plot_m_types(ax, PSET, colors, spacing=300.)
    ax.set_title('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel('layer')
    ax.text(-0.05, 1.05, alphabet[0],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)
    axis = ax.axis()

    ax = fig.add_subplot(gs[12:, :12])
    ax.axis('off')
    ax.set_xlim(left=axis[0], right=axis[1])

    # # table data with population sizes etc.
    h_spacing = 300.
    v_spacing = 100.
    keys = ['m_type', 'm_type', 'e_type', 'me_type', 'n_seg', 'POP_SIZE',
            'F_y', 'extrinsic_input_density', 'extrinsic_input_frequency',
            'pop_args[loc]', 'pop_args[scale]']
    labels = ['population ($X/Y$):', 'morphology type (m):',
              'electric type (e):', 'cell model #:',
              'segment count $n_\\mathrm{seg}$:', 'population count $N_X$:',
              r'Occurrence $F_X$:', r'$n_\mathrm{ext}$:',
              r'$\nu_\mathrm{ext}$ (s$^{-1}$):',
              r'$\overline{z}_X^\mathrm{soma}$ ($\mu$m):',
              r'$\sigma_{\overline{z},X}^\mathrm{soma}$ ($\mu$m)']
    # PSET.populationParameters
    z = -PSET.layer_data['thickness'].sum()
    ax.set_ylim(
        top=z -
        v_spacing,
        bottom=z -
        len(keys) *
        v_spacing +
        v_spacing)
    for i, (label, key) in enumerate(zip(labels, keys)):
        ax.text(axis[0], z - (i + 1) * v_spacing, label, ha='right', va='top')
        for j, popData in enumerate(PSET.populationParameters):
            if key == 'pop_args[loc]':
                ax.text(j * h_spacing, z - (i + 1) * v_spacing,
                        int(popData['pop_args']['loc']), ha='center', va='top')
            elif key == 'pop_args[scale]':
                ax.text(j * h_spacing,
                        z - (i + 1) * v_spacing,
                        int(popData['pop_args']['scale']),
                        ha='center',
                        va='top')
            elif key == 'me_type':
                ax.text(j *
                        h_spacing, z -
                        (i +
                         1) *
                        v_spacing, popData[key].lstrip(popData['m_type'] +
                                                       '_' +
                                                       popData['e_type']),
                        ha='center', va='top')
            elif key == 'n_seg':
                ax.text(j * h_spacing, z - (i + 1) * v_spacing,
                        n_segs[j], ha='center', va='top')
            elif key == 'F_y':
                ax.text(
                    j *
                    h_spacing,
                    z -
                    (
                        i +
                        1) *
                    v_spacing,
                    '{:.2f}'.format(
                        popData['POP_SIZE'] /
                        PSET.populationParameters['POP_SIZE'].sum()),
                    ha='center',
                    va='top')
            elif key == 'extrinsic_input_density':
                ax.text(j * h_spacing,
                        z - (i + 1) * v_spacing,
                        '{:.0f}'.format(int(areas[j] * popData[key])),
                        ha='center',
                        va='top')
            else:
                ax.text(j * h_spacing, z - (i + 1) * v_spacing,
                        popData[key], ha='center', va='top')

    # PANEL B. Connection probability
    ax = fig.add_subplot(gs[:3, 14:17])
    plotting.plot_connectivity(ax, PSET, data='connprob', cbarlabel='')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_title(r'$C_{YX}$')
    ax.text(-0.1, 1.15, alphabet[1],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL C. Mean number of synapses per connection
    ax = fig.add_subplot(gs[:3, 19:22])
    plot_multapseargs(ax, PSET, data='multapseargs', cbarlabel='', key='loc')
    ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_title(r'$\overline{n}_\mathrm{syn}$')
    ax.text(-0.1, 1.15, alphabet[2],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL D. Layer specificites of connections
    axes = [fig.add_subplot(gs[:3, i * 4 + 4:i * 4 + 8]) for i in range(5, 9)]
    plot_quantity_YXL(axes=axes, PSET=PSET,
                      quantity=PSET.L_YXL_m_types,
                      y=PSET.populationParameters['m_type'],
                      layers=PSET.layer_data['layer'],
                      label=r'$\mathcal{L}_{YXL}$')
    for ax in axes:
        ax.set_ylabel('')
    axes[0].text(-0.075, 1.15, alphabet[3],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16, fontweight='demibold',
                 transform=axes[0].transAxes)

    # PANEL E. Population illustration
    ax = fig.add_subplot(gs[6:, 13:20])
    # plot electrode contact points
    ax.plot(PSET.electrodeParams['x'], PSET.electrodeParams['z'], 'ko',
            markersize=5, clip_on=False)
    # plot ECoG electrode
    ax.plot([-PSET.ecogParameters['r'],
             PSET.ecogParameters['r']],
            [PSET.ecogParameters['z'][0]] * 2,
            'gray',
            lw=5,
            zorder=-1,
            clip_on=False)

    plotting.remove_axis_junk(ax)

    # draw the first NCELLS cells in each population
    NCELLS = 20
    CWD = PSET.CWD
    CELLPATH = PSET.CELLPATH
    # cell positions and rotations file:
    f = h5py.File(os.path.join(PSET.OUTPUTPATH,
                               'cell_positions_and_rotations.h5'), 'r')

    for i, data in enumerate(PSET.populationParameters):
        NRN = data["me_type"]
        os.chdir(os.path.join(CWD, CELLPATH, NRN))
        for j in range(NCELLS):
            try:
                cell = NetworkCell(**PSET.cellParameters[NRN])
                cell.set_pos(
                    x=f[NRN]['x'][j],
                    y=f[NRN]['y'][j],
                    z=f[NRN]['z'][j])
                cell.set_rotation(
                    x=f[NRN]['x_rot'][j],
                    y=f[NRN]['y_rot'][j],
                    z=f[NRN]['z_rot'][j])

                zips = []
                for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                    zips.append(list(zip(x, z)))

                polycol = PolyCollection(zips,
                                         edgecolors=colors[i],
                                         linewidths=0.05,
                                         facecolors=colors[i],
                                         label=NRN,
                                         zorder=f[NRN]['y'][j],
                                         )
                ax.add_collection(polycol)
                neuron.h('forall delete_section()')
            except IndexError:
                # NCELLS > cell count in population, plotting all there is.
                pass
        os.chdir(CWD)

    ax.axis(ax.axis('equal'))
    ax.set_ylim(-PSET.layer_data['thickness'].sum(), 0)
    # draw lines showing the layer boundaries
    ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()], -
              536.09990331936808, 536.09990331936808, 'k', lw=0.5)
    ax.set_yticks(PSET.layer_data['center'])
    ax.set_yticklabels(PSET.layer_data['layer'])
    ax.set_ylabel('layer')
    ax.set_xticks([-data['pop_args']['radius'], 0, data['pop_args']['radius']])
    ax.set_xlabel(r'$x$ ($\mu$m)')

    ax.text(-0.05, 1.05, alphabet[4],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL F. Population densities across depth
    ax = fig.add_subplot(gs[6:, 20:24])
    plotting.remove_axis_junk(ax)
    ax.text(-0.075, 1.05, alphabet[5],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # open file for reading
    f = h5py.File(os.path.join(PSET.OUTPUTPATH,
                               'cell_positions_and_rotations.h5'), 'r')
    # spatial bins across depth
    bins = np.arange(0, -PSET.layer_data['thickness'].sum(), -50)[::-1]
    for color, post, key in zip(colors,
                                PSET.populationParameters['m_type'],
                                PSET.populationParameters['me_type']):
        ax.hist(
            f[key]['z'],
            bins=bins,
            color=color,
            alpha=1,
            orientation='horizontal',
            histtype='step',
            label=r'{}'.format(post),
            clip_on=False)
    ax.set_xlabel('count')
    axis = ax.axis('tight')
    ax.axis(axis)
    ax.set_ylim(-PSET.layer_data['thickness'].sum(), 0)
    # draw lines showing the layer boundaries
    ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()],
              axis[0], axis[1], 'k', lw=0.5)
    ax.set_yticks(PSET.layer_data['center'])
    ax.set_yticklabels([])
    ax.legend(loc=8)
    ax.xaxis.set_major_locator(MaxNLocator(3))

    # PANEL G. Resulting densities of synapses across depth onto each cell type
    axes = [fig.add_subplot(gs[6:, i * 4 + 4:i * 4 + 8]) for i in range(5, 9)]
    axes[0].text(-0.075, 1.05, alphabet[6],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16, fontweight='demibold',
                 transform=axes[0].transAxes)

    # spatial bins across depth
    bins = np.arange(0, -PSET.layer_data['thickness'].sum(), -50)[::-1]

    # file output
    f = h5py.File(os.path.join(PSET.OUTPUTPATH, 'synapse_connections.h5'), 'r')
    for i, (m_post, post) in enumerate(zip(PSET.populationParameters['m_type'],
                                           PSET.populationParameters['me_type']
                                           )):
        ax = axes[i]
        plotting.remove_axis_junk(ax)
        ax.set_xlabel('count ($10^3$)')
        ax.set_yticklabels([])
        ax.set_title(r'$Y=${}'.format(m_post))
        for color, m_pre, pre in zip(colors,
                                     PSET.populationParameters['m_type'],
                                     PSET.populationParameters['me_type']):
            key = '{}:{}'.format(pre, post)
            ax.hist(
                f[key]['z'],
                bins=bins,
                color=color,
                alpha=1,
                orientation='horizontal',
                histtype='step',
                label=r'{}'.format(m_pre),
                clip_on=False,
                weights=1E-3 *
                np.ones(
                    f[key]['z'].size))
        axis = ax.axis('tight')
        ax.axis(axis)
        ax.set_ylim(-PSET.layer_data['thickness'].sum(), 0)
        # draw lines showing the layer boundaries
        ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()],
                  axis[0], axis[1], 'k', lw=0.5)
        ax.set_yticks(PSET.layer_data['center'])
        ax.xaxis.set_major_locator(MaxNLocator(2))

    fig.savefig(
        os.path.join(
            PSET.OUTPUTPATH,
            'figure_4.pdf'),
        bbox_inches='tight')

    plt.show()
