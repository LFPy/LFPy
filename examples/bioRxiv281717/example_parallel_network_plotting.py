#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''plotting/analysis routines on output of example_parallel_network.py

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
from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
import os
import numpy as np
import scipy.signal as ss
import h5py
from copy import copy
from LFPy import NetworkCell
from mpi4py import MPI
import neuron

# set up MPI environment
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# set default plotting parameters
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


def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.


    Parameters
    ----------
    x : numpy.ndarray
        Array to be downsampled along last axis.
    q : int
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    numpy.ndarray
        Array of downsampled signal.

    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except BaseException:
        # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except BaseException:
        return y[::q]


def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        scalebarpos=0,
        scalebarbasis='log2',
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    ''' draw some nice lines'''

    tvec = np.arange(data.shape[1]) * dt
    try:
        tinds = (tvec >= T[0]) & (tvec <= T[1])
    except TypeError:
        print(data.shape, T)
        raise Exception

    # apply temporal filter
    if filter:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    # subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i + 1))
        yticks.append(z)

    if scalebar:
        if scalebarbasis == 'log2':
            ax.plot([tvec[tinds][-1], tvec[tinds][-1]],
                    [-1 - scalebarpos, -2 - scalebarpos], lw=2,
                    color=color, clip_on=False)
            ax.text(tvec[tinds][-1] + np.diff(T) * 0.03, -1.5 - scalebarpos,
                    '$2^{' + '{}'.format(int(np.log2(vlimround))) +
                    '}$ ' + '{0}'.format(unit),
                    color=color, rotation='vertical',
                    va='center')
        elif scalebarbasis == 'log10':
            # recompute scale bar size to show it on scientific format
            vlimround10 = 10**np.round(np.log10(vlimround))
            if vlimround10 >= 1:
                vlimround10 = int(np.round(vlimround10))
            rescale = vlimround10 / vlimround
            ax.plot([tvec[tinds][-1], tvec[tinds][-1]],
                    np.array([0.5, -0.5]) * rescale - 1.5 - scalebarpos,
                    lw=2, color=color, clip_on=False)
            ax.text(tvec[tinds][-1] + np.diff(T) * 0.03, -1.5 - scalebarpos,
                    '{0} '.format(vlimround10) + '{0}'.format(unit),
                    color=color, rotation='vertical',
                    va='center')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r'time (ms)', labelpad=0.1)

    return vlimround


def remove_axis_junk(ax, lines=['right', 'top']):
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_connectivity(ax, PSET, cmap=plt.get_cmap('inferno'),
                      data='connprob', cbarlabel=r'$C_{YX}$'):
    '''make an imshow of the intranetwork connectivity'''
    masked_array = np.ma.array(
        PSET.connParams[data], mask=np.array(
            PSET.connParams[data]) == 0.)
    cmap = copy(cmap)
    cmap.set_bad('k', 0.5)
    # interpolation='nearest')
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


def plot_quantity_yXL(fig, left, bottom, top, PSET, quantity,
                      y=['p23', 'b23', 'nb23', 'p4', 'ss4(L23)', 'ss4(L4)',
                         'b4', 'nb4', 'p5(L23)', 'p5(L56)', 'b5', 'nb5',
                         'p6(L4)', 'p6(L56)', 'b6', 'nb6'],
                      label=r'$\mathcal{L}_{yXL}$',
                      layers=['L1', 'L2/3', 'L4', 'L5', 'L6'],
                      cmap=plt.get_cmap('inferno')):
    '''make a bunch of image plots, each showing the spatial normalized
    connectivity of synapses'''

    ncols = 3  # int(np.floor(np.sqrt(len(y))))
    nrows = int(len(y) // ncols)
    if len(y) % ncols > 0:
        nrows += 1

    # assess vlims
    vmin = 0
    vmax = 0
    for yi in y:
        if quantity[yi].max() > vmax:
            vmax = quantity[yi].max()

    gs = GridSpec(nrows, ncols, left=left, bottom=bottom, top=top)

    for i, yi in enumerate(y):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])

        masked_array = np.ma.array(quantity[yi], mask=quantity[yi] == 0)

        im = ax.pcolormesh(masked_array,
                           vmin=vmin, vmax=vmax,
                           cmap=cmap,
                           )
        ax.invert_yaxis()

        ax.axis(ax.axis('tight'))
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(np.arange(len(y)) + 0.5)
        ax.set_yticks(np.arange(len(layers)) + 0.5)

        if i % ncols == 0:
            ax.set_yticklabels(layers, )
            ax.set_ylabel('$L$', labelpad=0.)
        else:
            ax.set_yticklabels([])
        if i < ncols:
            ax.set_xlabel(r'$X$', labelpad=-1, fontsize=8)
            ax.set_xticklabels(y, rotation=90)
        else:
            ax.set_xticklabels([])
        ax.xaxis.set_label_position('top')

        ax.text(0.5, -0.13, r'$y=$' + yi,
                horizontalalignment='center',
                verticalalignment='center',
                #
                transform=ax.transAxes, fontsize=5.5)

        # colorbar
        if (i // ncols == 0) and (i % ncols) == ncols - 1:
            rect = np.array(ax.get_position().bounds)
            rect[0] += rect[2] + 0.01
            rect[1] = bottom
            rect[2] = 0.01
            rect[3] = top - bottom
            cax = fig.add_axes(rect)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(label, labelpad=0)


def plot_m_types(ax, PSET, colors, section=[
                 'dend', 'apic'], spacing=300, linewidths=0.05):
    '''draw comparison plot of each individual morphology'''
    CWD = PSET.CWD
    CELLPATH = PSET.CELLPATH
    n_segs = []
    areas = []

    for i, data in enumerate(PSET.populationParameters):

        NRN = data["me_type"]
        os.chdir(os.path.join(CWD, CELLPATH, NRN))
        cell = NetworkCell(**PSET.cellParameters[NRN])
        cell.set_pos(x=i * spacing, y=0, z=data['pop_args']['loc'])
        cell.set_rotation(x=np.pi / 2)
        n_segs += [cell.totnsegs]
        areas += [cell.area[cell.get_idx(section)].sum()]

        zips = []
        for x, z in cell.get_idx_polygons(projection=('x', 'z')):
            zips.append(list(zip(x, z)))

        polycol = PolyCollection(zips,
                                 edgecolors=colors[i],
                                 linewidths=linewidths,
                                 facecolors=colors[i],
                                 label=NRN,
                                 )
        ax.add_collection(polycol)
        os.chdir(CWD)

    axis = ax.axis(ax.axis('tight'))

    # draw lines showing the layer boundaries
    ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()]
              [:4], axis[0], axis[1] - 300, 'k', lw=0.5)
    ax.hlines(np.r_[0., -PSET.layer_data['thickness'].cumsum()]
              [4:], axis[0], axis[1], 'k', lw=0.5)

    # annotate hlines with values
    for z in np.r_[0., -PSET.layer_data['thickness'].cumsum()]:
        ax.text(
            axis[0],
            z,
            r'$z={}$'.format(
                int(z)) +
            r'$\mu$m',
            ha='right',
            va='center')

    ax.set_yticks(PSET.layer_data['center'])
    ax.set_yticklabels(PSET.layer_data['layer'])

    ax.set_xticks(np.arange(PSET.populationParameters.size) * spacing)
    ax.set_xticklabels(
        PSET.populationParameters['m_type'],
        rotation='vertical')

    ax.axis(ax.axis('equal'))
    ax.set_title('m-types')
    neuron.h("forall delete_section()")

    return n_segs, areas


if __name__ == '__main__':
    # get simulation parameters
    from example_parallel_network_parameters import PSET

    ##########################################################################
    # Plot simulated output
    ##########################################################################

    if not os.path.isdir(PSET.OUTPUTPATH):
        if RANK == 0:
            os.mkdir(PSET.OUTPUTPATH)
    COMM.Barrier()

    ############################################
    T = (PSET.TRANSIENT, PSET.tstop)
    colors = [
        plt.get_cmap(
            'Set1',
            PSET.populationParameters.size)(i) for i in range(
            PSET.populationParameters.size)]
    ############################################

    # plot m-types in network
    fig, ax = plt.subplots(1, 1, figsize=(PSET.populationParameters.size, 10))
    plot_m_types(ax, PSET, colors, spacing=300.)
    fig.savefig(os.path.join(PSET.OUTPUTPATH,
                             'example_parallel_network_m_types.pdf'),
                bbox_inches='tight')
    plt.close(fig)

    # plot connection probabilities between pre and postsynaptic populations
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(top=0.85)
    plot_connectivity(ax, PSET)
    fig.savefig(os.path.join(PSET.OUTPUTPATH,
                             'example_parallel_network_connectivity.pdf'),
                bbox_inches='tight')
    plt.close(fig)

    # plot layer specificity of connections between pre and postsynaptic cell
    # types
    fig = plt.figure()
    fig.suptitle('layer specificity of connections')
    plot_quantity_yXL(fig=fig, left=0.1, bottom=0.05, top=0.8, PSET=PSET,
                      quantity=PSET.L_YXL_m_types,
                      y=PSET.populationParameters['m_type'],
                      layers=PSET.layer_data['layer'],
                      label=r'$\mathcal{L}_{YXL}$')
    fig.savefig(os.path.join(PSET.OUTPUTPATH,
                             'example_parallel_network_L_YXL.pdf'),
                bbox_inches='tight')
    plt.close(fig)

    # plot summed LFP and contributions of leak and capacitive currents
    if RANK == 0 and PSET.COMPUTE_LFP:
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')
        for data, title, suffix, color in zip(
                [  # f['SUMMED_OUTPUT'].value['imem'],
                    # f['SUMMED_OUTPUT'].value['ipas'],
                    # f['SUMMED_OUTPUT'].value['icap'],
                    # f['SUMMED_OUTPUT'].value['isyn_e'],
                    # f['SUMMED_OUTPUT'].value['isyn_i'],
                    # f['SUMMED_OUTPUT'].value['isyn_e']
                    # + f['SUMMED_OUTPUT'].value['isyn_i'],
                    # f['SUMMED_OUTPUT'].value['imem']
                    # - f['SUMMED_OUTPUT'].value['ipas']
                    # - f['SUMMED_OUTPUT'].value['icap']
                    # - f['SUMMED_OUTPUT'].value['isyn_e']
                    # - f['SUMMED_OUTPUT'].value['isyn_i'],
                ] + [f['SUMMED_OUTPUT'].value[name]
                     for name in f['SUMMED_OUTPUT'].dtype.names],
                [  # 'extracellular potentials, summed',
                    # 'extracellular potential, leak currents',
                    # 'extracellular potential, capacitive currents',
                    # 'extracellular potential, exc. synaptic currents',
                    # 'extracellular potential, inh. synaptic currents',
                    # 'extracellular potential, exc. + inh. synaptic currents',
                    # 'extracellular potential, residual',
                ] + [name for name in f['SUMMED_OUTPUT'].dtype.names],
                [  # 'LFP',
                   # 'i_pas', 'i_cap', 'i_syn_e', 'i_syn_i', 'i_syn_ei', 'i_gX'
                ] + [name for name in f['SUMMED_OUTPUT'].dtype.names],
                [  # 'k',
                   # 'r', 'b', 'c', 'm', 'g', 'y'
                ] + [colors[i]
                     for i in range(PSET.populationParameters.size)]):
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(121)
            ax.set_title(title)
            vlimround = draw_lineplot(ax=ax,
                                      data=decimate(data, q=PSET.decimate_q),
                                      dt=PSET.dt * PSET.decimate_q,
                                      T=T, color=color)

            ax = fig.add_subplot(122)
            ax.set_title(title + r' (LP filtered, $f_\mathrm{crit}=100$ Hz)')
            vlimround = draw_lineplot(ax=ax,
                                      data=decimate(data, q=PSET.decimate_q),
                                      dt=PSET.dt * PSET.decimate_q,
                                      T=T, color=color,
                                      ztransform=True, filter=True,
                                      filterargs=PSET.filterargs)

            # save figure output
            fig.savefig(
                os.path.join(PSET.OUTPUTPATH,
                             'example_parallel_network_summed_{}.pdf'.format(
                                 suffix)),
                bbox_inches='tight')
            plt.close(fig)
        f.close()

    if RANK == 0 and PSET.COMPUTE_LFP:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax.set_title('extracellular signal variance')
        y = PSET.electrodeParams['z']
        yticklabels = ['ch. {}'.format(x + 1) for x in range(y.size)]
        tind = int(PSET.TRANSIENT / PSET.dt)
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')
        for data, label, color in zip([  # f['SUMMED_OUTPUT'].value['imem'],
            # f['SUMMED_OUTPUT'].value['ipas'],
            # f['SUMMED_OUTPUT'].value['icap'],
            # f['SUMMED_OUTPUT'].value['isyn_e'],
            # f['SUMMED_OUTPUT'].value['isyn_i'],
            # f['SUMMED_OUTPUT'].value['isyn_e']
            # + f['SUMMED_OUTPUT'].value['isyn_i'],
            # f['SUMMED_OUTPUT'].value['imem']
            # - f['SUMMED_OUTPUT'].value['ipas']
            # - f['SUMMED_OUTPUT'].value['icap']
            # - f['SUMMED_OUTPUT'].value['isyn_e']
            # - f['SUMMED_OUTPUT'].value['isyn_i']
        ] + [f['SUMMED_OUTPUT'].value[name]
             for name in f['SUMMED_OUTPUT'].dtype.names],
            [  # 'sum',
            # r'$i_\mathrm{pas}$', r'$i_\mathrm{cap}$',
            # r'$i_\mathrm{syn, E}$', r'$i_\mathrm{syn, I}$',
            # r'$i_\mathrm{syn, E}+i_\mathrm{syn, I}$', 'residual'
        ] + [name for name in f['SUMMED_OUTPUT'].dtype.names],
            [  # 'k',
            # 'r', 'b', 'c', 'm', 'g', 'y'
        ] + ['k'] + [colors[i]
                     for i in range(PSET.populationParameters.size)]):
            ax.semilogx(data[:, tind:].var(axis=1), y,
                        lw=2, label=label, color=color)
        f.close()
        ax.set_yticks(y)
        ax.set_yticklabels(yticklabels)
        ax.axis(ax.axis('tight'))
        ax.legend(loc='best')
        ax.set_xlabel(r'variance (mV$^2$)')

        ax = fig.add_subplot(212)
        ax.set_title(
            r'LP filtered signals ($f_\mathrm{crit}=100$ Hz, '
            + '4th order Butterworth, filtfilt)')
        b, a = ss.butter(**PSET.filterargs)
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')
        for data, label, color in zip([  # f['SUMMED_OUTPUT'].value['imem'],
            # f['SUMMED_OUTPUT'].value['ipas'],
            # f['SUMMED_OUTPUT'].value['icap'],
            # f['SUMMED_OUTPUT'].value['isyn_e'],
            # f['SUMMED_OUTPUT'].value['isyn_i'],
            # f['SUMMED_OUTPUT'].value['isyn_e']
            # + f['SUMMED_OUTPUT'].value['isyn_i'],
            # f['SUMMED_OUTPUT'].value['imem']
            # - f['SUMMED_OUTPUT'].value['ipas']
            # - f['SUMMED_OUTPUT'].value['icap']
            # - f['SUMMED_OUTPUT'].value['isyn_e']
            # - f['SUMMED_OUTPUT'].value['isyn_i']
        ] + [f['SUMMED_OUTPUT'].value[name]
             for name in f['SUMMED_OUTPUT'].dtype.names],
            [  # 'sum',
            # r'$i_\mathrm{pas}$', r'$i_\mathrm{cap}$', r'$i_\mathrm{syn, E}$',
            # r'$i_\mathrm{syn, I}$', r'$i_\mathrm{syn, E}+i_\mathrm{syn, I}$',
            # 'residual'
        ] + [name for name in f['SUMMED_OUTPUT'].dtype.names],
            [  # 'k',
            # 'r', 'b', 'c', 'm', 'g', 'y'
        ] + ['k'] + [colors[i]
                     for i in range(PSET.populationParameters.size)]):
            ax.semilogx(ss.filtfilt(b, a, data, axis=-1)
                        [:, tind:].var(axis=1), y, lw=2,
                        label=label, color=color)
        f.close()
        ax.set_yticks(y)
        ax.set_yticklabels(yticklabels)
        ax.axis(ax.axis('tight'))
        ax.set_xlabel(r'variance (mV$^2$)')

        fig.savefig(os.path.join(PSET.OUTPUTPATH,
                                 'example_parallel_network_variance.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

    # spike raster plot of all spiking activity from file
    if RANK == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')
        for i, name in enumerate(PSET.populationParameters['me_type']):
            x = []
            y = []
            ax.hlines(
                f['SPIKES'][name]['gids'].value.min(),
                T[0],
                T[1],
                'k',
                lw=0.25)
            for gid, spt in zip(f['SPIKES'][name]['gids'],
                                f['SPIKES'][name]['times']):
                if len(spt) > 0:
                    y += [gid] * spt.size
                    x += list(spt)
            ax.plot(x, y, '|',
                    color=colors[i], markersize=2, lw=2,
                    clip_on=True, label=name)
        f.close()
        ax.axis(ax.axis('tight'))
        ax.set_xlim(PSET.TRANSIENT, PSET.tstop)
        ax.set_ylim(ax.axis()[2] - 0.5, ax.axis()[3] + 0.5)
        ax.invert_yaxis()
        ax.legend(loc='best')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('gid')
        ax.set_title('spike raster')

        # save figure output
        fig.savefig(os.path.join(PSET.OUTPUTPATH,
                                 'example_parallel_network_raster.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

    # spike count rate histograms of all spiking activity from file
    if RANK == 0:
        fig, axes = plt.subplots(PSET.populationParameters.size, 1,
                                 figsize=(10, 10), sharex=True)
        fig.subplots_adjust(left=0.2)
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')

        dt = 10.  # bin size for histograms
        bins = np.arange(T[0], T[1] + dt, dt)

        axes[0].set_title(
            r'population spike time histogram, ($\Delta t={}$ ms)'.format(dt))

        for i, name in enumerate(PSET.populationParameters['me_type']):
            ax = axes[i]
            data = np.hstack(f['SPIKES'][name]['times'].value.flat)
            # , histtype='step', color=colors[i])
            ax.hist(data, bins=bins, color=colors[i])
            ax.axis(ax.axis('tight'))
            ax.set_xlim(PSET.TRANSIENT, PSET.tstop)
            ax.set_ylabel(name, rotation='horizontal', labelpad=50)
            if ax != axes[-1]:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (ms)')

        f.close()

        # save figure output
        fig.savefig(os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_spike_time_histogram.pdf'),
            bbox_inches='tight')
        plt.close(fig)

    # spike count histogram across populations from file
    if RANK == 0:
        n = PSET.populationParameters['me_type'].size
        ncols = int(np.floor(np.sqrt(n)))
        nrows = int(np.ceil(float(n) / ncols))
        gs = GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(
            'per-cell spike count hist., T={} s'.format(
                (PSET.tstop - PSET.TRANSIENT) / 1000.))

        bins = np.arange(42) * (PSET.tstop - PSET.TRANSIENT) / \
            1000.  # make count bins conform to bin size of 1 Hz.
        f = h5py.File(
            os.path.join(
                PSET.OUTPUTPATH,
                'example_parallel_network_output.h5'),
            'r')
        for i, name in enumerate(PSET.populationParameters['me_type']):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            x = []
            for spt in f['SPIKES'][name]['times']:
                if spt.size == 0:
                    x += [0]
                else:
                    if np.any(spt >= PSET.TRANSIENT):
                        x += [spt[spt >= PSET.TRANSIENT].size]
                    else:
                        x += [0]
            ax.hist(
                x,
                bins=bins,
                color=colors[i],
                clip_on=True,
                label=name)  # histtype='step',
            ax.axis(ax.axis('tight'))
            ax.set_title(name)

            if i >= (n - ncols):
                ax.set_xlabel('spike count')
            else:
                ax.set_xticklabels([])
            if i % ncols == 0:
                ax.set_ylabel('observations')
        f.close()
        # save figure output
        fig.savefig(os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_spike_count_hist.pdf'),
            bbox_inches='tight')
        plt.close(fig)
