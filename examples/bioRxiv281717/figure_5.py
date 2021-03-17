#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''plotting script for figure 5 in manuscript preprint on output of
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axisartist.axislines import SubplotZero
import os
import numpy as np
import h5py
from LFPy import FourSphereVolumeConductor, MEG
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


def plot_spike_raster(ax, PSET, T):
    f = h5py.File(
        os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_output.h5'),
        'r')
    for i, (m_name, name) in enumerate(
            zip(PSET.populationParameters['m_type'],
                PSET.populationParameters['me_type'])):
        x = []
        y = []
        ax.hlines(f['SPIKES'][name]['gids'][()].min(),
                  T[0], T[1], 'k', lw=0.25)
        for gid, spt in zip(f['SPIKES'][name]['gids'],
                            f['SPIKES'][name]['times']):
            if len(spt) > 0:
                y += [gid] * spt.size
                x += list(spt)
        x = np.array(x)
        y = np.array(y)
        inds = (x >= T[0]) & (x <= T[1])
        ax.plot(x[inds], y[inds], '|',
                color=colors[i], markersize=2,
                lw=2, clip_on=True, label=m_name)
    f.close()
    ax.set_xlim(T[0], T[1])
    ax.set_ylim(-0.5, PSET.populationParameters['POP_SIZE'].sum() + 0.5)
    ax.invert_yaxis()
    ax.legend(loc=1, markerscale=5)
    ax.set_xlabel('time (ms)', labelpad=0)
    ax.set_ylabel('cell ID')
    ax.set_title('spike raster')


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
    fig = plt.figure(figsize=(16, 15.5))
    gs = GridSpec(5, 3, hspace=0.3, wspace=0.2, left=0.06, right=0.95,
                  top=0.95, bottom=0.05)
    alphabet = 'ABCDEFGHIJKLMNOPQRST'

    # 90 deg rotation matrices around x-, y- and z-axis
    Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # PANEL A. Spike raster
    ax = fig.add_subplot(gs[:2, 0])
    plot_spike_raster(ax, PSET, T)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    plotting.remove_axis_junk(ax)
    ax.text(-0.1, 1.025, alphabet[0],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL B. Spike count histogram
    gs0 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:2, 1])
    axes = [
        fig.add_subplot(
            gs0[i]) for i in range(
            PSET.populationParameters['me_type'].size)]

    f = h5py.File(
        os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_output.h5'),
        'r')

    dt = 5.  # bin size for histograms
    bins = np.arange(T[0], T[1] + dt, dt)

    axes[0].set_title(r'spike-count histograms ($\Delta t={}$ ms)'.format(dt))

    for i, (m_name, name) in enumerate(
            zip(PSET.populationParameters['m_type'],
                PSET.populationParameters['me_type'])):
        ax = axes[i]
        plotting.remove_axis_junk(ax)
        data = np.hstack(f['SPIKES'][name]['times'][()])
        ax.hist(data, bins=bins, color=colors[i][:-1], label=m_name)
        ax.axis(ax.axis('tight'))
        ax.set_xlim(PSET.TRANSIENT, PSET.TRANSIENT + 1000.)
        ax.legend(loc=1)
        ax.set_ylabel('count')
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('time (ms)', labelpad=0)

    axes[0].text(-0.1, 1.1, alphabet[1],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16, fontweight='demibold',
                 transform=axes[0].transAxes)

    # PANEL C Extracellular potential
    gs0 = GridSpecFromSubplotSpec(12, 1, subplot_spec=gs[:2, 2])
    ax = fig.add_subplot(gs0[:-2])
    f = h5py.File(
        os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_output.h5'),
        'r')
    for data, title, color in zip(
        [f['SUMMED_OUTPUT'][()]['imem']],
        ['extracellular potentials, summed'],
            ['k']):
        ax.set_title(title)
        vlimround = plotting.draw_lineplot(
            ax=ax,
            data=plotting.decimate(
                data,
                q=PSET.decimate_q),
            dt=PSET.dt *
            PSET.decimate_q,
            T=T,
            color=color,
            scalebarbasis='log10')
    f.close()

    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(-0.1, 1.055, alphabet[2],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL D ECoG potential
    ax = fig.add_subplot(gs0[-1])
    f = h5py.File(
        os.path.join(
            PSET.OUTPUTPATH,
            'example_parallel_network_output.h5'),
        'r')
    data = f['SUMMED_ECOG'][()]['imem']
    title = 'ECoG potential, summed'
    color = 'k'
    ax.set_title(title)
    vlimround = plotting.draw_lineplot(
        ax=ax,
        data=plotting.decimate(
            f['SUMMED_OUTPUT'][()]['imem'][0, ].reshape((1, -1)),
            q=PSET.decimate_q),
        dt=PSET.dt * PSET.decimate_q,
        scalebar=False,
        T=T, color='0.5', scalebarpos=-1.5, scalebarbasis='log10')
    vlimround = plotting.draw_lineplot(
        ax=ax, data=plotting.decimate(data, q=PSET.decimate_q),
        dt=PSET.dt * PSET.decimate_q,
        vlimround=vlimround, scalebar=True,
        T=T, color=color, scalebarpos=-1.5, scalebarbasis='log10')

    lines = ax.get_lines()
    ax.legend(('ch. 1', 'ECoG'), loc=8, ncol=2,
              bbox_to_anchor=(0.5, -1.25), frameon=False)

    f.close()
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.text(-0.1, 1.175, alphabet[3],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL E. current dipole moment signal
    f = h5py.File(os.path.join(PSET.OUTPUTPATH,
                               'example_parallel_network_output.h5'), 'r')
    p_temp = np.zeros(f['CURRENT_DIPOLE_MOMENT'].shape)
    for name in f['CURRENT_DIPOLE_MOMENT'].dtype.names:
        p_temp += f['CURRENT_DIPOLE_MOMENT'][name]
    p_net = plotting.decimate(p_temp, q=PSET.decimate_q)
    p_net *= 1E-3  # nA um -> 1E-3 nA m unit conversion
    del p_temp

    gs0 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2, 0])
    for i, ylabel in enumerate(
            [r'$\mathbf{p \cdot \hat{x}}$',
             r'$\mathbf{p \cdot \hat{y}}$',
             r'$\mathbf{p \cdot \hat{z}}$']):
        ax = fig.add_subplot(gs0[i])
        if i == 0:
            ax.set_title(r'current dipole moment ($10^{-3}$ nA m)')
            ax.text(-0.1, 1.15, alphabet[4],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=ax.transAxes)

        plotting.remove_axis_junk(ax)
        t = np.arange(p_net.shape[1]) * PSET.dt * PSET.decimate_q
        inds = (t >= T[0]) & (t <= T[1])
        ax.plot(t[inds], p_net[i, inds], 'k', lw=1)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([])

    # panel F. Illustration of 4-sphere volume conductor model geometry
    ax = SubplotZero(fig, gs[2, 1])
    fig.add_subplot(ax)
    ax.set_title('four-sphere volume conductor model')

    for direction in ["xzero"]:
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

    theta = np.linspace(0, np.pi, 31)

    # draw some circles:
    for i, r, label in zip(range(4), PSET.foursphereParams['radii'], [
                           'brain', 'CSF', 'skull', 'scalp']):
        ax.plot(
            np.cos(theta) *
            r,
            np.sin(theta) *
            r,
            'C{}'.format(i),
            label=label +
            r', $r_%i=%i$ mm' %
            (i +
             1,
             r /
             1000),
            clip_on=False)

    # draw measurement points
    ax.plot(PSET.foursphereParams['r_electrodes'][:, 0],
            PSET.foursphereParams['r_electrodes'][:, 2],
            'ko',
            label='EEG/MEG sites')

    for i, (x, y, z) in enumerate(PSET.foursphereParams['r_electrodes']):
        ax.text(x, z + 2500, r'{}'.format(i + 1), ha='center')

    # dipole location
    ax.plot([0], [PSET.foursphereParams['radii'][0] +
                  PSET.layer_data['center'][3]], 'k.', label='dipole site')

    ax.axis('equal')
    ax.set_ylim(top=max(PSET.foursphereParams['radii']) + 5000)

    ax.set_xticks(np.r_[-np.array(PSET.foursphereParams['radii']),
                        0, PSET.foursphereParams['radii']])
    ax.set_xticklabels([])

    ax.legend(loc=(0.25, 0.05), frameon=False)

    ax.text(-0.1, 1.05, alphabet[5],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL G. EEG signal
    ax = fig.add_subplot(gs[2, 2])
    ax.set_title(r'surface potential $\phi_\mathbf{p}(\mathbf{r})$ ')

    f = h5py.File(os.path.join(PSET.OUTPUTPATH,
                               'example_parallel_network_output.h5'), 'r')

    # compute dipole potentials as the sum of contributions in
    # different positions
    phi_p = np.zeros(
        (PSET.foursphereParams['r_electrodes'].shape[0],
         f['CURRENT_DIPOLE_MOMENT'][name].shape[1]))
    for i, name in enumerate(PSET.populationParameters['me_type']):
        p = f['CURRENT_DIPOLE_MOMENT'][name]

        # four-sphere volume conductor
        sphere = FourSphereVolumeConductor(
            **PSET.foursphereParams
        )
        phi_p += sphere.get_dipole_potential(
            p=p,
            dipole_location=np.array([0, 0,
                                      PSET.foursphereParams['radii'][0]
                                      + PSET.layer_data['center'][3:][i % 2]])
            )

    vlimround = plotting.draw_lineplot(
        ax=ax,
        data=plotting.decimate(
            phi_p,
            q=PSET.decimate_q)[::-1, ] * 1E3,  # mV -> µV unit conversion
        unit=r'$\mu$V',
        dt=PSET.dt * PSET.decimate_q,
        T=T, color='k', scalebarbasis='log10')

    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_yticklabels(['{}'.format(i + 1)
                        for i in range(sphere.rxyz.shape[0])])
    ax.set_ylabel('position', labelpad=10)

    ax.text(-0.1, 1.05, alphabet[6],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)

    # PANEL H. tangential component of MEG signal (as recorded by squid
    # outside scull)

    # compute the radial unit vector from the center of the sphere to each
    # measurement point, then unit vectors along theta and phi
    r_hat = (sphere.rxyz.T / sphere.r).T
    theta = np.arccos(sphere.rxyz[:, 2] / sphere.r)
    phi = np.arctan2(sphere.rxyz[:, 1], sphere.rxyz[:, 0])
    theta_hat = np.array([np.cos(theta) * np.cos(phi),
                          np.cos(theta) * np.sin(phi),
                          -np.sin(phi)]).T
    phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(r_hat.shape[0])]).T

    for j, (unitvector, akse) in enumerate(zip(
        [theta_hat, phi_hat, r_hat],
        [r'\hat{\mathbf{\theta}}',
         r'\hat{\mathbf{\varphi}}',
         r'\hat{\mathbf{r}}'])):
        ax = fig.add_subplot(gs[3, j])
        ax.set_title(
            'surface magn. field '
            + r'$\mathbf{B}_\mathbf{p}(\mathbf{r}) \cdot %s$' % akse)

        # radial/tangential component of H at squid locations
        H_rt = np.zeros(phi_p.shape)
        for i, name in enumerate(PSET.populationParameters['me_type']):
            # dipole position
            dipole_position = np.array(
                [0, 0,
                 PSET.foursphereParams['radii'][0]
                 + PSET.layer_data['center'][3:][i % 2]])
            # create MEG object and compute magnetic field
            meg = MEG(sensor_locations=PSET.foursphereParams['r_electrodes'])
            H = meg.calculate_H(
                f['CURRENT_DIPOLE_MOMENT'][name],
                dipole_position)

            for k, (h, u) in enumerate(zip(H, unitvector)):
                H_rt[k, ] += np.dot(h.T, u)

        B_rt = H_rt * meg.mu  # unit mT (from nA/µm * Tm/A)
        vlimround = plotting.draw_lineplot(
            ax=ax,
            data=plotting.decimate(B_rt, q=PSET.decimate_q)[
                ::-1, ] * 1E12,  # mT --> fT unit conversion
            dt=PSET.dt * PSET.decimate_q, unit=r'fT',
            T=T, color='k', scalebarbasis='log10')

        ax.set_yticklabels(['{}'.format(i + 1)
                            for i in range(sphere.rxyz.shape[0])])
        ax.set_xlabel('')
        ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel('position', labelpad=10)
            ax.text(-0.1, 1.05, alphabet[7],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=ax.transAxes)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

        # PANEL I. tangential components of MEG signal (as recorded by squid
        # outside scull)
        # with dipole sources rotated 90 deg counterclockwise around x-axis
        ax = fig.add_subplot(gs[4, j])
        ax.set_title(
            'surface magn. field '
            + r'$\mathbf{B}_{R_x(\pi/2)\mathbf{p}}(\mathbf{r}) '
            + r'\cdot %s$' % akse)

        # radial/tangential component of H at squid locations
        H_rt = np.zeros(phi_p.shape)
        for i, name in enumerate(PSET.populationParameters['me_type']):
            # dipole position
            dipole_position = np.array(
                [0, 0,
                 PSET.foursphereParams['radii'][0]
                 + PSET.layer_data['center'][3:][i % 2]])
            # create MEG object and compute magnetic field
            meg = MEG(sensor_locations=PSET.foursphereParams['r_electrodes'])
            H = meg.calculate_H(
                np.dot(
                    Rx90,
                    f['CURRENT_DIPOLE_MOMENT'][name]),
                dipole_position)
            # compute the radial unit vector from the center of the sphere to
            # each
            # measurement point

            for k, (h, u) in enumerate(zip(H, unitvector)):
                H_rt[k, ] += np.dot(h.T, u)

        B_rt = H_rt * meg.mu  # unit mT (from nA/µm * Tm/A)
        vlimround = plotting.draw_lineplot(
            ax=ax,
            data=plotting.decimate(B_rt, q=PSET.decimate_q)[
                ::-1, ] * 1E12,  # mT --> fT unit conversion
            dt=PSET.dt * PSET.decimate_q, unit=r'fT',
            T=T, color='k', scalebarbasis='log10')

        ax.set_yticklabels(['{}'.format(i + 1)
                            for i in range(sphere.rxyz.shape[0])])
        ax.set_xlabel('time (ms)', labelpad=0)
        if j == 0:
            ax.set_ylabel('position', labelpad=10)
            ax.text(-0.1, 1.05, alphabet[8],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=ax.transAxes)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    fig.savefig(
        os.path.join(
            PSET.OUTPUTPATH,
            'figure_5.pdf'),
        bbox_inches='tight')

    plt.show()
