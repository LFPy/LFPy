#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that the complete set of cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads is unzipped in this folder.

Execution:

    python example_EPFL_neurons.py

Copyright (C) 2017 Computational Neuroscience Group, NMBU.

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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection, LineCollection
from glob import glob
import numpy as np
from warnings import warn
import scipy.signal as ss
import neuron
import LFPy
from mpi4py import MPI


plt.rcParams.update({
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'font.size': 8,
    'ytick.labelsize': 8,
    'xtick.labelsize': 8,
})

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# working dir
CWD = os.getcwd()
NMODL = 'hoc_combos_syn.1_0_10.allmods'

# load some required neuron-interface files
neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

# load only some layer 5 pyramidal cell types
neurons = glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_TTPC*'))[:1]
neurons += glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_MC*'))[:1]
neurons += glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_LBC*'))[:1]
neurons += glob(os.path.join('hoc_combos_syn.1_0_10.allzips', 'L5_NBC*'))[:1]

# flag for cell template file to switch on (inactive) synapses
add_synapses = False

# attempt to set up a folder with all unique mechanism mod files, compile, and
# load them all
if RANK == 0:
    if not os.path.isdir(NMODL):
        os.mkdir(NMODL)
    for NRN in neurons:
        for nmodl in glob(os.path.join(NRN, 'mechanisms', '*.mod')):
            while not os.path.isfile(os.path.join(NMODL,
                                                  os.path.split(nmodl)[-1])):
                if "win32" in sys.platform:
                    os.system("copy {} {}".format(nmodl, NMODL))
                else:
                    os.system('cp {} {}'.format(nmodl,
                                                os.path.join(NMODL, '.')))
    os.chdir(NMODL)
    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows. " +
             "Run mknrndll from NEURON bash in the folder %s" % NMODL +
             "and rerun example script")
    else:
        os.system('nrnivmodl')
    os.chdir(CWD)
COMM.Barrier()
if "win32" in sys.platform:
    if NMODL not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(NMODL + "/nrnmech.dll")
    neuron.nrn_dll_loaded.append(NMODL)
else:
    neuron.load_mechanisms(NMODL)

os.chdir(CWD)

FIGS = 'hoc_combos_syn.1_0_10.allfigures'
if not os.path.isdir(FIGS):
    os.mkdir(FIGS)


# load the LFPy SinSyn mechanism for stimulus
if "win32" in sys.platform:
    pth = os.path.join(LFPy.__path__[0], "test").replace(os.sep, posixpath.sep)
    if pth not in neuron.nrn_dll_loaded:
        neuron.h.nrn_load_dll(pth + "/nrnmech.dll")
    neuron.nrn_dll_loaded.append(pth)
else:
    neuron.load_mechanisms(os.path.join(LFPy.__path__[0], "test"))


def posixpth(pth):
    """
    Replace Windows path separators with posix style separators
    """
    return pth.replace(os.sep, posixpath.sep)


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


# PARAMETERS

# sim duration
tstop = 1000.
dt = 2**-6

PointProcParams = {
    'idx': 0,
    'pptype': 'SinSyn',
    'delay': 200.,
    'dur': tstop - 300.,
    'pkamp': 0.5,
    'freq': 0.,
    'phase': np.pi / 2,
    'bias': 0.,
    'record_current': False
}

# spike sampling
threshold = -20  # spike threshold (mV)
samplelength = int(2. / dt)

# filter settings for extracellular traces
b, a = ss.butter(N=3, Wn=(300 * dt * 2 / 1000, 5000 *
                          dt * 2 / 1000), btype='bandpass')
apply_filter = True

# communication buffer where all simulation output will be gathered on RANK 0
COMM_DICT = {}

COUNTER = 0
for i, NRN in enumerate(neurons):
    os.chdir(NRN)

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
    f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = get_templatename(f)
    f.close()

    print('Loading constants')
    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, posixpth(os.path.join('synapses', 'synapses.hoc')
                                       ))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    templatefile = posixpth(os.path.join(NRN, 'template.hoc'))
    for morphologyfile in glob(os.path.join('morphology', '*')):
        if COUNTER % SIZE == RANK:
            # Instantiate the cell(s) using LFPy
            cell = LFPy.TemplateCell(morphology=morphologyfile,
                                     templatefile=templatefile,
                                     templatename=templatename,
                                     templateargs=1 if add_synapses else 0,
                                     tstop=tstop,
                                     dt=dt,
                                     nsegs_method=None)

            # set view as in most other examples
            cell.set_rotation(x=np.pi / 2)

            pointProcess = LFPy.StimIntElectrode(cell, **PointProcParams)

            electrode = LFPy.RecExtElectrode(cell=cell,
                                             x=np.array([-40, 40., 0, 0]),
                                             y=np.array([0, 0, -40, 40]),
                                             z=np.zeros(4),
                                             sigma=0.3, r=5, n=50,
                                             N=np.array([[1, 0, 0],
                                                         [1, 0, 0],
                                                         [1, 0, 0],
                                                         [1, 0, 0]]),
                                             method='root_as_point')
            # run simulation
            cell.simulate(probes=[electrode])

            # compute LFP
            if apply_filter:
                LFP = ss.filtfilt(b, a, electrode.data, axis=-1)

            # detect action potentials from intracellular trace
            AP_train = np.zeros(cell.somav.size, dtype=int)
            crossings = ((cell.somav[:-1] < threshold) &
                         (cell.somav[1:] >= threshold))
            spike_inds = np.where(crossings)[0]
            # sampled spike waveforms for each event
            spw = np.zeros((crossings.sum() * LFP.shape[0], 2 * samplelength))
            tspw = np.arange(-samplelength, samplelength) * dt
            # set spike time where voltage gradient is largest
            n = 0  # counter
            for j, i in enumerate(spike_inds):
                inds = np.arange(i - samplelength, i + samplelength)
                w = cell.somav[inds]
                k = inds[:-1][np.diff(w) == np.diff(w).max()][0]
                AP_train[k] = 1
                # sample spike waveform
                for lfp in LFP:
                    spw[n, ] = lfp[np.arange(k - samplelength,
                                             k + samplelength)]
                    n += 1

            # fill in sampled spike waveforms and times of spikes in comm_dict
            COMM_DICT.update({
                os.path.split(NRN)[-1] + '_' +
                os.path.split(morphologyfile)[-1].strip('.asc'): dict(
                    spw=spw,
                )
            })

            # plot
            gs = GridSpec(2, 3)
            fig = plt.figure(figsize=(10, 8))
            fig.suptitle(NRN + '\n' +
                         os.path.split(morphologyfile)[-1].strip('.asc'))

            # morphology
            zips = []
            for x, z in cell.get_idx_polygons(projection=('x', 'z')):
                zips.append(list(zip(x, z)))
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors='k',
                                     rasterized=True)
            ax = fig.add_subplot(gs[:, 0])
            ax.add_collection(polycol)
            ax.plot(electrode.x, electrode.z, 'ro')
            ax.axis(ax.axis('equal'))
            ax.set_title('morphology')
            ax.set_xlabel('(um)', labelpad=0)
            ax.set_ylabel('(um)', labelpad=0)

            # soma potential and spikes
            ax = fig.add_subplot(gs[0, 1])
            ax.plot(cell.tvec, cell.somav, rasterized=True)
            ax.plot(cell.tvec, AP_train * 20 + 50)
            ax.axis(ax.axis('tight'))
            ax.set_title('soma voltage, spikes')
            ax.set_ylabel('(mV)', labelpad=0)

            # extracellular potential
            ax = fig.add_subplot(gs[1, 1])
            for lfp in LFP:
                ax.plot(cell.tvec, lfp, rasterized=True)
            ax.axis(ax.axis('tight'))
            ax.set_title('extracellular potential')
            ax.set_xlabel('(ms)', labelpad=0)
            ax.set_ylabel('(mV)', labelpad=0)

            # spike waveform
            ax = fig.add_subplot(gs[0, 2])
            n = electrode.x.size
            for j in range(n):
                zips = []
                for x in spw[j::n, ]:
                    zips.append(list(zip(tspw, x)))
                linecol = LineCollection(zips,
                                         linewidths=0.5,
                                         colors=plt.cm.Spectral(
                                             int(255. * j / n)),
                                         rasterized=True)
                ax.add_collection(linecol)
            ax.axis(ax.axis('tight'))
            ax.set_title('spike waveforms')
            ax.set_ylabel('(mV)', labelpad=0)

            # spike width vs. p2p amplitude
            ax = fig.add_subplot(gs[1, 2])
            w = []
            p2p = []
            for x in spw:
                j = x == x.min()
                i = x == x[np.where(j)[0][0]:].max()
                w += [(tspw[i] - tspw[j])[0]]
                p2p += [(x[i] - x[j])[0]]
            ax.plot(w, p2p, 'o', lw=0.1, markersize=5, mec='none')
            ax.set_title('spike peak-2-peak time and amplitude')
            ax.set_xlabel('(ms)', labelpad=0)
            ax.set_ylabel('(mV)', labelpad=0)

            fig.savefig(os.path.join(CWD, FIGS, os.path.split(NRN)[-1] + '_' +
                                     os.path.split(morphologyfile
                                                   )[-1].replace('.asc',
                                                                 '.pdf')),
                        dpi=200)
            plt.close(fig)

        COUNTER += 1
        os.chdir(CWD)


COMM.Barrier()

# gather sim output
if SIZE > 1:
    if RANK == 0:
        for i in range(1, SIZE):
            COMM_DICT.update(COMM.recv(source=i, tag=123))
            print('received from RANK {} on RANK {}'.format(i, RANK))
    else:
        print('sent from RANK {}'.format(RANK))
        COMM.send(COMM_DICT, dest=0, tag=123)
else:
    pass
COMM.Barrier()

# project data
if RANK == 0:
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('spike peak-2-peak time and amplitude')
    n = electrode.x.size
    for k in range(n):
        ax = fig.add_subplot(n, 2, k * 2 + 1)
        for key, val in COMM_DICT.items():
            spw = val['spw'][k::n, ]
            w = []
            p2p = []
            for x in spw:
                j = x == x.min()
                i = x == x[np.where(j)[0][0]:].max()
                w += [(tspw[i] - tspw[j])[0]]
                p2p += [(x[i] - x[j])[0]]
            if 'MC' in key:
                marker = 'x'
            elif 'NBC' in key:
                marker = '+'
            elif 'LBC' in key:
                marker = 'd'
            elif 'TTPC' in key:
                marker = '^'
            ax.plot(w, p2p, marker, lw=0.1,
                    markersize=5, mec='none', label=key, alpha=0.25)
        ax.set_xlabel('(ms)', labelpad=0)
        ax.set_ylabel('(mV)', labelpad=0)
        if k == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                      frameon=False, fontsize=7)
    fig.savefig(os.path.join(CWD, FIGS, 'P2P_time_amplitude.pdf'))
    print("wrote {}".format(os.path.join(CWD, FIGS, 'P2P_time_amplitude.pdf')))
    plt.close(fig)
else:
    pass
COMM.Barrier()
