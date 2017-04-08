#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate usage of LFPy.NetworkCell with a ball-and-stick type
morphology with active HH channels inserted in the soma and passive-leak
channels distributed throughout the apical stick. The corresponding morphology
and template-specification is in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

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
import matplotlib.pyplot as plt
from LFPy import NetworkCell, StimIntElectrode
import neuron
# class NetworkCell parameters
cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    passive=False,
    v_init=-65
    )
# create cell instance
cell = NetworkCell(
    tstart=0, tstop=100,
    **cellParameters
    )
# stimulus device
iclamp = StimIntElectrode(
    cell=cell,
    idx=0,
    pptype='IClamp',
    amp=0.5,
    dur=80,
    delay=10,
    record_current=True
    )
# run simulation
cell.simulate()

plt.subplot(2,1,1)
plt.plot(cell.tvec, iclamp.i)
plt.ylabel(r'$i_\mathrm{clamp}$ (nA)')
plt.subplot(2,1,2)
plt.plot(cell.tvec, cell.somav)
plt.ylabel(r'$V_\mathrm{soma}$ (mV)')
plt.xlabel(r'$t$ (ms)')
plt.show()

# customary cleanup of object references:
plt.close('all')
iclamp = None
cell = None
neuron.h('forall delete_section()')
