#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A very generic case where the same morphology on different file formats
is loaded in LFPy using a generic template specification, defined by
file LFPyCellTemplate.hoc

Execution:

    python example_LFPyCellTemplate.py

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
import LFPy
import neuron
import matplotlib.pyplot as plt


def plotcell(cell, color='k'):
    '''A plottin' function'''
    for sec in cell.allseclist:
        idx = cell.get_idx(sec.name())
        for i in idx:
            plt.plot(cell.x[i], cell.z[i], color=color)


# delete cell instances from previous script executions,
neuron.h('forall delete_section()')

# create some cell instances, set the positions, plot the morphologies
cell1 = LFPy.TemplateCell(
    morphology='morphologies/markram/CNG version/C010398B-P2.CNG.swc',
    templatefile='LFPyCellTemplate.hoc',
    templatename='LFPyCellTemplate',
    templateargs=None)
cell1.set_pos(0)
plotcell(cell=cell1, color='r')


cell2 = LFPy.TemplateCell(
    morphology='morphologies/markram/Source-Version/C010398B-P2.asc',
    templatefile='LFPyCellTemplate.hoc',
    templatename='LFPyCellTemplate',
    templateargs=None)
cell2.set_pos(200)
plotcell(cell=cell2, color='g')


cell3 = LFPy.TemplateCell(
    morphology='morphologies/markram/hoc-version/C010398B-P2.hoc',
    templatefile='LFPyCellTemplate.hoc',
    templatename='LFPyCellTemplate',
    templateargs=None)
cell3.set_pos(400)
plotcell(cell=cell3, color='b')
plt.show()
