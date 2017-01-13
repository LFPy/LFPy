#!/usr/bin/env python
'''
A very generic case where the same morphology on different file formats
is loaded in LFPy using a generic template specification, defined by
file LFPyCellTemplate.hoc
'''
import LFPy
import neuron
import numpy as np
import matplotlib.pyplot as plt

#A plottin' function
def plotcell(cell, color='k'):
    for sec in cell.cell.all:
        idx = cell.get_idx(sec.name())
        plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                 np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                 color=color)
    print(' ')

#delete cell instances from previous script executions,
neuron.h('forall delete_section()')

#create some cell instances, set the positions, plot the morphologies
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

