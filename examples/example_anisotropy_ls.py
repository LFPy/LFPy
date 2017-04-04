#!/usr/bin/env python
'''
Example plot for LFPy: Single-synapse contribution to the LFP
'''
import LFPy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from os.path import join


cell_parameters = {          # various cell parameters,
    'morphology' : join('morphologies', 'ball_and_stick.hoc'), # Mainen&Sejnowski, 1996
    'rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : True,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 5000.,
    'dt' : 2.**-2,      # simulation time step size
    'tstartms' : 0.,    # start time of simulation, recorders start at t=0
    'tstopms' : 2.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
}

# Create cell
cell = LFPy.Cell(**cell_parameters)
cell.set_pos(ypos=5.0, xpos=np.min(cell.xstart))

print cell.totnsegs

synapse_parameters = {
    'idx' : 0,
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 5.,                 # synaptic time constant
    'weight' : 1,            # synaptic weight
    'record_current' : False,    # record synapse current
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([1.]))

# Create a grid of measurement locations, in (um)
dx = 1

X, Z = np.mgrid[-50:50 + dx:dx, -10:10 + dx:dx]

# X = np.array([0])
# Z = np.array([12])

Y = np.zeros(X.shape)

sigma = 0.3
sigma_tensor = [0.3, 0.3, 0.45]

# Define electrode parameters
grid_electrode_parameters = {
    'sigma' : sigma,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten(),
    'method': 'som_as_point'
}

grid_electrode_parameters_tensor = {
    'sigma' : sigma_tensor,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten(),
    'method': 'som_as_point'
}

# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_isyn=False)

# cell.imem = np.ones((cell.imem.shape[0], 1))
# cell.imem[0, :] = -1
# cell.imem[1, :] = 1
# Create electrode objects

import time

t0 = time.time()
grid_electrode = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
grid_electrode.calc_lfp()
print("Original: %f" % (time.time() - t0))

t0 = time.time()
grid_electrode_tensor = LFPy.RecExtElectrode(cell, **grid_electrode_parameters_tensor)
grid_electrode_tensor.calc_lfp()
print("Tensor: %f" % (time.time() - t0))



print np.max(np.abs((grid_electrode.LFP - grid_electrode_tensor.LFP) / np.max(np.abs(grid_electrode.LFP))))

max_idx = np.argmax(np.abs(cell.imem[0,:]))

LFP = grid_electrode.LFP[:, max_idx].reshape(X.shape)
LFP_tensor = grid_electrode_tensor.LFP[:, max_idx].reshape(X.shape)


max = np.max(np.abs(np.r_[LFP, LFP_tensor]))
LFP /= max
LFP_tensor /= max

difference = LFP - LFP_tensor

fig = plt.figure(figsize=[10, 8])
fig.subplots_adjust(top=0.97, hspace=0.4)
ax = fig.add_subplot(311, aspect='equal', xlabel='x', ylabel='z', title="Sigma: %s S/m" % str(sigma),
                     ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                     xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])

im = ax.contourf(X, Z, LFP, 151, vmin=-np.max(np.abs(LFP)) / 1, vmax=np.max(np.abs(LFP)) / 1,
                 cmap='bwr',
                 zorder=-2)
cbar = plt.colorbar(im)

[plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]]) for idx in range(cell.totnsegs)]

ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=5,
        markeredgecolor='k',
        markerfacecolor='r')

ax2 = fig.add_subplot(312, aspect='equal', xlabel='x', ylabel='z', title="Sigma: %s S/m" % str(sigma_tensor),
                     ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                     xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])
[ax2.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]]) for idx in range(cell.totnsegs)]
ax2.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=5,
        markeredgecolor='k',
        markerfacecolor='r')
im = ax2.contourf(X, Z, LFP_tensor, 151, vmin=-np.max(np.abs(LFP_tensor)) / 1, vmax=np.max(np.abs(LFP_tensor )) / 1,
                  cmap='bwr',
                  zorder=-2)
cbar = plt.colorbar(im)

ax3 = fig.add_subplot(313, aspect='equal', xlabel='x', ylabel='z', title="Max difference:\n%f %s" % (100 * np.max(np.abs(difference)), "%") ,
                     ylim=[np.min(grid_electrode.z), np.max(grid_electrode.z)],
                     xlim=[np.min(grid_electrode.x), np.max(grid_electrode.x)])
# [ax3.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]]) for idx in range(cell.totnsegs)]

im = ax3.contourf(X, Z, difference, 151, vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)),
                  cmap='bwr',
                  zorder=-2)
cbar = plt.colorbar(im)

# ax3.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=5,
#         markeredgecolor='k',
#         markerfacecolor='r')

plt.savefig('example_anisotropy_soma_as_point.png', dpi=150)