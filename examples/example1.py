#!/usr/bin/env python
'''
Example plot for LFPy: Single-synapse contribution to the LFP
'''
import LFPy
import numpy as np
import os
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import zipfile
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from os.path import join
plt.interactive(1)
plt.close('all')


#Fetch Mainen&Sejnowski 1996 model files
if not os.path.isfile(join('cells', 'cells', 'j4a.hoc')):
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=2488&a=23&mime=application/zip')
    localFile = open('patdemo.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('patdemo.zip', 'r')
    myzip.extractall('.')
    myzip.close()



################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################

# Define cell parameters
cell_parameters = {          # various cell parameters,
    'morphology' : join('cells', 'cells', 'j4a.hoc'), # Mainen&Sejnowski, 1996
    'rm' : 30000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : True,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100.,
    'timeres_NEURON' : 2.**-3,   # [ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2.**-3,   # need binary representation
    'tstartms' : 0.,    # start time of simulation, recorders start at t=0
    'tstopms' : 100.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
}

# Create cell
cell = LFPy.Cell(**cell_parameters)
# Align cell
cell.set_rotation(x=4.99, y=-4.33, z=3.14)

# Define synapse parameters
synapse_parameters = {
    'idx' : cell.get_closest_idx(x=-200., y=0., z=800.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 5.,                 # synaptic time constant
    'weight' : .001,            # synaptic weight
    'record_current' : True,    # record synapse current
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([20.]))


# Create a grid of measurement locations, in (mum)
X, Z = np.mgrid[-700:701:50, -400:1201:50]
Y = np.zeros(X.shape)

# Define electrode parameters
grid_electrode_parameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()
}


# Define electrode parameters
point_electrode_parameters = {
    'sigma' : 0.3,  # extracellular conductivity
    'x' : np.array([-130., -220.]),
    'y' : np.array([   0.,    0.]),
    'z' : np.array([   0.,  700.]),
}



# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True,rec_isyn=True)

# Create electrode objects
grid_electrode = LFPy.RecExtElectrode(cell,**grid_electrode_parameters)
point_electrode = LFPy.RecExtElectrode(cell,**point_electrode_parameters)

# Calculate LFPs
grid_electrode.calc_lfp()
point_electrode.calc_lfp()

print("done")

#create a plot
#fig = prepfig(1, 8.5, 6.)
fig = plt.figure()

ax = fig.add_axes([.4,.1,.55,.8], aspect='equal', frameon=False)
cax = fig.add_axes([0.85, 0.4, 0.01, 0.2], frameon=False)

LFP = np.max(np.abs(grid_electrode.LFP),1).reshape(X.shape)
im = ax.contour(X, Z, np.log10(LFP), 
            50,
           #V=np.arange(-25.,-10.,0.1),
           cmap='rainbow',
           #linewidths=2,
           zorder=-2)
#cbar = fig.colorbar(im, fraction=0.1, shrink=0.4)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('$|\Phi(\mathbf{r}, t)|_\mathrm{max}$ (nV)')
cbar.outline.set_visible(False)
#get some log-linear tickmarks and ticklabels
ticks = np.arange(np.ceil(np.log10(LFP.min())), np.floor(np.log10(LFP.max())))
cbar.set_ticks(ticks)
cbar.set_ticklabels(10.**ticks * 1E6) #mV -> nV

#plot morphology
zips = []
for x, z in cell.get_idx_polygons():
    zips.append(list(zip(x, z)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='k')
ax.add_collection(polycol)

ax.plot([100, 200], [-400, -400], 'k', lw=1, clip_on=False)
ax.text(150, -470, r'100$\mu$m', va='center', ha='center')

ax.axis('off')


ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=5,
        markeredgecolor='k',
        markerfacecolor='r')

color_vec = ['blue','green']
for i in range(2):
    ax.plot(point_electrode_parameters['x'][i],
                    point_electrode_parameters['z'][i],'o',ms=6,
                    markeredgecolor='none',
                    markerfacecolor=color_vec[i])

plt.axes([.15,.10,.25,.25])
plt.plot(cell.tvec,point_electrode.LFP[0]*1e6,color=color_vec[0], clip_on=False)
plt.ylabel(r'$\Phi_2(\mathbf{r},t)$ (nV)') #rotation='horizontal')
plt.xlabel(r'$t$ (ms)')
plt.axis('tight')
ax = plt.gca()
for loc, spine in ax.spines.items():
    if loc in ['right', 'top']:
        spine.set_color('none')            
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


plt.axes([.15,.375,.25,.25])
plt.plot(cell.tvec,point_electrode.LFP[1]*1e6,color=color_vec[1], clip_on=False)
plt.ylabel(r'$\Phi_1(\mathbf{r}, t)$ (nV)') #rotation='horizontal')
plt.axis('tight')
ax = plt.gca()
for loc, spine in ax.spines.items():
    if loc in ['right', 'top']:
        spine.set_color('none')            
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticklabels([])

plt.axes([.15,.65,.25,.25])
plt.plot(cell.tvec,synapse.i*1E3, color='red', clip_on=False)
plt.ylabel(r'$i_\mathrm{syn}(t)$ (pA)') #, rotation='horizontal')
plt.axis('tight')
ax = plt.gca()
for loc, spine in ax.spines.items():
    if loc in ['right', 'top']:
        spine.set_color('none')            
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticklabels([])

# fig = plot_ex1(cell, electrode, X, Y, Z)
# Optionally save figure (uncomment the line below)
plt.savefig('example1.pdf', dpi=300)
