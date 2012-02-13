#!/usr/bin/env python
################################################################################
#
# This is an example scripts using LFPy with an active cell model adapted from
# Mainen and Sejnowski, Nature 1996, for the original files, see
# http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=2488
#
# This scripts is set up to use the model, where the active conductances are set
# in the file "active_declarations_example2.hoc", and uses the mechanisms from
# the .mod-files provided here. For this example to work, run "nrnivmodl" in
# this folder to compile these mechanisms
# (i.e. /$PATHTONEURON/nrn/x86_64/bin/nrnivmodl).
#
# A single excitatory synapse drive the neuron into producing a single action-
# potential, and the local field potential are calculated on a dense 2D-grid
# on around the soma.
#
################################################################################
#import some plotting stuff and the LFPy-module
import matplotlib.pylab as pl
import LFPy

#set some plotting parameters
pl.rcParams.update({'font.size' : 15,
    'figure.facecolor' : '1',
    'left': 0.1, 'wspace' : 0.5 ,'hspace' : 0.5})


################################################################################
# Function declarations
################################################################################

def plotstuff(cell, electrode):
    #creating array of points and corresponding diameters along structure
    for i in xrange(cell.xend.size):
        if i == 0:
            xcoords = pl.array([cell.xmid[i]])
            ycoords = pl.array([cell.ymid[i]])
            zcoords = pl.array([cell.zmid[i]])
            diams = pl.array([cell.diam[i]])    
        else:
            if cell.zmid[i] < 100 and cell.zmid[i] > -100 and \
                    cell.xmid[i] < 100 and cell.xmid[i] > -100:
                xcoords = pl.r_[xcoords, pl.linspace(cell.xstart[i],
                                            cell.xend[i], cell.length[i]*3)]   
                ycoords = pl.r_[ycoords, pl.linspace(cell.ystart[i],
                                            cell.yend[i], cell.length[i]*3)]   
                zcoords = pl.r_[zcoords, pl.linspace(cell.zstart[i],
                                            cell.zend[i], cell.length[i]*3)]   
                diams = pl.r_[diams, pl.linspace(cell.diam[i], cell.diam[i],
                                            cell.length[i]*3)]
    
    #sort along depth-axis
    argsort = pl.argsort(ycoords)
    
    #plotting
    fig = pl.figure(figsize=[15, 10])
    ax = fig.add_axes([0.1, 0.1, 0.533334, 0.8], frameon=False)
    ax.scatter(xcoords[argsort], zcoords[argsort], s=diams[argsort]**2*20,
               c=ycoords[argsort], edgecolors='none', cmap='gray')
    ax.plot(electrode.x, electrode.z, '.', marker='o', markersize=5, color='k')
    
    i = 0
    limLFP = abs(electrode.LFP).max()
    for LFP in electrode.LFP:
        tvec = cell.tvec*0.6 + electrode.x[i] + 2
        if abs(LFP).max() >= 1:
            factor = 2
            color='r'
        elif abs(LFP).max() < 0.25:
            factor = 50
            color='b'
        else:
            factor = 10
            color='g'
        trace = LFP*factor + electrode.z[i]
        ax.plot(tvec, trace, color=color, lw = 2)
        i += 1
    
    ax.plot([22, 28], [-60, -60], color='k', lw = 3)
    ax.text(22, -65, '10 ms')
    
    ax.plot([40, 50], [-60, -60], color='k', lw = 3)
    ax.text(42, -65, '10 $\mu$m')
    
    ax.plot([60, 60], [20, 30], color='r', lw=2)
    ax.text(62, 20, '5 mV')
    
    ax.plot([60, 60], [0, 10], color='g', lw=2)
    ax.text(62, 0, '1 mV')
    
    ax.plot([60, 60], [-20, -10], color='b', lw=2)
    ax.text(62, -20, '0.1 mV')
    
    
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.axis([-61, 61, -61, 61])
    
    ax.set_title('Location-dependent extracellular spike shapes')
    
    #plotting the soma trace    
    ax = fig.add_axes([0.75, 0.55, 0.2, 0.35])
    ax.plot(cell.tvec, cell.somav)
    ax.set_title('Somatic action-potential')
    ax.set_ylabel(r'$V_\mathrm{membrane}$ (mV)')
    
    #plotting the synaptic current
    ax = fig.add_axes([0.75, 0.1, 0.2, 0.35])
    ax.plot(cell.tvec, cell.synapses[0].i)
    ax.set_title('Synaptic current')
    ax.set_ylabel(r'$i_\mathrm{synapse}$ (nA)')
    ax.set_xlabel(r'time (ms)')

################################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
################################################################################

#define cell parameters used as input to cell-class
cellParameters = {
    'morphology' : 'morphologies/L5_Mainen96_wAxon_LFPy.hoc',
    'rm' : 30000,               # membrane resistance
    'cm' : 1.0,                 # membrane capacitance
    'Ra' : 150,                 # axial resistance
    'v_init' : -65,             # initial crossmembrane potential
    'e_pas' : -65,              # reversal potential passive mechs
    'passive' : True,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 500,           # segments are isopotential at this frequency
    'timeres_NEURON' : 2**-5,   # dt of LFP and NEURON simulation.
    'timeres_python' : 2**-5,
    'tstartms' : -10,           #start time, recorders start at t=0
    'tstopms' : 10,             #stop time of simulation
    'custom_code'  : ['active_declarations_example2.hoc'], # will run this file
}

#Synaptic parameters, corresponding to a NetCon synapse built into NEURON
synapseParameters = {
    'idx' : 0,               # insert synapse on index "0", the soma
    'e' : 0.,                # reversal potential of synapse
    'syntype' : 'Exp2Syn',   # conductance based double-exponential synapse
    'tau1' : 1.0,            # Time constant, rise
    'tau2' : 1.0,            # Time constant, decay
    'weight' : 0.03,         # Synaptic weight
    'record_current' : True, # Will enable synapse current recording
}

#Generate the grid in xz-plane over which we calculate local field potentials
x = pl.linspace(-50, 50, 11)
z = pl.linspace(-50, 50, 11)
X, Z = pl.meshgrid(x, z)
y = pl.zeros(X.size)

#define parameters for extracellular recording electrode, using optional method
electrodeParameters = {
    'sigma' : 0.3,              # extracellular conductivity
    'x' : X.reshape(-1),        # x,y,z-coordinates of contact points
    'y' : y,
    'z' : Z.reshape(-1),
    'method' : 'som_as_point',  #treat soma segment as sphere source
}

################################################################################
# Main simulation procedure, setting up extracellular electrode, cell, synapse
################################################################################

#close open figures
pl.close('all')

#create extracellular electrode object
electrode = LFPy.RecExtElectrode(**electrodeParameters)

#Initialize cell instance, using the LFPy.Cell class
cell = LFPy.Cell(**cellParameters)
#set the position of midpoint in soma to Origo (not needed, this is the default)
cell.set_pos(xpos = 0, ypos = 0, zpos = 0)
#rotate the morphology 90 degrees around z-axis
cell.set_rotation(z = pl.pi/2)

#attach synapse with parameters and set spike time
synapse = LFPy.Synapse(cell, **synapseParameters)
synapse.set_spike_times(pl.array([1]))

#perform NEURON simulation, results saved as attributes in the cell instance
cell.simulate(electrode = electrode, rec_isyn=True)

# Plotting of simulation results:
plotstuff(cell, electrode)
#pl.savefig('example2.pdf')

pl.show()


