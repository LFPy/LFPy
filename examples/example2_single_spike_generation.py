#!/usr/bin/env python

import pylab as pl
import LFPy
pl.rcParams.update({'font.size' : 12, 'figure.figsize' : [10, 10],
    'left': 0.1, 'wspace' : 0.5 ,'hspace' : 0.5})

#plot pops up by itself
pl.interactive(1)

################################################################################
# A couple of function declarations
################################################################################

def plotstuff(cell, electrode):
    fig = pl.figure()
    pl.subplot(221)
    pl.plot(cell.tvec,cell.somav)
    pl.xlabel('Time [ms]')
    pl.ylabel('Soma pot. [mV]')

    pl.subplot(223)
    for i in xrange(len(cell.synapses)):
        pl.plot(cell.tvec,cell.synapses[i].i,color=cell.synapses[i].color)
    pl.xlabel('Time [ms]')
    pl.ylabel('Syn. i [nA]')

    pl.subplot(122)
    for i in xrange(cell.xend.size):
        pl.plot([cell.xstart[i],cell.xend[i]],[cell.zstart[i],cell.zend[i]],color='k')
    for i in xrange(len(cell.synapses)):
        pl.plot([cell.synapses[i].x],[cell.synapses[i].z],\
            color=cell.synapses[i].color,marker=cell.synapses[i].marker)
    pl.plot(electrode.x, electrode.z, 'o')
    
    
    pl.axis('equal')
    pl.title('Morphology (XZ)')
    pl.xlabel(r'x [$\mu$m]')
    pl.ylabel(r'z [$\mu$m]')


################################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
################################################################################

cellParameters = {          #various cell parameters,
    'morphology' : 'morphologies/L5_Mainen96_wAxon_LFPy.hoc', #Mainen&Sejnowski, Nature, 1996
    'rm' : 30000,       #membrane resistance
    'cm' : 1.0,         #membrane capacitance
    'Ra' : 150,         #axial resistance
    'v_init' : -65,     #initial crossmembrane potential
    'e_pas' : -65,      #reversal potential passive mechs
    'passive' : True,   #switch on passive mechs
    'nsegs_method' : 'lambda_f',    #method for setting number of segments
    'lambda_f' : 100,               #frequency for segment
    'timeres_NEURON' : 2**-4,   #[ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2**-4,   #need binary representation
    'tstartms' : -10,  #start time of simulation, recorders start at t=0
    'tstopms' : 20,   #stop simulation at 1000 ms. these can be overridden
                        #by setting these arguments in cell.simulation()
    'custom_code'  : ['active_declarations.hoc'],    #Custom .hoc/.py-scripts
}

#Synaptic parameters
synapseParameters = {
    'idx' : 0,               #insert synapse on idx 0, the soma
    'e' : 0.,                #reversal potential
    'syntype' : 'Exp2Syn',   #conductance based exponential synapse
    'tau1' : 1.0,            #Time constant, rise
    'tau2' : 1.0,            #Time constant, decay
    'weight' : 0.03,         #Synaptic weight
    'record_current' : True, #Will enable synapse current recording
}


x = pl.linspace(-50, 50, 11)
z = pl.linspace(-50, 50, 11)
X, Z = pl.meshgrid(x, z)
y = pl.zeros(X.size)
electrodeParameters = {             #parameters for RecExtElectrode class
    'sigma' : 0.3,              #Extracellular potential
    'x' : X.reshape(-1),
    'y' : y,
    'z' : Z.reshape(-1)
}


################################################################################
# Main simulation procedure
################################################################################

pl.close('all')   #close open figures

#create extracellular electrode object
electrode = LFPy.RecExtElectrode(**electrodeParameters)

#Initialize cell instance, using the LFPy.Cell class
cell = LFPy.Cell(**cellParameters)

#attach synapse and set spike time
synapse = LFPy.Synapse(cell, **synapseParameters)
synapse.set_spike_times(pl.array([1]))

#perform NEURON simulation, results saved as attributes in the cell instance
cell.simulate(electrode = electrode, rec_isyn=True)



################################################################################

for i in xrange(cell.xend.size):
    if i == 0:
        xcoords = pl.array([cell.xmid[i]])
        ycoords = pl.array([cell.ymid[i]])
        zcoords = pl.array([cell.zmid[i]])
        diams = pl.array([cell.diam[i]])    
    else:
        if cell.zmid[i] < 70 and cell.zmid[i] > -70 and cell.xmid[i] < 70 and cell.xmid[i] > -70:
            xcoords = pl.r_[xcoords, pl.linspace(cell.xstart[i], cell.xend[i], cell.length[i]*3)]   
            ycoords = pl.r_[ycoords, pl.linspace(cell.ystart[i], cell.yend[i], cell.length[i]*3)]   
            zcoords = pl.r_[zcoords, pl.linspace(cell.zstart[i], cell.zend[i], cell.length[i]*3)]   
            diams = pl.r_[diams, pl.linspace(cell.diam[i], cell.diam[i], cell.length[i]*3)]   

argsort = pl.argsort(ycoords)
pl.scatter(xcoords[argsort], zcoords[argsort], s=diams[argsort]**2*25, c=ycoords[argsort], edgecolors='none', cmap='gray')
#for i in xrange(len(cell.synapses)):
#    pl.plot([cell.synapses[i].x], [cell.synapses[i].z], \
#        color='r', marker=cell.synapses[i].marker)
pl.plot(electrode.x, electrode.z, '.', marker='o', markersize=10)

i = 0
limLFP = abs(electrode.LFP).max()
for LFP in electrode.LFP:
    tvec = cell.tvec + electrode.x[i]
    trace = LFP + electrode.z[i]
    color = 'k'
    pl.plot(tvec, trace, color=color, lw = 2)
    i += 1

pl.axis([-55, 55, -55, 55])
pl.xlabel(r'x ($\mu$m)')
pl.ylabel(r'z ($\mu$m)')

