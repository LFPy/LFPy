#!/usr/bin/env python

import pylab as pl
import LFPy
#pl.rcParams.update({'font.size' : 10, 'figure.figsize' : [16,9],'wspace' : 0.5 ,'hspace' : 0.5})

#plot pops up by itself
pl.interactive(1)

################################################################################
# A couple of function declarations
################################################################################

#def plotstuff():
#    fig = pl.figure()
#    pl.subplot(321)
#    pl.plot(c.tvec,c.somav)
#    pl.xlabel('Time [ms]')
#    pl.ylabel('Soma pot. [mV]')
#
#    pl.subplot(323)
#    for i in xrange(len(c.synapses)):
#        pl.plot(c.tvec,c.synapses[i].i,color=c.synapses[i].color)
#    pl.xlabel('Time [ms]')
#    pl.ylabel('Syn. i [nA]')
#
#    pl.subplot(325)
#    absmaxLFP = abs(pl.array([e.LFP.max(),e.LFP.min()])).max()
#    pl.imshow(e.LFP,vmax=absmaxLFP/5,vmin=-absmaxLFP/5,origin='lower',
#           extent=(c.tvec[0],c.tvec[-1],e.z[0],e.z[-1]),cmap='jet_r',
#           interpolation='nearest')
#    pl.colorbar()
#    pl.axis('tight')
#    pl.xlabel('Time [ms]')
#    pl.ylabel('z [$\mu$m]')
#
#    pl.subplot(143)
#    for i in xrange(c.xend.size):
#        pl.plot([c.xstart[i],c.xend[i]],[c.zstart[i],c.zend[i]],color='k')
#    for i in xrange(len(c.synapses)):
#        pl.plot([c.synapses[i].x],[c.synapses[i].z],\
#            color=c.synapses[i].color,marker=c.synapses[i].marker)
#    for i in xrange(e.x.size):
#        pl.plot(e.x[i],e.z[i],color='g',marker='o')
#    pl.axis('equal')
#    pl.title('Morphology (XZ)')
#    pl.xlabel(r'x [$\mu$m]')
#    pl.ylabel(r'z [$\mu$m]')
#
#    pl.subplot(144)
#    for i in xrange(c.yend.size):
#        pl.plot([c.ystart[i],c.yend[i]],[c.zstart[i],c.zend[i]],color='k')
#    for i in xrange(len(c.synapses)):
#        pl.plot([c.synapses[i].y],[c.synapses[i].z],\
#            color=c.synapses[i].color,marker=c.synapses[i].marker)
#    for i in xrange(e.y.size):
#        pl.plot(e.y[i],e.z[i],color='g',marker='o')
#    pl.axis('equal')
#    pl.title('Morphology (YZ)')
#    pl.xlabel(r'y [$\mu$m]')

#def insert_synapses(synparams, section, n, spTimesFun, args):
#    #find n compartments to insert synapses onto
#    idx = c.get_rand_idx_area_norm(section=section, nidx=n)
#
#    #Insert synapses in an iterative fashion
#    for i in idx:
#        synparams.update({'idx' : int(i)})
#
#        # Some input spike train using the function call
#        spiketimes = spTimesFun(args[0], args[1], args[2], args[3])
#
#        # Create synapse(s) and setting times using the Synapse class in LFPy
#        s = LFPy.PointProcessSynapse(c,**synparams)
#        s.set_spike_times(c, spiketimes)

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
    'idx' : 0,              #insert synapse on idx 0, the soma
    'e' : 0.,               #reversal potential
    'syntype' : 'Exp2Syn',  #conductance based exponential synapse
    'tau1' : 0.1,            #Time constant, rise
    'tau2' : 1.0,            #Time constant, decay
    'weight' : 0.03,       #Synaptic weight
}


x = pl.arange(-50, 50, 21)
z = pl.arange(-50, 50, 21)
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
synapse.set_spike_times(cell, pl.array([-5]))

#perform NEURON simulation, results saved as attributes in the c instance
cell.simulate(electrode = electrode)



################################################################################

pl.plot(cell.tvec, cell.somav)


