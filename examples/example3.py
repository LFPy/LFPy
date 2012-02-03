#!/usr/bin/env python
################################################################################
#
# This is an example scripts using LFPy with an active cell model adapted from
# Mainen and Sejnowski, Nature 1996, for the original files, see
# http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=2488
#
# This scripts is set up to use the model, where the active conductances are set
# in the file "active_declarations_example3.hoc", and uses the mechanisms from
# the .mod-files provided here. For this example to work, run "nrnivmodl" in
# this folder to compile these mechanisms
# (i.e. /$PATHTONEURON/nrn/x86_64/bin/nrnivmodl).
#
################################################################################

# importing some modules, setting some matplotlib values for pl.plot.
import pylab as pl
import LFPy
pl.rcParams.update({'font.size' : 10, 'figure.figsize' : [16,9], 'wspace' : 0.5,
    'hspace' : 0.5})

#seed for random generation
pl.seed(9876543210)

#plot pops up by itself
pl.interactive(1)

################################################################################
# A couple of function declarations
################################################################################

def plotstuff():
    fig = pl.figure()
    pl.subplot(321)
    pl.plot(cell.tvec,cell.somav)
    pl.xlabel('Time [ms]')
    pl.ylabel('Soma pot. [mV]')

    pl.subplot(323)
    for i in xrange(len(cell.synapses)):
        pl.plot(cell.tvec,cell.synapses[i].i,color=cell.synapses[i].color)
    pl.xlabel('Time [ms]')
    pl.ylabel('Syn. i [nA]')

    pl.subplot(325)
    absmaxLFP = abs(pl.array([electrode.LFP.max(),electrode.LFP.min()])).max()
    pl.imshow(electrode.LFP,vmax=absmaxLFP/5,vmin=-absmaxLFP/5,origin='lower',
           extent=(cell.tvec[0],cell.tvec[-1],electrode.z[0],electrode.z[-1]),cmap='jet_r',
           interpolation='nearest')
    pl.colorbar()
    pl.axis('tight')
    pl.xlabel('Time [ms]')
    pl.ylabel('z [$\mu$m]')

    pl.subplot(143)
    for i in xrange(cell.xend.size):
        pl.plot([cell.xstart[i],cell.xend[i]],[cell.zstart[i],cell.zend[i]],color='k')
    for i in xrange(len(cell.synapses)):
        pl.plot([cell.synapses[i].x],[cell.synapses[i].z],\
            color=cell.synapses[i].color,marker=cell.synapses[i].marker)
    for i in xrange(electrode.x.size):
        pl.plot(electrode.x[i],electrode.z[i],color='g',marker='o')
    pl.axis('equal')
    pl.title('Morphology (XZ)')
    pl.xlabel(r'x [$\mu$m]')
    pl.ylabel(r'z [$\mu$m]')

    pl.subplot(144)
    for i in xrange(cell.yend.size):
        pl.plot([cell.ystart[i],cell.yend[i]],[cell.zstart[i],cell.zend[i]],color='k')
    for i in xrange(len(cell.synapses)):
        pl.plot([cell.synapses[i].y],[cell.synapses[i].z],\
            color=cell.synapses[i].color,marker=cell.synapses[i].marker)
    for i in xrange(electrode.y.size):
        pl.plot(electrode.y[i],electrode.z[i],color='g',marker='o')
    pl.axis('equal')
    pl.title('Morphology (YZ)')
    pl.xlabel(r'y [$\mu$m]')

def insert_synapses(synparams, section, n, spTimesFun, args):
    #find n compartments to insert synapses onto
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)

    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})

        # Some input spike train using the function call
        spiketimes = spTimesFun(args[0], args[1], args[2], args[3])

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell,**synparams)
        s.set_spike_times(spiketimes)

################################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
################################################################################

cellParameters = {          #various cell parameters,
    'morphology' : 'morphologies/L5_Mainen96_wAxon_LFPy.hoc', #Mainen&Sejnowski, Nature, 1996
    'rm' : 30000,       #membrane resistance
    'cm' : 1.0,        #membrane capacitance
    'Ra' : 150,         #axial resistance
    'v_init' : -65,     #initial crossmembrane potential
    'e_pas' : -65,      #reversal potential passive mechs
    'passive' : True,   #switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100,
    'timeres_NEURON' : 2**-3,   #[ms] dt's should be in powers of 2 for both,
    'timeres_python' : 2**-3,   #need binary representation
    'tstartms' : -100,  #start time of simulation, recorders start at t=0
    'tstopms' : 1000,   #stop simulation at 1000 ms. these can be overridden
                        #by setting these arguments in cell.simulation()
    'custom_code'  : ['active_declarations_example3.hoc'],    #Custom .hoc/.py-scripts
}

# Synaptic parameters taken from Hendrickson et al 2011
# Excitatory synapse parameters:
synapseParameters_AMPA = {
    'e' : 0,                    #reversal potential
    'syntype' : 'Exp2Syn',      #conductance based exponential synapse
    'tau1' : 1.,                #Time constant, rise
    'tau2' : 3.,                #Time constant, decay
    'weight' : 0.005,           #Synaptic weight
    'color' : 'r',              #for pl.plot
    'marker' : '.',             #for pl.plot
    'record_current' : True,    #record synaptic currents
}
# Excitatory synapse parameters
synapseParameters_NMDA = {         
    'e' : 0,
    'syntype' : 'Exp2Syn',
    'tau1' : 10.,
    'tau2' : 30.,
    'weight' : 0.005,
    'color' : 'm',
    'marker' : '.',
    'record_current' : True,
}
# Inhibitory synapse parameters
synapseParameters_GABA_A = {         
    'e' : -80,
    'syntype' : 'Exp2Syn',
    'tau1' : 1.,
    'tau2' : 12.,
    'weight' : 0.005,
    'color' : 'b',
    'marker' : '.',
    'record_current' : True
}
# where to insert, how many, and which input statistics
insert_synapses_AMPA_args = {
    'section' : 'apic',
    'n' : 100,
    'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
    'args' : [cellParameters['tstartms'], cellParameters['tstopms'], 2, 10]
}
insert_synapses_NMDA_args = {
    'section' : 'alldend',
    'n' : 10,
    'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
    'args' : [cellParameters['tstartms'], cellParameters['tstopms'], 5, 20]
}
insert_synapses_GABA_A_args = {
    'section' : 'dend',
    'n' : 100,
    'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
    'args' : [cellParameters['tstartms'], cellParameters['tstopms'], 2, 10]
}

# Define electrode geometry corresponding to a laminar electrode, where contact
# points have a radius r, surface normal vectors N, and LFP calculated as the
# average LFP in n random points on each contact:
N = pl.empty((16, 3))
for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
# put parameters in dictionary
electrodeParameters = {
    'sigma' : 0.3,              # Extracellular potential
    'x' : pl.zeros(16)+25,      # x,y,z-coordinates of electrode contacts
    'y' : pl.zeros(16),
    'z' : pl.linspace(-500,1000,16),
    'n' : 20,
    'r' : 10,
    'N' : N,
}

# Parameters for the cell.simulate() call, recording membrane- and syn.-currents
simulationParameters = {
    'rec_imem' : True,  # Record Membrane currents during simulation
    'rec_isyn' : True,  # Record synaptic currents
}

################################################################################
# Main simulation procedure
################################################################################

#close open figures
pl.close('all')   

#Initialize cell instance, using the LFPy.Cell class
cell = LFPy.Cell(**cellParameters)

#Insert synapses using the function defined earlier
insert_synapses(synapseParameters_AMPA, **insert_synapses_AMPA_args)
insert_synapses(synapseParameters_NMDA, **insert_synapses_NMDA_args)
insert_synapses(synapseParameters_GABA_A, **insert_synapses_GABA_A_args)

#perform NEURON simulation, results saved as attributes in the cell instance
cell.simulate(**simulationParameters)

# Initialize electrode geometry, then calculate the LFP, using the
# LFPy.RecExtElectrode class. Note that now cell is given as input to electrode
# and created after the NEURON simulations are finished
electrode = LFPy.RecExtElectrode(cell,**electrodeParameters)
electrode.calc_lfp()

#plotting some variables and geometry, saving output to .pdf.
plotstuff()
#pl.savefig('example3.pdf')
