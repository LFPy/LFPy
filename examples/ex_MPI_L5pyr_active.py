#!/usr/bin/env python
from mpi4py import MPI
import pylab as pl
from time import time
import LFPy

#seed for random generation (dont want this with MPI)
#pl.seed(9876543210)

'''
This is an example using LFPy and MPI to run several cell models similar to
ex_L5pyr_active.py in parallel, with cells positioned different in space.

To execute this scripts in parallel issue in the terminal::
$ mpirun -n 8 python ex_MPI_L5pyr_active.py

On Stallo the job may be submitted using the corresponding .sh script::
$ qsub ex_MPI_L5pyr_active.sh

or using an interactive session:
$ export LD_PRELOAD=libmkl_intel_lp64.so:libmkl_intel_thread.so:libmkl_core.so:libguide.so
$ mpirun -n 8 -x LD_PRELOAD python ex_MPI_L5pyr_active.py

Note that -n 2 don't get speedups, because one core is used to send and receive
and not for actual simulations.
'''

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
    absmaxLFP = abs(pl.array([cell.LFP.max(),cell.LFP.min()])).max()
    pl.imshow(cell.LFP, vmax=absmaxLFP/5, vmin=-absmaxLFP/5, origin='lower',
           extent=(cell.tvec[0], cell.tvec[-1], cell.electrode.z[0], cell.electrode.z[-1]),cmap='jet_r',
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
    for i in xrange(cell.electrode.x.size):
        pl.plot(cell.electrode.x[i],cell.electrode.z[i],color='g',marker='o')
    pl.axis('equal')
    pl.title('Morphology (XZ)')
    pl.xlabel(r'x [$\mu$m]')
    pl.ylabel(r'z [$\mu$m]')

    pl.subplot(144)
    for i in xrange(cell.yend.size):
        pl.plot([cell.ystart[i], cell.yend[i]], [cell.zstart[i], cell.zend[i]], color='k')
    for i in xrange(len(cell.synapses)):
        pl.plot([cell.synapses[i].y], [cell.synapses[i].z], \
            color=cell.synapses[i].color, marker=cell.synapses[i].marker)
    for i in xrange(cell.electrode.y.size):
        pl.plot(cell.electrode.y[i],cell.electrode.z[i], color='g', marker='o')
    pl.axis('equal')
    pl.title('Morphology (YZ)')
    pl.xlabel(r'y [$\mu$m]')

def insert_synapses(cell, synparams, section, n, spTimesFun, args):
    #find n compartments to insert synapses onto
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)

    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})

        # Some input spike train using the function call
        spiketimes = spTimesFun(args[0], args[1], args[2], args[3])

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.PointProcessSynapse(cell,**synparams)
        s.set_spike_times(cell, spiketimes)

def cellsim(cellposition={'xpos' : 0, 'ypos' : 0, 'zpos' : 0}):

    #Initialize cell instance, using the LFPy.Cell class
    cell = LFPy.CellWithElectrode(**cellParameters)
    cell.set_pos(**cellposition)
    #Insert synapses
    insert_synapses(cell, synparams_AMPA, **insert_synapses_AMPA_args)
    insert_synapses(cell, synparams_NMDA, **insert_synapses_NMDA_args)
    insert_synapses(cell, synparams_GABA_A, **insert_synapses_GABA_A_args)
    
    #perform NEURON simulation, results saved as attributes in the cell instance
    simulateParameters.update(electrodeParameters)
    cell.simulate(**simulateParameters)
    
    #for some reason, returning the cell-object is problematic, so some arrays
    #are returned
    return [cell.tvec, cell.somav, cell.LFP]

################################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
################################################################################

cellParameters = {          #various cell parameters,
    'morphology' : 'L5_Mainen96_wAxon_LFPy.hoc', #Mainen&Sejnowski, Nature, 1996
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
    'tstopms' : 200,   #stop simulation at 1000 ms. these can be overridden
                        #by setting these arguments in cell.simulation()
    'custom_code'  : ['my_active_declarations.hoc'],    #Custom .hoc/.py-scripts
}
#Synaptic parameters taken from Hendrickson et al 2011
synparams_AMPA = {         #Excitatory synapse parameters
    'e' : 0,           #reversal potential
    'syntype' : 'Exp2Syn',   #conductance based exponential synapse
    'tau1' : 1.,         #Time constant, rise
    'tau2' : 3.,         #Time constant, decay
    'weight' : 0.005,   #Synaptic weight
    'color' : 'r',      #for pl.plot
    'marker' : '.',     #for pl.plot
    'record_current' : True,    #record synaptic currents
}
synparams_NMDA = {         #Excitatory synapse parameters
    'e' : 0,           #reversal potential
    'syntype' : 'Exp2Syn',   #conductance based exponential synapse
    'tau1' : 10.,         #Time constant, rise
    'tau2' : 30.,         #Time constant, decay
    'weight' : 0.005,   #Synaptic weight
    'color' : 'm',      #for pl.plot
    'marker' : '.',     #for pl.plot
    'record_current' : True,    #record synaptic currents
}
synparams_GABA_A = {         #Inhibitory synapse parameters
    'e' : -80,
    'syntype' : 'Exp2Syn',
    'tau1' : 1.,
    'tau2' : 12.,
    'weight' : 0.005,
    'color' : 'b',
    'marker' : '.',
    'record_current' : True
}
#where to insert, how many, and which input statistics
insert_synapses_AMPA_args = {
    'section' : 'apic',
    'n' : 100,
    'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
    'args' : [cellParameters['tstartms'], cellParameters['tstopms'], 2, 20]
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
    'args' : [cellParameters['tstartms'], cellParameters['tstopms'], 2, 20]
}

N = pl.empty((62, 3))
for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
electrodeParameters = {             #parameters for electrode class
    'sigma' : 0.3,              #Extracellular potential
    'x' : pl.zeros(62)+25,      #Coordinates of electrode contacts
    'y' : pl.zeros(62),
    'z' : pl.linspace(-500,1000,62),
    'n' : 20,
    'r' : 10,
    'N' : N,
}
simulateParameters = {      #parameters for NEURON simulation
#    'rec_v' : True,    #record membrane potential for all compartments
#    'rec_isyn' : True,  #Record synaptic currents
#    'rec_ipas' : True, #record ohmic currents
#    'rec_icap' : True, #record capacitive currents
}

################################################################################
# Main simulation procedure
################################################################################

t0 = time()

POPULATION_SIZE = 7

cellpositions = [
    {'xpos' : -20, 'ypos' : 0, 'zpos' : -150},
    {'xpos' : 0, 'ypos' : 20, 'zpos' : -100},
    {'xpos' : 20, 'ypos' : 0, 'zpos' : -50},
    {'xpos' : 0, 'ypos' : -20, 'zpos' : 0},
    {'xpos' : -20, 'ypos' : 0, 'zpos' : 50},
    {'xpos' : 0, 'ypos' : 20, 'zpos' : 100},
    {'xpos' : 20, 'ypos' : 0, 'zpos' : 150},
]

#Initialization of MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
master_mode = comm.rank == 0
print 'size %i, rank %i, master_mode: %s' % (size, rank, str(master_mode))

#initialize workers
if not master_mode and size > 1:
    while(True):
        #receive parameters from master
        parameters = comm.recv(source=0)
        if parameters == None:
            break
        else:
            #send simulation results to master
            comm.send(cellsim(parameters), dest=0)

#master will send parameters to workers and receive simulation results unordered
if master_mode:
    #init container for simulation results
    outputs = []
    #using MPI
    if size > 1:
        dest = 1 #counter on [1, size-1]
        #sending parameters to workers
        for i in xrange(POPULATION_SIZE):
            parameters = cellpositions[i]
            comm.send(parameters, dest=dest)
            dest += 1
            if dest >= size:
                dest = 1
        #receiving simulation results from any worker, storing in container
        for i in xrange(POPULATION_SIZE):
            output = comm.recv(source=MPI.ANY_SOURCE)
            outputs.append(output)
        
        #killing workers
        for i in xrange(1, size):
            comm.send(None, dest=i)
    
    #serial mode
    else:
        for i in xrange(POPULATION_SIZE):
            parameters = cellpositions[i]
            output = cellsim(parameters)
            outputs.append(output)
    
    print 'execution time: %.3f seconds' %  (time()-t0)
    
    #plot some output
    pl.figure()
    i = 0
    for output in outputs:
        pl.plot(output[0], output[1], label='cell %i' % i)
        i += 1
    pl.xlabel('time (ms)')
    pl.ylabel(r'$V_mathrm{soma}$ (mV)')
    pl.legend()
    pl.savefig('ex_MPI_somatraces.pdf')
    
    
    LFPsum = pl.zeros(outputs[0][2].shape)
    for output in outputs:
        LFPsum += output[2]
    pl.figure()
    pl.imshow(LFPsum, origin='bottom', cmap='jet_r',
              vmin=-abs(LFPsum).max(), vmax=abs(LFPsum).max(),
              interpolation='nearest')
    pl.axis('tight')
    pl.colorbar()
    pl.title('Extracellular Field Potentials')
    pl.savefig('ex_MPI_LFPs.pdf')
    
        
    