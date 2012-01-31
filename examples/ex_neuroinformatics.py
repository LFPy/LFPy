#!/usr/bin/env python
import LFPy
import pylab as pl

#plot will pop up itself
pl.interactive(1)

cellParameters = {                          
    'morphology' : 'morphologies/L5_Mainen96_LFPy.hoc',  # morphology file
    'rm' : 30000,                           # membrane resistivity
    'cm' : 1.0,                             # membrane capacitance
    'Ra' : 150,                             # axial resistivity
    'timeres_NEURON' : 0.1,                 # dt for NEURON sim.
    'timeres_python' : 0.1,                 # dt for python output
    'tstartms' : -50,                       # start t of simulation
    'tstopms' : 50,                         # end t of simulation
}

cell = LFPy.Cell(**cellParameters)

synapseParameters = {
    'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
    'e' : 0,                                # reversal potential
    'syntype' : 'ExpSyn',                   # synapse type
    'tau' : 2,                              # syn. time constant
    'weight' : 0.01,                       # syn. weight
    'record_current' : True                 # syn. current record
}

synapse = LFPy.Synapse(cell, **synapseParameters)
synapse.set_spike_times(cell, pl.array([10, 15, 20, 25]))

cell.simulate(rec_imem=True, rec_isyn=True)                # run cell simulation

electrodeParameters = {
    'sigma' : 0.3,                          # conductivity
    'x' : pl.zeros(24) + 50,                # carthesian coords.
    'y' : pl.zeros(24),                     # of electrode points
    'z' : pl.arange(-200, 1000, 50),
}

electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
electrode.calc_lfp()               # run LFP simulation


def plotstuff():
    fig = pl.figure()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    pl.subplot(321)
    pl.plot(cell.tvec,cell.somav)
    pl.axis('tight')
    #pl.xlabel('Time (ms)')
    pl.ylabel(r'$V_\mathrm{soma}$ (mV)')
    pl.title('Somatic Potential')

    pl.subplot(323)
    for i in xrange(len(cell.synapses)):
        pl.plot(cell.tvec,cell.synapses[i].i,color=cell.synapses[i].color)
    pl.axis('tight')
    #pl.xlabel('Time (ms)')
    pl.ylabel(r'$i_\mathrm{syn.}$(nA)')
    pl.title('Synaptic Current')


    pl.subplot(325)
    absmaxLFP = abs(pl.array([electrode.LFP.max(),electrode.LFP.min()])).max()
    pl.imshow(electrode.LFP*1000,vmax=absmaxLFP*1000,vmin=-absmaxLFP*1000,origin='lower',
           extent=(cell.tvec[0],cell.tvec[-1],electrode.z[0],electrode.z[-1]),cmap='jet_r',
           interpolation='nearest')
    pl.xlabel('Time (ms)')
    pl.ylabel('z ($\mu$m)')
    pl.title('Extracellular Potential')
    pl.axis('tight')
    cb = pl.colorbar()
    cb.set_label('LFP ($\mu$V)')
    
    pl.subplot(122)
    for i in xrange(cell.xend.size):
        pl.plot([cell.xstart[i],cell.xend[i]],[cell.zstart[i],cell.zend[i]],color='k')
    for i in xrange(len(cell.synapses)):
        pl.plot([cell.synapses[i].x],[cell.synapses[i].z],\
            color=cell.synapses[i].color,marker=cell.synapses[i].marker, markersize=10)
    for i in xrange(electrode.x.size):
        pl.plot(electrode.x[i],electrode.z[i],color='g',marker='o')
    pl.axis('equal')
    pl.title('Morphology')
    pl.xlabel(r'x ($\mu$m)')
    pl.ylabel(r'z ($\mu$m)')


plotstuff()