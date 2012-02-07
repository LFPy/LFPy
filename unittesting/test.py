#!/usr/bin/env python
import LFPy
import matplotlib.pylab as pl

pl.interactive(1)
pl.close('all')

stickParams = {
    'morphology' : 'stick.hoc',
    'rm' : 30000,
    'cm' : 1,
    'Ra' : 150,
    'tstartms' : -100,
    'tstopms' : 100,
    'timeres_python' : 2**-4,
    'timeres_NEURON' : 2**-4,
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 1000,
    
}

electrodeParams = {
    'sigma' : 0.3,
    'x' : pl.ones(101) * 100.,
    'y' : pl.zeros(101),
    'z' : pl.linspace(1000, 0, 101),
    'method' : 'linesource'
}

stimParams = {
    'pptype' : 'SinSyn',
    'delay' : -100.,
    'dur' : 1000.,
    'pkamp' : 1.,
    'freq' : 100.,
    'phase' : -pl.pi/2,
    'bias' : 0.,
    'record_current' : True
}


electrode = LFPy.RecExtElectrode(**electrodeParams)

stick = LFPy.Cell(**stickParams)

synapse = LFPy.StimIntElectrode(stick, stick.get_closest_idx(0, 0, 1000),
                       **stimParams)
stick.simulate(electrode, rec_imem=True, rec_istim=True, rec_vmem=True)

pl.subplot(221)
pl.plot(stick.tvec, stick.pointprocesses[0].i)
pl.xlabel('time (ms)')
pl.ylabel('$i_\mathrm{electrode}$ (nA)')

pl.subplot(222)
for i in xrange(stick.totnsegs):
    pl.plot([stick.xstart[i], stick.xend[i]],
            [stick.zstart[i], stick.zend[i]],
             'k', lw=stick.diam[i])
pl.plot(synapse.x, synapse.z, '.', color='r', marker='o')
pl.plot(electrode.x, electrode.z, '.', color='b', marker='o')
pl.axis('equal')
pl.xlabel('x ($\mu$m)')
pl.ylabel('z ($\mu$m)')

pl.subplot(212)
pl.imshow(electrode.LFP, cmap='jet_r', interpolation='nearest')
pl.axis('tight')
cbar = pl.colorbar()
cbar.set_label('LFP (mV)')
