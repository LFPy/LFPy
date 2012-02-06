#!/usr/bin/env python
import LFPy
import matplotlib.pylab as pl

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
    'lambda_f' : 200,
    
}

electrodeParams = {
    'sigma' : 0.3,
    'x' : pl.ones(101) + 10.,
    'y' : pl.zeros(101),
    'z' : pl.linspace(1000, 0, 1001),
}

stimParams = {
    'pptype' : 'SinSyn',
    'delay' : 0.,
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

pl.subplot(211)
pl.plot(stick.tvec, stick.pointprocesses[0].i)

pl.subplot(212)
pl.imshow(electrode.LFP, cmap='jet_r')
pl.axis('tight')
pl.colorbar()

pl.show()