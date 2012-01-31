#!/usr/bin/env python
'''Copyright (C) 2012 Computational Neuroscience Group, UMB.
All rights reserved.'''
import cPickle
import pylab as pl
import scipy.signal as ss

def load(filename):
    '''Generic loading of cPickled objects from file'''
    filen = open(filename,'rb')
    obj = cPickle.load(filen)
    filen.close()
    return obj

def noise_brown(timevector, nrows=1, weight=1, highpassfilter=False):
    '''Return 1/f^2 noise of shape(nrows, timevector.size) obtained by taking 
    the cumulative sum of gaussian white noise, with rms weight. If 
    highpassfilter=True, the noise is high-pass filtered with a very low cutoff
    frequency'''
    if highpassfilter:
        fcut = 4./timevector.size
        [coeff_b, coeff_a] = ss.butter(1, fcut, btype='high')
    
    noise = pl.empty((nrows, timevector.size))    
    for i in xrange(nrows):
        signal = pl.normal(size=timevector.size+10000).cumsum()
        if highpassfilter:
            signal = ss.lfilter(coeff_b, coeff_a, signal)
        signal /= pl.rms_flat(signal)
        signal *= weight
        noise[i, :] = signal[10000:]
    
    return noise
    