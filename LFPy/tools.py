#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import cPickle
import pylab as pl
import scipy.signal as ss

def load(filename):
    '''generic loading of cPickled objects from file'''
    filen = open(filename,'rb')
    obj = cPickle.load(filen)
    filen.close()
    return obj

def noise_brown(t, n=1, weight=1, cutDC=False):
    '''return 1/f^2 noise of shape(n, t.size) obtained by taking the
    cumulative sum of gaussian white noise, with rms weight. If cutDC=True,
    the noise is high-pass filtered with a very low cutoff frequency'''
    noise = []
    if cutDC:
        fcut = 4./t.size
        [b, a] = ss.butter(1, fcut, btype='high')
        
    for i in xrange(n):
        x = pl.normal(size=t.size+10000).cumsum()
        if cutDC:
            x = ss.lfilter(b, a, x)
        x /= pl.rms_flat(x)
        x *= weight
        noise.append(x[10000:])
        
    return pl.array(noise)
    