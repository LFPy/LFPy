#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import cPickle
import pylab as pl

def load(filename):
    '''generic loading of cPickled objects from file'''
    filen = open(filename,'rb')
    obj = cPickle.load(filen)
    filen.close()
    return obj

def noise_brown(t, n=1, weight=1):
    '''return 1/f^2 noise of shape(n, t.size) obtained by taking the
    cumulative sum of gaussian white noise, with rms weight'''
    noise = []
    for i in xrange(n):
        x = pl.normal(size=t.size).cumsum()
        x /= pl.rms_flat(x)
        x *= weight
        noise.append(x)
    
    return pl.array(noise)
    