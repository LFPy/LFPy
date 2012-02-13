#!/usr/bin/env python
'''Copyright (C) 2012 Computational Neuroscience Group, UMB.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.'''

import cPickle
import pylab as pl
import scipy.signal as ss

def load(filename):
    '''Generic loading of cPickled objects from file'''
    filen = open(filename,'rb')
    obj = cPickle.load(filen)
    filen.close()
    return obj

def noise_brown(ncols, nrows=1, weight=1, filter=None, filterargs=None):
    '''Return 1/f^2 noise of shape(nrows, ncols obtained by taking 
    the cumulative sum of gaussian white noise, with rms weight.
    
    If filter != None, this function will apply the filter coefficients obtained
    by:
    ::
        
        >>> b, a = filter(**filterargs)
        >>> signal = scipy.signal.lfilter(b, a, signal)
    '''
    if filter != None:
        coeff_b, coeff_a = filter(**filterargs)
    
    noise = pl.empty((nrows, ncols))    
    for i in xrange(nrows):
        signal = pl.normal(size=ncols+10000).cumsum()
        if filter != None:
            signal = ss.lfilter(coeff_b, coeff_a, signal)
        noise[i, :] = signal[10000:]
        noise[i, :] /= pl.rms_flat(noise[i, :])
        noise[i, :] *= weight
    return noise
    