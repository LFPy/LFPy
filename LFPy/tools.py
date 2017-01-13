#!/usr/bin/env python
'''Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.'''

import numpy as np
import scipy.signal as ss

def load(filename):
    '''Generic loading of cPickled objects from file'''
    import pickle
    
    filen = open(filename,'rb')
    obj = pickle.load(filen)
    filen.close()
    return obj

def noise_brown(ncols, nrows=1, weight=1, filter=None, filterargs=None):
    '''Return 1/f^2 noise of shape(nrows, ncols obtained by taking 
    the cumulative sum of gaussian white noise, with rms weight.
    
    If filter is not None, this function will apply the filter coefficients obtained
    by:
    ::
        
        >>> b, a = filter(**filterargs)
        >>> signal = scipy.signal.lfilter(b, a, signal)
    '''
    from matplotlib.mlab import rms_flat

    if filter is not None:
        coeff_b, coeff_a = list(filter(**filterargs))
    
    noise = np.empty((nrows, ncols))    
    for i in range(nrows):
        signal = np.random.normal(size=ncols+10000).cumsum()
        if filter is not None:
            signal = ss.lfilter(coeff_b, coeff_a, signal)
        noise[i, :] = signal[10000:]
        noise[i, :] /= rms_flat(noise[i, :])
        noise[i, :] *= weight
    return noise

