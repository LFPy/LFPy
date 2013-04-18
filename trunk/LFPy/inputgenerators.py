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

import numpy as np

def get_rand_spike_times(synpos, nspikes, tstart, tstop):
    '''Return synpos times nspikes random spike times on the 
    interval [tstart, tstop]'''
    spiketimes = np.zeros([np.size(synpos), nspikes])
    for i in range(np.size(synpos)):
        spiketimes[i, :] = np.random.random_integers(tstart, tstop, nspikes)
    return spiketimes

def stationary_poisson(nsyn, lambd, tstart, tstop):
    '''Generate nsyn stationary possion processes with rate lambda 
    between tstart and tstop'''
    interval_s = (tstop-tstart)*.001
    spiketimes = []
    for i in range(nsyn):
        spikecount = np.random.poisson(interval_s*lambd)
        spikevec = np.empty(spikecount)
        if spikecount == 0:
            spiketimes.append(spikevec)
        else:
            spikevec = tstart + (tstop-tstart)*np.random.random(spikecount)
            spiketimes.append(np.sort(spikevec)) #sort them too!

    return spiketimes

def stationary_gamma(tstart, tstop, k=2, theta=10, tmin = -1E3, tmax=1E5):
    '''Generate spiketimes with interspike interval statistics according
    to gamma-distribution with 'shape' k and 'scale' theta between tstart and
    tstop. Spiketimes from tmin up to tmax is calculated,
    times between 0 and tstop are returned'''
    
    if tstop > tmax:
        tmax = tstop
    
    t = tmin
    spiketimes = []
    while t <= tmax:
        t = t + np.random.gamma(shape = k, scale = theta)
        if t >= tmin and t <= tstop:
            spiketimes.append(t)
    
    spiketimes = np.array(spiketimes)
    return spiketimes


def test_spiketimes(spiketime):
    '''Test and sort spike times'''
    spiketimes = []
    spikecount = 1
    spikevec = np.empty(spikecount)
    spikevec = spiketime
    spiketimes.append(np.sort(spikevec))
    return spiketimes

def get_normal_spike_times(nsyn, mu, sigma, tstart, tstop):
    '''Generate nsyn normal-distributed processes with mean mu and 
    deviation sigma'''
    spiketimes = []
    spikecount = nsyn
    spikevec = np.empty(spikecount)
    spikevec = np.random.normal(mu, sigma)
    while (np.squeeze(spikevec) <= tstart) and \
            (np.squeeze(spikevec) >= tstop):
        spikevec = np.random.normal(mu, sigma)
    
    spiketimes.append(np.sort(spikevec))
    return spiketimes

def get_normal_input_times(n, mu, sigma, tstart, tstop):
    '''Generates n normal-distributed prosesses with mean mu and 
    deviation sigma'''
    times = np.random.normal(mu, sigma, n)
    for i in range(n):
        while times[i] <= tstart or times[i] >= tstop:
            times[i] = np.random.normal(mu, sigma)
    return times
