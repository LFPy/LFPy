#!/usr/bin/env python
"""Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

from __future__ import division
import numpy as np
import scipy.stats
import warnings

def get_rand_spike_times(nsyn, nspikes, tstart, tstop):
    """Return synpos times nspikes random spike times on the 
    interval [tstart, tstop]
    
    Parameters
    ----------
    nsyn : int
    nspikes : int
    tstart : float
    tstop : float
    
    Returns
    -------
    ndarray
        shape (nsyn, nspikes) array with random spike times
    """
    warnings.warn("LFPy.inputgenerators.get_rand_spike_times will be removed shortly", DeprecationWarning)
    spiketimes = np.random.rand(nsyn*nspikes).reshape((nsyn, nspikes))
    spiketimes *= (tstop - tstart)
    spiketimes += tstart
    return spiketimes

def stationary_poisson(nsyn, rate, tstart, tstop):
    """Generate nsyn stationary possion processes with rate expectation rate 
    between tstart and tstop
    
    Parameters
    ----------
    nsyn : int
    rate : float
    tstart : float
    tstop : float
    
    Returns
    -------
    list of ndarrays
        list where each element is an ndarray with times as from a Poisson process
    """
    warnings.warn("LFPy.inputgenerators.stationary_poisson will be removed shortly", DeprecationWarning)
    interval_s = (tstop-tstart)*.001
    spiketimes = []
    for i in range(nsyn):
        spikecount = np.random.poisson(interval_s*rate)
        spikevec = np.zeros(spikecount)
        if spikecount == 0:
            spiketimes.append(spikevec)
        else:
            spikevec = tstart + (tstop-tstart)*np.random.random(spikecount)
            spiketimes.append(np.sort(spikevec)) #sort them too!

    return spiketimes

def stationary_gamma(tstart, tstop, k=2, theta=10, tmin = -1E3, tmax=1E6):
    """Generate spiketimes with interspike interval statistics according
    to gamma-distribution with 'shape' k and 'scale' theta between tstart and
    tstop. Spiketimes from tmin up to tmax is calculated,
    times between 0 and tstop are returned
    
    Parameters
    ----------
    tstart : float
    tstop : float
    k : float
    theta : float
    tmin : float
    tmax : float
    
    Returns
    -------
    ndarray

    """
    warnings.warn("LFPy.inputgenerators.stationary_gamma will be removed shortly", DeprecationWarning)
    
    if tstop > tmax:
        tmax = tstop
    
    t = tmin
    spiketimes = []
    while t <= tmax:
        t = t + np.random.gamma(shape = k, scale = theta)
        if t >= tstart and t <= tstop:
            spiketimes.append(t)
    
    spiketimes = np.array(spiketimes)
    return spiketimes


def get_normal_spike_times(nsyn, mu, sigma, tstart, tstop):
    """Generate nsyn normal-distributed processes with mean mu and 
    deviation sigma"""
    warnings.warn("LFPy.inputgenerators.get_normal_spike_times will be removed shortly", DeprecationWarning)
    spiketimes = []
    spikecount = nsyn
    spikevec = np.zeros(spikecount)
    spikevec = np.random.normal(mu, sigma)
    while (np.squeeze(spikevec) <= tstart) and \
            (np.squeeze(spikevec) >= tstop):
        spikevec = np.random.normal(mu, sigma)
    
    spiketimes.append(np.sort(spikevec))
    return spiketimes

def get_normal_input_times(n, mu, sigma, tstart, tstop):
    """Generates n normal-distributed prosesses with mean mu and 
    deviation sigma"""
    warnings.warn("LFPy.inputgenerators.get_normal_input_times will be removed shortly", DeprecationWarning)
    times = np.random.normal(mu, sigma, n)
    for i in range(n):
        while times[i] <= tstart or times[i] >= tstop:
            times[i] = np.random.normal(mu, sigma)
    return times



def get_activation_times_from_scipy_stats(n, tstart=0., tstop=1.E6,
                                          distribution=scipy.stats.expon,
                                          rvs_args=dict(loc=0, scale=1),
                                          maxiter=1E6):
    """
    Construct a length n list of ndarrays containing continously increasing
    random numbers on the interval [tstart, tstop], with intervals drawn from
    a chosen continuous random variable distribution subclassed from
    scipy.stats.rv_continous, e.g., scipy.stats.expon or scipy.stats.gamma.
    
    The most likely initial first entry is
    tstart + method.rvs(size=inf, **rvs_args).mean()
    
    
    Parameters
    ----------
    n : int
        number of ndarrays in list
    tstart : float
        minimum allowed value in ndarrays
    tstop : float
        maximum allowed value in ndarrays
    distribution : object
        subclass of scipy.stats.rv_continous. Distributions
        producing negative values should be avoided if continously increasing
        values should be obtained, i.e., the probability density function
        (distribution.pdf(**rvs_args)) should be 0 for x < 0, but this is not
        explicitly tested for.
    rvs_args : dict
        parameters for method.rvs method. If "size" is in dict, then tstop will
        be ignored, and each ndarray in output list will be
        distribution.rvs(**rvs_args).cumsum() + tstart. If size is not given in dict,
        then values up to tstop will be included
    maxiter : int
        maximum number of iterations
        
        
    Returns
    -------
    list of ndarrays
        length n list of arrays containing data
    
    Raises
    ------
    AssertionError
        if distribution does not have the 'rvs' attribute
    StopIteration
        if number of while-loop iterations reaches maxiter
    
    
    Examples
    --------
    Create n sets of activation times with intervals drawn from the exponential
    distribution, with rate expectation lambda 10 s^-1 (thus
    scale=1000 / lambda). Here we assume output in units of ms
    
    >>> from LFPy.inputgenerators import get_random_numbers_from_scipy_stats
    >>> import scipy.stats as st
    >>> import matplotlib.pyplot as plt
    >>> times = get_random_numbers_from_scipy_stats(n=10, tstart=0., tstop=1000.,
    >>>                                             distribution=st.expon,
    >>>                                             rvs_args=dict(loc=0.,
    >>>                                                           scale=100.))
    """
    try:
        assert hasattr(distribution, 'rvs')
    except AssertionError:
        raise AssertionError('distribution={} must have the attribute "rvs"'.format(distribution))
    
    times = []
    if 'size' in rvs_args.keys():
        for i in range(n):
            times += [distribution.rvs(**rvs_args).cumsum() + tstart]
    else:
        for i in range(n):
            values = distribution.rvs(size=1000, **rvs_args).cumsum() + tstart
            iter = 0
            while values[-1] < tstop and iter < maxiter:
                values = np.r_[values, distribution.rvs(size=1000, **rvs_args).cumsum() + values[-1]]
                iter += 1
            
            if iter == maxiter:
                raise StopIteration('maximum number of iterations reach. Con')
            
            
            times += [values[values < tstop]]
    
    return times
