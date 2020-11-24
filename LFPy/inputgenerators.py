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


import numpy as np
import scipy.stats


def get_activation_times_from_distribution(n, tstart=0., tstop=1.E6,
                                           distribution=scipy.stats.expon,
                                           rvs_args=dict(loc=0, scale=1),
                                           maxiter=1E6):
    """
    Construct a length n list of ndarrays containing continously increasing
    random numbers on the interval [tstart, tstop], with intervals drawn from
    a chosen continuous random variable distribution subclassed from
    scipy.stats.rv_continous, e.g., scipy.stats.expon or scipy.stats.gamma.

    The most likely initial first entry is
    ``tstart + method.rvs(size=inf, **rvs_args).mean()``

    Parameters
    ----------
    n: int
        number of ndarrays in list
    tstart: float
        minimum allowed value in ndarrays
    tstop: float
        maximum allowed value in ndarrays
    distribution: object
        subclass of scipy.stats.rv_continous. Distributions
        producing negative values should be avoided if continously increasing
        values should be obtained, i.e., the probability density function
        ``(distribution.pdf(**rvs_args))`` should be ``0`` for ``x < 0``,
        which is not explicitly tested for.
    rvs_args: dict
        parameters for method.rvs method. If "size" is in dict, then tstop will
        be ignored, and each ndarray in output list will be
        ``distribution.rvs(**rvs_args).cumsum() + tstart``. If size is not
        given in dict, then values up to tstop will be included
    maxiter: int
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

    >>> from LFPy.inputgenerators import get_activation_times_from_distribution
    >>> import scipy.stats as st
    >>> import matplotlib.pyplot as plt
    >>> times = get_activation_times_from_distribution(n=10, tstart=0.,
    >>>                                                tstop=1000.,
    >>>                                                distribution=st.expon,
    >>>                                                rvs_args=dict(loc=0.,
    >>>                                                scale=100.))
    """
    assert hasattr(distribution, 'rvs'), \
        'distribution={} must have the attribute "rvs"'.format(distribution)

    times = []
    if 'size' in rvs_args.keys():
        for i in range(n):
            times += [distribution.rvs(**rvs_args).cumsum() + tstart]
    else:
        for i in range(n):
            values = distribution.rvs(size=1000, **rvs_args).cumsum() + tstart
            iter = 0
            while values[-1] < tstop and iter < maxiter:
                values = np.r_[values, distribution.rvs(
                    size=1000, **rvs_args).cumsum() + values[-1]]
                iter += 1

            if iter == maxiter:
                raise StopIteration('maximum number of iterations reach. Con')

            times += [values[values < tstop]]

    return times
