#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: language_level=2

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t   LTYPE_t

cdef extern from "math.h":
    int floor(DTYPE_t x)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1, negative_indices=False] alias_method(
    np.ndarray[long, ndim=1, negative_indices=False] idx,
    np.ndarray[DTYPE_t, ndim=1, negative_indices=False] probs,
    int nsyn):
    """
    Alias method for drawing random numbers from a discrete probability
    distribution. See http://www.keithschwarz.com/darts-dice-coins/

    Parameters
    ----------
    idx: np.ndarray
        segment indices as array of ints
    probs: np.ndarray
        segment areas as array of floats
    nsyn: int
        number of randomized segment indices

    Returns
    -------
    out: np.ndarray
        integer array of randomly drawn segment indices

    """
    assert idx.size == probs.size, 'length of idx and probs arrays must be equal'

    # C-declare variables
    cdef np.ndarray[long, ndim=1, negative_indices=False] J, spc
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] q
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] rands
    cdef int nn, j, ad, K, kk

    # Construct the table.
    J, q = alias_setup(probs)

    # output array
    spc = np.zeros(nsyn, dtype=int)

    # prefetch random numbers, alias_draw needs nsyn x 2 numbers
    rands = np.random.rand(nsyn, 2)

    K = J.size
    # Generate variates using alias draw method
    for nn in range(nsyn):
        kk = floor(rands[nn, 0]*K)
        if rands[nn, 1] < q[kk]:
            spc[nn] = idx[kk]
        else:
            spc[nn] = idx[J[kk]]

    return spc


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef alias_setup(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] probs):
    """Set up function for alias method.
    See http://www.keithschwarz.com/darts-dice-coins/

    Parameters
    ----------
    probs: np.ndarray
        float array of segment areas

    Returns
    -------
    J: np.ndarray
        array of ints
    q: np.ndarray
        array of floats

    """
    # C-declare variables
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] q
    cdef np.ndarray[long, ndim=1, negative_indices=False] J, smaller, larger
    cdef long K
    cdef int small, large, kk, s_i, l_i
    cdef DTYPE_t prob

    K = probs.size
    q = probs*K
    J = np.zeros(K, dtype=int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = np.zeros(K, dtype=int)
    larger = np.zeros(K, dtype=int)
    s_i = 0
    l_i = 0
    for kk in range(K):
        if q[kk] < 1:
            smaller[s_i] = kk
            s_i += 1
        else:
            larger[l_i] = kk
            l_i += 1

    s_i -= 1
    l_i -= 1

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while s_i >= 0 and l_i >= 0:
        small = smaller[s_i]
        large = larger[l_i]

        J[small] = large
        q[large] = q[large] + q[small] - 1

        s_i -= 1

        if q[large] < 1:
            s_i += 1
            l_i -= 1
            smaller[s_i] = large

    return J, q
