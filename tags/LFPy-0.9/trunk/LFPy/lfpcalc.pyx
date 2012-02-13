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
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cpdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] calc_lfp_choose(c,
                    double x=0, double y=0, double z=0, double sigma=0.3,
                    r_limit=None,
                    timestep=None, t_indices=None, method='linesource'):
    '''Determine which method to use, line-source for soma default'''
    if method == 'som_as_point':
        return calc_lfp_som_as_point(c, x=x, y=y, z=z, sigma=sigma,
                                     r_limit=r_limit,
                                     timestep=timestep, t_indices=t_indices)
    elif method == 'linesource':
        return calc_lfp_linesource(c, x=x, y=y, z=z, sigma=sigma,
                                   r_limit=r_limit,
                                   timestep=timestep, t_indices=t_indices)
    elif method == 'pointsource':
        return calc_lfp_pointsource(c, x=x, y=y, z=z, sigma=sigma,
                                    r_limit=r_limit,
                                    timestep=timestep, t_indices=t_indices)


cpdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] calc_lfp_linesource(c,
                        double x=0,
                        double y=0,
                        double z=0,
                        double sigma=0.3,
                        r_limit=None,
                        timestep=None, t_indices=None):
    
    """Calculate electric field potential using the line-source method, all
    compartments treated as line sources, even soma."""
    # Handling the r_limits. If a r_limit is a single value, an array r_limit
    # of shape c.diam is returned.
    if type(r_limit) == int or type(r_limit) == float:
        r_limit = np.ones(np.shape(c.diam))*abs(r_limit)
    elif np.shape(r_limit) != np.shape(c.diam):
        raise Exception, 'r_limit is neither a float- or int- value, nor is \
            r_limit.shape() equal to c.diam.shape()'

    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart, xend, \
        ystart, yend, zstart, zend, deltaS, h, r2, l, \
        Ememi, Ememii, Ememiii, Emem, r_lims
    cdef np.ndarray[long, ndim=1, negative_indices=False] i, ii, iii

    if timestep != None:
        currmem = c.imem[:, timestep]
    if t_indices != None:
        currmem = c.imem[:, t_indices]
    else:
        currmem = c.imem


    #some variables for h, r2, r_soma calculations
    xstart = c.xstart
    xend = c.xend
    ystart = c.ystart
    yend = c.yend
    zstart = c.zstart
    zend = c.zend


    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)


    r_lims = r_limit
    r2 = _check_rlimit(r2, r_limit, h, deltaS)

    l = h + deltaS


    hnegi = h < 0
    hposi = h >= 0
    lnegi = l < 0
    lposi = l >= 0

    #case i, h < 0, l < 0
    [i] = np.where(hnegi & lnegi)
    #case ii, h < 0, l > 0
    [ii] = np.where(hnegi & lposi)
    #case iii, h > 0, l > 0
    [iii] = np.where(hposi & lposi)
    
    ##case i, h < 0, l < 0
    #[i] = np.where((h < 0) & (l < 1))
    ##case ii, h < 0, l >= 0
    #[ii] = np.where((h < 0) & (l >= 0))
    ##case iii, h >= 0, l >= 0
    #[iii] = np.where((h >= 0) & (l >= 1))
    #

    Ememi = _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h)
    Ememii = _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h)
    Ememiii = _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h)

    Emem = Ememi + Ememii + Ememiii

    return Emem.transpose()


cpdef np.ndarray[DTYPE_t, ndim=1] calc_lfp_som_as_point(c,
                          double x=0, double y=0, double z=0, double sigma=0.3,
                          r_limit=None,
                          timestep=None, t_indices=None):
    '''Calculate electric field potential using the line-source method,
    soma is treated as point/sphere source'''
    #Handling the r_limits. If a r_limit is a single value,
    #an array r_limit of shape c.diam is returned.
    if type(r_limit) != type(np.array([])):
        r_limit = np.array(r_limit)
    if r_limit.shape == ():
        s_limit = r_limit
        r_limit = np.ones(c.diam.size) * abs(r_limit)
    elif r_limit.shape == (2, ):
        s_limit = abs(r_limit[0])
        r_limit = np.ones(c.diam.size) * abs(r_limit[1])
    elif r_limit.shape == c.diam.shape:
        s_limit = r_limit[0]
        r_limit = r_limit
    else:
        raise Exception,  'r_limit is neither a float- or int- value, \
            on the form r_limit=[s_limit, r_limit],  \
            nor is shape(r_limit) equal to shape(c.diam)!'

    cdef np.ndarray[DTYPE_t, ndim=2] currmem
    cdef np.ndarray[DTYPE_t, ndim=1] xstart, xend, \
        ystart, yend, zstart, zend, deltaS, h, r2, l, \
        Ememi, Ememii, Ememiii, Emem
    cdef np.ndarray[long, ndim=1] i, ii, iii
    cdef double xmid, ymid, zmid


    if timestep != None:
        currmem = c.imem[:, timestep]
    if t_indices != None:
        currmem = c.imem[:, t_indices]
    else:
        currmem = c.imem

    #some variables for h, r2, r_soma calculations
    xstart = c.xstart
    xmid = c.xmid[0]
    xend = c.xend
    ystart = c.ystart
    ymid = c.ymid[0]
    yend = c.yend
    zstart = c.zstart
    zmid = c.zmid[0]
    zend = c.zend


    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)
    r_soma = _r_soma_calc(xmid, ymid, zmid, x, y, z)
    if r_soma < s_limit:
        print 'Adjusting r-distance to soma segment from %g to %g' \
                % (r_soma, s_limit)
        r_soma = s_limit

    # Check that no segment is close the electrode than r_limit
    if np.sum(np.nonzero( r2 < r_limit * r_limit )) > 0:
        for idx in np.nonzero( r2[1:] < r_limit[1:] * r_limit[1:] )[0]+1:
            if (h[idx] < r_limit[idx]) and \
            ((deltaS[idx] + h[idx]) > -r_limit[idx]):
                print 'Adjusting distance to segment ', str(idx), ' from ',\
                str(np.sqrt(r2[idx])), ' to ', str(r_limit[idx]), '.'
                r2[idx] = r_limit[idx] * r_limit[idx]

    l = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l < 0
    lposi = l >= 0

    # Ensuring that soma is not treated as line-source
    hnegi[0] = hposi[0] = lnegi[0] = lposi[0] = False

    #Line sources
    #case i,  h < 0,  l < 0
    [i] = np.where(hnegi & lnegi)
    #case ii,  h < 0,  l > 0
    [ii] = np.where(hnegi & lposi)
    #case iii,  h > 0,  l > 0
    [iii] = np.where(hposi & lposi)

    Ememi = _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h)
    Ememii = _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h)
    Ememiii = _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h)

    #Potential contribution from soma
    Emem0 = currmem[0] / (4 * np.pi * sigma * r_soma)

    #Summarizing all potential contributions
    Emem = Emem0 + Ememi + Ememiii + Ememii
    
    return Emem.transpose()


cdef double _r_soma_calc(double xmid, double ymid, double zmid,
                        double x, double y, double z):
    '''Calculate the distance to the soma midpoint'''
    r_soma = np.sqrt((x - xmid)*(x - xmid) + (y - ymid)*(y - ymid) + \
        (z - zmid)*(z - zmid))

    return r_soma


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _check_rlimit(
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r_limit,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS):
    # Check that no segment is close the electrode than r_limit
    cdef int idx
    if np.sum(np.nonzero( r2 < r_limit*r_limit )) > 0:
        for idx in np.nonzero( r2 < r_limit*r_limit )[0]:
            if (h[idx] < r_limit[idx]) and ((deltaS[idx]+h[idx])>-r_limit[idx]):
                print 'Adjusting distance to segment ',str(idx),' from ', \
                     str(np.sqrt(r2[idx])),' to ',str(r_limit[idx]),'.'
                r2[idx] = r_limit[idx]**2
    return r2


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _Ememi_calc(
               np.ndarray[long, ndim=1, negative_indices=False] i,
               np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem,
               double sigma,
               np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS,
               np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l,
               np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2,
               np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h):
    '''Subroutine used by calc_lfp_som_as_point()'''
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem_iT = \
        currmem[i].T
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS_i = \
        deltaS[i]

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l_i = l[i]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2_i = r2[i]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h_i = h[i]

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] aa = 4 * \
                                                    np.pi * sigma * deltaS_i
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] bb = np.sqrt(h_i *
                                                    h_i + r2_i) - h_i
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] cc = np.sqrt(l_i *
                                                    l_i + r2_i) - l_i
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] dd = \
                                                    np.log(bb / cc) / aa

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] Emem_i = \
                                                        np.dot(currmem_iT, dd)

    return Emem_i


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _Ememii_calc(
                np.ndarray[long, ndim=1, negative_indices=False] ii,
                np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem,
                double sigma,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h):
    '''Subroutine used by calc_lfp_som_as_point()'''
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem_iiT = \
        currmem[ii].transpose()
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS_ii = \
        deltaS[ii]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l_ii = l[ii]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2_ii = r2[ii]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h_ii = h[ii]

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] aa = 4 * \
                                                np.pi * sigma * deltaS_ii
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] bb = np.sqrt(
                                                h_ii*h_ii + r2_ii) - h_ii
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] cc = (l_ii +
                                        np.sqrt(l_ii*l_ii + r2_ii)) / r2_ii
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] dd = \
                                        np.log(bb * cc) / aa

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] Emem_ii = \
                                                        np.dot(currmem_iiT, dd)
    return Emem_ii


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _Ememiii_calc(
                 np.ndarray[long, ndim=1, negative_indices=False] iii,
                 np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem,
                 double sigma,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h):
    '''Subroutine used by calc_lfp_som_as_point()'''
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] currmem_iiiT = \
        currmem[iii].T
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l_iii = l[iii]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2_iii = r2[iii]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h_iii = h[iii]
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS_iii = \
        deltaS[iii]

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] aa = 4 * \
                                                np.pi * sigma * deltaS_iii
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] bb = np.sqrt(
                                                l_iii*l_iii + r2_iii) + l_iii
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] cc = np.sqrt(
                                                h_iii*h_iii + r2_iii) + h_iii
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] dd = \
                                                np.log(bb / cc) / aa

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] Emem_iii = \
    							np.dot(currmem_iiiT, dd)
    return Emem_iii


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _deltaS_calc(
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xend,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ystart,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zstart,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend):
    '''Subroutine used by calc_lfp_som_as_point()'''
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS = \
    np.sqrt( (xstart - xend)*(xstart - xend) \
        + (ystart - yend)*(ystart - yend) + (zstart-zend)*(zstart-zend))

    return deltaS


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _h_calc(
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xend,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ystart,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zstart,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS,
                double x, double y, double z):
    '''Subroutine used by calc_lfp_som_as_point()'''
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] aa, bb
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] cc, hh

    aa = np.array([x - xend, y - yend, z-zend])
    bb = np.array([xend - xstart, yend - ystart, zend - zstart])
    cc = np.dot(aa.T, bb).diagonal() 
    hh = cc / deltaS
    hh[0] = 0
    return hh

cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _r2_calc(
            np.ndarray[DTYPE_t, ndim=1] xend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend,
            double x, double y, double z,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h):
    '''Subroutine used by calc_lfp_*()'''
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2 = \
        (x-xend)*(x-xend) + (y-yend)*(y-yend) + (z-zend)*(z-zend) - h*h

    return abs(r2)

cpdef calc_lfp_pointsource(c, x=0, y=0, z=0, sigma=0.3,
                        r_limit=None, 
                        timestep=None, t_indices=None):
    '''Calculate local field potentials using the point-source equation on all
    compartments'''
    # Handling the r_limits. If a r_limit is a single value, an array r_limit
    # of shape c.diam is returned.
    if type(r_limit) == int or type(r_limit) == float:
        r_limit = np.ones(np.shape(c.diam))*abs(r_limit)
    elif np.shape(r_limit) != np.shape(c.diam):
        raise Exception, 'r_limit is neither a float- or int- value, nor is \
            r_limit.shape() equal to c.diam.shape()'

    if timestep != None:
        currmem = c.imem[:, timestep]
    if t_indices != None:
        currmem = c.imem[:, t_indices]
    else:
        currmem = c.imem

    r2 = (c.xmid - x)**2 + (c.ymid - y)**2 + (c.zmid - z)**2
    r2 = _check_rlimit_point(r2, r_limit)
    r = np.sqrt(r2)

    Emem = 1 / (4 * np.pi * sigma) * np.dot(currmem.T, 1/r)

    return Emem.transpose()

cdef _check_rlimit_point(r2, r_limit):
    '''Correct r2 so that r2 >= r_limit for all values'''
    [inds] = np.where(r2 < r_limit)
    r2[inds] = r_limit[inds]

    return r2
