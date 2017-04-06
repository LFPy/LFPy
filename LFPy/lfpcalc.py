#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


def calc_lfp_linesource(cell, x=0., y=0., z=0., sigma=0.3,
                        r_limit=None):
    """Calculate electric field potential using the line-source method, all
    compartments treated as line sources, including soma.
    
    Parameters
    ----------        
    cell: obj
        LFPy.Cell or LFPy.TemplateCell like instance
    x : float
        extracellular position, x-axis
    y : float
        extracellular position, y-axis
    z : float
        extracellular position, z-axis
    sigma : float
        extracellular conductivity
    r_limit : [None]/float/np.ndarray
        minimum distance to source current. Can be scalar or numpy array with
        a limit for each cell compartment. Defaults to [None]
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """
    # Handling the r_limits. If a r_limit is a single value, an array r_limit
    # of shape cell.diam is returned.
    if type(r_limit) == int or type(r_limit) == float:
        r_limit = np.ones(np.shape(cell.diam))*abs(r_limit)
    elif np.shape(r_limit) != np.shape(cell.diam):
        raise Exception('r_limit is neither a float- or int- value, nor is \
            r_limit.shape() equal to cell.diam.shape()')
    
    # if t_indices is not None:
    #     currmem = cell.imem[:, t_indices]
    # else:
    #     currmem = cell.imem

    #some variables for h, r2, r_soma calculations
    xstart = cell.xstart
    xend = cell.xend
    ystart = cell.ystart
    yend = cell.yend
    zstart = cell.zstart
    zend = cell.zend
    
    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)

    r2 = _check_rlimit(r2, r_limit, h, deltaS)
    
    l = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l < 0
    lposi = l >= 0

    mapping = np.zeros(cell.totnsegs)

    #case i, h < 0, l < 0
    [i] = np.where(hnegi & lnegi)
    #case ii, h < 0, l >= 0
    [ii] = np.where(hnegi & lposi)
    #case iii, h >= 0, l >= 0
    [iii] = np.where(hposi & lposi)

    mapping[i] = _linesource_calc_case1(l[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l[iii], r2[iii], h[iii])

    # Emem = np.dot(currmem.T, 1 / (4 * np.pi * sigma * deltaS) * mapping)
    
    return 1 / (4 * np.pi * sigma * deltaS) * mapping

def calc_lfp_soma_as_point(cell, x=0., y=0., z=0., sigma=0.3,
                           r_limit=None):
    """Calculate electric field potential using the line-source method,
    soma is treated as point/sphere source
    
    Parameters
    ----------
    cell: obj
        `LFPy.Cell` or `LFPy.TemplateCell` like instance
    x : float
        extracellular position, x-axis
    y : float
        extracellular position, y-axis
    z : float
        extracellular position, z-axis
    sigma : float
        extracellular conductivity in S/m
    r_limit : float or np.ndarray or None
        [None]/float/np.ndarray: minimum distance to source current.
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """
    #Handling the r_limits. If a r_limit is a single value,
    #an array r_limit of shape cell.diam is returned.
    if type(r_limit) != type(np.array([])):
        r_limit = np.array(r_limit)
    if r_limit.shape == ():
        s_limit = r_limit
        r_limit = np.ones(cell.diam.size) * abs(r_limit)
    elif r_limit.shape == (2, ):
        s_limit = abs(r_limit[0])
        r_limit = np.ones(cell.diam.size) * abs(r_limit[1])
    elif r_limit.shape == cell.diam.shape:
        s_limit = r_limit[0]
        r_limit = r_limit
    else:
        raise Exception('r_limit is neither a float- or int- value, \
            on the form r_limit=[s_limit, r_limit],  \
            nor is shape(r_limit) equal to shape(cell.diam)!')

    # if t_indices is not None:
    #     currmem = cell.imem[:, t_indices]
    # else:
    #     currmem = cell.imem

    #some variables for h, r2, r_soma calculations
    xstart = cell.xstart
    xmid = cell.xmid[0]
    xend = cell.xend
    ystart = cell.ystart
    ymid = cell.ymid[0]
    yend = cell.yend
    zstart = cell.zstart
    zmid = cell.zmid[0]
    zend = cell.zend

    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)
    r_soma = _r_soma_calc(xmid, ymid, zmid, x, y, z)
    if r_soma < s_limit:
        print('Adjusting r-distance to soma segment from %g to %g'
                % (r_soma, s_limit))
        r_soma = s_limit

    # Check that no segment is closer to the electrode than r_limit
    if np.sum(np.nonzero( r2 < r_limit*r_limit )) > 0:
        for idx in np.nonzero( r2[1:] < r_limit[1:] * r_limit[1:] )[0]+1:
            if (h[idx] < r_limit[idx]) and \
            ((deltaS[idx] + h[idx]) > -r_limit[idx]):
                print('Adjusting distance to segment %s from %.2f to %.2f.'
                      % (idx, r2[idx]**0.5, r_limit[idx]))
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
    i = np.where(hnegi & lnegi)
    #case ii,  h < 0,  l >= 0
    ii = np.where(hnegi & lposi)
    #case iii,  h >= 0,  l >= 0
    iii = np.where(hposi & lposi)

    #Summarizing all potential contributions

    mapping = np.zeros(cell.totnsegs)
    mapping[0] = 1 / r_soma
    deltaS[0] = 1.

    mapping[i] = _linesource_calc_case1(l[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l[iii], r2[iii], h[iii])

    # Emem = np.dot(currmem.T, 1 / (4 * np.pi * sigma * deltaS) * mapping)

    return 1 / (4 * np.pi * sigma * deltaS) * mapping

def _linesource_calc_case1(l_i, r2_i, h_i):
    """Calculates linesource contribution for case i"""
    bb = np.sqrt(h_i**2 + r2_i) - h_i
    cc = np.sqrt(l_i**2 + r2_i) - l_i
    dd = np.log(bb / cc)
    return dd

def _linesource_calc_case2(l_ii, r2_ii, h_ii):
    """Calculates linesource contribution for case ii"""
    bb = np.sqrt(h_ii**2 + r2_ii) - h_ii
    cc = (l_ii + np.sqrt(l_ii**2 + r2_ii)) / r2_ii
    dd = np.log(bb * cc)
    return dd
    
def _linesource_calc_case3(l_iii, r2_iii, h_iii):
    """Calculates linesource contribution for case iii"""
    bb = np.sqrt(l_iii**2 + r2_iii) + l_iii
    cc = np.sqrt(h_iii**2 + r2_iii) + h_iii
    dd = np.log(bb / cc)
    return dd

def _deltaS_calc(xstart, xend, ystart, yend, zstart, zend):
    """Returns length of each segment"""
    deltaS = np.sqrt((xstart - xend)**2 + (ystart - yend)**2 +
                     (zstart-zend)**2)
    return deltaS

def _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z):
    """Subroutine used by calc_lfp_*()"""
    aa = np.array([x - xend, y - yend, z-zend])
    bb = np.array([xend - xstart, yend - ystart, zend - zstart])
    cc = np.sum(aa*bb, axis=0)
    hh = cc / deltaS
    return hh

def _r2_calc(xend, yend, zend, x, y, z, h):
    """Subroutine used by calc_lfp_*()"""
    r2 = (x-xend)**2 + (y-yend)**2 + (z-zend)**2 - h**2
    return abs(r2)

def _check_rlimit(r2, r_limit, h, deltaS):
    """Check that no segment is close the electrode than r_limit"""
    if np.sum(np.nonzero(r2 < r_limit*r_limit)) > 0:
        for idx in np.nonzero( r2 < r_limit*r_limit )[0]:
            if (h[idx] < r_limit[idx]) and ((deltaS[idx]+h[idx]) >
                                             -r_limit[idx]):
                print('Adjusting distance to segment %s from %.2f to %.2f.'
                      % (idx, r2[idx]**0.5, r_limit[idx]))
                r2[idx] = r_limit[idx]**2
    return r2

def _r_soma_calc(xmid, ymid, zmid, x, y, z):
    """calculate the distance to soma midpoint"""
    r_soma = np.sqrt((x - xmid)**2 + (y - ymid)**2 + (z - zmid)**2)
    return r_soma

def calc_lfp_pointsource(cell, x=0, y=0, z=0, sigma=0.3,
                        r_limit=None):
    """Calculate extracellular potentials using the point-source
    equation on all compartments

    Parameters
    ----------
    cell: obj
        LFPy.Cell or LFPy.TemplateCell like instance
    x : float
        extracellular position, x-axis
    y : float
        extracellular position, y-axis
    z : float
        extracellular position, z-axis
    sigma : float
        extracellular conductivity
    r_limit : [None]/float/np.ndarray
        minimum distance to source current
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """
    # Handling the r_limits. If a r_limit is a single value, an array r_limit
    # of shape cell.diam is returned.
    if type(r_limit) == int or type(r_limit) == float:
        r_limit = np.ones(np.shape(cell.diam))*abs(r_limit)
    elif np.shape(r_limit) != np.shape(cell.diam):
        raise Exception('r_limit is neither a float- or int- value, nor is \
            r_limit.shape() equal to cell.diam.shape()')

    r2 = (cell.xmid - x)**2 + (cell.ymid - y)**2 + (cell.zmid - z)**2
    r2 = _check_rlimit_point(r2, r_limit)
    r = np.sqrt(r2)
    
    mapping = 1 / (4 * np.pi * sigma * r)
    
    return mapping

def _check_rlimit_point(r2, r_limit):
    """Correct r2 so that r2 >= r_limit**2 for all values"""
    inds = r2 < r_limit*r_limit
    r2[inds] = r_limit[inds]*r_limit[inds]
    
    return r2
