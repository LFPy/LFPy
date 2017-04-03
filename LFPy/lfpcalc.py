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

def calc_lfp_choose(cell, x=0., y=0., z=0., sigma=0.3,
                    r_limit=None,
                    t_indices=None, method='linesource'):
    """
    Determine which method to use. line-source for soma default

    TODO: x, y, z, sigma, r_limit, timestep, t_indices are given default values
    both in this function and in the calc_lfp_* funcions. Cleaner to
    only give default values at one level? From looking at functions here,
    one would think r_limit defaults to None, while it actually defaults to
    cell.diam/2 as set in recextelectrode.py

    Parameters
    ----------
    cell : obj
        LFPy.Cell or LFPy.TemplateCell instance
    x : float
        extracellular position, x-axis. Defaults to 0.
    y : float
        extracellular position, y-axis. Defaults to 0.
    z : float
        extracellular position, z-axis. Defaults to 0.
    sigma : float
        extracellular conductivity. Defaults to 0.3
    r_limit : [None]/float/np.ndarray:
        minimum distance to source current
    t_indices : np.ndarray or None. optional
        If array is given, LFP is calculated at specified timesteps
    method : str
        switch for choosing underlying methods. Can be one of the following:
        'linesource'/'pointsource'/'som_as_point'. Defaults to linesource

    """
    if method == 'som_as_point':
        return calc_lfp_som_as_point(cell, x=x, y=y, z=z, sigma=sigma,
                                     r_limit=r_limit,
                                     t_indices=t_indices)
    elif method == 'linesource':
        if type(sigma) in [list, np.ndarray]:
            return calc_lfp_linesource_anisotropic(cell, x=x, y=y, z=z, sigma=sigma,
                                       r_limit=r_limit,
                                        t_indices=t_indices)
        else:
            return calc_lfp_linesource(cell, x=x, y=y, z=z, sigma=sigma,
                                       r_limit=r_limit,
                                        t_indices=t_indices)
    elif method == 'pointsource':
        if type(sigma) in [list, np.ndarray]:
            # print("Given tensor conductivity. Using anisotropic formalism.")
            return calc_lfp_pointsource_anisotropic(cell, x=x, y=y, z=z, sigma=sigma,
                                    r_limit=r_limit,
                                    t_indices=t_indices)
        else:
            return calc_lfp_pointsource(cell, x=x, y=y, z=z, sigma=sigma,
                                        r_limit=r_limit,
                                        t_indices=t_indices)

def dist(x1, x2, p):
    """
    Returns distance and closest point on line between x1 and x2 from point p
    """
    px = x2[0]-x1[0]
    py = x2[1]-x1[1]
    pz = x2[2]-x1[2]

    delta = px*px + py*py + pz*pz

    u = ((p[0] - x1[0]) * px + (p[1] - x1[1]) * py + (p[2] - x1[2]) * pz) / float(delta)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x_ = x1[0] + u * px
    y_ = x1[1] + u * py
    z_ = x1[2] + u * pz

    closest_point = np.array([x_, y_, z_])
    dx = x_ - p[0]
    dy = y_ - p[1]
    dz = z_ - p[2]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

    return dist, closest_point



def calc_lfp_linesource_anisotropic(cell, x=0., y=0., z=0., sigma=0.3,
                        r_limit=None,
                        t_indices=None):
    """Calculate electric field potential using the line-source method, all
    compartments treated as line sources, even soma.

    Parameters
    ----------
    cell: obj
        LFPy.Cell or LFPy.TemplateCell instance
    x : float
        extracellular position, x-axis
    y : float
        extracellular position, y-axis
    z : float
        extracellular position, z-axis
    sigma : array
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

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem

    #some variables for h, r2, r_soma calculations
    xstart = cell.xstart.copy()
    xend = cell.xend.copy()
    ystart = cell.ystart.copy()
    yend = cell.yend.copy()
    zstart = cell.zstart.copy()
    zend = cell.zend.copy()

    dx2 = (xend - xstart)**2
    dy2 = (yend - ystart)**2
    dz2 = (zend - zstart)**2
    a = (sigma[1] * sigma[2] * dx2 +
         sigma[0] * sigma[2] * dy2 +
         sigma[0] * sigma[1] * dz2)
    pos = np.array([x, y, z])

    b = -2 * (sigma[1] * sigma[2] * (x - xstart) * (xend - xstart) +
              sigma[0] * sigma[2] * (y - ystart) * (yend - ystart) +
              sigma[0] * sigma[1] * (z - zstart) * (zend - zstart))
    c = (sigma[1] * sigma[2] * (x - xstart)**2 +
         sigma[0] * sigma[2] * (y - ystart)**2 +
         sigma[0] * sigma[1] * (z - zstart)**2)

    mapping = np.zeros(cell.totnsegs)

    for idx in range(cell.totnsegs):

        # pos = np.array([x, y, z])
        # print "start pos: ", pos
        a_ = a[idx]
        b_ = b[idx]
        c_ = c[idx]

        p1 = np.array([xstart[idx], ystart[idx], zstart[idx]])
        p2 = np.array([xend[idx], yend[idx], zend[idx]])
        l_vec = p2 - p1

        r, closest_point = dist(p1, p2, pos)
        p_ = pos.copy()

        if r < r_limit[idx]:

            # print "too close", closest_point, pos
            if np.abs(r) < 1e-6:
                # print "r is zero"

                if np.abs(l_vec[0]) < 1e-12:
                    p_[0] += r_limit[idx]
                    # print "here x"x
                elif np.abs(l_vec[1]) < 1e-12:
                    p_[1] = p_[1] + r_limit[idx]
                elif np.abs(l_vec[2]) < 1e-12:
                    p_[2] += r_limit[idx]
                else:
                    # print "Displacing like "
                    displace_vec = np.array([-l_vec[1], l_vec[0], 0])
                    displace_vec = displace_vec / np.sqrt(np.sum(displace_vec**2)) * r_limit[idx]
                    # print pos, p1, p2, displace_vec
                    p_[:] += displace_vec
            else:

                p_[:] = pos + (pos - closest_point) * (r_limit[idx] - r) / r

            if np.sqrt(np.sum((p_ - closest_point)**2)) - r_limit[idx] > 1e-9:
                print p_, closest_point
                raise RuntimeError("Segment adjustment not working")

            b_ = -2 * (sigma[1] * sigma[2] * (p_[0] - xstart[idx]) * (xend[idx] - xstart[idx]) +
                       sigma[0] * sigma[2] * (p_[1] - ystart[idx]) * (yend[idx] - ystart[idx]) +
                       sigma[0] * sigma[1] * (p_[2] - zstart[idx]) * (zend[idx] - zstart[idx]))
            c_ = (sigma[1] * sigma[2] * (p_[0] - xstart[idx])**2 +
                  sigma[0] * sigma[2] * (p_[1] - ystart[idx])**2 +
                  sigma[0] * sigma[1] * (p_[2] - zstart[idx])**2)

        if c_ == 0:
            raise RuntimeError("C should not be zero!")
        if np.abs(b_) <= 1e-12:
            # print("Case 0"), a_, b_, c_
            # one_over_r_part = 1. / np.sqrt(a_) * np.log(np.sqrt(a_ / c_) + np.sqrt(a_ / c_) * np.sqrt(1 + c_ / a_))
            one_over_r_part = 1. / np.sqrt(a_) * np.log(np.sqrt(a_ / c_) + np.sqrt(a_ / c_ + 1))
            # if np.isnan(one_over_r_part).any():
            #     print "NaN ", idx, pos
            #     print "Case 1", one_over_r_part, a_, b_, c_, 4 * a_ * c_ - b_**2
            #     raise RuntimeError("NaN")
            mapping[idx] = one_over_r_part

        elif np.abs(4 * a_ * c_ - b_**2) < 1e-6:

            # print "Case 3", a_, b_, c_,  -1. / np.sqrt(a_) * np.log(np.abs(2 * a_ / b_ + 1))
            # print pos, a_, b_, c_, xstart[idx], ystart[idx], zstart[idx], xend[idx], yend[idx], zend[idx]
            # one_over_r_part = 1. / np.sqrt(a_) * (np.log(np.abs((2 * a_ + b_)/b_)))
            # one_over_r_part = 1. / np.sqrt(a_) * np.log(np.sqrt(a_ / c_) + 1.)

            # if 1 - np.sqrt(a_ / c_) > 0:
            if np.abs(a_ - c_) < 1e-6:
                one_over_r_part = 1. / np.sqrt(a_) * np.log(np.abs(1 + np.sqrt(a_ / c_)))
            else:
                # one_over_r_part = -1. / np.sqrt(a_) * np.log(np.abs(1 - np.sqrt(a_ / c_)))
                # if b_ < 0:
                one_over_r_part = 1. / np.sqrt(a_) * np.abs(np.log(np.abs(np.sign(b_) * np.sqrt(a_/c_) + 1)))
            # else:
            #     one_over_r_part = 1. / np.sqrt(a_) * np.log(np.abs(np.sqrt(a_ / c_) + 1))

            # if np.isnan(one_over_r_part).any():
            #     print "NaN ", idx, pos
            #     print "Case 3", one_over_r_part, a_, b_, c_, 4 * a_ * c_ - b_**2
            #     raise RuntimeError("NaN")

            mapping[idx] = one_over_r_part

            # print one_over_r_part
        elif 4 * a_ * c_ < b_**2:
            # print("Case 2")
            numerator = np.abs(2 * a_ + b_ + 2 * np.sqrt(a_ * (a_ + b_ + c_)))
            denumerator = np.abs(b_ + 2 * np.sqrt(a_ * c_))
            one_over_r_part = 1. / np.sqrt(a_) * np.log(numerator / denumerator)
            # if np.isnan(one_over_r_part).any():
            #     print "NaN ", idx, pos
            #     print "Case 2", a_, b_, c_, numerator, denumerator
            #     raise RuntimeError("NaN")
            mapping[idx] = one_over_r_part


        elif 4 * a_ * c_ > b_**2:
            # print "arcsinh"
            one_over_r_part = 1. / np.sqrt(a_) * (np.arcsinh((2 * a_ + b_) / np.sqrt(4 * a_ * c_ - b_**2))-
                                                  np.arcsinh((b_) / np.sqrt(4 * a_ * c_ - b_**2)))
            # print("This happend")
            # if np.isnan(one_over_r_part).any():
            #     print "NaN ", idx, pos
            #     print "Case 4", one_over_r_part, a_, b_, c_, 4 * a_ * c_ - b_**2
            #     raise RuntimeError("NaN")

            mapping[idx] = one_over_r_part

            # one_over_r_part = 1. / np.sqrt(a[idx]) * (np.arcsinh((2 * a[idx] + b[idx]) / np.sqrt(4 * a[idx] * c[idx] - b[idx]**2)) - np.arcsinh(b[idx] / np.sqrt(4 * a[idx] * c[idx] - b[idx]**2)))
            # phi += currmem[idx] * one_over_r_part
        else:
            raise RuntimeError
    if np.isnan(mapping).any():
        raise RuntimeError("NaN")

    phi = 1 / (4 * np.pi) * np.dot(currmem.T, mapping)

    return phi.T

def calc_lfp_linesource(cell, x=0., y=0., z=0., sigma=0.3,
                        r_limit=None,
                        t_indices=None):
    """Calculate electric field potential using the line-source method, all
    compartments treated as line sources, even soma.
    
    Parameters
    ----------        
    cell: obj
        LFPy.Cell or LFPy.TemplateCell instance
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
    
    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem

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

    #case i, h < 0, l < 0
    [i] = np.where(hnegi & lnegi)
    #case ii, h < 0, l >= 0
    [ii] = np.where(hnegi & lposi)
    #case iii, h >= 0, l >= 0
    [iii] = np.where(hposi & lposi)

    # print r2
    Ememi = _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h)
    Ememii = _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h)
    Ememiii = _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h)

    Emem = Ememi + Ememii + Ememiii
    
    return Emem.transpose()

def calc_lfp_som_as_point(cell, x=0., y=0., z=0., sigma=0.3,
                          r_limit=None,
                          t_indices=None):
    """Calculate electric field potential using the line-source method,
    soma is treated as point/sphere source
    
    Parameters
    ----------
    cell: obj
        `LFPy.Cell` or `LFPy.TemplateCell` instance
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

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem

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
    
    Ememi = _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h)
    Ememii = _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h)
    Ememiii = _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h)

    #Potential contribution from soma
    Emem0 = currmem[0]/(4 * np.pi * sigma * r_soma)

    #Summarizing all potential contributions
    Emem = Emem0 + Ememi + Ememiii + Ememii

    return Emem.transpose()

def _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h):
    """Subroutine used by calc_lfp_*()"""
    currmem_iT = currmem[i].transpose()
    deltaS_i = deltaS[i]
    l_i = l[i]
    r2_i = r2[i]
    h_i = h[i]
    
    aa = 4 * np.pi * sigma * deltaS_i
    bb = np.sqrt(h_i**2 + r2_i) - h_i
    cc = np.sqrt(l_i**2 + r2_i) - l_i
    dd = np.log(bb / cc) / aa
    
    Emem_i = np.dot(currmem_iT, dd)
    
    return Emem_i

def _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h):
    """Subroutine used by calc_lfp_*()"""
    currmem_iiT = currmem[ii].transpose()
    deltaS_ii = deltaS[ii]
    l_ii = l[ii]
    r2_ii = r2[ii]
    h_ii = h[ii]

    aa = 4 * np.pi * sigma * deltaS_ii
    bb = np.sqrt(h_ii**2 + r2_ii) - h_ii
    cc = (l_ii + np.sqrt(l_ii**2 + r2_ii)) / r2_ii
    dd = np.log(bb * cc) / aa
    
    Emem_ii = np.dot(currmem_iiT, dd)
    
    return Emem_ii
    
def _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h):
    """Subroutine used by calc_lfp_*()"""
    currmem_iiiT = currmem[iii].transpose()
    l_iii = l[iii]
    r2_iii = r2[iii]
    h_iii = h[iii]
    deltaS_iii = deltaS[iii]

    aa = 4 * np.pi * sigma * deltaS_iii
    bb = np.sqrt(l_iii**2 + r2_iii) + l_iii
    cc = np.sqrt(h_iii**2 + r2_iii) + h_iii
    dd = np.log(bb / cc) / aa

    Emem_iii = np.dot(currmem_iiiT, dd)
    
    return Emem_iii

def _deltaS_calc(xstart, xend, ystart, yend, zstart, zend):
    """Subroutine used by calc_lfp_*()"""
    deltaS = np.sqrt((xstart - xend)**2 + (ystart - yend)**2 +
                     (zstart-zend)**2)
    return deltaS

def _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z):
    """Subroutine used by calc_lfp_*()"""
    aa = np.array([x - xend, y - yend, z-zend])
    bb = np.array([xend - xstart, yend - ystart, zend - zstart])
    cc = np.dot(aa.T, bb).diagonal()
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
            if (h[idx] < r_limit[idx]) and ((deltaS[idx]+h[idx]) > -r_limit[idx]):
                print('Adjusting distance to segment %s from %.2f to %.2f.'
                      % (idx, r2[idx]**0.5, r_limit[idx]))
                r2[idx] = r_limit[idx]**2
            # if r2[idx] == 0:
            #     print r2
            #     raise RuntimeError(r2)
    return r2

def _r_soma_calc(xmid, ymid, zmid, x, y, z):
    """calculate the distance to soma midpoint"""
    r_soma = np.sqrt((x - xmid)**2 + (y - ymid)**2 +
        (z - zmid)**2)

    return r_soma

def calc_lfp_pointsource(cell, x=0, y=0, z=0, sigma=0.3,
                        r_limit=None, 
                        t_indices=None):
    """Calculate extracellular potentials using the point-source
    equation on all compartments

    Parameters
    ----------
    cell: obj
        LFPy.Cell or LFPy.TemplateCell instance
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

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem
    
    r2 = (cell.xmid - x)**2 + (cell.ymid - y)**2 + (cell.zmid - z)**2
    r2 = _check_rlimit_point(r2, r_limit)
    r = np.sqrt(r2)
    
    Emem = 1 / (4 * np.pi * sigma) * np.dot(currmem.T, 1/r)
    
    return Emem.transpose()

def calc_lfp_pointsource_anisotropic(cell, x=0, y=0, z=0, sigma=[0.3, 0.3, 0.3],
                        r_limit=None,
                        t_indices=None):
    """Calculate extracellular potentials using the anisotropic point-source
    equation on all compartments

    Parameters
    ----------
    cell: obj
        LFPy.Cell or LFPy.TemplateCell instance
    x : float
        extracellular position, x-axis
    y : float
        extracellular position, y-axis
    z : float
        extracellular position, z-axis
    sigma : array
        extracellular conductivity in [x,y,z]-direction
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

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem

    dx2 = (cell.xmid - x)**2
    dy2 = (cell.ymid - y)**2
    dz2 = (cell.zmid - z)**2

    r2 = dx2 + dy2 + dz2
    # if (r2 == 0).any():
    #     dx2[r2 == 0] += 0.001
    #     r2[r2 == 0] += 0.001

    close_idxs = r2 < r_limit**2

    # For anisotropic media, the direction in which to move points matter.
    # Radial distance between point source and electrode is scaled to r_limit
    r2_scale_factor = r_limit[close_idxs]**2 / r2[close_idxs]
    dx2[close_idxs] *= r2_scale_factor
    dy2[close_idxs] *= r2_scale_factor
    dz2[close_idxs] *= r2_scale_factor

    # r2 = _check_rlimit_point(r2, r_limit)
    # r = np.sqrt(r2)
    sigma_r = np.sqrt(sigma[1] * sigma[2] * dx2
                    + sigma[0] * sigma[2] * dy2
                    + sigma[0] * sigma[1] * dz2)

    Emem = 1 / (4 * np.pi) * np.dot(currmem.T, 1./sigma_r)

    return Emem.transpose()



def _check_rlimit_point(r2, r_limit):
    """Correct r2 so that r2 >= r_limit**2 for all values"""
    inds = r2 < r_limit*r_limit
    r2[inds] = r_limit[inds]*r_limit[inds]
    
    return r2
