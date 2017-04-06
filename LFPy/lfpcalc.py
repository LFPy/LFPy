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


def return_dist_from_segments(xstart, ystart, zstart, xend, yend, zend, p):
    """
    Returns distance and closest point on line segments from point p
    """
    px = xend-xstart
    py = yend-ystart
    pz = zend-zstart

    delta = px*px + py*py + pz*pz
    u = ((p[0] - xstart) * px + (p[1] - ystart) * py + (p[2] - zstart) * pz) / delta
    u[u > 1] = 1
    u[u < 0] = 0

    closest_point = np.array([xstart + u * px,
                              ystart + u * py,
                              zstart + u * pz])
    dist = np.sqrt(np.sum((closest_point.T - p)**2, axis=1))
    return dist, closest_point


def calc_lfp_linesource_anisotropic(cell, x, y, z, sigma,
                        r_limit, t_indices=None):
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
        extracellular conductivity [sigma_x, sigma_y, sigma_z]
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """

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
    l_vecs = np.array([xend - xstart,
                      yend - ystart,
                      zend - zstart])

    pos = np.array([x, y, z])

    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart, xend, yend, zend, pos)

    dx2 = (xend - xstart)**2
    dy2 = (yend - ystart)**2
    dz2 = (zend - zstart)**2
    a = (sigma[1] * sigma[2] * dx2 +
         sigma[0] * sigma[2] * dy2 +
         sigma[0] * sigma[1] * dz2)

    b = -2 * (sigma[1] * sigma[2] * (x - xstart) * (xend - xstart) +
              sigma[0] * sigma[2] * (y - ystart) * (yend - ystart) +
              sigma[0] * sigma[1] * (z - zstart) * (zend - zstart))
    c = (sigma[1] * sigma[2] * (x - xstart)**2 +
         sigma[0] * sigma[2] * (y - ystart)**2 +
         sigma[0] * sigma[1] * (z - zstart)**2)

    for idx in np.where(rs <r_limit)[0]:
        r, closest_point, l_vec = rs[idx], closest_points[:, idx], l_vecs[:, idx]

        p_ = pos.copy()
        if np.abs(r) < 1e-12:
            # print "r is zero"
            if np.abs(l_vec[0]) < 1e-12:
                p_[0] += r_limit[idx]
            elif np.abs(l_vec[1]) < 1e-12:
                p_[1] += r_limit[idx]
            elif np.abs(l_vec[2]) < 1e-12:
                p_[2] += r_limit[idx]
            else:
                displace_vec = np.array([-l_vec[1], l_vec[0], 0])
                displace_vec = displace_vec / np.sqrt(np.sum(displace_vec**2)) * r_limit[idx]
                p_[:] += displace_vec
        else:
            p_[:] = pos + (pos - closest_point) * (r_limit[idx] - r) / r

        if np.sqrt(np.sum((p_ - closest_point)**2)) - r_limit[idx] > 1e-9:
            print p_, closest_point
            raise RuntimeError("Segment adjustment not working")

        b[idx] = -2 * (sigma[1] * sigma[2] * (p_[0] - xstart[idx]) * (xend[idx] - xstart[idx]) +
                   sigma[0] * sigma[2] * (p_[1] - ystart[idx]) * (yend[idx] - ystart[idx]) +
                   sigma[0] * sigma[1] * (p_[2] - zstart[idx]) * (zend[idx] - zstart[idx]))
        c[idx] = (sigma[1] * sigma[2] * (p_[0] - xstart[idx])**2 +
              sigma[0] * sigma[2] * (p_[1] - ystart[idx])**2 +
              sigma[0] * sigma[1] * (p_[2] - zstart[idx])**2)

    [i] = np.where(np.abs(b) <= 1e-6)
    [iia] = np.where(np.bitwise_and(np.abs(4 * a * c - b*b) < 1e-6, np.abs(a - c) < 1e-6))
    [iib] = np.where(np.bitwise_and(np.abs(4 * a * c - b*b) < 1e-6, np.abs(a - c) >= 1e-6))
    [iii] = np.where(np.bitwise_and(4 * a * c - b*b < -1e-6, np.abs(b) > 1e-6))
    [iiii] = np.where(np.bitwise_and(4 * a * c - b*b > 1e-6, np.abs(b) > 1e-6))

    if len(i) + len(iia) + len(iib) + len(iii) + len(iiii) != cell.totnsegs:
        print a, b, c
        print i, iia, iib, iii, iiii
        raise RuntimeError

    # if len(iiii) != cell.totnsegs:
    #     print len(i), len(iia), len(iib), len(iii), len(iiii)
    mapping = np.zeros(cell.totnsegs)
    mapping[i] = _anisotropic_line_source_case_i(a[i], c[i])
    mapping[iia] = _anisotropic_line_source_case_iia(a[iia], c[iia])
    mapping[iib] = _anisotropic_line_source_case_iib(a[iib], b[iib], c[iib])
    mapping[iii] = _anisotropic_line_source_case_iii(a[iii], b[iii], c[iii])
    mapping[iiii] = _anisotropic_line_source_case_iiii(a[iiii], b[iiii], c[iiii])

    if np.isnan(mapping).any():
        raise RuntimeError("NaN")

    phi = 1 / (4 * np.pi) * np.dot(currmem.T, mapping / np.sqrt(a))

    return phi.T


def calc_lfp_soma_as_point_anisotropic(cell, x, y, z, sigma,
                                       r_limit, t_indices=None):
    """Calculate electric field potential, soma is treated as point source, all
    compartments except soma are treated as line sources.

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
        extracellular conductivity [sigma_x, sigma_y, sigma_z]
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem

    xstart = cell.xstart
    xend = cell.xend
    ystart = cell.ystart
    yend = cell.yend
    zstart = cell.zstart
    zend = cell.zend
    l_vecs = np.array([xend - xstart,
                      yend - ystart,
                      zend - zstart])

    pos = np.array([x, y, z])

    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart, xend, yend, zend, pos)

    dx2 = (xend - xstart)**2
    dy2 = (yend - ystart)**2
    dz2 = (zend - zstart)**2
    a = (sigma[1] * sigma[2] * dx2 +
         sigma[0] * sigma[2] * dy2 +
         sigma[0] * sigma[1] * dz2)

    b = -2 * (sigma[1] * sigma[2] * (x - xstart) * (xend - xstart) +
              sigma[0] * sigma[2] * (y - ystart) * (yend - ystart) +
              sigma[0] * sigma[1] * (z - zstart) * (zend - zstart))
    c = (sigma[1] * sigma[2] * (x - xstart)**2 +
         sigma[0] * sigma[2] * (y - ystart)**2 +
         sigma[0] * sigma[1] * (z - zstart)**2)

    for idx in np.where(rs <r_limit)[0]:
        r, closest_point, l_vec = rs[idx], closest_points[:, idx], l_vecs[:, idx]

        p_ = pos.copy()
        if np.abs(r) < 1e-12:
            # print "r is zero"
            if np.abs(l_vec[0]) < 1e-12:
                p_[0] += r_limit[idx]
            elif np.abs(l_vec[1]) < 1e-12:
                p_[1] += r_limit[idx]
            elif np.abs(l_vec[2]) < 1e-12:
                p_[2] += r_limit[idx]
            else:
                displace_vec = np.array([-l_vec[1], l_vec[0], 0])
                displace_vec = displace_vec / np.sqrt(np.sum(displace_vec**2)) * r_limit[idx]
                p_[:] += displace_vec
        else:
            p_[:] = pos + (pos - closest_point) * (r_limit[idx] - r) / r

        if np.sqrt(np.sum((p_ - closest_point)**2)) - r_limit[idx] > 1e-9:
            print p_, closest_point
            raise RuntimeError("Segment adjustment not working")

        b[idx] = -2 * (sigma[1] * sigma[2] * (p_[0] - xstart[idx]) * (xend[idx] - xstart[idx]) +
                   sigma[0] * sigma[2] * (p_[1] - ystart[idx]) * (yend[idx] - ystart[idx]) +
                   sigma[0] * sigma[1] * (p_[2] - zstart[idx]) * (zend[idx] - zstart[idx]))
        c[idx] = (sigma[1] * sigma[2] * (p_[0] - xstart[idx])**2 +
              sigma[0] * sigma[2] * (p_[1] - ystart[idx])**2 +
              sigma[0] * sigma[1] * (p_[2] - zstart[idx])**2)

    [i] = np.where(np.abs(b) <= 1e-6)
    [iia] = np.where(np.bitwise_and(np.abs(4 * a * c - b*b) < 1e-6, np.abs(a - c) < 1e-6))
    [iib] = np.where(np.bitwise_and(np.abs(4 * a * c - b*b) < 1e-6, np.abs(a - c) >= 1e-6))
    [iii] = np.where(np.bitwise_and(4 * a * c - b*b < -1e-6, np.abs(b) > 1e-6))
    [iiii] = np.where(np.bitwise_and(4 * a * c - b*b > 1e-6, np.abs(b) > 1e-6))

    if len(i) + len(iia) + len(iib) + len(iii) + len(iiii) != cell.totnsegs:
        print a, b, c
        print i, iia, iib, iii, iiii
        raise RuntimeError

    # if len(iiii) != cell.totnsegs:
    #     print len(i), len(iia), len(iib), len(iii), len(iiii)
    mapping = np.zeros(cell.totnsegs)
    mapping[i] = _anisotropic_line_source_case_i(a[i], c[i])
    mapping[iia] = _anisotropic_line_source_case_iia(a[iia], c[iia])
    mapping[iib] = _anisotropic_line_source_case_iib(a[iib], b[iib], c[iib])
    mapping[iii] = _anisotropic_line_source_case_iii(a[iii], b[iii], c[iii])
    mapping[iiii] = _anisotropic_line_source_case_iiii(a[iiii], b[iiii], c[iiii])

    if np.isnan(mapping).any():
        raise RuntimeError("NaN")

    mapping /= np.sqrt(a)

    # Treat soma as point source
    dx2_soma = (cell.xmid[0] - x)**2
    dy2_soma = (cell.ymid[0] - y)**2
    dz2_soma = (cell.zmid[0] - z)**2

    r2_soma = dx2_soma + dy2_soma + dz2_soma

    if np.abs(r2_soma) < 1e-6:
        dx2_soma += 0.001
        r2_soma += 0.001


    if r2_soma < r_limit[0]**2:
        # For anisotropic media, the direction in which to move points matter.
        # Radial distance between point source and electrode is scaled to r_limit
        r2_scale_factor = r_limit[0]*r_limit[0] / r2_soma
        dx2_soma *= r2_scale_factor
        dy2_soma *= r2_scale_factor
        dz2_soma *= r2_scale_factor

    mapping[0] = 1/np.sqrt(sigma[1] * sigma[2] * dx2_soma
                    + sigma[0] * sigma[2] * dy2_soma
                    + sigma[0] * sigma[1] * dz2_soma)

    phi = 1 / (4 * np.pi) * np.dot(currmem.T, mapping)
    return phi.T


def _anisotropic_line_source_case_i(a, c):
    return np.log(np.sqrt(a / c) + np.sqrt(a / c + 1))


def _anisotropic_line_source_case_iia(a, c):
    return np.log(np.abs(1 + np.sqrt(a / c)))


def _anisotropic_line_source_case_iib(a, b, c):
    return np.abs(np.log(np.abs(np.sign(b) * np.sqrt(a/c) + 1)))


def _anisotropic_line_source_case_iii(a, b, c):
    return np.log(np.abs((2 * a + b + 2 * np.sqrt(a * (a + b + c)))
                        / (b + 2 * np.sqrt(a * c))))


def _anisotropic_line_source_case_iiii(a, b, c):
    return (np.arcsinh((2 * a + b) / np.sqrt(4 * a * c - b*b)) -
                        np.arcsinh(b / np.sqrt(4 * a * c - b*b)))


def calc_lfp_linesource(cell, x, y, z, sigma, r_limit, t_indices=None):
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
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """

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

    Emem = np.dot(currmem.T, 1 / (4 * np.pi * sigma * deltaS) * mapping)
    
    return Emem.T

def calc_lfp_soma_as_point(cell, x, y, z, sigma, r_limit, t_indices=None):
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
    r_limit : np.ndarray
         minimum distance to source current for each compartment.
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """

    s_limit = r_limit[0]

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

    #Summarizing all potential contributions

    mapping = np.zeros(cell.totnsegs)
    mapping[0] = 1 / r_soma
    deltaS[0] = 1.

    mapping[i] = _linesource_calc_case1(l[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l[iii], r2[iii], h[iii])

    Emem = np.dot(currmem.T, 1 / (4 * np.pi * sigma * deltaS) * mapping)

    return Emem.T

def _linesource_calc_case1(l_i, r2_i, h_i):
    """Calculates linesource contribution for case i"""
    bb = np.sqrt(h_i*h_i + r2_i) - h_i
    cc = np.sqrt(l_i*l_i + r2_i) - l_i
    dd = np.log(bb / cc)
    return dd

def _linesource_calc_case2(l_ii, r2_ii, h_ii):
    """Calculates linesource contribution for case ii"""
    bb = np.sqrt(h_ii*h_ii + r2_ii) - h_ii
    cc = (l_ii + np.sqrt(l_ii*l_ii + r2_ii)) / r2_ii
    dd = np.log(bb * cc)
    return dd
    
def _linesource_calc_case3(l_iii, r2_iii, h_iii):
    """Calculates linesource contribution for case iii"""
    bb = np.sqrt(l_iii*l_iii + r2_iii) + l_iii
    cc = np.sqrt(h_iii*h_iii + r2_iii) + h_iii
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
            if (h[idx] < r_limit[idx]) and ((deltaS[idx]+h[idx]) > -r_limit[idx]):
                print('Adjusting distance to segment %s from %.2f to %.2f.'
                      % (idx, r2[idx]**0.5, r_limit[idx]))
                r2[idx] = r_limit[idx]*r_limit[idx]
    return r2

def _r_soma_calc(xmid, ymid, zmid, x, y, z):
    """calculate the distance to soma midpoint"""
    r_soma = np.sqrt((x - xmid)**2 + (y - ymid)**2 + (z - zmid)**2)
    return r_soma

def calc_lfp_pointsource(cell, x, y, z, sigma, r_limit, t_indices=None):
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
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem
    
    r2 = (cell.xmid - x)**2 + (cell.ymid - y)**2 + (cell.zmid - z)**2
    r2 = _check_rlimit_point(r2, r_limit)
    r = np.sqrt(r2)
    
    Emem = 1 / (4 * np.pi * sigma) * np.dot(currmem.T, 1/r)
    
    return Emem.T

def calc_lfp_pointsource_anisotropic(cell, x, y, z, sigma, r_limit,
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
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    t_indices : [None]/np.ndarray
        calculate LFP at specific timesteps
    """

    if t_indices is not None:
        currmem = cell.imem[:, t_indices]
    else:
        currmem = cell.imem

    dx2 = (cell.xmid - x)**2
    dy2 = (cell.ymid - y)**2
    dz2 = (cell.zmid - z)**2

    r2 = dx2 + dy2 + dz2
    if (np.abs(r2) < 1e-6).any():
        dx2[np.abs(r2) < 1e-6] += 0.001
        r2[np.abs(r2) < 1e-6] += 0.001

    close_idxs = r2 < r_limit*r_limit

    # For anisotropic media, the direction in which to move points matter.
    # Radial distance between point source and electrode is scaled to r_limit
    r2_scale_factor = r_limit[close_idxs]*r_limit[close_idxs] / r2[close_idxs]
    dx2[close_idxs] *= r2_scale_factor
    dy2[close_idxs] *= r2_scale_factor
    dz2[close_idxs] *= r2_scale_factor

    # r2 = _check_rlimit_point(r2, r_limit)
    # r = np.sqrt(r2)
    sigma_r = np.sqrt(sigma[1] * sigma[2] * dx2
                    + sigma[0] * sigma[2] * dy2
                    + sigma[0] * sigma[1] * dz2)

    Emem = 1 / (4 * np.pi) * np.dot(currmem.T, 1./sigma_r)
    return Emem.T


def _check_rlimit_point(r2, r_limit):
    """Correct r2 so that r2 >= r_limit**2 for all values"""
    inds = r2 < r_limit*r_limit
    r2[inds] = r_limit[inds]*r_limit[inds]
    return r2
