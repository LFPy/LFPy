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
    u[u > 1] = 1.0
    u[u < 0] = 0.0

    closest_point = np.array([xstart + u * px,
                              ystart + u * py,
                              zstart + u * pz])
    dist = np.sqrt(np.sum((closest_point.T - p)**2, axis=1))
    return dist, closest_point


def calc_lfp_linesource_anisotropic(cell, x, y, z, sigma, r_limit):
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
    """

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

    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart,
                                                   xend, yend, zend, pos)

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

    for idx in np.where(rs < r_limit)[0]:
        r, closest_point, l_vec = rs[idx], closest_points[:, idx], l_vecs[:, idx]

        p_ = pos.copy()
        if np.abs(r) < 1e-12:
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
            print(p_, closest_point)

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
        print(a, b, c)
        print(i, iia, iib, iii, iiii)
        raise RuntimeError

    mapping = np.zeros(cell.totnsegs)
    mapping[i] = _anisotropic_line_source_case_i(a[i], c[i])
    mapping[iia] = _anisotropic_line_source_case_iia(a[iia], c[iia])
    mapping[iib] = _anisotropic_line_source_case_iib(a[iib], b[iib], c[iib])
    mapping[iii] = _anisotropic_line_source_case_iii(a[iii], b[iii], c[iii])
    mapping[iiii] = _anisotropic_line_source_case_iiii(a[iiii], b[iiii], c[iiii])

    if np.isnan(mapping).any():
        raise RuntimeError("NaN")

    return 1 / (4 * np.pi) * mapping / np.sqrt(a)


def calc_lfp_soma_as_point_anisotropic(cell, x, y, z, sigma, r_limit):
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
    """

    xstart = cell.xstart
    xend = cell.xend
    ystart = cell.ystart
    yend = cell.yend
    zstart = cell.zstart
    zend = cell.zend
    l_vecs = np.array([xend - xstart, yend - ystart, zend - zstart])

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

    for idx in np.where(rs < r_limit)[0]:
        r, closest_point, l_vec = rs[idx], closest_points[:, idx], l_vecs[:, idx]

        p_ = pos.copy()
        if np.abs(r) < 1e-12:
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
            print(p_, closest_point)

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
        print(a, b, c)
        print(i, iia, iib, iii, iiii)
        raise RuntimeError

    mapping = np.zeros(cell.totnsegs)
    mapping[i] = _anisotropic_line_source_case_i(a[i], c[i])
    mapping[iia] = _anisotropic_line_source_case_iia(a[iia], c[iia])
    mapping[iib] = _anisotropic_line_source_case_iib(a[iib], b[iib], c[iib])
    mapping[iii] = _anisotropic_line_source_case_iii(a[iii], b[iii], c[iii])
    mapping[iiii] = _anisotropic_line_source_case_iiii(a[iiii], b[iiii], c[iiii])

    if np.isnan(mapping).any():
        raise RuntimeError("NaN")

    mapping /= np.sqrt(a)

    # Get compartment indices for somatic compartments (to be treated as point
    # sources)
    try:
        somainds = cell.get_idx("soma")
    except Exception:
        raise Exception('Call {}.get_idx("soma") failed in method LFPy.lfpcalc.calc_lfp_soma_as_point'.format(cell))
    
    dx2_soma = (cell.xmid[somainds] - x)**2
    dy2_soma = (cell.ymid[somainds] - y)**2
    dz2_soma = (cell.zmid[somainds] - z)**2

    r2_soma = dx2_soma + dy2_soma + dz2_soma

    # Go through and correct all (if any) somatic idxs that are too close
    for close_idx in np.where(np.abs(r2_soma) < 1e-6)[0]:
        dx2_soma[close_idx] += 0.001
        r2_soma[close_idx] += 0.001

    for close_idx in np.where(r2_soma < r_limit[somainds]**2)[0]:
        # For anisotropic media, the direction in which to move points matter.
        # Radial distance between point source and electrode is scaled to r_limit
        r2_scale_factor = r_limit[somainds[close_idx]]*r_limit[somainds[close_idx]] / r2_soma[close_idx]
        dx2_soma[close_idx] *= r2_scale_factor
        dy2_soma[close_idx] *= r2_scale_factor
        dz2_soma[close_idx] *= r2_scale_factor

    mapping[somainds] = 1/np.sqrt(sigma[1] * sigma[2] * dx2_soma
                    + sigma[0] * sigma[2] * dy2_soma
                    + sigma[0] * sigma[1] * dz2_soma)

    return 1. / (4 * np.pi) * mapping


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


def calc_lfp_linesource(cell, x, y, z, sigma, r_limit):

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
    """

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

    too_close_idxs = np.where(r2 < r_limit*r_limit)[0]
    r2[too_close_idxs] = r_limit[too_close_idxs]**2
    l = h + deltaS
    hnegi = h < 0
    hposi = h >= 0
    lnegi = l < 0
    lposi = l >= 0

    mapping = np.zeros(len(cell.xstart))

    #case i, h < 0, l < 0
    [i] = np.where(hnegi & lnegi)
    #case ii, h < 0, l >= 0
    [ii] = np.where(hnegi & lposi)
    #case iii, h >= 0, l >= 0
    [iii] = np.where(hposi & lposi)


    mapping[i] = _linesource_calc_case1(l[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l[iii], r2[iii], h[iii])
    return 1 / (4 * np.pi * sigma * deltaS) * mapping


def calc_lfp_soma_as_point(cell, x, y, z, sigma, r_limit):

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
    """
    # get compartment indices for somatic compartments (to be treated as point
    # sources)
    try:
        somainds = cell.get_idx("soma")
    except Exception:
        raise Exception('Call {}.get_idx("soma") failed in method LFPy.lfpcalc.calc_lfp_soma_as_point'.format(cell))

    #some variables for h, r2, r_soma calculations
    xstart = cell.xstart
    xmid = cell.xmid[somainds]
    xend = cell.xend
    ystart = cell.ystart
    ymid = cell.ymid[somainds]
    yend = cell.yend
    zstart = cell.zstart
    zmid = cell.zmid[somainds]
    zend = cell.zend

    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)
    r_soma = _r_soma_calc(xmid, ymid, zmid, x, y, z)
    if np.any(r_soma < r_limit[somainds]):
        print('Adjusting r-distance to soma segments')
        r_soma[r_soma < r_limit[somainds]
               ] = r_limit[somainds][r_soma < r_limit[somainds]]

    too_close_idxs = np.where(r2 < r_limit*r_limit)[0]
    r2[too_close_idxs] = r_limit[too_close_idxs]**2
    l = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l < 0
    lposi = l >= 0

    # Ensuring that soma is not treated as line-source
    hnegi[somainds] = hposi[somainds] = lnegi[somainds] = lposi[somainds] = False

    #Line sources
    #case i,  h < 0,  l < 0
    i = np.where(hnegi & lnegi)
    #case ii,  h < 0,  l >= 0
    ii = np.where(hnegi & lposi)
    #case iii,  h >= 0,  l >= 0
    iii = np.where(hposi & lposi)

    #Summarizing all potential contributions
    mapping = np.zeros(cell.totnsegs)
    mapping[somainds] = 1 / r_soma
    deltaS[somainds] = 1.

    mapping[i] = _linesource_calc_case1(l[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l[iii], r2[iii], h[iii])

    return 1 / (4 * np.pi * sigma * deltaS) * mapping


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


def _r_soma_calc(xmid, ymid, zmid, x, y, z):
    """calculate the distance to soma midpoint"""
    r_soma = np.sqrt((x - xmid)**2 + (y - ymid)**2 + (z - zmid)**2)
    return r_soma


def calc_lfp_pointsource(cell, x, y, z, sigma, r_limit):
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
    """

    r2 = (cell.xmid - x)**2 + (cell.ymid - y)**2 + (cell.zmid - z)**2
    r2 = _check_rlimit_point(r2, r_limit)
    mapping = 1 / (4 * np.pi * sigma * np.sqrt(r2))
    return mapping


def calc_lfp_pointsource_anisotropic(cell, x, y, z, sigma, r_limit):
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
    """

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

    sigma_r = np.sqrt(sigma[1] * sigma[2] * dx2
                    + sigma[0] * sigma[2] * dy2
                    + sigma[0] * sigma[1] * dz2)

    mapping = 1 / (4 * np.pi * sigma_r)
    return mapping


def _check_rlimit_point(r2, r_limit):
    """Correct r2 so that r2 >= r_limit**2 for all values"""
    inds = r2 < r_limit*r_limit
    r2[inds] = r_limit[inds]*r_limit[inds]
    return r2


def calc_lfp_pointsource_moi(cell, x, y, z, sigma_T, sigma_S, sigma_G,
                             steps, h, r_limit, **kwargs):
    """Calculate extracellular potentials using the point-source
    equation on all compartments for in vitro Microelectrode Array (MEA) slices

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
    sigma_T : float
        extracellular conductivity in tissue slice
    sigma_G : float
        Conductivity of MEA glass electrode plane.
        Should normally be zero for MEA set up.
    sigma_S : float
        Conductivity of saline bath that tissue slice is immersed in
    steps : int
        Number of steps to average over the in technically infinite sum
    h : float
        Slice thickness in um.
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    """

    dx2 = (x - cell.xmid)**2
    dy2 = (y - cell.ymid)**2
    dz2 = (z - cell.zmid)**2

    dL2 = dx2 + dy2
    inds = np.where(dL2 + dz2 < r_limit*r_limit)[0]
    dL2[inds] = r_limit[inds]*r_limit[inds] - dz2[inds]

    def _omega(dz):
        return 1/np.sqrt(dL2 + dz*dz)

    WTS = (sigma_T - sigma_S)/(sigma_T + sigma_S)
    WTG = (sigma_T - sigma_G)/(sigma_T + sigma_G)

    mapping = _omega(z - cell.zmid)
    mapping += (WTS * _omega(z + cell.zmid - 2*h) +
                WTG * _omega(z + cell.zmid))

    n = np.arange(1, steps)
    a = (WTS*WTG)**n[:, None] * (WTS * _omega(z + cell.zmid - 2*(n[:, None] + 1)*h) +
                                 WTG * _omega(z + cell.zmid + 2*n[:, None]*h) +
                                 _omega(z - cell.zmid + 2*n[:, None]*h) +
                                 _omega(z - cell.zmid - 2*n[:, None]*h))
    mapping += np.sum(a, axis=0)
    mapping *= 1/(4*np.pi*sigma_T)

    return mapping


def calc_lfp_linesource_moi(cell, x, y, z, sigma_T, sigma_S, sigma_G,
                             steps, h, r_limit, **kwargs):
    """Calculate extracellular potentials using the line-source
    equation on all compartments for in vitro Microelectrode Array (MEA) slices

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
    sigma_T : float
        extracellular conductivity in tissue slice
    sigma_G : float
        Conductivity of MEA glass electrode plane.
        Should normally be zero for MEA set up, and for this method,
        only zero valued sigma_G is supported.
    sigma_S : float
        Conductivity of saline bath that tissue slice is immersed in
    steps : int
        Number of steps to average over the in technically infinite sum
    h : float
        Slice thickness in um.
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    """

    if np.abs(z) > 1e-9:
        raise RuntimeError("This method can only handle electrodes "
                           "at the MEA plane z=0")
    if np.abs(sigma_G) > 1e-9:
        raise RuntimeError("This method can only handle sigma_G=0, i.e.,"
                           "a non-conducting MEA glass electrode plane.")

    xstart = cell.xstart
    xend = cell.xend
    ystart = cell.ystart
    yend = cell.yend
    zstart = cell.zstart
    zend = cell.zend
    x0, y0, z0 = cell.xstart, cell.ystart, cell.zstart
    x1, y1, z1 = cell.xend, cell.yend, cell.zend

    pos = np.array([x, y, z])
    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart,
                                                   xend, yend, zend, pos)
    z0_ = z0.copy()
    z0_[np.where(rs < r_limit)] = r_limit[np.where(rs < r_limit)]

    ds = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    factor_a = ds*ds
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    a_x = x - x0
    a_y = y - y0
    W = (sigma_T - sigma_S)/(sigma_T + sigma_S)
    num = np.zeros(factor_a.shape)
    den = np.zeros(factor_a.shape)

    def _omega(a_z):
        #See Rottman integration formula 46) page 137 for explanation

        factor_b = - a_x*dx - a_y*dy - a_z*dz
        factor_c = a_x*a_x + a_y*a_y + a_z*a_z
        b_2_ac = factor_b*factor_b - factor_a * factor_c

        case1_idxs = np.where(np.abs(b_2_ac) <= 1e-12)
        case2_idxs = np.where(np.abs(b_2_ac) > 1e-12)

        if not len(case1_idxs) == 0:
            num[case1_idxs] = factor_a[case1_idxs] + factor_b[case1_idxs]
            den[case1_idxs] = factor_b[case1_idxs]
        if not len(case2_idxs) == 0:
            num[case2_idxs] = (factor_a[case2_idxs] + factor_b[case2_idxs] +
                               ds[case2_idxs]*np.sqrt(factor_a[case2_idxs] +
                               2*factor_b[case2_idxs] + factor_c[case2_idxs]))
            den[case2_idxs] = (factor_b[case2_idxs] +
                               ds[case2_idxs]*np.sqrt(factor_c[case2_idxs]))
        return np.log(num/den)

    mapping = _omega(-z0_)
    n = 1
    while n < steps:
        mapping += W**n * (_omega(2*n*h - z0_) + _omega(-2*n*h - z0_))
        n += 1

    mapping *= 2/(4*np.pi*sigma_T * ds)

    return mapping


def calc_lfp_soma_as_point_moi(cell, x, y, z, sigma_T, sigma_S, sigma_G,
                             steps, h, r_limit, **kwargs):
    """Calculate extracellular potentials for in vitro
    Microelectrode Array (MEA) slices, where soma (compartment zero) is
    treated as a point source, and all other compartments as line sources.

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
    sigma_T : float
        extracellular conductivity in tissue slice
    sigma_G : float
        Conductivity of MEA glass electrode plane.
        Should normally be zero for MEA set up, and for this method,
        only zero valued sigma_G is supported.
    sigma_S : float
        Conductivity of saline bath that tissue slice is immersed in
    steps : int
        Number of steps to average over the in technically infinite sum
    h : float
        Slice thickness in um.
    r_limit : np.ndarray
        minimum distance to source current for each compartment
    """

    if np.abs(z) > 1e-9:
        raise RuntimeError("This method can only handle electrodes "
                           "at the MEA plane z=0")
    if np.abs(sigma_G) > 1e-9:
        raise RuntimeError("This method can only handle sigma_G=0, i.e.,"
                           "a non-conducting MEA glass electrode plane.")

    xstart = cell.xstart
    xend = cell.xend
    ystart = cell.ystart
    yend = cell.yend
    zstart = cell.zstart
    zend = cell.zend
    x0, y0, z0 = cell.xstart, cell.ystart, cell.zstart
    x1, y1, z1 = cell.xend, cell.yend, cell.zend

    pos = np.array([x, y, z])
    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart,
                                                   xend, yend, zend, pos)
    z0_ = np.array(z0)
    if np.any(rs < r_limit):
        z0_[rs < r_limit] = r_limit


    ds = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    factor_a = ds*ds
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    a_x = x - x0
    a_y = y - y0
    W = (sigma_T - sigma_S)/(sigma_T + sigma_S)
    num = np.zeros(factor_a.shape)
    den = np.zeros(factor_a.shape)

    def _omega(a_z):
        #See Rottman integration formula 46) page 137 for explanation

        factor_b = - a_x*dx - a_y*dy - a_z*dz
        factor_c = a_x*a_x + a_y*a_y + a_z*a_z
        b_2_ac = factor_b*factor_b - factor_a * factor_c

        case1_idxs = np.where(np.abs(b_2_ac) <= 1e-12)
        case2_idxs = np.where(np.abs(b_2_ac) > 1e-12)

        if not len(case1_idxs) == 0:
            num[case1_idxs] = factor_a[case1_idxs] + factor_b[case1_idxs]
            den[case1_idxs] = factor_b[case1_idxs]
        if not len(case2_idxs) == 0:
            num[case2_idxs] = (factor_a[case2_idxs] + factor_b[case2_idxs] +
                               ds[case2_idxs]*np.sqrt(factor_a[case2_idxs] +
                               2*factor_b[case2_idxs] + factor_c[case2_idxs]))
            den[case2_idxs] = (factor_b[case2_idxs] +
                               ds[case2_idxs]*np.sqrt(factor_c[case2_idxs]))
        return np.log(num/den)

    mapping = _omega(-z0_)
    n = 1
    while n < steps:
        mapping += W**n * (_omega(2*n*h - z0) + _omega(-2*n*h - z0))
        n += 1

    mapping *= 2/(4*np.pi*sigma_T * ds)

    # NOW DOING SOMA

    # get compartment indices for somatic compartments (to be treated as point
    # sources)
    try:
        somainds = cell.get_idx("soma")
    except Exception:
        raise Exception('Call {}.get_idx("soma") failed in method LFPy.lfpcalc.calc_lfp_soma_as_point'.format(cell))

    dx2 = (x - cell.xmid[somainds])**2
    dy2 = (y - cell.ymid[somainds])**2
    dz2 = (z - cell.zmid[somainds])**2

    dL2 = dx2 + dy2
    inds = np.where(dL2 + dz2 < r_limit[somainds]*r_limit[somainds])[0]
    dL2[inds] = r_limit[inds]*r_limit[inds] - dz2[inds]

    def _omega(dz):
        return 1/np.sqrt(dL2 + dz*dz)

    mapping[somainds] = _omega(z - cell.zmid[somainds])
    mapping[somainds] += (W * _omega(cell.zmid[somainds] - 2*h) +
                       _omega(cell.zmid[somainds]))

    n = np.arange(1, steps)
    a = (W)**n[:, None] * (W * _omega(+ cell.zmid[somainds] - 2*(n[:, None] + 1)*h) +
                           2 * _omega(+ cell.zmid[somainds] + 2*n[:, None]*h) +
                               _omega(+ cell.zmid[somainds] - 2*n[:, None]*h))
    mapping[somainds] += np.sum(a, axis=0)
    mapping[somainds] *= 1/(4*np.pi*sigma_T)

    return mapping