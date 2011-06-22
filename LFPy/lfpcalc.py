#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import pylab as pl
import LFPy

def calc_lfp_choose(c, x=0, y=0, z=0, sigma=0.3, \
                    r_limit=None, from_file=False, \
                    timestep=None, t_indices=None, som_as_point=False):
    '''Determine which method to use, line-source for soma default'''
    if som_as_point:
        return calc_lfp_som_as_point(c, x=x, y=y, z=z, sigma=sigma, \
                                     r_limit=r_limit, from_file=from_file, \
                                     t_indices=t_indices)
    else:
        return calc_lfp_linesource(c, x=x, y=y, z=z, sigma=sigma, \
                                   r_limit=r_limit, from_file=from_file, \
                                   t_indices=t_indices)


def calc_lfp_linesource(c, x=0, y=0, z=0, sigma=0.3, \
                        r_limit=None, from_file=False, timestep=None, t_indices=None):
    '''Calculate electric field potential using the line-source method, all
    compartments treated as line sources, even soma.'''
    if from_file:
        c = LFPy.tools.load(c)

    # Handling the r_limits. If a r_limit is a single value, an array r_limit
    # of shape c.diam is returned.
    if type(r_limit) == int or type(r_limit) == float:
        r_limit = pl.ones(pl.shape(c.diam))*abs(r_limit)
    elif pl.shape(r_limit) != pl.shape(c.diam):
        raise Exception, 'r_limit is neither a float- or int- value, nor is \
            r_limit.shape() equal to c.diam.shape()'

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

    r2 = _check_rlimit(r2, r_limit, h, deltaS)

    l = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l < 0
    lposi = l >= 0

    #case i, h < 0, l < 0
    [i] = pl.where(hnegi & lnegi)
    #case ii, h < 0, l > 0
    [ii] = pl.where(hnegi & lposi)
    #case iii, h > 0, l > 0
    [iii] = pl.where(hposi & lposi)

    Ememi = _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h)
    Ememii = _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h)
    Ememiii = _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h)

    Emem = Ememi + Ememii + Ememiii

    return Emem.transpose()

def calc_lfp_som_as_point(c, x=0, y=0, z=0, sigma=0.3, \
                          r_limit=None, from_file=False, timestep=None, t_indices=None):
    '''Calculate electric field potential using the line-source method,
    soma is treated as point/sphere source'''

    if from_file:
        c = LFPy.tools.load(c)

    #Handling the r_limits. If a r_limit is a single value,
    #an array r_limit of shape c.diam is returned.
    if type(r_limit) != type(pl.array([])):
        r_limit = pl.array(r_limit)
    if r_limit.shape == ():
        s_limit = r_limit
        r_limit = pl.ones(c.diam.size) * abs(r_limit)
    elif r_limit.shape == (2, ):
        s_limit = abs(r_limit[0])
        r_limit = pl.ones(c.diam.size) * abs(r_limit[1])
    elif r_limit.shape == c.diam.shape:
        s_limit = r_limit[0]
        r_limit = r_limit
    else:
        raise Exception,  'r_limit is neither a float- or int- value, \
            on the form r_limit=[s_limit, r_limit],  \
            nor is shape(r_limit) equal to shape(c.diam)!'

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
    if pl.sum(pl.nonzero( r2 < r_limit*r_limit )) > 0:
        for idx in pl.nonzero( r2[1:] < r_limit[1:] * r_limit[1:] )[0]+1:
            if (h[idx] < r_limit[idx]) and \
            ((deltaS[idx] + h[idx]) > -r_limit[idx]):
                print 'Adjusting distance to segment ', str(idx), ' from ',\
                str(pl.sqrt(r2[idx])), ' to ', str(r_limit[idx]), '.'
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
    i = pl.where(hnegi & lnegi)
    #case ii,  h < 0,  l > 0
    ii = pl.where(hnegi & lposi)
    #case iii,  h > 0,  l > 0
    iii = pl.where(hposi & lposi)

    Ememi = _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h)
    Ememii = _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h)
    Ememiii = _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h)

    #Potential contribution from soma
    Emem0 = currmem[0]/(4 * pl.pi * sigma * r_soma)

    #Summarizing all potential contributions
    Emem = Emem0 + Ememi + Ememiii + Ememii

    return Emem.transpose()

def _Ememi_calc(i, currmem, sigma, deltaS, l, r2, h):
    '''Subroutine used by calc_lfp_som_as_point()'''
    currmem_iT = currmem[i].transpose()
    deltaS_i = deltaS[i]
    l_i = l[i]
    r2_i = r2[i]
    h_i = h[i]
    #sigma = sigma

    aa = 1 / ( 4 * pl.pi * sigma * deltaS_i)
    bb = pl.sqrt(h_i**2 + r2_i) - h_i
    cc = pl.sqrt(l_i**2 + r2_i) - l_i
    dd = aa * pl.log(bb / cc)

    Emem_i = pl.dot(currmem_iT, dd)

    return Emem_i

def _Ememii_calc(ii, currmem, sigma, deltaS, l, r2, h):
    '''Subroutine used by calc_lfp_som_as_point()'''
    currmem_iiT = currmem[ii].transpose()
    deltaS_ii = deltaS[ii]
    l_ii = l[ii]
    r2_ii = r2[ii]
    h_ii = h[ii]

    aa = 1 / (4 * pl.pi * sigma * deltaS_ii)
    bb = pl.sqrt(h_ii**2 + r2_ii) - h_ii
    cc = (l_ii + pl.sqrt(l_ii**2 + r2_ii)) / r2_ii
    dd = aa * pl.log(bb * cc)

    Emem_ii = pl.dot(currmem_iiT, dd)

    return Emem_ii

def _Ememiii_calc(iii, currmem, sigma, deltaS, l, r2, h):
    '''Subroutine used by calc_lfp_som_as_point()'''
    currmem_iiiT = currmem[iii].transpose()
    l_iii = l[iii]
    r2_iii = r2[iii]
    h_iii = h[iii]
    deltaS_iii = deltaS[iii]

    aa = 1 / (4 * pl.pi * sigma * deltaS_iii)
    bb = pl.sqrt(l_iii**2 + r2_iii) + l_iii
    cc = pl.sqrt(h_iii**2 + r2_iii) + h_iii
    dd = aa * pl.log(bb / cc)

    Emem_iii = pl.dot(currmem_iiiT, dd)

    return Emem_iii

def _deltaS_calc(xstart, xend, ystart, yend, zstart, zend):
    '''Subroutine used by calc_lfp_som_as_point()'''
    deltaS = pl.sqrt( (xstart - xend)**2 + (ystart - yend)**2 + \
        (zstart-zend)**2)

    return deltaS

def _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z):
    '''Subroutine used by calc_lfp_som_as_point()'''
    h  = pl.zeros(xstart.size)
    for i in xrange(1, xstart.size):
        aa = [x - xend[i], y - yend[i], z - zend[i]]
        bb = [xend[i] - xstart[i], yend[i] - ystart[i], \
          zend[i] - zstart[i]]
        cc = pl.dot(aa, bb)
        h[i] = cc / deltaS[i]
    return h

def _r2_calc(xend, yend, zend, x, y, z, h):
    '''Subroutine used by calc_lfp_*()'''
    r2 = (x-xend)**2 + (y-yend)**2 + (z-zend)**2 - h**2
    
    return abs(r2)

def _check_rlimit(r2, r_limit, h, deltaS):
    '''Check that no segment is close the electrode than r_limit'''
    if pl.sum(pl.nonzero( r2 < r_limit*r_limit )) > 0:
        for idx in pl.nonzero( r2 < r_limit*r_limit )[0]:
            if (h[idx] < r_limit[idx]) and ((deltaS[idx]+h[idx])>-r_limit[idx]):
                print 'Adjusting distance to segment ',str(idx),' from ', \
                     str(pl.sqrt(r2[idx])),' to ',str(r_limit[idx]),'.'
                r2[idx] = r_limit[idx]**2
    return r2

def _r_soma_calc(xmid, ymid, zmid, x, y, z):
    '''calculate the distance to soma midpoint'''
    r_soma = pl.sqrt((x - xmid)**2 + (y - ymid)**2 + \
        (z - zmid)**2)

    return r_soma
