#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017 Computational Neuroscience Group, NMBU.

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
from scipy.special import eval_legendre, lpmv
import numpy as np

class FourSphereVolumeConductor(object):
    """
    Main class for computing extracellular potentials in a four-sphere
    volume conductor model that assumes homogeneous, isotropic, linear
    (frequency independent) conductivity within the inner sphere and outer
    shells. The conductance outside the outer shell is 0 (air).

    Parameters
    ----------
    radii : list, dtype=float
        Len 4 list with the outer radii in units of (µm) for the 4
        concentric shells in the four-sphere model: brain, csf, skull and
        scalp, respectively.
    sigmas : list, dtype=float
        Len 4 list with the electrical conductivity in units of (S/m) of
        the four shells in the four-sphere model: brain, csf, skull and
        scalp, respectively.
    r : ndarray, dtype=float
        Shape (n_contacts, 3) array containing n_contacts electrode locations
        in cartesian coordinates in units of (µm).
        All r_el in r must be less than or equal to scalp radius
        and larger than the distance between dipole and sphere
        center: |rz| < r_el <= radii[3].
    rz : ndarray, dtype=float
        Shape (3, ) array containing the position of the current dipole in
        cartesian coordinates. Units of (µm).

    Examples
    --------
    Compute extracellular potential from current dipole moment in four-sphere
    head model

    >>> import LFPy
    >>> import numpy as np
    >>> radii = [79000., 80000., 85000., 90000.]
    >>> sigmas = [0.3, 1.5, 0.015, 0.3]
    >>> r = np.array([[0., 0., 90000.], [0., 85000., 0.]])
    >>> rz = np.array([0., 0., 78000.])
    >>> sphere_model = LFPy.FourSphereVolumeConductor(radii, sigmas, r, rz)
    >>> # current dipole moment
    >>> p = np.array([[10., 10., 10.]]*10) # 10 timesteps
    >>> # compute potential
    >>> potential = sphere_model.calc_potential(p)

    """

    def __init__(self, radii, sigmas, r, rz):
        """Initialize class FourSphereVolumeConductor"""
        self.r1 = radii[0]
        self.r2 = radii[1]
        self.r3 = radii[2]
        self.r4 = radii[3]

        self.sigma1 = sigmas[0]
        self.sigma2 = sigmas[1]
        self.sigma3 = sigmas[2]
        self.sigma4 = sigmas[3]

        self.r12 = self.r1 / self.r2
        self.r21 = self.r2 / self.r1
        self.r23 = self.r2 / self.r3
        self.r32 = self.r3 / self.r2
        self.r34 = self.r3 / self.r4
        self.r43 = self.r4 / self.r3

        self.sigma12 = self.sigma1 / self.sigma2
        self.sigma21 = self.sigma2 / self.sigma1
        self.sigma23 = self.sigma2 / self.sigma3
        self.sigma32 = self.sigma3 / self.sigma2
        self.sigma34 = self.sigma3 / self.sigma4
        self.sigma43 = self.sigma4 / self.sigma3

        self.rxyz = r
        self.rzloc = rz
        self.rz = np.sqrt(np.sum(rz ** 2))
        self.rz1 = self.rz / self.r1
        self.r = np.sqrt(np.sum(r ** 2, axis=1))

    def calc_potential(self, p):
        """
        Return electric potential from current dipole moment p

        Parameters
        ----------
        p : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing the x,y,z components of the
            current dipole moment in units of (nA*µm) for all timesteps

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) FourSphereVolumeConductor.r in units
            of (mV) for all timesteps of current dipole moment p

        """
        p_rad, p_tan = self._decompose_dipole(p)
        pot_rad = self._calc_rad_potential(p_rad)
        pot_tan = self._calc_tan_potential(p_tan)

        pot_tot = pot_rad + pot_tan
        mask = np.isnan(pot_tot)
        return np.ma.masked_array(pot_tot, mask=mask)

    def _decompose_dipole(self, p):
        """
        Decompose current dipole moment vector in radial and tangential terms

        Parameters
        ----------
        p : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing the x,y,z-components of the
            current dipole moment in units of (nA*µm) for all timesteps

        Returns:
        -------
        p_rad : ndarray, dtype=float
            Shape (n_timesteps, 3) array, radial part of p, parallel to self.rz
        p_tan : ndarray, dtype=float
            Shape (n_timesteps, 3) array, tangential part of p,
            orthogonal to self.rz
        """
        p_rad = (np.dot(p, self.rzloc)/self.rz ** 2
                 ).reshape(len(p), 1) * self.rzloc.reshape(1, len(self.rzloc))
        p_tan = p - p_rad

        return p_rad, p_tan

    def _calc_rad_potential(self, p_rad):
        """
        Return potential from radial dipole p_rad at location rz measured at r

        Parameters
        ----------
        p_rad : ndarray, dtype=float
            Shape (n_timesteps, 3) array, radial part of p
            in units of (nA*µm), parallel to self.rz

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the extracecllular
            potential at n_contacts contact point(s) FourSphereVolumeConductor.r
            in units of (mV) for all timesteps of p_rad
        """

        p_tot = np.linalg.norm(p_rad, axis=1)
        theta = self.calc_theta()
        s_vector = self._sign_rad_dipole(p_rad)
        phi_const = s_vector * p_tot / (4 * np.pi * self.sigma1 * self.rz ** 2)
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            el_rad = self.r[el_point]
            theta_point = theta[el_point]

            if el_rad <= self.rz:
                n_terms[el_point] = np.nan
                UserWarning('Electrode must be farther away from brain center than dipole. r = %s, rz = %s', self.r, self.rz)
            elif el_rad <= self.r1:
                n_terms[el_point] = self._potential_brain_rad(el_rad,
                                                             theta_point)
            elif el_rad <= self.r2:
                n_terms[el_point] = self._potential_csf_rad(el_rad,
                                                           theta_point)
            elif el_rad <= self.r3:
                n_terms[el_point] = self._potential_skull_rad(el_rad,
                                                             theta_point)
            elif el_rad <= (self.r4):
                n_terms[el_point] = self._potential_scalp_rad(el_rad,
                                                             theta_point)
            else:
                n_terms[el_point] = 0.
                UserWarning('Electrode located outside head model. Maximum r = %s µm.',
                                 self.r4, '\n your r = ', self.r)
        potential = phi_const * n_terms
        return potential

    def _calc_tan_potential(self, p_tan):
        """
        Return potential from tangential dipole P at location rz measured at r

        Parameters
        ----------
        p_tan : ndarray, dtype=float
            Shape (n_timesteps, 3) array, tangential part of p
            in units of (nA*µm), orthogonal to self.rz

        Returns
        _______
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the extracecllular
            potential at n_contacts contact point(s) FourSphereVolumeConductor.r
            in units of (mV) for all timesteps of p_tan
        """
        theta = self.calc_theta()
        phi = self.calc_phi(p_tan)
        p_tot = np.linalg.norm(p_tan, axis=1)
        phi_hom = - p_tot / (4 * np.pi * self.sigma1 * self.rz ** 2) * np.sin(phi)
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            el_rad = self.r[el_point]
            theta_point = theta[el_point]
            if el_rad <= self.rz:
                n_terms[el_point] = np.nan
                UserWarning('Electrode must be farther away from brain center than dipole. r = %s, rz = %s', self.r, self.rz)
            elif el_rad <= self.r1:
                n_terms[el_point] = self._potential_brain_tan(el_rad, theta_point)
            elif el_rad <= self.r2:
                n_terms[el_point] = self._potential_csf_tan(el_rad, theta_point)
            elif el_rad <= self.r3:
                n_terms[el_point] = self._potential_skull_tan(el_rad, theta_point)
            elif el_rad <= self.r4:
                n_terms[el_point] = self._potential_scalp_tan(el_rad, theta_point)
            else:
                n_terms[el_point] = 0.
                UserWarning('Electrode located outside head model. Maximum r = %s µm.',
                                 self.r4, '\n your r = ', self.r)
        potential = phi_hom * n_terms
        return potential

    def calc_theta(self):
        """
        Return polar angle(s) between rzloc and contact point location(s)

        Returns
        -------
        theta : ndarray, dtype=float
            Shape (n_contacts, ) array containing polar angle
            in units of (radians) between z-axis and n_contacts contact
            point location vector(s) in FourSphereVolumeConductor.rxyz
            z-axis is defined in the direction of rzloc and the radial dipole.
        """
        cos_theta = np.dot(self.rxyz, self.rzloc) / (np.linalg.norm(self.rxyz, axis=1) * np.linalg.norm(self.rzloc))
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        return theta

    def calc_phi(self, p_tan):
        """
        Return azimuthal angle between x-axis and contact point locations(s)

        Parameters
        ----------
        p_tan : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing
            tangential component of current dipole moment in units of (nA*µm)

        Returns
        -------
        phi : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing azimuthal angle
            in units of (radians) between x-axis vector(s) and projection of
            contact point location vector(s) rxyz into xy-plane.
            z-axis is defined in the direction of rzloc.
            y-axis is defined in the direction of p (orthogonal to rzloc).
            x-axis is defined as cross product between p and rzloc (x).
        """
        proj_rxyz_rz = (np.dot(self.rxyz,
                        self.rzloc) / np.sum(self.rzloc **
                        2)).reshape(len(self.rxyz),1) * self.rzloc.reshape(1,3)
                        # projection of rxyz onto rzloc
        rxy = self.rxyz - proj_rxyz_rz  # projection of rxyz into xy-plane
        x = np.cross(p_tan, self.rzloc)  # vector giving direction of x-axis
        cos_phi = np.dot(rxy, x.T)/np.dot(np.linalg.norm(rxy,
                         axis=1).reshape(len(rxy),1),np.linalg.norm(x,
                         axis=1).reshape(1, len(x)))
        cos_phi = np.nan_to_num(cos_phi)
        phi_temp = np.arccos(cos_phi) # nb: phi_temp is in range [0, pi]
        phi = phi_temp
        range_test = np.dot(rxy, p_tan.T)  # if range_test < 0, phi > pi
        for i in range(len(self.r)):
            for j in range(len(p_tan)):
                if range_test[i,j] < 0:
                    phi[i,j] = 2*np.pi - phi_temp[i,j]
        return phi

    def _sign_rad_dipole(self, p):
        """
        Determine whether radial dipoles are pointing inwards or outwards

        Parameters
        ----------
        p : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing the current dipole moment
             in cartesian coordinates for all n_timesteps in units of (nA*µm)

        Returns
        -------
        sign_vector : ndarray
            Shape (n_timesteps, ) array containing +/-1 for all
            current dipole moments in p.
            If radial part of p[i] points outwards, sign_vector[i] = 1.
            If radial part of p[i] points inwards, sign_vector[i] = -1.

        """
        sign_vector = np.ones(len(p))
        radial_test = np.dot(p, self.rzloc) / (np.linalg.norm(p, axis=1) * self.rz)
        for i in range(len(p)):
            if np.abs(radial_test[i] + 1) < 10 ** -8:
                sign_vector[i] = -1.
        return sign_vector

    def _potential_brain_rad(self, r, theta):
        """
        Return factor for calculation of potential in brain from rad. dipole

        Parameters
        ----------
        r : float
            Distance from origin to brain electrode location in units of (µm)
        theta : float
            Polar angle between brain electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summationfactor for calculation of electrical potential in brain
            from radial current dipole moment. (unitless)
        """
        n = np.arange(1, 100)
        c1n = self._calc_c1n(n)
        consts = n*(c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_csf_rad(self, r, theta):
        """
        Return factor for calculation of potential in CSF from rad. dipole

        Parameters
        ----------
        r : float
            Distance from origin to CSF electrode location in units of (µm)
        theta : float
            Polar angle between CSF electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in CSF
            from radial current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c2n = self._calc_c2n(n)
        d2n = self._calc_d2n(n, c2n)
        consts = n*(c2n * (r / self.r2) ** n + d2n * (self.r2 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_skull_rad(self, r, theta):
        """
        Return factor for calculation of potential in skull from rad. dipole

        Parameters
        ----------
        r : float
            Distance from origin to skull electrode location in units of (µm)
        theta : float
            Polar angle between skull electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in skull
            from radial current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c3n = self._calc_c3n(n)
        d3n = self._calc_d3n(n, c3n)
        consts = n*(c3n * (r / self.r3) ** n + d3n * (self.r3 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_scalp_rad(self, r, theta):
        """
        Return factor for calculation of potential in scalp from radial dipole

        Parameters
        ----------
        r : float
            Distance from origin to scalp electrode location in units of (µm)
        theta : float
            Polar angle between scalp electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in scalp
            from radial current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c4n = self._calc_c4n(n)
        d4n = self._calc_d4n(n, c4n)
        consts = n*(c4n * (r / self.r4) ** n + d4n * (self.r4 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_brain_tan(self, r, theta):
        """
        Return factor for calculation of potential in brain from tan. dipole

        Parameters
        ----------
        r : float
            Distance from origin to brain electrode location in units of (µm)
        theta : float
            Polar angle between brain electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in brain
            from tangential current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c1n = self._calc_c1n(n)
        consts = (c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def _potential_csf_tan(self, r, theta):
        """
        Return factor for calculation of potential in CSF from tan. dipole

        Parameters
        ----------
        r : float
            Distance from origin to CSF electrode location in units of (µm)
        theta : float
            Polar angle between CSF electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in CSF
            from tangential current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c2n = self._calc_c2n(n)
        d2n = self._calc_d2n(n, c2n)
        consts = c2n*(r/self.r2)**n + d2n*(self.r2/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def _potential_skull_tan(self, r, theta):
        """
        Return factor for calculation of potential in skull from tan. dipole

        Parameters
        ----------
        r : float
            Distance from origin to skull electrode location in units of (µm)
        theta : float
            Polar angle between skull electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in skull
            from tangential current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c3n = self._calc_c3n(n)
        d3n = self._calc_d3n(n, c3n)
        consts = c3n*(r/self.r3)**n + d3n*(self.r3/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def _potential_scalp_tan(self, r, theta):
        """
        Return factor for calculation of potential in scalp from tan. dipole

        Parameters
        ----------
        r : float
            Distance from origin to scalp electrode location in units of (µm)
        theta : float
            Polar angle between scalp electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum : float
            Summation factor for calculation of electrical potential in scalp
            from tangential current dipole moment. (unitless)
        """
        n = np.arange(1,100)
        c4n = self._calc_c4n(n)
        d4n = self._calc_d4n(n, c4n)
        consts = c4n*(r/self.r4)**n + d4n*(self.r4/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def _calc_vn(self, n):
        r_const = (self.r34 ** n - self.r43 ** (n + 1)) / ((n + 1) / n * self.r34 ** n + self.r43 ** (n + 1))
        v = (n / (n + 1) * self.sigma34 - r_const) / (self.sigma34 + r_const)
        return v

    def _calc_yn(self, n):
        vn = self._calc_vn(n)
        r_const = (n / (n + 1) * self.r23 ** n - vn * self.r32 ** (n + 1)) / (self.r23 ** n + vn * self.r32 ** (n + 1))
        y = (n / (n + 1) * self.sigma23 - r_const) / (self.sigma23 + r_const)
        return y

    def _calc_zn(self, n):
        yn = self._calc_yn(n)
        z = (self.r12 ** n - (n + 1) / n * yn * self.r21 ** (n + 1)) / (self.r12 ** n + yn * self.r21 ** (n + 1))
        return z

    def _calc_c1n(self, n):
        zn = self._calc_zn(n)
        c = ((n + 1) / n * self.sigma12 + zn) / (self.sigma12 - zn) * self.rz1**(n+1)
        return c

    def _calc_c2n(self, n):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        c2 = (c1 + self.rz1**(n+1)) / (self.r12 ** n + yn * self.r21 ** (n + 1))
        return c2

    def _calc_d2n(self, n, c2):
        yn = self._calc_yn(n)
        d2 = yn * c2
        return d2

    def _calc_c3n(self, n):
        vn = self._calc_vn(n)
        c2 = self._calc_c2n(n)
        d2 = self._calc_d2n(n, c2)
        c3 = (c2 + d2) / (self.r23 ** n + vn * self.r32 ** (n + 1))
        return c3

    def _calc_d3n(self, n, c3):
        vn = self._calc_vn(n)
        d3 = vn * c3
        return d3

    def _calc_c4n(self, n):
        c3 = self._calc_c3n(n)
        d3 = self._calc_d3n(n, c3)
        c4 = (n + 1) / n * (c3 + d3) / ((n + 1) / n * self.r34 ** n + self.r43 ** (n + 1))
        return c4

    def _calc_d4n(self, n, c4):
        d4 = n / (n + 1) * c4
        return d4


class InfiniteVolumeConductor(object):
    """
    Main class for computing extracellular potentials with current dipole
    approximation in an infinite 3D volume conductor model that assumes
    homogeneous, isotropic, linear (frequency independent) conductivity

    Parameters
    ----------
    sigma : float
        Electrical conductivity in extracellular space in units of (S/cm)

    Examples
    --------
    Computing the potential from dipole moment valid in the far field limit.
    Theta correspond to the dipole alignment angle from the vertical z-axis.
    >>> import LFPy
    >>> import numpy as np
    >>> inf_model = LFPy.InfiniteVolumeConductor(sigma=0.3)
    >>> p = np.array([[10., 10., 10.]])
    >>> r = np.array([[1000., 0., 5000.]])
    >>> phi_p = inf_model.get_dipole_potential(p, r)

    """

    def __init__(self, sigma=0.3):
        "Initialize class InfiniteVolumeConductor"
        self.sigma = sigma

    def get_dipole_potential(self, p, r):
        """
        Return electric potential from current dipole with current dipole approx

        p : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing the x,y,z components of the
            current dipole moment in units of (nA*µm) for all timesteps

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) FourSphereVolumeConductor.r in units
            of (mV) for all timesteps of current dipole moment p

        """
        dotprod = np.dot(r, p.T)
        r_factor = np.linalg.norm(r, axis=1)**3
        phi = 1./(4*np.pi*self.sigma)*(dotprod.T/ r_factor).T
        return phi


def get_current_dipole_moment(dist, current):
    """
    Return current dipole moment vector P and P_tot of cell.

    Parameters
    ----------
    current : ndarray, dtype=float
        Either an array containing all transmembrane currents
        from all compartments of the cell, or an array of all
        axial currents between compartments in cell in units of nA
    dist : ndarray, dtype=float
        When input current is an array of axial currents,
        the dist is the length of each axial current.
        When current is the an array of transmembrane
        currents, dist is the position vector of each
        compartment middle. Unit is (µm).

    Returns
    -------
    P : ndarray, dtype=float
        Array containing the current dipole moment for all
        timesteps in the x-, y- and z-direction in units of (nA*µm)
    P_tot : ndarray, dtype=float
        Array containing the magnitude of the
        current dipole moment vector for all timesteps in units of (nA*µm)

    Examples
    --------
    Get current dipole moment vector and scalar moment from axial currents
    computed from membrane potentials
    >>> import LFPy
    >>> import numpy as np
    >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=False)
    >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,1000),
    >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
    >>> syn.set_spike_times(np.mgrid[20:100:20])
    >>> cell.simulate(rec_vmem=True, rec_imem=False)
    >>> d_list, i_axial = cell.get_axial_currents()
    >>> P_ax, P_ax_tot = LFPy.get_current_dipole_moment(d_list, i_axial)

    Get current dipole moment vector and scalar moment from transmembrane
    currents using the extracellular mechanism in NEURON
    >>> import LFPy
    >>> import numpy as np
    >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=True)
    >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,1000),
    >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
    >>> syn.set_spike_times(np.mgrid[20:100:20])
    >>> cell.simulate(rec_vmem=False, rec_imem=True)
    >>> P_imem, P_imem_tot = LFPy.get_current_dipole_moment(np.c_[cell.xmid,
    >>>                                                          cell.ymid,
    >>>                                                          cell.zmid],
    >>>                                                    cell.imem)

    """
    P = np.dot(current.T, dist)
    P_tot = np.sqrt(np.sum(P**2, axis=1))
    return P, P_tot


class MEG(object):
    """
    Basic class for computing magnetic field from current dipole moment.
    For this purpose we use the Biot-Savart law derived from Maxwell's equations
    under the assumption of negligible magnetic induction effects (Nunez and
    Srinivasan, Oxford University Press, 2006):

    .. math:: \mathbf{H} = \\frac{\\mathbf{p} \\times \\mathbf{R}}{4 \pi R^3}

    where :math:`\mathbf{p}` is the current dipole moment, :math:`\mathbf{R}`
    the vector between dipole source location and measurement location, and
    :math:`R=|\mathbf{R}|`

    Note that the magnetic field :math:`\mathbf{H}` is related to the magnetic
    field :math:`\mathbf{B}` as :math:`\mu_0 \mathbf{H} = \mathbf{B}-\mathbf{M}`
    where :math:`\mu_0` is the permeability of free space (very close to
    permebility of biological tissues). :math:`\mathbf{M}` denotes material
    magnetization (also ignored)


    Parameters
    ----------
    sensor_locations : ndarray, dtype=float
        shape (n_locations x 3) array with x,y,z-locations of measurement
        devices where magnetic field of current dipole moments is calculated


    Examples
    --------
    Define cell object, create synapse, compute current dipole moment

    >>> import LFPy, os, numpy as np, matplotlib.pyplot as plt
    >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
    >>>                  passive=True)
    >>> cell.set_pos(0., 0., 0.)
    >>> syn = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=0.01, record_current=True)
    >>> syn.set_spike_times_w_netstim()
    >>> cell.simulate(rec_isyn=syn.record_current, rec_current_dipole_moment=True)

    Compute the dipole location as an average of segment locations weighted by membrane area

    >>> dipole_location = (cell.area * np.c_[cell.xmid, cell.ymid, cell.zmid].T / cell.area.sum()).sum(axis=1)

    Instantiate the LFPy.MEG object, compute and plot the magnetic signal in a sensor location

    >>> sensor_locations = np.array([[1E4, 0, 0]])
    >>> meg = LFPy.MEG(sensor_locations)
    >>> H = meg.calculate_H(cell.current_dipole_moment, dipole_location)
    >>> plt.subplot(311)
    >>> plt.plot(cell.tvec, cell.somav)
    >>> plt.subplot(312)
    >>> plt.plot(cell.tvec, syn.i)
    >>> plt.subplot(313)
    >>> plt.plot(cell.tvec, H[0])

    Raises
    ------
    AssertionError
        If dimensionality of sensor_locations is wrong


    """
    def __init__(self, sensor_locations):
        """
        Initialize class MEG
        """
        try:
            assert(sensor_locations.ndim == 2)
        except AssertionError:
            raise AssertionError('sensor_locations.ndim != 2')
        try:
            assert(sensor_locations.shape[1] == 3)
        except AssertionError:
            raise AssertionError('sensor_locations.shape[1] != 3')

        # set attributes
        self.sensor_locations = sensor_locations


    def calculate_H(self, current_dipole_moment, dipole_location):
        """
        Parameters
        ----------
        current_dipole_moment : ndarray, dtype=float
            shape (n_timesteps x 3) array with x,y,z-components of current-
            dipole moment time series data in units of (nA µm)
        dipole_location : ndarray, dtype=float
            shape (3, ) array with x,y,z-location of dipole in units of (µm)

        Returns
        -------
        ndarray, dtype=float
            shape (n_locations x n_timesteps x 3) array with x,y,z-components of the magnetic
            field :math:`\mathbf{H}` in units of (nA/µm)

        Raises
        ------
        AssertionError
            If dimensionality of current_dipole_moment and/or dipole_location is wrong
        """
        try:
            assert(current_dipole_moment.ndim == 2)
        except AssertionError:
            raise AssertionError('current_dipole_moment.ndim != 2')
        try:
            assert(current_dipole_moment.shape[1] == 3)
        except AssertionError:
            raise AssertionError('current_dipole_moment.shape[1] != 3')
        try:
            assert(dipole_location.shape == (3, ))
        except AssertionError:
            raise AssertionError('dipole_location.shape != (3, )')


        # container
        H = np.empty((self.sensor_locations.shape[0],
                      current_dipole_moment.shape[0],
                      3))
        # iterate over sensor locations
        for i, r in enumerate(self.sensor_locations):
            R = r - dipole_location
            assert(R.ndim==1 and R.size == 3)
            try:
                assert(np.allclose(R, np.zeros(3)) == False)
            except AssertionError:
                raise AssertionError('Identical dipole and sensor location.')
            H[i, ] = np.cross(current_dipole_moment,
                              R) / (4 * np.pi * np.sqrt((R**2).sum())**3)

        return H
