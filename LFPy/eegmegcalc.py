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


from scipy.special import eval_legendre, lpmv, legendre
import numpy as np
from warnings import warn


class OneSphereVolumeConductor(object):
    """
    Computes extracellular potentials within and outside a spherical volume-
    conductor model that assumes homogeneous, isotropic, linear (frequency
    independent) conductivity in and outside the sphere with a radius R. The
    conductivity in and outside the sphere must be greater than 0, and the
    current source(s) must be located within the radius R.

    The implementation is based on the description of electric potentials of
    point charge in an dielectric sphere embedded in dielectric media, which is
    mathematically equivalent to a current source in conductive media, as
    published by Deng (2008), Journal of Electrostatics 66:549-560

    Parameters
    ----------
    r : ndarray, dtype=float
        shape(3, n_points) observation points in space in spherical coordinates
        (radius, theta, phi) relative to the center of the sphere.
    R : float
        sphere radius (µm)
    sigma_i : float
        electric conductivity for radius r <= R (S/m)
    sigma_o : float
        electric conductivity for radius r > R (S/m)

    Examples
    --------
    Compute the potential for a single monopole along the x-axis:

    >>> # import modules
    >>> from LFPy import OneSphereVolumeConductor
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # observation points in spherical coordinates (flattened)
    >>> X, Y = np.mgrid[-15000:15100:1000., -15000:15100:1000.]
    >>> r = np.array([np.sqrt(X**2 + Y**2).flatten(),
    >>>               np.arctan2(Y, X).flatten(),
    >>>               np.zeros(X.size)])
    >>> # set up class object and compute electric potential in all locations
    >>> sphere = OneSphereVolumeConductor(r, R=10000.,
    >>>                                   sigma_i=0.3, sigma_o=0.03)
    >>> Phi = sphere.calc_potential(rs=8000, I=1.).reshape(X.shape)
    >>> # plot
    >>> fig, ax = plt.subplots(1,1)
    >>> im=ax.contourf(X, Y, Phi,
    >>>              levels=np.linspace(Phi.min(), np.median(Phi[np.isfinite(Phi)])*4, 30))
    >>> circle = plt.Circle(xy=(0,0), radius=sphere.R, fc='none', ec='k')
    >>> ax.add_patch(circle)
    >>> fig.colorbar(im, ax=ax)
    >>> plt.show()
    """
    def __init__(self,
                 r,
                 R=10000.,
                 sigma_i=0.3,
                 sigma_o=0.03):
        """initialize class OneSphereVolumeConductor"""
        # check inputs
        try:
            assert(r.shape[0] == 3)
            assert(r.ndim == 2)
        except AssertionError as ae:
            raise AssertionError('r must be a shape (3, n_points) ndarray')
        try:
            assert((type(R) is float) or (type(R) is int))
        except AssertionError as ae:
            raise AssertionError('sphere radius R must be a float value')
        try:
            assert((sigma_i > 0) & (sigma_o > 0))
        except AssertionError as ae:
            raise AssertionError('sigma_i and sigma_o must both be positive values')

        self.r = r
        self.R = R
        self.sigma_i = sigma_i
        self.sigma_o = sigma_o


    def calc_potential(self, rs, I, min_distance=1., n_max=1000):
        """
        Return the electric potential at observation points for source current
        with magnitude I as function of time.

        Parameters
        ----------
        rs : float
            monopole source location along the horizontal x-axis (µm)
        I : float or ndarray, dtype float
            float or shape (n_tsteps, ) array containing source current (nA)
        min_distance : None or float
            minimum distance between source location and observation point (µm)
            (in order to avoid singular values)
        n_max : int
            Number of elements in polynomial expansion to sum over
            (see Deng 2008).

        Returns
        -------
        Phi : ndarray
            shape (n-points, ) ndarray of floats if I is float like. If I is
            an 1D ndarray, and shape (n-points, I.size) ndarray is returned.
            Unit (mV).
        """
        try:
            assert(type(rs) in [int, float, np.float64])
            assert(abs(rs) < self.R)
        except AssertionError as ae:
            raise AssertionError('source location rs must be a float value and |rs| must be less than sphere radius R')
        try:
            assert((min_distance is None) or (type(min_distance) in [float, int, np.float64]))
        except AssertionError:
            raise AssertionError('min_distance must be None or a float')

        r = self.r[0]
        theta = self.r[1]


        # add harmonical contributions due to inhomogeneous media
        inds_i = r <= self.R
        inds_o = r > self.R

        # observation points r <= R
        phi_i = np.zeros(r.size)
        for j, (theta_i, r_i) in enumerate(zip(theta[inds_i], r[inds_i])):
            coeffs_i = np.zeros(n_max)
            for n in range(n_max):
                coeffs_i[n] = ((self.sigma_i - self.sigma_o)*(n+1))  / (self.sigma_i*n + self.sigma_o*(n+1)) * ((r_i*rs)/self.R**2)**n
            poly_i = np.polynomial.legendre.Legendre(coeffs_i)
            phi_i[np.where(inds_i)[0][j]] = poly_i(np.cos(theta_i))
        phi_i[inds_i] *= 1./self.R

        # observation points r > R
        phi_o = np.zeros(r.size)
        for j, (theta_o, r_o) in enumerate(zip(theta[inds_o], r[inds_o])):
            coeffs_o = np.zeros(n_max)
            for n in range(n_max):
                coeffs_o[n] = (self.sigma_i*(2*n+1) ) / (self.sigma_i*n + self.sigma_o*(n+1)) * (rs / r_o)**n
            poly_o = np.polynomial.legendre.Legendre(coeffs_o)
            phi_o[np.where(inds_o)[0][j]] = poly_o(np.cos(theta_o))
        phi_o[inds_o] *= 1./r[inds_o]

        # potential in homogeneous media
        if min_distance is None:
            phi_i[inds_i] += 1. / np.sqrt(r[r <= self.R]**2 + rs**2 - 2*r[inds_i]*rs*np.cos(theta[inds_i]))
        else:
            denom = np.sqrt(r[inds_i]**2 + rs**2 - 2*r[inds_i]*rs*np.cos(theta[inds_i]))
            denom[denom < min_distance] = min_distance
            phi_i[inds_i] += 1./denom



        if type(I) is np.ndarray:
            try:
                assert(np.all(np.isfinite(I)))
                assert(np.all(np.isreal(I)))
                assert(I.ndim == 1)
            except AssertionError:
                raise AssertionError('input argument I must be float or 1D ndarray with float values')

            return np.dot((phi_i + phi_o).reshape((1, -1)).T,
                I.reshape((1, -1))) / (4.*np.pi*self.sigma_i)
        else:
            try:
                assert(np.isfinite(I)) and (np.shape(I) == ())
            except AssertionError:
                raise AssertionError('input argument I must be float or 1D ndarray with float values')
            return I / (4.*np.pi*self.sigma_i)*(phi_i + phi_o)


    def calc_mapping(self, cell, n_max=1000):
        """
        Compute linear mapping between transmembrane currents of LFPy.Cell like
        object instantiation and extracellular potential in and outside of
        sphere. Cell position must be set in space, using the method
        ``Cell.set_pos(**kwargs)``.

        Parameters
        ----------
        cell : LFPy.Cell like instance
            Instantiation of class LFPy.Cell, TemplateCell or NetworkCell.
        n_max : int
            Number of elements in polynomial expansion to sum over
            (see Deng 2008).

        Examples
        --------
        Compute extracellular potential in one-sphere volume conductor model
        from LFPy.Cell object:

        >>> # import modules
        >>> import LFPy
        >>> import os
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.collections import PolyCollection
        >>> # create cell
        >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
        >>>                  tstop=10.)
        >>> cell.set_pos(z=9800.)
        >>> # stimulus
        >>> syn = LFPy.Synapse(cell, idx=cell.totnsegs-1, syntype='Exp2Syn', weight=0.01)
        >>> syn.set_spike_times(np.array([1.]))
        >>> # simulate
        >>> cell.simulate(rec_imem=True)
        >>> # observation points in spherical coordinates (flattened)
        >>> X, Z = np.mgrid[-500:501:10., 9500:10501:10.]
        >>> Y = np.zeros(X.shape)
        >>> r = np.array([np.sqrt(X**2 + Z**2).flatten(),
        >>>               np.arccos(Z / np.sqrt(X**2 + Z**2)).flatten(),
        >>>               np.arctan2(Y, X).flatten()])
        >>> # set up class object and compute mapping between segment currents
        >>> # and electric potential in space
        >>> sphere = LFPy.OneSphereVolumeConductor(r=r, R=10000.,
        >>>                                        sigma_i=0.3, sigma_o=0.03)
        >>> mapping = sphere.calc_mapping(cell, n_max=1000)
        >>> # pick out some time index for the potential and compute potential
        >>> ind = cell.tvec==2.
        >>> Phi = np.dot(mapping, cell.imem)[:, ind].reshape(X.shape)
        >>> # plot potential
        >>> fig, ax = plt.subplots(1,1)
        >>> zips = []
        >>> for x, z in cell.get_idx_polygons(projection=('x', 'z')):
        >>>     zips.append(zip(x, z))
        >>> polycol = PolyCollection(zips,
        >>>                          edgecolors='none',
        >>>                          facecolors='gray')
        >>> vrange = 1E-3 # limits for color contour plot
        >>> im=ax.contour(X, Z, Phi,
        >>>              levels=np.linspace(-vrange, vrange, 41))
        >>> circle = plt.Circle(xy=(0,0), radius=sphere.R, fc='none', ec='k')
        >>> ax.add_collection(polycol)
        >>> ax.add_patch(circle)
        >>> ax.axis(ax.axis('equal'))
        >>> ax.set_xlim(X.min(), X.max())
        >>> ax.set_ylim(Z.min(), Z.max())
        >>> fig.colorbar(im, ax=ax)
        >>> plt.show()

        Returns
        -------
        ndarray
            Shape (n_points, n_compartments) mapping between individual
            segments and extracellular potential in extracellular locations

        Notes
        -----
        Each segment is treated as a point source in space. The minimum
        source to measurement site distance will be set to the diameter of
        each segment

        """
        # midpoint position of compartments in spherical coordinates
        radius = np.sqrt(cell.xmid**2 + cell.ymid**2 + cell.zmid**2)
        theta = np.arccos(cell.zmid/radius)
        phi = np.arctan2(cell.ymid, cell.xmid)
        diam = cell.diam

        # since the sources must be located on the x-axis, we keep a copy
        # of the unrotated coordinate system for the contact points:
        r_orig = np.copy(self.r)

        # unit current amplitude
        I = 1.

        # initialize mapping array
        mapping = np.zeros((self.r.shape[1], radius.size))

        # compute the mapping for each compartment
        for i, (radius_i, theta_i, _, diam_i) in enumerate(zip(radius, theta, phi, diam)):
            self.r = np.array([r_orig[0], # radius unchanged
                               r_orig[1] - theta_i, # rotate relative to source location
                               r_orig[2]]) # phi unchanged (don't enter equations)
            mapping[:, i] = self.calc_potential(radius_i, I=I, min_distance=diam_i, n_max=n_max)

        # reset measurement locations
        self.r = r_orig

        # return mapping between segment currents and contrib in each
        # measurement location
        return mapping


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
    r_electrodes : ndarray, dtype=float
        Shape (n_contacts, 3) array containing n_contacts electrode locations
        in cartesian coordinates in units of (µm).
        All ``r_el`` in ``r_electrodes`` must be less than or equal to scalp
        radius and larger than the distance between dipole and sphere
        center: ``|rz| < r_el <= radii[3]``.

    Examples
    --------
    Compute extracellular potential from current dipole moment in four-sphere
    head model:

    >>> import LFPy
    >>> import numpy as np
    >>> radii = [79000., 80000., 85000., 90000.]
    >>> sigmas = [0.3, 1.5, 0.015, 0.3]
    >>> r = np.array([[0., 0., 90000.], [0., 85000., 0.]])
    >>> rz = np.array([0., 0., 78000.])
    >>> sphere_model = LFPy.FourSphereVolumeConductor(radii, sigmas, r)
    >>> # current dipole moment
    >>> p = np.array([[10., 10., 10.]]*10) # 10 timesteps
    >>> # compute potential
    >>> potential = sphere_model.calc_potential(p, rz)

    """

    def __init__(self, radii, sigmas, r_electrodes, iter_factor = 2./99.*1e-6):
        """Initialize class FourSphereVolumeConductor"""
        self.r1 = float(radii[0])
        self.r2 = float(radii[1])
        self.r3 = float(radii[2])
        self.r4 = float(radii[3])

        self.sigma1 = float(sigmas[0])
        self.sigma2 = float(sigmas[1])
        self.sigma3 = float(sigmas[2])
        self.sigma4 = float(sigmas[3])

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

        self.rxyz = r_electrodes
        self.r = np.sqrt(np.sum(r_electrodes ** 2, axis=1))

        self.iteration_stop_factor = iter_factor
        self._check_params()

    def _check_params(self):
        """Check that radii, sigmas and r contain reasonable values"""
        if (self.r1 < 0) or (self.r1 > self.r2) or (self.r2 > self.r3) or (self.r3 > self.r4):
            raise RuntimeError('Radii of brain (radii[0]), CSF (radii[1]), '
                               'skull (radii[2]) and scalp (radii[3]) '
                               'must be positive ints or floats such that '
                               '0 < radii[0] < radii[1] < radii[2] < radii[3].')

        if (self.sigma1 < 0) or (self.sigma2 < 0) or (self.sigma3 < 0) or (self.sigma4 < 0):
            raise RuntimeError('Conductivities (sigmas), must contain positive'
                               ' ints or floats.')

        if any(r > self.r4 for r in self.r):
            raise ValueError('Electrode located outside head model.'
                               'r > radii[3]. r = %s, r4 = %s',
                               self.r, self.r4)

    def _rz_params(self, rz):
        """Define dipole location vector, and check that dipole is located in
        the brain, closer to the center than any measurement location."""
        self._rzloc = rz
        self._rz = np.sqrt(np.sum(rz ** 2))
        self._z = self._rzloc/self._rz
        if self._rz == 0:
            raise RuntimeError('Placing dipole in center of head model causes division by zero.')

        self._rz1 = self._rz / self.r1

        if self._rz1 >= 1:
            raise RuntimeError('Dipole should be placed inside brain, i.e. |rz| < |r1|')

        elif self._rz1 > 0.99999:
            warn('Dipole should be placed minimum ~1µm away from brain surface, '
                  'to avoid extremely slow convergence.')

        elif self._rz1 > 0.9999:
            warn('Computation time might be long due to slow convergence. '
                 'Can be avoided by placing dipole further away from brain surface.')

        if any(r < self._rz for r in self.r):
            raise RuntimeError('Electrode must be farther away from '
                               'brain center than dipole: r > rz.'
                               'r = %s, rz = %s', self.r, self._rz)

        # compute theta angle between rzloc and rxyz
        self._theta = self.calc_theta()


    def calc_potential(self, p, rz):
        """
        Return electric potential from current dipole moment p.

        Parameters
        ----------
        p : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing the x,y,z components of the
            current dipole moment in units of (nA*µm) for all timesteps.
        rz : ndarray, dtype=float
            Shape (3, ) array containing the position of the current dipole in
            cartesian coordinates. Units of (µm).

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) FourSphereVolumeConductor.r in units
            of (mV) for all timesteps of current dipole moment p.

        """

        self._rz_params(rz)
        n_contacts = self.r.shape[0]
        n_timesteps = p.shape[0]

        if np.linalg.norm(p) != 0:
            p_rad, p_tan = self._decompose_dipole(p)
        else:
            p_rad = np.zeros((n_timesteps, 3))
            p_tan = np.zeros((n_timesteps, 3))
        if np.linalg.norm(p_rad) != 0.:
            pot_rad = self._calc_rad_potential(p_rad)
        else:
            pot_rad = np.zeros((n_contacts, n_timesteps))

        if np.linalg.norm(p_tan) != 0.:
            pot_tan = self._calc_tan_potential(p_tan)
        else:
            pot_tan = np.zeros((n_contacts, n_timesteps))

        pot_tot = pot_rad + pot_tan
        return pot_tot

    def calc_potential_from_multi_dipoles(self, cell, timepoints=None):
        """
        Return electric potential from multiple current dipoles from cell.

        By multiple current dipoles we mean the dipoles computed from all
        axial currents in a neuron simulation, typically two
        axial currents per compartment, except for the root compartment.

        Parameters
        ----------
        cell : LFPy Cell object, LFPy.Cell
        timepoints : ndarray, dtype=int
            array of timepoints at which you want to compute
            the electric potential. Defaults to None. If not given,
            all simulation timesteps will be included.

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) electrode_locs in units
            of (mV) for all timesteps of neuron simulation.

        Examples
        --------
        Compute extracellular potential from neuron simulation in
        four-sphere head model. Instead of simplifying the neural activity to
        a single dipole, we compute the contribution from every multi dipole
        from all axial currents in neuron simulation:

        >>> import LFPy
        >>> import numpy as np
        >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=False)
        >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,100),
        >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
        >>> syn.set_spike_times(np.mgrid[20:100:20])
        >>> cell.simulate(rec_vmem=True, rec_imem=False)
        >>> radii = [200., 300., 400., 500.]
        >>> sigmas = [0.3, 1.5, 0.015, 0.3]
        >>> electrode_locs = np.array([[50., -50., 250.]])
        >>> timepoints = np.array([0,100])
        >>> MD_4s = LFPy.FourSphereVolumeConductor(radii,
        >>>                                        sigmas,
        >>>                                        electrode_locs)
        >>> phi = MD_4s.calc_potential_from_multi_dipoles(cell,
        >>>                                               timepoints)
        """
        multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(timepoints)
        N_elec = self.rxyz.shape[0]
        Ni, Nt, Nd = multi_p.shape
        potential = np.zeros((N_elec, Nt))
        for i in range(Ni):
            pot = self.calc_potential(multi_p[i], multi_p_locs[i])
            potential += pot
        return potential

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
            Shape (n_timesteps, 3) array, radial part of p, parallel to self._rz
        p_tan : ndarray, dtype=float
            Shape (n_timesteps, 3) array, tangential part of p,
            orthogonal to self._rz
        """
        z_ = np.expand_dims(self._z, -1) # reshape z-axis vector
        p_rad = np.dot(np.dot(p, z_), z_.T)
        p_tan = p - p_rad
        return p_rad, p_tan

    def _calc_rad_potential(self, p_rad):
        """
        Return potential from radial dipole p_rad at location rz measured at r

        Parameters
        ----------
        p_rad : ndarray, dtype=float
            Shape (n_timesteps, 3) array, radial part of p
            in units of (nA*µm), parallel to self._rz

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the extracecllular
            potential at n_contacts contact point(s) FourSphereVolumeConductor.r
            in units of (mV) for all timesteps of p_rad
        """

        p_tot = np.linalg.norm(p_rad, axis=1)
        s_vector = self._sign_rad_dipole(p_rad)
        phi_const = s_vector * p_tot / (4 * np.pi * self.sigma1 * self._rz ** 2)
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            r_point = self.r[el_point]
            theta_point = self._theta[el_point]
            if r_point <= self.r1:
                n_terms[el_point] = self._potential_brain_rad(r_point,
                                                             theta_point)
            elif r_point <= self.r2:
                n_terms[el_point] = self._potential_csf_rad(r_point,
                                                           theta_point)
            elif r_point <= self.r3:
                n_terms[el_point] = self._potential_skull_rad(r_point,
                                                             theta_point)
            else:
                n_terms[el_point] = self._potential_scalp_rad(r_point,
                                                             theta_point)
        potential = phi_const * n_terms
        return potential

    def _calc_tan_potential(self, p_tan):
        """
        Return potential from tangential dipole P at location rz measured at r

        Parameters
        ----------
        p_tan : ndarray, dtype=float
            Shape (n_timesteps, 3) array, tangential part of p
            in units of (nA*µm), orthogonal to self._rz

        Returns
        _______
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the extracecllular
            potential at n_contacts contact point(s) FourSphereVolumeConductor.r
            in units of (mV) for all timesteps of p_tan
        """
        phi = self.calc_phi(p_tan)
        p_tot = np.linalg.norm(p_tan, axis=1)
        phi_hom = - p_tot / (4 * np.pi * self.sigma1 * self._rz ** 2) * np.sin(phi)
        n_terms = np.zeros((len(self.r),1))
        for el_point in range(len(self.r)):
            r_point = self.r[el_point]
            theta_point = self._theta[el_point]
            # if r_electrode is orthogonal to p_tan, i.e. theta = 0 or theta = pi,
            # there is no contribution to electric potential from p_tan
            if (theta_point == 0.) or (theta_point == np.pi):
                n_terms[el_point] = 0
            elif r_point <= self.r1:
                n_terms[el_point] = self._potential_brain_tan(r_point, theta_point)
            elif r_point <= self.r2:
                n_terms[el_point] = self._potential_csf_tan(r_point, theta_point)
            elif r_point <= self.r3:
                n_terms[el_point] = self._potential_skull_tan(r_point, theta_point)
            else:
                n_terms[el_point] = self._potential_scalp_tan(r_point, theta_point)
        potential = n_terms * phi_hom

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
        cos_theta = np.dot(self.rxyz, self._rzloc) / (np.linalg.norm(self.rxyz, axis=1) * np.linalg.norm(self._rzloc))
        theta = np.arccos(cos_theta)
        return theta

    def calc_phi(self, p_tan):
        """
        Return azimuthal angle between x-axis and contact point locations(s)

        Parameters
        ----------
        p_tan : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing
            tangential component of current dipole moment in units of (nA*µm)

        Returns
        -------
        phi : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing azimuthal angle
            in units of (radians) between x-axis vector(s) and projection of
            contact point location vector(s) rxyz into xy-plane.
            z-axis is defined in the direction of rzloc.
            y-axis is defined in the direction of p_tan (orthogonal to rzloc).
            x-axis is defined as cross product between p_tan and rzloc (x).
        """

        # project rxyz onto z-axis (rzloc)
        proj_rxyz_rz = self.rxyz*self._z
        # find projection of rxyz in xy-plane
        rxy = self.rxyz - proj_rxyz_rz
        # define x-axis
        x = np.cross(p_tan, self._z)

        phi = np.zeros((len(self.rxyz), len(p_tan)))
        # create masks to avoid computing phi when phi is not defined
        mask = np.ones(phi.shape, dtype=bool)
        # phi is not defined when theta= 0,pi or |p_tan| = 0
        mask[(self._theta == 0) | (self._theta == np.pi)] = np.zeros(len(p_tan))
        mask[:,np.abs(np.linalg.norm(p_tan, axis=1)) == 0] = 0

        cos_phi = np.zeros(phi.shape)
        # compute cos_phi using mask to avoid zerodivision
        cos_phi[mask] = np.dot(rxy, x.T)[mask] / np.outer(np.linalg.norm(rxy,
                  axis=1), np.linalg.norm(x, axis=1))[mask]

        # compute phi in [0, pi]
        phi[mask] = np.arccos(cos_phi[mask])

        # nb: phi in [-pi, pi]. since p_tan defines direction of y-axis,
        # phi < 0 when rxy*p_tan < 0
        phi[np.dot(rxy, p_tan.T) < 0] *= -1

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
        sign_vector = np.sign(np.dot(p, self._rzloc))
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        # while const > self.iteration_stop_factor*coeff_sum:
        while const > 2./99.*1e-12*coeff_sum:
            c1n = self._calc_c1n(n)
            const = n*(c1n * (r / self.r1) ** n + (self._rz / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        consts = np.insert(consts, 0, 0) # since the legendre function starts with P0
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        # while const > self.iteration_stop_factor*coeff_sum:
        while const > 2./99.*1e-6*coeff_sum:
            term1 = self._calc_csf_term1(n,r)
            term2 = self._calc_csf_term2(n,r)
            const = n*(term1 + term2)
            coeff_sum += const
            consts.append(const)
            n += 1
        consts = np.insert(consts, 0, 0) # since the legendre function starts with P0
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        # while const > self.iteration_stop_factor*coeff_sum:
        while const > 2./99.*1e-6*coeff_sum:
            c3n = self._calc_c3n(n)
            d3n = self._calc_d3n(n, c3n)
            const = n*(c3n * (r / self.r3) ** n + d3n * (self.r3 / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        consts = np.insert(consts, 0, 0) # since the legendre function starts with P0
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        # while const > self.iteration_stop_factor*coeff_sum:
        while const > 2./99.*1e-6*coeff_sum:
            c4n = self._calc_c4n(n)
            d4n = self._calc_d4n(n, c4n)
            const = n*(c4n * (r / self.r4) ** n + d4n * (self.r4 / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        consts = np.insert(consts, 0, 0) # since the legendre function starts with P0
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor*coeff_sum:
            c1n = self._calc_c1n(n)
            const = (c1n * (r / self.r1) ** n + (self._rz / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,np.arange(1,n))])
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor*coeff_sum:
            term1 = self._calc_csf_term1(n,r)
            term2 = self._calc_csf_term2(n,r)
            const = term1 + term2
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,np.arange(1,n))])
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor*coeff_sum:
            c3n = self._calc_c3n(n)
            d3n = self._calc_d3n(n, c3n)
            const = c3n * (r / self.r3) ** n + d3n * (self.r3 / r) ** (n + 1)
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,np.arange(1,n))])
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
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor*coeff_sum:
            c4n = self._calc_c4n(n)
            d4n = self._calc_d4n(n, c4n)
            const = c4n * (r / self.r4) ** n + d4n * (self.r4 / r) ** (n + 1)
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,np.arange(1,n))])
        return pot_sum

    def _calc_vn(self, n):
        r_const = ((self.r34 ** (2*n + 1) - 1) /
                  ((n + 1) / n * self.r34 ** (2*n + 1) + 1))
        if self.sigma23 + r_const == 0.0:
            v = 1e12
        else:
            v = (n / (n + 1) * self.sigma34 - r_const) / (self.sigma34 + r_const)
        return v

    def _calc_yn(self, n):
        vn = self._calc_vn(n)
        r_const = ((n / (n + 1) * self.r23 ** (2*n + 1) - vn) /
                  (self.r23 ** (2*n + 1) + vn))
        if self.sigma23 + r_const == 0.0:
            y = 1e12
        else:
            y = (n / (n + 1) * self.sigma23 - r_const) / (self.sigma23 + r_const)
        return y

    def _calc_zn(self, n):
        yn = self._calc_yn(n)
        z = (self.r12 ** (2*n+1) - (n + 1) / n * yn) / (self.r12 ** (2*n+1) + yn)
        return z

    def _calc_c1n(self, n):
        zn = self._calc_zn(n)
        c1 = (((n + 1) / n * self.sigma12 + zn) / (self.sigma12 - zn) * self._rz1**(n+1))
        return c1

    def _calc_c2n(self, n):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        c2 = ((c1 + self._rz1**(n+1)) * self.r12 ** (n + 1) /
             (self.r12 ** (2 * n + 1) + yn))
        return c2

    def _calc_d2n(self, n, c2):
        yn = self._calc_yn(n)
        d2 = yn * c2
        return d2

    def _calc_c3n(self, n):
        vn = self._calc_vn(n)
        c2 = self._calc_c2n(n)
        d2 = self._calc_d2n(n, c2)
        c3 = (c2 + d2) * self.r23 ** (n + 1) / (self.r23 ** (2*n + 1) + vn)
        return c3

    def _calc_d3n(self, n, c3):
        vn = self._calc_vn(n)
        d3 = vn * c3
        return d3

    def _calc_c4n(self, n):
        c3 = self._calc_c3n(n)
        d3 = self._calc_d3n(n, c3)
        c4 = ((n + 1) / n * self.r34 ** (n + 1) * (c3 + d3) /
             ((n + 1) / n * self.r34 ** (2*n + 1) + 1))
        return c4

    def _calc_d4n(self, n, c4):
        d4 = n / (n + 1) * c4
        return d4

    def _calc_csf_term1(self, n, r):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        term1 = ((c1 + self._rz1 ** (n + 1)) * self.r12*((self.r1*r)/
                (self.r2 ** 2)) **n / (self.r12**(2*n+1) + yn))
        return term1

    def _calc_csf_term2(self, n, r):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        term2 = (yn*(c1 + self._rz1 ** (n + 1))/
                (r/self.r2*((self.r1 * r) / self.r2**2) ** n +
                (r / self.r1) ** (n+1)*yn))
        return term2


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
    Theta correspond to the dipole alignment angle from the vertical z-axis:

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
        Return electric potential from current dipole with current dipole
        approximation

        p : ndarray, dtype=float
            Shape (n_timesteps, 3) array containing the x,y,z components of the
            current dipole moment in units of (nA*µm) for all timesteps
        r : ndarray, dtype=float
            Shape (n_contacts, 3) array contaning the displacement vectors
            from dipole location to measurement location

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

    def get_multi_dipole_potential(self, cell, electrode_locs, timepoints=None):
        """
        Return electric potential from multiple current dipoles from cell

        By multiple current dipoles we mean the dipoles computed from all
        axial currents in a neuron simulation, typically two
        axial currents per compartment, except for the root compartment.

        Parameters
        ----------
        cell : Cell object from LFPy
        electrode_locs : ndarray, dtype=float
            Shape (n_contacts, 3) array containing n_contacts electrode
            locations in cartesian coordinates in units of (µm).
            All ``r_el`` in electrode_locs must be placed so that ``|r_el|`` is
            less than or equal to scalp radius and larger than
            the distance between dipole and sphere
            center: ``|rz| < |r_el| <= radii[3]``.
        timepoints : ndarray, dtype=int
            array of timepoints at which you want to compute
            the electric potential. Defaults to None. If not given,
            all simulation timesteps will be included.

        Returns
        -------
        potential : ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) electrode_locs in units
            of (mV) for all timesteps of neuron simulation

        Examples
        --------
        Compute extracellular potential from neuron simulation in
        four-sphere head model. Instead of simplifying the neural activity to
        a single dipole, we compute the contribution from every multi dipole
        from all axial currents in neuron simulation:

        >>> import LFPy
        >>> import numpy as np
        >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=False)
        >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,100),
        >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
        >>> syn.set_spike_times(np.mgrid[20:100:20])
        >>> cell.simulate(rec_vmem=True, rec_imem=False)
        >>> sigma = 0.3
        >>> timepoints = np.array([10, 20, 50, 100])
        >>> electrode_locs = np.array([[50., -50., 250.]])
        >>> MD_INF = LFPy.InfiniteVolumeConductor(sigma)
        >>> phi = MD_INF.get_multi_dipole_potential(cell, electrode_locs,
        >>>                                         timepoints = timepoints)
        """

        multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(timepoints=timepoints)
        N_elec = electrode_locs.shape[0]
        Ni, Nt, Nd = multi_p.shape
        potentials = np.zeros((N_elec, Nt))
        for i in range(Ni):
            p = multi_p[i]
            r = electrode_locs - multi_p_locs[i]
            pot = self.get_dipole_potential(p, r)
            potentials += pot
        return potentials

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
        dist is the length of each axial current.
        When current is an array of transmembrane
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
    computed from membrane potentials:

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
    currents using the extracellular mechanism in NEURON:

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

    .. math:: \\mathbf{H} = \\frac{\\mathbf{p} \\times \\mathbf{R}}{4 \\pi R^3}

    where :math:`\\mathbf{p}` is the current dipole moment, :math:`\\mathbf{R}`
    the vector between dipole source location and measurement location, and
    :math:`R=|\\mathbf{R}|`

    Note that the magnetic field :math:`\\mathbf{H}` is related to the magnetic
    field :math:`\\mathbf{B}` as :math:`\\mu_0 \\mathbf{H} = \\mathbf{B}-\\mathbf{M}`
    where :math:`\\mu_0` is the permeability of free space (very close to
    permebility of biological tissues). :math:`\\mathbf{M}` denotes material
    magnetization (also ignored)


    Parameters
    ----------
    sensor_locations : ndarray, dtype=float
        shape (n_locations x 3) array with x,y,z-locations of measurement
        devices where magnetic field of current dipole moments is calculated.
        In unit of (µm)
    mu : float
        Permeability. Default is permeability of vacuum (:math:`\\mu_0 = 4*\\pi*10^{-7}` T*m/A)


    Examples
    --------
    Define cell object, create synapse, compute current dipole moment:

    >>> import LFPy, os, numpy as np, matplotlib.pyplot as plt
    >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
    >>>                  passive=True)
    >>> cell.set_pos(0., 0., 0.)
    >>> syn = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=0.01, record_current=True)
    >>> syn.set_spike_times_w_netstim()
    >>> cell.simulate(rec_current_dipole_moment=True)

    Compute the dipole location as an average of segment locations weighted by membrane area:

    >>> dipole_location = (cell.area * np.c_[cell.xmid, cell.ymid, cell.zmid].T / cell.area.sum()).sum(axis=1)

    Instantiate the LFPy.MEG object, compute and plot the magnetic signal in a sensor location:

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
    def __init__(self, sensor_locations, mu=4*np.pi*1E-7):
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
        self.mu = mu


    def calculate_H(self, current_dipole_moment, dipole_location):
        """
        Compute magnetic field H from single current-dipole moment localized
        somewhere in space

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
            shape (n_locations x n_timesteps x 3) array with x,y,z-components
            of the magnetic field :math:`\\mathbf{H}` in units of (nA/µm)

        Raises
        ------
        AssertionError
            If dimensionality of current_dipole_moment and/or dipole_location
            is wrong
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


    def calculate_H_from_iaxial(self, cell):
        """
        Computes the magnetic field in space from axial currents computed from
        membrane potential values and axial resistances of multicompartment
        cells.

        See:
        Blagoev et al. (2007) Modelling the magnetic signature of neuronal
        tissue. NeuroImage 37 (2007) 137–148
        DOI: 10.1016/j.neuroimage.2007.04.033

        for details on the biophysics governing magnetic fields from axial
        currents.

        Parameters
        ----------
        cell : object
            LFPy.Cell-like object. Must have attribute vmem containing recorded
            membrane potentials in units of mV

        Examples
        --------
        Define cell object, create synapse, compute current dipole moment:

        >>> import LFPy, os, numpy as np, matplotlib.pyplot as plt
        >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test', 'ball_and_sticks.hoc'),
        >>>                  passive=True)
        >>> cell.set_pos(0., 0., 0.)
        >>> syn = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=0.01, record_current=True)
        >>> syn.set_spike_times_w_netstim()
        >>> cell.simulate(rec_vmem=True)

        Instantiate the LFPy.MEG object, compute and plot the magnetic signal in a sensor location:

        >>> sensor_locations = np.array([[1E4, 0, 0]])
        >>> meg = LFPy.MEG(sensor_locations)
        >>> H = meg.calculate_H_from_iaxial(cell)
        >>> plt.subplot(311)
        >>> plt.plot(cell.tvec, cell.somav)
        >>> plt.subplot(312)
        >>> plt.plot(cell.tvec, syn.i)
        >>> plt.subplot(313)
        >>> plt.plot(cell.tvec, H[0])

        Returns
        -------
        H : ndarray, dtype=float
            shape (n_locations x n_timesteps x 3) array with x,y,z-components
            of the magnetic field :math:`\\mathbf{H}` in units of (nA/µm)
        """
        i_axial, d_vectors, pos_vectors = cell.get_axial_currents_from_vmem()
        R = self.sensor_locations
        H = np.zeros((R.shape[0], cell.tvec.size, 3))

        for i, R_ in enumerate(R):
            for i_, d_, r_ in zip(i_axial, d_vectors, pos_vectors):
                r_rel = R_ - r_
                H[i, :, :] += np.dot(i_.reshape((-1, 1)),
                                     np.cross(d_, r_rel).reshape((1, -1))
                                     ) / (4*np.pi*np.sqrt((r_rel**2).sum())**3)
        return H
