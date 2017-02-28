# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:24:00 2015

@author: solveig
"""

from __future__ import division
from scipy.special import eval_legendre, lpmv
import numpy as np

class FourSphereVolumeConductor(object):
    """
    Main class for computing extracellular potentials with four-sphere model.

    Parameters
    __________
    radii : list of floats in [micro m]
            List containing the outer radii of the 4 concentric shells in the
            four-sphere model: brain, csf, skull and scalp, respectively.
    sigmas : list of floats in [S/cm]
             List containing 4 floats, i.e. the electrical conductivity of the
             four media in the four-sphere model:
             brain, csf, skull and scalp, respectively
    r : ndarray [micro m]
        Array containing n electrode locations in cartesian coordinates.
        r.shape = (n, 3)
        All r_el in r must be smaller than the scalp radius and larger than the
        distance between dipole and sphere center: |rz| < r_el < radii[3]
    rz : ndarray [micro m]
         Array containing the position of the current dipole in cartesian
         coordinates.

    Examples
    ________
    Here's an example on how to use the FourSphereVolumeConductor class.

    >>> import LFPy
    >>> import numpy as np
    >>> radii = [79000., 80000., 85000., 90000.]
    >>> sigmas = [0.3, 1.5, 0.015, 0.3]
    >>> r = np.array([[0., 0., 90000.], [0., 85000., 0.]])
    >>> rz = np.array([0., 0., 78000.])
    >>> sphere_model = LFPy.FourSphereVolumeConductor(radii, sigmas, r, rz)
    >>> # current dipole moment
    >>> P = np.array([[10., 10., 10.]])
    >>> potential = sphere_model.calc_potential(P)

    """

    def __init__(self, radii, sigmas, r, rz):

        self.k1 = 1E6  # from mV to nV
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

        # if self.r < self.rz:
        #     raise ValueError('Electrode must be farther away from brain center than dipole. r = %s, rz = %s', self.r, self.rz)
        # elif self.r > self.r4:
        #     raise ValueError('Electrode located outside head model. Your r = %s µm. Maximum r = %s µm.', self.r, self.r4)

    def calc_potential(self, p):
        """
        Parameters
        __________
        p : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns
        _______
        potential : ndarray [nV]
            Array containing the electric potential at point self.r.

        """
        p_rad, p_tan = self.decompose_dipole(p)
        pot_rad = self.calc_rad_potential(p_rad)
        pot_tan = self.calc_tan_potential(p_tan)

        pot_tot = pot_rad + pot_tan
        return pot_tot

    def decompose_dipole(self, p):
        """Decompose current dipole moment vector in radial and tangential terms
        Parameters
        __________
        p : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns:
        ________
        p_rad : ndarray [1E-15 Am]
                Radial part of p, parallel to self.rz
        p_tan : ndarray [1E-15 Am]
                Tangential part of p, orthogonal to self.rz
        """
        p_rad = (np.dot(p, self.rzloc)/self.rz ** 2).reshape(len(p),1) * self.rzloc.reshape(1, len(self.rzloc))
        p_tan = p - p_rad

        return p_rad, p_tan

    def calc_rad_potential(self, p_rad):
        """Return potential from radial dipole p_rad at location rz measured at r
        Parameters
        __________
        P : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns
        _______
        potential : ndarray [nV]
            Array containing the current dipole moment at point r.
        """

        p_tot = np.linalg.norm(p_rad, axis=1)
        theta = self.calc_theta()
        s_vector = self._sign_rad_dipole(p_rad)
        phi_const = s_vector * p_tot / (4 * np.pi * self.sigma1 * self.rz ** 2) * self.k1
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            el_rad = self.r[el_point]
            theta_point = theta[el_point]
            if el_rad <= self.r1:
                n_terms[el_point] = self.potential_brain_rad(el_rad, theta_point)
            elif el_rad <= self.r2:
                n_terms[el_point] = self.potential_csf_rad(el_rad, theta_point)
            elif el_rad <= self.r3:
                n_terms[el_point] = self.potential_skull_rad(el_rad, theta_point)
            elif el_rad <= (self.r4):
                n_terms[el_point] = self.potential_scalp_rad(el_rad, theta_point)
            elif el_rad <= (self.r4+1E-6):
                el_rad = self.r4
                n_terms[el_point] = self.potential_scalp_rad(el_rad, theta_point)
            else:
                n_terms[el_point] = np.nan
                raise ValueError('Electrode located outside head model. Maximum r = %s µm.', self.r4, '\n your r = ', self.r)
        potential = phi_const * n_terms
        return potential

    def calc_tan_potential(self, p_tan):
        """Return potential from tangential dipole P at location rz measured at r
        Parameters
        __________
        p : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns
        _______
        potential : ndarray [nV]
            Array containing the current dipole moment at point r.
        """
        theta = self.calc_theta()
        phi = self.calc_phi(p_tan)
        p_tot = np.linalg.norm(p_tan, axis=1)
        phi_hom = - p_tot / (4 * np.pi * self.sigma1 * self.rz ** 2) * np.sin(phi) * self.k1
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            el_rad = self.r[el_point]
            theta_point = theta[el_point]
            if el_rad <= self.r1:
                n_terms[el_point] = self.potential_brain_tan(el_rad, theta_point)
            elif el_rad <= self.r2:
                n_terms[el_point] = self.potential_csf_tan(el_rad, theta_point)
            elif el_rad <= self.r3:
                n_terms[el_point] = self.potential_skull_tan(el_rad, theta_point)
            elif el_rad <= self.r4:
                n_terms[el_point] = self.potential_scalp_tan(el_rad, theta_point)
            else:
                n_terms[el_point] = np.nan
        potential = phi_hom * n_terms
        return potential

    def calc_theta(self):
        """Calculate angle between radial dipole and measurement location
        Returns
        _______
        phi : ndarray [radians]
              Array containing polar angle between z-axis and
              electrode location vector rxyz.
              Z-axis is defined in the direction of rzloc and the radial dipole.
        """
        cos_theta = np.dot(self.rxyz, self.rzloc) / (np.linalg.norm(self.rxyz, axis=1) * np.linalg.norm(self.rzloc))
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        return theta

    def calc_phi(self, p):
        """Calculate angle between tangential dipole and measurement location
           Parameters
           __________
           p : ndarray [1E-15 Am]
               Array containing tangential component of current dipole moment
           Returns
           _______
           phi : ndarray [radians]
                 Array containing azimuthal angle between x-axis and
                 projection of electrode location vector rxyz into xy-plane,
                 rxy.
                 Z-axis is defined in the direction of rzloc.
                 Y-axis is defined in the direction of p (orthogonal to rzloc).
                 X-axis is defined as cross product between p and rzloc (x).
                 """
        proj_rxyz_rz = (np.dot(self.rxyz,
                        self.rzloc) / np.sum(self.rzloc **
                        2)).reshape(len(self.rxyz),1) * self.rzloc.reshape(1,3)
                        # projection of rxyz onto rzloc
        rxy = self.rxyz - proj_rxyz_rz  # projection of rxyz into xy-plane
        x = np.cross(p, self.rzloc)  # vector giving direction of x-axis
        cos_phi = np.dot(rxy, x.T)/np.dot(np.linalg.norm(rxy,
                         axis=1).reshape(len(rxy),1),np.linalg.norm(x,
                         axis=1).reshape(1, len(x)))
        cos_phi = np.nan_to_num(cos_phi)
        phi_temp = np.arccos(cos_phi) # nb: phi_temp is in range [0, pi]
        phi = phi_temp
        range_test = np.dot(rxy, p.T)  # if range_test < 0, phi > pi
        for i in range(len(self.r)):
            for j in range(len(p)):
                if range_test[i,j] < 0:
                    phi[i,j] = 2*np.pi - phi_temp[i,j]
        return phi

    def _sign_rad_dipole(self, p):
        """Flip radial dipole pointing inwards (i.e. we only use p_tot),
        and add a -1 to the s-vector, so that the potential can be
        calculated as if the dipole were pointing outwards,
        and later be multiplied by -1 to get the right potential."""
        sign_vector = np.ones(len(p))
        radial_test = np.dot(p, self.rzloc) / (np.linalg.norm(p, axis=1) * self.rz)
        for i in range(len(p)):
            if np.abs(radial_test[i] + 1) < 10 ** -8:
                sign_vector[i] = -1.
        return sign_vector

    def potential_brain_rad(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1, 100)
        c1n = self._calc_c1n(n)
        consts = n*(c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_csf_rad(self, r, theta):
        """Calculate potential in CSF from radial dipole
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c2n = self._calc_c2n(n)
        d2n = self._calc_d2n(n, c2n)
        consts = n*(c2n * (r / self.r2) ** n + d2n * (self.r2 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_skull_rad(self, r, theta):
        """Calculate potential in skull from radial dipole
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c3n = self._calc_c3n(n)
        d3n = self._calc_d3n(n, c3n)
        consts = n*(c3n * (r / self.r3) ** n + d3n * (self.r3 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_scalp_rad(self, r, theta):
        """Calculate potential in scalp from radial dipole
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c4n = self._calc_c4n(n)
        d4n = self._calc_d4n(n, c4n)
        consts = n*(c4n * (r / self.r4) ** n + d4n * (self.r4 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_brain_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c1n = self._calc_c1n(n)
        consts = (c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def potential_csf_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c2n = self._calc_c2n(n)
        d2n = self._calc_d2n(n, c2n)
        consts = c2n*(r/self.r2)**n + d2n*(self.r2/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def potential_skull_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c3n = self._calc_c3n(n)
        d3n = self._calc_d3n(n, c3n)
        consts = c3n*(r/self.r3)**n + d3n*(self.r3/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def potential_scalp_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
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
    Main class for computing potentials with current dipole approximation.

    Parameters
    __________
    sigma : float [S/cm]
            Electrical conductivity in extracellular space.
    Examples
    ________
    Computing the potential from dipole moment valid in the far field limit.
    Theta correspond to the dipole alignment angle from the vertical z-axis.
    >>> import LFPy
    >>> import numpy as np
    >>> inf_model = LFPy.InfiniteVolumeConductor(sigma=0.3)
    >>> P = np.array([[10., 10., 10.]])
    >>> Phi_P, theta = inf_model.get_dipole_potential(P,
    >>>                                               x=np.array([1000]),
    >>>                                               y=np.array([0]),
    >>>                                               z=np.array([5000]))

    """

    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def get_dipole_potential(self, P, x, y, z):
        '''
        Compute the electric potential in infinite homogeneous linear conductive
        media from a current dipole moment at a distance [x, y, z]

        Parameters
        ----------
        P : ndarray [1E-15 mA]
            Array containing the current dipole moment for
            all timesteps in the x-, y- and z-direction.
        x, y, z : ndarray,
            arrays of coordinates relative to soma midpoint where potential is
            computed
        sigma : float [ohm/m]
            Extracellular Conductivity.

        Returns
        -------
        theta : ndarray [radians]
            Angle between phi(t) and distance vector from
            electrode to current dipole location,
            calculated for all timesteps.
        phi : ndarray [nV]
            Array containing the current dipole moment at all
            points in x-, y-, z-grid for all timesteps.

        '''
        phi = np.zeros((x.size, P.shape[-1]))
        for j, dist in enumerate(np.c_[x, y, z]):
            cos_theta = np.dot(P, dist) / (np.linalg.norm(dist) *
                                           np.linalg.norm(P, axis=1))
            cos_theta = np.nan_to_num(cos_theta)
            theta = np.arccos(cos_theta)
            phi[j, :] = 1. / (4*np.pi*self.sigma) *\
                             np.linalg.norm(P, axis=1)*cos_theta/(dist**2).sum()

        return phi, theta

def get_current_dipole_moment(dist, current):
    """
    Return current dipole moment vector P and P_tot of cell.

    Parameters
    ----------
    current : ndarray [nA]
        Either an array containing all transmembrane currents
        from all compartments of the cell. Or an array of all
        axial currents between compartments in cell.
    dist : ndarray [microm]
        When input current is an array of axial currents,
        the dist is the length of each axial current.
        When current is the an array of transmembrane
        currents, dist is the position vector of each
        compartment middle.

    Returns
    -------
    P : ndarray [10^-15 mA]
        Array containing the current dipole moment for all
        timesteps in the x-, y- and z-direction.
    P_tot : ndarray [10^-15 mA]
        Array containing the magnitude of the
        current dipole moment vector for all timesteps.

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
