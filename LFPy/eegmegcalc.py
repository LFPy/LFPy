#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Copyright (C) 2020 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
import lfpykit
from lfpykit.eegmegcalc import NYHeadModel  # noqa: F401
import numpy as np


class FourSphereVolumeConductor(lfpykit.eegmegcalc.FourSphereVolumeConductor):
    """
    Main class for computing extracellular potentials in a four-sphere
    volume conductor model that assumes homogeneous, isotropic, linear
    (frequency independent) conductivity within the inner sphere and outer
    shells. The conductance outside the outer shell is 0 (air).

    This class implements the corrected 4-sphere model described in [1]_, [2]_

    References
    ----------
    .. [1] Næss S, Chintaluri C, Ness TV, Dale AM, Einevoll GT and Wójcik DK
        (2017) Corrected Four-sphere Head Model for EEG Signals. Front. Hum.
        Neurosci. 11:490. doi: 10.3389/fnhum.2017.00490
    .. [2] Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling
        of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG Signals
        With LFPy 2.0. Front. Neuroinform. 12:92. doi: 10.3389/fninf.2018.00092

    See also
    --------
    InfiniteVolumeConductor
    MEG

    Parameters
    ----------
    r_electrodes: ndarray, dtype=float
        Shape (n_contacts, 3) array containing n_contacts electrode locations
        in cartesian coordinates in units of (µm).
        All ``r_el`` in ``r_electrodes`` must be less than or equal to scalp
        radius and larger than the distance between dipole and sphere
        center: ``|rz| < r_el <= radii[3]``.
    radii: list, dtype=float
        Len 4 list with the outer radii in units of (µm) for the 4
        concentric shells in the four-sphere model: brain, csf, skull and
        scalp, respectively.
    sigmas: list, dtype=float
        Len 4 list with the electrical conductivity in units of (S/m) of
        the four shells in the four-sphere model: brain, csf, skull and
        scalp, respectively.
    iter_factor: float
        iteration-stop factor

    Examples
    --------
    Compute extracellular potential from current dipole moment in four-sphere
    head model:

    >>> from lfpykit.eegmegcalc import FourSphereVolumeConductor
    >>> import numpy as np
    >>> radii = [79000., 80000., 85000., 90000.]  # (µm)
    >>> sigmas = [0.3, 1.5, 0.015, 0.3]  # (S/m)
    >>> r_electrodes = np.array([[0., 0., 90000.], [0., 85000., 0.]]) # (µm)
    >>> sphere_model = FourSphereVolumeConductor(r_electrodes, radii,
    >>>                                          sigmas)
    >>> # current dipole moment
    >>> p = np.array([[10.]*10, [10.]*10, [10.]*10]) # 10 timesteps (nA µm)
    >>> dipole_location = np.array([0., 0., 78000.])  # (µm)
    >>> # compute potential
    >>> sphere_model.get_dipole_potential(p, dipole_location)  # (mV)
    array([[1.06247669e-08, 1.06247669e-08, 1.06247669e-08, 1.06247669e-08,
            1.06247669e-08, 1.06247669e-08, 1.06247669e-08, 1.06247669e-08,
            1.06247669e-08, 1.06247669e-08],
           [2.39290752e-10, 2.39290752e-10, 2.39290752e-10, 2.39290752e-10,
            2.39290752e-10, 2.39290752e-10, 2.39290752e-10, 2.39290752e-10,
            2.39290752e-10, 2.39290752e-10]])
    """
    def __init__(self,
                 r_electrodes,
                 radii=None,
                 sigmas=None,
                 iter_factor=2. / 99. * 1e-6):
        """
        Initialize class FourSphereVolumeConductor
        """
        if radii is None:
            radii = [79000., 80000., 85000., 90000.]
        if sigmas is None:
            sigmas = [0.3, 1.5, 0.015, 0.3]
        super().__init__(r_electrodes=r_electrodes,
                         radii=radii,
                         sigmas=sigmas,
                         iter_factor=iter_factor)

    def get_dipole_potential_from_multi_dipoles(self, cell, timepoints=None):
        """
        Return electric potential from multiple current dipoles from cell.

        By multiple current dipoles we mean the dipoles computed from all
        axial currents in a neuron simulation, typically two
        axial currents per compartment, except for the root compartment.

        Parameters
        ----------
        cell: LFPy Cell object, LFPy.Cell
        timepoints: ndarray, dtype=int
            array of timepoints at which you want to compute
            the electric potential. Defaults to None. If not given,
            all simulation timesteps will be included.

        Returns
        -------
        potential: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) electrode_locs in units
            of [mV] for all timesteps of neuron simulation.

        Examples
        --------
        Compute extracellular potential from neuron simulation in
        four-sphere head model. Instead of simplifying the neural activity to
        a single dipole, we compute the contribution from every multi dipole
        from all axial currents in neuron simulation:

        >>> import os
        >>> import LFPy
        >>> from LFPy import FourSphereVolumeConductor
        >>> import numpy as np
        >>> cell = LFPy.Cell(os.path.join(LFPy.__path__[0], 'test',
        >>>                               'ball_and_sticks.hoc'),
        >>>                  v_init=-65, cm=1., Ra=150,
        >>>                  passive=True,
        >>>                  passive_parameters=dict(g_pas=1/1E4, e_pas=-65))
        >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,100),
        >>>                    syntype='ExpSyn', e=0., tau=1., weight=0.001)
        >>> syn.set_spike_times(np.mgrid[20:100:20])
        >>> cell.simulate(rec_vmem=True, rec_imem=False)
        >>> cell.set_pos(0, 0, 78800)
        >>> radii = [79000., 80000., 85000., 90000.]
        >>> sigmas = [0.3, 1.5, 0.015, 0.3]
        >>> r_electrodes = np.array([[0., 0., 90000.]])
        >>> MD_4s = FourSphereVolumeConductor(r_electrodes=r_electrodes,
        >>>                                   radii=radii,
        >>>                                   sigmas=sigmas)
        >>> phi = MD_4s.get_dipole_potential_from_multi_dipoles(cell)
        """
        multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(
            timepoints)
        N_elec = self.rxyz.shape[0]
        Ni, Nd, Nt = multi_p.shape
        potential = np.zeros((N_elec, Nt))
        for i in range(Ni):
            pot = self.get_dipole_potential(multi_p[i], multi_p_locs[i])
            potential += pot
        return potential


class InfiniteVolumeConductor(lfpykit.eegmegcalc.InfiniteVolumeConductor):
    """
    Main class for computing extracellular potentials with current dipole
    moment :math:`\\mathbf{P}` in an infinite 3D volume conductor model that
    assumes homogeneous, isotropic, linear (frequency independent)
    conductivity :math:`\\sigma`. The potential :math:`V` is computed as [1]_:

    .. math:: V = \\frac{\\mathbf{P} \\cdot \\mathbf{r}}{4 \\pi \\sigma r^3}

    Parameters
    ----------
    sigma: float
        Electrical conductivity in extracellular space in units of (S/cm)

    See also
    --------
    FourSphereVolumeConductor
    MEG

    References
    ----------
    .. [1] Nunez and Srinivasan, Oxford University Press, 2006

    Examples
    --------
    Computing the potential from dipole moment valid in the far field limit.
    Theta correspond to the dipole alignment angle from the vertical z-axis:

    >>> from lfpykit.eegmegcalc import InfiniteVolumeConductor
    >>> import numpy as np
    >>> inf_model = InfiniteVolumeConductor(sigma=0.3)
    >>> p = np.array([[10.], [10.], [10.]])  # [nA µm]
    >>> r = np.array([[1000., 0., 5000.]])  # [µm]
    >>> inf_model.get_dipole_potential(p, r)  # [mV]
    array([[1.20049432e-07]])
    """

    def __init__(self, sigma=0.3):
        """
        Initialize class InfiniteVolumeConductor
        """
        super().__init__(sigma=sigma)

    def get_multi_dipole_potential(self, cell,
                                   electrode_locs, timepoints=None):
        """
        Return electric potential from multiple current dipoles from cell

        The multiple current dipoles corresponds to dipoles computed from all
        axial currents in a neuron simulation, typically two
        axial currents per compartment, excluding the root compartment.

        Parameters
        ----------
        cell: LFPy.Cell object
        electrode_locs: ndarray, dtype=float
            Shape (n_contacts, 3) array containing n_contacts electrode
            locations in cartesian coordinates in units of [µm].
            All ``r_el`` in electrode_locs must be placed so that ``|r_el|`` is
            less than or equal to scalp radius and larger than
            the distance between dipole and sphere
            center: ``|rz| < |r_el| <= radii[3]``.
        timepoints: ndarray, dtype=int
            array of timepoints at which you want to compute
            the electric potential. Defaults to None. If not given,
            all simulation timesteps will be included.

        Returns
        -------
        potential: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) electrode_locs in units
            of [mV] for all timesteps of neuron simulation

        Examples
        --------
        Compute extracellular potential from neuron simulation in
        four-sphere head model. Instead of simplifying the neural activity to
        a single dipole, we compute the contribution from every multi dipole
        from all axial currents in neuron simulation:

        >>> import LFPy
        >>> from lfpykit.eegmegcalc import InfiniteVolumeConductor
        >>> import numpy as np
        >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=False)
        >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,100),
        >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
        >>> syn.set_spike_times(np.mgrid[20:100:20])
        >>> cell.simulate(rec_vmem=True, rec_imem=False)
        >>> sigma = 0.3
        >>> timepoints = np.array([10, 20, 50, 100])
        >>> electrode_locs = np.array([[50., -50., 250.]])
        >>> MD_INF = InfiniteVolumeConductor(sigma)
        >>> phi = MD_INF.get_multi_dipole_potential(cell, electrode_locs,
        >>>                                         timepoints = timepoints)
        """
        multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(
            timepoints=timepoints)
        N_elec = electrode_locs.shape[0]
        Ni, Nd, Nt = multi_p.shape
        potentials = np.zeros((N_elec, Nt))
        for i in range(Ni):
            p = multi_p[i]
            r = electrode_locs - multi_p_locs[i]
            pot = self.get_dipole_potential(p, r)
            potentials += pot
        return potentials


class MEG(lfpykit.eegmegcalc.MEG):
    """
    Basic class for computing magnetic field from current dipole moment.
    For this purpose we use the Biot-Savart law derived from Maxwell's
    equations under the assumption of negligible magnetic induction
    effects [1]_:

    .. math:: \\mathbf{H} = \\frac{\\mathbf{p} \\times \\mathbf{R}}{4 \\pi R^3}

    where :math:`\\mathbf{p}` is the current dipole moment, :math:`\\mathbf{R}`
    the vector between dipole source location and measurement location, and
    :math:`R=|\\mathbf{R}|`

    Note that the magnetic field :math:`\\mathbf{H}` is related to the magnetic
    field :math:`\\mathbf{B}` as

    .. math:: \\mu_0 \\mathbf{H} = \\mathbf{B}-\\mathbf{M}

    where :math:`\\mu_0` is the permeability of free space (very close to
    permebility of biological tissues). :math:`\\mathbf{M}` denotes material
    magnetization (also ignored)

    Parameters
    ----------
    sensor_locations: ndarray, dtype=float
        shape (n_locations x 3) array with x,y,z-locations of measurement
        devices where magnetic field of current dipole moments is calculated.
        In unit of [µm]
    mu: float
        Permeability. Default is permeability of vacuum
        (:math:`\\mu_0 = 4*\\pi*10^{-7}` T*m/A)

    See also
    --------
    FourSphereVolumeConductor
    InfiniteVolumeConductor

    References
    ----------
    .. [1] Nunez and Srinivasan, Oxford University Press, 2006

    Examples
    --------
    Define cell object, create synapse, compute current dipole moment:

    >>> import LFPy, os, numpy as np, matplotlib.pyplot as plt
    >>> from LFPy import MEG
    >>> # create LFPy.Cell object
    >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
    >>>                                          'ball_and_sticks.hoc'),
    >>>                  passive=True)
    >>> cell.set_pos(0., 0., 0.)
    >>> # create single synaptic stimuli at soma (idx=0)
    >>> syn = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=0.01, tau=5,
    >>>                    record_current=True)
    >>> syn.set_spike_times_w_netstim()
    >>> # simulate, record current dipole moment
    >>> cell.simulate(rec_current_dipole_moment=True)
    >>> # Compute the dipole location as an average of segment locations
    >>> # weighted by membrane area:
    >>> dipole_location = (cell.area * np.c_[cell.xmid, cell.ymid, cell.zmid].T
    >>>                    / cell.area.sum()).sum(axis=1)
    >>> # Define sensor site, instantiate MEG object, get transformation matrix
    >>> sensor_locations = np.array([[1E4, 0, 0]])
    >>> meg = MEG(sensor_locations)
    >>> M = meg.get_transformation_matrix(dipole_location)
    >>> # compute the magnetic signal in a single sensor location:
    >>> H = M @ cell.current_dipole_moment.T
    >>> # plot output
    >>> plt.figure(figsize=(12, 8), dpi=120)
    >>> plt.subplot(311)
    >>> plt.plot(cell.tvec, cell.somav)
    >>> plt.ylabel(r'$V_{soma}$ (mV)')
    >>> plt.subplot(312)
    >>> plt.plot(cell.tvec, syn.i)
    >>> plt.ylabel(r'$I_{syn}$ (nA)')
    >>> plt.subplot(313)
    >>> plt.plot(cell.tvec, H[0].T)
    >>> plt.ylabel(r'$H$ (nA/um)')
    >>> plt.xlabel('$t$ (ms)')
    >>> plt.legend(['$H_x$', '$H_y$', '$H_z$'])
    >>> plt.show()

    Raises
    ------
    AssertionError
        If dimensionality of sensor_locations is wrong
    """
    def __init__(self, sensor_locations, mu=4 * np.pi * 1E-7):
        super().__init__(sensor_locations=sensor_locations, mu=mu)

    def calculate_H_from_iaxial(self, cell):
        """
        Computes the magnetic field in space from axial currents computed from
        membrane potential values and axial resistances of multicompartment
        cells.

        See [1]_ for details on the biophysics governing magnetic fields from
        axial currents.

        Parameters
        ----------
        cell: object
            LFPy.Cell-like object. Must have attribute vmem containing recorded
            membrane potentials in units of mV

        References
        ----------
        .. [1] Blagoev et al. (2007) Modelling the magnetic signature of
            neuronal tissue. NeuroImage 37 (2007) 137–148
            DOI: 10.1016/j.neuroimage.2007.04.033

        Examples
        --------
        Define cell object, create synapse, compute current dipole moment:

        >>> import LFPy, os, numpy as np, matplotlib.pyplot as plt
        >>> from lfpykit.eegmegcalc import MEG
        >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
        >>>                                          'ball_and_sticks.hoc'),
        >>>                  passive=True)
        >>> cell.set_pos(0., 0., 0.)
        >>> syn = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=0.01,
        >>>                    record_current=True)
        >>> syn.set_spike_times_w_netstim()
        >>> cell.simulate(rec_vmem=True)
        >>> # Instantiate the MEG object, compute and plot the magnetic
        >>> # signal in a sensor location:
        >>> sensor_locations = np.array([[1E4, 0, 0]])
        >>> meg = MEG(sensor_locations)
        >>> H = meg.calculate_H_from_iaxial(cell)
        >>> plt.subplot(311)
        >>> plt.plot(cell.tvec, cell.somav)
        >>> plt.subplot(312)
        >>> plt.plot(cell.tvec, syn.i)
        >>> plt.subplot(313)
        >>> plt.plot(cell.tvec, H[0])
        >>> plt.show()

        Returns
        -------
        H: ndarray, dtype=float
            shape (n_locations x 3 x n_timesteps) array with x,y,z-components
            of the magnetic field :math:`\\mathbf{H}` in units of (nA/µm)
        """
        i_axial, d_vectors, pos_vectors = cell.get_axial_currents_from_vmem()
        R = self.sensor_locations
        H = np.zeros((R.shape[0], 3, cell.tvec.size))

        for i, R_ in enumerate(R):
            for i_, d_, r_ in zip(i_axial, d_vectors, pos_vectors):
                r_rel = R_ - r_
                H[i, :, :] += (i_.reshape((-1, 1))
                               @ np.cross(d_, r_rel).reshape((1, -1))).T \
                    / (4 * np.pi * np.sqrt((r_rel**2).sum())**3)
        return H
