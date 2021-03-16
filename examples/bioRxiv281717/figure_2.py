#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation and plotting script reproducing figure 2 of:

Multimodal modeling of neural network activity: computing LFP, ECoG, EEG and
MEG signals with LFPy2.0
Espen Hagen, Solveig Næss, Torbjørn V Ness, Gaute T Einevoll
bioRxiv 281717; doi: https://doi.org/10.1101/281717
"""
# import of modules
import LFPy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.collections import PolyCollection
import scipy.signal as ss
import neuron

# set some plotting parameters
fontsize = 14
titlesize = 16
legendsize = 12
plt.rcParams.update({
    'axes.xmargin': 0.0,
    'axes.ymargin': 0.0,
    'axes.labelsize': fontsize,
    'axes.titlesize': titlesize,
    'figure.titlesize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': legendsize,
})

##########################################################################
# Main script, set parameters and create cell, synapse and electrode objects
##########################################################################

# clear out old section references in NEURON
neuron.h('forall delete_section()')

# parameters for Cell instance with simplified morphology
cellParameters = {
    'morphology': 'simplemorpho_modded.hoc',
    'tstop': 10.,      # sim duration
    'dt': 2**-4,       # sim time step size
    'v_init': -65,     # intitial voltage
    'cm': 1.,          # membrane capacitance
    'Ra': 150.,        # axial resistivity
    'passive': True,   # turn on passive mechanism
    'passive_parameters': {'g_pas': 1. / 3E4,   # passive leak conductance
                           'e_pas': -65.},    # leak reversal potential
    'pt3d': True,      # keep morphological detail
}
# create cell, set alignment
cell = LFPy.Cell(**cellParameters)
cell.set_rotation(x=np.pi / 2)

# synapse location index
synidx = cell.get_idx(['apic[3]'])[1]

# parameters for synapse instance
synapseParameters = {
    'idx': synidx,
    'e': 0,                                # reversal potential
    'syntype': 'Exp2Syn',                   # synapse type
    'tau1': 0.1,                              # syn. rise time constant
    'tau2': 1.,                              # syn. decay time constant
    'weight': 0.002,                        # syn. weight
    'record_current': True                 # syn. current record
}
# create synapse and set activation time
synapse = LFPy.Synapse(cell, **synapseParameters)
synapse.set_spike_times(np.array([2.]))

# extracellular electrode parameters
dx = 1.  # spatial resolution
x = np.arange(-100., 140. + dx, dx)
z = np.arange(-85., 255. + dx, dx)
X, Z = np.meshgrid(x, z)
electrodeParameters = {
    'x': X.flatten(),
    'y': np.zeros(X.size),
    'z': Z.flatten(),
    'sigma': 0.3,
    'method': 'root_as_point',
}
# instantiate electrode
electrode = LFPy.RecExtElectrode(cell=cell, **electrodeParameters)

# instantiate current dipole moment
current_dipole_moment = LFPy.CurrentDipoleMoment(cell)

# compute cell response, current dipole moment and extracellular potential
cell.simulate(probes=[electrode, current_dipole_moment],
              rec_imem=True,
              rec_vmem=True)


# compute effective dipole location as the cell's 'center of gravity of areas'
R_cell = (cell.area * np.c_[cell.x.mean(axis=-1),
                            cell.y.mean(axis=-1),
                            cell.z.mean(axis=-1)].T /
          cell.area.sum()).sum(axis=1)

# compute the electric potential of the dipole using
# \phi = \frac{1}{4 pi \sigma} \frac{\mathbf{p} \cdot \hat\mathbf{R}}{R^2}
# where
# \hat\mathbf{R} = \frac{\mathbf{R}}{R}
dx_p = 10.
x_p = np.arange(-1200., 1200. + dx_p, dx_p)
z_p = np.arange(-1700., 1700. + dx_p, dx_p)
X_p, Z_p = np.meshgrid(x_p, z_p)
Y_p = np.zeros(X_p.shape)
R = np.c_[X_p.flatten(), Y_p.flatten(), Z_p.flatten()]
R_rel = R - R_cell
R_scalar = np.sqrt((R_rel**2).sum(axis=1))
phi_p = 1. / (4 * np.pi * electrode.sigma) \
    * ((R_rel @ current_dipole_moment.data).T
        / R_scalar**3)  # (omega*m*nA*µm/µm^3=mV)
# mask out values in spatial locations in vicinity of the cell:
mask = np.zeros(phi_p.shape).astype(bool)
mask[:, R_scalar < 500.] = True
phi_p = np.ma.masked_array(phi_p, mask=mask)


# compute potential on dipole grid
electrodeParameters_p = {
    'x': X_p.flatten(),
    'y': np.zeros(X_p.size),
    'z': Z_p.flatten(),
    'sigma': 0.3,
    'method': 'root_as_point',
}
electrode_p = LFPy.RecExtElectrode(cell=cell, **electrodeParameters_p)
electrode_p.data = electrode_p.get_transformation_matrix() @ cell.imem
LFP_p = np.ma.masked_array(electrode_p.data, mask=(np.equal(mask.T, False)))


# Compute the magnetic field strengt |\mathbf{H}| at locations corresponding
# the electrode grid from the dipole moment using the Biot-Savart law
# H = (p x R) / (4 pi u_0 |R|**3)
H = np.zeros((current_dipole_moment.data.shape[1], R_rel.shape[0], 3))
for i, r in enumerate(R_rel):
    if R_scalar[i] > 500:
        H[:, i, :] = np.cross(current_dipole_moment.data.T, r) / \
            (4 * np.pi * np.sqrt((r**2).sum())**3)


# for locations within 500 um, compute the magnetic field from axial currents
i_axial, d_vectors, pos_vectors = cell.get_axial_currents_from_vmem()
inds = np.where(R_scalar < 500.)[0]
for i in inds:
    R_ = R[i, ]
    for i_, d_, r_ in zip(i_axial, d_vectors.T, pos_vectors):
        r_rel = R_ - r_
        H[:, i, :] += np.dot(i_.reshape((-1, 1)),
                             np.cross(d_, r_rel).reshape((1, -1))
                             ) / (4 * np.pi * np.sqrt((r_rel**2).sum())**3)

# set up four-sphere head model params
_theta = np.linspace(-np.pi / 4, np.pi / 4, 9)
_x = 90000. * np.sin(_theta)
_y = np.zeros(_theta.size)
_z = 90000. * np.cos(_theta)
foursphereParams = {
    'radii': [79000., 80000., 85000., 90000.],  # shell radii
    'sigmas': [0.3, 1.5, 0.015, 0.3],  # shell conductivity
    'r_electrodes': np.c_[_x, _y, _z],  # contact coordinates
}
dipole_position = np.array([0, 0, 78000.])      # dipole location


#######################################
# Define some plotting helper functions
#######################################
def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        scalebarpos=0,
        scalebarbasis='log2',
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    ''' draw some nice lines'''

    tvec = np.arange(data.shape[1]) * dt
    try:
        tinds = (tvec >= T[0]) & (tvec <= T[1])
    except TypeError:
        print(data.shape, T)
        raise Exception

    # apply temporal filter
    if filter:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    # subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i + 1))
        yticks.append(z)

    if scalebar:
        if scalebarbasis == 'log2':
            ax.plot([tvec[tinds][-1],
                     tvec[tinds][-1]],
                    [-1 - scalebarpos, - 2 - scalebarpos],
                    lw=2, color=color, clip_on=False)
            ax.text(tvec[tinds][-1] + np.diff(T) * 0.03, -1.5 - scalebarpos,
                    '$2^{' + '{}'.format(int(np.log2(vlimround))) +
                    '}$ ' + '{0}'.format(unit),
                    color=color, rotation='vertical',
                    va='center')
        elif scalebarbasis == 'log10':
            # recompute scale bar size to show it on scientific format
            vlimround10 = 10**np.round(np.log10(vlimround))
            if vlimround10 >= 1:
                vlimround10 = int(np.round(vlimround10))
            rescale = vlimround10 / vlimround
            ax.plot([tvec[tinds][-1], tvec[tinds][-1]],
                    np.array([0.5, -0.5]) * rescale - 1.5 - scalebarpos,
                    lw=2, color=color, clip_on=False)
            ax.text(tvec[tinds][-1] + np.diff(T) * 0.03, -1.5 - scalebarpos,
                    '{0} '.format(vlimround10) + '{0}'.format(unit),
                    color=color, rotation='vertical',
                    va='center')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r'time (ms)', labelpad=0.1)

    return vlimround


def remove_axis_junk(ax, lines=['right', 'top']):
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


#######################################
# Figure
#######################################

# plot and annotate
plt.close('all')
fig = plt.figure(figsize=(16, 12))
fig.subplots_adjust()
gs = GridSpec(
    9,
    3,
    left=0.05,
    right=0.98,
    wspace=0.15,
    hspace=-0.2,
    bottom=0.05,
    top=0.95)
alphabet = 'ABCDEF'

# set up subplot panels
ax0 = fig.add_subplot(gs[:6, 0], aspect='equal')  # LFP forward model ill.
ax0.axis('off')
ax0.set_title('extracellular potential')
ax1 = fig.add_subplot(gs[:6, 1], aspect='equal')  # dipole moment ill.
ax1.axis('off')
ax1.set_title('extracellular potential')
ax2 = fig.add_subplot(gs[:6, 2], aspect='equal')  # dipole moment ill.
ax2.axis('off')
ax2.set_title('magnetic field')
# ax3 = fig.add_subplot(gs[0, 3], aspect='equal')  # spherical shell model ill.
# ax3.set_title('4-sphere volume conductor')
# ax4 = fig.add_subplot(gs[1, 3],
# aspect='equal'
# )                 # MEG/EEG forward model ill.
# ax4.set_title('EEG and MEG signal detection')

ax3 = SubplotZero(fig, gs[7:, 0])
fig.add_subplot(ax3)
ax3.set_title('4-sphere volume conductor', verticalalignment='bottom')
ax4 = fig.add_subplot(gs[7:, 1])  # EEG
ax4.set_title(r'scalp electric potential $\phi_\mathbf{p}(\mathbf{r})$')
ax5 = fig.add_subplot(gs[7:, 2], sharey=ax4)  # MEG
# ax5.set_title('scalp magnetic field')

# morphology - line sources for panels A and B
zips = []
xz = cell.get_idx_polygons()
for x, z in xz:
    zips.append(list(zip(x, z)))
for ax in [ax0]:
    polycol = PolyCollection(zips,
                             linewidths=(0.5),
                             edgecolors='k',
                             facecolors='none',
                             zorder=-5)
    ax.add_collection(polycol)

    # morphology mid points
    ax.plot(cell.x.mean(axis=-1), cell.z.mean(axis=-1), 'o',
            mec='none', mfc='k',
            markersize=3, zorder=0)

    # mark synapse location
    ax.plot(
        synapse.x + 5,
        synapse.z,
        '> ',
        mec='r',
        mfc='r',
        markersize=10,
        zorder=1)

for ax in [ax1, ax2]:
    polycol = PolyCollection(zips,
                             linewidths=(0.5),
                             edgecolors='gray',
                             facecolors='gray',
                             zorder=-5)
    ax.add_collection(polycol)

    # mark synapse location
    ax.plot(
        synapse.x + 5,
        synapse.z,
        '> ',
        mec='r',
        mfc='r',
        markersize=5,
        zorder=1)

# morphology - offset with pt3d info for panel A
zips = []
offset = 0.
for x, z in cell.get_pt3d_polygons():
    zips.append(list(zip(x - offset, z + offset)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='gray',
                         alpha=0.5,
                         zorder=-6)
ax0.add_collection(polycol)

# some extracellular position and annotation
ax0.plot(-80,
         30,
         'o',
         markeredgecolor='none',
         markerfacecolor='b',
         markersize=5,
         zorder=0,
         clip_on=False)
ax0.text(-80, 40, r'$\phi({\bf r}, t)$', horizontalalignment='center')
idx = cell.get_idx('apic')[2]
ax0.plot([cell.x[idx].mean(), -80], [cell.z[idx].mean(), 30],
         '-.k', linewidth=1.0, clip_on=False)
ax0.fill(xz[idx][0], xz[idx][1], edgecolor='none', facecolor='k', zorder=-7)
ax0.text(
    cell.x[idx].mean() + 5,
    cell.z[idx].mean(),
    r'$I_n^\mathrm{m}({\bf r}_n, t)$',
    verticalalignment='center')
ax0.text(
    cell.x[idx].mean() - 40,
    cell.z[idx].mean() - 10,
    r'$|{\bf r}-{\bf r}_n|$',
    horizontalalignment='center')


# draw LFPs at t=|i_syn|_max
t_max = abs(synapse.i) == abs(synapse.i).max()
vmax = 2
im = ax0.pcolormesh(X, Z, electrode.data[:, t_max].reshape(X.shape) * 1E3,
                    cmap=plt.get_cmap('PRGn', 51), zorder=-10,
                    vmin=-vmax,
                    vmax=vmax,
                    rasterized=True,
                    shading='auto')
bbox = np.array(ax0.get_position())
cax = fig.add_axes([bbox[0][0] + (bbox[1][0] - bbox[0][0]) / 4,
                    bbox[0][1], (bbox[1][0] - bbox[0][0]) / 2, 0.015])
axcb = fig.colorbar(im, cax=cax, orientation='horizontal')
axcb.outline.set_visible(False)
axcb.set_label(r'$\phi$ ($\mu$V)', labelpad=0)
axcb.set_ticks([-vmax, 0, vmax])

# plot arrow representing dipole moment magnitude and direction
ax0.annotate("",  # r"$\mathbf{p}(I_n^{(\mathrm{m})}(t), \mathbf{r}_n)$",
             xy=(R_cell[0], R_cell[2]),
             xytext=(R_cell[0]
                     + current_dipole_moment.data[:, t_max][0, 0] * 5,
                     R_cell[2]
                     + current_dipole_moment.data[:, t_max][2, 0] * 5),
             arrowprops=dict(arrowstyle="<-", lw=3, color='k',
                             shrinkA=0, shrinkB=0
                             ),
             zorder=100)
ax0.text(R_cell[0] + current_dipole_moment.data[:, t_max][0, 0] * 5 / 2 + 1,
         R_cell[2] + current_dipole_moment.data[:, t_max][2, 0] * 5 / 2,
         r"$\mathbf{p}(I_n^\mathrm{m}(t), \mathbf{r}_n)$")


# create axes for synapse input current
synax = fig.add_axes([0.29, 0.82, bbox[1][0] - 0.27, 0.08])
synax.plot(cell.tvec, synapse.i, 'r', lw=1)
synax.set_xticks([0, 5, 10])
synax.set_xticklabels([])
synax.set_title('I')
# synax.set_xlabel('time (ms)')
synax.set_ylabel(r'$i_\mathrm{syn}(t)$ (nA)')

# axes for somatic voltage
vax = fig.add_axes([0.29, 0.70, bbox[1][0] - 0.27, 0.08])
vax.plot(cell.tvec, cell.somav, 'r', lw=1)
vax.set_xticks([0, 5, 10])
vax.set_xticklabels([])
vax.set_title('II')
# synax.set_xlabel('time (ms)')
vax.set_ylabel(r'$V_\mathrm{soma}(t)$ (mV)')


# create axes for extracellular potential
lfpax = fig.add_axes([0.29, 0.58, bbox[1][0] - 0.27, 0.08])
lfpax.plot(cell.tvec, electrode.data[(
    electrode.x == -80) & (electrode.z == 30), ].ravel() * 1E3, 'b', lw=1)
lfpax.set_xticks([0, 5, 10])
lfpax.set_xticklabels([])
lfpax.set_title('III')
# lfpax.set_xlabel('time (ms)')
lfpax.set_ylabel(r'$\phi(\mathbf{r}, t)$ ($\mu$V)')

# create axes for current dipole moment
pax = fig.add_axes([0.29, 0.46, bbox[1][0] - 0.27, 0.08])
for i, x in enumerate(current_dipole_moment.data[::2, :]):
    pax.plot(
        cell.tvec,
        x *
        1E-3,
        label=r'$\mathbf{\hat{u}=\hat{%s}}$' %
        ('xz'[i]),
        lw=1)  # nA um -> 1E-3 nA m unit conversion
pax.set_xticks([0, 5, 10])
pax.set_xlabel('time (ms)')
pax.set_ylabel(r'$\mathbf{p} \cdot \hat{\mathbf{u}}}$ ($10^{-3}$ nA m)')
pax.set_title('IV')
pax.legend(loc=8, bbox_to_anchor=(0.5, -1.25))


# scale bars
ax0.plot([60, 60], [-80, -70], 'k', lw=1, clip_on=False)
ax0.text(62, -80, r'$10 \mu$m', fontsize=12)

# axis cross
ax0.annotate("", xy=(-90, -80),
             xycoords='data', xytext=(-70, -80), textcoords='data',
             arrowprops=dict(arrowstyle="<|-",
                             connectionstyle="arc3,rad=0", facecolor='black'))
ax0.annotate("", xy=(-90, -80),
             xycoords='data', xytext=(-90, -60), textcoords='data',
             arrowprops=dict(arrowstyle="<|-",
                             connectionstyle="arc3,rad=0", facecolor='black'))
ax0.text(-70, -90, 'x', ha='right')
ax0.text(-100, -70, 'z')


# Plot dipole moment potential at t=|i_syn|_max
# t_max = abs(synapse.i) == abs(synapse.i).max()
vmax = 5.
im = ax1.pcolormesh(X_p, Z_p, phi_p[t_max, ].reshape(X_p.shape) * 1E6,
                    cmap=plt.get_cmap('PRGn', 51), zorder=-10,
                    vmin=-vmax,
                    vmax=vmax,
                    rasterized=True,
                    shading='auto')
_ = ax1.pcolormesh(X_p, Z_p, LFP_p.T[t_max, ].reshape(X_p.shape) * 1E6,
                   cmap=plt.get_cmap('PRGn', 51), zorder=-10,
                   vmin=-vmax,
                   vmax=vmax,
                   rasterized=True,
                   shading='auto')
# draw circle between "close field" and "far field"
phi = np.linspace(0, 2 * np.pi, 37)
r = 500
x = r * np.cos(phi) + R_cell[0]
y = r * np.sin(phi) + R_cell[2]
ax1.plot(x, y, ':w', lw=2)
ax1.text(
    R_cell[0],
    r + R_cell[2] + 10,
    r'$\phi_\mathbf{p}(\mathbf{r}, t)$',
    ha='center',
    va='bottom',
    color='w')
ax1.text(
    R_cell[0],
    r + R_cell[2] - 10,
    r'$\phi(\mathbf{r}, t)$',
    ha='center',
    va='top',
    color='w')

bbox = np.array(ax1.get_position())
cax = fig.add_axes([bbox[0][0] + (bbox[1][0] - bbox[0][0]) / 4,
                    bbox[0][1], (bbox[1][0] - bbox[0][0]) / 2, 0.015])
axcb = fig.colorbar(im, cax=cax, orientation='horizontal')
axcb.outline.set_visible(False)
axcb.set_label(r'$\phi$ (nV)', labelpad=0)
axcb.set_ticks([-vmax, 0, vmax])

# scale bars
ax1.plot([800, 800], [-1650, -1550], 'k', lw=1, clip_on=False)
ax1.text(820, -1650, r'$100 \mu$m', fontsize=12)

# axis cross
ax1.annotate("", xy=(-1100, -1600), xycoords='data',
             xytext=(-900, -1600), textcoords='data',
             arrowprops=dict(arrowstyle="<|-", connectionstyle="arc3,rad=0",
                             facecolor='black'))
ax1.annotate("", xy=(-1100, -1600), xycoords='data', xytext=(-1100, -1400),
             textcoords='data',
             arrowprops=dict(arrowstyle="<|-", connectionstyle="arc3,rad=0",
                             facecolor='black'))
ax1.text(-900, -1700, 'x', ha='right')
ax1.text(-1200, -1500, 'z')

# plot arrow representing dipole moment magnitude and direction
insetax = []
for ax in [ax1, ax2]:

    # plot arrow representing dipole moment magnitude and direction
    ax.annotate("",  # r"$\mathbf{p}(I_n(t), \mathbf{r}_n)$",
                xy=(R_cell[0], R_cell[2]),
                xytext=(R_cell[0]
                        + current_dipole_moment.data[:, t_max][0, 0] * 50,
                        R_cell[2]
                        + current_dipole_moment.data[:, t_max][2, 0] * 50),
                arrowprops=dict(arrowstyle="<-", lw=3, color='k',
                                shrinkA=0, shrinkB=0
                                ),
                zorder=100)
    ax.text(R_cell[0] + current_dipole_moment.data[:, t_max][0, 0] * 25 + 10,
            R_cell[2] + current_dipole_moment.data[:, t_max][2, 0] * 25,
            r"$\mathbf{p}$")
    # r"$\mathbf{p}(I_n^{(\mathrm{m})}(t), \mathbf{r}_n)$")

    # plot points where we show signal values
    ax.plot([0 + R_cell[0]], [750 + R_cell[2]], 'o', color='C0')
    ax.plot([-750 + R_cell[0]], [0 + R_cell[2]], 'o', color='C1')
    ax.plot([0 + R_cell[0]], [-750 + R_cell[2]], 'o', color='C2')
    ax.plot([750 + R_cell[0]], [0 + R_cell[2]], 'o', color='C3')

    bbox = np.array(ax.get_position())
    insetax.append(fig.add_axes(
        [bbox[0, 0] + bbox[0, 1] / 4, 0.43, bbox[0, 1] / 4, 0.1]))


# Plot scalar y-component of magnetic field H at t=|i_syn|_max
vmax = 2.
mu = 4 * np.pi * 1E-7
im = ax2.pcolormesh(X_p, Z_p,
                    # mT -> fT unit conversion
                    H[t_max, :, 1].reshape(X_p.shape) * mu * 1E12,
                    cmap=plt.get_cmap('BrBG', 51), zorder=-10,
                    vmin=-vmax,
                    vmax=vmax,
                    rasterized=True,
                    shading='auto')

phi = np.linspace(0, 2 * np.pi, 37)
r = 500
x = r * np.cos(phi) + R_cell[0]
y = r * np.sin(phi) + R_cell[2]
ax2.plot(x, y, ':w', lw=2)
ax2.text(
    R_cell[0],
    r + R_cell[2] + 10,
    r'$\mathbf{B}_\mathbf{p}\cdot\mathbf{\hat{y}}$',
    ha='center',
    va='bottom',
    color='k')
ax2.text(
    R_cell[0],
    r + R_cell[2] - 10,
    r'$\mathbf{B}\cdot\mathbf{\hat{y}}$',
    ha='center',
    va='top',
    color='k')

bbox = np.array(ax2.get_position())
cax = fig.add_axes([bbox[0][0] + (bbox[1][0] - bbox[0][0]) / 4,
                    bbox[0][1], (bbox[1][0] - bbox[0][0]) / 2, 0.015])
axcb = fig.colorbar(im, cax=cax, orientation='horizontal')
axcb.outline.set_visible(False)
axcb.set_label(r'$\mathbf{B}\cdot\hat{\mathbf{y}}$ (fT)', labelpad=0)
axcb.set_ticks([-vmax, 0, vmax])

# scale bars
ax2.plot([800, 800], [-1650, -1550], 'k', lw=1, clip_on=False)
ax2.text(820, -1650, r'$100 \mu$m', fontsize=12)

# axis cross
ax2.annotate("", xy=(-1100, -1600), xycoords='data', xytext=(-900, -1600),
             textcoords='data',
             arrowprops=dict(arrowstyle="<|-", connectionstyle="arc3,rad=0",
                             facecolor='black'))
ax2.annotate("", xy=(-1100, -1600), xycoords='data', xytext=(-1100, -1400),
             textcoords='data',
             arrowprops=dict(arrowstyle="<|-", connectionstyle="arc3,rad=0",
                             facecolor='black'))
ax2.text(-900, -1700, 'x', ha='right')
ax2.text(-1200, -1500, 'z')


# plot in insetaxes the dipole moment potential and magnetic field magnitude
for i, (x, z) in enumerate(zip([R_cell[0], -
                                750 +
                                R_cell[0], R_cell[0], 750 +
                                R_cell[0]], [750 +
                                             R_cell[2], R_cell[2], -
                                             750 +
                                             R_cell[2], R_cell[2]])):
    # ind = ((X_p == x) & (Z_p==z)).flatten()
    ind = (((X_p - x)**2 == ((X_p - x)**2).min()) &
           ((Z_p - z)**2 == ((Z_p - z)**2).min())).flatten()
    insetax[0].plot(cell.tvec, phi_p[:, ind] * 1E6,
                    'C{}'.format(i), lw=1)  # mV->nV
    insetax[1].plot(cell.tvec, H[:, ind, 1] * mu * 1E12,
                    'C{}'.format(i), lw=1)  # mT -> fT

insetax[0].set_ylabel(r'$\phi_\mathbf{p}(t)$ (nV)')
insetax[0].set_xticks([0, 5, 10])
insetax[0].set_xlabel('time (ms)')

insetax[1].set_ylabel(
    r'$\mathbf{B}_\mathbf{p}(t) \cdot \hat{\mathbf{y}}$ (fT)')
insetax[1].set_xticks([0, 5, 10])
insetax[1].set_xlabel('time (ms)')


# panel D. Illustration of 4-sphere volume conductor model geometry
# ax3.set_title('four-sphere volume conductor model')
for direction in ["xzero"]:
    ax3.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    ax3.axis[direction].set_visible(False)


theta = np.linspace(0, np.pi, 31)

# draw some circles:
for i, r, label in zip(range(4), foursphereParams['radii'], [
                       'brain', 'CSF', 'skull', 'scalp']):
    ax3.plot(
        np.cos(theta) *
        r,
        np.sin(theta) *
        r,
        'C{}'.format(i),
        label=label +
        r', $r_%i=%i$ mm' %
        (i +
         1,
         r /
         1000),
        clip_on=False)

# draw measurement points
ax3.plot(foursphereParams['r_electrodes'][:, 0],
         foursphereParams['r_electrodes'][:, 2], 'ko', label='EEG/MEG sites')
for i, (x, y, z) in enumerate(foursphereParams['r_electrodes']):
    # theta = np.arcsin(x / foursphereParams['radii'][-1])
    # if x >= 0:
    #     ax3.text(x, z+5000, r'${}\pi$'.format(theta / np.pi))
    # else:
    #     ax3.text(x, z+5000, r'${}\pi$'.format(theta / np.pi), ha='right')
    ax3.text(x, z + 2500, r'{}'.format(i + 1), ha='center')

# dipole location
ax3.plot([0], [dipole_position[-1]], 'k.', label='dipole site')
ax3.axis('equal')
ax3.set_xticks(np.r_[-np.array(foursphereParams['radii']),
                     0, foursphereParams['radii']])
ax3.set_xticklabels([])
ax3.legend(loc=(0.25, 0.15), frameon=False)


# four-sphere volume conductor
sphere = LFPy.FourSphereVolumeConductor(
    **foursphereParams
)
phi_p = sphere.get_dipole_potential(current_dipole_moment.data,
                                    dipole_location=dipole_position)

# import example_parallel_network_plotting as plotting
vlimround = draw_lineplot(ax=ax4,
                          data=phi_p * 1E9,  # mV -> pV unit conversion
                          unit=r'pV',
                          dt=cell.dt, ztransform=False,
                          T=(0, cell.tstop), color='k', scalebarbasis='log10')
# ax4.set_xticklabels([])
ax4.set_yticklabels([r'{}'.format(i + 1) for i in range(phi_p.shape[0])])
ax4.set_ylabel('position')
ax4.set_xlabel('time (ms)')


# 90 deg rotation matrices around x-, y- and z-axis
Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# compute the radial unit vector from the center of the sphere to each
# measurement point, then unit vectors along theta and phi
r_hat = (sphere.rxyz.T / sphere.r).T
theta = np.arccos(sphere.rxyz[:, 2] / sphere.r)
phi = np.arctan2(sphere.rxyz[:, 1], sphere.rxyz[:, 0])
theta_hat = np.array([np.cos(theta) * np.cos(phi),
                      np.cos(theta) * np.sin(phi),
                      -np.sin(phi)]).T
phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(r_hat.shape[0])]).T


ax5.set_title(
    r"scalp magnetic field "
    + r"$\mathbf{B}_\mathbf{p}(\mathbf{r}) \cdot \hat{\mathbf{\varphi}}$")
# radial component of H at squid locations
# create MEG object and compute magnetic field
meg = LFPy.MEG(sensor_locations=foursphereParams['r_electrodes'])
H = meg.calculate_H(current_dipole_moment.data, dipole_position)

H_phi = np.zeros(phi_p.shape)
for j, (h, u) in enumerate(zip(H, phi_hat)):
    H_phi[j, ] += h.T @ u


vlimround = draw_lineplot(
    ax=ax5,
    data=H_phi * meg.mu * 1E12,  # mT -> fT unit conv.
    dt=cell.dt, unit=r'fT',
    ztransform=False,
    label=r'$\mathbf{B}_\mathbf{p}(\mathbf{r}) \cdot \hat{\mathbf{\varphi}}$',
    T=(0, cell.tstop), color='k', scalebarbasis='log10')

ax5.set_yticklabels([r'{}'.format(i + 1) for i in range(phi_p.shape[0])])
ax5.set_xlabel('time (ms)', labelpad=0)
ax5.set_ylabel('')


for i, ax in enumerate([ax0, ax1, ax2]):
    ax.text(-0.05, 1.05, alphabet[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)
for i, ax in enumerate([ax3, ax4, ax5]):
    ax.text(-0.05, 1.1, alphabet[i + 3],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='demibold',
            transform=ax.transAxes)


fig.savefig('figure_2.pdf', bbox_inches='tight', dpi=300)
plt.show()
