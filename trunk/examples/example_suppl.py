#!/usr/bin/env python
'''some plottin' functions used by the example scripts'''
import numpy as np
import matplotlib.pyplot as plt
import neuron


def plot_ex1(cell, electrode, X, Y, Z):
    '''
    plot the morphology and LFP contours, synaptic current and soma trace
    '''
    #some plot parameters
    t_show = 30 #time point to show LFP
    tidx = np.where(cell.tvec == t_show)
    #contour lines:
    n_contours = 200
    n_contours_black = 40
    
    #This is the extracellular potential, reshaped to the X, Z mesh
    LFP = np.arcsinh(electrode.LFP[:, tidx]).reshape(X.shape)
    
    #figure object
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
        wspace=0.4, hspace=0.4)
    
    # Plot LFP around the cell with in color and with equipotential lines
    ax1 = fig.add_subplot(121, aspect='equal', frameon=False)
    
    #plot_morphology(plot_synapses=True)
    for sec in neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color='k')
    for i in range(len(cell.synapses)):
        ax1.plot([cell.synapses[i].x], [cell.synapses[i].z],
            color=cell.synapses[i].color, marker=cell.synapses[i].marker, 
            markersize=10)
    
    #contour lines
    ct1 = ax1.contourf(X, Z, LFP, n_contours)
    ct1.set_clim((-0.00007, 0.00002))
    ct2 = ax1.contour(X, Z, LFP, n_contours_black, colors='k')
    
    # Plot synaptic input current
    ax2 = fig.add_subplot(222)
    ax2.plot(cell.tvec, cell.synapses[0].i)
    
    # Plot soma potential
    ax3 = fig.add_subplot(224)
    ax3.plot(cell.tvec, cell.somav)
    
    # Figure formatting and labels
    fig.suptitle('example 1', fontsize=14)
    
    ax1.set_title('LFP at t=' + str(t_show) + ' ms', fontsize=12)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    
    ax2.set_title('synaptic input current', fontsize=12)
    ax2.set_ylabel('(nA)')
    ax2.set_xlabel('time (ms)')
    
    ax3.set_title('somatic membrane potential', fontsize=12)
    ax3.set_ylabel('(mV)')
    ax3.set_xlabel('time (ms)')
    
    return fig


def plot_ex2(cell, electrode):
    '''example2.py plotting function'''
    #creating array of points and corresponding diameters along structure
    for i in range(cell.xend.size):
        if i == 0:
            xcoords = np.array([cell.xmid[i]])
            ycoords = np.array([cell.ymid[i]])
            zcoords = np.array([cell.zmid[i]])
            diams = np.array([cell.diam[i]])    
        else:
            if cell.zmid[i] < 100 and cell.zmid[i] > -100 and \
                    cell.xmid[i] < 100 and cell.xmid[i] > -100:
                xcoords = np.r_[xcoords, np.linspace(cell.xstart[i],
                                            cell.xend[i], cell.length[i]*3)]   
                ycoords = np.r_[ycoords, np.linspace(cell.ystart[i],
                                            cell.yend[i], cell.length[i]*3)]   
                zcoords = np.r_[zcoords, np.linspace(cell.zstart[i],
                                            cell.zend[i], cell.length[i]*3)]   
                diams = np.r_[diams, np.linspace(cell.diam[i], cell.diam[i],
                                            cell.length[i]*3)]
    
    #sort along depth-axis
    argsort = np.argsort(ycoords)
    
    #plotting
    fig = plt.figure(figsize=[15, 10])
    ax = fig.add_axes([0.1, 0.1, 0.533334, 0.8], frameon=False)
    ax.scatter(xcoords[argsort], zcoords[argsort], s=diams[argsort]**2*20,
               c=ycoords[argsort], edgecolors='none', cmap='gray')
    ax.plot(electrode.x, electrode.z, '.', marker='o', markersize=5, color='k')
    
    i = 0
    limLFP = abs(electrode.LFP).max()
    for LFP in electrode.LFP:
        tvec = cell.tvec*0.6 + electrode.x[i] + 2
        if abs(LFP).max() >= 1:
            factor = 2
            color='r'
        elif abs(LFP).max() < 0.25:
            factor = 50
            color = 'b'
        else:
            factor = 10
            color = 'g'
        trace = LFP*factor + electrode.z[i]
        ax.plot(tvec, trace, color=color, lw = 2)
        i += 1
    
    ax.plot([22, 28], [-60, -60], color='k', lw = 3)
    ax.text(22, -65, '10 ms')
    
    ax.plot([40, 50], [-60, -60], color='k', lw = 3)
    ax.text(42, -65, '10 $\mu$m')
    
    ax.plot([60, 60], [20, 30], color='r', lw=2)
    ax.text(62, 20, '5 mV')
    
    ax.plot([60, 60], [0, 10], color='g', lw=2)
    ax.text(62, 0, '1 mV')
    
    ax.plot([60, 60], [-20, -10], color='b', lw=2)
    ax.text(62, -20, '0.1 mV')
    
    
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.axis([-61, 61, -61, 61])
    
    ax.set_title('Location-dependent extracellular spike shapes')
    
    #plotting the soma trace    
    ax = fig.add_axes([0.75, 0.55, 0.2, 0.35])
    ax.plot(cell.tvec, cell.somav)
    ax.set_title('Somatic action-potential')
    ax.set_ylabel(r'$V_\mathrm{membrane}$ (mV)')
    
    #plotting the synaptic current
    ax = fig.add_axes([0.75, 0.1, 0.2, 0.35])
    ax.plot(cell.tvec, cell.synapses[0].i)
    ax.set_title('Synaptic current')
    ax.set_ylabel(r'$i_\mathrm{synapse}$ (nA)')
    ax.set_xlabel(r'time (ms)')

    return fig



def plot_ex3(cell, electrode):
    '''plotting function used by example3/4'''
    fig = plt.figure(figsize=[12, 8])
    
    #plot the somatic trace
    ax = fig.add_axes([0.1, 0.7, 0.5, 0.2])
    ax.plot(cell.tvec, cell.somav)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Soma pot. [mV]')
    
    #plot the synaptic current
    ax = fig.add_axes([0.1, 0.4, 0.5, 0.2])
    for i in range(len(cell.synapses)):
        ax.plot(cell.tvec, cell.synapses[i].i, color=cell.synapses[i].color)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Syn. i [nA]')
    
    #plot the LFP as image plot
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.2])
    absmaxLFP = abs(np.array([electrode.LFP.max(), electrode.LFP.min()])).max()
    im = ax.pcolormesh(cell.tvec, electrode.z, electrode.LFP,
                  vmax=absmaxLFP, vmin=-absmaxLFP,
           cmap='spectral_r')
    
    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.02
    cax = fig.add_axes(rect)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('LFP (mV)')
    ax.axis(ax.axis('tight'))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('z [$\mu$m]')
    
    #plot the morphology, electrode contacts and synapses
    ax = fig.add_axes([0.65, 0.1, 0.25, 0.8], frameon=False)
    for sec in neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color='k')
    for i in range(len(cell.synapses)):
        ax.plot([cell.synapses[i].x], [cell.synapses[i].z], \
            color=cell.synapses[i].color, marker=cell.synapses[i].marker)
    for i in range(electrode.x.size):
        ax.plot(electrode.x[i], electrode.z[i], color='g', marker='o')
    plt.axis('equal')
    plt.axis(np.array(plt.axis())*0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

