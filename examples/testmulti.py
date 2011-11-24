#!/usr/bin/env python
import LFPy
from LFPy import Electrode, lfpcalc
import pylab as pl
from multiprocessing import Process, Queue, freeze_support, cpu_count


class ElectrodeThreaded(Electrode):
    '''Inherit class Electrode, but using multiprocessing to distribute
    calculations of LFP for each electrode contact'''
    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 N=None, r=None, n=0, r_z=None, colors=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None):
        '''Initialization of class ElectrodeThreaded, with electrode setup
        inherited from class ElectrodeSetup'''
        Electrode.__init__(self, cell, sigma, x, y, z,
                                N, r, n, r_z, perCellLFP,
                                method, color, marker, from_file, cellfile)
    
    def _lfp_el_pos_calc_dist(self, c, r_limit, sigma=0.3, radius=10, n=10,
                             m=50, N=None, t_indices=None, 
                             method='linesource'):
        '''
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface with radius r. The locations of these n points on
        the electrode surface are random,  within the given radius'''
        lfp_el_pos = pl.zeros(self.LFP.shape)
        offsets = {}
        circle = {}
        
        if __name__ == '__main__':
            freeze_support()
            NUMBER_OF_PROCESSES = cpu_count()
            task_queue = Queue()
            done_queue = Queue()

            TASKS = pl.arange(len(self.x))
            
            for task in TASKS:
                task_queue.put(int(task))       
            for i in xrange(NUMBER_OF_PROCESSES):
                Process(target=self._lfp_el_pos_calc_dist_i,
                             args=(task_queue,
                             c, i, r_limit, sigma, radius, n,
                             m, N, t_indices, method,
                             done_queue)).start()
            for n in xrange(TASKS.size):
                [i, lfp, offset, circle] = done_queue.get() 
                lfp_el_pos[i], offsets[i], circle[i] = lfp, offset, circle                
            for i in xrange(NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            
            task_queue.close()
            done_queue.close()
        else:
            raise Exception, "'__name__' != '__main__'"
                
        return circle,  offsets,  lfp_el_pos
        
            
    def _lfp_el_pos_calc_dist_i(self, task_queue,
                    c, i, r_limit, sigma, radius, n, m, N, t_indices, method,
                    done_queue):       
        
        for nnn in iter(task_queue.get, 'STOP'):
            if n > 1:
                lfp_e = pl.zeros((n, self.LFP.shape[1]))
    
                offs = pl.zeros((n, 3))
                r2 = pl.zeros(n)
    
                crcl = pl.zeros((m, 3))
    
                for j in xrange(n):
                    A = [(pl.rand()-0.5)*radius*2, (pl.rand()-0.5)*radius*2,
                            (pl.rand()-0.5)*radius*2]
                    offs[j, ] = pl.cross(N[i, ], A)
                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
                    while r2[j] > radius**2:
                        A = [(pl.rand()-0.5)*radius*2, (pl.rand()-0.5)*radius*2,
                            (pl.rand()-0.5)*radius*2]
                        offs[j, ] = pl.cross(N[i, ], A)
                        r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
    
                x_n = offs[:, 0] + self.x[i]
                y_n = offs[:, 1] + self.y[i]
                z_n = offs[:, 2] + self.z[i]
    
                for j in xrange(m):
                    B = [(pl.rand()-0.5), (pl.rand()-0.5), (pl.rand()-0.5)]
                    crcl[j, ] = pl.cross(N[i, ], B)
                    crcl[j, ] = crcl[j, ]/pl.sqrt(crcl[j, 0]**2 +
                                               crcl[j, 1]**2 + \
                                               crcl[j, 2]**2)*radius
    
                crclx = crcl[:, 0] + self.x[i]
                crcly = crcl[:, 1] + self.y[i]
                crclz = crcl[:, 2] + self.z[i]
    
                for j in xrange(n):
                    variables = {
                        'x' : x_n[j],
                        'y' : y_n[j],
                        'z' : z_n[j],
                        'r_limit' : r_limit,
                        'sigma' : sigma,
                        't_indices' : t_indices,
                        'method' : method,
                    }
                    lfp_e[j, ] = lfpcalc.calc_lfp_choose(c, **variables)
    
                lfp_el_pos = lfp_e.mean(axis=0)
    
            else:
                lfp_el_pos = lfpcalc.calc_lfp_choose(c,
                    x=self.x[i], y=self.y[i], z=self.z[i], r_limit = r_limit,
                    sigma=sigma, t_indices=t_indices)
            offsets = {
                'x_n' : x_n,
                'y_n' : y_n,
                'z_n' : z_n
            }
            circle = {
                'x' : crclx,
                'y' : crcly,
                'z' : crclz
            }
            done_queue.put([i, lfp_el_pos, offsets, circle])





pl.interactive(1)

cellParameters = {                          
    'morphology' : 'L5_Mainen96_LFPy.hoc',  # morphology file
    'rm' : 30000,                           # membrane resistivity
    'cm' : 1.0,                             # membrane capacitance
    'Ra' : 150,                             # axial resistivity
    'timeres_NEURON' : 0.1,                # dt for NEURON sim.
    'timeres_python' : 0.1,                 # dt for python output
    'tstartms' : -50,                         # start t of simulation
    'tstopms' : 50,                        # end t of simulation
}

cell = LFPy.Cell(**cellParameters)

synapseParameters = {
    'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
    'e' : 0,                                # reversal potential
    'syntype' : 'ExpSyn',                   # synapse type
    'tau' : 2,                              # syn. time constant
    'weight' : 0.01,                       # syn. weight
    'record_current' : True                 # syn. current record
}

synapse = LFPy.PointProcessSynapse(cell, **synapseParameters)
synapse.set_spike_times(cell, pl.array([10, 15, 20, 25]))

cell.simulate(rec_isyn=True)                # run cell simulation

N = pl.empty((24, 3))
for i in xrange(N.shape[0]):
    N[i,] = [1, 0, 0] #normal unit vec. to contacts
electrodeParameters = {
    'sigma' : 0.3,                          # conductivity
    'x' : pl.zeros(24) + 50,                # carthesian coords.
    'y' : pl.zeros(24),                     # of electrode points
    'z' : pl.arange(-200, 1000, 50),
    'n' : 5,
    'r' : 10,
    'N' : N,
}


elserial = LFPy.Electrode(cell, **electrodeParameters)
elserial.calc_lfp()               # run LFP simulation

electrode = ElectrodeThreaded(cell, **electrodeParameters)
electrode.calc_lfp()               # run LFP simulation


def plotstuff():
    fig = pl.figure()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    pl.subplot(321)
    pl.plot(cell.tvec,cell.somav)
    pl.axis('tight')
    #pl.xlabel('Time (ms)')
    pl.ylabel(r'$V_\mathrm{soma}$ (mV)')
    pl.title('Somatic Potential')

    pl.subplot(323)
    for i in xrange(len(cell.synapses)):
        pl.plot(cell.tvec,cell.synapses[i].i,color=cell.synapses[i].color)
    pl.axis('tight')
    #pl.xlabel('Time (ms)')
    pl.ylabel(r'$i_\mathrm{syn.}$(nA)')
    pl.title('Synaptic Current')


    pl.subplot(325)
    absmaxLFP = abs(pl.array([electrode.LFP.max(),electrode.LFP.min()])).max()
    pl.imshow(electrode.LFP*1000,vmax=absmaxLFP*1000,vmin=-absmaxLFP*1000,origin='lower',
           extent=(cell.tvec[0],cell.tvec[-1],electrode.z[0],electrode.z[-1]),cmap='jet_r',
           interpolation='nearest')
    pl.xlabel('Time (ms)')
    pl.ylabel('z ($\mu$m)')
    pl.title('Extracellular Potential')
    pl.axis('tight')
    cb = pl.colorbar()
    cb.set_label('LFP ($\mu$V)')
    
    pl.subplot(122)
    for i in xrange(cell.xend.size):
        pl.plot([cell.xstart[i],cell.xend[i]],[cell.zstart[i],cell.zend[i]],color='k')
    for i in xrange(len(cell.synapses)):
        pl.plot([cell.synapses[i].x],[cell.synapses[i].z],\
            color=cell.synapses[i].color,marker=cell.synapses[i].marker, markersize=10)
    for i in xrange(electrode.x.size):
        pl.plot(electrode.x[i],electrode.z[i],color='g',marker='o')
    pl.axis('equal')
    pl.title('Morphology')
    pl.xlabel(r'x ($\mu$m)')
    pl.ylabel(r'z ($\mu$m)')


#plotstuff()
