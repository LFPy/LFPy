#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import pylab as pl
import sys
from LFPy import lfpcalc, tools
from multiprocessing import Process, Queue, freeze_support, cpu_count

class ElectrodeSetup(object):
    '''Main electrode class used in LFPy.
    The class takes dict.electrode geometries as input, or may produce itself
    by means of calling functions self.el_pos*(). It also takes LFPy.cell.Cell
    object or dictionaries/lists containing these as input,  i.e.

    Example script (electrode "similar" to Thomas Recording tetrode);
    ::
        from pylab import *
        import LFPy
    
        morphology = 'L5_Mainen96_LFPy.hoc'
        electrode = {
            'sigma' : 0.3   # infinite homogenous extracellular conductivity
            'x' : pl.array([0, -1, pl.sin(pl.pi/6), pl.sin(pl.pi/6)])*25.,
            'y' : pl.array([0, 0, -pl.cos(pl.pi/6), pl.cos(pl.pi/6)])*25.,
            'z' : pl.array([-50., 0, 0, 0]),   # x, y, z-coordinates
            'color' : 'g',          #plots color,  opt.
            'marker' : 'o',                 #plot marker,  opt.
            'tetr_r' : 25.,                 #optional,  for plotting
            'N' : pl.array([    [0, 0, -1],
                                [-1*pl.cos(pl.pi/9), 0, -1*pl.sin(pl.pi/9)],
                                [pl.sin(pl.pi/6)*pl.cos(pl.pi/9), \
                                    -pl.cos(pl.pi/9)*pl.cos(pl.pi/9), \
                                    -1*pl.sin(pl.pi/9)],
                                [-pl.sin(pl.pi/6)*pl.cos(pl.pi/9), \
                                    -pl.cos(pl.pi/9)*pl.cos(pl.pi/9), \
                                    1*pl.sin(pl.pi/9)]]),
                            # opt. electrode contact surface normals
            'r' : 7.,       # opt. electrode contact radius
            'n' : 20,       # opt. n random points on electrode contact for avrg.
            'r_z': pl.array([[-1E199, -50.0001, -50, 75, 1E99], [0, 0, 7, 48, 48]]),
            # opt. radius of electrode as function of z
            'colors' : ['r', 'g', 'b', 'y'] #opt. color of electrode points in plots
            'perCellLFP' : False, #opt storing LFP-contrib from all cells
            'from_file' : False, #option for importing cpickled cell objects
            'cellfile' : None, #path to file with cell object(s)
        }
        c = LFPy.Cell(morphology=morphology)
        s = LFPy.Synapse(c, idx = 0)
        s.set_spike_times(c, sptimes=pl.array([10, 20]))
    
        c.simulate(tstopms=50, rec_v=True)
    
        el = LFPy.Electrode(c, **electrode)
        el.calc_lfp_threaded()
        LFP_dist = el.LFP
    
        el['N']
        el['LFP']
        el.calc_lfp_threaded()
        LFP_point = el.LFP
    
        LFP_diff = LFP_point - LFP_dist
    
        figure()
        for i in xrange(el.x.size):
            plot(c.tvec, LFP_dist[i, ], label='dist,  el. %i' % i)
        for i in xrange(el.x.size):
            plot(c.tvec, LFP_point[i, ], linestyle='--', label='point,  el. %i' % i)
        legend()
    
        figure()
        for i in xrange(el.x.size):
            plot(c.tvec, LFP_diff[i, ],  label='el. %i' % i)
        legend()
    '''
    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 color='g', marker='o',
                 N=None, r=None, n=0, r_z=None, colors=None,
                 perCellLFP=False, method='linesource', 
                 from_file=False, cellfile=None, verbose=False):
        '''Initialize class Electrode'''
        self.sigma = sigma
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.marker = marker
        if N != None:
            if N.shape[-1] == 3:
                self.N = N
            else:
                self.N = N.T
                if N.shape[-1] != 3:
                    raise Exception, 'N.shape must be (n contacts, 1, 3)!'
        else:
            self.N = N
            
        self.r = r
        self.n = n
        self.r_z = r_z
        self.colors = colors
        self.perCellLFP = perCellLFP
        
        self.method = method
        self.verbose = verbose

        if from_file:
            if type(cellfile) == type(str()):
                cell = tools.load(cellfile)
            elif type(cellfile) == type([]):
                cell = []
                for fil in cellfile:
                    cell.append(tools.load(fil))
            else:
                raise ValueError, 'cell either string or list of strings'

        self._import_c(cell)
        
                
        #test that currents sum towards zero        
        try:
            self._test_imem_sum()
        except:
            pass
        

    class cell():
        '''empty object that cell-specific variables are stored in'''
        def __init__(self):
            '''just some empty class'''
            pass

    def _import_c(self, cell):
        '''Keeps the relevant variables for LFP-calculation from cell'''
        #keeping these variables:
        variables = [
            'somav',
            'somaidx',
            'tvec',
            'timeres_python',
            'imem',
            'diam',
            'xstart',
            'xmid',
            'xend',
            'ystart',
            'ymid',
            'yend',
            'zstart',
            'zmid',
            'zend',
            'totnsegs',
            'synapses',
        ]
        
        #redefine list of cells as dict of cells
        if type(cell) == list:
            k = 0
            celldict = {}
            for c in cell:
                celldict[k] = c
                k += 1
            cell = celldict
        
        self.c = {}
        if str(type(cell))[:12] == "<class 'LFPy.cell.Cell'>"[:12] or \
                    str(type(cell))[:12] == "<class 'cell.Cell'>"[:12]:
            self.c[0] = self.cell()
            for v in variables:
                setattr(self.c[0], v, getattr(cell, v))
        elif type(cell) == dict:
            for k in cell:
                if str(type(cell[k]))[:12] != "<class 'LFPy.cell.Cell'>"[:12]:
                    raise Exception,  "Error! <type(cell[%s])> something else \
                    than <LFPy.cell.Cell object>" % str(k)
                self.c[k] = self.cell()
                for v in variables:
                    setattr(self.c[k], v, getattr(cell[k], v))
        else:
            raise Exception, \
            "Error! <type(cell)> something else than <LFPy.cell.Cell object> \
            or <dict>"

        setattr(self, 'tvec', self.c[self.c.keys()[0]].tvec)
        setattr(self, 'dt', self.c[self.c.keys()[0]].timeres_python)
        
        self.nCells = pl.array(self.c.keys()).size
    
    def _test_imem_sum(self, tolerance=1E-12):
        '''test that the membrane currents sum to zero'''
        for k in self.c:
            sum_imem = self.c[k].imem.sum(axis=0)
            if abs(sum_imem).max() >= tolerance:
                print 'Membrane currents do not sum towards zero! They should!'
                [inds] = pl.where((abs(sum_imem) >= tolerance))
                for i in inds:
                    print 'membrane current sum cell %i, timestep %i: %.3e' \
                        % (k, i, sum_imem[i])
            else:
                pass

class ElectrodeThreaded(ElectrodeSetup):
    '''This is an electrode implementation that uses Python multiprocessing to
    distribute the LFP calculations on different cpu-cores'''
    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 color='g', marker='o',
                 N=None, r=None, n=0, r_z=None, colors=None,
                 perCellLFP=False, method='linesource', 
                 from_file=False, cellfile=None):
        '''Initialization of class ElectrodeThreaded, with electrode setup
        inherited from class ElectrodeSetup'''
        ElectrodeSetup.__init__(self, cell, sigma, x, y, z, color, marker,
                                 N, r, n, r_z, colors, perCellLFP, method,
                                 from_file, cellfile)
            
    def calc_lfp_threaded(self, t_indices=None, __name__='__main__',
                          NUMBER_OF_PROCESSES=None, ):
        '''Calculate LFP on electrode geometry from all cell instances.
        Will choose distributed calculated if electrode contain 'n', 'N', and
        'r'. Will try to distribute jobs to available processes'''
        #create self.LFP if not already there
        if hasattr(self, 'LFP'):
            del self.LFP
            print 'deleted electrode.LFP, consecutive calls to calc_lfp_th...!'
        if not hasattr(self,  'LFP'):
            if t_indices != None:
                if self.perCellLFP:
                    LFP_temp = pl.zeros((self.nCells, self.x.size,
                                         t_indices.size))
                else:
                    LFP_temp = pl.zeros((self.nCells, self.x.size,
                                         t_indices.size))
                self.LFP = pl.zeros((self.x.size, t_indices.size))
            else:
                if self.perCellLFP:
                    LFP_temp = pl.zeros((self.nCells, self.x.size,
                                         self.tvec.size))
                else:
                    LFP_temp = pl.zeros((self.nCells, self.x.size,
                                         self.tvec.size))
                self.LFP = pl.zeros((self.x.size, self.tvec.size))
        
        #multiprocess initialization
        if __name__ == '__main__':
            freeze_support()
            if NUMBER_OF_PROCESSES == None:
                NUMBER_OF_PROCESSES = cpu_count() #Number of threads
            #Creating multiproc. queues for tasks and results
            task_queue = Queue()
            done_queue = Queue()

            #One task per cell instance
            TASKS = pl.array(self.c.keys())

            #Distribute TASKS, call Process, fetching results for LFP-calc.
            #and stopping and closing threads and queues
            for task in TASKS:
                task_queue.put(task)
            for i in xrange(NUMBER_OF_PROCESSES):
                Process(target=self._calc_lfp_thread,
                    args=(task_queue, t_indices, NUMBER_OF_PROCESSES,
                          done_queue)).start()
            for i in xrange(TASKS.size):
                [self.circle, self.offsets, n, LFP] = done_queue.get()
                LFP_temp[n, :, :] = LFP
            for i in xrange(NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            task_queue.close()
            done_queue.close()
        
        if self.perCellLFP:
            self.CellLFP = LFP_temp
        
        self.LFP = LFP_temp.sum(axis=0)

    def _calc_lfp_thread(self, task_queue, t_indices,
                          NUMBER_OF_PROCESSES, done_queue):
        '''Single thread, calculating LFP from single cell.
        Should be called by calc_lfp_threaded'''
        #local declaration of LFP, which will be saved in done_queue
        if t_indices != None:
            LFP = pl.zeros((self.x.size, t_indices.size))
        else:
            LFP = pl.zeros((self.x.size, self.tvec.size))
        variables = {
            'sigma' : self.sigma,
            't_indices' : t_indices,
            'method' : self.method,
        }

        #iterate over elements in the task queue
        for k in iter(task_queue.get, 'STOP'):
            #choosing spatial averaging for each electrode contact
            if self.n != None and self.N != None and self.r != None:
                variables.update({
                    'r_limit' : self.c[k].diam / 2,
                    'radius' : self.r,
                    'n' : self.n,
                    'N' : self.N,
                    't_indices' : t_indices,
                    'NUMBER_OF_PROCESSES' : NUMBER_OF_PROCESSES,
                    })
                #Calling function which will calculate LFP, distributed
                if sys.version_info < (2, 6, 6): #fix for older Pythons
                    del variables['NUMBER_OF_PROCESSES']
                    [circle, offsets, LFP] = self._lfp_el_pos_calc_dist(
                        self.c[k], **variables)
                else:
                    [circle, offsets, LFP] = \
                    self._lfp_el_pos_calc_dist_threaded(self.c[k], **variables)
                done_queue.put([circle, offsets, k, LFP])
            #case for no averaging
            else:
                variables.update({
                    'r_limit' : self.c[k].diam / 2
                    })
                if sys.version_info < (2, 6, 6): #fix for runs on Stallo
                    for i in xrange(self.x.size):
                        variables.update({
                            'x' : self.x[i],
                            'y' : self.y[i],
                            'z' : self.z[i],
                        })
                        if hasattr(self, 'r_drift'):
                            LFP = pl.zeros(self.LFP.shape[1])
                            for j in xrange(self.LFP.shape[1]):
                                variables.update({
                                    'x' : self.x[i] + self.r_drift['x'][i],
                                    'y' : self.y[i] + self.r_drift['y'][i],
                                    'z' : self.z[i] + self.r_drift['z'][i],
                                    'timestep' : j
                                })
                                LFP[i, j] = lfpcalc.calc_lfp_choose(self.c[k], 
                                                                    **variables)
                        else:
                            LFP[i, ] = lfpcalc.calc_lfp_choose(self.c[k], 
                                                               **variables)
                else:
                    LFP = self._calc_lfp_simple_threaded(self.c[k], LFP, 
                                                          **variables)
                circle = None,
                offsets = None
                
                done_queue.put([circle, offsets, k, LFP])
            if self.verbose:
                print 'Calculated potential contribution, cell %i.' % k
        
    
    def _calc_lfp_simple_threaded(self, c, LFP, **variables):

        __name__='__main__'
        if __name__ == '__main__':              # This is important, apparently
            output = {}                         # results saved in dict
            freeze_support()                    # let NUMBER_OF_PROCESSES calls
                                                # finish before new jobs are qd
            NUMBER_OF_PROCESSES = cpu_count()   # Using all available cores
            task_queue = Queue()                # queue for tasks
            done_queue = Queue()                # simulation results put here
            
            TASKS = pl.arange(self.x.size)
            
            for task in TASKS:
                task_queue.put(int(task))       
            for i in xrange(NUMBER_OF_PROCESSES):
                Process(target=self._calc_lfp_simple_thread,
                        args=(task_queue, done_queue, c, variables)).start()
            for n in xrange(TASKS.size):
                [i, lfp] = done_queue.get()
                LFP[i, ] = lfp
            for i in xrange(NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            
            task_queue.close()      # No more jobs can be sent to the queues,
            done_queue.close()      # necessary for consecutive parallel jobs
            
            return LFP
    
    def _calc_lfp_simple_thread(self, task_queue,
                                 done_queue, c, variables):
        for i in iter(task_queue.get, 'STOP'):
            variables.update({
                        'x' : self.x[i],
                        'y' : self.y[i],
                        'z' : self.z[i],
                    })
            if hasattr(self, 'r_drift'):
                LFP = pl.zeros(self.LFP.shape[1])
                for j in xrange(self.LFP.shape[1]):
                    variables.update({
                        'x' : self.x[i] + self.r_drift['x'][i],
                        'y' : self.y[i] + self.r_drift['y'][i],
                        'z' : self.z[i] + self.r_drift['z'][i],
                        'timestep' : j
                    })
                    LFP[j] = lfpcalc.calc_lfp_choose(c, **variables)
            else:
                LFP = lfpcalc.calc_lfp_choose(c, **variables)
                
            done_queue.put([i, LFP])
    
    def _lfp_el_pos_calc_dist_threaded(self, c, r_limit, sigma=0.3, radius=10,
                n=10, m=50, N=None, t_indices=None, NUMBER_OF_PROCESSES=None, 
                method='linesource',
                __name__='__main__'):
        '''Calc. of LFP over an n-point integral approximation over flat
        electrode surface with radius r. The locations of these n points on
        the electrode surface are random,  within the given radius.
        The method will spawn a few multiprocessing threads'''
        offsets = {}
        circle = {}

        if __name__ == '__main__':
            freeze_support()
            if NUMBER_OF_PROCESSES == None:
                NUMBER_OF_PROCESSES = cpu_count()
            dist_task_queue = Queue()
            dist_done_queue = Queue()

            TASKS = pl.arange(self.x.size)
            tempLFP = pl.zeros((self.x.size, self.LFP.shape[0],
                                self.LFP.shape[1]))
            
            for task in TASKS:
                dist_task_queue.put(task)
            for i in xrange(NUMBER_OF_PROCESSES):
                Process(target=self._lfp_el_pos_calc_dist_thread,
                    args=(dist_task_queue, c, r_limit, sigma, radius, n, m, N,
                    t_indices, method, dist_done_queue)).start()
            for i in xrange(TASKS.size):
                [circl, offset, tempLFP[i,]] = dist_done_queue.get()
                if type(circl) == type({}) and type(offsets) == type({}):
                    circle.update(circl)
                    offsets.update(offset)
            for i in xrange(NUMBER_OF_PROCESSES):
                dist_task_queue.put('STOP')

            dist_task_queue.close()
            dist_done_queue.close()
        
        lfp_el_pos = tempLFP.sum(axis=0)
        return offsets, circle, lfp_el_pos

    def _lfp_el_pos_calc_dist_thread(self, dist_task_queue, c, r_limit, sigma,
            radius, n, m, N, t_indices, method, dist_done_queue):
        '''spawn thread called by self._lfp_el_pos_calc_dist_threaded()'''
        for i in iter(dist_task_queue.get, 'STOP'):
            lfp_el_pos = pl.zeros(self.LFP.shape)
            offsets = {}
            circle = {}
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
                                               crcl[j, 1]**2 + 
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
                        'method' : self.method,
                    }
                    lfp_e[j, ] = lfpcalc.calc_lfp_choose(c, **variables)

                lfp_el_pos[i] = lfp_e.mean(axis=0)
                offsets[i] = {
                'x_n' : x_n,
                'y_n' : y_n,
                'z_n' : z_n
                }
                circle[i] = {
                'x' : crclx,
                'y' : crcly,
                'z' : crclz
                }
            else:
                lfp_el_pos[i] = lfpcalc.calc_lfp_choose(c, \
                    x=self.x[i], y=self.y[i], z=self.z[i], r_limit = r_limit,
                    sigma=sigma, t_indices=t_indices)
                
            
            dist_done_queue.put([circle, offsets, lfp_el_pos])

class Electrode(ElectrodeSetup):
    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 color='g', marker='o',
                 N=None, r=None, n=0, r_z=None, colors=None,
                 perCellLFP=False, method='linesource', 
                 from_file=False, cellfile=None):
        '''This is the regular implementation of the Electrode class
        that calculates the LFP serially using one core'''
        ElectrodeSetup.__init__(self, cell, sigma, x, y, z, color, marker,
                                N, r, n, r_z, colors, perCellLFP,
                                method, from_file, cellfile)
        
    def calc_lfp(self, t_indices=None):
        '''Calculate LFP on electrode geometry from all cell instances.
        Will chose distributed calculated if electrode contain 'n', 'N', and 'r'
        '''
        for k in self.c.iterkeys():
            if not hasattr(self,  'LFP'):
                if t_indices != None:
                    self.LFP = pl.zeros((self.x.size, t_indices.size))
                else:
                    self.LFP = pl.zeros((self.x.size, self.tvec.size))
            if t_indices != None:
                LFP_temp = pl.zeros((self.nCells, self.x.size, t_indices.size))
            else:
                LFP_temp = pl.zeros((self.nCells, self.x.size, self.tvec.size))
                
            variables = {
                'sigma' : self.sigma,
                't_indices' : t_indices,
                'method' : self.method,
            }
            if self.n != None and self.N != None and self.r != None:
                variables.update({
                    'r_limit' : self.c[k].diam/2,
                    'radius' : self.r,
                    'n' : self.n,
                    'N' : self.N,
                    't_indices' : t_indices,
                    })
                [self.circle, self.offsets, LFP_temp[k, :, :]] = \
                    self._lfp_el_pos_calc_dist(self.c[k], **variables)
            else:
                variables.update({
                    'r_limit' : self.c[k].diam/2
                })
                for i in xrange(self.x.size):
                    variables.update({
                        'x' : self.x[i],
                        'y' : self.y[i],
                        'z' : self.z[i],
                    })
                    if hasattr(self, 'r_drift'):
                        for j in xrange(self.LFP.shape[1]):
                            variables.update({
                                'x' : self.x[i] + self.r_drift['x'][i],
                                'y' : self.y[i] + self.r_drift['y'][i],
                                'z' : self.z[i] + self.r_drift['z'][i],
                                'timestep' : j
                            })
                            LFP_temp[k, i, j] = LFP_temp[k, i, j] + \
                                lfpcalc.calc_lfp_choose(self.c[k], **variables)
                    else:
                        LFP_temp[k, i, ] = LFP_temp[k, i, ] + \
                                lfpcalc.calc_lfp_choose(self.c[k], **variables)

            if self.verbose:
                print 'Calculated potential contribution, cell %i.' % k
        if self.perCellLFP:
            self.CellLFP = LFP_temp
        
        self.LFP = LFP_temp.sum(axis=0)

    def _lfp_el_pos_calc_dist(self, c, r_limit, sigma=0.3, radius=10, n=10,
                             m=50, N=None, t_indices=None, 
                             method='linesource'):
        '''Calc. of LFP over an n-point integral approximation over flat
        electrode surface with radius r. The locations of these n points on
        the electrode surface are random,  within the given radius'''
        lfp_el_pos = pl.zeros(self.LFP.shape)
        offsets = {}
        circle = {}
        for i in xrange(len(self.x)):
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

                lfp_el_pos[i] = lfp_e.mean(axis=0)

            else:
                lfp_el_pos[i] = lfpcalc.calc_lfp_choose(c, \
                    x=self.x[i], y=self.y[i], z=self.z[i], r_limit = r_limit, \
                    sigma=sigma, t_indices=t_indices)
            offsets[i] = {
                'x_n' : x_n,
                'y_n' : y_n,
                'z_n' : z_n
            }
            circle[i] = {
                'x' : crclx,
                'y' : crcly,
                'z' : crclz
            }
        return circle,  offsets,  lfp_el_pos
