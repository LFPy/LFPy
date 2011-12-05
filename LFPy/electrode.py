#!/usr/bin/env python
'''Copyright (C) 2011 Computational Neuroscience Group, UMB.
All rights reserved.'''
import pylab as pl
import sys
from LFPy import lfpcalc, tools
from multiprocessing import Process, Queue, freeze_support, cpu_count

class ElectrodeSetup(object):
    '''
    Electrode superclass, does import a LFPy.cell.Cell object or either list
    or dictionary containing such Cell objects. Keyword arguments determine
    properties of latter LFP-calculations
    
    Arguments:
    ::
        sigma   : extracellular conductivity
        x, y, z : coordinates or arrays of coordinates. Must be same length
        N       : Normal vector [x, y, z] of contact surface, default None
        r       : radius of contact surface, default None
        n       : if N != None and r > 0, the number of points to use for each
                  contact point in order to calculate average
        color   : color of electrode contact points in plots
        marker  : marker of electrode contact points in plots
        from_file   : if True, load cell object from file
        cellfile    : path to cell pickle
        verbose : Flag for verbose output        
    '''
    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 N=None, r=None, n=0, r_z=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None, verbose=False):
        '''Initialize class ElectrodeSetup'''
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


class Electrode(ElectrodeSetup):
    '''
    Electrode class with inheritance from LFPy.ElectrodeSetup able to actually
    calculate local field potentials from LFPy.Cell objects. **kwargs are passed
    on to LFPy.ElectrodeSetup
    
    Usage:
    ::
        import LFPy
        import pylab as pl
        
        N = pl.empty((16, 3))
        for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
        electrodeParameters = {             #parameters for electrode class
            'sigma' : 0.3,              #Extracellular potential
            'x' : pl.zeros(16)+25,      #Coordinates of electrode contacts
            'y' : pl.zeros(16),
            'z' : pl.linspace(-500,1000,16),
            'n' : 20,
            'r' : 10,
            'N' : N,
        }
        
        cellParameters = {                          
            'morphology' : 'L5_Mainen96_LFPy.hoc',  # morphology file
            'rm' : 30000,                           # membrane resistivity
            'cm' : 1.0,                             # membrane capacitance
            'Ra' : 150,                             # axial resistivity
            'timeres_NEURON' : 2**-4,                # dt for NEURON sim.
            'timeres_python' : 2**-4,                 # dt for python output
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
        
        cell.simulate()
        
        electrode = LFPy.Electrode(cell, **electrodeParameters)
        electrode.calc_lfp()
        pl.matshow(electrode.LFP)
    '''

    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 N=None, r=None, n=0, r_z=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None):
        '''This is the regular implementation of the Electrode class
        that calculates the LFP serially using a single core'''
        ElectrodeSetup.__init__(self, cell, sigma, x, y, z,
                                N, r, n, r_z, perCellLFP,
                                method, color, marker, from_file, cellfile)
        
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
                LFP_temp[k, :, :] = self._loop_over_contacts(k, LFP_temp, variables)

            if self.verbose:
                print 'Calculated potential contribution, cell %i.' % k
        if self.perCellLFP:
            self.CellLFP = LFP_temp
        
        self.LFP = LFP_temp.sum(axis=0)

    def _loop_over_contacts(self, k, LFP_temp, variables):
        '''Loop over electrode contacts, and will return LFP_temp filled'''
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
            
        return LFP_temp[k, :, :]

    
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


class ElectrodeThreaded(Electrode):
    '''Inherit class Electrode, but using multiprocessing to distribute
    calculations of LFP for each electrode contact
    
    Will only work for case when averaging over electrode geometry area
    currently
    '''
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

    def _loop_over_contacts(self, k, LFP_temp, variables,
                            __name__='__main__', NUMBER_OF_PROCESSES=None):
        '''Monkeypatching to include multiprocessing!
        Loop over electrode contacts, and will return LFP_temp filled'''
        
        if __name__ == '__main__':
            freeze_support()
            if NUMBER_OF_PROCESSES == None:
                NUMBER_OF_PROCESSES = cpu_count()
            elif type(NUMBER_OF_PROCESSES) != int:
                raise ValueError, 'NUMBER_OF_PROCESSES != int, %s' \
                                                    % str(NUMBER_OF_PROCESSES)
            else:
                pass
            task_queue = Queue()
            done_queue = Queue()

            TASKS = pl.arange(len(self.x))
            
            for task in TASKS:
                task_queue.put(int(task))       
            for i in xrange(NUMBER_OF_PROCESSES):
                Process(target=self._loop_over_contacts_thread,
                             args=(task_queue, k, LFP_temp, variables,
                             done_queue)).start()
            for n in xrange(TASKS.size):
                [i, lfp_temp] = done_queue.get()
                LFP_temp[k, i, :] += lfp_temp                 
            for i in xrange(NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            
            task_queue.close()
            done_queue.close()
        else:
            raise Exception, "'__name__' != '__main__'"
        
        return LFP_temp[k, :, :]

    def _loop_over_contacts_thread(self, task_queue, k, LFP_temp, variables,
                             done_queue):
        """thread for each contact point"""
        for i in iter(task_queue.get, 'STOP'):
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
        
            done_queue.put([i, LFP_temp[k, i, :]])
        
    
    def _lfp_el_pos_calc_dist(self, c, r_limit, sigma=0.3, radius=10, n=10,
                             m=50, N=None, t_indices=None, 
                             method='linesource', __name__="__main__",
                             NUMBER_OF_PROCESSES=None):
        '''
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface with radius r. The locations of these n points on
        the electrode surface are random,  within the given radius'''
        lfp_el_pos = pl.zeros(self.LFP.shape)
        offsets = {}
        circle = {}
        
        if __name__ == "__main__":
            freeze_support()
            if NUMBER_OF_PROCESSES == None:
                NUMBER_OF_PROCESSES = cpu_count()
            elif type(NUMBER_OF_PROCESSES) != int:
                raise ValueError, 'NUMBER_OF_PROCESSES != int, %s' \
                                                    % str(NUMBER_OF_PROCESSES)
            else:
                pass
            task_queue = Queue()
            done_queue = Queue()

            TASKS = pl.arange(len(self.x))
            
            for task in TASKS:
                task_queue.put(int(task))       
            for i in xrange(NUMBER_OF_PROCESSES):
                Process(target=self._lfp_el_pos_calc_dist_i,
                             args=(task_queue,
                             c, r_limit, sigma, radius, n,
                             m, N, t_indices, method,
                             done_queue)).start()
            for n in xrange(TASKS.size):
                [index, lfp, offset, circle] = done_queue.get() 
                lfp_el_pos[index], offsets[index], circle[index] = lfp, offset, circle                
            for i in xrange(NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            
            task_queue.close()
            done_queue.close()
        else:
            raise Exception, "'__name__' != '__main__'"
                
        return circle,  offsets,  lfp_el_pos
        
            
    def _lfp_el_pos_calc_dist_i(self, task_queue,
                    c, r_limit, sigma, radius, n, m, N, t_indices, method,
                    done_queue):       
        for index in iter(task_queue.get, 'STOP'):
            if n > 1:
                lfp_e = pl.zeros((n, self.LFP.shape[1]))
    
                offs = pl.zeros((n, 3))
                r2 = pl.zeros(n)
    
                crcl = pl.zeros((m, 3))
    
                for j in xrange(n):
                    A = [(pl.rand()-0.5)*radius*2, (pl.rand()-0.5)*radius*2,
                            (pl.rand()-0.5)*radius*2]
                    offs[j, ] = pl.cross(N[index, ], A)
                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
                    while r2[j] > radius**2:
                        A = [(pl.rand()-0.5)*radius*2, (pl.rand()-0.5)*radius*2,
                            (pl.rand()-0.5)*radius*2]
                        offs[j, ] = pl.cross(N[index, ], A)
                        r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
    
                x_n = offs[:, 0] + self.x[index]
                y_n = offs[:, 1] + self.y[index]
                z_n = offs[:, 2] + self.z[index]
    
                for j in xrange(m):
                    B = [(pl.rand()-0.5), (pl.rand()-0.5), (pl.rand()-0.5)]
                    crcl[j, ] = pl.cross(N[index, ], B)
                    crcl[j, ] = crcl[j, ]/pl.sqrt(crcl[j, 0]**2 +
                                               crcl[j, 1]**2 + \
                                               crcl[j, 2]**2)*radius
    
                crclx = crcl[:, 0] + self.x[index]
                crcly = crcl[:, 1] + self.y[index]
                crclz = crcl[:, 2] + self.z[index]
    
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
                    x=self.x[index], y=self.y[index], z=self.z[index], r_limit = r_limit,
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
            done_queue.put([index, lfp_el_pos, offsets, circle])


    




#class ElectrodeThreaded(ElectrodeSetup):
#    '''
#    class with inheritance from class ElectrodeSetup, calculating LFPs using
#    available cpu-cores.
#    
#    Usage:
#    ::
#        electrode = LFPy.ElectrodeThreaded(cell, **kwargs)
#        electrode.calc_lfp_threaded()
#    '''
#    
#    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
#                 N=None, r=None, n=0, r_z=None, colors=None,
#                 perCellLFP=False, method='linesource', 
#                 color='g', marker='o',
#                 from_file=False, cellfile=None):
#        '''Initialization of class ElectrodeThreaded, with electrode setup
#        inherited from class ElectrodeSetup'''
#        ElectrodeSetup.__init__(self, cell, sigma, x, y, z,
#                                N, r, n, r_z, perCellLFP,
#                                method, color, marker, from_file, cellfile)
#            
#    def calc_lfp_threaded(self, t_indices=None, __name__='__main__',
#                          NUMBER_OF_PROCESSES=None, ):
#        '''Calculate LFP on electrode geometry from all cell instances.
#        Will choose distributed calculated if electrode contain 'n', 'N', and
#        'r'. Will try to distribute jobs to available processes'''
#        #create self.LFP if not already there
#        if hasattr(self, 'LFP'):
#            del self.LFP
#            print 'deleted electrode.LFP, consecutive calls to calc_lfp_th...!'
#        if not hasattr(self,  'LFP'):
#            if t_indices != None:
#                if self.perCellLFP:
#                    LFP_temp = pl.zeros((self.nCells, self.x.size,
#                                         t_indices.size))
#                else:
#                    LFP_temp = pl.zeros((self.nCells, self.x.size,
#                                         t_indices.size))
#                self.LFP = pl.zeros((self.x.size, t_indices.size))
#            else:
#                if self.perCellLFP:
#                    LFP_temp = pl.zeros((self.nCells, self.x.size,
#                                         self.tvec.size))
#                else:
#                    LFP_temp = pl.zeros((self.nCells, self.x.size,
#                                         self.tvec.size))
#                self.LFP = pl.zeros((self.x.size, self.tvec.size))
#        
#        #multiprocess initialization
#        if __name__ == '__main__':
#            freeze_support()
#            if NUMBER_OF_PROCESSES == None:
#                NUMBER_OF_PROCESSES = cpu_count() #Number of threads
#            #Creating multiproc. queues for tasks and results
#            task_queue = Queue()
#            done_queue = Queue()
#
#            #One task per cell instance
#            TASKS = pl.array(self.c.keys())
#
#            #Distribute TASKS, call Process, fetching results for LFP-calc.
#            #and stopping and closing threads and queues
#            for task in TASKS:
#                task_queue.put(task)
#            for i in xrange(NUMBER_OF_PROCESSES):
#                Process(target=self._calc_lfp_thread,
#                    args=(task_queue, t_indices, NUMBER_OF_PROCESSES,
#                          done_queue)).start()
#            for i in xrange(TASKS.size):
#                [self.circle, self.offsets, n, LFP] = done_queue.get()
#                LFP_temp[n, :, :] = LFP
#            for i in xrange(NUMBER_OF_PROCESSES):
#                task_queue.put('STOP')
#            task_queue.close()
#            done_queue.close()
#        
#        if self.perCellLFP:
#            self.CellLFP = LFP_temp
#        
#        self.LFP = LFP_temp.sum(axis=0)
#
#    def _calc_lfp_thread(self, task_queue, t_indices,
#                          NUMBER_OF_PROCESSES, done_queue):
#        '''Single thread, calculating LFP from single cell.
#        Should be called by calc_lfp_threaded'''
#        #local declaration of LFP, which will be saved in done_queue
#        if t_indices != None:
#            LFP = pl.zeros((self.x.size, t_indices.size))
#        else:
#            LFP = pl.zeros((self.x.size, self.tvec.size))
#        variables = {
#            'sigma' : self.sigma,
#            't_indices' : t_indices,
#            'method' : self.method,
#        }
#
#        #iterate over elements in the task queue
#        for k in iter(task_queue.get, 'STOP'):
#            #choosing spatial averaging for each electrode contact
#            if self.n != None and self.N != None and self.r != None:
#                variables.update({
#                    'r_limit' : self.c[k].diam / 2,
#                    'radius' : self.r,
#                    'n' : self.n,
#                    'N' : self.N,
#                    't_indices' : t_indices,
#                    'NUMBER_OF_PROCESSES' : NUMBER_OF_PROCESSES,
#                    })
#                #Calling function which will calculate LFP, distributed
#                if sys.version_info < (2, 6, 6): #fix for older Pythons
#                    del variables['NUMBER_OF_PROCESSES']
#                    [circle, offsets, LFP] = self._lfp_el_pos_calc_dist(
#                        self.c[k], **variables)
#                else:
#                    [circle, offsets, LFP] = \
#                    self._lfp_el_pos_calc_dist_threaded(self.c[k], **variables)
#                done_queue.put([circle, offsets, k, LFP])
#            #case for no averaging
#            else:
#                variables.update({
#                    'r_limit' : self.c[k].diam / 2
#                    })
#                if sys.version_info < (2, 6, 6): #fix for runs on Stallo
#                    for i in xrange(self.x.size):
#                        variables.update({
#                            'x' : self.x[i],
#                            'y' : self.y[i],
#                            'z' : self.z[i],
#                        })
#                        if hasattr(self, 'r_drift'):
#                            LFP = pl.zeros(self.LFP.shape[1])
#                            for j in xrange(self.LFP.shape[1]):
#                                variables.update({
#                                    'x' : self.x[i] + self.r_drift['x'][i],
#                                    'y' : self.y[i] + self.r_drift['y'][i],
#                                    'z' : self.z[i] + self.r_drift['z'][i],
#                                    'timestep' : j
#                                })
#                                LFP[i, j] = lfpcalc.calc_lfp_choose(self.c[k], 
#                                                                    **variables)
#                        else:
#                            LFP[i, ] = lfpcalc.calc_lfp_choose(self.c[k], 
#                                                               **variables)
#                else:
#                    LFP = self._calc_lfp_simple_threaded(self.c[k], LFP, 
#                                                          **variables)
#                circle = None,
#                offsets = None
#                
#                done_queue.put([circle, offsets, k, LFP])
#            if self.verbose:
#                print 'Calculated potential contribution, cell %i.' % k
#        
#    
#    def _calc_lfp_simple_threaded(self, c, LFP, **variables):
#
#        __name__='__main__'
#        if __name__ == '__main__':              # This is important, apparently
#            output = {}                         # results saved in dict
#            freeze_support()                    # let NUMBER_OF_PROCESSES calls
#                                                # finish before new jobs are qd
#            NUMBER_OF_PROCESSES = cpu_count()   # Using all available cores
#            task_queue = Queue()                # queue for tasks
#            done_queue = Queue()                # simulation results put here
#            
#            TASKS = pl.arange(self.x.size)
#            
#            for task in TASKS:
#                task_queue.put(int(task))       
#            for i in xrange(NUMBER_OF_PROCESSES):
#                Process(target=self._calc_lfp_simple_thread,
#                        args=(task_queue, done_queue, c, variables)).start()
#            for n in xrange(TASKS.size):
#                [i, lfp] = done_queue.get()
#                LFP[i, ] = lfp
#            for i in xrange(NUMBER_OF_PROCESSES):
#                task_queue.put('STOP')
#            
#            task_queue.close()      # No more jobs can be sent to the queues,
#            done_queue.close()      # necessary for consecutive parallel jobs
#            
#            return LFP
#    
#    def _calc_lfp_simple_thread(self, task_queue,
#                                 done_queue, c, variables):
#        for i in iter(task_queue.get, 'STOP'):
#            variables.update({
#                        'x' : self.x[i],
#                        'y' : self.y[i],
#                        'z' : self.z[i],
#                    })
#            if hasattr(self, 'r_drift'):
#                LFP = pl.zeros(self.LFP.shape[1])
#                for j in xrange(self.LFP.shape[1]):
#                    variables.update({
#                        'x' : self.x[i] + self.r_drift['x'][i],
#                        'y' : self.y[i] + self.r_drift['y'][i],
#                        'z' : self.z[i] + self.r_drift['z'][i],
#                        'timestep' : j
#                    })
#                    LFP[j] = lfpcalc.calc_lfp_choose(c, **variables)
#            else:
#                LFP = lfpcalc.calc_lfp_choose(c, **variables)
#                
#            done_queue.put([i, LFP])
#    
#    def _lfp_el_pos_calc_dist_threaded(self, c, r_limit, sigma=0.3, radius=10,
#                n=10, m=50, N=None, t_indices=None, NUMBER_OF_PROCESSES=None, 
#                method='linesource',
#                __name__='__main__'):
#        '''Calc. of LFP over an n-point integral approximation over flat
#        electrode surface with radius r. The locations of these n points on
#        the electrode surface are random,  within the given radius.
#        The method will spawn a few multiprocessing threads'''
#        offsets = {}
#        circle = {}
#
#        if __name__ == '__main__':
#            freeze_support()
#            if NUMBER_OF_PROCESSES == None:
#                NUMBER_OF_PROCESSES = cpu_count()
#            dist_task_queue = Queue()
#            dist_done_queue = Queue()
#
#            TASKS = pl.arange(self.x.size)
#            tempLFP = pl.zeros((self.x.size, self.LFP.shape[0],
#                                self.LFP.shape[1]))
#            
#            for task in TASKS:
#                dist_task_queue.put(task)
#            for i in xrange(NUMBER_OF_PROCESSES):
#                Process(target=self._lfp_el_pos_calc_dist_thread,
#                    args=(dist_task_queue, c, r_limit, sigma, radius, n, m, N,
#                    t_indices, method, dist_done_queue)).start()
#            for i in xrange(TASKS.size):
#                [circl, offset, tempLFP[i,]] = dist_done_queue.get()
#                if type(circl) == type({}) and type(offsets) == type({}):
#                    circle.update(circl)
#                    offsets.update(offset)
#            for i in xrange(NUMBER_OF_PROCESSES):
#                dist_task_queue.put('STOP')
#
#            dist_task_queue.close()
#            dist_done_queue.close()
#        
#        lfp_el_pos = tempLFP.sum(axis=0)
#        return offsets, circle, lfp_el_pos
#
#    def _lfp_el_pos_calc_dist_thread(self, dist_task_queue, c, r_limit, sigma,
#            radius, n, m, N, t_indices, method, dist_done_queue):
#        '''spawn thread called by self._lfp_el_pos_calc_dist_threaded()'''
#        for i in iter(dist_task_queue.get, 'STOP'):
#            lfp_el_pos = pl.zeros(self.LFP.shape)
#            offsets = {}
#            circle = {}
#            if n > 1:
#                lfp_e = pl.zeros((n, self.LFP.shape[1]))
#
#                offs = pl.zeros((n, 3))
#                r2 = pl.zeros(n)
#                crcl = pl.zeros((m, 3))
#
#                for j in xrange(n):
#                    A = [(pl.rand()-0.5)*radius*2, (pl.rand()-0.5)*radius*2,
#                        (pl.rand()-0.5)*radius*2]
#                    offs[j, ] = pl.cross(N[i, ], A)
#                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
#                    while r2[j] > radius**2:
#                        A = [(pl.rand()-0.5)*radius*2, (pl.rand()-0.5)*radius*2,
#                            (pl.rand()-0.5)*radius*2]
#                        offs[j, ] = pl.cross(N[i, ], A)
#                        r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
#
#                x_n = offs[:, 0] + self.x[i]
#                y_n = offs[:, 1] + self.y[i]
#                z_n = offs[:, 2] + self.z[i]
#
#                for j in xrange(m):
#                    B = [(pl.rand()-0.5), (pl.rand()-0.5), (pl.rand()-0.5)]
#                    crcl[j, ] = pl.cross(N[i, ], B)
#                    crcl[j, ] = crcl[j, ]/pl.sqrt(crcl[j, 0]**2 +
#                                               crcl[j, 1]**2 + 
#                                               crcl[j, 2]**2)*radius
#
#                crclx = crcl[:, 0] + self.x[i]
#                crcly = crcl[:, 1] + self.y[i]
#                crclz = crcl[:, 2] + self.z[i]
#
#                for j in xrange(n):
#                    variables = {
#                        'x' : x_n[j],
#                        'y' : y_n[j],
#                        'z' : z_n[j],
#                        'r_limit' : r_limit,
#                        'sigma' : sigma,
#                        't_indices' : t_indices,
#                        'method' : self.method,
#                    }
#                    lfp_e[j, ] = lfpcalc.calc_lfp_choose(c, **variables)
#
#                lfp_el_pos[i] = lfp_e.mean(axis=0)
#                offsets[i] = {
#                'x_n' : x_n,
#                'y_n' : y_n,
#                'z_n' : z_n
#                }
#                circle[i] = {
#                'x' : crclx,
#                'y' : crcly,
#                'z' : crclz
#                }
#            else:
#                lfp_el_pos[i] = lfpcalc.calc_lfp_choose(c, \
#                    x=self.x[i], y=self.y[i], z=self.z[i], r_limit = r_limit,
#                    sigma=sigma, t_indices=t_indices)
#                
#            
#            dist_done_queue.put([circle, offsets, lfp_el_pos])

