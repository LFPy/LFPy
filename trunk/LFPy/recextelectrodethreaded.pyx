#!/usr/bin/env python
'''Copyright (C) 2012 Computational Neuroscience Group, UMB.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.'''

from LFPy import RecExtElectrode, lfpcalc
from multiprocessing import Process, Queue, freeze_support, cpu_count
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


class RecExtElectrodeThreaded(RecExtElectrode):
    '''Inherit class RecExtElectrode, but using multiprocessing to distribute
    calculations of LFP for each electrode contact
    '''
    def __init__(self, cell, sigma=0.3, x=100, y=0, z=0,
                 N=None, r=None, n=0, r_z=None, colors=None,
                 perCellLFP=False, method='linesource', 
                 color='g', marker='o',
                 from_file=False, cellfile=None,
                 NUMBER_OF_PROCESSES=None, verbose=False,
                 seedvalue=None):
        '''Initialization of class ElectrodeThreaded, with electrode setup
        inherited from class RecExtElectrodeSetup'''
        RecExtElectrode.__init__(self, cell, sigma, x, y, z,
                                N, r, n, r_z, perCellLFP,
                                method, color, marker, from_file,
                                cellfile, verbose, seedvalue)
        #set the numbers of processes to use, if none use all available cores
        if NUMBER_OF_PROCESSES == None:
            NUMBER_OF_PROCESSES = cpu_count()
        elif type(NUMBER_OF_PROCESSES) != int:
            raise ValueError, 'NUMBER_OF_PROCESSES != int, %s' \
                                                % str(NUMBER_OF_PROCESSES)
        self.NUMBER_OF_PROCESSES = NUMBER_OF_PROCESSES
            

    def _loop_over_contacts(self, int k, variables,
                            __name__='__main__'
                            ):
        '''Monkeypatching function to use Python multiprocessing!
        Loop over electrode contacts, and will return LFP_temp filled'''

        cdef np.ndarray[DTYPE_t, ndim=2] LFP_temp
        cdef np.ndarray[long, ndim=1] TASKS
        cdef int i, task

        if variables['t_indices'] != None:
            LFP_temp = np.zeros((self.x.size, variables['t_indices'].size))
        else:
            LFP_temp = np.zeros((self.x.size, self.tvec.size))

        if __name__ == '__main__':
            freeze_support()
            task_queue = Queue()
            done_queue = Queue()

            TASKS = np.arange(len(self.x))
            
            for task in TASKS:
                task_queue.put(int(task))       
            for i in xrange(self.NUMBER_OF_PROCESSES):
                Process(target=self._loop_over_contacts_thread,
                             args=(task_queue, k, variables,
                             done_queue)).start()
            for n in xrange(TASKS.size):
                [i, lfp_temp] = done_queue.get()
                LFP_temp[i, :] += lfp_temp                 
            for i in xrange(self.NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            
            task_queue.close()
            done_queue.close()
        else:
            raise Exception, "'__name__' != '__main__'"
        
        return LFP_temp

    def _loop_over_contacts_thread(self, task_queue,
                                   int k, variables,
                             done_queue):
        '''thread calculating the LFP in each contact point called from
        self._loop_over_contacts'''
        
        cdef np.ndarray[DTYPE_t, ndim=2] LFP_temp
        cdef int i, j
        
        if variables['t_indices'] != None:
            LFP_temp = np.zeros((self.x.size, variables['t_indices'].size))
        else:
            LFP_temp = np.zeros((self.x.size, self.tvec.size))

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
                    LFP_temp[i, j] = LFP_temp[i, j] + \
                        lfpcalc.calc_lfp_choose(self.c[k], **variables)
            else:
                LFP_temp[i, :] = LFP_temp[i, :] + \
                        lfpcalc.calc_lfp_choose(self.c[k], **variables)
        
            done_queue.put([i, LFP_temp[i, :]])
        
    
    def _lfp_el_pos_calc_dist(self, int k,
                             np.ndarray[DTYPE_t, ndim=1] r_limit,
                             double sigma=0.3, double radius=10, int n=10,
                             int m=50, N=None, t_indices=None, 
                             str method='linesource',
                             str __name__="__main__"
                             ):
        '''
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface with radius r. The locations of these n points on
        the electrode surface are random,  within the given radius'''
        cdef int i, task, index
        cdef np.ndarray[long, ndim=1] TASKS
        cdef np.ndarray[DTYPE_t, ndim=2] lfp_el_pos
        cdef np.ndarray[DTYPE_t, ndim=1] lfp
        
        lfp_el_pos = np.zeros(self.LFP.shape)
        offsets = {}
        circles = {}
        
        if __name__ == "__main__":
            freeze_support()
            task_queue = Queue()
            done_queue = Queue()

            TASKS = np.arange(len(self.x))
            
            for task in TASKS:
                task_queue.put(task)       
            for i in xrange(self.NUMBER_OF_PROCESSES):
                Process(target=self._lfp_el_pos_calc_dist_i,
                             args=(task_queue,
                             k, r_limit, sigma, radius, n,
                             m, N, t_indices, method,
                             done_queue)).start()
            for i in xrange(TASKS.size):
                [index, lfp, offset, circle] = done_queue.get() 
                lfp_el_pos[index, :], offsets[index], circles[index] = lfp, offset, circle                
            for i in xrange(self.NUMBER_OF_PROCESSES):
                task_queue.put('STOP')
            
            task_queue.close()
            done_queue.close()
        else:
            raise Exception, "'__name__' != '__main__'"
                
        return circles,  offsets,  lfp_el_pos
        
            
    def _lfp_el_pos_calc_dist_i(self, task_queue,
                    int k,
                    np.ndarray[DTYPE_t, ndim=1] r_limit,
                    double sigma,
                    double radius,
                    int n,
                    int m,
                    np.ndarray[DTYPE_t, ndim=2] N,
                    t_indices, method,
                    done_queue):
        '''
        Multiprocessing thread used by self_lfp_el_pos_calc_dist(), distributing
        calculations of the LFP on each contact point
        '''
        cdef int index, j
        cdef np.ndarray[DTYPE_t, ndim=2] lfp_e, crcl, offs
        cdef np.ndarray[DTYPE_t, ndim=1] r2, A, B, x_n, y_n, z_n, \
                                            crclx, crcly, crclz, lfp_el_pos 
        
        for index in iter(task_queue.get, 'STOP'):
            if n > 1:
                lfp_e = np.zeros((n, self.LFP.shape[1]))
                offs = np.zeros((n, 3))
                r2 = np.zeros(n)
                crcl = np.zeros((m, 3))
                
                #assert the same random numbers are drawn every time
                if self.seedvalue != None:
                    np.random.seed(self.seedvalue)
                
                for j in xrange(n):
                    A = np.array([(np.random.rand()-0.5)*radius*2,
                            (np.random.rand()-0.5)*radius*2,
                            (np.random.rand()-0.5)*radius*2])
                    offs[j, ] = np.cross(N[index, ], A)
                    r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
                    while r2[j] > radius**2:
                        A = np.array([(np.random.rand()-0.5)*radius*2,
                            (np.random.rand()-0.5)*radius*2,
                            (np.random.rand()-0.5)*radius*2])
                        offs[j, ] = np.cross(N[index, ], A)
                        r2[j] = offs[j, 0]**2 + offs[j, 1]**2 + offs[j, 2]**2
    
                x_n = offs[:, 0] + self.x[index]
                y_n = offs[:, 1] + self.y[index]
                z_n = offs[:, 2] + self.z[index]
    
                for j in xrange(m):
                    B = np.array([(np.random.rand()-0.5),
                        (np.random.rand()-0.5), (np.random.rand()-0.5)])
                    crcl[j, ] = np.cross(N[index, ], B)
                    crcl[j, ] = crcl[j, ]/np.sqrt(crcl[j, 0]**2 +
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
                    lfp_e[j, ] = lfpcalc.calc_lfp_choose(self.c[k], **variables)
    
                lfp_el_pos = lfp_e.mean(axis=0)    
            else:
                lfp_el_pos = lfpcalc.calc_lfp_choose(self.c[k],
                    x=self.x[index], y=self.y[index], z=self.z[index],
                    r_limit = r_limit,
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
