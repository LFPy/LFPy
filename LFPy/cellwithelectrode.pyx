import numpy as np
cimport numpy as np
import neuron
from LFPy import Cell, Electrode

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

class CellWithElectrode(Cell):
    '''
    Subclass of LFPy.Cell with some changes to simulate() and
    _run_simulation(), to facilitate the usage of class Electrode to
    construct a coefficient matrix that is multiplied with the membrane
    currents at every timestep to obtain the LFP, thus the membrane
    currents are not stored unless rec_i=True in simulate()
    
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
        
        cell = LFPy.CellWithElectrode(**cellParameters)
        
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
        
        cell.simulate(rec_isyn=True, **electrodeParameters)
        
        pl.matshow(cell.LFP)
    '''
    
    def __init__(self, **kwargs):
        '''Clone of LFPy.Cell before monkeypatching the simulate() and
        _run_simulation() functions. simulate() accepts kwargs that is passed
        on to LFPy.Electrode, but uses a subclass of LFPy.Electrode under the
        hood.
        Class CellWithElectrode accepts the same kwargs as class Cell'''
        Cell.__init__(self, **kwargs)
        
        if self.timeres_NEURON != self.timeres_python:
            raise ValueError, 'timeres_NEURON != timeres_python'

    def simulate(self, rec_i=False, rec_v=False, rec_ipas=False, rec_icap=False,
                 rec_isyn=False, rec_vsyn=False, rec_istim=False, **kwargs):
        '''Start NEURON simulation and record variables.
        **kwargs are the electrode parameters corresponding to the
        input to the class LFPy.Electrode'''
        self._set_soma_volt_recorder()
        self._set_time_recorder()
        
        if rec_i:
            self._set_imem_recorders()
        if rec_v:
            self._set_voltage_recorders()
        if rec_ipas:
            self._set_ipas_recorders()
        if rec_icap:
            self._set_icap_recorders()
        
        #run fadvance until t >= tstopms
        self._run_simulation(**kwargs)
        
        self._collect_tvec()
        
        self.somav = np.array(self.somav)
        
        if rec_i:
            self._calc_imem()
        
        if rec_ipas:
            self._calc_ipas()
        
        if rec_icap:
            self._calc_icap()
        
        if rec_v:
            self._collect_vmem()
        
        if rec_isyn:
            self._collect_isyn()
        
        if rec_vsyn:
            self._collect_vsyn()
        
        if rec_istim:
            self._collect_istim()
    
    def _run_simulation(self, **kwargs):
        '''Running the actual simulation in NEURON, simulations in NEURON
        is now interruptable. kwargs are passed on to class 
        ElectrodeDetermineCoeffs'''
        
        #initialize electrode, get coefficient array, initialize LFP
        self.electrode = ElectrodeDetermineCoeffs(cell=self, **kwargs)
        self.electrodecoeffs = self.electrode.electrodecoeffs
                
        neuron.h.dt = self.timeres_NEURON
        
        cvode = neuron.h.CVode()
        
        #don't know if this is the way to do, but needed for variable dt method
        if neuron.h.dt <= 1E-8:
            cvode.active(1)
            cvode.atol(0.001)
        else:
            cvode.active(0)
        
        #initialize state
        neuron.h.finitialize(self.v_init)
        
        #initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()
        
        #Starting simulation at t != 0
        if self.tstartms != None:
            neuron.h.t = self.tstartms
        
        self.loadspikes()
        
        #temp vector to store membrane currents at each timestep and other
        #c-declared variables used in while loop
        cdef int i, j
        cdef int totnsegs = self.totnsegs
        cdef double tstopms = self.tstopms
        cdef double counter, interval
        cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] imem = \
            np.empty(totnsegs)
        cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] area = \
            self.area
        cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] \
            electrodecoeffs = self.electrodecoeffs 
        cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] LFP = \
            np.empty((self.tstopms / self.timeres_python + 1,
                     self.electrodecoeffs.shape[0]))
        
        
        #print sim.time at intervals
        if tstopms > 1000:
            interval = 1. / self.timeres_NEURON * 100
        else:
            interval = 1. / self.timeres_NEURON * 10
                
        counter = 0
        j = 0
        while neuron.h.t < tstopms:
            if neuron.h.t >= 0:
                i = 0
                for sec in self.allseclist:
                    for seg in sec:
                        imem[i] = seg.i_membrane
                        i += 1
                imem *= area * 1E-2
                LFP[j, ] = np.dot(electrodecoeffs, imem)
                j += 1
            
            neuron.h.fadvance()
            counter += 1
            if np.mod(counter, interval) == 0:
                print 't = %.0f' % neuron.h.t
        
        #calculate LFP after final fadvance()
        i = 0
        for sec in self.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane
                i += 1
        imem *= area * 1E-2
        LFP[j, ] = np.dot(electrodecoeffs, imem)
        
        self.LFP = LFP.T
    
class ElectrodeDetermineCoeffs(Electrode):
    def __init__(self, **kwargs):
        '''Uses Electrodesetup class, and exludes loading of cell.imem 
        and cell.tvec.
        The class works by faking membrane currents as an eye matrix with 
        size N X N where N is the number of segments, used to create the 
        electrodecoeffs matrix, which may then give the LFP when doing
        the dot-product electrodecoeffs * i_membrane at every timestep
        '''
        Electrode.__init__(self, **kwargs)
        
        #Non default procedure for class
        totnsegs = kwargs['cell'].totnsegs
        self.dt = kwargs['cell'].timeres_NEURON
        self.tvec = np.arange(totnsegs) * self.dt
        self.c[0].imem = np.eye(totnsegs)
        self.c[0].tvec = np.arange(totnsegs) * self.dt
        self.calc_lfp()
        self.electrodecoeffs = self.LFP
        del self.c[0].imem
        del self.LFP
        
    def _import_c(self, cell):
        '''Keeps the relevant variables for LFP-calculation from cell'''
        #keeping these variables:
        variables = [
            'somav',
            'somaidx',
            'timeres_python',
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
                    str(type(cell))[:12] == "<class 'cellwithelectrode.CellWithElectrode'>"[:12]:
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
        
        self.nCells = np.array(self.c.keys()).size
    
