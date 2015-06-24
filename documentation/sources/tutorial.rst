=============
LFPy Tutorial
=============

This tutorial will guide you through the first steps with LFPy. It is based on ``example1.py`` in the examples folder.

Let us start by importing LFPy, as well as numpy
::

    import LFPy
    import numpy as np

Then we define a dictionary which describes the properties of the cell we want to simulate
::

    cell_parameters = {         
        'morphology' : 'morphologies/L5_Mainen96_LFPy.hoc',     # Mainen&Sejnowski, Nature, 1996
        'tstartms' : -100.,                 # start time of simulation, recorders start at t=0
        'tstopms' : 200.,                   # stop simulation at 200 ms. 
    }

The only mandatory entry is ``morphology``, which should point to a ``hoc`` file specifying the neuron's morphology. Here we also set the start and end times (in milliseconds). Much more options are available (such as specifying
passive parameters of the cell or adding custom NEURON mechanisms), but for now we leave them at default values.

The Cell object is created using
::

    cell = LFPy.Cell(**cell_parameters)

Let us now add a synapse. Again, we define the synapse parameters first
::

    synapse_parameters = {
        'idx' : cell.get_closest_idx(x=0., y=0., z=800.),
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSyn',       # synapse type
        'tau' : 2.,                 # syn. time constant
        'weight' : .1,              # syn. weight
    }

and create a Synapse object connected to our cell
::

    synapse = LFPy.Synapse(cell, **synapse_parameters)
    
Let us assume we want the synapse to be activated by a single spike at t = 50ms. We have to pass the spike time to the synapse using
::

    synapse.set_spike_times(np.array([50.]))
    
We now have a cell with a synapse, and we can run the simulation
::
    
    cell.simulate(rec_imem=True)

Note the ``rec_imem=True`` argument, this means that the transmembrane currents will be saved - we need the currents to calculate the extracellular potential. 

The final element is the extracellular recording electrode. Again, we start by defining the parameters
::

    electrode_parameters = {
        'sigma' : 0.3,   # extracellular conductivity
        'x' : np.array([0]),
        'y' : np.array([0]),
        'z' : np.array([50])
    }

Here we define a single electrode contact at x = 0, y = 0, z = 50 microns, but a whole array of electrodes can be specified by passing array arguments. 

The electrode (recording from ``cell``) is created using
::

    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    
Finally, we calculate the extracellular potential at the specified electrode location
::
    
    electrode.calc_lfp()
    
The resulting LFP is stored in ``electrode.LFP``.


More examples
=============

More examples of LFPy usage are provided in the "trunk/examples" folder in the
source code release, displaying different usage scenarios;

- `example1.py <https://github.com/LFPy/LFPy/blob/master/examples/example1.py>`_
- `example2.py <https://github.com/LFPy/LFPy/blob/master/examples/example2.py>`_
- `example3.py <https://github.com/LFPy/LFPy/blob/master/examples/example3.py>`_
- `example4.py <https://github.com/LFPy/LFPy/blob/master/examples/example4.py>`_
- `example5.py <https://github.com/LFPy/LFPy/blob/master/examples/example5.py>`_
- `example6.py <https://github.com/LFPy/LFPy/blob/master/examples/example6.py>`_
- `example7.py <https://github.com/LFPy/LFPy/blob/master/examples/example7.py>`_
- `example_mpi.py <https://github.com/LFPy/LFPy/blob/master/examples/example_mpi.py>`_

The examples rely on files present inside the `examples folder <https://github.com/LFPy/LFPy/blob/master/examples>`_,
such as morphology files and .mod files.

The easiest way of accessing all of these files is cloning the examples directory using git (https://git-scm.com):
::
    
    git clone https://github.com/LFPy/LFPy.git
    cd LFPy/examples

