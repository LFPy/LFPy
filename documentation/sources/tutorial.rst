=============
LFPy Tutorial
=============

This tutorial will guide you through a few first steps with LFPy. It is based on ``example1.py`` in the ``LFPy/examples`` folder.
In order to obtain all necessary files, please obtain the LFPy source files as described on the `Download page <information.html#download-lfpy>`_

Change directory to the LFPy examples folder, and start the interactive `IPython <https://ipython.org>`_ interpreter
::
    
    $ cd <path/to/LFPy>/examples
    $ ipython

Let us start by importing ``LFPy``, as well as the ``numpy`` and ``os`` modules
::

    >>> import os
    >>> import LFPy
    >>> import numpy as np

Then we define a dictionary which describes the properties of the cell we want to simulate
::

    >>> cell_parameters = {         
    >>>     'morphology' : os.path.join('morphologies', 'L5_Mainen96_LFPy.hoc'),     # Mainen&Sejnowski, Nature, 1996
    >>>     'tstart' : 0.,          # start time of simulation, recorders start at t=0
    >>>     'tstop' : 20.,          # stop simulation at 20 ms.
    >>>     'v_init' : -70.0,       # initial voltage at tstart
    >>>     'Ra' : 35.4,            # axial resistivity
    >>>     'cm' : 1.0,             # membrane capacitance
    >>>     'passive' : True,       # switch on passive leak resistivity
    >>>     'passive_parameters' : {'e_pas': -70.0, # passive leak reversal potential
    >>>                             'g_pas': 0.001} # passive leak conductivity
    >>> }

The only mandatory entry is ``morphology``, which should point to a ``hoc`` file specifying the neuron's morphology.
Here we also set the start and end times (in milliseconds). Many more options are available (such as adding custom
NEURON mechanisms), but for now we leave other parameters at their default values.

Then, the Cell object is created issuing
::

    >>> cell = LFPy.Cell(**cell_parameters)

Let us now add a synapse to the cell. Again, we define the synapse parameters first
::

    >>> synapse_parameters = {
    >>>     'idx' : cell.get_closest_idx(x=0., y=0., z=800.), # segment index for synapse
    >>>     'e' : 0.,                   # reversal potential
    >>>     'syntype' : 'ExpSyn',       # synapse type
    >>>     'tau' : 2.,                 # syn. time constant
    >>>     'weight' : .1,              # syn. weight
    >>>     'record_current' : True           # record syn. current
    >>> }

and create a Synapse object connected to our cell
::

    >>> synapse = LFPy.Synapse(cell, **synapse_parameters)
    
Let us assume we want the synapse to be activated by a single spike at t = 5 ms. We have to pass the spike time to the synapse using
::

    >>> synapse.set_spike_times(np.array([5.]))
    
We now have a cell with a synapse, and we can run the simulation
::
    
    >>> cell.simulate(rec_imem=True)

Note the ``rec_imem=True`` argument, this means that the transmembrane currents will be saved - we need the currents to calculate the extracellular potential.
An array with all transmembrane currents for the total number of segments ``cell.totnsegs`` is now accessible as ``cell.imem``, the somatic voltage as ``cell.tvec``,
all with time stamps ``cell.tvec``.
::

    >>> print(cell.tvec.shape, cell.somav.shape, cell.totnsegs, cell.imem.shape)

The final element is the extracellular recording electrode. Again, we start by defining the parameters
::

    >>> electrode_parameters = {
    >>>     'sigma' : 0.3,   # extracellular conductivity
    >>>     'x' : np.array([0]),
    >>>     'y' : np.array([0]),
    >>>     'z' : np.array([50])
    >>> }

Here we define a single electrode contact at x = 0, y = 0, z = 50 µm, but a whole array of electrodes can be specified by passing array arguments. 

The electrode (recording from ``cell``) is created using
::

    >>> electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    
Finally, we calculate the extracellular potential at the specified electrode location
::
    
    >>> electrode.calc_lfp()
    
The resulting LFP is stored in ``electrode.LFP``.
::
    
    >>> print(electrode.LFP.shape)

Finally, the cell morphology and synapse location can be plotted
::

    >>> from matplotlib.collections import PolyCollection
    >>> import matplotlib.pyplot as plt
    >>> zips = []
    >>> for x, z in cell.get_idx_polygons(projection=('x', 'z')):
    >>>     zips.append(zip(x, z))
    >>> polycol = PolyCollection(zips,
    >>>                          edgecolors='none',
    >>>                          facecolors='gray')
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.add_collection(polycol)
    >>> ax.plot(cell.xmid[synapse.idx], cell.zmid[synapse.idx], 'ro')
    >>> ax.axis(ax.axis('equal'))
    >>> plt.show()

As well as the simulated output
::

    >>> fig = plt.figure()
    >>> plt.subplot(311)
    >>> plt.plot(cell.tvec, synapse.i)
    >>> plt.subplot(312)
    >>> plt.plot(cell.tvec, cell.somav)
    >>> plt.subplot(313)
    >>> plt.plot(cell.tvec, electrode.LFP.T)
    >>> plt.show()

More examples
=============

More examples of LFPy usage are provided in the ``trunk/examples`` folder in the
source code release, displaying different usages of LFPy.

The examples rely on files present inside the `examples folder <https://github.com/LFPy/LFPy/blob/master/examples>`_,
such as morphology files (.hoc) and NEURON NMODL (.mod) files.

The easiest way of accessing all of these files is cloning the examples directory using git (https://git-scm.com):
::
    
    $ git clone https://github.com/LFPy/LFPy.git
    $ cd LFPy/examples

The files provided are

- `example1.py <https://github.com/LFPy/LFPy/blob/master/examples/example1.py>`_
- `example2.py <https://github.com/LFPy/LFPy/blob/master/examples/example2.py>`_
- `example3.py <https://github.com/LFPy/LFPy/blob/master/examples/example3.py>`_
- `example4.py <https://github.com/LFPy/LFPy/blob/master/examples/example4.py>`_
- `example5.py <https://github.com/LFPy/LFPy/blob/master/examples/example5.py>`_
- `example6.py <https://github.com/LFPy/LFPy/blob/master/examples/example6.py>`_
- `example7.py <https://github.com/LFPy/LFPy/blob/master/examples/example7.py>`_
- `example8.py <https://github.com/LFPy/LFPy/blob/master/examples/example8.py>`_
- `example_mpi.py <https://github.com/LFPy/LFPy/blob/master/examples/example_mpi.py>`_
- `example_EPFL_neurons.py <https://github.com/LFPy/LFPy/blob/master/examples/example_EPFL_neurons.py>`_
- `example_LFPyCellTemplate.py <https://github.com/LFPy/LFPy/blob/master/examples/example_LFPyCellTemplate.py>`_
- `example_MEA.py <https://github.com/LFPy/LFPy/blob/master/examples/example_MEA.py>`_
- `example_anisotropy.py <https://github.com/LFPy/LFPy/blob/master/examples/example_anisotropy.py>`_
- `example_loadL5bPCmodelsEH.py <https://github.com/LFPy/LFPy/blob/master/examples/example_loadL5bPCmodelsEH.py>`_
- `example_network/example_Network.py <https://github.com/LFPy/LFPy/blob/master/examples/example_network/example_Network.py>`_
- `example_EEG.py <https://github.com/LFPy/LFPy/blob/master/examples/example_EEG.py>`_



