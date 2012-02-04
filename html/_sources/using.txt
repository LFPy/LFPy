=============
Notes on LFPy
=============

Using your own models
=====================

Importing morphology
--------------------

Cell morphology can be specified manually in a ``hoc`` file. For a simple example, see
``examples/morphologies/example_morphology``. Note the following conventions:

-  soma needs to be a list (use ``soma[0]``, not ``soma``),
-  use ``soma`` as the name for the soma compartment,
-  use a name starting with ``dend`` for the dendrites.

::

    /* ----------------------------------------------------
    example_morphology.hoc

    This hoc file creates a neuron of the following shape:

                \       
                 \     
                  \   /
                   \ /
                    V
                    |
                    |
                    |
                    O
                
    Note the conventions:
     - soma needs to be a list (soma[0], not soma),
     - use soma for the soma compartment,
     - use a name starting with dend for the dendrites.
    -----------------------------------------------------*/


    create soma[1]
    create dend[3]

    soma[0] {
        pt3dadd(0, 0, 0, 25)
        pt3dadd(0, 0, 35, 25)
    }

    dend[0] {
        pt3dadd(0, 0, 35, 5)
        pt3dadd(0, 0, 150, 5)
    }

    dend[1] {
        pt3dadd(0, 0, 150, 2)
        pt3dadd(-50, 20, 200, 1)
    }

    dend[2] {
        pt3dadd(0, 0, 150, 2)
        pt3dadd(30, 0, 160, 2)
    }

    connect dend[0](0), soma[0](1)
    connect dend[1](0), dend[0](1)
    connect dend[2](0), dend[0](1)

Using NEURON mechanisms
-----------------------

Custom NEURON mechanisms can be loaded from external ``hoc`` files - see ``example2.py`` and ``example3.py``. Remeber to compile any ``mod`` files (if needed) using ``nrnivmodl`` (or ``mknrdll`` on Mac OS).  


Units
=====

Units follow the NEURON conventions.
The units in LFPy for given quantities are:

+-------------+-----------+
| What        | Unit      |
+=============+===========+
| Potentials  | [mV]      |
+-------------+-----------+
| Currents    | [nA]      |
+-------------+-----------+
| conductance | [S/cm2]   |
+-------------+-----------+
| capacitance | [muF/cm2] |
+-------------+-----------+
| Coordinates | [mum]     |
+-------------+-----------+

Note: resistance, conductance and capacitance are usually specific values, i.e per membrane area (lowercase r_m, g, c_m)
Depending on the mechanism files, some may use different units altogether, but this should be taken care of internally by NEURON.


.. Documentation
.. ===============
.. 
.. To rebuild this documentation from the LFPy-release root folder, issue in terminal
.. ::
..     export LC_ALL=en_US.UTF-8
..     sphinx-build-2.* -b html documentation/sphinx_files/. html
