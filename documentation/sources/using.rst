=============
Notes on LFPy
=============

Morphology files
================

Cell morphologies can be specified manually in a ``hoc`` file. For a simple example, see
``examples/morphologies/example_morphology.hoc``. Note the following conventions:

-  Sections should be named according to the following scheme:
   
   -  ``soma*[1]`` for somatic sections, ``*`` is optional
   -  ``dend*[]`` for dendritic sections
   -  ``apic*[]`` for apical dendrite sections
   -  ``axon*[]`` for axonal sections
-  Sections must be defined as SectionList (use ``soma[1]``, not ``soma``)


Also the morphologies exported from the NEURON simulator 
(for example using Cell Builder -> Export) should
work with LFPy, but often ``create soma`` must be corrected to
``create soma[1]`` in those files.


NEURON convention for creating morphology files in ``hoc``:
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

Other file formats
------------------

Support for SWC, NeuroLucida3 and NeuroML file formats is added, but consider this
experimental and is poorly tested. If there is something wrong with the files, strange behaviour may occur or LFPy may even fail
to load the desired morphology at all.


Using NEURON mechanisms
=======================

Custom NEURON mechanisms can be loaded from external ``.hoc``- or ``.py``-files - see ``example2.py`` and ``example3.py``.
Python function definitions with arguments can also be given as input to the ``Cell``-class, specifying model specific conductances etc.
Remeber to compile any ``mod`` files (if needed) using ``nrnivmodl`` (or ``mknrdll`` on Mac OS).

These model specific declarations is typically run after the ``Cell``-class try to read the morphology file,
and before optionally running the ``_set_nsegs()`` and ``_collect_geometry()`` procedures.


Units
=====

Units follow the NEURON conventions.
The units in LFPy for given quantities are:

+-------------+-----------+
| What        | Unit      |
+=============+===========+
| Potential   | [mV]      |
+-------------+-----------+
| Current     | [nA]      |
+-------------+-----------+
| Conductance | [S/cm2]   |
+-------------+-----------+
| Capacitance | [μF/cm2]  |
+-------------+-----------+
| Dimension   | [μm]      |
+-------------+-----------+

Note: resistance, conductance and capacitance are usually specific values, i.e per membrane area (lowercase r_m, g, c_m)
Depending on the mechanism files, some may use different units altogether, but this should be taken care of internally by NEURON.

