.. LFPy documentation master file, created by
    sphinx-quickstart on Mon Oct  3 13:36:41 2011.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

.. image:: logo_liten.png

LFPy Homepage
=============

LFPy is a `Python <http://www.python.org>`_ package for calculation of extracellular potentials from multicompartment neuron models.
It relies on the `NEURON simulator <http://www.neuron.yale.edu/neuron/>`_ and uses the 
`Python interface <http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.001.2009/abstract>`_ it provides.

LFPy provides a set of easy-to-use Python classes for setting up your model, running your simulations and calculating the extracellular potentials arising from activity in your model neuron. If you have a model working in `NEURON <www.neuron.yale.edu>`_
already, it is likely that it can be adapted to work with LFPy.

The extracellular potentials are calculated from transmembrane currents in multi-compartment neuron models using the line-source method (Holt & Koch, J Comp Neurosci 1999), but a simpler point-source method is also available. The calculations assume that the neuron are surrounded by an infinite extracellular medium with homogeneous and frequency independent conductivity, and compartments are assumed to be at least at a minimal distance from the electrode (which can be specified by the user). For more information on the biophysics underlying the numerical framework used see this coming book chapter:

- K.H. Pettersen, H. Linden, A.M. Dale and G.T. Einevoll: Extracellular spikes and current-source density, in *Handbook of Neural Activity Measurement*, edited by R. Brette and A. Destexhe, Cambridge, to appear `[preprint PDF, 5.7MB] <http://arken.umb.no/~gautei/forskning/PettersenLindenDaleEinevoll-BookChapter-revised.pdf>`_

In the present version LFPy is mainly designed for simulation of single neurons.

LFPy was developed in the `Computational Neuroscience Group <http://compneuro.umb.no>`_, `Department of Mathemathical Sciences and Technology <http://www.umb.no/imt>`_,
at the `Norwegian University of Life Sciences <http://www.umb.no>`_ ,
in collaboration with the `Laboratory of Neuroinformatics <http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics>`_, 
`Nencki Institute of Experimental Biology <http://www.nencki.gov.pl>`_, Warsaw, Poland. The effort was supported by 
`International Neuroinformatics Coordinating Facility <http://incf.org>`_ (`INCF <http://incf.org>`_) and the `The Research Council of Norway <http://www.forskningsradet.no/english/>`_ (eScience, NevroNor).

This scientific software is released under the GNU Public License `GPLv3 <http://www.gnu.org/copyleft/gpl.html>`_.

Contents
========
.. toctree::
    :maxdepth: 3
   
    information
    tutorial
    using
    contributors
    contact
    classes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

