.. LFPy documentation master file, created by
    sphinx-quickstart on Mon Oct  3 13:36:41 2011.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.


LFPy documentation
==================

.. image:: logo_liten.png

What is LFPy?
=============

LFPy is a `Python <http://www.python.org>`_-package for calculation of extracellular potentials from multicompartment neuron models.
It relies on the `NEURON simulator <http://www.neuron.yale.edu/neuron/>`_ and uses the 
`Python interface <http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.001.2009/abstract>`_ it provides.


LFPy provides a set of easy-to-use Python classes for setting up your model, running your simulations and calculating the extracellular potentials arising from activity in your model neuron. If you have a model working in `NEURON <www.neuron.yale.edu>`_
already, it is likely that it can be adapted to work with LFPy.


The extracellular potentials are calculated using the following assumptions:
 * The extracellular potential is zero for every compartment when calculating membrane currents
 * The transmembrane current of each compartment are treates as either point- or line-sources in the extracellular space
 * Extracellular potential can not be calculated inside the model cell's volume, LFPy adjust the distance to each compartment to it's radius if the extracellular distance is less than this radius
 * Infinite and homogenous extracellular conductivity of the extracellular conductivity
 * Linear and frequency independent extracellular medium
 * Linear superposition of the extracellular potential contribution from all current-sources.


LFPy was developed in the `Computational Neuroscience Group <http://compneuro.umb.no>`_, `Department of Mathemathical Sciences and Technology <http://www.umb.no/imt>`_,
at the `Norwegian University of Life Sciences <http://www.umb.no>`_ ,
in collaboration with the `Laboratory of Neuroinformatics <http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics>`_, 
`Nencki Institute of Experimental Biology <http://www.nencki.gov.pl>`_, Warsaw, Poland. The effort was supported by 
`International Neuroinformatics Coordinating Facility <http://incf.org>`_ (`INCF <http://incf.org>`_) and the e-science program of `The Research Council of Norway <http://www.forskningsradet.no/english/>`_.



Contents
========
.. toctree::
    :maxdepth: 3
   
    information
    tutorial
    using
    classes
    contributors
..    citations

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

