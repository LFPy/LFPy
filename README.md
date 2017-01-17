====
LFPy
====

LFPy is a Python-module for calculation of extracellular potentials from multicompartment neuron models.
It relies on the NEURON simulator (http://www.neuron.yale.edu/neuron) and uses the 
Python interface (http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.001.2009/abstract) it provides.

LFPy provides a set of easy-to-use Python classes for setting up your model, running your simulations and calculating the extracellular potentials arising from activity in your model neuron. If you have a model working in NEURON (www.neuron.yale.edu)
already, it is likely that it can be adapted to work with LFPy.

The extracellular potentials are calculated from transmembrane currents in multi-compartment neuron models using the line-source method (Holt & Koch, J Comp Neurosci 1999),
but a simpler point-source method is also available. The calculations assume that the neuron are surrounded by an infinite extracellular medium with homogeneous and frequency
independent conductivity, and compartments are assumed to be at least at a minimal distance from the electrode (which can be specified by the user). For more information on
the biophysics underlying the numerical framework used see this coming book chapter:

- K.H. Pettersen, H. Linden, A.M. Dale and G.T. Einevoll: Extracellular spikes and current-source density, in *Handbook of Neural Activity Measurement*, edited by R. Brette and A. Destexhe, Cambridge, to appear (preprint PDF, 5.7MB http://arken.umb.no/~gautei/forskning/PettersenLindenDaleEinevoll-BookChapter-revised.pdf

In the present version, LFPy is mainly designed for simulation of single neurons, described in our recent paper on the package in Frontiers in Neuroinformatics entitled "LFPy: A tool for biophysical simulation of extracellular potentials generated by detailed model neurons".
The article can be found at http://dx.doi.org/10.3389%2Ffninf.2013.00041

Citation:
Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH and Einevoll GT (2013). LFPy: A tool for biophysical simulation of extracellular potentials generated by detailed model neurons. Front. Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041

LFPy was developed in the Computational Neuroscience Group, Department of Mathemathical Sciences and Technology (http://www.nmbu.no/imt),
at the Norwegian University of Life Sciences (http://www.nmbu.no),
in collaboration with the Laboratory of Neuroinformatics (http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics), 
Nencki Institute of Experimental Biology (http://www.nencki.gov.pl), Warsaw, Poland. The effort was supported by 
International Neuroinformatics Coordinating Facility (http://incf.org), the Research Council of Norway (http://www.forskningsradet.no/english) (eScience, NevroNor) and EU-FP7 (BrainScaleS, http://www.brainscales.org).

For updated information on LFPy and online documentation, see the LFPy homepage (http://LFPy.github.io).

This scientific software is released under the GNU Public License GPLv3.

===========
Code status
===========
[![Build Status](https://travis-ci.org/espenhgn/LFPy.svg?branch=dev)](https://travis-ci.org/espenhgn/LFPy)
[![Coverage Status](https://coveralls.io/repos/github/espenhgn/LFPy/badge.svg?branch=dev)](https://coveralls.io/github/espenhgn/LFPy)


============
Requirements
============

To install LFPy you will need the following:

- Python modules numpy, scipy and matplotlib
- NEURON (from http://www.neuron.yale.edu) compiled as a Python module, so the following should execute without error in Python console::
    
    import neuron
    neuron.test()

- Cython (C-extensions for python, http://cython.org) to speed up simulations of extracellular fields


============
Installation
============

There are few options to install LFPy:

1.  From the Python Package Index with only local access using pip
    ::
        
        pip install --user LFPy


    as sudoer:
    ::
    
        sudo pip install LFPy


2.  From the Python Package Index with only local access using easy_install
    ::
    
        easy_install --user LFPy

    as sudoer:
    ::
    
        sudo easy_install LFPy

3.  From source:
    ::
    
        tar -xzf LFPy-x.x.tar.gz
        cd LFPy-x.x
        (sudo) python setup.py install (--user)

4.  Development version from the GitHub repository:
    ::

        git clone https://github.com/LFPy/LFPy.git
        cd LFPy
        (sudo) python setup.py install (--user)


=============
Documentation
=============

To generate the html documentation, issue from the LFPy source code directory:
::
    
    sphinx-build -b html /path/to/LFPy/documentation/sources path/to/dest

The main html file is now in path/to/dest/index.html

