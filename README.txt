
LFPy
====

LFPy is a Python-module for calculation of extracellular potentials from multicompartment neuron models.
It relies on the NEURON simulator (http://www.neuron.yale.edu/neuron) and uses the 
Python interface (http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.001.2009/abstract) it provides.


LFPy provides a set of easy-to-use Python classes for setting up your model, running your simulations and calculating the extracellular potentials arising from activity in your model neuron. If you have a model working in `NEURON <www.neuron.yale.edu>`_
already, it is likely that it can be adapted to work with LFPy.

The extracellular potentials are calculated from transmembrane currents in multi-compartment neuron models using the line-source method (Holt & Koch, J Comp Neurosci 1999), but a simpler point-source method is also available. The calculations assume that the neuron are surrounded by an infinite extracellular medium with homogeneous and frequency independent conductivity, and compartments are assumed to be at least at a minimal distance from the electrode (which can be specified by the user). For more information on the biophysics underlying the numerical framework used see this coming book chapter:

- K.H. Pettersen, H. Linden, A.M. Dale and G.T. Einevoll: Extracellular spikes and current-source density, in *Handbook of Neural Activity Measurement*, edited by R. Brette and A. Destexhe, Cambridge, to appear `[preprint PDF, 5.7MB] <http://arken.umb.no/~gautei/forskning/PettersenLindenDaleEinevoll-BookChapter-revised.pdf>`_

In the present version LFPy is mainly designed for simulation of single neurons.

LFPy was developed in the Computational Neuroscience Group (http://compneuro.umb.no), Department of Mathemathical Sciences and Technology <http://www.umb.no/imt>`_,
at the Norwegian University of Life Sciences (http://www.umb.no) ,
in collaboration with the Laboratory of Neuroinformatics (http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics), 
Nencki Institute of Experimental Biology (http://www.nencki.gov.pl), Warsaw, Poland. The effort was supported by 
International Neuroinformatics Coordinating Facility (http://incf.org) and the the Research Council of Norway (http://www.forskningsradet.no/english) (eScience, NevroNor).

For updated information on LFPy and online documentation, see the LFPy homepage (http://compneuro.umb.no/LFPy).

This scientific software is released under the GNU Public License GPLv3.

Requirements
============

To install LFPy you will need the following:

- Python modules numpy, scipy and matplotlib
- NEURON (from http://www.neuron.yale.edu) compiled as a Python module, so the following should execute without error in Python console::
    
    import neuron
    neuron.test()

- Cython (C-extensions for python, http://cython.org) to speed up simulations of extracellular fields


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

4.  Development version from subversion repository:
    ::
    
        svn checkout http://bebiservice.umb.no/svn-public/LFPy-release/
        cd LFPy-release/trunk
        (sudo) python setup.py install (--user)
