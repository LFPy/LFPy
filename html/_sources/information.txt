===============
Getting started
===============

Requirements
============

To install LFPy you will need the following:

1.  Python, seems to work fine with recent python since 2.6.x (2.6.6 or newer recommended), 2.7.x works fine.
2.  Python modules numpy, scipy, matplotlib
3.  NEURON (from http://www.neuron.yale.edu) compiled as a Python module, so the following should execute without error in Python console:
    ::
    
        import neuron
        neuron.test()

	
   
4.  Cython (C-extensions for python, cython.org) to speed up simulations of extracellular fields

Linux
-----

In Ubuntu 10.4LTS you may use Synaptic to install python2.6-dev, numpy, scipy and matplotlib. Note that the cython version in Ubuntu repositories (0.11) will not work, please install the current version from http://cython.org (tested with 0.15.1). The easiest way to install NEURON as a Python module is to download a .deb package from 
Eilif Muller's webpage: http://neuralensemble.org/people/eilifmuller/software.html. 

(The command to compile NEURON
mechanisms is then /opt/nrn/i686/bin/nrnivmodl)

Mac OS X
--------

The easiest way to install Neuron as a Python module is again to use Eilif Muller's unofficial installer (http://neuralensemble.org/people/eilifmuller/software.html) which works with the Enthought Python Distribution (EPD). EPD provides numpy, scipy and matplotlib, so you only need to install cython from http://cython.org. 

Installation
============  

1.  Download the LFPy source distribution (LFPy-1.0.tar.gz) and unpack.

2.  In LFPy-1.0 run
    ::
    
        sudo python setup.py install
    
    or ::
    
        python setup.py install --user
    
    in case you want to install in your home directory (e.g. if you do not have root access).
    
3.  Now you should be able to 
    ::  
    
        import LFPy
    

4.  There are some basic usage examples provided in 
    ::
    
        LFPy-1.0/examples/

    To execute:
    :: 
    
        cd examples
        python script.py
        #or
        ipython
        >>>run script.py

	We suggest you start with ``example1.py``.
	
5.  HTML documentation is available at::

		LFPy-1.0/html/index.html


.. Documentation
.. ===============
.. 
.. To rebuild this documentation from the LFPy-release root folder, issue in terminal
.. ::
..     export LC_ALL=en_US.UTF-8
..     sphinx-build-2.* -b html documentation/sphinx_files/. html
