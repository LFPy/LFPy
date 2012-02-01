Requirements
============

The basic requirements for the release branch of LFPy is as follows

1.  Subversion, in order to download the code from the repository
2.  Python, seems to work fine with recent python since 2.6.x (2.6.6 or newer recommended), 2.7.x works fine.
3.  Python modules numpy, scipy, matplotlib
4.  NEURON (from www.neuron.yale.edu) compiled as a Python module, so the following should execute without error in the terminal:
    ::
    
        ipython
        >>> import neuron
        >>> neuron.test()
   
5.  Cython (C-extensions for python, cython.org) to speed up simulations of extracellular fields

Linux
-----

In Ubuntu 10.4LTS you may use Synaptic to install subversion, python2.6-dev, numpy, scipy and matplotlib. Note that the cython version in Ubuntu repositories (0.11) will not work, please install the current version from cython.org (tested with 0.15.1). The easiest way to install NEURON as a Python module is to download a .deb package from Eilif Muller's webpage (http://neuralensemble.org/people/eilifmuller/software.html). (The command to compile NEURON
mechanisms is then /opt/nrn/i686/bin/nrnivmodl)

Installation
============  

New installation instructions:

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
    

1.  In a terminal, ``'cd'`` to a directory where you want your LFPy files to be, as an example this could be '/home/yourname/work'. 

2.  Download LFPy using SVN by typing:
    :: 
    
        svn checkout https://bebiservice.umb.no/svn-private/LFPy-release

    This will create a folder named 'LFPy-release'  in the current folder containing all files belonging to the LFPy package.

3.  Now you need to add the following lines to your '.bashrc' or '.bash_profile' or similar file:
    :: 
    
        LFPYPATH="/home/yourname/work/LFPy-release/"
        PYTHONPATH="${PYTHONPATH}:${LFPYPATH}"
        export LFPYPATH
        export PYTHONPATH

    Change the exact path to where you downloaded the files in previous steps. This is to make the LFPy installation available everywhere (as we don't install LFPy as a "proper" python module at the moment)

4.  In order to speed up LFP-calculations, the Cython code (lfpycalc.pyx) need to be compiled every time it has been updated from the terminal;
    :: 
    
        cd $LFPYPATH/LFPy
        python setup.py build_ext -i

    Otherwise the corresponding .py files will be used, which is slower. 
    
5.  Compile the NEURON mechanisms in LFPy-release/neuron (using nrnivmodl).

6.  Now try if it all works. In a python shell in a fresh terminal, try
    ::
    
        import LFPy

    if there are no complaints, it should be all done!

7.  There are some basic usage examples provided in 
    ::
    
        $LFPYPATH/examples/

    To execute:
    :: 
    
        cd $LFPYPATH/examples/
        python script.py
        #or
        ipython
        >>>run script.py

Units
=====

As of now, units follow the NEURON conventions.
The units in LFPy for given quantities are:

+-------------+-----------+
| What        | unit      |
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
Depending on the mechanism files, some may use different units altogether, but this should be taken care of internally by NEURON, right?.
