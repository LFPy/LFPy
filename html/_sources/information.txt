What is LFPy?
=============

LFPy is a `Python <http://www.python.org>`_-module for calculating extracellular potentials from multicompartment neuron models.
It relies on calculating compartmental membrane currents using the `NEURON <www.neuron.yale.edu>`_-simulator, a under the hood, and
provides a set of easy-to-use Python-classes for setting up your model in Python, and run your simulations and have simultaneous
model predicitions of the extracellular potentials arising from activity of your model neuron. If you have a model working in `NEURON <www.neuron.yale.edu>`_
already, it is likely that it can be adapted to work with LFPy.

The extracellular potentials are calculated using the following assumptions:
 * The extracellular potential is zero for every compartment when calculating membrane currents
 * The transmembrane current of each compartment are treates as either point- or line-sources in the extracellular space
 * Extracellular potential can not be calculated inside the model cell's volume, LFPy adjust the distance to each compartment to it's radius
 if the extracellular distance is less than this radius
 * Infinite and homogenous extracellular conductivity of the extracellular conductivity
 * Linear and frequency independent extracellular medium
 * Linear superposition of the extracellular contribution from all current-sources.


LFPy was developed in the `Computational Neuroscience Group <http://compneuro.umb.no>`_, `Department of Mathemathical Sciences and Technology <http://www.umb.no/imt>`_,
at the `Norwegian University of Life Sciences <http://www.umb.no>`_ ,
in collaboration with the `Laboratory of Neuroinformatics <http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics>`_, 
`Nencki Institute of Experimental Biology <http://www.nencki.gov.pl>`_, Warsaw, Poland. The effort was supported by 
`International Neuroinformatics Coordinating Facility <http://incf.org>`_ (`INCF <http://incf.org>`_) and the e-science program of `The Research Council of Norway <http://www.forskningsradet.no/english/>`_.


Getting LFPy
============

Downlad stable versions of LFPy here:

`LFPy-0.9.tar.gz <http://compneuro.umb.no/LFPy/downloads/LFPy-0.9.tar.gz>`_

Or, download the development version of LFPy using `subversion <http://subversion.apache.org/>`_ into a local folder:
::
    
    svn checkout http://bebiservice.umb.no/svn-public/LFPy-release

The development code is then found under "trunk/", and different numbered releases under "tags/".

To browse the source codes online, see http://bebiservice.umb.no/projects-public/LFPy-release/browser.



Getting started
===============

Dependencies
------------

To install LFPy you will need the following:

1.  Python, seems to work fine with recent python since 2.6.x (2.6.6 or newer recommended), 2.7.x works fine.
    Not tested with Python 3.x.

2.  Python modules numpy, scipy, matplotlib

3.  `NEURON <http://www.neuron.yale.edu>`_ compiled as a Python module, so the following should execute without error in Python console:
    ::
    
        import neuron
        neuron.test()
    
    If this step fails, see the next section.
    
    LFPy was tested to work with NEURON 7.2 and the 7.3 development branch, we had some issues with a precompiled version of NEURON 7.1.

4.  `Cython <http://cython.org>`_ (C-extensions for python) to speed up simulations of extracellular fields. Tested with version > 1.4.,
    and known to fail with version 1.1.


Installing LFPy
---------------

1.  Download the LFPy source distribution (LFPy-x.x.tar.gz) and unpack.

2.  In "LFPy-x.x/" run
    ::
    
        sudo python setup.py install
    
    or ::
    
        python setup.py install --user
    
    in case you want to install in your home directory (e.g. if you do not have root access).
    
    If you only want to have LFPy in one place, you can add this working folder to your $PYTHONPATH, and just compile the Cython extensions;
    ::
    
        python setup.py build_ext -i
    
3.  In a fresh python-session you should now be able to 
    ::  
    
        import LFPy

4.  There are some basic usage examples provided in 
    ::
    
        LFPy-x.x/examples/

    We suggest you start with ``example1.py``
    :: 
    
        cd examples
        python example1.py

    or ::
    
        cd examples
        ipython
        >>> run "example1.py"
    
    For the example2.py and example3.py, the NEURON mechanisms (.mod-files) must be compiled using "nrnivmodl" from your NEURON installation first.
    
5.  HTML documentation is available at::

		LFPy-x.x/html/index.html



Installing NEURON with Python
-----------------------------

Some users have difficulties install NEURON as a Python module,
depending on their platform. 
We will provide some explanations here, and otherwise direct to the NEURON download pages;
http://www.neuron.yale.edu/neuron/download and http://www.neuron.yale.edu/neuron/download/getstd. The NEURON forums are
also a useful resource for installation problems.

Dependencies: Ubuntu 10.4LTS and other Debian-based Linux versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The instructions below show how to meet all the requirements starting from a clean Ubuntu 10.4 for the installation of NEURON from the development branch. 

Start by installing the required packages
::

    sudo apt-get install mercurial autoconf libtool
    sudo apt-get install libxext-dev libncurses-dev
    sudo apt-get install bison flex
    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib
    sudo apt-get install ipython

The cython version provided in Ubuntu 10.4LTS is out of date, compile a more recent version yourself.
Download Cython (Cython-0.15.1.tar.gz, or newer) from `Cython.org <http://www.cython.org>`_, unpack and install;
::
    
    sudo python setup.py install

Linux/Unix installation
^^^^^^^^^^^^^^^^^^^^^^^

Now get the source code of NEURON using mercurial
::

    cd $HOME
    mkdir neuron
    cd neuron

    hg clone http://www.neuron.yale.edu/hg/neuron/iv
    hg clone http://www.neuron.yale.edu/hg/neuron/nrn

Compile and install InterViews
::
    
    cd iv
    sh build.sh 
    ./configure --prefix=`pwd`
    make
    make install
    
Compile and install NEURON
::

    cd ../nrn
    sh build.sh 
    ./configure --prefix=`pwd` --with-iv=$HOME/neuron/iv --with-nrnpython=/usr/bin/python
    make
    make install

Install NEURON as a Python module
::

    cd src/nrnpython/
    sudo python setup.py install
    
(or ``python setup.py install --user`` if you want to install the Python package in your home folder). 
    
Now you should be able to ``import neuron`` from Python console and run a small test with success;
::

    cd $HOME
    ipython
    import neuron
    neuron.test()
    
You might want to add the folder with NEURON executables to your PATH, so that you can easily compile NEURON mechanisms using ``nrnivmodl``
::
    
    export PATH=$PATH:$HOME/neuron/nrn/x86_64/bin


Dependencies and installation on Mac OS X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the development work and testing of LFPy has been done on Max OS X 10.6.* Snow Leopard and 10.7.* Lion. Our preferred way of building Python 
has been through MacPorts; http://www.macports.org. Here is an step-by-step explanation on how to compile NEURON agains that installation of Python.

To start using MacPorts, follow the instructions on http://www.macports.org/install.php.

Building a python 2.7 environment using MacPorts issue in Terminal:
::
    
    sudo port install python27 py27-ipython py27-numpy py27-matplotlib py27-scipy py27-cython

Make the installed Python and IPython default:
::

    sudo port select --set python python27
    sudo port select --set ipython ipython27
    
Install the necessary packages for cloning into repository and compiling NEURON:
::

    sudo port install automake autoconf libtool libxext xorg-libXext ncurses mercurial bison flex

Install NEURON from the bleeding edge source code. The following recipe assumes a 64 bit build of NEURON and Python on OSX 10.7 Lion, so change
"x86_64-apple-darwin10.7.0" throughout to facilitate your system accordingly;
::

    #create a directory in home directory                                                                                                                                                               
    cd $HOME
    mkdir nrn64
    cd nrn64
    
    #creating directories                                                                                                                                                                               
    sudo mkdir /Applications/NEURON-7.2
    sudo mkdir /Applications/NEURON-7.2/iv
    sudo mkdir /Applications/NEURON-7.2/nrn
    
    #Downloading bleeding edge source code                                                                                                                                                              
    hg clone http://www.neuron.yale.edu/hg/neuron/iv
    hg clone http://www.neuron.yale.edu/hg/neuron/nrn
    cd iv
    hg up
    cd ../nrn
    hg up
    cd ..
        
    #compiling and installing IV under folder /Applications/nrn7.2                                                                                                                                             
    cd iv
    sh build.sh
    ./configure --prefix=/Applications/NEURON-7.2/iv \
        --build=x86_64-apple-darwin10.7.0 --host=x86_64-apple-darwin10.7.0
    
    make
    sudo make install
    
    #Building NEURON with InterViews, you may have to alter the path --with-nrnpython=/python-path                                                                                                      
    cd $HOME/nrn64/nrn
    sh build.sh
    ./configure --prefix=/Applications/NEURON-7.2/nrn --with-iv=/Applications/NEURON-7.2/iv \
        --with-x --x-includes=/usr/X11/include/ --x-libraries=/usr/X11/lib/ \
        --with-nrnpython=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python \
        --host=x86_64-apple-darwin10.7.0 --build=x86_64-apple-darwin10.7.0
    
    make
    sudo make install
    sudo make install after_install
    
    #You should now have a working NEURON application under Applications. Small test;                                                                                                                   
    #sudo /Applications/NEURON-7.2/nrn/x86_64/bin/neurondemo                                                                                                                                            
    
    #Final step is to install neuron as a python module                                                                                                                                                 
    cd src/nrnpython
    sudo python setup.py install

If you prefer to use Enthough Python distribution, see this forum post: https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=2191 

Windows
^^^^^^^

We have not tested this, but there is probably some way or another to compile NEURON as a Python module on the Windows platform.
If so, LFPy should work as well. 