===============
Getting started
===============

Requirements
============

To install LFPy you will need the following:

1.  Python, seems to work fine with recent python since 2.6.x (2.6.6 or newer recommended), 2.7.x works fine.

2.  Python modules numpy, scipy, matplotlib

3.  `NEURON <http://www.neuron.yale.edu>`_ compiled as a Python module, so the following should execute without error in Python console:
    ::
    
        import neuron
        neuron.test()
    LFPy was tested with NEURON 7.2 and 7.3. 

4.  `Cython <http://cython.org>`_ (C-extensions for python) to speed up simulations of extracellular fields

Ubuntu 10.4LTS
--------------

The instructions below show how to meet all the requirements starting from a clean Ubuntu 10.4 installation. 

Start by installing the required packages
::

    sudo apt-get install mercurial autoconf libtool
    sudo apt-get install libxext-dev libncurses-dev
    sudo apt-get install bison flex
    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib
    sudo apt-get install ipython

Now get the source code of NEURON
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
    
(or `python setup.py install --user` if you want to install the Python package in your home folder). 
    
Now you should be able to `import neuron` from Python console
::

    cd $HOME
    ipython
    import neuron
    
You might want to add the folder with NEURON executables to your PATH, so that you can easily compile NEURON mechanisms using `nrnivmodl`
::
    
    export PATH=$PATH:$HOME/neuron/nrn/i686/bin

Now download Cython (Cython-0.15.1.tar.gz, or newer) from http://www.cython.org, unpack and install 
::

    sudo python setup.py install
    
You are now ready to download and install LFPy.


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

    We suggest you start with ``example1.py``
    :: 
    
        cd examples
        python example1.py
    
    or 
    ::
        
        cd examples
        ipython
        execfile("example1.py")
	
5.  HTML documentation is available at::

		LFPy-1.0/html/index.html


.. Documentation
.. ===============
.. 
.. To rebuild this documentation from the LFPy-release root folder, issue in terminal
.. ::
..     export LC_ALL=en_US.UTF-8
..     sphinx-build-2.* -b html documentation/sphinx_files/. html
