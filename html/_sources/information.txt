Getting started
===============

Requirements
------------

To install LFPy you will need the following:

1.  Python, seems to work fine with recent python since 2.6.x (2.6.6 or newer recommended), 2.7.x works fine.
    Not tested with Python 3.x.

2.  Python modules numpy, scipy, matplotlib

3.  `NEURON <http://www.neuron.yale.edu>`_ compiled as a Python module, so the following should execute without error in Python console:
    ::
    
        import neuron
        neuron.test()
    LFPy was tested to work with NEURON 7.2 and 7.3, we had issues with NEURON 7.1.

4.  `Cython <http://cython.org>`_ (C-extensions for python) to speed up simulations of extracellular fields. Tested with version > 1.4.

Installing NEURON with Python
=============================

Some users have difficulties install NEURON as a Python module,
depending on their platform. We will provide some explanations here, and otherwise direct to the NEURON download pages;
http://www.neuron.yale.edu/neuron/download and http://www.neuron.yale.edu/neuron/download/getstd. The NEURON forums are
also a useful resource for installation problems.

Dependencies Ubuntu 10.4LTS and other Debian based Linux versions
-----------------------------------------------------------------

The instructions below show how to meet all the requirements starting from a clean Ubuntu 10.4 installation of NEURON from the development branch. 

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
-----------------------

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


Dependencies and installation on Mac OSX
----------------------------------------

Most of the development work and testing of LFPy has been done on Max OSX 10.6.* Snow Leopard and 10.7.* Lion. Our preferred way of building Python
and NEURON have been through MacPorts; www.macports.org. 

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

Install NEURON from the bleeding edge source code:
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
    sudo make after_install
    
    #You should now have a working NEURON application under Applications. Small test;                                                                                                                   
    #sudo /Applications/NEURON-7.2/nrn/x86_64/bin/neurondemo                                                                                                                                            
    
    #Final step is to install neuron as a python module                                                                                                                                                 
    cd src/nrnpython
    sudo python setup.py install

If you rather use Enthough Python distribution, see this forum post: https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=2191 

Installation of LFPy
====================

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

    or ::
    
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
