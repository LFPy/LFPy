
Download LFPy
=============

Downlad the latest stable version of LFPy from the Python Package Index: `http://pypi.python.org/pypi/LFPy <https://pypi.python.org/pypi/LFPy>`_

Or, download the development version of LFPy using `git <https://git-scm.com>`_ from `GitHub.com <https://github.com/LFPy/LFPy>`_ into a local folder:
::

    cd <where to put repositories>
    git clone https://github.com/LFPy/LFPy.git

The LFPy source code and examples is then found under "LFPy".

The stable versions of LFPy can be accessed by listing and checking out tags, e.g.,
::

    cd <path to LFPy>
    git tag -l
    git checkout <tag name>


To browse the documentation and source codes online, see `http://lfpy.readthedocs.io/classes.html <http://lfpy.readthedocs.io/classes.html>`_ or `https://github.com/LFPy/LFPy <https://github.com/LFPy/LFPy>`_


Developing LFPy
===============

As development of LFPy is now moved onto GitHub (https://github.com/LFPy), one can now fully benefit on working with forks of LFPy, implement new features, and share code improvements through pull requests.
We hope that LFPy can be improved continously through such a collaborative effort.

A list of various LFPy issues, bugs, feature requests etc. is found `here <https://github.com/LFPy/LFPy/issues>`_.
If you want to contribute code fixes or new features in LFPy, send us a `pull request <https://github.com/LFPy/LFPy/pulls>`_.


Getting started
===============

Dependencies
------------

To install LFPy you will need the following:

1.  Python. LFPy's unit-test suite is integrated with and continuously tested using `Travis-CI <https://travis-ci.org>`_. Tests are run using NEURON 7.4 and Python 2.7, 3.4, 3.5 and 3.6, as well as other Python dependencies listed next.
    The code build testing status, results of the last test, test coverage test using `Coveralls <https://coveralls.io>`_ and documentation status can be seen here:

    .. image:: https://travis-ci.org/LFPy/LFPy.svg?branch=master
        :target: https://travis-ci.org/LFPy/LFPy
        :alt: TravisCI Status

    .. image:: https://coveralls.io/repos/github/LFPy/LFPy/badge.svg?branch=master
        :target: https://coveralls.io/github/LFPy/LFPy?branch=master
        :alt: Coveralls Status

    .. image:: https://readthedocs.org/projects/lfpy/badge/?version=latest
        :target: http://lfpy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


2.  Python modules setuptools, numpy, scipy, matplotlib, Cython, h5py, mpi4py, csa

    Depending on how LFPy is obtained, missing dependencies will be installed automatically (using ``pip``). Manual installation of dependencies can be done using ``pip`` and the  ``requirements.txt`` file which is supplied with the LFPy source codes (since v2.0).
    See next section for details.

3.  Some example files may rely on additional Python dependencies. If some examples should fail to run due to some ``ImportError``, identify the missing dependency name and run
    ::

        pip install <package>

    Alternatively, use the system default package manager, for example
    ::

        sudo apt install python-<package> # or
        conda install package

4.  `NEURON <http://www.neuron.yale.edu>`_ compiled as a Python module, so the following should execute without error in Python console:
    ::

        import neuron
        neuron.test()

    If this step fails, see the next section.

    LFPy now requires NEURON 7.4 or newer, and should be compiled with MPI in order to support new parallel functionality in LFPy, i.e., execution of networks in parallel.

5.  `Cython <http://cython.org>`_ (C-extensions for python) to speed up simulations of extracellular fields. Tested with version > 0.14,
    and known to fail with version 0.11. LFPy works without Cython, but simulations may run slower and is therefore not recommended.


Installing LFPy
---------------

There are few options to install LFPy:

1.  From the Python Package Index with only local access using pip
    ::

        pip install LFPy --user

    as sudoer (in general not recommended as system Python files may be overwritten):
    ::

        sudo pip install LFPy

    Upgrading LFPy from the Python package index (without attempts at upgrading dependencies):
    ::

        pip install --upgrade --no-deps LFPy --user

    LFPy release candidates can be installed as
    ::

        pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple LFPy --user

2.  From source:
    ::

        tar -xzf LFPy-x.x.tar.gz
        cd LFPy-x.x
        (sudo) pip install -r requirements.txt (--user) # install dependencies
        (sudo) python setup.py install (--user)


3.  From development version in our git (https://git-scm.com) repository:
    ::

        git clone https://github.com/LFPy/LFPy.git
        cd LFPy
        # git tag -l # list versions
        # git checkout <tag name> # choose particular version
        (sudo) pip install -r requirements.txt (--user) # install dependencies
        (sudo) python setup.py install (--user)


4.  If you only want to have LFPy in one place, you can also build the LFPy Cython and NEURON NMODL extensions in the source directory.
    That can be quite useful for active LFPy development.
    ::

        python setup.py develop --user


    With any changes in LFPy ``*.pyx`` source files, rebuild LFPy.

    In case of problems, it may be necessary to remove temporary and compiled files from the git repository before new attempts at building LFPy can be made:
    ::

        git clean -n # list files that will be removed
        git clean -fd # remove files


In a fresh terminal and python-session you should now be able to issue:
::

    import LFPy


Uninstalling LFPy
-----------------

Some times it may be necessary to remove installed versions of LFPy. Depending on how LFPy was installed in the first place, it should under most circumstances suffice to execute
::

    (sudo) pip uninstall LFPy

If several versions was installed in the past, repeat until no more LFPy files are found.


Documentation
=============

To generate the html documentation using Sphinx, issue from the LFPy source code directory:
::

    sphinx-build -b html <path to LFPy>/doc <path to output>

The main html file is in ``<path to output>/index.html``. Numpydoc and the ReadTheDocs theme may be needed:
::

    pip install numpydoc --user
    pip install sphinx-rtd-theme --user


Installing NEURON with Python
-----------------------------

For most users, and even though NEURON (http://neuron.yale.edu) provides a working Python interpreter, making NEURON work as a Python module may be quite straightforward using pre-built
Python distributions such as the Anaconda Scientific Python distribution (http://continuum.io) or Enthought Canopy (https://www.enthought.com/products/canopy/). We here provide some short step-by-step recipes on
how to set up a working Python environment using Anaconda with the standard pre-built NEURON binaries on Linux, OSX and Windows.


Ubuntu 18.04.1 LTS 64-bit with Anaconda Scientific Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Probably the simplest solution relying on no source code compilation.
This recipe is tested on a clean installation of Ubuntu 18.04.1 running in VirtualBox (https://www.virtualbox.org).
The VirtualBox guest additions were installed.

1.  Download and install Anaconda using the 64-bit Linux installer script from https://www.anaconda.com/download (https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh)

    Open Terminal application, then issue:
    ::

        cd $HOME/Downloads
        bash Anaconda3-5.3.1-Linux-x86_64.sh

    Accept the license and default options. Allow the installer to modify your ``.bashrc`` file. Installing MS Visual Studio Code (VSCode) is optional.

2.  Download and install the 64-bit Debian/Ubuntu .deb file with NEURON from https://www.neuron.yale.edu/neuron/download (https://neuron.yale.edu/ftp/neuron/versions/v7.6/7.6.2/nrn-7.6.2.x86_64-linux-py-36-35-27.deb)
3.  The ``readline``, ``ncurses``, ``libtool`` library files as well as ``make`` may be needed by NEURON. Install them from the terminal calling
    ::

        sudo apt install libreadline-dev libncurses-dev libtool make

4.  Edit your ``.bashrc`` or similar file located in the ``$HOME`` folder, e.g., by calling in the terminal ``gedit $HOME/.bashrc``, to include the lines:
    ::

        # make NEURON python module available to Anaconda python
        export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"

5.  Open a fresh terminal window (or type ``source ~/.bashrc`` in the terminal)

6.  Activate the ``base`` conda environment:
    ::

        conda activate


7.  Install LFPy dependencies (not installed by default) using conda
    ::

        conda install mpi4py # numpy matplotlib scipy h5py


8.  Clone into LFPy using Git (https://git-scm.com), installable by calling ``sudo apt install git`` in the terminal:
    ::

        git clone https://github.com/LFPy/LFPy.git
        cd LFPy


9.  Build LFPy from source inplace (without moving files)
    ::

        python setup.py develop --user

    or perform a local installation of LFPy:
    ::

        python setup.py install --user

9.  Test the installation from the terminal
    ::

        nosetests

    which will run through the LFPy test suite. Hopefully without errors.


Python 2.7
""""""""""

1.  Follow the above steps up until point 6. Then in the terminal create a new environment based on Python 2.7
    (assuming that the Python 3 version of Anaconda was installed):
    ::

        conda create -n py27 python=2.7 anaconda
        conda activate py27

2.  Continue with step 7 above.


Ubuntu 18.04.1 LTS w. system Python 3.6
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Another option is to rely on the system Python installation (Python 3.6.7).
Make sure that nothing in ``$PATH`` points to an Anaconda-installed version of Python (may break e.g., ``pip3``)

1.  Install dependencies through the terminal:
    ::

        sudo apt install libreadline-dev libncurses-dev libtool make
        sudo apt install ipython3 cython3 jupyter-notebook python3-nose \
          python3-numpy python3-scipy python3-matplotlib python3-mpi4py python3-h5py

2.  Download and install the 64-bit Debian/Ubuntu .deb file with NEURON from https://www.neuron.yale.edu/neuron/download (https://neuron.yale.edu/ftp/neuron/versions/v7.6/nrn-7.6.x86_64-linux.deb)

3.  Edit your ``.bashrc`` or similar file located in the ``$HOME`` folder, e.g., by calling in the terminal ``gedit $HOME/.bashrc``, to include the lines:
    ::

        # make NEURON python module available to Anaconda python
        export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"

4.  Open a fresh terminal window (or type ``source ~/.bashrc`` in the terminal)

5.  Clone into LFPy using Git (https://git-scm.com), installable by calling ``sudo apt install git`` in the terminal:
    ::

        git clone https://github.com/LFPy/LFPy.git
        cd LFPy


6.  Build LFPy from source inplace (without moving files)
    ::

        python3 setup.py develop --user

    or perform a local installation of LFPy:
    ::

        python3 setup.py install --user

7.  Test the installation from the terminal
    ::

        nosetests3

    which will run through the LFPy test suite. Hopefully without errors.


OSX 10.12.x with Anaconda Scientific Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By far the simplest solution relying on no source code compilation.

1.  Download and install Anaconda using the 64-bit graphical installer from http://continuum.io/downloads
2.  Download and install the 64-bit Mac ``.pkg`` file with NEURON from http://www.neuron.yale.edu/neuron/download.
    Do not choose to let the NEURON installer edit the ``~/.bash_profile`` file. The default file to edit is ``~/.profile`` (see below).
3.  Edit your .profile or similar file located in the $HOME folder, e.g., by calling in the Terminal.app ``open -t $HOME/.profile``, to include the lines:
    ::

        # make nrniv, mknrnivmodl, etc. available from the command line
        export PATH=/Applications/NEURON-7.5/nrn/x86_64/bin/:$PATH

        # Append the path to the NEURON python extension module to PYTHONPATH
        export PYTHONPATH=/Applications/NEURON-7.5/nrn/lib/python:$PYTHONPATH


4.  Open a fresh terminal window

5.  Install LFPy dependencies (not installed by default) using conda
    ::

        conda install mpi4py

6.  Clone into LFPy using Git:
    ::

        git clone https://github.com/LFPy/LFPy.git

7.  Build LFPy from source (without moving files)
    ::

        python setup.py develop

8.  Test the installation from the terminal
    ::

        python -c "import LFPy"
        NEURON -- VERSION 7.5 master (6b4c19f) 2017-09-25
        Duke, Yale, and the BlueBrain Project -- Copyright 1984-2016
        See http://neuron.yale.edu/neuron/credits

If everything worked, you now have a working Python/NEURON/LFPy environment.



Windows with Anaconda Scientific Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Windows 10 Pro/Education (64-bit) install instructions:

1.  Download and install Anaconda Python from https://www.anaconda.com/download.
2.  Download and install NEURON from https://www.neuron.yale.edu/neuron/download.
    Tick the box to "Set DOS environment" (Otherwise Anaconda Python will not find the NEURON python module)
3.  Download and install the Visual Studio C++ Build Tools 2015 from: https://www.microsoft.com/en-us/download/details.aspx?id=48159.
4.  Download and install Git from https://git-scm.com/downloads
5.  Download and install Microsoft MPI from the Official Microsoft Download Center: https://www.microsoft.com/en-us/download/details.aspx?id=55494.
    Choose the file "MSMpiSetup.exe".
6.  Open the Anaconda Prompt under the Anaconda* folder in the start menu
7.  Optionally, create a separate conda environment for LFPy:
    ::

        conda create -n LFPy python=3.6 mpi4py numpy scipy matplotlib h5py Cython jupyter
        conda activate LFPy

    For every future session, the LFPy environment needs to be used by issuing `conda activate LFPy`. From hereon, skip to step 9. 
8.  Install additional LFPy dependencies listed in `requirements.txt` using ``conda`` (to avoid package clashes with i.e., ``pip install <package_name>``)
    ::

        conda install mpi4py

9.  Clone into LFPy using Git:
    ::

        git clone https://github.com/LFPy/LFPy.git

10.  Build LFPy from source (without moving files)
    ::

        python setup.py develop

11. NEURON NMODL (.mod) files will not be autocompiled when building LFPy as on MacOS/Linux, as the mknrndll script
    cannot be run directly in the Anaconda Prompt. To fix this, run the ``bash`` file in the NEURON program group,
    change directory within "bash" to the ``<LFPy>/LFPy/test`` folder,  then run ``mknrndll``


Installing NEURON with Python from source
-----------------------------------------

Some users have difficulties installing NEURON as a Python module,
depending on their platform.
We will provide some explanations here, and otherwise direct to the NEURON download pages;
https://www.neuron.yale.edu/neuron/download and https://www.neuron.yale.edu/neuron/download/getstd.
The NEURON forum (https://www.neuron.yale.edu/phpBB/) is
also a useful resource for installation problems.

Dependencies: Ubuntu 18.04 LTS and other Debian-based Linux versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The instructions below show how to meet all the requirements starting from a clean Ubuntu 18.4 for the installation of NEURON from the development branch.

Start by installing required packages. We aim to link with the system Python installation, not Anaconda Python.
For Anaconda installations, make sure that the correct Python installations is found during NEURON's configure step below.
::

    sudo apt install git build-essential autoconf libtool
    sudo apt install libxext-dev libncurses-dev zlib1g-dev
    sudo apt install bison flex libx11-dev
    sudo apt install openmpi-bin libopenmpi-dev
    sudo apt install python3-dev python3-numpy python3-scipy python3-matplotlib
    sudo apt install ipython3 cython3


Linux/Unix installation of NEURON from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch the source code of NEURON using git
::

    cd $HOME
    mkdir neuron
    cd neuron

    git clone https://github.com/neuronsimulator/iv.git
    git clone https://github.com/neuronsimulator/nrn.git

Set compilers, here using the GNU Compiler Collection (GCC). A compute cluster may have the Intel Compiler binaries (``icc/icpc/mpiicc/mpiicpc``)
::

    export CC=gcc
    export CXX=g++
    export MPICC=mpicc
    export MPICXX=mpicxx

Optional, compile and install InterViews binaries to the folder ``$HOME/.local`` folder (which should be in the default ``$PATH`` on most systems)
::

    cd iv
    sh build.sh
    ./configure --prefix=~/.local
    make -j4
    make install

Compile and install NEURON with InterViews and MPI. To disable InterViews, use ``--without-iv`` during the configuration step.
::

    cd ../nrn
    sh build.sh
    ./configure --prefix=~/.local --with-iv=~/.local --with-nrnpython=/usr/bin/python3.6 --with-mpi=/usr/bin/mpirun --with-paranrn
    make -j4
    make install

You might want to add the folder with NEURON binaries to your ``$PATH``, so that you can easily compile NEURON mechanisms using ``nrnivmodl`` from the terminal.
Add the following line to your ``$HOME/.bashrc`` (or equivalent) file:
::

    export PATH=$HOME/.local/x86_64/bin:$PATH

Start a new terminal tab or type ``source $HOME/.bashrc`` to activate.

Install NEURON as a Python module
::

    cd src/nrnpython/
    python3 setup.py install --user


Now you should be able to ``import neuron`` from Python console and run a small test with success;
::

    cd $HOME
    ipython3
    >>> import neuron
    >>> neuron.test()



NEURON dependencies and installation on Mac OSX from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the development work and testing of LFPy has been done on MacOS (10.6-). Our preferred way of building Python
has been through MacPorts; http://www.macports.org. Here is an step-by-step explanation on how to compile NEURON agains that installation of Python.
Simpler solutions are stipulated above.

To start using MacPorts, follow the instructions on http://www.macports.org/install.php.

Building a python 2.7 environment using MacPorts issue in Terminal:
::

    sudo port install python27 py27-ipython py27-numpy py27-matplotlib py27-scipy py27-cython py27-mpi4py py27-h5py

Make the installed Python and IPython default:
::

    sudo port select --set python python27
    sudo port select --set ipython ipython27

Install the necessary packages for cloning into repository and compiling NEURON:
::

    sudo port install automake autoconf libtool xorg-libXext ncurses mercurial bison flex

Install NEURON from the bleeding edge source code. The following recipe assumes a 64 bit build of NEURON and Python on MacOS 10.12, so change
"x86_64-apple-darwin16.7.0" throughout to facilitate your system accordingly,
as found by running ``./config.guess`` in the root of the NEURON source code folder;
::

    #create a directory in home directory
    cd $HOME
    mkdir nrn64
    cd nrn64

    #creating directories
    sudo mkdir /Applications/NEURON-7.5
    sudo mkdir /Applications/NEURON-7.5/iv
    sudo mkdir /Applications/NEURON-7.5/nrn

    #Downloading bleeding edge source code
    hg clone http://www.neuron.yale.edu/hg/neuron/iv
    hg clone http://www.neuron.yale.edu/hg/neuron/nrn
    cd iv

    #compiling and installing IV under folder /Applications/nrn7.5
    sh build.sh
    ./configure --prefix=/Applications/NEURON-7.5/iv \
            --build=x86_64-apple-darwin16.7.0 --host=x86_64-apple-darwin16.7.0 \
            --x-includes=/usr/X11/include --x-libraries=/usr/X11/lib
    make
    sudo make install

    #Building NEURON with InterViews, you may have to alter the path --with-nrnpython=/python-path
    cd $HOME/nrn64/nrn
    sh build.sh
    ./configure --prefix=/Applications/NEURON-7.5/nrn \
            --with-nrnpython=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python \
            --host=x86_64-apple-darwin16.7.0 --build=x86_64-apple-darwin16.7.0 \
            --with-paranrn \
            --with-mpi \
            --with-iv=/Applications/NEURON-7.5/iv \
            CFLAGS='-O3 -Wno-return-type -Wno-implicit-function-declaration -Wno-implicit-int -fPIC' \
            CXXFLAGS='-O3 -Wno-return-type -fPIC'
    make
    sudo make install
    sudo make install after_install

    #You should now have a working NEURON application under Applications. Small test;
    #sudo /Applications/NEURON-7.5/nrn/x86_64/bin/neurondemo

    #Final step is to install neuron as a python module
    cd src/nrnpython
    sudo python setup.py install


LFPy on the Neuroscience Gateway Portal
=======================================

LFPy is installed on the Neuroscience Gateway Portal (NSG, see http://www.nsgportal.org), and can be used to execute simulations with LFPy both serially and in parallel applications on high-performance computing facilities.
The access to the NSG is entirely free, and access to other neuronal simulation software (NEST, NEURON, etc.) is also provided. The procedure for getting started with LFPy on the NSG is quite straightforward through their web-based interface:

1.  First, apply for a NSG user account by filling out their application form and sending it by email (follow instructions on http://www.nsgportal.org/portal2)
2.  After approval, log in using your credentials, change password if necessary
3.  As a first step after log in, create a new folder, e.g., named "LFPyTest" and with some description. This will be the home for your input files and output files, and should contain empty Data and Tasks folders
4.  Press the "Data (0)" folder in the left margin. Press the "Upload/Enter Data" button, showing the Upload File interface. Add a label, e.g., "LFPyTest".
5.  Next, LFPy simulation files have to be uploaded. As an example, download the example LFPy files https://github.com/espenhgn/LFPy/blob/master/examples/nsg_example/L5_Mainen96_wAxon_LFPy.hoc
    and https://github.com/espenhgn/LFPy/blob/master/examples/nsg_example/nsg_example.py into a new local folder "nsg_example". Modify as needed.
6.  Zip the "nsg_example" folder, upload it to the NSG (cf. step 4) and press "Save"
7.  Press "Tasks (0)" in the left margin and "Create New Task"
8.  Enter some Description, e.g., "LFPyTest", and "Select Input Data". Hook off "LFPyTest" and press "Select Data"
9.  Next, press "Select Tool", and then "Python (2.7.x)"
10. Then, go to the "Set Parameters" tab. This allows for specifying simulation time, main simulation script, and number of parallel threads. Set "Maximum Hours" to 0.1,
    and "Main Input Python Filename" to "nsg_example.py". Node number and number of cores per node should both be 1. Press "Save Parameters"
11. Everything that is needed has been set up, thus "Save and Run Task" in the Task Summary tab is all that is needed to start the job, but expect some delay for it to start.
12. Once the job is finished, you will be notified by email, or keep refreshing the Task window. The simulation output can be accessed through "View Output". Download the "output.tar.gz" file and unzip it.
    Among the output files, including stdout.txt and stderr.txt text files and jobscript details, the included folder "nsg_example" will contain the input files and any output files.
    For this particular example, only a pdf image file is generated, "nsg_example.pdf"
