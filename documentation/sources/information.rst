
Download LFPy
=============

Downlad the latest version of LFPy from the Python Package Index: `http://pypi.python.org/pypi/LFPy <https://pypi.python.org/pypi/LFPy>`_

Or, download the development version of LFPy using `git <https://git-scm.com>`_ from `GitHub.com <https://github.com/LFPy/LFPy>`_ into a local folder:
::
    
    cd /where/to/put/repositories
    git clone https://github.com/LFPy/LFPy.git

The LFPy source code and examples is then found under "LFPy".

The stable versions of LFPy can be accessed by listing and checking out tags, e.g.,
::
    
    cd /path/to/LFPy
    git tag -l
    git checkout tags/<tag name>
    

To browse the source codes online, see http://lfpy.github.io/classes.html.


Developing LFPy
===============

As development of LFPy is now moved onto GitHub (https://github.com/LFPy), one can now fully benefit on working with forks of LFPy, implement new features, and share code improvements through pull requests.
We hope that LFPy can be improved continously through such a collaborative effort.


Getting started
===============

Dependencies
------------

To install LFPy you will need the following:

1.  Python, seems to work fine with recent python since 2.6.x (2.6.6 or newer recommended), 2.7.x works fine.
    LFPy has only been thoroughly checked for code consistency with Python 3.x., and can be assumed to be untested with Python 3.x. 

2.  Python modules numpy, scipy, matplotlib

3.  `NEURON <http://www.neuron.yale.edu>`_ compiled as a Python module, so the following should execute without error in Python console:
    ::
    
        import neuron
        neuron.test()
    
    If this step fails, see the next section.
    
    LFPy has been tested to work with NEURON 7.2 and the 7.3 branches, we had some issues with a precompiled version of NEURON 7.1.

4.  `Cython <http://cython.org>`_ (C-extensions for python) to speed up simulations of extracellular fields. Tested with version > 0.14,
    and known to fail with version 0.11. LFPy works without Cython, but simulations will run slower and is therefore not recommended.


Installing LFPy
---------------

There are few options to install LFPy:

1.  From the Python Package Index with only local access using pip
    ::
        
        pip install --user LFPy


    as sudoer:
    ::
    
        sudo pip install LFPy
    
    Upgrading LFPy from the Python package index (without attempts at upgrading dependencies):
    ::
        
        pip install --upgrade --no-deps LFPy
        

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


4.  From development version in our git (https://git-scm.com) repository:
    ::
    
        git clone https://github.com/LFPy/LFPy.git
        cd LFPy
        (sudo) python setup.py install (--user)

    
5.  If you only want to have LFPy in one place, you can add this working folder to your $PYTHONPATH, and just compile the Cython extensions inplace;
    ::
    
        python setup.py build_ext -i

    
In a fresh terminal and python-session you should now be able to issue: 
::  

    import LFPy


Installing NEURON with Python
-----------------------------

For most users, and even though NEURON (http://neuron.yale.edu) provides a working Python interpreter, making NEURON work as a Python module may be quite straightforward using pre-built
Python distributions such as the Anaconda Scientific Python distribution (http://continuum.io) or Enthought Canopy (https://www.enthought.com/products/canopy/). We here provide some short step-by-step recipes on
how to set up a working Python environment using Anaconda with the standard pre-built NEURON binaries on Linux, OSX and Windows.


Ubuntu 10.04 LTS 64-bit with Anaconda Scientific Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By far the simplest solution relying on no source code compilation.

1. Download and install Anaconda using the 64-bit Linux installer script from http://continuum.io/downloads
2. Download and install the 64-bit Debian/Ubuntu .deb file with NEURON from http://www.neuron.yale.edu/neuron/download
3. Edit your .bashrc or similar file located in the $HOME folder, e.g., by calling in the Terminal.app "gedit $HOME/.bashrc", to include the lines:

::

    # make NEURON python module available to Anaconda python
    export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"


4. Open a fresh terminal window
5. Install LFPy, e.g., from the python package index  (or see above)

::
    
    pip install LFPy
    
6. Test the installation

::
    
    espen@espen-VirtualBox:~$ which python
    /home/ehagen/anaconda/bin/python
    espen@espen-VirtualBox:~$ python -c "import LFPy"
    NEURON -- VERSION 7.3 (1078:2b0c984183df) 2014-04-04
    Duke, Yale, and the BlueBrain Project -- Copyright 1984-2014
    See http://www.neuron.yale.edu/neuron/credits

If everything worked, one should now have a working Python/NEURON/LFPy environment.


OSX 10.9.x with Anaconda Scientific Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By far the simplest solution relying on no source code compilation.

1. Download and install Anaconda using the 64-bit graphical installer from http://continuum.io/downloads
2. Download and install the 64-bit Mac .dmg file with NEURON from http://www.neuron.yale.edu/neuron/download
3. Edit your .bash_profile or similar file located in the $HOME folder, e.g., by calling in the Terminal.app "open -t $HOME/.bash_profile", to include the lines:

::
    
    # added by Anaconda 2.0.1 installer
    export PATH="/Users/ehagen_test/anaconda/bin:$PATH"
    
    # make neuron executable from terminal
    export PATH="/Applications/NEURON-7.3/nrn/x86_64/bin:$PATH"
    
    # make NEURON python module available to Anaconda python
    export PYTHONPATH="/Applications/NEURON-7.3/nrn/lib/python/:$PYTHONPATH"

4. Open a fresh terminal window
5. Install LFPy, e.g., from the python package index (or see above)

::
    
    pip install LFPy
    
6. Test the installation from the terminal

::
    
    inm6078:VirtualBox VMs ehagen$ which python
    /Users/ehagen/anaconda/bin/python
    inm6078:VirtualBox VMs ehagen$ python -c "import LFPy"
    NEURON -- VERSION 7.3 (1078:2b0c984183df) 2014-04-04
    Duke, Yale, and the BlueBrain Project -- Copyright 1984-2014
    See http://www.neuron.yale.edu/neuron/credits

If everything worked, one should now have a working Python/NEURON/LFPy environment.



Windows with Anaconda Scientific Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We have had some success on getting LFPy and NEURON to run on the Windows platform.

The following procedure was tested on a 32 bit Windows 7 Ultimate virtual machine, using Anaconda Python distribution and the MinGW-built release of NEURON.
However, the procedure may or may not be similar on other versions of Windows.

1.  Download and install 32-bit Anaconda Python using the graphical installer from http://continuum.io/downloads
2.  Download and install the 32-bit (MinGW) setup.exe file for NEURON from http://www.neuron.yale.edu/neuron/download
3.  Some PATH variables are needed to enable Anaconda Python and NEURON to interact nicely.
    Go through "Control Panel" --> "System and Security" --> "System" --> "Advanced System Settings" --> "Environment Variables"
    and add the following variables and values to the list of "System variables":
    ::
        
        Append ";/nrn73w/bin/" to "Path"
        Create new entry: "NEURONHOME", value "/nrn73w/"
        Create second entry "PYTHONPATH", value "/nrn73w/lib/python/"
    
    if running with 64 bit Windows and software, exchange "nrn73w" with "nrn73w64" and "bin" with "bin64" above. 

4.  Issue in the "Anaconda Command Prompt":
    ::
    
        pip install LFPy --user

5.  Check if "import neuron" and "import LFPy" works both from within Spyder, IPython and the NEURON supplied Python interpreter.
        

If everything worked, one should now have a working Python/NEURON/LFPy environment.




Installing NEURON with Python from source
-----------------------------------------

Some users have difficulties install NEURON as a Python module,
depending on their platform. 
We will provide some explanations here, and otherwise direct to the NEURON download pages;
http://www.neuron.yale.edu/neuron/download and http://www.neuron.yale.edu/neuron/download/getstd. The NEURON forum is
also a useful resource for installation problems.

Dependencies: Ubuntu 10.04 LTS and other Debian-based Linux versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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



Linux/Unix installation of NEURON from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


NEURON dependencies and installation on Mac OSX from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    sudo port install automake autoconf libtool xorg-libXext ncurses mercurial bison flex

Install NEURON from the bleeding edge source code. The following recipe assumes a 64 bit build of NEURON and Python on OSX 10.7 Lion, so change
"x86_64-apple-darwin10.7.0" throughout to facilitate your system accordingly,
as found by running "./config.guess" in the root of the NEURON source code;
::

    #create a directory in home directory                                                                                                                                                               
    cd $HOME
    mkdir nrn64
    cd nrn64
    
    #creating directories                                                                                                                                                                               
    sudo mkdir /Applications/NEURON-7.3
    sudo mkdir /Applications/NEURON-7.3/iv
    sudo mkdir /Applications/NEURON-7.3/nrn
    
    #Downloading bleeding edge source code                                                                                                                                                              
    hg clone http://www.neuron.yale.edu/hg/neuron/iv
    hg clone http://www.neuron.yale.edu/hg/neuron/nrn
    cd iv
        
    #compiling and installing IV under folder /Applications/nrn7.3                                                                                                                                             
    sh build.sh
    ./configure --prefix=/Applications/NEURON-7.3/iv \
        --build=x86_64-apple-darwin10.7.0 --host=x86_64-apple-darwin10.7.0
    
    make
    sudo make install
    
    #Building NEURON with InterViews, you may have to alter the path --with-nrnpython=/python-path                                                                                                      
    cd $HOME/nrn64/nrn
    sh build.sh
    ./configure --prefix=/Applications/NEURON-7.3/nrn --with-iv=/Applications/NEURON-7.3/iv \
        --with-x --x-includes=/usr/X11/include/ --x-libraries=/usr/X11/lib/ \
        --with-nrnpython=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python \
        --host=x86_64-apple-darwin10.7.0 --build=x86_64-apple-darwin10.7.0
    
    make
    sudo make install
    sudo make install after_install
    
    #You should now have a working NEURON application under Applications. Small test;                                                                                                                   
    #sudo /Applications/NEURON-7.3/nrn/x86_64/bin/neurondemo                                                                                                                                            
    
    #Final step is to install neuron as a python module                                                                                                                                                 
    cd src/nrnpython
    sudo python setup.py install


LFPy on the Neuroscience Gateway Portal
=======================================

LFPy is installed on the Neuroscience Gateway Portal (NSG, see http://www.nsgportal.org), and can be used to execute simulations with LFPy both serially and in parallel applications on high-performance computing facilities.
The access to the NSG is entirely free, and access to other neuronal simulation software (NEST, NEURON, etc.) is also provided. The procedure for getting started with LFPy on the NSG is quite straightforward through their web-based interface:

1. First, apply for a NSG user account by filling out their application form and sending it by email (follow instructions on http://www.nsgportal.org/portal2)
2. After approval, log in using your credentials, change password if necessary
3. As a first step after log in, create a new folder, e.g., named "LFPyTest" and with some description. This will be the home for your input files and output files, and should contain empty Data and Tasks folders
4. Press the "Data (0)" folder in the left margin. Press the "Upload/Enter Data" button, showing the Upload File interface. Add a label, e.g., "LFPyTest".
5. Next, LFPy simulation files have to be uploaded. As an example, download the example LFPy files https://github.com/espenhgn/LFPy/blob/master/examples/nsg_example/L5_Mainen96_wAxon_LFPy.hoc
   and https://github.com/espenhgn/LFPy/blob/master/examples/nsg_example/nsg_example.py into a new local folder "nsg_example". Modify as needed. 
6. Zip the "nsg_example" folder, upload it to the NSG (cf. step 4) and press "Save"
7. Press "Tasks (0)" in the left margin and "Create New Task"
8. Enter some Description, e.g., "LFPyTest", and "Select Input Data". Hook off "LFPyTest" and press "Select Data"
9. Next, press "Select Tool", and then "Python (2.7.x)"
10. Then, go to the "Set Parameters" tab. This allows for specifying simulation time, main simulation script, and number of parallel threads. Set "Maximum Hours" to 0.1,
    and "Main Input Python Filename" to "nsg_example.py". Node number and number of cores per node should both be 1. Press "Save Parameters"
11. Everything that is needed has been set up, thus "Save and Run Task" in the Task Summary tab is all that is needed to start the job, but expect some delay for it to start. 
12. Once the job is finished, you will be notified by email, or keep refreshing the Task window. The simulation output can be accessed through "View Output". Download the "output.tar.gz" file and unzip it.
    Among the output files, including stdout.txt and stderr.txt text files and jobscript details, the included folder "nsg_example" will contain the input files and any output files.
    For this particular example, only a pdf image file is generated, "nsg_example.pdf"
