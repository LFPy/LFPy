LFPy  
====

Summary
-------

LFPy is a Python module for calculation of extracellular potentials from multicompartment neuron models.
It relies on the NEURON simulator (<http://www.neuron.yale.edu/neuron>) and uses the
Python interface (<http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.001.2009/abstract>) it provides.

Latest changes
--------------

Just updated LFPy? Please check the latest release notes: <https://github.com/LFPy/LFPy/releases>

Usage
-----
A brief video tutorial on LFPy is available here: <https://youtu.be/gCQkyTHZ1lw>

LFPy is preinstalled at the EBRAINS collaboratory, and you can test LFPy online without installation, by clicking this button:

[![](https://nest-simulator.org/TryItOnEBRAINS.png)](https://lab.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FLFPy%2Ftutorial_at_EBRAINS&urlpath=tree%2Ftutorial_at_EBRAINS%2F&branch=main)

Note that you might need to be logged into an EBRAINS account for the link to work.
To get a free EBRAINS account, sign up here: https://www.ebrains.eu/page/sign-up

A basic simulation of extracellular potentials of a multicompartment neuron model set up with LFPy:

    >>> # import modules
    >>> import LFPy
    >>> from LFPy import Cell, Synapse, LineSourcePotential
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # create Cell
    >>> cell = Cell(morphology=''.join(LFPy.__path__ +
    >>>                                ['/test/ball_and_sticks.hoc']),
    >>>             passive=True,  # NEURON 'pas' mechanism
    >>>             tstop=100,  # ms
    >>>            )
    >>> # create Synapse
    >>> synapse = Synapse(cell=cell,
    >>>                   idx=cell.get_idx("soma[0]"),  # soma segment index
    >>>                   syntype='Exp2Syn',  # two-exponential synapse
    >>>                   weight=0.005,  # max conductance (uS)
    >>>                   e=0,  # reversal potential (mV)
    >>>                   tau1=0.5,  # rise time constant
    >>>                   tau2=5.,  # decay time constant
    >>>                   record_current=True,  # record synapse current
    >>>                  )
    >>> synapse.set_spike_times(np.array([20., 40]))  # set activation times
    >>> # create extracellular predictor
    >>> lsp = LineSourcePotential(cell=cell,
    >>>                           x=np.zeros(11) + 10,  # x-coordinates of contacts (µm)
    >>>                           y=np.zeros(11),  # y-coordinates
    >>>                           z=np.arange(11)*20,  # z-coordinates
    >>>                           sigma=0.3,  # extracellular conductivity (S/m)
    >>>                          )
    >>> # execute simulation
    >>> cell.simulate(probes=[lsp])  # compute measurements at run time
    >>> # plot results
    >>> fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    >>> axes[0].plot(cell.tvec, synapse.i)
    >>> axes[0].set_ylabel('i_syn (nA)')
    >>> axes[1].plot(cell.tvec, cell.somav)
    >>> axes[1].set_ylabel('V_soma (nA)')
    >>> axes[2].pcolormesh(cell.tvec, lsp.z, lsp.data, shading='auto')
    >>> axes[2].set_ylabel('z (µm)')
    >>> axes[2].set_xlabel('t (ms)')

Code status
-----------

[![PyPI version](https://badge.fury.io/py/LFPy.svg)](https://badge.fury.io/py/LFPy)
[![flake8 lint](https://github.com/LFPy/LFPy/actions/workflows/flake8.yml/badge.svg)](https://github.com/LFPy/LFPy/actions/workflows/flake8.yml)
![Python application](https://github.com/LFPy/LFPy/workflows/Python%20application/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/LFPy/LFPy/badge.svg?branch=master)](https://coveralls.io/github/LFPy/LFPy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/lfpy/badge/?version=latest)](http://lfpy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LFPy/LFPy_binder_examples/master)
[![DOI](https://zenodo.org/badge/78627256.svg)](https://zenodo.org/badge/latestdoi/78627256)

Conda-forge status
------------------

[![Conda Recipe](https://img.shields.io/badge/recipe-lfpy-green.svg)](https://anaconda.org/conda-forge/lfpy)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/lfpy.svg)](https://anaconda.org/conda-forge/lfpy)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/lfpy.svg)](https://anaconda.org/conda-forge/lfpy)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/lfpy.svg)](https://anaconda.org/conda-forge/lfpy)

Information
-----------

LFPy provides a set of easy-to-use Python classes for setting up your model, running your simulations and calculating the extracellular potentials arising from activity in your model neuron. If you have a model working in NEURON (www.neuron.yale.edu) already, it is likely that it can be adapted to work with LFPy.

The extracellular potentials are calculated from transmembrane currents in multicompartment neuron models using the line-source method (Holt & Koch, J Comp Neurosci 1999),
but a simpler point-source method is also available. The calculations assume that the neuron are surrounded by an infinite extracellular medium with homogeneous and frequency
independent conductivity, and compartments are assumed to be at least at a minimal distance from the electrode (which can be specified by the user). For more information on
the biophysics underlying the numerical framework used see this coming book chapter:

- K.H. Pettersen, H. Linden, A.M. Dale and G.T. Einevoll: Extracellular spikes and current-source density, in *Handbook of Neural Activity Measurement*, edited by R. Brette and A. Destexhe, Cambridge, to appear (preprint PDF, 5.7MB <http://www.csc.kth.se/~helinden/PettersenLindenDaleEinevoll-BookChapter-revised.pdf>

The first release of LFPy (v1.x) was mainly designed for simulation extracellular potentials of single neurons, described in our paper on the package in Frontiers in Neuroinformatics entitled "LFPy: A tool for biophysical simulation of extracellular potentials generated by detailed model neurons".
The article can be found at <https://dx.doi.org/10.3389/fninf.2013.00041>.
Since version 2 (LFPy v2.x), the tool also facilitates simulations of extracellular potentials and current dipole moment from ongoing activity in recurrently connected networks of multicompartment neurons, prediction of EEG scalp surface potentials,
MEG scalp surface magnetic fields, as described in the publication "Multimodal modeling of neural network activity: computing LFP, ECoG, EEG and MEG signals with LFPy2.0" by Espen Hagen, Solveig Naess, Torbjoern V Ness, Gaute T Einevoll, found at <https://dx.doi.org/10.3389/fninf.2018.00092>.

Citing LFPy
-----------

- LFPy v2.x: Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG Signals With LFPy 2.0. Front. Neuroinform. 12:92. doi: 10.3389/fninf.2018.00092. <https://dx.doi.org/10.3389/fninf.2018.00092>

- LFPy v1.x: Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH and Einevoll GT (2013). LFPy: A tool for biophysical simulation of extracellular potentials generated by detailed model neurons. Front. Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041. <https://dx.doi.org/10.3389/fninf.2013.00041>

LFPy was developed in the Computational Neuroscience Group, Department of Mathemathical Sciences and Technology (<http://www.nmbu.no/imt>),
at the Norwegian University of Life Sciences (<http://www.nmbu.no>),
in collaboration with the Laboratory of Neuroinformatics (<http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics>),
Nencki Institute of Experimental Biology (<http://www.nencki.gov.pl>), Warsaw, Poland. The effort was supported by
International Neuroinformatics Coordinating Facility (<http://incf.org>), the Research Council of Norway (<http://www.forskningsradet.no/english>) (eScience, NevroNor),
EU-FP7 (BrainScaleS, <http://www.brainscales.org>),
the European Union Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreement No. 785907 and No. 945539 [Human Brain Project (HBP) SGA2, SGA3 and EBRAINS].

For updated information on LFPy and online documentation, see the LFPy homepage (<http://lfpy.readthedocs.io>).

Tutorial slides on LFPy
-----------------------

- Slides for OCNS 2019 meeting tutorial [T8: Biophysical modeling of extracellular potentials (using LFPy)](https://www.cnsorg.org/cns-2019-tutorials#T8) hosted in Barcelona, Spain on LFPy: [CNS2019 LFPy tutorial slides](https://LFPy.github.io/downloads/CNS2019_LFPy_tutorial.pdf "slides")
- Older tutorial slides can be found at [https://github.com/LFPy/LFPy.github.io/tree/master/downloads](https://github.com/LFPy/LFPy.github.io/tree/master/downloads)

Related projects
----------------

LFPy has been used extensively in ongoing and published work, and may be a required dependency by the publicly available Python modules:

- ViSAPy - Virtual Spiking Activity in Python (<https://github.com/espenhgn/ViSAPy>, <http://software.incf.org/software/visapy>),
  as described in Hagen, E., et al. (2015), J Neurosci Meth, DOI:`10.1016/j.jneumeth.2015.01.029 <http://dx.doi.org/10.1016/j.jneumeth.2015.01.029>`_

- ViMEAPy that can be used to incorporate heterogeneous conductivity in calculations of extracellular potentials with LFPy
  (<https://bitbucket.org/torbness/vimeapy>, <http://software.incf.org/software/vimeapy>). ViMEAPy and it's application is described
  in Ness, T. V., et al. (2015), Neuroinform, DOI:`10.1007/s12021-015-9265-6 <http://dx.doi.org/10.1007/s12021-015-9265-6>`_.

- hybridLFPy - biophysics-based hybrid scheme for calculating the local field potential (LFP) of spiking activity in simplified
  point-neuron network models (<https://github.com/INM-6/hybridLFPy>),
  as described in Hagen, E. and Dahmen, D., et al. (2016), Cereb Cortex, DOI:`10.1093/cercor/bhw237 <http://dx.doi.org/10.1093/cercor/bhw237>`_

- MEArec - Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays (<https://github.com/alejoe91/MEArec>)
 as described in Buccino, A.P., Einevoll, G.T. MEArec: A Fast and Customizable Testbench Simulator for Ground-truth Extracellular Spiking Activity. Neuroinform (2020). <https://doi.org/10.1007/s12021-020-09467-7>

Requirements
------------

Dependencies should normally be automatically installed.
For manual preinstallation of dependencies, the following packages are needed:

- Python modules numpy, scipy, matplotlib, h5py
- MEAutility (<https://github.com/alejoe91/MEAutility>)
- LFPykit (<https://github.com/LFPy/LFPykit>)
- NEURON (from <http://www.neuron.yale.edu>) and corresponding Python module. The following should execute without error in a Python console:

        >>> import neuron
        >>> neuron.test()

- Cython (C-extensions for python, <http://cython.org>) to speed up simulations of extracellular fields

Installation
------------

There are few options to install LFPy:

1. From the Python Package Index with only local access using pip:

        pip install LFPy --user

    as sudoer (in general not recommended as system Python files may be overwritten):

        sudo pip install LFPy

    Upgrading LFPy from the Python package index (without attempts at upgrading dependencies):

        pip install --upgrade --no-deps LFPy --user

    LFPy release candidates can be installed as:

        pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple LFPy --user

2. From the Python Package Index with only local access using easy_install:

        easy_install --user LFPy

    As sudoer:

        sudo easy_install LFPy

3. From source:

        tar -xzf LFPy-x.x.tar.gz
        cd LFPy-x.x
        (sudo) python setup.py develop (--user)

4. Development version from the GitHub repository:

        git clone https://github.com/LFPy/LFPy.git
        cd LFPy
        (sudo) pip install -r requirements.txt (--user) # install dependencies
        (sudo) python setup.py develop (--user)

5. Anaconda Python (<https://www.anaconda.com>, macos/linux only):

    Add the conda-forge (<https://conda-forge.org>) as channel:

        conda config --add channels conda-forge
        conda config --set channel_priority strict  # suggested

    Create a new conda environment with LFPy and activate it:

        conda create -n lfpy python=3 pip lfpy  # creates new Python 3.x conda environment named lfpy with pip and LFPy and their dependencies
        conda activate lfpy  # activate the lfpy environment
        python -c "import LFPy; LFPy.run_tests()"  # check that installation is working

    LFPy can also be installed in existing conda environments if the dependency tree is solvable:

        conda activate <environment>
        conda install lfpy  # installs LFPy and its dependencies in the current conda environment

Uninstall
---------

To remove installed LFPy files it should suffice to issue (repeat until no more LFPy files are found):

    (sudo) pip uninstall LFPy

In case LFPy was installed using conda in an environment, it can be uninstalled by issuing:

    conda uninstall lfpy

Docker
------

We provide a Docker (<https://www.docker.com>) container recipe file with LFPy.
To get started, install Docker and issue either:

    # build Dockerfile from GitHub
    $ docker build -t lfpy https://raw.githubusercontent.com/LFPy/LFPy/master/Dockerfile
    $ docker run -it -p 5000:5000 lfpy:latest

or

    # build local Dockerfile (obtained by cloning repo, checkout branch etc.)
    $ docker build -t lfpy - < Dockerfile
    $ docker run -it -p 5000:5000 lfpy:latest

If the docker file should fail for some reason it is possible to store the build log and avoid build caches by issuing

    docker build --no-cache --progress=plain -t lfpy - < Dockerfile 2>&1 | tee lfpy.log

If the build is successful, the ``--mount`` option can be used to mount a folder on the host to a target folder as:

    docker run --mount type=bind,source="$(pwd)",target=/opt -it -p 5000:5000 lfpy

which mounts the present working dirctory (``$(pwd)``) to the ``/opt`` directory of the container.
Try mounting the ``LFPy`` source directory for example (by setting ``source="<path-to-LFPy>"``).
Various LFPy example files can then be found in the folder ``/opt/LFPy/examples/``
when the container is running.

Jupyter notebook servers running from within the
container can be accessed after invoking them by issuing:

    cd /opt/LFPy/examples/
    jupyter-notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root

and opening the resulting URL in a browser on the host computer, similar to:
<http://127.0.0.1:5000/?token=dcf8f859f859740fc858c568bdd5b015e0cf15bfc2c5b0c1>

HTML Documentation
------------------

To generate the html documentation also hosted at <https://lfpy.rtfd.io> using Sphinx,
issue from the LFPy source code directory:

    cd doc
    make html

The main html file is in ``_build/html/index.html``.
m2r2, Numpydoc and the Sphinx ReadTheDocs theme may be needed:

    pip install m2r2 --user
    pip install numpydoc --user
    pip install sphinx-rtd-theme --user

Physical units in LFPy
----------------------

Physical units follow the NEURON conventions found [here](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).
The units in LFPy for given quantities are:

| Quantity                   | Symbol    | Unit      |
|----------------------------|-----------|-----------|
| Spatial dimensions         | x,y,z,d   | [μm]      |
| Potential                  | v, Phi, Φ | [mV]      |
| Reversal potential         | E         | [mV]      |
| Current                    | i         | [nA]      |
| Membrane capacitance       | c_m       | [μF/cm2]  |
| Conductance                | g         | [S/cm2]   |
| Synaptic conductance       | g         | [µS]      |
| Extracellular conductivity | sigma, σ  | [S/m]     |
| Current dipole moment      | P         | [nA µm]   |
| Magnetic field             | H         | [nA/µm]   |
| Magnetic permeability      | µ, mu     | [T m/A]   |
| Current Source Density     | CSD       | [nA/µm3]  |  

Note: resistance, conductance and capacitance are usually specific values, i.e per membrane area (lowercase r_m, g, c_m)
Depending on the mechanism files, some may use different units altogether, but this should be taken care of internally by NEURON.
