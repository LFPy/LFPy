LFPy  
====

LFPy is a Python-module for calculation of extracellular potentials from multicompartment neuron models.
It relies on the NEURON simulator (http://www.neuron.yale.edu/neuron) and uses the
Python interface (http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.001.2009/abstract) it provides.

You can now test some LFPy examples online without installation: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LFPy/LFPy_binder_examples/master)

LFPy provides a set of easy-to-use Python classes for setting up your model, running your simulations and calculating the extracellular potentials arising from activity in your model neuron. If you have a model working in NEURON (www.neuron.yale.edu)
already, it is likely that it can be adapted to work with LFPy.

The extracellular potentials are calculated from transmembrane currents in multi-compartment neuron models using the line-source method (Holt & Koch, J Comp Neurosci 1999),
but a simpler point-source method is also available. The calculations assume that the neuron are surrounded by an infinite extracellular medium with homogeneous and frequency
independent conductivity, and compartments are assumed to be at least at a minimal distance from the electrode (which can be specified by the user). For more information on
the biophysics underlying the numerical framework used see this coming book chapter:

- K.H. Pettersen, H. Linden, A.M. Dale and G.T. Einevoll: Extracellular spikes and current-source density, in *Handbook of Neural Activity Measurement*, edited by R. Brette and A. Destexhe, Cambridge, to appear (preprint PDF, 5.7MB http://arken.umb.no/~gautei/forskning/PettersenLindenDaleEinevoll-BookChapter-revised.pdf

The first release of LFPy (v1.x) was mainly designed for simulation extracellular potentials of single neurons, described in our paper on the package in Frontiers in Neuroinformatics entitled "LFPy: A tool for biophysical simulation of extracellular potentials generated by detailed model neurons".
The article can be found at http://dx.doi.org/10.3389%2Ffninf.2013.00041.
Since version 2 (LFPy v2.x), the tool also facilitates simulations of extracellular potentials and current dipole moment from ongoing activity in recurrently connected networks of multicompartment neurons, prediction of EEG scalp surface potentials,
MEG scalp surface magnetic fields, as described in the bioRXiv preprint "Multimodal modeling of neural network activity: computing LFP, ECoG, EEG and MEG signals with LFPy2.0" by Espen Hagen, Solveig Naess, Torbjoern V Ness, Gaute T Einevoll, found at https://doi.org/10.1101/281717.

Citing LFPy:

- LFPy v2.x: Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG Signals With LFPy 2.0. Front. Neuroinform. 12:92. doi: 10.3389/fninf.2018.00092. https://dx.doi.org/10.3389/fninf.2018.00092
- LFPy v1.x: Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH and Einevoll GT (2013). LFPy: A tool for biophysical simulation of extracellular potentials generated by detailed model neurons. Front. Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041. https://dx.doi.org/10.3389/fninf.2013.00041

LFPy was developed in the Computational Neuroscience Group, Department of Mathemathical Sciences and Technology (http://www.nmbu.no/imt),
at the Norwegian University of Life Sciences (http://www.nmbu.no),
in collaboration with the Laboratory of Neuroinformatics (http://www.nencki.gov.pl/en/laboratory-of-neuroinformatics),
Nencki Institute of Experimental Biology (http://www.nencki.gov.pl), Warsaw, Poland. The effort was supported by
International Neuroinformatics Coordinating Facility (http://incf.org), the Research Council of Norway (http://www.forskningsradet.no/english) (eScience, NevroNor) and EU-FP7 (BrainScaleS, http://www.brainscales.org).

For updated information on LFPy and online documentation, see the LFPy homepage (http://lfpy.readthedocs.io).

This scientific software is released under the GNU Public License GPLv3.

Code status
===========
[![PyPI version](https://badge.fury.io/py/LFPy.svg)](https://badge.fury.io/py/LFPy)
[![Build Status](https://travis-ci.org/LFPy/LFPy.svg?branch=master)](https://travis-ci.org/LFPy/LFPy)
[![Coverage Status](https://coveralls.io/repos/github/LFPy/LFPy/badge.svg?branch=master)](https://coveralls.io/github/LFPy/LFPy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/lfpy/badge/?version=latest)](http://lfpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/78627256.svg)](https://zenodo.org/badge/latestdoi/78627256)

Conda-forge release info
========================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-lfpy-green.svg)](https://anaconda.org/conda-forge/lfpy) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/lfpy.svg)](https://anaconda.org/conda-forge/lfpy) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/lfpy.svg)](https://anaconda.org/conda-forge/lfpy) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/lfpy.svg)](https://anaconda.org/conda-forge/lfpy) |

Requirements
============

To install LFPy you will need the following:

- Python modules numpy, scipy, matplotlib, h5py, mpi4py, Cython
- NEURON (from http://www.neuron.yale.edu, v7.6.4 or newer) and corresponding Python module. The following should execute without error in a Python console:

        import neuron
        neuron.test()

- Cython (C-extensions for python, http://cython.org) to speed up simulations of extracellular fields


Installation
============

There are few options to install LFPy:

1.  From the Python Package Index with only local access using pip:

        pip install LFPy --user

    as sudoer (in general not recommended as system Python files may be overwritten):

        sudo pip install LFPy

    Upgrading LFPy from the Python package index (without attempts at upgrading dependencies):

        pip install --upgrade --no-deps LFPy --user

    LFPy release candidates can be installed as:

        pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple LFPy --user


2.  From the Python Package Index with only local access using easy_install:

        easy_install --user LFPy

    As sudoer:

        sudo easy_install LFPy

3.  From source:

        tar -xzf LFPy-x.x.tar.gz
        cd LFPy-x.x
        (sudo) python setup.py develop (--user)

4.  Development version from the GitHub repository:

        git clone https://github.com/LFPy/LFPy.git
        cd LFPy
        (sudo) pip install -r requirements.txt (--user) # install dependencies
        (sudo) python setup.py develop (--user)

5. Anaconda Python (https://www.anaconda.com, macos/linux only):

    Add the conda-forge (https://conda-forge.org) as channel:

        conda config --add channels conda-forge
        conda config --set channel_priority strict  # suggested

    Install LFPy either issuing

        conda install lfpy  # installs LFPy and its dependencies in the current conda environment

    or

        conda create -n lfpy lfpy  # creates new conda environment named lfpy with LFPy and its dependencies
        conda activate lfpy        # activate the lfpy environment

Uninstall
=========

To remove installed LFPy files it should suffice to issue (repeat until no more LFPy files are found):

    (sudo) pip uninstall LFPy

In case LFPy was installed using conda in an environment, it can be uninstalled by issuing:

    conda uninstall lfpy

Docker
======

We provide a Docker (https://www.docker.com) container build file with LFPy.
To get started, install Docker and issue:

    docker build -t lfpy https://raw.githubusercontent.com/LFPy/LFPy/master/Dockerfile
    docker run -it -p 5000:5000 lfpy

Various LFPy example files can be found in the folder ``/opt/LFPy/examples/``
when the container is running. Jupyter notebook servers running from within the
container can be accessed after invoking them by issuing:

    cd /opt/LFPy/examples/
    jupyter notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root

and opening the resulting URL in a browser the host computer, similar to:
http://127.0.0.1:5000/?token=dcf8f859f859740fc858c568bdd5b015e0cf15bfc2c5b0c1


Documentation
=============

To generate the html documentation using Sphinx, issue from the LFPy source code directory:

    sphinx-build -b html <path to LFPy>/doc <path to output>

The main html file is in ``<path to output>/index.html``. Numpydoc and the ReadTheDocs theme may be needed:

    pip install numpydoc --user
    pip install sphinx-rtd-theme --user
