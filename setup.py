#!/usr/bin/env python

from distutils.core import setup

setup(
    name = "LFPy",
    version = "0.1",
    package_dir={'trunk': 'LFPy'},
    packages = ['LFPy'],
    author = "The LFPy Community",
    description = "A simulation environment for LFP built on NEURON",
    )
