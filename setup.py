#!/usr/bin/env python

from distutils.core import setup 
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "LFPy",
    version = "1.0", 
    author = "The LFPy Community",
    packages = ['LFPy'], 
    cmdclass = { 'build_ext' : build_ext}, 
    ext_modules = [
        Extension('LFPy.lfpcalc', 
        ['LFPy/lfpcalc.pyx'],
        include_dirs=[numpy.get_include()]),
        Extension('LFPy.run_simulation', 
        ['LFPy/run_simulation.pyx'],
        include_dirs=[numpy.get_include()]),
        Extension('LFPy.recextelectrodethreaded', 
        ['LFPy/recextelectrodethreaded.pyx'],
        include_dirs=[numpy.get_include()])
        ],
    url='http://pypi.python.org/pypi/LFPy/',
    license='LICENSE.txt',
    description='A simulation environment for LFP built on NEURON',
    long_description=open('README.txt').read()
    )