#!/usr/bin/env python

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
from sys import version
if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

from distutils.core import setup 
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

with open('README.txt') as file:
    long_description = file.read()

setup(
    name = "LFPy",
    version = "0.9.2", 
    maintainer = "Espen Hagen",
        maintainer_email = 'ehagen@umb.no',
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
    url='http://compneuro.umb.no/LFPy/',
    license='LICENSE.txt',
    description='A module for modeling Local Field Potentials built on NEURON',
    long_description=long_description,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        ],
    requires = [
        'numpy', 'scipy', 'matplotlib', 'neuron', 'Cython'
        ],
    provides = ['LFPy'],
    )