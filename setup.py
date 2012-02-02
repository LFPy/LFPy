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

setup(
    name = "LFPy",
    version = "1.0", 
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
    url='http://pypi.python.org/pypi/LFPy/',
    license='LICENSE.txt',
    description='A simulation environment for LFP built on NEURON',
    long_description=open('README.txt').read(),
	classifiers=[
	          'License :: OSI Approved :: GNU General Public License (GPL)',
	          'Programming Language :: Python',
	          ],
    )