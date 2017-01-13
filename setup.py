#!/usr/bin/env python
'''LFPy setup.py file'''

import os
import shutil
try:
    from setuptools import setup, Extension
except ImportError as ie:
    raise ie, 'please install setuptools'
try:
    import numpy
    from Cython.Distutils import build_ext
    cmdclass = { 'build_ext' : build_ext}
    ext_modules = [
        Extension('LFPy.lfpcalc',
        ['LFPy/lfpcalc.pyx'],
        include_dirs=[numpy.get_include()]),
        Extension('LFPy.run_simulation',
        ['LFPy/run_simulation.pyx'],
        include_dirs=[numpy.get_include()]),
        Extension('LFPy.alias_method',
        ['LFPy/alias_method.pyx'],
        include_dirs=[numpy.get_include()]),
        ]
except ImportError as ie:
    print("'from Cython.Distutils import build_ext' or 'import numpy' failed!")
    print("Cython extensions will not be compiled, and")
    print("simulations in LFPy may run slower")
    cmdclass = {}
    ext_modules = []

#try and locate the nrnivmodl script of NEURON in PATH so that the
#NEURON extension file LFPy/sinsyn.mod can be compiled in place and be copied
#as part of the package_data, allowing unit tests to run
from distutils.spawn import find_executable, spawn
if find_executable('nrnivmodl') is not None:
    os.chdir('LFPy')
    for path in ['x86_64', 'i686', 'powerpc']:
        if os.path.isdir(path):
            shutil.rmtree(path)
    spawn([find_executable('nrnivmodl')])
    os.chdir('..')
else:
    print("nrnivmodl script not found in PATH, thus NEURON .mod files could" +
          "not be compiled, and LFPy.test() functions will fail")

with open('README.md') as file:
    long_description = file.read()

setup(
    name = "LFPy",
    version = "1.1.3",
    maintainer = "Espen Hagen",
    maintainer_email = 'espen.hagen@fys.uio.no',
    packages = ['LFPy'],
    package_data = {'LFPy' : ['stick.hoc', 'sinsyn.mod',
                              os.path.join('i686', '*'),
                              os.path.join('i686', '.libs', '*'),
                              os.path.join('x86_64', '*'),
                              os.path.join('x86_64', '.libs', '*'),
                              os.path.join('powerpc', '*'),
                              os.path.join('powerpc', '.libs', '*'),
                              ]},
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    url='http://LFPy.github.io',
    download_url = 'https://github.com/LFPy/LFPy/tarball/v1.1.3',
    license='LICENSE',
    description='A module for modeling Local Field Potentials built on NEURON',
    long_description=long_description,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable',
        ],
    install_requires = [
        'numpy', 'scipy', 'matplotlib', 'Cython'
        ],
    provides = ['LFPy'],
    )
