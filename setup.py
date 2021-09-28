#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LFPy setuptools file

"""

from distutils.spawn import spawn
import os
import sys
import shutil

d = {}
exec(open(os.path.join('LFPy', 'version.py')).read(), None, d)
version = d['version']

try:
    from setuptools import setup, Extension
except ImportError as ie:
    raise ie('please install setuptools')
try:
    import numpy
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
    ext_modules = [Extension('LFPy.run_simulation',
                             [os.path.join('LFPy', 'run_simulation.pyx')],
                             include_dirs=[numpy.get_include()]),
                   Extension('LFPy.alias_method',
                             [os.path.join('LFPy', 'alias_method.pyx')],
                             include_dirs=[numpy.get_include()]),
                   ]
except ImportError:
    print("'from Cython.Distutils import build_ext' or 'import numpy' failed!")
    print("Cython extensions will not be compiled, and")
    print("simulations in LFPy may run slower")
    cmdclass = {}
    ext_modules = []


# try and locate the nrnivmodl or mknrndll script of NEURON in PATH so that the
# NEURON NMODL files LFPy/test/*.mod can be compiled in place and be copied
# as part of the package_data, allowing unit tests to run
if not any(arg in sys.argv for arg in ['sdist', 'upload']):
    if shutil.which('nrnivmodl') is not None:
        os.chdir(os.path.join('LFPy', 'test'))
        for path in ['x86_64', 'arm64', 'aarch64']:
            if os.path.isdir(path):
                shutil.rmtree(path)
        spawn([shutil.which('nrnivmodl')])
        os.chdir(os.path.join('..', '..'))
    elif shutil.which('mknrndll') is not None:
        os.chdir(os.path.join('LFPy', 'test'))
        if os.path.isfile("nrnmech.dll"):
            os.remove("nrnmech.dll")
        spawn([shutil.which('mknrndll')])
        os.chdir(os.path.join('..', '..'))
    else:
        print("nrnivmodl/mknrndll script not found in PATH, thus NMODL " +
              "files could not be compiled. LFPy.test() functions will fail")


with open('README.md') as file:
    long_description = file.read()

setup(
    name="LFPy",
    version=version,
    maintainer="LFPy-team",
    maintainer_email='lfpy@users.noreply.github.com',
    packages=['LFPy'],
    package_data={'LFPy': ['*.pyx',
                           os.path.join('test', '*.hoc'),
                           os.path.join('test', '*.py'),
                           os.path.join('test', 'sinsyn.mod'),
                           os.path.join('test', 'expsyni.mod'),
                           ]},
    include_package_data=True,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    url='http://LFPy.readthedocs.io',
    download_url='https://github.com/LFPy/LFPy/tarball/v{}'.format(version),
    license='LICENSE',
    description=('A module for modeling extracellular potentials of '
                 'multicompartment neuron models built on NEURON'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable',
    ],
    install_requires=[
        'neuron>=7.7.2',
        'numpy>=1.8',
        'scipy>=0.14',
        'Cython>=0.20',
        'h5py>=2.5',
        'mpi4py>=1.2',
        'LFPykit>=0.3'
    ],
    extras_require={'tests': ['pytest'],
                    'docs': ['sphinx', 'numpydoc', 'sphinx_rtd_theme']
                    },
    dependency_links=[],
    provides=['LFPy'],
)
