#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LFPy setuptools file

"""

import os
from distutils.util import convert_path

d = {}
ver_path = convert_path(os.path.join('LFPy', 'version.py'))
with open(ver_path, 'rt') as f:
    exec(f.read(), d)
version = d['version']

try:
    from setuptools import setup, Extension
except ImportError as err:
    print(f'please install setuptools: {err}')
    raise

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


with open('README.md', 'rt') as file:
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
    download_url=f'https://github.com/LFPy/LFPy/tarball/v{version}',
    license='LICENSE',
    description=('A module for modeling extracellular potentials of '
                 'multicompartment neuron models built on NEURON'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
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
        'lfpykit>=0.5'
    ],
    extras_require={'tests': ['pytest'],
                    'docs': ['sphinx', 'numpydoc', 'sphinx_rtd_theme']
                    },
    dependency_links=[],
    provides=['LFPy'],
)
