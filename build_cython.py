"""Build script for Cython extensions, used by poetry's build system."""

import os


def build(setup_kwargs):
    """Build Cython extensions when invoked by poetry."""
    try:
        import numpy as np
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
        from setuptools import Extension

        ext_modules = [
            Extension(
                'LFPy.run_simulation',
                [os.path.join('LFPy', 'run_simulation.pyx')],
                include_dirs=[np.get_include()],
            ),
            Extension(
                'LFPy.alias_method',
                [os.path.join('LFPy', 'alias_method.pyx')],
                include_dirs=[np.get_include()],
            ),
        ]
        setup_kwargs.update({
            'ext_modules': cythonize(ext_modules),
            'cmdclass': {'build_ext': build_ext},
        })
    except ImportError:
        print("'Cython' or 'numpy' not found – Cython extensions will not be "
              "compiled and simulations in LFPy may run slower.")
