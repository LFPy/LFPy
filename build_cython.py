"""Build script for Cython extensions, used by poetry's build system."""

import os
import shutil
from pathlib import Path


def build(setup_kwargs):
    """Provide Cython extensions to Poetry's generated setup script."""
    import numpy as np
    from Cython.Build import cythonize
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
    setup_kwargs["ext_modules"] = cythonize(ext_modules)


def main():
    """Build Cython extensions in place."""
    try:
        from Cython.Distutils import build_ext
        from setuptools import Distribution

        print("Building Cython extensions")
        setup_kwargs = {}
        build(setup_kwargs)
        distribution = Distribution({
            "name": "LFPy",
            **setup_kwargs,
        })

        cmd = build_ext(distribution)
        cmd.ensure_finalized()
        cmd.run()

        for output in cmd.get_outputs():
            output = Path(output)
            relative_extension = output.relative_to(cmd.build_lib)
            print(f"Copying {output} -> {relative_extension}")
            shutil.copyfile(output, relative_extension)
    except ImportError:
        print("'Cython' or 'numpy' not found – Cython extensions will not be "
              "compiled and simulations in LFPy may run slower.")


if __name__ == "__main__":
    main()
