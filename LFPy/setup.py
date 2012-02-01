from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = { 'build_ext' : build_ext},
    ext_modules = [
        Extension("lfpcalc",
        ["lfpcalc.pyx"],
        include_dirs=[numpy.get_include()]
        ),
        Extension("recextelectrodethreaded",
        ["recextelectrodethreaded.pyx"],
        include_dirs=[numpy.get_include()]
        ),
        Extension("run_simulation",
        ["run_simulation.pyx"],
        include_dirs=[numpy.get_include()]
        ),
])
