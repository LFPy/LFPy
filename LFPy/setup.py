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
        #Extension("electrode",
        #["electrode.py"],
        #include_dirs=[numpy.get_include()]
        #),
        #Extension("cell",
        #["cell.py"],
        #include_dirs=[numpy.get_include()]
        #),
        Extension("cellwithelectrode",
        ["cellwithelectrode.pyx"],
        include_dirs=[numpy.get_include()]
        )
])
