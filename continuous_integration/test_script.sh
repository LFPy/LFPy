#!/bin/bash

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# build LFPy inplace (compile cython extensions, NMODL files)
python setup.py build_ext -i

# run tests
nosetests --with-coverage --cover-package=LFPy

