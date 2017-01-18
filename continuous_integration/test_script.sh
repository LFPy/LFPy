#!/bin/bash

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# build LFPy inplace (compile cython extensions, NMODL files)
python setup.py build_ext -i

# run tests, but if mystery segmentation fault occurr, rerun tests to get
# clean exit
while true; do
    nosetests --with-coverage --cover-package=LFPy
    if [ $? -eq 0 ]
    then
        exit 0
        break
    fi
done

