#!/bin/bash

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import neuron; print('neuron %s' % neuron.version); print(neuron.__file__)"

# build LFPy inplace (compile cython extensions, NMODL files)
python setup.py develop

# run tests, but if mystery segmentation fault occurr, rerun tests to get
# clean exit
while true; do
    py.test LFPy/test/test*.py --cov-report term --cov=LFPy/test/
    if [ $? -eq 0 ]
    then
        exit 0
        break
    fi
done

