#!/bin/bash
export PATH=$HOME/.local/nrn/x86_64/bin:$PATH

cd $HOME
wget https://github.com/neuronsimulator/nrn/archive/7.7.0.tar.gz
tar -xf 7.7.0.tar.gz
cd nrn-7.7.0
echo "installing NEURON:"
echo "running sh build.sh"
sh build.sh #> /dev/null
echo "running ./configure"
./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython=python --with-mpi --disable-rx3d > /dev/null
echo "running make"
make -j8 > /dev/null
echo "running make install"
make install > /dev/null
cd src/nrnpython
echo "installing neuron python module"
python setup.py install # > /dev/null
cd $TRAVIS_BUILD_DIR
