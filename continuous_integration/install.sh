#!/bin/bash
export PATH=$HOME/.local/nrn/x86_64/bin:$PATH

cd $HOME
wget https://github.com/neuronsimulator/nrn/archive/master.tar.gz
tar -xf master.tar.gz
cd nrn-master
echo "installing NEURON:"
echo "running sh build.sh"
sh build.sh > /dev/null
echo "running ./configure"
./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython=python --with-mpi > /dev/null
echo "running make"
make > /dev/null
echo "running make install"
make install > /dev/null
cd src/nrnpython
echo "installing neuron python module"
python setup.py install > /dev/null
cd $TRAVIS_BUILD_DIR

