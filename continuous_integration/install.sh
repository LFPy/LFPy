#!/bin/bash
export PATH=$HOME/.local/nrn/x86_64/bin:$PATH

cd $HOME
wget https://neuron.yale.edu/ftp/neuron/versions/v7.5/nrn-7.5.tar.gz
tar -xf nrn-7.5.tar.gz
cd nrn-7.5
echo "installing NEURON:"
echo "running sh build.sh"
sh build.sh > /dev/null
echo "running ./configure"
./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython --with-mpi > /dev/null
echo "running make"
make > /dev/null
echo "running make install"
make install > /dev/null
cd src/nrnpython
echo "installing neuron python module"
python setup.py install > /dev/null
cd $TRAVIS_BUILD_DIR

