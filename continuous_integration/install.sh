#!/bin/bash
export PATH=$HOME/.local/nrn/x86_64/bin:$PATH

#if hash nrniv 2>/dev/null; then
#    echo "NEURON appear to be installed"
#else
export PWD=pwd
cd $HOME
wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz
tar -xf nrn-7.4.tar.gz
cd nrn-7.4
echo "installing NEURON:"
echo "running sh build.sh"
sh build.sh > /dev/null
echo "running ./configure"
./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython > /dev/null
echo "running make"
make > /dev/null
echo "running make install"
make install > /dev/null
cd src/nrnpython
echo "installing neuron python module"
python setup.py install --user
cd $PWD
#fi

