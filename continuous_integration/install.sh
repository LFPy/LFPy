#!/bin/bash
#pip install scipy cython coveralls #numpy nose
# install NEURON
export PWD=pwd
cd $HOME
wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz
tar -xf nrn-7.4.tar.gz
cd nrn-7.4
echo "installing NEURON:"
echo "running sh build.sh"
sh build.sh > build.outp
echo "running ./configure"
./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython > configure.outp
echo "running make"
make -j4 > make.outp
echo "running make install"
make install > make_install.outp
cd src/nrnpython
echo "installing neuron python module"
python setup.py install --user
cd $PWD
export PATH=$HOME/.local/bin:$HOME/.local/nrn/x86_64/bin:$PATH
#export PATH=$HOME/.local/bin:$PATH


