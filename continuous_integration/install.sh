#!/bin/bash/sh
pip install scipy cython #numpy nose
# install NEURON
export PWD=pwd
cd $HOME
wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz
tar -xf nrn-7.4.tar.gz
cd nrn-7.4
sh build.sh
./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython
make -j4
make install
cd src/nrnpython
python setup.py install --user
cd $PWD
export PATH=$HOME/.local/bin:$HOME/.local/nrn/x86_64/bin:$PATH
#export PATH=$HOME/.local/bin:$PATH


