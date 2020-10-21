FROM debian:latest

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        gcc \
        git \
        libncurses-dev \
        python3 \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-h5py \
        python3-mpi4py \
        python3-yaml \
        python3-pytest \
        python3-pip \
        cython3 \
        jupyter \
        ipython3

RUN pip3 install neuron
RUN pip3 install git+https://github.com/LFPy/LFPy.git@master#egg=LFPy
