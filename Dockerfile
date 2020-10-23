# -------- base ---------
FROM debian:latest AS base

RUN apt-get update && \
    apt-get install -y \
        wget \
        bash \
        build-essential \
        make \
        gcc \
        g++ \
        git \
        libncurses-dev \
        python3 \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-h5py \
        python3-yaml \
        python3-pytest \
        python3-pip \
        cython3 \
        jupyter \
        ipython3


# ------ latest -----------

FROM base AS latest

RUN apt-get update && \
    apt-get install -y \
    python3-mpi4py

RUN pip3 install neuron
RUN pip3 install git+https://github.com/LFPy/LFPy.git@master#egg=LFPy
