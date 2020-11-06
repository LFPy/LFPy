# -------- base ---------
FROM buildpack-deps:focal AS base

RUN apt-get update && \
    apt-get install -y \
        wget \
        libncurses-dev \
        libmpich-dev \
        mpich \
        python3 \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        #python3-h5py \
        python3-yaml \
        python3-pytest \
        python3-pip \
        cython3 \
        jupyter \
        ipython3

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10


# ------ latest -----------

FROM base AS latest

RUN pip install h5py
RUN pip install mpi4py
RUN pip install neuron
RUN pip install git+https://github.com/LFPy/LFPy.git@2.2.dev0#egg=LFPy
