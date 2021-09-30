# -------- base ---------
FROM buildpack-deps:focal AS base

RUN apt-get update && \
    apt-get install -y \
        wget \
        python3 \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-yaml \
        python3-pytest \
        python3-pip \
        python3-h5py \
        python3-pandas \
        python3-mpi4py \
        cython3 \
        jupyter \
        ipython3 \
        cmake \
        bison \
        flex \
        libopenmpi-dev \
        libncurses-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10


# ------ latest -----------
# FROM base AS latest

# --- install NEURON from source
# (only way to get builds on arm64/aarch64 hosts working) -------
RUN cd /opt
RUN git clone --depth=1 --branch 8.0.0 https://github.com/neuronsimulator/nrn

RUN cd nrn && \
    mkdir build && \
    cd build

# export MPI_LIB_NRN_PATH=$CONDA_PREFIX/lib/libmpicxx.dylib
# export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DCURSES_NEED_NCURSES=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_RX3D=ON \
  -DNRN_ENABLE_MPI=ON \
  ../nrn

RUN make
RUN make install
RUN cd src/nrnpython && \
    python setup.py install

RUN cd
RUN rm -rf nrn



# --- Install LFPy ----
RUN git clone --depth=1 --branch master https://github.com/LFPy/LFPy
RUN cd LFPy && \
    pip install .
RUN cd
RUN rm -rf LFPy
