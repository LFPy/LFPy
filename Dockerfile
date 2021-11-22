# -------- base ---------
FROM buildpack-deps:hirsute AS base

RUN apt-get update && \
    apt-get install -y \
        python3-dev \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-pytest \
        python3-pip \
        cython3 \
        jupyter \
        ipython3 \
        cmake \
        bison \
        flex \
        libmpich-dev \
	mpich \
        libncurses-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10


# ------ latest -----------
# FROM base AS latest

# --- install NEURON from source
# (only way to get builds on arm64/aarch64 hosts working) -------
RUN cd /opt
RUN git clone --depth=1 -b 8.0.0 https://github.com/neuronsimulator/nrn

RUN mkdir nrn-bld && cd nrn-bld

# export MPI_LIB_NRN_PATH=$CONDA_PREFIX/lib/libmpicxx.dylib
# export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DCURSES_NEED_NCURSES=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_RX3D=ON \
  -DNRN_ENABLE_MPI=ON \
  ../nrn

RUN cmake --build . --parallel 4 --target install && \
  cd ..

RUN cd
RUN rm -rf nrn && rm -rf nrn-bld



# --- Install LFPy ----
RUN git clone --depth=1 --branch master https://github.com/LFPy/LFPy
RUN cd LFPy && \
    pip install .
RUN cd
RUN rm -rf LFPy


# ---install misc using pip
# RUN pip install pandas
