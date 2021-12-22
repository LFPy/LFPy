# -------- base ---------
FROM buildpack-deps:hirsute AS base

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3-dev=3.9.4-1 \
    python3-scipy=1.6.0-2 \
    python3-matplotlib=3.3.4-1 \
    python3-pytest=6.0.2-2ubuntu1 \
    python3-pip=20.3.4-1ubuntu2 \
    python3-pandas=1.1.5+dfsg-2 \
    cython3=0.29.21-1ubuntu3 \
    jupyter=4.7.1-1\
    jupyter-notebook=6.2.0-1 \
    ipython3=7.20.0-1 \
    cmake=3.18.4-2ubuntu1 \
    bison=2:3.7.5+dfsg-1 \
    flex=2.6.4-8 \
    libmpich-dev=3.4.1-3build1 \
    libncurses-dev=6.2+20201114-2build1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10

# --- install NEURON from source
# (only way to get builds on arm64/aarch64 hosts working) -------
RUN git clone --depth 1 -b 8.0.0 https://github.com/neuronsimulator/nrn.git /usr/src/nrn
RUN mkdir nrn-bld

RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/ \
  -DCURSES_NEED_NCURSES=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_RX3D=ON \
  -DNRN_ENABLE_PYTHON=ON \
  -S /usr/src/nrn \
  -B nrn-bld

RUN cmake --build nrn-bld --parallel 4 --target install

# add nrnpython to PYTHONPATH
ENV PYTHONPATH /usr/local/lib/python:${PYTHONPATH}

# clean up
RUN rm -r /usr/src/nrn
RUN rm -r nrn-bld

# --- Install LFPy ----
RUN pip install git+https://github.com/LFPy/LFPy@master#egg=LFPy
