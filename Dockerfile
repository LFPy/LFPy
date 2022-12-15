# -------- base ---------
FROM buildpack-deps:jammy AS base

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3-dev=3.10.6-1~22.04 \
    python3-scipy=1.8.0-1exp2ubuntu1 \
    python3-matplotlib=3.5.1-2build1 \
    python3-pytest=6.2.5-1ubuntu2 \
    python3-pip=22.0.2+dfsg-1 \
    python3-pandas=1.3.5+dfsg-3 \
    python3-h5py=3.6.0-2build1 \
    cython3=0.29.28-1ubuntu3 \
    cmake=3.22.1-1ubuntu1.22.04.1 \
    bison=2:3.8.2+dfsg-1build1 \
    flex=2.6.4-8build2 \
    libmpich-dev=4.0-3 \
    libncurses-dev=6.3-2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

# --- Install LFPy ----
RUN pip install --no-cache-dir mpi4py==3.1.4 && \
  pip install --no-cache-dir jupyterlab==3.5.1 && \
  pip install --no-cache-dir git+https://github.com/LFPy/LFPy@master#egg=LFPy && \
  pip cache purge