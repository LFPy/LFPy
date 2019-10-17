FROM continuumio/miniconda3

RUN conda config --add channels conda-forge

RUN conda create -n lfpy python=3.7 lfpy ipython jupyter git matplotlib make gxx_linux-64

RUN echo "source activate lfpy" > ~/.bashrc
ENV PATH /opt/conda/envs/lfpy/bin:$PATH

RUN git clone https://github.com/LFPy/LFPy.git /opt/LFPy
