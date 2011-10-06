#!/bin/sh
#PBS -lnodes=1:ppn=8
#PBS -lwalltime=0:10:00
#PBS -A nn4661k

cd $PBS_O_WORKDIR
export LD_PRELOAD=libmkl_intel_lp64.so:libmkl_intel_thread.so:libmkl_core.so:libguide.so
mpirun -n 8 -x LD_PRELOAD python ex_MPI_L5pyr_active.py
wait