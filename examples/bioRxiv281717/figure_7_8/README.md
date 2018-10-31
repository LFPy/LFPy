figure_7_8
==========
Parameterspace and corresponding simulation files to run benchmarks of LFPy2.0's
parallel performance with networks as presented in figure 7 and 8 in:

     Multimodal modeling of neural network activity: computing LFP, ECoG, EEG and MEG signals with LFPy2.0
     Espen Hagen, Solveig Næss, Torbjørn V Ness, Gaute T Einevoll
     bioRxiv 281717; doi: https://doi.org/10.1101/281717

These files are mostly here for future reference - a single successful run could
cost as much as 50,000 core hours or more!!!!

The main simulation scripts are similar to those in the above folder, except
that the number of populations is two, with variable population sizes,
and calculations of set up times etc.

Files
=====
* example_parallel_network.py: Main simulation script for network. See file header for set up instructions. 
* example_parallel_network_parameters.py: Corresponding parameter script
* example_parallel_network_methods.py: Helper script
* example_parallel_network_plotting.py: Helper script
* example_parrallel_network_parameterspace.py: Main script to generate parameter files, job scripts and submission of jobs to the cluster queue using slurm.
* figure_7_8.py: Script to generate figure output
