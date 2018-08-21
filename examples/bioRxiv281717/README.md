bioRxiv 281717
==============

This folder contains the simulation files for the preprint describing the LFPy v2.0 release:

     Multimodal modeling of neural network activity: computing LFP, ECoG, EEG and MEG signals with LFPy2.0
     Espen Hagen, Solveig Næss, Torbjørn V Ness, Gaute T Einevoll
     bioRxiv 281717; doi: https://doi.org/10.1101/281717

Files
=====
* requirements.txt: pip requirements file to install modules needed for the examples. 
* figure_2.py: Simulation and plotting script reproducing figure 2 of the manuscript
* simplemorpho_modded.hoc: morphology file used for figure 2.
* example_parallel_network.py: Main simulation script for network, needed for figure 4-6. See docstring in file for set up details.
* example_parallel_network_parameters.py: Corresponding parameter script
* example_parallel_network.job: SLURM workload manager job submission script. See main sim. script for details.
* example_parallel_network_methods.py: Helper script
* example_parallel_network_plotting.py: Helper script
* figure_4.py: Generates figure 4
* figure_5.py: Generates figure 5
* figure_6.py: Generates figure 6
* figure_7_8/*: Code for running benchmark simulations and generation of figures 7 and 8

