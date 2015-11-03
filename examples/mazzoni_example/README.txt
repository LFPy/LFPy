This is an example simulation code as used in 

Mazzoni A, Lindén H, Cuntz H, Lansner A, Panzeri S and Einevoll GT. Computing the local
field potential (LFP) from integrate-and-fire network models. PLoS Computational
Biology (accepted).

The example provides a downsized simulation of 50 cells* (compared to 5000 in the paper) where the detailed morphologies have been replaced with two template ball-and-stick neurons for all pyramidal cell and interneurons respectively. 

The simulation can be run with or without an MPI environment (set by the parameter 'with_MPI' in 'ref_sim.py'. If the simulation is run without MPI all cell with be simulated serially. If the simulation is run with MPI the simulations of invidual cells are distributed on all available MPI processes. 

The code example consists of the following files:

Simulation:
- ref_sim.py : Defines parameters and runs the population simulation
- single_cell.py: Called by ref_sim.py and runs the simulation for each cell in the population

Auxilliary files:
- exp2isyn.mod: Definition of synapse model
- Cmatrix_15.1.out: Connectivity of the LIF network (each row i contains ids of all presynaptic neurons to cell i)
- spiketimes_15.1.out: Spiketimes in the LIF network (each row i contains all spikes of neuron i)
- pyr1.hoc: Cell morphology for pyramidal neurons 
- int1.hoc: Cell morphology for interneurons
- PCsXYZ.txt: Soma positions pyramidal cells
- INTsXYZ.txt: Soma positions for interneurons

How to run the simulation example:
1. Compile the synapse model by calling 'nrnivmodl' (Linux/OSX) or 'mknrndll' (Windows), only necessary first time the example is run.
2. Run 'python ref_sim.py' 

Output data:
- The code will produce a data output file 'results/intput15.p15' containing all output variables in a single file (if run without MPI) or in several files (one for each rank) if the example is run within a MPI environment.
- The data is stored in pickled file and can be loaded by the following code snipped:

"
import numpy as np
import pickle
data = pickle.load( open("results/input15_output_data_rank0.p15", "rb" ) )
"

data is a dictionary containing all simulation output:
- data['LFP'] contains the LFP contributions from all single neurons at all electrode positions: 'data['LFP'][cell_id][electrode_id]

* A further difference with the simulations used in the paper is that thalamic and corticortical inputs are not included. Code for the external input is commented out in 'single_cell.py' but can be included by uncommented the code on lines 121-159.


# HL 2015-11-02