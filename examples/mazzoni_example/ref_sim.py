import numpy as np
from single_cell import simulate_cells_serially

with_MPI = False

# Define all parameters ##################################################

population_parameters = {
    'global_seed': 0,
    'radius': 250.,  # 250
    'exc_ids': np.arange(0, 40),
    'inh_ids': np.arange(40, 50),
    'cells_to_simulate': np.arange(50),
    'n_thalamic_synapses': 100,  # 100
    'n_external_synapses': 100,
    'data_dir': '',
    'save_to_dir': 'results/',
    'input_dir': '',
}


cell_parameters = {
    'morphology': '',  # set in 'single_cell.py'
    'dt': 2**-4,  # 2**-4=0.0625 ms
    'nsegs_method': 'lambda100',
    'Ra': 150.,
    'cm': 1.0,
    'v_init': -65.,
    'passive': True,
    'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
    'tstart': -50.,  # start time of simulation, recorders start at t=0
    'tstop': 1000,   # stop simulation at 1000 ms. These can be overridden
}

synapse_parameters = {}

synapse_parameters['exc_exc'] = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .4,                # syn. time constant
    'tau2': 2.,                # syn. time constant
    'weight': .070,            # syn. weight
    'record_current': True,
}

synapse_parameters['inh_exc'] = {
    'idx': 0,  # to be set later
    'e': -90.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .25,                # syn. time constant
    'tau2': 5.,                # syn. time constant
    'weight': -.145,            # syn. weight
    'record_current': True,
}

synapse_parameters['thalamic_exc'] = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .4,                # syn. time constant
    'tau2': 2.,                # syn. time constant
    'weight': 0.091,            # syn. weight
    'record_current': True,
}


synapse_parameters['external_exc'] = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .4,                # syn. time constant
    'tau2': 2.,                # syn. time constant
    'weight': .070,            # syn. weight
    'record_current': True,
}

synapse_parameters['exc_inh'] = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .2,                # syn. time constant
    'tau2': 1.,                # syn. time constant
    'weight': .093,            # syn. weight
    'record_current': True,
}


synapse_parameters['inh_inh'] = {
    'idx': 0,  # to be set later
    'e': -90.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .25,                # syn. time constant
    'tau2': 5.,                # syn. time constant
    'weight': -.092,            # syn. weight
    'record_current': True,
}

synapse_parameters['thalamic_inh'] = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .2,                # syn. time constant
    'tau2': 1.,                # syn. time constant
    'weight': 0.126,            # syn. weight
    'record_current': True,
}

synapse_parameters['external_inh'] = {
    'idx': 0,  # to be set later
    'e': 0.,                   # reversal potential
    'syntype': 'Exp2ISyn',       # synapse type
    'tau1': .2,                # syn. time constant
    'tau2': 1.,                # syn. time constant
    'weight': .093,            # syn. weight
    'record_current': True,
}


synapse_position_parameters = {}

synapse_position_parameters['exc_exc'] = {
    'z_min': -300.,
    'z_max': 300.
}

synapse_position_parameters['inh_exc'] = {
    'z_min': -300.,
    'z_max': 0.
}

synapse_position_parameters['thalamic_exc'] = {
    'z_min': -300.,
    'z_max': 300.
}

synapse_position_parameters['external_exc'] = {
    'z_min': -300,
    'z_max': 300.
}

synapse_position_parameters['exc_inh'] = {
    'z_min': -250.,
    'z_max': 0.
}

synapse_position_parameters['inh_inh'] = {
    'z_min': -300.,
    'z_max': 0.
}

synapse_position_parameters['thalamic_inh'] = {
    'z_min': -300.,
    'z_max': 0.
}

synapse_position_parameters['external_inh'] = {
    'z_min': -300,
    'z_max': 0.
}

Z = np.arange(-400, 400, 25)

electrode_parameters = {
    'sigma': 0.3,      # extracellular conductivity

    'z': Z,
    'x': np.zeros(Z.size),  # electrode requires 1d vector of positions
    'y': np.zeros(Z.size),
}

# Run simulation #########################################################

if __name__ == '__main__':
    for stim in [15]:
        # If simulating using MPI
        if with_MPI:
            from mpi4py import MPI

            # MPI related stuff
            COMM = MPI.COMM_WORLD
            SIZE = COMM.Get_size()
            RANK = COMM.Get_rank()

            n_cells_to_simulate = len(
                population_parameters['cells_to_simulate'])

            n_cells_per_RANK = int(np.ceil(1. * n_cells_to_simulate / SIZE))

            if RANK * n_cells_per_RANK < n_cells_to_simulate:

                top_lim = min(
                    (RANK + 1) * n_cells_per_RANK,
                    n_cells_to_simulate)

                cell_ids = np.arange(RANK * n_cells_per_RANK, top_lim)

                print(
                    "Simulating RANK" +
                    str(RANK) +
                    ', cells' +
                    str(cell_ids) +
                    ". top_lim " +
                    str(top_lim))

                print('input' + str(stim))

                simulate_cells_serially(
                    stimolo=stim,
                    cell_ids=cell_ids,
                    data_name='input' +
                    str(stim) + '_output_data_rank' +
                    str(RANK) + '.p',
                    population_parameters=population_parameters,
                    cell_parameters=cell_parameters,
                    synapse_parameters=synapse_parameters,
                    synapse_position_parameters=synapse_position_parameters,
                    electrode_parameters=electrode_parameters)

        else:

            # If simulating on one machine without MPI

            print('input' + str(stim))

            simulate_cells_serially(
                stimolo=stim,
                cell_ids=population_parameters['cells_to_simulate'],
                data_name='input' + str(stim) + 'output_data.p',
                population_parameters=population_parameters,
                cell_parameters=cell_parameters,
                synapse_parameters=synapse_parameters,
                synapse_position_parameters=synapse_position_parameters,
                electrode_parameters=electrode_parameters)
