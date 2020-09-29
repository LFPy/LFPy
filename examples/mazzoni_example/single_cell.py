import LFPy
import neuron  # new
import numpy as np
import pickle
import os


def simulate_cells_serially(stimolo,
                            cell_ids,
                            data_name,
                            population_parameters,
                            cell_parameters,
                            synapse_parameters,
                            synapse_position_parameters,
                            electrode_parameters):

    print('input' + str(stimolo))

    # Load emitted spikes, 1st column: spike time, 2nd column: pre cell id
#    print "Loading locally emitted spikes"
    local_spikes_filename = population_parameters['input_dir'] + \
        'spiketimes_' + str(stimolo) + '.1.out'
    local_spikes = np.loadtxt(local_spikes_filename)
    local_sp_times = local_spikes[:, 0]
    local_sp_ids = local_spikes[:, 1]

    # Load connectivity, 1st column post id, 2nd column pre id
    connectivity_filename = population_parameters['input_dir'] \
        + 'Cmatrix_' + str(stimolo) + '.1.out'
    connectivity_file = open(connectivity_filename, "r")
    lines = connectivity_file.readlines()
    incoming_connections = []
    for line in lines:
        incoming_connections.append(np.array(line.split(), dtype='int'))
    connectivity_file.close()

    pre_cells = {}
    pre_cells['exc_exc'] = population_parameters['exc_ids']
    pre_cells['exc_inh'] = population_parameters['exc_ids']
    pre_cells['inh_exc'] = population_parameters['inh_ids']
    pre_cells['inh_inh'] = population_parameters['inh_ids']

    # n_thalamic_synapses = population_parameters['n_thalamic_synapses']
    # n_external_synapses = population_parameters['n_external_synapses']

    # setup data dictionary

    output_data = {}
    output_data['somav'] = {}
    output_data['LFP'] = {}
    output_data['somapos'] = {}
    output_data['tot_isyn'] = {}

    for i_cell, cell_id in enumerate(cell_ids):

        if cell_id in population_parameters['exc_ids']:
            print(str(cell_id))
            cell_parameters.update({'morphology': 'pyr1.hoc'})
            cell_parameters['passive_parameters'].update(
                {'g_pas': 1. / 20000.})

        elif cell_id in population_parameters['inh_ids']:
            indin = int(cell_id - max(population_parameters['exc_ids']))
            print(str(indin))
            cell_parameters.update({'morphology': 'int1.hoc'})
            cell_parameters['passive_parameters'].update(
                {'g_pas': 1. / 10000.})

        print("Setting up cell " + str(cell_id))

        cell_seed = population_parameters['global_seed'] + cell_id

        print("Setting random seed: " + str(cell_seed))

        np.random.seed(cell_seed)

        neuron.h('forall delete_section()')  # new
        cell = LFPy.Cell(**cell_parameters)  # new

        # load true position
        if cell_id in population_parameters['exc_ids']:
            cell_pos = np.loadtxt('PCsXYZ.txt')
            x, y, z = cell_pos[cell_id]

        elif cell_id in population_parameters['inh_ids']:
            cell_pos = np.loadtxt('INTsXYZ.txt')
            x, y, z = cell_pos[int(cell_id -
                                   int(min(population_parameters['inh_ids'])))]

        cell.set_pos(x=x, y=y, z=z)

        if cell_id in population_parameters['exc_ids']:
            local_synapse_types = ['exc_exc', 'inh_exc']
            # thalamic_synapse_type = 'thalamic_exc'
            # external_synapse_type = 'external_exc'

        elif cell_id in population_parameters['inh_ids']:
            local_synapse_types = ['exc_inh', 'inh_inh']
            # thalamic_synapse_type = 'thalamic_inh'
            # external_synapse_type = 'external_inh'

        for synapse_type in local_synapse_types:

            print("Setting up local synapses: ", synapse_type)

            pre_ids = incoming_connections[cell_id]
            # n_synapses = len(pre_ids)

            for i_synapse, pre_id in enumerate(pre_ids):
                if pre_id in pre_cells[synapse_type]:
                    syn_idx = int(cell.get_rand_idx_area_norm(
                        **synapse_position_parameters[synapse_type]))
                    synapse_parameters[synapse_type].update({'idx': syn_idx})
                    synapse = LFPy.Synapse(cell,
                                           **synapse_parameters[synapse_type])
                    spike_times =\
                        local_sp_times[np.where(local_sp_ids == pre_id)[0]]
                    synapse.set_spike_times(spike_times)

        print("Setting up thalamic synapses")

        # Load thalamic input spike times, 1st column time,
        # 2nd column post cell id
        #   print "Loading thalamic input spikes"
#         thalamic_spikes_filename = population_parameters['input_dir'] \
#              +'ths/th_'+str(stimolo)+'_'+str(cell_id)+'.out'

#         print thalamic_spikes_filename
#
#         thalamic_spike_times = np.loadtxt(thalamic_spikes_filename)

#         synapse_ids =\
#                  np.random.randint(0,n_thalamic_synapses,len(thalamic_spike_times))

#         for i_synapse in xrange(n_thalamic_synapses):
#             syn_idx = int(cell.get_rand_idx_area_norm(\
#                         **synapse_position_parameters[thalamic_synapse_type]))
#             synapse_parameters[thalamic_synapse_type].update({'idx':syn_idx})
#             synapse = LFPy.Synapse(cell, \
#                             **synapse_parameters[thalamic_synapse_type])
#             spike_times = \
#                     thalamic_spike_times[np.where(synapse_ids==i_synapse)[0]]
#             synapse.set_spike_times(spike_times)

        print("Setting up external synapses")

        # Load external cortico-cortical input rate
        #  print "Loading external input spikes"
#         external_spikes_filename = population_parameters['input_dir'] \
#              + 'ccs/cc_'+str(stimolo)+'_'+str(cell_id)+'.out'
#         external_spike_times = np.loadtxt(external_spikes_filename)
#
#         synapse_ids =\
#              np.random.randint(0,n_external_synapses,len(external_spike_times))
#
#         for i_synapse in xrange(n_external_synapses):
#             syn_idx = int(cell.get_rand_idx_area_norm(\
#                         **synapse_position_parameters[external_synapse_type]))
#
#             synapse_parameters[external_synapse_type].update({'idx':syn_idx})
#             synapse = LFPy.Synapse(cell,\
#                             **synapse_parameters[external_synapse_type])
#             spike_times =\
#                 external_spike_times[np.where(synapse_ids==i_synapse)[0]]
#             synapse.set_spike_times(spike_times)

        # Run simulation
        print("Running simulation...")
        cell.simulate(rec_imem=True)

        # Calculate LFP
        print("Calculating LFP")
        electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
        electrode.calc_lfp()

        # Store data
        print("Storing data")
        output_data['LFP'][cell_id] = electrode.LFP
        output_data['somav'][cell_id] = cell.somav
        output_data['somapos'][cell_id] = cell.somapos

    output_data['tvec'] = cell.tvec

    print("Saving data to file")

    if not os.path.isdir(population_parameters['save_to_dir']):
        os.mkdir(population_parameters['save_to_dir'])
    print(output_data)
    pickle.dump(output_data, open(
        population_parameters['save_to_dir'] + data_name + str(stimolo), "wb"))
