#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Define ParameterSpace for benchmarking of scaling with different MPI pools
'''
import os
import numpy as np
import operator
import pickle
import hashlib
import parameters as ps


def sort_deep_dict(d):
    '''
    sort arbitrarily deep dictionaries into tuples

    Arguments
    ---------
    d: dict

    Returns:
    x: list of tuples of tuples of tuples ...
    '''
    x = sorted(iter(d.items()), key=operator.itemgetter(0))
    for i, (key, value) in enumerate(x):
        if isinstance(value, dict) or isinstance(value, ps.ParameterSet):
            y = sorted(iter(value.items()), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if isinstance(v, dict) or isinstance(v, ps.ParameterSet):
                    y[j] = (k, sort_deep_dict(v))
    return x


def get_unique_id(paramset):
    '''
    create a unique hash key for input dictionary

    Arguments
    ---------
    paramset: dict
        parameter dictionary

    Returns
    -------
    key: str
        hash key

    '''
    sorted_params = sort_deep_dict(paramset)
    string = pickle.dumps(sorted_params)
    key = hashlib.md5(string).hexdigest()
    return key


PSPACES = dict()

# check scaling with MPI pool size
PSPACES['MPI'] = ps.ParameterSpace(dict())
PSPACES['MPI'].update(dict(
    # Population sizes
    POP_SIZE_REF=[2400, 480],

    # allow different seeds for different network iterations
    GLOBALSEED=ps.ParameterRange([1234, 65135, 216579876]),

    # MPI pool size
    MPISIZE=ps.ParameterRange([120, 240, 480, 960, 1920, 2880]),

    # bool flag switching LFP calculations on or off (faster)
    COMPUTE_LFP=ps.ParameterRange([False, True]),

    # population size scaling (multiplied with values in
    # populationParams['POP_SIZE']):
    POPSCALING=ps.ParameterRange([1.]),

    # preserve expected synapse in-degree or total number of connections
    PRESERVE=ps.ParameterRange(['indegree'])
))

PSPACES['MPI5'] = ps.ParameterSpace(dict())
PSPACES['MPI5'].update(dict(
    # Population sizes
    POP_SIZE_REF=[12000, 2400],

    # allow different seeds for different network iterations
    GLOBALSEED=ps.ParameterRange([1234, 65135, 216579876]),

    # MPI pool size
    MPISIZE=ps.ParameterRange([600, 1200, 2400, 4800]),

    # bool flag switching LFP calculations on or off (faster)
    COMPUTE_LFP=ps.ParameterRange([False, True]),

    # population size scaling (multiplied with values in
    # populationParams['POP_SIZE']):
    POPSCALING=ps.ParameterRange([1.]),

    # preserve expected synapse in-degree or total number of connections
    PRESERVE=ps.ParameterRange(['indegree'])
))


# check scaling with population size
PSPACES['POP'] = ps.ParameterSpace(dict())
PSPACES['POP'].update(dict(
    # Population sizes
    POP_SIZE_REF=[2400, 480],

    # allow different seeds for different network iterations
    GLOBALSEED=ps.ParameterRange([1234, 65135, 216579876]),

    # MPI pool size
    MPISIZE=ps.ParameterRange([480]),

    # bool flag switching LFP calculations on or off (faster)
    COMPUTE_LFP=ps.ParameterRange([False, True]),

    # population size scaling (multiplied with values in
    # populationParams['POP_SIZE']):
    POPSCALING=ps.ParameterRange([0.2, 0.25, 0.5, 1., 2.0, 4.0]),

    # preserve expected synapse in-degree or total number of connections
    # across population scalings
    PRESERVE=ps.ParameterRange(['total', 'indegree'])
))

PSPACES['POP5'] = ps.ParameterSpace(dict())
PSPACES['POP5'].update(dict(
    # Population sizes
    POP_SIZE_REF=[12000, 2400],

    # allow different seeds for different network iterations
    GLOBALSEED=ps.ParameterRange([1234, 65135, 216579876]),

    # MPI pool size
    MPISIZE=ps.ParameterRange([2400]),

    # bool flag switching LFP calculations on or off (faster)
    COMPUTE_LFP=ps.ParameterRange([False, True]),

    # population size scaling (multiplied with values in
    # populationParams['POP_SIZE']):
    POPSCALING=ps.ParameterRange([0.2, 0.25, 0.5, 1., 2.0, 4.0]),

    # preserve expected synapse in-degree or total number of connections
    # across population scalings
    PRESERVE=ps.ParameterRange(['total', 'indegree'])
))

# PSPACES['TEST'] = ps.ParameterSpace(dict())
# PSPACES['TEST'].update(dict(
#     # Population sizes
#     POP_SIZE_REF = [2400, 480],
#
#     # allow different seeds for different network iterations
#     GLOBALSEED = ps.ParameterRange([1234, 123456, 1234665465,
#                                     1343645757, 12423, 982736]),
#
#     # MPI pool size
#     MPISIZE = ps.ParameterRange([480]),
#
#     # bool flag switching LFP calculations on or off (faster)
#     COMPUTE_LFP = ps.ParameterRange([True]),
#
#     # population size scaling (multiplied with values in
#     # populationParams['POP_SIZE']):
#     POPSCALING = ps.ParameterRange([1.,]),
#
#     # preserve expected synapse in-degree or total number of connections
#     # across population scalings
#     PRESERVE = ps.ParameterRange(['indegree'])
#     ))

jobscript_stallo = '''#!/bin/bash
##################################################################
# SBATCH --job-name {}
# SBATCH --time {}
# SBATCH -o {}
# SBATCH -e {}
# SBATCH --mem-per-cpu=2000MB
# SBATCH --ntasks {}
##################################################################
# from here on we can run whatever command we want
unset DISPLAY # DISPLAY env variable somehow problematic with Slurm
srun --mpi=pmi2 python example_parallel_network.py {}
'''

jobscript_jureca = '''#!/bin/bash
##################################################################
# SBATCH --job-name {}
# SBATCH --time {}
# SBATCH -o {}
# SBATCH -e {}
# SBATCH -N {}
# SBATCH --ntasks {}
# SBATCH --mem-per-cpu={}
# SBATCH --exclusive
##################################################################
# from here on we can run whatever command we want
unset DISPLAY # DISPLAY somehow problematic with Slurm
srun python example_parallel_network.py {}
'''

LOGDIR = 'logs'
OUTPUTDIR = 'output'
JOBDIR = 'jobs'
PSETDIR = 'parameters'


if __name__ == '__main__':
    for f in [LOGDIR, JOBDIR, PSETDIR]:
        if not os.path.isdir(f):
            os.mkdir(f)
        else:
            os.system('rm {}'.format(os.path.join(f, '*')))

    runningjobs = []

    for PSPACE in PSPACES.values():
        for pset in PSPACE.iter_inner():
            # get identifier
            ps_id = get_unique_id(pset)
            print(ps_id)

            # write parameterset file
            pset.save(os.path.join(PSETDIR, ps_id + '.txt'))

            # memory (MB)
            mem_per_cpu = 4000

            # walltime (2400 seconds per 480 MPI threads and popscaling 1 and
            # neuron count 2880)
            wt = 2400 * 480 / pset.MPISIZE * pset.POPSCALING * \
                np.sum(pset.POP_SIZE_REF) / 2880.
            wt = '%i:%.2i:%.2i' % (wt // 3600,
                                   (wt - wt // 3600 * 3600) // 60,
                                   (wt - wt // 60 * 60))

            # logfile
            logfile = os.path.join(LOGDIR, ps_id + '.txt')

            # write and submit jobscript
            try:
                if os.environ['HOSTNAME'].rfind('jr') >= 0:  # JURECA
                    NPERNODE = 24
                    N = int(pset.MPISIZE / NPERNODE)
                    jobscript = jobscript_jureca.format(
                        ps_id, wt, logfile, logfile, N, pset.MPISIZE,
                        mem_per_cpu, ps_id)
                elif os.environ['HOSTNAME'].rfind('stallo') >= 0 \
                        or os.environ['HOSTNAME'].rfind('local') >= 0:
                    jobscript = jobscript_stallo.format(
                        ps_id, wt, logfile, logfile, pset.MPISIZE,
                        mem_per_cpu, ps_id)
                else:
                    pass  # modify as needed, or assuming runs are done locally
                # write jobscript file
                fpath = os.path.join(JOBDIR, ps_id + '.sh')
                f = open(fpath, 'w')
                f.writelines(jobscript)
                f.close()

                # submit job
                if ps_id not in runningjobs:
                    os.system('sbatch {}'.format(fpath))

                runningjobs.append(ps_id)
            except KeyError:
                pass
