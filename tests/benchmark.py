import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
                #path to tracts, may have to adjust if the file is moved
import tracts
import numpy
import scipy
import matplotlib.pyplot as plt
from pprint import pprint
from timeit import default_timer as time

'''
Tests for component methods of tracts core
'''

def benchmark_PTD(migration_matrix, bins, Ls, runs):
    print('Benchmarking Phase-Type Distibution')
    time_total = 0
    for run in range(runs):
        start = time()
        PTD = tracts.PhaseTypeDistribution(migration_matrix)
        PTD_hists = [PTD.tractlength_histogram_multi_windowed(population_number, bins, Ls) for population_number in range(len(migration_matrix[0]))]
        if run == 0:
            for population_number, PTD_hist in enumerate(PTD_hists):
                print(f'\nPopulation {population_number} distribution:\n{PTD_hist}')
        time_total += time()-start
    print(f'Time averaged over {runs} runs: {time_total/runs}')
    return time_total/runs

def benchmark_demography(migration_matrix, bins, Ls, runs):
    print('Benchmarking Phase-Type Distibution')
    time_total = 0
    for run in range(runs):
        start = time()
        demo = tracts.demographic_model(migration_matrix)
        PTD_hists = [numpy.array(demo.expectperbin(Ls, population_number, bins)) for population_number in range(len(migration_matrix[0]))]
        if run == 0:
            for population_number, PTD_hist in enumerate(PTD_hists):
                print(f'\nPopulation {population_number} distribution:\n{PTD_hist}')
        time_total += time()-start
    print(f'Time averaged over {runs} runs: {time_total/runs}')
    return time_total/runs

def benchmark(runs):
    '''
    Test that phase-type distributions gives the same result as demographic_model.expectperbin()
    ''' 

    bins = [0.1, 0.15561805, 0.21123611, 0.26685416, 0.32247222, 0.37809027,
            0.43370833, 0.48932638, 0.54494444, 0.60056249, 0.65618055, 0.7117986,
            0.76741666, 0.82303471, 0.87865277, 0.93427082, 0.98988888, 1.04550693,
            1.10112498, 1.15674304, 1.21236109, 1.26797915, 1.3235972, 1.37921526,
            1.43483331, 1.49045137, 1.54606942, 1.60168748, 1.65730553, 1.71292359,
            1.76854164, 1.8241597,  1.87977775, 1.93539581, 1.99101386, 2.04663191,
            2.10224997, 2.15786802, 2.21348608, 2.26910413, 2.32472219, 2.38034024,
            2.4359583, 2.49157635, 2.54719441, 2.60281246, 2.65843052, 2.71404857,
            2.76966663]
    Ls = [2.7809027354399998, 1.7928357829, 1.5953700000000002, 1.728997512, 1.27096679, 1.1710800000000001, 1.31433, 2.6344739744, 2.24544150046, 2.1288176, 2.0397778748, 1.9301017528000002, 1.8692348767000002, 1.7021899999999999, 1.6825700000000001]
    pops = [0,1]
    A = numpy.array([[0,0], [0,0],[0.1,0],[0,0],[0,0],[0,0],[0.2,0.8]])
    B = numpy.array([[0,0], [0,0], [0,0.1], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.6,0.4]])
    C = numpy.array([[0,0], [0,0], [0,0.1], [0,0.1], [0,0.1], [0,0.1], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.1,0], [0.6,0.4]])

    benchmark_PTD(C, bins, Ls, runs)
    benchmark_demography(C, bins, Ls, runs)

benchmark(100)