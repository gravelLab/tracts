from timeit import default_timer as time

import numpy

import tracts
from tests.test_data import bins, Ls

"""
Tests for component methods of tracts core
"""


def benchmark_PTD(migration_matrix, bins, Ls, runs):
    print('Benchmarking Phase-Type Distribution')
    time_total = 0
    for run in range(runs):
        start = time()
        PTD = tracts.PhaseTypeDistribution(migration_matrix)
        PTD_hists = [PTD.tractlength_histogram_multi_windowed(population_number, bins, Ls) for population_number in
                     range(len(migration_matrix[0]))]
        if run == 0:
            for population_number, PTD_hist in enumerate(PTD_hists):
                print(f'\nPopulation {population_number} distribution:\n{PTD_hist}')
        time_total += time() - start
    print(f'Time averaged over {runs} runs: {time_total / runs}')
    return time_total / runs


def benchmark_demography(migration_matrix, bins, Ls, runs):
    print('Benchmarking Phase-Type Distribution')
    time_total = 0
    for run in range(runs):
        start = time()
        demo = tracts.DemographicModel(migration_matrix)
        PTD_hists = [numpy.array(demo.expectperbin(Ls, population_number, bins)) for population_number in
                     range(len(migration_matrix[0]))]
        if run == 0:
            for population_number, PTD_hist in enumerate(PTD_hists):
                print(f'\nPopulation {population_number} distribution:\n{PTD_hist}')
        time_total += time() - start
    print(f'Time averaged over {runs} runs: {time_total / runs}')
    return time_total / runs


def run_benchmark(migration_matrix, runs):
    """
    Test that phase-type distributions gives the same result as demographic_model.expectperbin()
    """
    # pops = [0, 1]
    benchmark_PTD(migration_matrix, bins, Ls, runs)
    benchmark_demography(migration_matrix, bins, Ls, runs)


def benchmark_A(runs):
    A = numpy.array([[0, 0], [0, 0], [0.1, 0], [0, 0], [0, 0], [0, 0], [0.2, 0.8]])
    run_benchmark(A, runs)


def benchmark_B(runs):
    B = numpy.array([[0, 0], [0, 0], [0, 0.1], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0],
                     [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.6, 0.4]])
    run_benchmark(B, runs)


def benchmark_C(runs):
    C = numpy.array(
        [[0, 0], [0, 0], [0, 0.1], [0, 0.1], [0, 0.1], [0, 0.1], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0],
         [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0],
         [0.1, 0], [0.6, 0.4]])
    run_benchmark(C, runs)


# benchmark_A(100)
benchmark_B(100)
# benchmark_C(100)
