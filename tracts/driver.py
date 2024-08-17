import tracts
import numpy
import ruamel.yaml
import re
import logging
import os
import __main__
import sys
from pathlib import Path
import numbers
logger = logging.getLogger(__name__)

filepath_error_additional_message = '\nPlease ensure that the file path is either absolute, or relative to the working directory, script directory, or the directory of the driver yaml.'

def locate_file_path(filename, absolute_driver_yaml_path=None):
    for filepath, method_name in ((Path(filename), 'using working directory'), 
                                  (Path(__main__.__file__).parent / filename, 'using script directory'), 
                                  (absolute_driver_yaml_path.parent / filename if isinstance(absolute_driver_yaml_path, Path) else Path(''), 'using driver yaml')
                                ):
        logger.info(f'{method_name}: {filepath}')
        if filepath.is_file():
            logger.info(f'Found {filename} {method_name}.')
            return filepath
    for pathname in sys.path:
        if (Path(pathname) / filename).is_file():
            logger.info(f'Found {filename} from {pathname}.')
            return Path(pathname) / filename
    #raise IndexError(f'The file {filename} could not be found. {filepath_error_additional_message}')
    return None

def run_tracts(driver_filename):
    driver_path = locate_file_path(driver_filename)
    driver_spec = load_driver_file(driver_path)

    chromosome_list = parse_chromosomes(driver_spec['samples']['chromosomes'])
    logger.info(f'Chromosomes: {chromosome_list}')
    individual_filenames = parse_individual_filenames(driver_spec['samples']['individual_names'], driver_spec['samples']['filename_format'], labels = driver_spec['samples']['labels'], directory=driver_spec['samples']['directory'], absolute_driver_yaml_path=driver_path)

    exclude_tracts_below_cM = driver_spec['exclude_tracts_below_cm'] if 'exclude_tracts_below_cm' in driver_spec else 10

    # load the population
    pop = tracts.population(filenames_by_individual=individual_filenames, selectchrom=chromosome_list)
    (bins, data) = pop.get_global_tractlengths(npts=50, exclude_tracts_below_cM=exclude_tracts_below_cM)

    logger.info(f'Bins: {bins}')

    time_scaling_factor = driver_spec['time_scaling_factor'] if 'time_scaling_factor' in driver_spec else 1

    model = load_model_from_driver(driver_spec, driver_path)

    population_labels = model.population_indices.keys()

    logger.info(f'Model Parameters: {model.free_params}')

    if 'fix_parameters_from_ancestry_proportions' in driver_spec:
        ancestry_proportions = calculate_ancestry_proportions(pop, population_labels)
        model.fix_ancestry_proportions(driver_spec['fix_parameters_from_ancestry_proportions'], ancestry_proportions)
    
    func = get_time_scaled_model_func(model, time_scaling_factor)
    bound = get_time_scaled_model_bounds(model, time_scaling_factor)

    if type(driver_spec['start_params']) is not dict:
        raise KeyError('You must specify initial parameters or parameter ranges under "start_params".')
       
    params_found, likelihoods = run_model_multi_params(func, bound, pop, population_labels, parse_start_params(driver_spec['start_params'], driver_spec['repetitions'], driver_spec['seed'], model, time_scaling_factor), exclude_tracts_below_cM=exclude_tracts_below_cM)
    print("likelihoods found: ", likelihoods)
    optimal_params = min(zip(params_found,likelihoods), key=lambda x: x[1])[0]
    optimal_params = scale_select_indices(optimal_params, model.is_time_param(), time_scaling_factor)
    if 'output_filename_format' in driver_spec:
        output_simulation_data(pop, optimal_params, model, driver_spec)
    return

def load_driver_file(driver_path):
    if driver_path is None:
        raise OSError(f'Driver yaml file could not be found. {filepath_error_additional_message}')
    driver_spec = None
    with driver_path.open() as file, ruamel.yaml.YAML(typ="safe") as yaml:
        driver_spec = yaml.load(file)
    if not isinstance(driver_spec, dict):
        raise ValueError('Driver yaml file was invalid.')
    return driver_spec

def load_model_from_driver(driver_spec, driver_path):
    if 'model_filename' not in driver_spec:
        raise ValueError('You must specify the file path to your model under "model_filename".')
    model_path = locate_file_path(driver_spec['model_filename'], driver_path)
    if model_path is None:
        raise OSError(f'Model yaml file could not be found. {filepath_error_additional_message}')
    model = tracts.ParametrizedDemography.load_from_YAML(model_path.resolve())
    return model
    
def parse_chromosomes(chromosome_spec, chromosomes=None):
    if chromosomes == None:
        chromosomes = []
    if isinstance(chromosome_spec, int):
        chromosomes.append(chromosome_spec)
        return chromosomes
    if isinstance(chromosome_spec, list):
        [parse_chromosomes(subspec, chromosomes) for subspec in chromosome_spec]
        return chromosomes
    try:
        chromosome_spec = chromosome_spec.split('-')
        chromosomes.extend(range(int(chromosome_spec[0]), int(chromosome_spec[1])+1))
        return chromosomes
    except Exception as e:
        raise ValueError('Chromosomes should be an int, range (ie: 1-22), or list.') from e
    
    
def parse_individual_filenames(individual_names, filename_string, labels = ['A', 'B'], directory='', absolute_driver_yaml_path=None):
    def _find_individual_file(individual_name, label_val):
        filepath = locate_file_path(directory+filename_string.format(name = individual_name, label = label_val), absolute_driver_yaml_path)
        if filepath is None:
            raise IndexError(f'File for individiual {individual_name} ("{directory+filename_string.format(name = individual_name, label = label_val)}") could not be found. {filepath_error_additional_message}')
        return str(filepath)
    individual_filenames = {individual_name: [_find_individual_file(individual_name, label_val) for label_val in labels] for individual_name in individual_names}
    return individual_filenames

def parse_start_params(start_param_bounds, repetitions=1, seed=None, model: tracts.ParametrizedDemography=None, time_scaling_factor=1):
    #num_params = len(model.free_params) - len(model.params_fixed_by_ancestry)
    num_params = len(model.free_params)
    rng = numpy.random.default_rng(seed=seed)
    start_params = rng.random((repetitions, num_params))
    for param_name, param_info in model.free_params.items():
        if param_name in model.params_fixed_by_ancestry:
            start_params[:,param_info['index']] = 0
            continue
        if param_name not in start_param_bounds:
            raise KeyError(f"Initial values were not specified for parameter '{param_name}'.")
        if isinstance(start_param_bounds[param_name], numbers.Number):
            start_params[:,param_info['index']] = start_param_bounds[param_name]
        else:
            try:
                bounds = [float(bound) for bound in start_param_bounds[param_name].split('-')]
                assert len(bounds) == 2
                start_params[:,param_info['index']] *= bounds[1] - bounds[0]
                start_params[:,param_info['index']] += bounds[0]
            except Exception as e:
                raise ValueError("Initial values must be specified as a range (ie: 5-7) or a number.") from e
        if param_info['type'] == 'time':
            start_params[:,param_info['index']] *= 1/time_scaling_factor
    #logger.info(f' Start Params: \n {start_params}')
    if model.params_fixed_by_ancestry is not None:
        start_params = numpy.transpose([start_params[:,param_info['index']] for param_name, param_info in model.free_params.items() if param_name not in model.params_fixed_by_ancestry])
    logger.info(f' Start Params: \n {start_params}')
    return start_params
    
def scale_select_indices(arr, indices_to_scale, scaling_factor = 1):
    if len(indices_to_scale) != len(arr):
        raise ValueError(f'Length of array ({len(arr)}) was not equal to length of indices_to_scale ({len(indices_to_scale)}).')
    return (numpy.multiply(indices_to_scale, scaling_factor - 1) + 1) * arr

def get_time_scaled_model_func(model, time_scaling_factor):
    return lambda params: model.get_migration_matrix(scale_select_indices(params, model.is_time_param(), time_scaling_factor))

def get_time_scaled_model_bounds(model, time_scaling_factor):
    return lambda params: model.check_invalid(scale_select_indices(params, model.is_time_param(), time_scaling_factor))

def randomize(arr, a, b):
    # takes an array and multiplies every element by a factor between a and b,
    # uniformly.
    return ((b-a) * numpy.random.random(arr.shape) + a)*arr

def run_model_multi_params(model_func, bound_func, population, population_labels, startparams_list, exclude_tracts_below_cM=0, modelling_method=tracts.PhaseTypeDistribution):
    optimal_params = []
    likelihoods = []
    for start_params in startparams_list:
        logger.info(f'Start params: {start_params}')
        params_found, likelihood_found = run_model(model_func, bound_func, population, population_labels, start_params, exclude_tracts_below_cM=exclude_tracts_below_cM, modelling_method=modelling_method)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
    return optimal_params, likelihoods

def run_model(model_func, bound_func, population, population_labels, startparams, exclude_tracts_below_cM=0, modelling_method=tracts.PhaseTypeDistribution):
    Ls = population.Ls
    nind = population.nind
    (bins, data) = population.get_global_tractlengths(npts=50, exclude_tracts_below_cM=exclude_tracts_below_cM)
    data = [data[poplab] for poplab in population_labels]
    xopt = tracts.optimize_cob(startparams, bins, Ls, data, nind, model_func, outofbounds_fun=bound_func, epsilon=1e-2, modelling_method=modelling_method)
    optmod = modelling_method(model_func(xopt))
    optlik = optmod.loglik(bins, Ls, data, nind)
    return xopt, optlik

def output_simulation_data(sample_population, optimal_params, model: tracts.ParametrizedDemography, driver_spec):

    if 'output_directory' in driver_spec:
        output_dir = driver_spec['output_directory']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = ''

    output_filename_format = driver_spec['output_filename_format']
    exclude_tracts_below_cM = driver_spec['exclude_tracts_below_cm'] if 'exclude_tracts_below_cm' in driver_spec else 10
    (bins, data) = sample_population.get_global_tractlengths(npts=50, exclude_tracts_below_cM=exclude_tracts_below_cM)
    Ls = sample_population.Ls
    nind = sample_population.nind

    with open(output_dir + output_filename_format.format(label='tract_length_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, bins)))

    with open(output_dir + output_filename_format.format(label='sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            fdat.write("\t".join(map(str, data[population])) + "\n")

    optimal_model = tracts.PhaseTypeDistribution(model.get_migration_matrix(optimal_params))
    
    with open(output_dir + output_filename_format.format(label='migration_matrix'), 'w') as fmig2:
        for line in optimal_model.migration_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")

    with open(output_dir + output_filename_format.format(label='predicted_tract_distribution'), 'w') as fpred2:
        for popnum in range(len(data)):
            fpred2.write("\t".join(map(
                str,
                nind * numpy.array(optimal_model.expectperbin(Ls, popnum, bins))))
        + "\n")

    with open(output_dir + output_filename_format.format(label='optimal_parameters'), 'w') as fpars2:
        fpars2.write("\t".join(map(str, optimal_params)) + "\n")

def calculate_ancestry_proportions(sample_population: tracts.population, population_labels):
    # calculate the proportion of ancestry in each individual
    bypopfrac = [[] for i in range(len(population_labels))]
    for ind in sample_population.indivs:
        for ii, poplab in enumerate(population_labels):
            bypopfrac[ii].append(ind.ancestryProps([poplab]))
    return numpy.mean(bypopfrac, axis=1).flatten()
