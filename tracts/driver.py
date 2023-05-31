import tracts
import numpy
import ruamel.yaml
import re
import logging
import os
import __main__
import sys
from pathlib import Path
logger = logging.getLogger(__name__)

def locate_file_path(filename, absolute_driver_yaml_path=None):
    for filepath, method_name in ((Path(filename), 'from working directory.'), 
                                  (Path(__main__.__file__).parent / filename, 'using script directory.'), 
                                  (Path(sys.path[0]) / filename, 'using sys.path[0]'),
                                  (absolute_driver_yaml_path.parent / filename if isinstance(absolute_driver_yaml_path, Path) else Path(''), 'using driver yaml.')
                                ):
        #logger.info(f'{method_name}: {filepath}')
        if filepath.is_file():
            logger.info(f'Found {filename} {method_name}.')
            return filepath
    return None

def run_tracts(driver_filename):
    driver_path = locate_file_path(driver_filename)
    if driver_path is None:
        raise OSError('Driver yaml file could not be found.')
    driver_spec = None
    with driver_path.open() as file, ruamel.yaml.YAML(typ="safe") as yaml:
        driver_spec = yaml.load(file)
    assert isinstance(driver_spec, dict), ".yaml file was invalid."
    directory = driver_spec['samples']['directory']
    names = driver_spec['samples']['individual_names']
    chromosome_list = parse_chromosomes(driver_spec['samples']['chromosomes'])
    individual_filenames = parse_individual_filenames(driver_spec['samples']['individual_names'], driver_spec['samples']['filename_format'], labels = driver_spec['samples']['labels'], directory=driver_spec['samples']['directory'], absolute_driver_yaml_path=driver_path)

    # load the population
    pop = tracts.population(filenames_by_individual=individual_filenames, selectchrom=chromosome_list)
    (bins, data) = pop.get_global_tractlengths(npts=50)

    logger.info(f'Bins: {bins}')
    Ls = pop.Ls
    nind = pop.nind

    time_scaling_factor = driver_spec['time_scaling_factor'] if 'time_scaling_factor' in driver_spec else 1

    if 'model_filename' not in driver_spec:
        raise ValueError('You must specify the file path to your model under "model_filename".')
    model_path = locate_file_path(driver_spec['model_filename'], driver_path)
    if model_path is None:
        raise OSError('Model yaml file could not be found.')
    model = tracts.ParametrizedDemography.load_from_YAML(model_path.resolve())
    func = lambda params: model.get_migration_matrix(scale_select_indices(params, model.is_time_param(), time_scaling_factor))
    bound = lambda params: model.out_of_bounds(scale_select_indices(params, model.is_time_param(), time_scaling_factor))

    population_labels = model.population_indices.keys()

    if 'fix_ancestry_proportions' in driver_spec and driver_spec['fix_ancestry_proportions'] == True:
        ancestry_proportions = calculate_ancestry_proportions(pop, population_labels)
        if 'params_to_fix' not in driver_spec:
            raise ValueError('You must specify which parameters to calculate from known ancestry proportions under "params_to_fix"')
        model.fix_ancestry_proportions(driver_spec['params_to_fix'], ancestry_proportions)

    if type(driver_spec['start_params']) is not list or len(driver_spec['start_params']) < 1:
        raise ValueError('You must specify initial parameters under "start_params".')    
    params_found, likelihoods = run_model_multi_params(func, bound, pop, population_labels, parse_start_params(driver_spec['start_params'], model.is_time_param(), 1/time_scaling_factor))
    print("likelihoods found: ", likelihoods)
    optimal_params = min(zip(params_found,likelihoods), key=lambda x: x[1])[0]
    optimal_params = scale_select_indices(optimal_params, model.is_time_param(), time_scaling_factor)
    if 'output_filename_format' in driver_spec:
        output_simulation_data(pop, optimal_params, model, driver_spec)
    return

def parse_chromosomes(chromosome_spec):
    if isinstance(chromosome_spec, int):
        return str(chromosome_spec)
    if isinstance(chromosome_spec, list):
        return [parse_chromosomes(subspec) for subspec in chromosome_spec]
    try:
        chromosome_spec = chromosome_spec.split('-')
        return
    except Exception as e:
        raise ValueError('Chromosomes should be an int, range (ie: 1-22), or list.') from e
    
def parse_individual_filenames(individual_names, filename_string, labels = ['A', 'B'], directory='', absolute_driver_yaml_path=None):
    def _find_individual_file(individual_name, label_val):
        filepath = locate_file_path(directory+filename_string.format(name = individual_name, label = label_val), absolute_driver_yaml_path)
        if filepath is None:
            raise IndexError(f'File for individiual {individual_name} ("{directory+filename_string.format(name = individual_name, label = label_val)}") could not be found.')
        return str(filepath)
    individual_filenames = {individual_name: [_find_individual_file(individual_name, label_val) for label_val in labels] for individual_name in individual_names}
    return individual_filenames

def parse_start_params(start_params_list, indices_to_scale, time_scaling_factor=1):
    for startparams in start_params_list:
        try: 
            if type(startparams) == dict:
                repetitions = startparams['repetitions'] if 'repetitions' in startparams else 1
                if repetitions > 1 and 'perturbation' not in startparams:
                    logging.warning('You are running the optimizer multiple times without perturbing the start parameters.')
                for rep in range(repetitions):
                    randomized_params = randomize(numpy.array(startparams['values']), startparams['perturbation'][0], startparams['perturbation'][1])
                    yield scale_select_indices(randomized_params, indices_to_scale, time_scaling_factor)
            else:
                yield scale_select_indices(numpy.array(startparams), indices_to_scale, time_scaling_factor)
        except Exception as e:
            raise ValueError(f'Could not parse start_params ({startparams}).') from e
        
def scale_select_indices(arr, indices_to_scale, scaling_factor = 1):
    assert len(indices_to_scale) == len(arr)
    return (numpy.multiply(indices_to_scale, scaling_factor - 1) + 1) * arr

def randomize(arr, a, b):
    # takes an array and multiplies every element by a factor between a and b,
    # uniformly.
    return ((b-a) * numpy.random.random(arr.shape) + a)*arr

def run_model_multi_params(model_func, bound_func, population, population_labels, startparams_generator, cutoff = 2):
    optimal_params = []
    likelihoods = []
    for start_params in startparams_generator:
        logger.info(f'Start params: {start_params}')
        params_found, likelihood_found = run_model(model_func, bound_func, population, population_labels, start_params)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
    return optimal_params, likelihoods

def run_model(model_func, bound_func, population, population_labels, startparams):
    cutoff = 2
    Ls = population.Ls
    nind = population.nind
    (bins, data) = population.get_global_tractlengths(npts=50)
    data = [data[poplab] for poplab in population_labels]
    xopt = tracts.optimize_cob(startparams, bins, Ls, data, nind, model_func, outofbounds_fun=bound_func, cutoff=cutoff, epsilon=1e-2)
    optmod = tracts.demographic_model(model_func(xopt))
    optlik = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
    return xopt, optlik

def output_simulation_data(sample_population, optimal_params, model: tracts.ParametrizedDemography, driver_spec):

    if 'output_directory' in driver_spec:
        output_dir = driver_spec['output_directory']
        try:
            os.mkdir(output_dir)
        except OSError as error:
            logging.warn(error)
    else:
        output_dir = ''

    output_filename_format = driver_spec['output_filename_format']
    (bins, data) = sample_population.get_global_tractlengths(npts=50)
    Ls = sample_population.Ls
    nind = sample_population.nind

    with open(output_dir + output_filename_format.format(label='tract_length_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, bins)))

    with open(output_dir + output_filename_format.format(label='sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            fdat.write("\t".join(map(str, data[population])) + "\n")

    optimal_model = tracts.demographic_model(model.get_migration_matrix(optimal_params))
    
    with open(output_dir + output_filename_format.format(label='migration_matrix'), 'w') as fmig2:
        for line in optimal_model.mig:
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
