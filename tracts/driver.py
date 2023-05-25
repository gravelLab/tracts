import tracts
import numpy
import ruamel.yaml
import re
import logging

def run_tracts(filename):
    driver_spec = None
    with open(filename) as file, ruamel.yaml.YAML(typ="safe") as yaml:
        driver_spec = yaml.load(file)
    assert isinstance(driver_spec, dict), ".yaml file was invalid."
    directory = driver_spec['samples']['directory']
    names = driver_spec['samples']['individual_names']
    chromosome_list = parse_chromosomes(driver_spec['samples']['chromosomes'])
    individual_filenames = parse_individual_filenames(driver_spec['samples']['individual_names'], driver_spec['samples']['filename_format'], labels = driver_spec['samples']['labels'], directory=driver_spec['samples']['directory'])
    #[print(individual_filenames)]
    # load the population
    pop = tracts.population(filenames_by_individual=individual_filenames, selectchrom=chromosome_list)
    (bins, data) = pop.get_global_tractlengths(npts=50)

    # choose order of populations and sort data accordingly

    print(bins)
    Ls = pop.Ls
    nind = pop.nind

    time_scaling_factor = driver_spec['time_scaling_factor'] if 'time_scaling_factor' in driver_spec else 1

    if 'model_filename' not in driver_spec:
        raise ValueError('You must specify the file path to your model under "model_filename".')
    model = tracts.ParametrizedDemography.load_from_YAML(driver_spec['model_filename'])
    func = lambda params: model.get_migration_matrix(scale_select_indices(params, model.is_time_param(), time_scaling_factor))
    bound = lambda params: model.out_of_bounds(scale_select_indices(params, model.is_time_param(), time_scaling_factor))

    population_labels = model.population_indices.keys()
    data = [data[poplab] for poplab in population_labels]

    if type(driver_spec['start_params']) is not list or len(driver_spec['start_params']) < 1:
        raise ValueError('You must specify initial parameters under "start_params".')    

    # you can also look at the "_mig" output file for a
    # generation-by-generation breakdown of the migration rates.
    optimal_params, likelihoods = run_model_multi_params(func, bound, pop, parse_start_params(driver_spec['start_params'], model.is_time_param(), 1/time_scaling_factor))
    print("likelihoods found: ", likelihoods)
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
    
def parse_individual_filenames(individual_names, filename_string, labels = ['A', 'B'], directory=''):
    individual_filenames = {}
    for individual_name in individual_names:
        #print([filename_string.format(name = individual_name, label = label_val) for label_val in labels])
        individual_filenames[individual_name] = [directory+filename_string.format(name = individual_name, label = label_val) for label_val in labels]
    return individual_filenames

def parse_start_params(start_params_list, indices_to_scale, time_scaling_factor=1):
    for startparams in start_params_list:
        try: 
            if type(startparams) == dict:
                if startparams['repetitions'] > 1 and 'perturbation' not in startparams:
                    logging.warning('You are running the optimizer multiple times without perturbing the start parameters.')
                for rep in range(startparams['repetitions']):
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

def run_model_multi_params(model_func, bound_func, population, startparams_generator, cutoff = 2):
    optimal_params = []
    likelihoods = []
    Ls = population.Ls
    nind = population.nind
    (bins, data) = population.get_global_tractlengths(npts=50)
    for start_params in startparams_generator:
        print(f'Start params:{start_params}')
        params_found, likelihood_found = run_model(model_func, bound_func, population, start_params)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
    return optimal_params, likelihoods

def run_model(model_func, bound_func, population, startparams):
    cutoff = 2
    Ls = population.Ls
    nind = population.nind
    (bins, data) = population.get_global_tractlengths(npts=50)
    labels = ['EUR', 'AFR']
    data = [data[poplab] for poplab in labels]
    xopt = tracts.optimize_cob(startparams, bins, Ls, data, nind, model_func, outofbounds_fun=bound_func, cutoff=cutoff, epsilon=1e-2)
    optmod = tracts.demographic_model(model_func(xopt))
    optlik = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
    return xopt, optlik