import tracts
import numpy
import ruamel.yaml

def run_tracts(filename):
    driver_spec = None
    with open(filename) as file, ruamel.yaml.YAML(typ="safe") as yaml:
        driver_spec = yaml.load(file)
    assert isinstance(driver_spec, dict), ".yaml file was invalid."
    directory = driver_spec['samples']['directory']
    names = driver_spec['samples']['individual_names']
    chromosome_list = parse_chromosomes(driver_spec['samples']['chromosomes'])
    if isinstance(driver_spec['samples']['chromosomes'], int):
        chromosomes = range(1,driver_spec['samples']['chromosomes'])
    else:
        chromosomes = driver_spec['samples']['chromosomes']
    chromosomes = [f'{i}' for i in chromosomes]



    cutoff = driver_data['parameters']['bin_cutoff']

    # only trio individuals
    names = [
        "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
        "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
        "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
        "NA20357", "NA20356"
    ]

    chroms = [f'{i}' for i in range(1, 23)]

    # load the population
    pop = tracts.population(names=names, fname=(directory, "", ".bed"), selectchrom=chromosomes)
    (bins, data) = pop.get_global_tractlengths(npts=50)

    # choose order of populations and sort data accordingly
    labels = ['EUR', 'AFR']
    data = [data[poplab] for poplab in labels]

    # (initial European proportion, and time). Times are
    # measured in units of hundred generations (i.e.,
    # multiply the number by 100 to get the time in
    # generations). The reason is that some python
    # optimizers do a poor job when the parameters (time
    # and ancestry proportions) are of different
    # magnitudes.
    startparams = numpy.array([0.173632,  0.0683211])
    # you can also look at the "_mig" output file for a
    # generation-by-generation breakdown of the migration rates.

    Ls = pop.Ls
    nind = pop.nind
    if driver_spec['model_filename'] is None:
        raise ValueError('You must specify the file path to your model under "model_filepath".')
    model = tracts.ParametrizedDemography.load_from_YAML(driver_spec['model_filepath'])
    func = model.get_migration_matrix
    bound = model.out_of_bounds
    start_params_list = driver_spec['start_params']
    optimal_params, likelihoods = run_model_multi_params(func, bound, pop, start_params_list)
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

def run_model_multi_params(model_func, bound_func, population, startparams_list, cutoff = 2):
    optimal_params = []
    likelihoods = []
    Ls = population.Ls
    nind = population.nind
    (bins, data) = population.get_global_tractlengths(npts=50)
    for start_params in startparams_list:
        print(start_params)
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

def randomize(arr, scale=2):
    # takes an array and multiplies every element by a factor between 0 and 2,
    # uniformly. caps at 1.
    return list(map(lambda i: min(i, 1), scale * numpy.random.random(arr.shape) * arr))