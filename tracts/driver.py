import logging
import numbers
import os
import sys
from pathlib import Path
from typing import Callable
import numpy as np
import ruamel.yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
#warnings.simplefilter('always')
from scipy.special import logit, expit
from tracts.population import Population
from tracts.core import optimize_cob, optimize_cob_sex_biased_fixed_values
import tracts.hybrid_pedigree as HP
from tracts.phase_type_distribution import PhTMonoecious, PhTDioecious
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.demography.parameter import Parameter ,ParamType
from tracts.demography import DemographicModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# File handler (logs EVERYTHING >= INFO)
file_handler = logging.FileHandler("tracts.log") #TODO enable file naming
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Stream handler (logs only WARNING+ to stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)




filepath_error_additional_message = ('\nPlease ensure that the file path is either absolute,'
                                     ' or relative to the working directory, script directory,'
                                     ' or the directory of the driver yaml.')

def locate_file_path(filename: str, script_dir: str | Path | None, absolute_driver_yaml_path: str | Path = None):
    # Define search methods and paths

    search_methods = [
        (Path(filename), "working directory"),
        (Path(script_dir) / filename if script_dir else Path(""), "script directory"),
        (
            (absolute_driver_yaml_path.parent / filename if isinstance(absolute_driver_yaml_path, Path) else Path("")),
            "driver yaml"
        )
    ]

    for filepath, method_name in search_methods:
        logger.info(f'{method_name}: {filepath}')
        if filepath.is_file():
            logger.info(f'Found {filename} using {method_name}.')
            return filepath
    for pathname in sys.path:
        if (Path(pathname) / filename).is_file():
            logger.info(f'Found {filename} from {pathname}.')
            return Path(pathname) / filename
    return None

def run_tracts(driver_filename, script_dir=None):

    driver_path = locate_file_path(filename=driver_filename, script_dir=script_dir)
    driver_spec = load_driver_file(driver_path)

    
    print('------------------------------------------------------------------------------------------------\n')
    print('Running tracts 2.0 with driver file:', driver_filename,'\n')
    print('Reading data, demographic model and driver specifications...\n')
    print('------------------------------------------------------------------------------------------------\n')

    # Set autosomal and allosomal models for admixture
    if hasattr(driver_spec, 'ad_model_autosomes') :
        ad_model_autosomes = driver_spec.ad_model_autosomes
        if not ad_model_autosomes in ['DC','DF','M','H-DC','H-DF']:
            print('The model for autosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), M (for Monoecious), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_autosomes = DC by default.')
            ad_model_autosomes = 'DC'
    else:
        print('Model for autosomal admixture not specified. Setting DC by default.')
        ad_model_autosomes = 'DC'
    
    if hasattr(driver_spec, 'ad_model_allosomes'):
        ad_model_allosomes = driver_spec.ad_model_allosomes
        if not ad_model_allosomes in ['DC','DF','H-DC','H-DF']:
            print('The model for allosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_allosomes = DC by default.')
            ad_model_allosomes = 'DC'
    else:
        print('Model for allosomal admixture not specified. Setting DC by default.')
        ad_model_allosomes = 'DC'


    exclude_tracts_below_cM = driver_spec.exclude_tracts_below_cm
    print(f'excluding_tracts_below Defaulting to {exclude_tracts_below_cM} cM.')
    

    npts = driver_spec.npts

    # Currently assumes allosomes is a single label. May change in the future
    allosome_labels = driver_spec.samples.allosomes
    allosome_label = allosome_labels[0] if len(allosome_labels) > 0 else None

    # Load the population
    
    pop = load_population(driver_path, driver_spec, script_dir, allosome_labels = allosome_labels) 
    pop.unknown_labels = driver_spec.unknown_labels_for_smoothing
    
    pop.smooth_unknowns(allosome_labels = allosome_labels)
    _bins, _data = pop.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM) # we do this here just to get the population labels and 
                                                                                                    # validate that these correspond to to model population labels
    
    time_scaling_factor = driver_spec.time_scaling_factor

    model = load_model_from_driver(driver_spec=driver_spec, script_dir=script_dir, driver_path=driver_path, allosome_label=allosome_label)

    ancestor_labels = model.population_indices.keys()

    data_labels =  _data.keys()
       
    for label in data_labels:
        if label not in ancestor_labels and label not in pop.unknown_labels:
            raise ValueError(f"Population label '{label}' found in data but not in model or labels to be smoothed over. data labels: {data_labels}, model labels: {ancestor_labels}, " \
            "unknown labels: {pop.unknown_labels}")
    
    
    

    ancestry_proportions = pop.calculate_ancestry_proportions(ancestor_labels)
    print("\nData autosome proportions:", ancestry_proportions )
    
    if len(allosome_labels)>=1:
        allosome_proportions = pop.calculate_allosome_proportions(ancestor_labels, allosome_label)
        print("Data allosome proportions:", allosome_proportions )




    if len(driver_spec.fix_parameters_from_ancestry_proportions)>0:
        
        if allosome_label:
            model.parameter_handler.set_up_fixed_parameters(model,
                driver_spec.fix_parameters_from_ancestry_proportions,
                {
                    f'{model.parametrized_populations[0]}_autosomal':ancestry_proportions,
                    f'{model.parametrized_populations[0]}_{allosome_label}': allosome_proportions
                }
            )
        else:
            model.set_up_fixed_parameters(params_to_fix_by_ancestry=driver_spec.fix_parameters_from_ancestry_proportions, 
                                                                           proportions= {model.parametrized_populations[0]:ancestry_proportions})
    else:
        model.set_up_fixed_parameters([],{})
    func = get_time_scaled_model_func(model, time_scaling_factor) # time parameters need to be rescaled for some optimizers

    bound = get_time_scaled_model_bounds(model, time_scaling_factor)


    #The following should be moved to core.py or base_demography.py (in the FixedParamHandler class)
    time_to_physical_function = lambda x:np.exp(x) # This converts from optimizer units to physical units
    rate_to_physical_function = lambda x: expit(x)
    def to_minus1_plus1(x):
        return 2 * expit(x) - 1
    
    sex_bias_to_physical_function = to_minus1_plus1
    
    to_physical_params_functions = {ParamType.TIME: time_to_physical_function, 
                                    ParamType.RATE: rate_to_physical_function, 
                                    ParamType.SEX_BIAS: sex_bias_to_physical_function} 
    

    time_to_optimizer_function = lambda x: np.log(x)
    rate_to_optimizer_function = lambda x: logit(x)
    
    def from_minus1_plus1(y):

        with np.errstate(divide='ignore', invalid='ignore'):
            log_result = np.log1p(y) - np.log1p(-y)
        log_result = np.where(np.isfinite(log_result), log_result, -1e32)
        return log_result

    sex_bias_to_optimizer_function = from_minus1_plus1

    to_optimizer_params_functions  = {ParamType.TIME: time_to_optimizer_function, 
                                    ParamType.RATE: rate_to_optimizer_function, 
                                    ParamType.SEX_BIAS: sex_bias_to_optimizer_function}
    model.parameter_handler.to_physical_params_functions = to_physical_params_functions
    model.parameter_handler.to_optimizer_params_functions = to_optimizer_params_functions


    
    max_iter = driver_spec.maximum_iterations
    
    pop_dict = model.population_indices.items()
    
    print("Model parameters :",[param_name for param_name in model.model_base_params.keys()])

    physical_start_params = parse_start_params(driver_spec.start_params, driver_spec.repetitions, 
                                      driver_spec.seed, model)
    optimizer_start_params = [model.parameter_handler.convert_to_optimizer_params(params) for params in physical_start_params]   

    if len(physical_start_params) > 1:

        print("\nMultiple starting parameters were generated. These will be converted to optimizer units and used for multiple optimization runs.")

    else:
        print("\nA single set of starting parameters was generated. It will be converted to optimizer units and used for optimization.")

    header = f"{'Run':>3} | {'Physical start parameters':<45} | {'Optimizer start parameters'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i, (phys, opt) in enumerate(zip(physical_start_params, optimizer_start_params)):
        assert np.isclose(phys, model.parameter_handler.convert_to_physical_params(opt)).all()
        if bound(opt)<0:
            print("Warning, starting parameters are out of bounds.")
        phys_str = ", ".join(f"{x:.4g}" for x in phys)
        opt_str = ", ".join(f"{x:.4g}" for x in opt)
        print(f"{1+i:>3} | [{phys_str:<43}] | [{opt_str}]")
    print("-" * len(header))

    # Get keys from first run
    first_props = model.proportions_from_matrices(func(optimizer_start_params[0]))
    keys = list(first_props.keys())

    print("\nStarting ancestry proportions for the starting parameters (in physical units)")
    header = f"{'Run':>3} | " + " | ".join(f"{k:<35}" for k in keys)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
   
    for i, opt in enumerate(optimizer_start_params):
        try: 
            props = model.proportions_from_matrices(func(opt))
        except ValueError:
            print("Could not compute starting ancestry proportions - likely due to out of bounds starting parameters.")


        row_values = []
        for k in keys:
            arr = props[k]
            arr_str = ", ".join(f"{x:.4g}" for x in arr)
            row_values.append(f"[{arr_str:<33}]")
    
        print(f"{1+i:>3} | " + " | ".join(row_values))
    print("-" * len(header))
          
    params_found, likelihoods = run_model_multi_init(func, bound, pop, ancestor_labels,
                                                        optimizer_start_params,
                                                        population_dict=pop_dict,
                                                        parameter_handler=model.parameter_handler,
                                                        max_iter=max_iter,
                                                        exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                        modelling_method=PhTDioecious if allosome_label else PhTMonoecious,
                                                        ad_model_autosomes = ad_model_autosomes, ad_model_allosomes=ad_model_allosomes, npts=npts)

    formatted_likelihoods = [float(x) for x in likelihoods]
    
    if len(formatted_likelihoods) > 1:

        print("\n---------------------------------------------------------------------------")
        print("Results from multiple optimization runs with different starting parameters:")
        header = f"{'Run':>3} | {'LogLik':>12} | Found parameters (in optimizer units)"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i, (params, ll) in enumerate(zip(params_found, formatted_likelihoods)):
            params_str = ", ".join(f"{p:.4g}" for p in params)
            print(f"{1+i:>3} | {float(ll):>12.6g} | [{params_str}]")
        print("-" * len(header))
    
    optimal_params, optimal_likelihood = max(zip(params_found, formatted_likelihoods), key=lambda x: x[1])
    optimal_params = model.parameter_handler.convert_to_physical_params(optimal_params)

    print("\nFinal parameters in physical units and corresponding likelihood:")
    param_names = list(model.model_base_params.keys())
    header = f"{'LogLik':>12} | " + " ".join(f"{name:>12}" for name in param_names)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    values_str = " ".join(f"{x:>12.4g}" for x in optimal_params)
    print(f"{float(optimal_likelihood):>12.6g} | {values_str}")
    print("-" * len(header))

    bound = model.get_violation_score(optimal_params, verbose = True)

    if hasattr(driver_spec, "output_filename_format"):
        if allosome_label:
            output_simulation_data_sex_biased(pop, optimal_params, model, driver_spec, ad_model_autosomes=ad_model_autosomes, ad_model_allosomes=ad_model_allosomes)
        else:
            output_simulation_data(pop, optimal_params, model, driver_spec)


import ruamel.yaml as yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from pydantic_core import PydanticUndefined

# ---------- Models ----------

class SamplesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str
    individual_names: List[str]
    male_names: List[str] | str = "auto" 
    filename_format: str
    labels: List[str] = Field(default_factory=lambda: ["A", "B"])
    chromosomes: str
    allosomes: List[str]=[]


class StartParamsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")



class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unknown_labels_for_smoothing : List[str] = []
    samples: SamplesConfig
    model_filename: str
    start_params: StartParamsConfig
    repetitions: int =1 
    seed: int
    maximum_iterations: int|None=None 
    npts: int = 50
    exclude_tracts_below_cm: float = 1
    time_scaling_factor: float = 1
    fix_parameters_from_ancestry_proportions: List[str] = []
    output_directory: str = ""
    output_filename_format: str
    ad_model_autosomes: str = "M"
    ad_model_allosomes: str = "DC"


# ---------- Loader ----------

def load_driver_file(driver_path: str) -> InferenceConfig:
    if driver_path is None:
        raise OSError(f'Driver yaml file could not be found. {filepath_error_additional_message}')
    
    yaml_loader = yaml.YAML(typ="safe")

    with open(driver_path, "r") as f:
        driver_spec = yaml_loader.load(f)
    
    missing = [] # Check for required missing parameters in the driver file
    for name, field in InferenceConfig.model_fields.items():
        # Field is required if it has no default and no default factory
        is_required = field.default is PydanticUndefined and field.default_factory is None
        # Only add to missing if it's required and not in driver_spec
        if is_required and name not in driver_spec:
            missing.append(name)

    if missing:
        raise ValueError(f"Missing required driver parameters: {', '.join(missing)}")

    return InferenceConfig.model_validate(driver_spec)



#def load_driver_file(driver_path):
#    if driver_path is None:
#        raise OSError(f'Driver yaml file could not be found. {filepath_error_additional_message}')
#    with driver_path.open() as file, ruamel.yaml.YAML(typ="safe") as yaml:
#        driver_spec = yaml.load(file)
#    if not isinstance(driver_spec, dict):
#        raise ValueError('Driver yaml file was invalid.')
#    return driver_spec

def load_population(driver_path, driver_spec, script_dir=None, allosome_labels=None):
    individual_filenames = parse_individual_filenames(driver_spec.samples.individual_names,
                                                      driver_spec.samples.filename_format,
                                                      labels=driver_spec.samples.labels,
                                                      directory=driver_spec.samples.directory,
                                                      script_dir=script_dir,
                                                      absolute_driver_yaml_path=driver_path)
    
    
    male_list = driver_spec.samples.male_names
 
    chromosome_list = parse_chromosomes(driver_spec.samples.chromosomes)
    logger.info(f'Chromosomes: {chromosome_list}')
    pop = Population(filenames_by_individual=individual_filenames, selectchrom=chromosome_list, allosomes=allosome_labels if allosome_labels else [], male_list = male_list)
    if len(allosome_labels)>=1 and allosome_labels is not None:
        assert(allosome_labels[0] == 'X'), "Currently only X allosome is supported for male determination. Should be first allosome. "
    
    if len(allosome_labels) >0:
        pop.set_males(male_list = male_list, allosome_label = allosome_labels[0]) 
    return pop


def load_model_from_driver(driver_spec, script_dir, driver_path, allosome_label=None):
    if not hasattr( driver_spec, 'model_filename') :
        raise ValueError('You must specify the file path to your model under "model_filename".')
    model_path = locate_file_path(filename=driver_spec.model_filename,
                                  script_dir=script_dir,
                                  absolute_driver_yaml_path=driver_path)
    if model_path is None:
        raise FileNotFoundError(f'Model yaml file {driver_spec.model_filename} could not be found. {filepath_error_additional_message}')
    if allosome_label:
        model = ParametrizedDemographySexBiased.load_from_YAML(str(model_path.resolve()))
        model.allosome_label=allosome_label
    else:    
        model = ParametrizedDemography.load_from_YAML(str(model_path.resolve()))
    return model



def parse_chromosomes(chromosome_spec: list | str | int, chromosomes: None | list=None):
    if chromosomes is None:
        chromosomes = []
    if isinstance(chromosome_spec, int):
        chromosomes.append(chromosome_spec)
        return chromosomes
    if isinstance(chromosome_spec, list):
        [parse_chromosomes(subspec, chromosomes) for subspec in chromosome_spec]
        return chromosomes
    try:
        chromosome_spec = chromosome_spec.split('-')
        chromosomes.extend(range(int(chromosome_spec[0]), int(chromosome_spec[1]) + 1))
        return chromosomes
    except Exception as e:
        raise ValueError('Chromosomes should be an int, range (ie: 1-22), or list.') from e


def parse_individual_filenames(individual_names, filename_string, script_dir: str, labels=['A', 'B'], directory='',
                               absolute_driver_yaml_path=None):

    def _find_individual_file(individual_name, label_val):
        filepath = locate_file_path(filename=directory + filename_string.format(name=individual_name, label=label_val),
                                    script_dir=script_dir,
                                    absolute_driver_yaml_path=absolute_driver_yaml_path)
        if filepath is None:
            raise FileNotFoundError(
                f'File for individual {individual_name} '
                f'("{directory + filename_string.format(name=individual_name, label=label_val)}")'
                f' could not be found. {filepath_error_additional_message}')
        return str(filepath)

    individual_filenames = {individual_name: [_find_individual_file(individual_name, label_val)
                                              for label_val in labels]
                            for individual_name in individual_names}
    return individual_filenames


def parse_start_params(start_param_bounds, repetitions=1, seed=None, model: ParametrizedDemography = None,
                       time_scaling_factor=1):
    """outputs a 1D array of starting parameters for optimization. Returns all base_model_parameters in physical units""" 
    
    num_params = len(model.model_base_params)
    rng = np.random.default_rng(seed=seed)
    start_params = rng.random((repetitions, num_params))
    for param_name, param_info in model.model_base_params.items():
        if param_name in model.params_fixed_by_ancestry:
            start_params[:, param_info.index] = param_info.bounds[0] #this will be replaced, set to arbitrary feasible value
            
            continue

        try: 
            getattr(start_param_bounds, param_name)
        except:
            raise KeyError(f"Initial values were not specified for parameter '{param_name}'.")

        if isinstance(getattr(start_param_bounds, param_name), numbers.Number):
            start_params[:, param_info.index] = getattr(start_param_bounds, param_name)
        else:
            try:
                bounds = [float(bound) for bound in getattr(start_param_bounds, param_name).split(':')] 
                # Intervals are specified as "min:max" to avoid confusion with negative values.

                assert len(bounds) == 2
                start_params[:, param_info.index] *= bounds[1] - bounds[0]
                start_params[:, param_info.index] += bounds[0]
            except Exception as e:
                raise ValueError("Initial values must be specified as min:max or a single value.") from e
        
    
    
    logger.info(f' Start Params: \n {start_params}')
    return start_params


def scale_select_indices(arr, indices_to_scale, scaling_factor=1):
    if len(indices_to_scale) != len(arr):
        raise ValueError(
            f'Length of array ({len(arr)}) was not equal to length of indices_to_scale ({len(indices_to_scale)}).')
    return (np.multiply(indices_to_scale, scaling_factor - 1) + 1) * arr


def get_time_scaled_model_func(model: ParametrizedDemography, time_scaling_factor: float) -> Callable[[np.ndarray], dict[str, np.ndarray]]:
    return lambda params: model.get_migration_matrices(
        model.parameter_handler.convert_to_physical_params(params))


def get_time_scaled_model_bounds(model, time_scaling_factor, verbose = False):
    return lambda params: model.get_violation_score(model.parameter_handler.convert_to_physical_params(params), verbose = verbose)


def randomize(arr, a, b):
    # takes an array and multiplies every element by a factor between a and b,
    # uniformly.
    return ((b - a) * np.random.random(arr.shape) + a) * arr


def run_model_multi_init(model_func: Callable, bound_func: Callable, population: Population, population_labels: list[str], 
                          start_params_list: list[np.ndarray], population_dict : dict, parameter_handler = None , 
                          max_iter: int=None, exclude_tracts_below_cM: int = 0, 
                          modelling_method: type = PhTMonoecious, ad_model_autosomes = 'DC', 
                          ad_model_allosomes = 'DC', npts: int = 50) -> tuple[list[np.ndarray], list[float]]:
    """
    Runs the model multiple times with different initial parameters.

    Parameters
    ----------
    
        model_func: Callable
    	    A function that takes parameters and returns migration matrices.
        bound_func: Callable
    	    A function that calculates the violation score for the parameters. 	
        population: :class:`tracts.population.Population`
    	    The population object containing individual data.
        population_labels: list[str]
    	    A list of labels corresponding to the populations.	
        start_params_list: list[np.ndarray]
    	    A list of initial parameter arrays to start the optimization.	
        exclude_tracts_below_cM: int, optional
    	    Minimum tract length in centimorgans to exclude from analysis. Default is 0.
        modelling_method: type, optional
    	    The method used for modeling. Default is PhTMonoecious.
        npts: int, optional
            Number of bins for the tract length histogram. Default is 50.

    Returns
    ----------
    
    tuple[list[np.ndarray], list[float]]
    	A tuple containing two lists: (i) a list of optimal parameters found for each set of starting parameters and (ii) a list of likelihoods corresponding to the optimal parameters.
    """
    optimal_params = []
    likelihoods = []
    for start_params in start_params_list:
        print('\nOptimization run #', len(optimal_params)+1,"\n--------------------")
        logger.info(f'Start params: {start_params}')
        params_found, likelihood_found = run_model(model_func, bound_func, population, population_labels, start_params,
                                                   population_dict,
                                                   parameter_handler=parameter_handler,
                                                   max_iter=max_iter,
                                                   exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                   modelling_method=modelling_method, ad_model_autosomes=ad_model_autosomes,ad_model_allosomes=ad_model_allosomes, npts=npts)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
    return optimal_params, likelihoods


def run_model(model_func, bound_func, population: Population, population_labels, startparams, population_dict, parameter_handler=None, max_iter=None, exclude_tracts_below_cM=0,
              modelling_method=PhTMonoecious, ad_model_autosomes='DC', ad_model_allosomes='DC', npts=0):
    if modelling_method == PhTDioecious:
        return run_model_sex_biased(model_func,bound_func, population, population_labels, startparams, population_dict, parameter_handler, max_iter, exclude_tracts_below_cM, ad_model_autosomes=ad_model_autosomes, ad_model_allosomes=ad_model_allosomes, npts=npts)
    Ls = population.Ls
    nind = population.nind
    bins, data = population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
    data = [data[poplab] for poplab in population_labels]
    model_func_sample_pop = lambda params:list(model_func(params).values())[0]
    xopt = optimize_cob(startparams, bins, Ls, data, nind, model_func_sample_pop, outofbounds_fun=bound_func, epsilon=1e-2,
                        modelling_method=modelling_method)
    optmod = modelling_method(model_func_sample_pop(xopt))
    optlik = optmod.loglik(bins, Ls, data, nind)
    return xopt, optlik

def run_model_sex_biased(model_func, bound_func, population: Population, population_labels, startparams, population_dict, parameter_handler = None, max_iter=None, exclude_tracts_below_cM=0, ad_model_autosomes='DC',ad_model_allosomes='DC',npts=0):
  
    #optimal_params, optimal_likelihood = optimize_cob_sex_biased(startparams, population, model_func, bound_func, p_dict = population_dict, exclude_tracts_below_cM=exclude_tracts_below_cM, maxiter=max_iter, epsilon=1e-2,verbose=1, ad_model_autosomes=ad_model_autosomes, ad_model_allosomes=ad_model_allosomes, npts=npts)
    optimal_params, optimal_likelihood = optimize_cob_sex_biased_fixed_values(startparams, population, model_func, 
                                                                              parameter_handler= parameter_handler, 
                                                                              outofbounds_fun = bound_func, 
                                                                              p_dict = population_dict, 
                                                                              exclude_tracts_below_cM=exclude_tracts_below_cM, 
                                                                              maxiter=max_iter, epsilon=1e-2,verbose=1, ad_model_autosomes=ad_model_autosomes, ad_model_allosomes=ad_model_allosomes, npts=npts)
    
    
    
    
    return optimal_params, optimal_likelihood

def output_simulation_data(sample_population, optimal_params, model: ParametrizedDemography, driver_spec):


    output_dir = driver_spec.output_directory
    if not os.path.exists(output_dir) and len(output_dir)>0:
        os.mkdir(output_dir)


    output_filename_format = driver_spec.output_filename_format
    exclude_tracts_below_cM = driver_spec.exclude_tracts_below_cm

    npts = driver_spec.npts
    (bins, data) = sample_population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)


    Ls = sample_population.Ls
    nind = sample_population.nind

    with open(output_dir + output_filename_format.format(label='tract_length_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, bins)))

    with open(output_dir + output_filename_format.format(label='sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            fdat.write("\t".join(map(str, data[population])) + "\n")

    optimal_model = PhTMonoecious(list(model.get_migration_matrices(optimal_params).values())[0])

    with open(output_dir + output_filename_format.format(label='migration_matrix'), 'w') as fmig2:
        for line in optimal_model.migration_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")

    with open(output_dir + output_filename_format.format(label='predicted_tract_distribution'), 'w') as fpred2:
        for popnum in range(len(data)):
            fpred2.write("\t".join(map(
                str,
                nind * np.array(optimal_model.tract_length_histogram_multi_windowed(popnum, bins, Ls))))
                         + "\n")

    param_names = list(model.model_base_params.keys())
    params_file_path = output_dir + output_filename_format.format(label="optimal_parameters") + ".txt"

    with open(params_file_path, "w") as f:
        f.write("parameter\tvalue\n")
        for name, value in zip(param_names, optimal_params):
            f.write(f"{name}\t{value}\n")
            

    # Plot results
    
    for pop, pop_num in model.population_indices.items():
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        fig.suptitle(f"Ancestor: {pop}", fontsize=16, fontweight="bold")

        # --- 1st row: autosome ---
        axes.bar(
            bins[:-1],
            data[pop],
            width=np.diff(bins),
            align='edge',
            alpha=0.6,
            color='tab:blue',
            edgecolor='white',
            label="Data"
        )
        axes.plot(
            0.5 * (bins[:-1] + bins[1:]),
            nind * np.array(optimal_model.tract_length_histogram_multi_windowed(pop_num, bins, Ls)),
            color='tab:orange',
            lw=2,
            label="Predicted"
        )
        axes.set_title("Autosome", fontsize=12, fontweight="bold")
        axes.set_xlabel("Tract Length (M)")
        axes.set_ylabel("Count")
        axes.legend(frameon=False)
        axes.grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir + output_filename_format.format(label=f"{pop}_tract_histograms.png"))
        plt.close(fig)
        

def output_simulation_data_sex_biased(sample_population: Population, optimal_params, 
                                      model: ParametrizedDemographySexBiased, driver_spec, ad_model_autosomes='DC', ad_model_allosomes='DC'):
    """
    Creates output graphs to compare data and the theoretical tract length distribution inferred by the model.
    """
    
    if hasattr(driver_spec, 'output_directory'):
        output_dir = driver_spec.output_directory
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = ''

    output_filename_format = driver_spec.output_filename_format
    if hasattr(driver_spec, 'exclude_tracts_below_cM'):
        exclude_tracts_below_cM = driver_spec.exclude_tracts_below_cM
    else:
        exclude_tracts_below_cM = 10

    if hasattr(driver_spec, 'npts'):
        npts = driver_spec.npts
    else:
        npts = 50

    matrices = model.get_migration_matrices(optimal_params)
    
    [male_matrix, female_matrix] = [matrix for matrix in matrices.values()]
    output_filename_format = driver_spec.output_filename_format
    autosome_bins, autosome_data = sample_population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
    Ls = sample_population.Ls
    nind = sample_population.nind

    allosome_bins, allosome_data = sample_population.get_global_allosome_tractlengths('X',npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
    allosome_length = sample_population.allosome_lengths['X']
    female_data = allosome_data[SexType.FEMALE]
    male_data = allosome_data[SexType.MALE]
    
    num_males = sample_population.num_males
    num_females = sample_population.num_females

    if ad_model_autosomes in ['DC','DF']:
        autosome_predicted={pop:PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_autosomes).tract_length_histogram_multi_windowed(pop_num, autosome_bins, Ls) for pop, pop_num in model.population_indices.items()}
    elif ad_model_autosomes == 'M':
        autosome_predicted={pop:PhTMonoecious(0.5*(female_matrix+male_matrix), rho=1).tract_length_histogram_multi_windowed(pop_num, autosome_bins, Ls) for pop, pop_num in model.population_indices.items()}
    elif ad_model_autosomes == 'H-DC':
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DC', rr_f=1, rr_m=1, X_chr=False, X_chr_male=False, N_cores=5, population_number= pop_num, bins=autosome_bins, chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    else:
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DF', rr_f=1, rr_m=1, X_chr=False, X_chr_male=False, N_cores=5, population_number= pop_num, bins=autosome_bins, chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    
    if ad_model_allosomes in ['DC','DF']:
        female_predicted = {pop: PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True).tract_length_histogram_multi_windowed(pop_num, allosome_bins, [allosome_length]) for pop, pop_num in model.population_indices.items()}
        male_predicted = {pop: PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True, X_chromosome_male=True).tract_length_histogram_multi_windowed(pop_num, allosome_bins, [allosome_length]) for pop, pop_num in model.population_indices.items()}
    elif ad_model_allosomes == 'H-DC':
        female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DC', rr_f=1, rr_m=1, X_chr=True, X_chr_male=False, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
        male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DC', rr_f=1, rr_m=1, X_chr=True, X_chr_male=True, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
    else:
        female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DF', rr_f=1, rr_m=1, X_chr=True, X_chr_male=False, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
        male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DF', rr_f=1, rr_m=1, X_chr=True, X_chr_male=True, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
    
    # Save results
    
    with open(output_dir + output_filename_format.format(label='tract_length_autosome_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, autosome_bins)))
    with open(output_dir + output_filename_format.format(label='tract_length_allosome_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, allosome_bins)))

    with open(output_dir + output_filename_format.format(label='autosome_sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            try:
                fdat.write("\t".join(map(str, autosome_data[population])) + "\n")
            except KeyError:
                autosome_data[population] = np.zeros(len(autosome_bins)).tolist()
                print(f'Population {population} not found in autosome data.')
    with open(output_dir + output_filename_format.format(label='female_allosome_sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            try:
                fdat.write("\t".join(map(str, female_data[population])) + "\n")
            except KeyError:
                female_data[population] = np.zeros(len(allosome_bins)).tolist()
                print(f'Population {population} not found in female allosome data.')
    with open(output_dir + output_filename_format.format(label='male_allosome_sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            try:
                fdat.write("\t".join(map(str, male_data[population])) + "\n")
            except KeyError:
                male_data[population] = np.zeros(len(allosome_bins)).tolist()
                print(f'Population {population} not found in male allosome data.')
            
    with open(output_dir + output_filename_format.format(label='female_migration_matrix'), 'w') as fmig2:
        for line in female_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")
    with open(output_dir + output_filename_format.format(label='male_migration_matrix'), 'w') as fmig2:
        for line in male_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")
    
    with open(output_dir + output_filename_format.format(label='autosome_predicted_tract_distribution'), 'w') as fpred2:
        for pop, pop_num in model.population_indices.items():
            fpred2.write("\t".join(map(
                str,
                [nind * num_tracts for num_tracts in autosome_predicted[pop]]))
                         + "\n")
    with open(output_dir + output_filename_format.format(label='female_allosome_predicted_tract_distribution'), 'w') as fpred2:
        for pop, pop_num in model.population_indices.items():
            fpred2.write("\t".join(map(
                str,
                [num_females * num_tracts for num_tracts in female_predicted[pop]]))
                         + "\n")
    with open(output_dir + output_filename_format.format(label='male_allosome_predicted_tract_distribution'), 'w') as fpred2:
        for pop, pop_num in model.population_indices.items():
            fpred2.write("\t".join(map(
                str,
                [num_males * num_tracts for num_tracts in male_predicted[pop]]))
                         + "\n")

    param_names = list(model.model_base_params.keys())
    params_file_path = output_dir + output_filename_format.format(label="optimal_parameters") + ".txt"

    with open(params_file_path, "w") as f:
        f.write("parameter\tvalue\n")
        for name, value in zip(param_names, optimal_params):
            f.write(f"{name}\t{value}\n")

    pop_names = list(model.population_indices.keys())
    n_pops = len(pop_names)

    # Colorblind-friendly palette
    okabe_ito = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]

    if n_pops <= len(okabe_ito):
        colors = okabe_ito[:n_pops]
    else:
        # fallback if there are more populations than Okabe-Ito colors
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n_pops)]

    pop_colors = {pop: colors[i] for i, pop in enumerate(pop_names)}


    def _bin_centers(bins):
        return 0.5 * (bins[:-1] + bins[1:])


    def _plot_panel(
        xbins,
        observed_dict,
        predicted_dict,
        scale_factor,
        title,
        ylabel,
        output_path,
        xlabel="Tract Length (M)"):

        fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)

        x_centers = _bin_centers(xbins)
        x_left = xbins[:-1]

        population_handles = []

        for pop in pop_names:
            color = pop_colors[pop]

            # Observed data as points
            y_obs = np.asarray(observed_dict[pop], dtype=float)
            ax.scatter(
                x_centers,
                y_obs,
                s=30,
                color=color,
                alpha=0.95,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

            # Predicted data as line
            y_pred = scale_factor * np.asarray(predicted_dict[pop], dtype=float)
            ax.step(
                x_left,
                y_pred,
                where="post",
                color=color,
                lw=2.2,
                alpha=0.95,
                zorder=2,
            )

            # One legend entry per population: line + marker together
            population_handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    lw=2.2,
                    marker='o',
                    markersize=6,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    label=pop
                )
            )

        # Main styling
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10)

        # Legend 1: populations by color
        legend_pop = ax.legend(
            handles=population_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
            fontsize=10,
            ncol=min(len(pop_names), 4),
            title="Source population",
            title_fontsize=10,
        )

        # Legend 2: glyph meaning
        glyph_handles = [
            Line2D(
                [0], [0],
                linestyle="None",
                marker='o',
                color='0.35',
                markerfacecolor='0.35',
                markeredgecolor="white",
                markersize=6,
                label="Observed"
            ),
            Line2D(
                [0], [0],
                linestyle='-',
                color='0.35',
                lw=2.2,
                label="Predicted"
            ),
        ]

        legend_glyph = ax.legend(
            handles=glyph_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
            frameon=False,
            fontsize=10,
            ncol=2,
        )

        ax.add_artist(legend_pop)

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    # --- Autosomes ---
    _plot_panel(
        xbins=autosome_bins,
        observed_dict=autosome_data,
        predicted_dict=autosome_predicted,
        scale_factor=nind,
        title="Autosomal tract length distributions",
        ylabel="Count",
        output_path=os.path.join(
            output_dir,
            output_filename_format.format(label="autosomes_all_populations.png")
        ),
    )

    # --- Male allosomes ---
    _plot_panel(
        xbins=allosome_bins,
        observed_dict=male_data,
        predicted_dict=male_predicted,
        scale_factor=num_males,
        title="Male X-chromosome tract length distributions",
        ylabel="Count",
        output_path=os.path.join(
            output_dir,
            output_filename_format.format(label="male_allosomes_all_populations.png")
        ),
    )

    # --- Female allosomes ---
    _plot_panel(
        xbins=allosome_bins,
        observed_dict=female_data,
        predicted_dict=female_predicted,
        scale_factor=num_females,
        title="Female X-chromosome tract length distributions",
        ylabel="Count",
        output_path=os.path.join(
            output_dir,
            output_filename_format.format(label="female_allosomes_all_populations.png")
        ),
    )
    print('Results saved to : ' + output_dir)

   
