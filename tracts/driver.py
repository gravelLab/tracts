import logging
from logging.handlers import MemoryHandler
import numbers
import os
import sys
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import ruamel.yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
#warnings.simplefilter('always')
from scipy.special import logit, expit
from scipy.stats import poisson
from tracts.population import Population
from tracts.core import optimize_cob, optimize_cob_sex_biased, optimize_cob_sex_biased_fixed_values
import tracts.hybrid_pedigree as HP
from tracts.phase_type_distribution import PhTMonoecious, PhTDioecious
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.demography.parameter import Parameter ,ParamType
from tracts.demography import DemographicModel

logger = logging.getLogger("tracts")
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Stream handler: warnings/errors to stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(formatter)

# Memory buffer for early log messages
memory_handler = logging.handlers.MemoryHandler(
    capacity = 10000,  
    flushLevel=logging.CRITICAL,
    target=None
)
memory_handler.setLevel(logging.INFO)

# Add handlers only once
if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(memory_handler)

filepath_error_additional_message = (
    '\nPlease ensure that the file path is either absolute,'
    ' or relative to the working directory, script directory,'
    ' or the directory of the driver yaml.'
)

def locate_file_path(
    filename: str,
    script_dir: str | Path | None,
    absolute_driver_yaml_path: str | Path | None = None,
    verbose: bool = False,
):
    search_methods = [
        (Path(filename), "working directory"),
        (Path(script_dir) / filename if script_dir else None, "script directory"),
        (
            absolute_driver_yaml_path.parent / filename
            if isinstance(absolute_driver_yaml_path, Path) else None,
            "driver yaml",
        ),
    ]

    for filepath, method_name in search_methods:
        if filepath is None:
            continue
        if verbose:
            logger.debug(f"{method_name}: {filepath}")
        if filepath.is_file():
            if verbose:
                logger.debug(f"Found {filename} using {method_name}.")
            return filepath

    for pathname in sys.path:
        candidate = Path(pathname) / filename
        if candidate.is_file():
            if verbose:
                logger.debug(f"Found {filename} from {pathname}.")
            return candidate

    return None

def set_log_file(log_filename: str | Path):
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add the real file handler
    logger.addHandler(file_handler)

    # Flush buffered records to it
    memory_handler.setTarget(file_handler)
    memory_handler.flush()

    # Remove memory handler
    logger.removeHandler(memory_handler)
    memory_handler.close()

def run_tracts(driver_filename, script_dir=None):

    driver_path = locate_file_path(filename=driver_filename, script_dir=script_dir)
    driver_spec = load_driver_file(driver_path)

    if hasattr(driver_spec, "log_filename") and driver_spec.log_filename:
        log_path = Path(driver_spec.log_filename)
        if log_path.suffix == "":
            log_path = log_path.with_suffix(".log")
        log_filename = log_path

    else:
        log_filename = "tracts.log"
        logger.warning(f"No log filename specified in driver file. Defaulting to {log_filename} in the working directory.")

    set_log_file(log_filename)

    logger.info(f"Using log file: {log_filename}")
    logger.info(f"Running tracts 2.0 with driver file: {driver_filename}")

    print('------------------------------------------------------------------------------------------------\n')
    print('Running tracts 2.0 with driver file:', driver_filename,'\n')
    print('Reading data, demographic model and driver specifications...\n')
    print('------------------------------------------------------------------------------------------------\n')   

    if not hasattr(driver_spec, "output_filename_format"):
        raise Exception("Please specify an output filename format in the driver file under 'output_filename_format'.")
        
    if hasattr(driver_spec, 'verbose') :
        verbose = driver_spec.verbose
    else:
        verbose = 0

    if hasattr(driver_spec, 'two_steps_optimization') :
        two_steps_optimization = driver_spec.two_steps_optimization
    else:
        two_steps_optimization = True
    
    if hasattr(driver_spec, 'run_optimize_cob') :
        run_optimize_cob = driver_spec.run_optimize_cob
    else:
        run_optimize_cob = False

    # Set autosomal and allosomal models for admixture
    if hasattr(driver_spec, 'ad_model_autosomes') :
        ad_model_autosomes = driver_spec.ad_model_autosomes
        if not ad_model_autosomes in ['DC','DF','M','H-DC','H-DF']:
            print('The model for autosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), M (for Monoecious), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_autosomes = DC by default.')
            ad_model_autosomes = 'DC'
    else:
        print('Model for autosomal admixture not specified. Setting DC by default.')
        ad_model_autosomes = 'DC'
    
    # Currently assumes allosomes is a single label. May change in the future
    allosome_labels = driver_spec.samples.allosomes
    allosome_label = allosome_labels[0] if len(allosome_labels) > 0 else None

    if hasattr(driver_spec, 'ad_model_allosomes') and allosome_label is not None:
        ad_model_allosomes = driver_spec.ad_model_allosomes
        if not ad_model_allosomes in ['DC','DF','H-DC','H-DF']:
            print('The model for allosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_allosomes = DC by default.')
            ad_model_allosomes = 'DC'
    elif allosome_label is not None:
        print('Model for allosomal admixture not specified. Setting DC by default.')
        ad_model_allosomes = 'DC'
    else:
        print('No allosomes found in the sample. Modelling only autosomal admixture.')
        ad_model_allosomes = None

    exclude_tracts_below_cM = driver_spec.exclude_tracts_below_cm
    print(f'excluding_tracts_below set to {exclude_tracts_below_cM} cM.')
    
    npts = driver_spec.npts

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
    
    print("Ancestries:", [ancestry for ancestry in ancestor_labels] )
    
    print("Data autosome proportions:", ancestry_proportions )
    
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
        mult_params_message = "Multiple starting parameters were generated. These will be converted to optimizer units and used for multiple optimization runs."
        logger.info(mult_params_message)
        print("\n"+mult_params_message)

    else:
        single_params_message = "A single set of starting parameters was generated. It will be converted to optimizer units and used for optimization."
        logger.info(single_params_message)
        print("\n"+single_params_message)

    header = f"{'Run':>3} | {'Starting parameters':<45}"
    line = "-" * len(header) 

    for l in (line, header, line):
        print(l)
        logger.info(l)

    for i, (phys, opt) in enumerate(zip(physical_start_params, optimizer_start_params)):
        assert np.isclose(phys, model.parameter_handler.convert_to_physical_params(opt)).all()
        if bound(opt)<0:
            print("Warning, starting parameters are out of bounds.")
        phys_str = ", ".join(f"{x:.4g}" for x in phys)
        start_param_message = f"{1+i:>3} | [{phys_str:<43}]"
        print(start_param_message)
        logger.info(start_param_message)
    print("-" * len(header))

    # Get keys from first run
    first_props = model.proportions_from_matrices(func(optimizer_start_params[0]))
    tract_types = list(first_props.keys())

    start_ancestry_props_message = "Starting ancestry proportions for the starting parameters"
    header = f"{'Run':>3} | " + " | ".join(f"{k:<35}" for k in tract_types)
    line = "-" * len(header)
    #print("\n" + start_ancestry_props_message)
    logger.info(start_ancestry_props_message)
    for l in (line, header, line):
        #print(l)
        logger.info(l)

    for i, opt in enumerate(optimizer_start_params):
        try: 
            props = model.proportions_from_matrices(func(opt))

        except ValueError:
            print("Could not compute starting ancestry proportions - likely due to out of bounds starting parameters.")

        row_values = []
        for k in tract_types:
            arr = props[k]
            arr_str = ", ".join(f"{x:.4g}" for x in arr)
            row_values.append(f"[{arr_str:<33}]")

        anc_line = f"{1+i:>3} | " + " | ".join(row_values)
        print(anc_line)
        logger.info(anc_line)
    print(line)
          
    params_found, likelihoods = run_model_multi_init(model_func=func,
                                                    bound_func=bound,
                                                    population=pop, 
                                                    population_labels=ancestor_labels,
                                                    start_params_list=optimizer_start_params,
                                                    population_dict=pop_dict,
                                                    parameter_handler=model.parameter_handler,
                                                    max_iter=max_iter,
                                                    exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                    ad_model_autosomes = ad_model_autosomes, 
                                                    ad_model_allosomes=ad_model_allosomes,
                                                    npts=npts, 
                                                    verbose=verbose, 
                                                    two_steps_optimization=two_steps_optimization, 
                                                    run_optimize_cob=run_optimize_cob)

    formatted_likelihoods = [float(x) for x in likelihoods]
    physical_found_params = [model.parameter_handler.convert_to_physical_params(f_param) for f_param in params_found]

    if len(formatted_likelihoods) > 1:

        print("\n---------------------------------------------------------------------------")
        results_message = "Results from multiple optimization runs with different starting parameters:"
        header = f"{'Run':>3} | {'LogLik':>12} | Found parameters"
        line = "-" * len(header)
        for l in (results_message, line, header, line):
            print(l)
            logger.info(l)  
        
        for i, (params, ll) in enumerate(zip(physical_found_params, formatted_likelihoods)):
            params_str = ", ".join(f"{p:.4g}" for p in params)
            param_line = f"{1+i:>3} | {float(ll):>12.6g} | [{params_str}]"
            print(param_line)
            logger.info(param_line)
        print(line)
    
    optimal_params, optimal_likelihood = max(zip(physical_found_params, formatted_likelihoods), key=lambda x: x[1])
    #optimal_params = model.parameter_handler.convert_to_physical_params(optimal_params)

    final_message = "Final parameters and corresponding likelihood:"
    param_names = list(model.model_base_params.keys())
    header = f"{'LogLik':>12} | " + " ".join(f"{name:>12}" for name in param_names)
    line = "-" * len(header)
    print("\n" + final_message)
    for l in (line, header, line):
        print(l)
        logger.info(l)  
    
    values_str = " ".join(f"{x:>12.4g}" for x in optimal_params)
    loglik_message = f"{float(optimal_likelihood):>12.6g} | {values_str}"
    logger.info(loglik_message)
    print(loglik_message)
    print(line)

    bound = model.get_violation_score(optimal_params, verbose = True)

    output_simulation_data_sex_biased(pop, optimal_params, model, driver_spec, ad_model_autosomes=ad_model_autosomes, ad_model_allosomes=ad_model_allosomes)

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
    log_filename: Optional[str] = "tracts.log"
    ad_model_autosomes: str = "M"
    ad_model_allosomes: str = "DC"
    verbose: int = 0
    log_scale: bool = True
    two_steps_optimization: bool = True
    run_optimize_cob: bool = False

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
    logger.info(f'Allosomes: {allosome_labels if allosome_labels else []}')
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


def parse_individual_filenames(
    individual_names,
    filename_string,
    script_dir: str | Path | None,
    labels=['A', 'B'],
    directory='',
    absolute_driver_yaml_path=None,
):
    resolved_files = []

    def _find_individual_file(individual_name, label_val):
        requested_filename = directory + filename_string.format(
            name=individual_name,
            label=label_val
        )

        filepath = locate_file_path(
            filename=requested_filename,
            script_dir=script_dir,
            absolute_driver_yaml_path=absolute_driver_yaml_path,
            verbose=False,
        )

        if filepath is None:
            raise FileNotFoundError(
                f'File for individual {individual_name} '
                f'("{requested_filename}") could not be found.'
                f'{filepath_error_additional_message}'
            )

        resolved_files.append(filepath)
        return str(filepath)

    individual_filenames = {
        individual_name: [
            _find_individual_file(individual_name, label_val)
            for label_val in labels
        ]
        for individual_name in individual_names
    }

    logger.info("Found %d input .bed files.", len(resolved_files))
    for path in resolved_files:
        logger.info("  - %s", path)

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
        
    
    
    logger.info(f'Starting parameters in physical units: {start_params}')
    
    if len(model.params_fixed_by_ancestry) > 0:
        start_params = [model.parameter_handler.compute_params_fixed_by_ancestry(start_param_set)
         for start_param_set in start_params]
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
                          max_iter: int=None, exclude_tracts_below_cM: int = 0, ad_model_autosomes = 'DC', 
                          ad_model_allosomes = 'DC', npts: int = 50, verbose: int = 0, two_steps_optimization: bool = True, run_optimize_cob: bool = False) -> tuple[list[np.ndarray], list[float]]:
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
        opt_run_message = f"\nOptimization run #{len(optimal_params)+1}\n--------------------"
        print(opt_run_message)
        logger.info(opt_run_message)
        logger.debug(f'Starting parameters in optimizer units: {start_params}')
        params_found, likelihood_found = run_model(model_func=model_func,
                                                   bound_func=bound_func, 
                                                   population=population, 
                                                   population_labels=population_labels, 
                                                   startparams=start_params,
                                                   population_dict=population_dict,
                                                   parameter_handler=parameter_handler,
                                                   max_iter=max_iter,
                                                   exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                   ad_model_autosomes=ad_model_autosomes,
                                                   ad_model_allosomes=ad_model_allosomes,
                                                   npts=npts,
                                                   verbose=verbose,
                                                   two_steps_optimization=two_steps_optimization, 
                                                   run_optimize_cob=run_optimize_cob)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
    return optimal_params, likelihoods

def run_model(model_func, bound_func, population: Population, population_labels, startparams, population_dict, parameter_handler=None, 
              max_iter=None, exclude_tracts_below_cM=0, ad_model_autosomes='DC', ad_model_allosomes='DC', npts=0, verbose=0, two_steps_optimization = True, run_optimize_cob = False):
    
    if not run_optimize_cob:
        return run_model_sex_biased(
            model_func=model_func,
            bound_func=bound_func,
            population=population,
            startparams=startparams,
            population_dict=population_dict,
            parameter_handler=parameter_handler,
            max_iter=max_iter,
            exclude_tracts_below_cM=exclude_tracts_below_cM,
            ad_model_autosomes=ad_model_autosomes,
            ad_model_allosomes=ad_model_allosomes,
            npts=npts,
            verbose=verbose,
            two_steps_optimization=two_steps_optimization,
        )
    elif ad_model_allosomes is None:
        Ls = population.Ls
        nind = population.nind
        bins, data = population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
        data = [data[poplab] for poplab in population_labels]
        model_func_sample_pop = lambda params:list(model_func(params).values())[0]
        xopt = optimize_cob(startparams, bins, Ls, data, nind, model_func_sample_pop, outofbounds_fun=bound_func, epsilon=1e-2)
        optmod = PhTMonoecious(model_func_sample_pop(xopt))
        optlik = optmod.loglik(bins, Ls, data, nind)
        return xopt, optlik
    else:
        raise Exception("The optimize_cob method does not accept allosomal admixture.")

def run_model_sex_biased(model_func, bound_func, population: Population, startparams, population_dict, parameter_handler = None, max_iter=None, 
                         exclude_tracts_below_cM=0, ad_model_autosomes='DC',ad_model_allosomes='DC',npts=0, verbose=0, two_steps_optimization = True):
  
    if not two_steps_optimization:
        optimal_params, optimal_likelihood = optimize_cob_sex_biased(p0=startparams, 
                                                                    population=population,
                                                                    model_func=model_func, 
                                                                    parameter_handler=parameter_handler,
                                                                    outofbounds_fun=bound_func,
                                                                    p_dict = population_dict,
                                                                    exclude_tracts_below_cM=exclude_tracts_below_cM, 
                                                                    maxiter=max_iter,
                                                                    verbose=verbose,
                                                                    ad_model_autosomes=ad_model_autosomes, 
                                                                    ad_model_allosomes=ad_model_allosomes,
                                                                    npts=npts)
    else:
        optimal_params, optimal_likelihood = optimize_cob_sex_biased_fixed_values(p0=startparams, 
                                                                                population=population, 
                                                                                model_func=model_func, 
                                                                                parameter_handler= parameter_handler, 
                                                                                outofbounds_fun = bound_func, 
                                                                                p_dict = population_dict, 
                                                                                exclude_tracts_below_cM=exclude_tracts_below_cM, 
                                                                                maxiter=max_iter,
                                                                                verbose=verbose,
                                                                                ad_model_autosomes=ad_model_autosomes, 
                                                                                ad_model_allosomes=ad_model_allosomes,
                                                                                npts=npts)    
    
    return optimal_params, optimal_likelihood
       

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
    exclude_tracts_below_cM = driver_spec.exclude_tracts_below_cm
    npts = driver_spec.npts
    log_scale = driver_spec.log_scale

    matrices = model.get_migration_matrices(optimal_params)
    matrix_list = [matrix for matrix in matrices.values()]

    if ad_model_allosomes is not None:
        [male_matrix, female_matrix] = matrix_list
    else:
        male_matrix = matrix_list[0]
        female_matrix = matrix_list[0]

    output_filename_format = driver_spec.output_filename_format
    autosome_bins, autosome_data = sample_population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
    Ls = sample_population.Ls
    nind = sample_population.nind

    if ad_model_autosomes in ['DC','DF']:
        autosome_predicted={pop:PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_autosomes).tract_length_histogram_multi_windowed(pop_num, autosome_bins, Ls) for pop, pop_num in model.population_indices.items()}
    elif ad_model_autosomes == 'M':
        autosome_predicted={pop:PhTMonoecious(0.5*(female_matrix+male_matrix), rho=1).tract_length_histogram_multi_windowed(pop_num, autosome_bins, Ls) for pop, pop_num in model.population_indices.items()}
    elif ad_model_autosomes == 'H-DC':
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DC', rr_f=1, rr_m=1, X_chr=False, X_chr_male=False, N_cores=5, population_number= pop_num, bins=autosome_bins, chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    else:
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DF', rr_f=1, rr_m=1, X_chr=False, X_chr_male=False, N_cores=5, population_number= pop_num, bins=autosome_bins, chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    
    # Save autosome results
    
    with open(output_dir + output_filename_format.format(label='tract_length_autosome_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, autosome_bins)))
    
    with open(output_dir + output_filename_format.format(label='autosome_sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            try:
                fdat.write("\t".join(map(str, autosome_data[population])) + "\n")
            except KeyError:
                autosome_data[population] = np.zeros(len(autosome_bins)).tolist()
                print(f'Population {population} not found in autosome data.')

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

    if ad_model_allosomes is not None:
        allosome_bins, allosome_data = sample_population.get_global_allosome_tractlengths('X',npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
        allosome_length = sample_population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
    
        num_males = sample_population.num_males
        num_females = sample_population.num_females
 
        if ad_model_allosomes in ['DC','DF']:
            female_predicted = {pop: PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True).tract_length_histogram_multi_windowed(pop_num, allosome_bins, [allosome_length]) for pop, pop_num in model.population_indices.items()}
            male_predicted = {pop: PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True, X_chromosome_male=True).tract_length_histogram_multi_windowed(pop_num, allosome_bins, [allosome_length]) for pop, pop_num in model.population_indices.items()}
        elif ad_model_allosomes == 'H-DC':
            female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DC', rr_f=1, rr_m=1, X_chr=True, X_chr_male=False, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
            male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DC', rr_f=1, rr_m=1, X_chr=True, X_chr_male=True, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
        else:
            female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DF', rr_f=1, rr_m=1, X_chr=True, X_chr_male=False, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
            male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(female_matrix, male_matrix, TP=2, D_model='DF', rr_f=1, rr_m=1, X_chr=True, X_chr_male=True, N_cores=5, population_number= pop_num, bins=allosome_bins, chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
    
        # Save allosome results

        with open(output_dir + output_filename_format.format(label='tract_length_allosome_bins'), 'w') as fbins:
            fbins.write("\t".join(map(str, allosome_bins)))

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

    # Save optimal parameters

    param_names = list(model.model_base_params.keys())
    params_file_path = output_dir + output_filename_format.format(label="optimal_parameters") + ".txt"

    with open(params_file_path, "w") as f:
        f.write("parameter\tvalue\n")
        for name, value in zip(param_names, optimal_params):
            f.write(f"{name}\t{value}\n")

    # Produce and display plots 

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
        xlabel="Tract Length (M)",
        alpha_ci=0.05):

        fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)

        x_centers = _bin_centers(xbins)
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

            # Predicted mean counts per bin: length K
            y_pred_bin = scale_factor * np.asarray(predicted_dict[pop], dtype=float)

            # Poisson prediction interval per bin: also length K
            y_low_bin = np.asarray(poisson.ppf(alpha_ci / 2, y_pred_bin), dtype=float)
            y_high_bin = np.asarray(poisson.ppf(1 - alpha_ci / 2, y_pred_bin), dtype=float)

            # Extend to length K+1 for step plotting
            y_pred_step = np.r_[y_pred_bin, y_pred_bin[-1]]
            y_low_step = np.r_[y_low_bin, y_low_bin[-1]]
            y_high_step = np.r_[y_high_bin, y_high_bin[-1]]

            # Step line
            ax.step(
                xbins,
                y_pred_step,
                where="post",
                color=color,
                lw=2.2,
                alpha=0.95,
                zorder=2,
            )

            # Shadow for prediction interval
            ax.fill_between(
                xbins,
                y_low_step,
                y_high_step,
                step="post",
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )

            # One legend entry per population
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
        if log_scale:
            ax.set_yscale("log") # Log-scale
            ax.set_ylim(bottom=0.5)
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

    if ad_model_allosomes is not None:
    
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
    logger.info('Results saved to : ' + output_dir)

   
