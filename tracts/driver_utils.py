import numbers
import os
import sys
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import poisson
from tracts.population import Population
from tracts.phase_type import hybrid_pedigree as HP
from tracts.phase_type import PhTMonoecious, PhTDioecious
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.demography.parameter import ParamType
import ruamel.yaml as yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from pydantic_core import PydanticUndefined
import logging
logger = logging.getLogger(__name__)
from datetime import datetime


# --------------- Locate driver file ---------------


def locate_file_path(filename: str, 
                    script_dir: str | Path | None,
                    absolute_driver_yaml_path: str | Path | None = None,
                    verbose: bool = False) -> Optional[Path]:
    
    """
    Locates the file path for a given filename by searching in multiple locations. The search order is as follows:
    1. Working directory
    2. Script directory (if provided)
    3. Directory of the driver yaml file (if provided)
    4. Directories in sys.path

    Parameters
    ----------
    filename: str
        The name of the file to locate.
    script_dir: str | Path | None
        The directory of the script, if provided.
    absolute_driver_yaml_path: str | Path | None
        The absolute path to the driver yaml file, if provided.
    verbose: bool
        If True, logs the search process.
    
    Returns
    -------
    Optional[Path]
        The path to the located file, or None if the file is not found.
    """
                 
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

# --------------- Models ---------------

class SamplesConfig(BaseModel):
    """
    Configuration for the samples used in the inference. 
    This includes information about the directory where the sample files are located,
    the names of the individuals and populations, and the format of the filenames.
    The configuration also specifies which chromosomes to include in the analysis and any allosomes to consider.

    Attributes
    ----------  
    directory: str
        The directory where the sample files are located.
    individual_names: List[str]
        A list of individual names corresponding to the sample files. 
    male_names: List[str] | None
        A list of individual names corresponding to male individuals, or None to automatically determine based on the presence of allosomes.
    filename_format: str
        The format of the sample filenames, which should include placeholders for the individual name and chromosome (e.g. "{individual}_{chromosome}.txt").
    labels: List[str]
        A list of population labels corresponding to the sample files. Defaults to ["A", "B"].
    chromosomes: str
        A string specifying which chromosomes to include in the analysis. 
    allosomes: List[str]
        A list of allosome chromosome names. Currenly only supporting "X".

    """
    model_config = ConfigDict(extra="forbid")

    directory: str
    individual_names: List[str]
    male_names: List[str] | None = None
    filename_format: str
    labels: List[str] = Field(default_factory=lambda: ["A", "B"])
    chromosomes: str
    allosomes: List[str]=[]


class StartParamsConfig(BaseModel):
    """
    Configuration for the starting parameters used in the optimization.
    """
    model_config = ConfigDict(extra="allow")


class InferenceConfig(BaseModel):
    """
    Configuration for the inference process. This determines the list of parameteres that can be processed
    from the driver file, together with their types and default values. Only parameters specified in this class will be processed
    and additional parameters in the driver file will rise an error. This is to ensure that the driver file is correctly specified
    and to provide clear error messages for missing or misspelled parameters. See online documentation for details on how to specify parameters in the driver file.

    Attributes
    ----------
    unknown_labels_for_smoothing: List[str]
        A list of population labels for which to apply smoothing to the tract length distribution. Defaults to an empty list.
    samples: SamplesConfig
        The configuration for the samples used in the inference.
    model_filename: str
        The filename of the demographic model to use for the inference. 
    start_params: StartParamsConfig
        The configuration for the starting parameters used in the optimization.
    repetitions: int
        The number of repetitions to perform for the optimization. Defaults to 1.
    seed: int
        The random seed to use for the optimization.
    maximum_iterations: int | None
        The maximum number of iterations to perform for the optimization. Defaults to None, which means no limit on the number of iterations.
    npts: int
        The number of grid points to use to define the tract length histogram. Defaults to 50.
    exclude_tracts_below_cm: float
        The minimum tract length in centiMorgans to include in the analysis. Tracts shorter than this length will be excluded. Defaults to 1 cM.
    fix_parameters_from_ancestry_proportions: List[str]
        A list of parameter names to fix based on the ancestry proportions. See online documentation for details.
    output_directory: str | None
        The directory where the output files will be saved. 
    output_filename_format: str
        The format of the output filenames.
    log_filename : str, Optional
        The filename of the log file to write to. If None, no log file will be created. Defaults to "tracts.log".
    ad_model_autosomes: str
        The admixture model to use for the autosomes. Must be one in ["M", "DC", "DF", "H-DC", "H-DF]. See online documentation for details. Defaults to "M".
    ad_model_allosomes: str
        The admixture model to use for the allosomes. Must be one in ["DC", "DF", "H-DC", "H-DF]. See online documentation for details. Defaults to "DC".
    verbose_log: int
        The verbosity level for logging. Defaults to 1.
    verbose_screen: int
        The verbosity level for screen prints. Defaults to 30.
    log_scale: bool
        Whether to use log scale to plot the tract length distribution. Defaults to True.
    two_steps_optimization: bool
        Whether to perform a two-step optimization process, where the first step optimizes only the non-sex-bias parameters on autosomal data and the second step optimizes sex-bias parameters using both autosomal and allosomal data. Defaults to True.
    use_autosomes_for_sex_bias: bool
        Whether step 2 should include autosomal data in addition to allosomal data. Defaults to False.
    """

    model_config = ConfigDict(extra="forbid")
    unknown_labels_for_smoothing : List[str] = []
    samples: SamplesConfig
    model_filename: str
    start_params: StartParamsConfig
    repetitions: int =1 
    seed: int
    maximum_iterations: int|None = None 
    npts: int = 50
    exclude_tracts_below_cm: float = 1
    fix_parameters_from_ancestry_proportions: List[str] = []
    output_directory: str|None= None
    output_filename_format: str
    log_filename: Optional[str] = "tracts.log"
    ad_model_autosomes: str = "DC"
    ad_model_allosomes: str = "DC"
    verbose_log: int = 1
    verbose_screen: int = 30
    log_scale: bool = True
    two_steps_optimization: bool = True
    use_autosomes_for_sex_bias: bool = False


# --------------- Driver file setup ---------------


filepath_error_additional_message = (
    '\nPlease ensure that the file path is either absolute,'
    ' or relative to the working directory, script directory,'
    ' or the directory of the driver yaml.'
)

def load_driver_file(driver_path: str) -> InferenceConfig:
    """
    Loads the driver file and validates that it contains all required parameters for the inference. 
    See online documentation for details on how to specify parameters in the driver file.

    Parameters
    ----------
        driver_path: str
            The path to the driver yaml file.
    Returns
    -------
        InferenceConfig
            The configuration for the inference process, as specified in the driver file.

    """
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


# --------------- Loader ---------------


def parse_individual_filenames(
    individual_names: List[str],
    filename_string,
    script_dir: str | Path | None,
    labels=['A', 'B'],
    directory: str = '',
    absolute_driver_yaml_path=None):
    
    """
    Parses the individual filenames based on the provided format and locates their paths. 

    Parameters
    ----------
    individual_names: List[str]
        A list of individual names corresponding to the sample files.
    filename_string: str
        The format of the sample filenames. This should include placeholders for the individual name and haploid copy (e.g. "{name}_{label}.txt").
    script_dir: str | Path | None
        The directory containing the script.
    labels: List[str]
        A list of labels for the haploid copies.
    directory: str
        The directory containing the sample files.
    absolute_driver_yaml_path: str | None
        The absolute path to the driver yaml file

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping individual names to a list of file paths.

    """
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


def parse_chromosomes(chromosome_spec: list | str | int, chromosomes: None | list=None):
    """
    Parses a chromosome specification and returns a list of chromosome numbers.

    Parameters
    ----------
    chromosome_spec: list | str | int
        The chromosome specification, which can be an integer, a string representing a range, or a list of specifications.
    chromosomes: None | list
        A list to which the parsed chromosome numbers will be appended.

    Returns
    -------
    list
        A list of chromosome numbers.
    """

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


def load_population(driver_path: str, driver_spec: InferenceConfig, script_dir: str | Path | None=None, allosome_labels: List[str] | None=None):
    """
    Loads the population data based on the specifications in the driver file. 

    Parameters   
    ----------
    driver_path: str
        The path to the driver yaml file.
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file.
    script_dir: str | Path | None
        The directory containing the script.
    allosome_labels: List[str] | None
        A list of allosome chromosome names.

    """

    individual_filenames = parse_individual_filenames(driver_spec.samples.individual_names,
                                                      driver_spec.samples.filename_format,
                                                      labels=driver_spec.samples.labels,
                                                      directory=driver_spec.samples.directory,
                                                      script_dir=script_dir,
                                                      absolute_driver_yaml_path=driver_path)
    
    allosome_labels = allosome_labels if allosome_labels is not None else []
    male_list = driver_spec.samples.male_names
    chromosome_list = parse_chromosomes(driver_spec.samples.chromosomes)
    logger.info(f'Chromosomes: {chromosome_list}')
    logger.info(f'Allosomes: {allosome_labels}')
    pop = Population(filenames_by_individual=individual_filenames,
                    selectchrom=chromosome_list,
                    allosomes=allosome_labels,
                    male_list = male_list)
    if len(allosome_labels)>=1:
        assert(allosome_labels[0] == 'X'), "Currently only X allosome is supported for male determination. Should be first allosome."
    if len(allosome_labels)>0:
        pop.set_males(male_list=male_list,
                    allosome_label=allosome_labels[0]) 
    return pop


def load_model_from_driver(driver_spec: InferenceConfig, script_dir: str | Path | None, driver_path: str, allosome_label: str | None=None):
    """
    Loads the demographic model based on the specifications in the driver file. The model is expected to be defined in a separate yaml file, 
    whose path is specified in the driver file under "model_filename". See online documentation for details on how to specify the model yaml file and its contents.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file.
    script_dir: str | Path | None
        The directory containing the script.
    driver_path :str
        The path to the driver yaml file.
    allosome_label: str | None
        The label of the allosome chromosome, if any. This is used to determine whether allosomal admixture is modelled.

    Returns
    -------
    ParametrizedDemography | ParametrizedDemographySexBiased
        The loaded demographic model, which can be either a ParametrizedDemography or a ParametrizedDemographySexBiased depending on whether allosomal admixture is modelled.
    """ 

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

def parse_start_params(start_param_bounds, model: ParametrizedDemography, repetitions: int=1, seed: float | None = None,
                       sample_param_names: set[str] | None = None, fixed_param_values: dict[str, float] | None = None):
    """
    Produces starting parameters for optimization in physical units. Only produces starting parameters that are compatible with well-defined migration matrices.
    
    Parameters
    ----------
    start_param_bounds
        An object containing attributes corresponding to each parameter in model.model_base_parameters, where the value of each attribute is either a single number (if the starting value for that parameter should be fixed) or a string of the form "min:max" specifying the range from which to sample starting values for that parameter. The parameters specified in start_param_bounds must match those in model.model_base_parameters, and an error will be raised if any parameters are missing or if any extra parameters are included.
    model: ParametrizedDemography
        The demographic model for which to produce starting parameters. 
    repetitions: int
        The number of sets of starting parameters to produce. Defaults to 1.
    seed: float | None
        The random seed to use for sampling starting parameters. Defaults to None.
    sample_param_names: set[str] | None
        Optional subset of parameter names to sample from ``start_param_bounds``.
        If provided, all other non-ancestry-fixed parameters must be supplied in
        ``fixed_param_values``.
    fixed_param_values: dict[str, float] | None
        Optional parameter values to hold fixed while sampling the remaining
        parameters.
    
    Returns
    -------
    list[np.ndarray]: A list of arrays of starting parameters in physical units, where each array corresponds to a set of starting parameters for one repetition of the optimization. The parameters are ordered according to their order in model.model_base_parameters.
    
    Notes
    -----
    Starting-parameter specifications are parsed once per parameter and stored as either
    ``("fixed", value)`` or ``("range", (min, max))``. For each candidate vector,
    independent Uniform(0,1) draws are generated and then transformed per parameter:
    fixed parameters are assigned directly, while ranged parameters are mapped to
    Uniform(min, max) via an affine transform. Parameters fixed by ancestry are not
    sampled from user input and are initialized from the configured ancestry-fixed
    behavior.

    Feasibility is checked by evaluating ``model.get_violation_score(candidate)``.
    Candidates are accepted only when the returned score is non-negative. Any
    ``ValueError`` raised during validation is treated as infeasible, and candidate
    generation continues until the requested number of feasible starts is collected
    or the attempt limit is reached.    
    """ 
    
    num_params = len(model.model_base_params)
    rng = np.random.default_rng(seed=seed)
    sampled_param_names = set(sample_param_names) if sample_param_names is not None else None
    fixed_param_values = {} if fixed_param_values is None else dict(fixed_param_values)

    # ------- Support Pydantic models, plain mappings, and attribute-style config objects -------
    start_param_values = None
    use_attribute_lookup = False
    instance_start_param_values = {}
    missing_attr = object()

    if isinstance(start_param_bounds, Mapping):
        start_param_values = dict(start_param_bounds)
    else:
        model_dump = getattr(start_param_bounds, "model_dump", None)
        if callable(model_dump):
            dumped_values = model_dump()
            if isinstance(dumped_values, Mapping):
                start_param_values = dict(dumped_values)

    if start_param_values is None:
        # Keep compatibility with objects that expose values via class attributes, properties, __slots__, or other descriptors (not just __dict__).
        use_attribute_lookup = True
        try:
            instance_start_param_values = vars(start_param_bounds)
        except TypeError:
            instance_start_param_values = {}

    def has_start_param(param_name: str) -> bool:
        if use_attribute_lookup:
            if param_name in instance_start_param_values:
                return True
            return inspect.getattr_static(start_param_bounds, param_name, missing_attr) is not missing_attr
        return param_name in start_param_values

    def get_start_param(param_name: str):
        if use_attribute_lookup:
            return getattr(start_param_bounds, param_name)
        return start_param_values[param_name]

    # ------- Parse and validate start-parameter specifications once to avoid repeated parsing while resampling -------

    parsed_specs = {}
    for param_name, param_info in model.model_base_params.items():
        if param_name in fixed_param_values:
            parsed_specs[param_name] = ("fixed", float(fixed_param_values[param_name]))
            continue

        if sampled_param_names is not None and param_name not in sampled_param_names and param_name not in model.params_fixed_by_ancestry:
            raise KeyError(
                f"Parameter '{param_name}' must be provided in fixed_param_values when sampling only a subset of parameters."
            )

        if param_name in model.params_fixed_by_ancestry: # Ancestry-fixed parameters do not need to be present in start_param_bounds and default to model lower bound.
            if not has_start_param(param_name):
                parsed_specs[param_name] = ("fixed", float(param_info.bounds[0]))
                continue

            user_value = get_start_param(param_name)
            if isinstance(user_value, numbers.Number):
                parsed_specs[param_name] = ("fixed", float(user_value))
            else:
                try:
                    bounds = [float(bound) for bound in user_value.split(':')]
                    assert len(bounds) == 2
                    parsed_specs[param_name] = ("fixed", bounds[0])
                except Exception as e:
                    raise ValueError("Initial values must be specified as min:max or a single value.") from e
            continue

        if not has_start_param(param_name):
            raise KeyError(f"Initial values were not specified for parameter '{param_name}'.")

        user_value = get_start_param(param_name)

        if isinstance(user_value, numbers.Number):
            parsed_specs[param_name] = ("fixed", float(user_value)) # Initial value set as a single number and not as a range.
        else:
            try:
                bounds = [float(bound) for bound in user_value.split(':')]  # Intervals are specified as "min:max" to avoid confusion with negative values.
                assert len(bounds) == 2
                parsed_specs[param_name] = ("range", (bounds[0], bounds[1]))
            except Exception as e:
                raise ValueError("Initial values must be specified as min:max or a single value.") from e

    # ------- Helper functions to sample starting parameters and check feasibility -------

    def _draw_candidate() -> np.ndarray:
        """
        Draw a single candidate vector of starting parameters. Each parameter is sampled independently based on the parsed specification:
        "fixed": use the provided fixed value, "range": sample uniformly from the interval [min, max].
        """
        candidate = rng.random(num_params)  # Base random draws: independent Uniform(0,1) for each parameter.
        for param_name, param_info in model.model_base_params.items():
            mode, spec = parsed_specs[param_name]
            if mode == "fixed":
                candidate[param_info.index] = spec
            else:
                low, high = spec
                candidate[param_info.index] = candidate[param_info.index] * (high - low) + low # Affine transformation to Uniform(min, max)
        return candidate

    def _is_feasible(start_param_set: np.ndarray) -> bool:
        """
        Return whether a proposed starting parameter vector is feasible. A parameter set is
        considered feasible when the model reports a non-negative violation score. Any ValueError
        raised during validation is treated as infeasible.
        """
        # Suppress warning logs during feasibility checks; only accepted starts are returned.
        demography_logger = logging.getLogger("tracts.demography.base_parametrized_demography")
        original_level = demography_logger.level
        demography_logger.setLevel(logging.ERROR)
        try:
            _tol = 1e-10  # Tolerance for floating-point rounding at the boundary of feasibility
            return model.get_violation_score(start_param_set) >= -_tol
        except (ValueError, Exception):
            return False
        finally:
            demography_logger.setLevel(original_level)

    start_params = []
    max_attempts = max(1000, 100*repetitions)
    attempts = 0

    while len(start_params) < repetitions and attempts < max_attempts:
        attempts += 1
        candidate = _draw_candidate()

        if len(model.params_fixed_by_ancestry) > 0:
            demography_logger = logging.getLogger("tracts.demography.base_parametrized_demography")
            original_level = demography_logger.level
            demography_logger.setLevel(logging.ERROR)
            try:
                candidate = model.parameter_handler.compute_params_fixed_by_ancestry(candidate)
            except (ValueError, AssertionError):
                demography_logger.setLevel(original_level)
                continue
            finally:
                demography_logger.setLevel(original_level)
            # Re-apply any values from fixed_param_values that compute_params_fixed_by_ancestry
            # may have overridden (e.g. sex-bias params held at 0 during step-1 of a two-step
            # optimisation are still in params_fixed_by_ancestry on the shared model object).
            for param_name, value in fixed_param_values.items():
                if param_name in model.params_fixed_by_ancestry:
                    candidate[model.model_base_params[param_name].index] = value

        if _is_feasible(candidate):
            start_params.append(candidate)

    if len(start_params) < repetitions:
        raise ValueError(f"Could not generate {repetitions} feasible starting parameter sets after {attempts} attempts. Try widening valid start ranges.")
        
    return start_params


def collapse_identical_start_params(start_params: list[np.ndarray], step_label: str) -> list[np.ndarray]:
    """
    Collapse repeated identical starting-parameter sets to a single repetition.

    This is used when all generated starting parameters for a step are the same,
    for example because the inputs were specified as single fixed values or
    because ancestry-fixed parameters made every draw collapse to the same point.

    Parameters
    ----------
    start_params: list[np.ndarray]
        Candidate starting-parameter sets for one optimization step.
    step_label: str
        Human-readable label used in the warning message (for example,
        ``"step 1"`` or ``"step 2"``).

    Returns
    -------
    list[np.ndarray]
        The original list if it contains zero or one entry, or a single-entry
        list containing the unique start if every entry is identical.
    """
    if len(start_params) <= 1:
        return start_params

    first = np.asarray(start_params[0], dtype=float)
    if all(np.allclose(np.asarray(candidate, dtype=float), first) for candidate in start_params[1:]):
        logger.warning(f"All generated starting parameters for {step_label} are identical: running a single optimization instead of multiple repetitions.")
        return [np.array(first, copy=True)]

    return start_params


# ---------- Conversion between optimizer and physical parameters ---------

def get_time_scaled_model_func(model: ParametrizedDemography) -> Callable[[np.ndarray], dict[str, np.ndarray]]:
    """
    Computes a function that takes in optimizer parameters, converts them to physical parameters using the model's parameter handler, and returns the migration matrices for those parameters.
    This is necessary because some optimizers may require parameters to be on a different scale (e.g. log scale) than the physical parameters used in the model, so this function serves as a wrapper to apply the necessary transformations before passing parameters to the model.
    
    Parameters
    ----------
    model: ParametrizedDemography
        The demographic model for which to compute the migration matrices.

    Returns
    -------
    Callable[[np.ndarray], dict[str, np.ndarray]]
        A function that takes in optimizer parameters, converts them to physical parameters, and returns the migration matrices for those parameters.
    """
    return lambda params: model.get_migration_matrices(model.parameter_handler.convert_to_physical_params(params))


def get_time_scaled_model_bounds(model: ParametrizedDemography, verbose = False):
    """
    Computes a function that takes in optimizer parameters, converts them to physical parameters using the model's parameter handler, and returns the violation score for those parameters.
    This is necessary because some optimizers may require parameters to be on a different scale (e.g. log scale) than the physical parameters used in the model, so this function serves as a wrapper to apply the necessary transformations before passing parameters to the model.
    
    Parameters
    ----------
    model: ParametrizedDemography
        The demographic model for which to compute the violation score.
    verbose: bool
        Whether to print detailed information about the violation score. Defaults to False.

    Returns
    -------
    Callable[[np.ndarray], float]
        A function that takes in optimizer parameters, converts them to physical parameters, and returns the violation score for those parameters.
    """
    return lambda params: model.get_violation_score(model.parameter_handler.convert_to_physical_params(params), verbose = verbose)


def scale_select_indices(arr, indices_to_scale, scaling_factor=1):
    if len(indices_to_scale) != len(arr):
        raise ValueError(
            f'Length of array ({len(arr)}) was not equal to length of indices_to_scale ({len(indices_to_scale)}).')
    return (np.multiply(indices_to_scale, scaling_factor - 1) + 1) * arr



# --------------- Output production ---------------


def output_simulation_data_sex_biased(sample_population: Population,
                                    optimal_params: np.ndarray, 
                                    optimal_likelihood:float,
                                    model: ParametrizedDemographySexBiased,
                                    driver_spec: InferenceConfig,
                                    ad_model_autosomes: str='DC', 
                                    ad_model_allosomes: str='DC'):
    """
    Creates output graphs to compare data and the theoretical tract length distribution inferred by the model. Also saves
    migration matrices, tract length distributions, and optimal parameters to output files.
    For details on the output files and graphs produced, see online documentation.

    Parameters
    ----------
    sample_population: :class:`tracts.population.Population`
        The population for which to output simulation data.
    optimal_params: np.ndarray
        The optimal parameters for the model.
    model: ParametrizedDemographySexBiased
        The demographic model for which to output simulation data.
    driver_spec: InferenceConfig
        The driver specification containing output configuration.
    ad_model_autosomes: str
        The model for autosomal admixture. Defaults to 'DC'.
    ad_model_allosomes: str
        The model for allosomal admixture. Defaults to 'DC'.
    """
    
    # ------ Create output directory if it doesn't exist ------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    formatted_output_directory =  driver_spec.output_directory.format(date=timestamp)
    
    output_dir = Path.cwd() if not formatted_output_directory  else Path(formatted_output_directory)
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    # ------- Set up output filename format and load required parameters for output production ------
    output_filename_format = driver_spec.output_filename_format
    exclude_tracts_below_cM = driver_spec.exclude_tracts_below_cm
    npts = driver_spec.npts
    log_scale = driver_spec.log_scale

    matrices = model.get_migration_matrices(optimal_params)
    matrix_list = [matrix for matrix in matrices.values()]

    if ad_model_allosomes is not None:
        [male_matrix, female_matrix] = matrix_list # One male-specific and one female-specific migration matrix is produce when allosomal admixture is modelled.
    else:
        male_matrix = matrix_list[0] # If allosomal admixture is not modelled, only one migration matrix is produce. # NOTE: This can be updated in future development to allow for sex-bias inference from autosomal data.
        female_matrix = matrix_list[0]

    # ------- Get tract length distributions for data and model predictions for autosomes ------
    # Autosomal data
    autosome_bins, autosome_data = sample_population.get_global_tractlengths(npts=npts,
                                                                            exclude_tracts_below_cM=exclude_tracts_below_cM)
    Ls = sample_population.Ls
    nind = sample_population.nind

    # Autosomal admixture model predictions
    if ad_model_autosomes in ['DC','DF']:
        autosome_predicted={pop:PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            sex_model=ad_model_autosomes).tract_length_histogram_multi_windowed(population_number=pop_num,
                                                                                                                bins=autosome_bins,
                                                                                                                chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    elif ad_model_autosomes == 'M':
        autosome_predicted={pop:PhTMonoecious(migration_matrix=0.5*(female_matrix+male_matrix),
                                            rho=1).tract_length_histogram_multi_windowed(population_number=pop_num,
                                                                                        bins=autosome_bins,
                                                                                        chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    elif ad_model_autosomes == 'H-DC':
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DC',
                                                                            rho_f=1,
                                                                            rho_m=1,
                                                                            X_chr=False,
                                                                            X_chr_male=False,
                                                                            N_cores=5,
                                                                            population_number=pop_num,
                                                                            bins=autosome_bins,
                                                                            chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    else:
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DF',
                                                                            rho_f=1,
                                                                            rho_m=1,
                                                                            X_chr=False,
                                                                            X_chr_male=False,
                                                                            N_cores=5,
                                                                            population_number=pop_num,
                                                                            bins=autosome_bins,
                                                                            chrom_lengths=Ls) for pop, pop_num in model.population_indices.items()}
    
    # Save autosome results
    with open(output_dir / output_filename_format.format(label='tract_length_autosome_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, autosome_bins)))
    with open(output_dir / output_filename_format.format(label='autosome_sample_tract_distribution'), 'w') as fdat:
        for population in model.population_indices.keys():
            try:
                fdat.write("\t".join(map(str, autosome_data[population])) + "\n")
            except KeyError:
                autosome_data[population] = np.zeros(len(autosome_bins)).tolist()
                print(f'Population {population} not found in autosome data.')
    with open(output_dir / output_filename_format.format(label='female_migration_matrix'), 'w') as fmig2:
        for line in female_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")
    with open(output_dir / output_filename_format.format(label='male_migration_matrix'), 'w') as fmig2:
        for line in male_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")
    with open(output_dir / output_filename_format.format(label='autosome_predicted_tract_distribution'), 'w') as fpred2:
        for pop, pop_num in model.population_indices.items():
            fpred2.write("\t".join(map(
                str,
                [nind * num_tracts for num_tracts in autosome_predicted[pop]]))
                         + "\n")

    # Allosomal data and predictions (if applicable)
    if ad_model_allosomes is not None:
        # Allosomal data
        allosome_bins, allosome_data = sample_population.get_global_allosome_tractlengths(allosome='X',
                                                                                        npts=npts,
                                                                                        exclude_tracts_below_cM=exclude_tracts_below_cM)
        allosome_length = sample_population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
        num_males = sample_population.num_males
        num_females = sample_population.num_females
 
        # Allosomal admixture model predictions
        if ad_model_allosomes in ['DC','DF']:
            female_predicted = {pop: PhTDioecious(migration_matrix_f=female_matrix,
                                                migration_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                sex_model=ad_model_allosomes,
                                                X_chromosome=True).tract_length_histogram_multi_windowed(population_number=pop_num,
                                                                                                        bins=allosome_bins,
                                                                                                        chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
            male_predicted = {pop: PhTDioecious(migration_matrix_f=female_matrix,
                                                migration_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                sex_model=ad_model_allosomes,
                                                X_chromosome=True,
                                                X_chromosome_male=True).tract_length_histogram_multi_windowed(population_number=pop_num,
                                                                                                            bins=allosome_bins,
                                                                                                            chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
        elif ad_model_allosomes == 'H-DC':
            female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                                mig_matrix_m=male_matrix,
                                                                                TP=2,
                                                                                D_model='DC',
                                                                                rho_f=1,
                                                                                rho_m=1,
                                                                                X_chr=True,
                                                                                X_chr_male=False,
                                                                                N_cores=5,
                                                                                population_number=pop_num,
                                                                                bins=allosome_bins,
                                                                                chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
            male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DC',
                                                                            rho_f=1,
                                                                            rho_m=1,
                                                                            X_chr=True,
                                                                            X_chr_male=True,
                                                                            N_cores=5,
                                                                            population_number=pop_num,
                                                                            bins=allosome_bins,
                                                                            chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
        else:
            female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                                mig_matrix_m=male_matrix,
                                                                                TP=2,
                                                                                D_model='DF',
                                                                                rho_f=1,
                                                                                rho_m=1,
                                                                                X_chr=True,
                                                                                X_chr_male=False,
                                                                                N_cores=5,
                                                                                population_number=pop_num,
                                                                                bins=allosome_bins,
                                                                                chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
            male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DF',
                                                                            rho_f=1, rho_m=1,
                                                                            X_chr=True,
                                                                            X_chr_male=True,
                                                                            N_cores=5,
                                                                            population_number=pop_num,
                                                                            bins=allosome_bins,
                                                                            chrom_lengths=[allosome_length]) for pop, pop_num in model.population_indices.items()}
    
        # Save allosome results
        with open(output_dir / output_filename_format.format(label='tract_length_allosome_bins'), 'w') as fbins:
            fbins.write("\t".join(map(str, allosome_bins)))
        with open(output_dir / output_filename_format.format(label='female_allosome_sample_tract_distribution'), 'w') as fdat:
            for population in model.population_indices.keys():
                try:
                    fdat.write("\t".join(map(str, female_data[population])) + "\n")
                except KeyError:
                    female_data[population] = np.zeros(len(allosome_bins)).tolist()
                    print(f'Population {population} not found in female allosome data.')
        with open(output_dir / output_filename_format.format(label='male_allosome_sample_tract_distribution'), 'w') as fdat:
            for population in model.population_indices.keys():
                try:
                    fdat.write("\t".join(map(str, male_data[population])) + "\n")
                except KeyError:
                    male_data[population] = np.zeros(len(allosome_bins)).tolist()
                    print(f'Population {population} not found in male allosome data.')           
        with open(output_dir / output_filename_format.format(label='female_allosome_predicted_tract_distribution'), 'w') as fpred2:
            for pop, pop_num in model.population_indices.items():
                fpred2.write("\t".join(map(
                    str,
                    [num_females * num_tracts for num_tracts in female_predicted[pop]]))
                            + "\n")
        with open(output_dir / output_filename_format.format(label='male_allosome_predicted_tract_distribution'), 'w') as fpred2:
            for pop, pop_num in model.population_indices.items():
                fpred2.write("\t".join(map(
                    str,
                    [num_males * num_tracts for num_tracts in male_predicted[pop]]))
                            + "\n")

    # ------ Save optimal parameters -------
    param_names = list(model.model_base_params.keys())
    params_file_path = output_dir / output_filename_format.format(label="optimal_parameters.txt")
    with open(params_file_path, "w") as f:

        f.write("parameter\tvalue\n")
        for name, value in zip(param_names, optimal_params):
            f.write(f"{name}\t{value}\n")
        f.write(f"likelihood {optimal_likelihood:>12.6g}\n")

    # ------ Produce and display plots -------
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
        xbins: np.ndarray,
        observed_dict: dict,
        predicted_dict: dict,
        scale_factor: float,
        title: str,
        ylabel: str,
        output_path: str,
        xlabel: str="Tract Length (M)",
        alpha_ci: float=0.05,
        subtitle: str = None):

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

            # Predicted mean counts per bin
            y_pred_bin = scale_factor * np.asarray(predicted_dict[pop], dtype=float)

            # Poisson prediction interval per bin
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

        # Main styling — both anchored to axes x=0.5 so they share the same centre
        ax.text(0.5, 1.08, title, transform=ax.transAxes,
                ha='center', va='bottom', clip_on=False,
                fontsize=14, fontweight='bold', fontfamily='Cantarell')
        if subtitle is not None:
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                    ha='center', va='bottom', clip_on=False,
                    fontsize=10, color='0.4')
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


    # --- Produce plot for autosomes ---
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
        subtitle=f"Log-likelihood: {optimal_likelihood:.6g}"
    )

    if ad_model_allosomes is not None:
    
        # --- Produce plot for allosomes in male individuals ---
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
            subtitle=f"Log-likelihood: {optimal_likelihood:.6g}"
        )

        # --- Produce plot for allosomes in female individuals ---
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
            subtitle=f"Log-likelihood: {optimal_likelihood:.6g}"
        )

    
    # Final message
    print('Results saved to : ' + str(output_dir))
    logger.info('Results saved to : ' + str(output_dir))



# --------------- Helper function to summarize optimization results and choose best likelihood ---------------


def _summarize_step_results(params_found: list[np.ndarray], likelihoods: list[float], parameter_handler: FixedParametersHandler,
                            param_names: list[str], step_label: str | None = None,
                            likelihood_tolerance: float = 0.5) -> tuple[np.ndarray, float]:
    """
    Print per-run optimization results and select the best run.

    Parameters
    ----------
    params_found: list[np.ndarray]
        A list of arrays of parameters found by the optimization runs, where each array corresponds to one run and is in optimizer parameter space.
    likelihoods: list[float] 
        A list of likelihoods corresponding to each set of parameters found by the optimization runs.
    parameter_handler: FixedParametersHandler
        The parameter handler for the model.
    param_names: list[str]
        A list of parameter names corresponding to the parameters in the model, used for printing results.
    step_label: str | None
        A label for the optimization step, used for printing results.
    likelihood_tolerance: float
        Absolute tolerance used to decide whether a run reached a likelihood
        value close to the best one.

    Returns
    -------
    np.ndarray
        An array of the optimal parameters in physical parameter space, corresponding to the highest likelihood among the runs.
    float
        The optimal likelihood as a float.
    """
    formatted_likelihoods = [float(x) for x in likelihoods]
    step_prefix = f"In {step_label}: " if step_label else ""
    
    prev_time_param_logging = parameter_handler.enable_time_param_logging # Keep time-transition warnings tied to optimization iterations, not to post-run summary conversions.
    parameter_handler.enable_time_param_logging = False
    try:
        physical_found_params = [
            parameter_handler.convert_to_physical_params(found, report_non_admissible=False)
            for found in params_found
        ]
    finally:
        parameter_handler.enable_time_param_logging = prev_time_param_logging

    if len(formatted_likelihoods) > 1:
        results_message = f"\n{step_prefix}Results from multiple optimization runs with different starting parameters:"
        found_param_col_widths = [max(len(name), 12) for name in param_names]
        header = f"{'Run':>3} | {'LogLik':>12} | " + " | ".join(
            f"{name:>{w}}" for name, w in zip(param_names, found_param_col_widths)
        )
        line = "-" * len(header)
        for message_line in (results_message, line, header, line):
            print(message_line)
            logger.info(message_line)

        for i, (params, ll) in enumerate(zip(physical_found_params, formatted_likelihoods)):
            params_str = " | ".join(
                f"{p:>{w}.4g}" for p, w in zip(params, found_param_col_widths)
            )
            param_line = f"{1+i:>3} | {float(ll):>12.6g} | {params_str}"
            print(param_line)
            logger.info(param_line)
        print(line)

    optimal_params, optimal_likelihood = max(
        zip(physical_found_params, formatted_likelihoods),
        key=lambda x: x[1],
    )

    if len(formatted_likelihoods) > 1:
        close_to_best_count = sum(
            np.isclose(ll, optimal_likelihood, atol=likelihood_tolerance, rtol=0.0)
            for ll in formatted_likelihoods
        )
        if close_to_best_count == 1:
            warning_message = (
                f"{step_prefix}final likelihoods close to the optimum were found only once among "
                f"{len(formatted_likelihoods)} runs (tolerance={likelihood_tolerance:g})."
            )
            logger.warning(warning_message)

    return optimal_params, float(optimal_likelihood)


def _print_step_header_block(parameter_handler: FixedParametersHandler, start_params_list: list[np.ndarray] | None = None,
                            bound_func: Callable[[np.ndarray], float] | None = None, title_message: str | None = None, display_param_indices: list[int] | None = None) -> None:
    """
    Print starting-parameter information before optimization runs begin.

    This helper is for internal use only. It prints a starting-parameter table that is
    logged and shown once before optimization runs begin.

    Parameters
    ----------
    parameter_handler: FixedParametersHandler
        Used to derive parameter labels and convert optimizer-space values for display.
    start_params_list: list[np.ndarray] | None
        Starting parameters in optimizer units. If provided together with
        ``bound_func``, a starting-parameters table is printed.
    bound_func: Callable[[np.ndarray], float] | None
        Function returning a violation score in optimizer space. Used to flag
        out-of-bounds starting values when printing the table.
    title_message: str | None
        Optional title shown above the starting-parameters table.
    display_param_indices: list[int] | None
        Indices of parameters to display in the table. If None, defaults to
        current free-parameter indices from ``parameter_handler``.
    """
    if start_params_list is None or bound_func is None:
        return

    if display_param_indices is None:
        if hasattr(parameter_handler, "free_parameters_indices"):
            display_param_indices = list(parameter_handler.free_parameters_indices)
        else:
            display_param_indices = list(range(len(parameter_handler.convert_to_physical_params(start_params_list[0], report_non_admissible=False))))

    physical_params_list = [
        parameter_handler.convert_to_physical_params(params, report_non_admissible=False)[display_param_indices]
        for params in start_params_list
    ]
    if hasattr(parameter_handler, "indices_to_labels"):
        param_names = list(parameter_handler.indices_to_labels(display_param_indices))
    else:
        param_names = [str(index) for index in display_param_indices]
    param_col_widths = [max(len(name), 12) for name in param_names]

    if title_message is None:
        title_message = "Starting parameters"

    print(title_message)
    logger.info(title_message)

    table_header = f"{'Run':>3} | " + " | ".join(
        f"{name:>{w}}" for name, w in zip(param_names, param_col_widths)
    )
    table_line = "-" * len(table_header)

    for l in (table_line, table_header, table_line):
        print(l)
        logger.info(l)

    for i, (phys, opt) in enumerate(zip(physical_params_list, start_params_list)):
        assert np.isclose(
            phys,
            parameter_handler.convert_to_physical_params(opt)[display_param_indices]
        ).all()
        demography_logger = logging.getLogger("tracts.demography.base_parametrized_demography")
        _prev_level = demography_logger.level
        demography_logger.setLevel(logging.ERROR)
        try:
            out_of_bounds = bound_func(opt) < 0
        finally:
            demography_logger.setLevel(_prev_level)
        if out_of_bounds:
            warning_message = "Warning: starting parameters are out of bounds."
            print(warning_message)
            logger.info(warning_message)
        values_str = " | ".join(
            f"{x:>{w}.4g}" for x, w in zip(phys, param_col_widths)
        )
        start_param_message = f"{1+i:>3} | {values_str}"
        print(start_param_message)
        logger.info(start_param_message)

    print(table_line)
    logger.info(table_line)


def _normalize_multi_init_result(result):
    """
    Normalize outputs from multi-initialization optimization runs.

    This helper accepts legacy 2-item return values as well as the current
    3-item return shape and always returns a 3-tuple:
    ``(params_found, likelihoods, full_likelihoods)``.

    Parameters
    ----------
    result
        Tuple returned by ``run_model_multi_init``. Supported forms are: ``(params_found, likelihoods)`` or ``(params_found, likelihoods, full_likelihoods)``.

    Returns
    -------
    tuple
        A 3-item tuple ``(params_found, likelihoods, full_likelihoods)``. If ``result`` has only two items, ``full_likelihoods`` is filled with
        ``None`` values matching the number of runs.

    Raises
    ------
    ValueError
        If ``result`` does not contain exactly 2 or 3 items.
    """
    if len(result) == 3:
        return result
    if len(result) == 2:
        params_found, likelihoods = result
        return params_found, likelihoods, [None] * len(params_found)
    raise ValueError("run_model_multi_init must return either 2 or 3 values.")


def _get_display_param_indices(parameter_handler: FixedParametersHandler,
                               model,
                               two_steps_optimization: bool,
                               steps: list[int | str] | None = None) -> list[int]:
    """
    Compute which parameter columns should be shown in starting-parameter tables.

    The selected indices depend on whether optimization is single-step or
    two-step and, for two-step mode, which step is active.

    Parameters
    ----------
    parameter_handler: FixedParametersHandler
        Parameter handler containing free/fixed parameter metadata.
    model
        Demography model used as a fallback source for parameter metadata when ``parameter_handler.demography`` is unavailable.
    two_steps_optimization: bool
        Whether optimization runs in two-step mode.
    steps: list[int | str] | None
        Active step selection (e.g. ``[1]``, ``[2]``, ``[1, 2]``,
        ``["step1"]``, ``["step2"]``).

    Returns
    -------
    list[int]
        Parameter indices to display in the table for the active optimization  context.
    """
    if not two_steps_optimization:
        if hasattr(parameter_handler, "free_parameters_indices"):
            return list(parameter_handler.free_parameters_indices)
        return list(range(len(model.model_base_params)))

    step_2_only = bool(steps) and all(step in (2, "step2") for step in steps)
    model_base_params = (
        parameter_handler.demography.model_base_params
        if hasattr(parameter_handler, "demography")
        else model.model_base_params
    )
    user_params_fixed_by_value = getattr(parameter_handler, "user_params_fixed_by_value", {})
    params_fixed_by_ancestry = getattr(parameter_handler, "params_fixed_by_ancestry", {})

    if step_2_only:
        return list(range(len(model_base_params)))

    return [
        idx for idx, (name, info) in enumerate(model_base_params.items())
        if (
            info.type != ParamType.SEX_BIAS
            and name not in user_params_fixed_by_value
            and name not in params_fixed_by_ancestry
        )
    ]


def _print_run_intro(parameter_handler: FixedParametersHandler,
                     model,
                     start_params_list: list[np.ndarray],
                     bound_func: Callable[[np.ndarray], float],
                     title_message: str,
                     two_steps_optimization: bool,
                     autosomes_in_step_2: bool,
                     steps: list[int | str] | None = None) -> None:
    """
    Print the optimization subtitle and starting-parameter table for a run.

    This helper centralizes the pre-run console/log output shown before each optimization phase. Time-parameter transition logging is temporarily
    disabled while printing the starting-parameter table to avoid emitting admissibility transition warnings during display-only conversions.

    Parameters
    ----------
    parameter_handler: FixedParametersHandler
        Parameter handler used for subtitle generation and parameter conversion.
    model
        Model object used as fallback metadata source when needed.
    start_params_list: list[np.ndarray]
        Starting parameters (in optimizer units) for all runs in the phase.
    bound_func: Callable[[np.ndarray], float]
        Bound/violation function used to flag out-of-bounds starts in the
        table.
    title_message: str
        Title printed above the starting-parameter table.
    two_steps_optimization: bool
        Whether optimization is single-step or two-step.
    autosomes_in_step_2: bool
        In two-step mode, whether step 2 uses autosomal data in addition to
        allosomal data.
    steps: list[int | str] | None
        Active step selection used to compute subtitle and displayed columns.
    """
    if hasattr(parameter_handler, "demography"):
        subtitle_message = _get_optimization_subtitle(
            parameter_handler=parameter_handler,
            two_steps_optimization=two_steps_optimization,
            autosomes_in_step_2=autosomes_in_step_2,
            steps=steps,
        )
    else:
        all_params = model.model_base_params
        if not two_steps_optimization:
            free_params = list(all_params.keys())
            subtitle_message = f"Optimizing model likelihood over parameters {str(free_params)}."
        else:
            normalized_steps = set(steps or [1, 2])
            if 2 in normalized_steps and 1 not in normalized_steps:
                free_params = [
                    name for name, info in all_params.items()
                    if info.type == ParamType.SEX_BIAS
                ]
                step_2_data = "autosomal + allosomal" if autosomes_in_step_2 else "allosomal"
                subtitle_message = f"Step 2 : Optimizing {step_2_data} likelihood over parameters : {str(free_params)}."
            else:
                free_params = [
                    name for name, info in all_params.items()
                    if info.type != ParamType.SEX_BIAS
                ]
                subtitle_message = f"Step 1 : Optimizing autosomal likelihood over parameters {str(free_params)}."

    for line in ["-" * len(subtitle_message), subtitle_message, "-" * len(subtitle_message)]:
        print(line)
        logger.info(line)

    display_param_indices = _get_display_param_indices(
        parameter_handler=parameter_handler,
        model=model,
        two_steps_optimization=two_steps_optimization,
        steps=steps,
    )

    prev_time_param_logging = parameter_handler.enable_time_param_logging
    parameter_handler.enable_time_param_logging = False
    try:
        _print_step_header_block(
            parameter_handler=parameter_handler,
            start_params_list=start_params_list,
            bound_func=bound_func,
            title_message=title_message,
            display_param_indices=display_param_indices,
        )
    finally:
        parameter_handler.enable_time_param_logging = prev_time_param_logging


def _get_optimization_subtitle(parameter_handler: FixedParametersHandler,
                              two_steps_optimization: bool,
                              autosomes_in_step_2: bool,
                              steps: list[int | str] | None = None) -> str:
    """
    Build the user-readable optimization subtitle for the active run phase. This helper determines which parameter set is being optimized in the
    current context and returns the corresponding subtitle string used in console/log headers.

    Behavior
    --------
    - Single-step mode: reports all currently free parameters.
    - Two-step mode, step 2 only: reports sex-bias parameters only.
    - Two-step mode, otherwise: reports non-sex-bias parameters (step 1 view).
    - In two-step mode, parameters fixed by value or ancestry are excluded
      from the displayed list.

    Parameters
    ----------
    parameter_handler: FixedParametersHandler
        Provides parameter metadata and fixed-parameter information.
    two_steps_optimization: bool
        Whether optimization is configured to run in two-step mode.
    autosomes_in_step_2: bool
        If True, the step-2 subtitle references autosomal + allosomal data;
        otherwise it references allosomal data only.
    steps: list[int | str] | None
        Optional explicit step selection. Accepted labels are ``1``/``"step1"`` and ``2``/``"step2"``. If None, both steps are assumed.

    Returns
    -------
    str
        Subtitle describing the active optimization step and parameter subset.
    """
    if not two_steps_optimization:
        free_params = list(parameter_handler.indices_to_labels(parameter_handler.free_parameters_indices))
        return f"Optimizing model likelihood over parameters {str(free_params)}."

    normalized_steps = set()
    if steps is None:
        normalized_steps = {1, 2}
    else:
        for s in steps:
            if s in [1, "step1"]:
                normalized_steps.add(1)
            elif s in [2, "step2"]:
                normalized_steps.add(2)

    all_params = parameter_handler.demography.model_base_params

    if 2 in normalized_steps and 1 not in normalized_steps:
        free_params = [
            name for name, info in all_params.items()
            if info.type == ParamType.SEX_BIAS
            and name not in parameter_handler.user_params_fixed_by_value
            and name not in parameter_handler.params_fixed_by_ancestry
        ]
        step_2_data = "autosomal + allosomal" if autosomes_in_step_2 else "allosomal"
        return f"Step 2 : Optimizing {step_2_data} likelihood over parameters : {str(free_params)}."

    free_params = [
        name for name, info in all_params.items()
        if info.type != ParamType.SEX_BIAS
        and name not in parameter_handler.user_params_fixed_by_value
        and name not in parameter_handler.params_fixed_by_ancestry
    ]
    return f"Step 1 : Optimizing autosomal likelihood over parameters {str(free_params)}."