import numbers
import os
import sys
import inspect
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from scipy.stats import poisson
from tracts.population import Population
from tracts.genetic_model import GeneticModel
from tracts.phase_type import hybrid_pedigree as HP
from tracts.phase_type import PhTMonoecious, PhTDioecious
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.demography.parameter import ParamType
import ruamel.yaml as yaml
import shutil
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from pydantic_core import PydanticUndefined
import logging
logger = logging.getLogger(__name__)


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

class ModelsConfig(BaseModel):
    """
    Configuration for the demographic and admixture models used in the inference.

    Attributes
    ----------
    model_filename: str
        The filename of the demographic model to use for the inference. 
    implicit_population: str | None
        The name of the population to use as the implicit population in the discrete founder event (if any), whose proportion is set to one minus the sum of the proportions contributed by the other source populations.
        The corresponding rate and sex-bias parameters will not be optimized and their optimal values will be derived from the optimal values of the rest of parameters. If None, defaults to the first source population specified in the founder event.
    ad_model_autosomes: str
        The admixture model to use for the autosomes. Must be one in ["M", "DC", "DF", "H-DC", "H-DF]. See online documentation for details. Defaults to "DC".
    ad_model_allosomes: str
        The admixture model to use for the allosomes. Must be one in ["DC", "DF", "H-DC", "H-DF]. See online documentation for details. Defaults to "DC".   
    rho_f: float
        The female-specific recombination rate. Defaults to 1.
    rho_m: float
        The male-specific recombination rate. Defaults to 1.
    TP: int
        The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models. Ignored if not applicable. Defaults to 2.
    """
    model_config = ConfigDict(extra="forbid")
    implicit_population: str | None = None
    model_filename: str
    ad_model_autosomes: str = "DC"
    ad_model_allosomes: str = "DC"
    rho_f: float = 1
    rho_m: float = 1
    TP: int = 2

class StartParamsConfig(BaseModel):
    """
    Configuration for the starting parameters used in the optimization.
    """
    model_config = ConfigDict(extra="allow")

class ParamBoundsConfig(BaseModel):
    """
    Optional lower/upper admissibility bounds for model parameters, specified as ``"min:max"``
    strings (same syntax as ``start_params`` interval bounds, see ``StartParamsConfig``), e.g.::

        bounds:
          t: 1:20
          REUR: 0.1:0.9

    Any parameter not listed here keeps its default bounds (determined by its ``ParamType``: RATE
    in (0, 1), SEX_BIAS in (-1, 1), TIME in (min_time, max_time), UNTYPED unbounded -- see
    ``tracts.demography.parameter.ParamType``). A bound given here narrows (intersects with,
    rather than replaces) that default range: see ``parse_param_bounds``.
    """
    model_config = ConfigDict(extra="allow")

class OptimizationConfig(BaseModel):
    """
    Configuration for the optimization process used in the inference.

    Attributes
    ----------
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
    fix_parameters_by_value: dict[str, float]
        A dict mapping parameter names to their corresponding user-defined fixed values. These parameters are not optimized nor computed from ancestry proportions.
    unknown_labels_for_smoothing: List[str]
        A list of population labels for which to apply smoothing to the tract length distribution. Defaults to an empty list.
    two_steps_optimization: bool
        Whether to perform a two-step optimization process, where the first step optimizes only the non-sex-bias parameters on autosomal data and the second step optimizes sex-bias parameters using both autosomal and allosomal data. Defaults to True.
    use_autosomes_for_sex_bias: bool
        Whether step 2 should include autosomal data in addition to allosomal data. Defaults to False.
    N_cores: int
        The number of CPU cores to use for parallel processing, when the hybrid-pedigree refinements of the DF or DC models are used. Ignored if the hybrid-pedigree refinements are not used. Defaults to 1.
    n_reoptimizations: int
        The number of times to repeat: fixing the sex-bias parameters at their most recently
        optimized values, then re-running the optimization. Defaults to 0 (not run).
    reoptimization_likelihood_tolerance: float
        Absolute tolerance used to decide whether a re-optimization repetition (see
        ``run_sex_bias_fixing_reoptimizations``) has stopped improving the likelihood. Defaults
        to 1e-3.
    rerun_optimization_on_boundaries: bool
        Whether to re-run the optimization (see ``run_boundary_reoptimization``) when one or more
        sex-bias parameters have an optimal value near their +-1 boundary. Defaults to True.
    boundary_tol: float
            The tolerance for determining if a parameter is at its boundary value. Defaults to 0.1.
    near_one: float
        The value to which a sex-bias parameter is fixed when it is near its +-1 boundary. This is used to avoid parameters getting stuck at the boundary.
        When a sex-bias parameter is near its +-1 boundary and gets fixed by value for the boundary
        re-optimization (see ``run_boundary_reoptimization``), it is fixed at ``+-near_one`` rather
        than at its actual (possibly less extreme, e.g. ``1 - boundary_tol``) optimal value.
        Defaults to 0.999.
    repetitions_likelihood_tolerance: float
        Absolute tolerance used to decide whether a run (among the ``repetitions`` runs from
        different starting parameters) reached a likelihood value close to the best one. A
        warning is logged if only one run out of several is found to be within this tolerance of
        the best. Defaults to 0.5.
    bounds_proximity_tol: float
        Relative tolerance, as a fraction of a parameter's admissible range (``upper - lower``,
        see ``bounds``), used at the end of the run to decide whether a final optimal parameter
        value is close to a bound. Only bounds that the user narrowed below their default,
        type-determined value (via ``bounds``) are checked, and only on the narrowed side (see
        ``check_optimal_params_near_bounds``): a parameter sitting at its natural type boundary
        (e.g. a sex-bias parameter at +-1) is not flagged. Defaults to 0.05 (5% of the admissible range).
    """
    model_config = ConfigDict(extra="forbid")
    repetitions: int =1
    seed: int
    maximum_iterations: int|None = None
    npts: int = 50
    exclude_tracts_below_cm: float = 1
    fix_parameters_from_ancestry_proportions: List[str] = []
    fix_parameters_by_value: dict[str, float] = {}
    unknown_labels_for_smoothing : List[str] = []
    two_steps_optimization: bool = True
    use_autosomes_for_sex_bias: bool = False
    N_cores: int = Field(default=1, ge=1)
    n_reoptimizations: int = Field(default=0, ge=0)
    reoptimization_likelihood_tolerance: float = Field(default=1e-3, ge=0)
    rerun_optimization_on_boundaries: bool = True
    boundary_tol: float = Field(default=0.1, ge=0)
    near_one: float = Field(default=0.999, gt=0, lt=1)
    repetitions_likelihood_tolerance: float = Field(default=0.5, ge=0)
    bounds_proximity_tol: float = Field(default=0.05, ge=0, le=0.5)


class OutputConfig(BaseModel):
    """
    Configuration for the output of the inference process.

    Attributes
    ----------
    output_directory: str | None
        The directory where the output files will be saved. 
    output_filename_format: str
        The format of the output filenames.
    log_filename : str, Optional
        The filename of the log file to write to. If None, no log file will be created. Defaults to "tracts.log".
    verbose_log: int
        The verbosity level for logging. Defaults to 1.
    verbose_screen: int
        The verbosity level for screen prints. Defaults to 30.
    log_scale: bool
        Whether to use log scale to plot the tract length distribution. Defaults to True.
    plot_migration_matrices: bool
        Whether to plot the final mean migration matrix together with the sex-bias values per pulse.
    """
    model_config = ConfigDict(extra="forbid")
    output_directory: str|None= None
    output_filename_format: str
    log_filename: Optional[str] = "tracts.log"
    verbose_log: int = 1
    verbose_screen: int = 30
    log_scale: bool = True
    plot_migration_matrices: bool = True

class InferenceConfig(BaseModel):
    """
    Configuration for the inference process. This determines the list of parameters that can be processed
    from the driver file, together with their types and default values. Only parameters specified in this class will be processed
    and additional parameters in the driver file will raise an error. This is to ensure that the driver file is correctly specified
    and to provide clear error messages for missing or misspelled parameters. See online documentation for details on how to specify parameters in the driver file.

    Attributes
    ----------
    samples: SamplesConfig
        The configuration for the samples used in the inference.
    models: ModelsConfig
        The configuration for the demographic and admixture models used in the inference.
    start_params: StartParamsConfig
        The configuration for the starting parameters used in the optimization.
    bounds: ParamBoundsConfig
        Optional per-parameter admissibility bounds, narrowing the default bounds determined by
        each parameter's type. Defaults to an empty ``ParamBoundsConfig()`` (no narrowing) if
        omitted from the driver file. See ``ParamBoundsConfig``.
    optim: OptimizationConfig
        The configuration for the optimization process used in the inference.
    output: OutputConfig
        The configuration for the output of the inference process.
    """
    model_config = ConfigDict(extra="forbid")
    samples: SamplesConfig
    models: ModelsConfig
    start_params: StartParamsConfig
    bounds: ParamBoundsConfig = ParamBoundsConfig()
    optim: OptimizationConfig
    output: OutputConfig


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


def get_admixture_models(driver_spec: InferenceConfig):
    """
    Validates the admixture models specified in the driver file and returns the models for autosomes and allosomes.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file.

    Returns
    -------
    tuple[str, str | None, str | None]
        A tuple containing the admixture model for autosomes, the admixture model for allosomes (or None if no allosomes are specified), and the allosome label (or None if no allosomes are specified).
    """

    # Autosomal admixture model is correctly specified
    ad_model_autosomes = driver_spec.models.ad_model_autosomes
    if not driver_spec.models.ad_model_autosomes in ['DC','DF','M','H-DC','H-DF']:
        print('The model for autosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), M (for Monoecious), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_autosomes = DC by default.')
        ad_model_autosomes = 'DC'
        
    # Check whether allosomes are present in the sample
    allosome_labels = driver_spec.samples.allosomes
    allosome_label = allosome_labels[0] if len(allosome_labels) > 0 else None  # Currently assumes allosomes is a single label. May change in the future

    # Allosomal admixture model is correctly specified
    if hasattr(driver_spec.models, "ad_model_allosomes") and allosome_label is not None:
        ad_model_allosomes = driver_spec.models.ad_model_allosomes
        if not ad_model_allosomes in ['DC','DF','H-DC','H-DF']:
            print('The model for allosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_allosomes = DC by default.')
            ad_model_allosomes = 'DC'
    elif allosome_label is not None:
        print('Model for allosomal admixture not specified. Setting DC by default.')
        ad_model_allosomes = 'DC'
    else:
        print('No allosomes specified in the driver file. Modelling only autosomal admixture.')
        ad_model_allosomes = None # This will trigger the code to not model allosomal admixture.
    
    if ad_model_allosomes is not None:
            admixture_model_message = (
                f"Admixture is modelled with the {ad_model_autosomes} model for autosomes "
                f"and with the {ad_model_allosomes} model for allosomes."
            )
    else:
        admixture_model_message = f"Admixture is modelled with the {ad_model_autosomes} model for autosomes."
        print(admixture_model_message)
        logger.info(admixture_model_message)

    return ad_model_autosomes, ad_model_allosomes, allosome_label


def get_ancestry_proportions(driver_spec: InferenceConfig, population: Population, ancestor_labels: list[str], allosome_label: str):
    """
    Computes and reports the observed ancestry proportions for a population, based on autosomal data and,
    if allosomes are specified in the driver file, allosomal data as well.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file. Used to check whether allosomes are specified in the sample configuration.
    population: Population
        The population for which to calculate ancestry proportions.
    ancestor_labels: list[str]
        A list of ancestry labels for which to calculate the ancestry proportions.
    allosome_label: str
        The label for the allosome to use when calculating allosome ancestry proportions. Only used if allosomes are specified in the driver file.

    Returns
    -------
    tuple[list[float], list[float]]
        A tuple ``(autosome_proportions, allosome_proportions)``, where ``autosome_proportions`` are the ancestry proportions calculated from autosomal data,
        averaged across all individuals in the population, and ``allosome_proportions`` are the corresponding proportions calculated from allosomal data.
        If ``driver_spec.samples.allosomes`` is empty, ``allosome_proportions`` is an empty list.
    """
    autosome_proportions = population.calculate_ancestry_proportions(ancestor_labels)
        
    print(f"Ancestries: {', '.join(ancestor_labels)}")
    autosomal_ancestry_message = f"Data autosome proportions: {np.array2string(autosome_proportions, separator=' ')}"
    print(autosomal_ancestry_message)
    logger.info(autosomal_ancestry_message)

    if len(driver_spec.samples.allosomes)>=1:
        allosome_proportions = population.calculate_allosome_proportions(population_labels=ancestor_labels,
                                                                        allosome_label=allosome_label)
        allosomal_ancestry_message = f"Data allosome proportions: {np.array2string(allosome_proportions, separator=' ')}"
        print(allosomal_ancestry_message)
        logger.info(allosomal_ancestry_message)
    else:
        allosome_proportions = []

    return autosome_proportions, allosome_proportions


def _reorder_ancestry_proportions(old_ancestor_labels: list[str], new_ancestor_labels: list[str],
                                  autosome_proportions: np.ndarray, allosome_proportions: np.ndarray | list):
    """
    Permutes observed ancestry proportions computed under ``old_ancestor_labels`` so that they line
    up with ``new_ancestor_labels`` instead. Used when the implicit population changes during
    boundary re-optimization: this reorders ``demographic_model.population_indices`` (the implicit
    population is always placed last; see ``ParametrizedDemography(SexBiased).load_from_YAML``), so
    any previously-computed proportions must be realigned to match wherever they are reused
    afterwards, whether for display or to fix parameters by ancestry proportion.

    Parameters
    ----------
    old_ancestor_labels: list[str]
        The population order that ``autosome_proportions``/``allosome_proportions`` were computed
        in (i.e. the ``ancestor_labels`` originally passed to ``get_ancestry_proportions``).
    new_ancestor_labels: list[str]
        The population order to realign to.
    autosome_proportions: np.ndarray
        Observed autosomal ancestry proportions, in ``old_ancestor_labels`` order.
    allosome_proportions: np.ndarray | list
        Observed allosomal ancestry proportions, in ``old_ancestor_labels`` order, or ``[]`` if
        allosomes are not modelled.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | list]
        ``(autosome_proportions, allosome_proportions)`` reordered to match ``new_ancestor_labels``.
        Returned unchanged if the two label orders are already identical.
    """
    if new_ancestor_labels == old_ancestor_labels:
        return autosome_proportions, allosome_proportions

    reorder_idx = [old_ancestor_labels.index(label) for label in new_ancestor_labels]
    autosome_proportions = np.asarray(autosome_proportions)[reorder_idx]
    if len(allosome_proportions) > 0:
        allosome_proportions = np.asarray(allosome_proportions)[reorder_idx]
    return autosome_proportions, allosome_proportions


def check_population_labels(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, population: Population, data: dict[str, np.ndarray]):
    """
    Validates that the population labels in the data correspond to the model population labels. 
    Raises a ValueError if any label in the data is not found in the model or in the list of unknown labels to be smoothed over.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to validate the population labels.
    population: Population
        The population for which to validate the population labels.
    data: dict[str, np.ndarray]
        A dictionary containing the data for which to validate the population labels. The keys of the dictionary are the population labels.
    """

    ancestor_labels = demographic_model.population_indices.keys()
    data_labels =  data.keys()
           
    for label in data_labels:
        if label not in ancestor_labels and label not in population.unknown_labels:
            raise ValueError(f"Population label '{label}' found in data but not in model or labels to be smoothed over. data labels: {data_labels}, model labels: {ancestor_labels}, " \
            f"unknown labels: {population.unknown_labels}")


def setup_fixed_parameters(driver_spec: InferenceConfig, demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased,
                           allosome_label: str, autosome_proportions: list[float], allosome_proportions: list[float],
                           print_details: bool = True):

    """
    Sets up fixed parameters in the demographic model based on the specifications in the driver file.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file.
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to set up fixed parameters.
    allosome_label: str
        The label for the allosome to use when setting up fixed parameters. Only used if allosomes are specified in the driver file.
    autosome_proportions: list[float]
        The ancestry proportions calculated from autosomal data, averaged across all individuals in the population.
    allosome_proportions: list[float]
        The ancestry proportions calculated from allosomal data, averaged across all individuals in the population. Only used if allosomes are specified in the driver file.
    print_details: bool
        Whether to print the model parameters list and which parameters were fixed (from ancestry
        proportions or by value), and the boundary-value warning. The corresponding log messages
        are always emitted regardless. Defaults to True; set to False for the quieter
        boundary re-optimization, where this has already been reported once for the original model.

    """
    if len(driver_spec.optim.fix_parameters_from_ancestry_proportions) > 0 or len(driver_spec.optim.fix_parameters_by_value) > 0: # Set up fixed parameters if specified in the driver
            
        # Check for non-overlapping fixed parameters
        overlapping_params = [_param for _param in driver_spec.optim.fix_parameters_from_ancestry_proportions if _param in driver_spec.optim.fix_parameters_by_value.keys()]
        if len(overlapping_params) > 0:
            raise ValueError(f"Parameters {', '.join(overlapping_params)} are specified to be fixed both from ancestry proportions and by value. Please choose only one fixing strategy per parameter.")

        if allosome_label:
            demographic_model.parameter_handler.set_up_fixed_parameters(demography=demographic_model,
                                                                    params_to_fix_by_ancestry=driver_spec.optim.fix_parameters_from_ancestry_proportions,
                                                                    proportions={
                                                                    f'{demographic_model.parametrized_populations[0]}_autosomal':autosome_proportions,
                                                                    f'{demographic_model.parametrized_populations[0]}_{allosome_label}': allosome_proportions
                                                                    },
                                                                    user_params_to_fix_by_value=driver_spec.optim.fix_parameters_by_value
                                                                    )
        else:
            demographic_model.set_up_fixed_parameters(params_to_fix_by_ancestry=driver_spec.optim.fix_parameters_from_ancestry_proportions,
                                                    proportions= {demographic_model.parametrized_populations[0]:autosome_proportions},
                                                    user_params_to_fix_by_value=driver_spec.optim.fix_parameters_by_value) 
    else: # No parameters to fix 
        demographic_model.set_up_fixed_parameters([],{})

    if len(driver_spec.optim.fix_parameters_from_ancestry_proportions) > 0:
        ancestry_fixed_params = ", ".join(driver_spec.optim.fix_parameters_from_ancestry_proportions)
        anc_message = f"The following parameters have been fixed from ancestry proportions: {ancestry_fixed_params}"
        logger.info(anc_message)
        if print_details:
            print(anc_message)
    if len(driver_spec.optim.fix_parameters_by_value) > 0:
        value_fixed_params = ", ".join(driver_spec.optim.fix_parameters_by_value.keys())
        value_message = f"The following parameters have been fixed by value: {value_fixed_params}"
        fixed_at_one = {_param_name: _param_value for _param_name, _param_value in driver_spec.optim.fix_parameters_by_value.items() if np.isclose(_param_value, 1.0, atol=1e-5) or np.isclose(_param_value, -1.0, atol=1e-5)}
        logger.info(value_message)
        if print_details:
            print(value_message)
            if len(fixed_at_one) > 0:
                print("Warning: fixing rate or sex-bias parameters at boundary values may lead to suboptimal results. Consider fixing at 0.99 or -0.99 instead.")



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


def get_param_names_by_type(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased) -> tuple[list[str], list[str], list[str]]:
    """
    Derives a demographic model's free base parameter names, split into sex-bias and
    non-sex-bias subsets, straight from its ``model_base_params``. Since this is fully
    determined by the demographic model itself, callers that already hold a demographic model
    (or a ``GeneticModel`` wrapping one) never need to separately track or pass around these
    three lists.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model whose parameter names are derived.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        A tuple ``(model_param_names, sex_bias_param_names, non_sex_bias_param_names)``, where:

        * ``model_param_names`` is the list of all parameter names, in the order given by
          ``demographic_model.model_base_params``.
        * ``sex_bias_param_names`` is the subset of ``model_param_names`` corresponding to
          sex-bias parameters.
        * ``non_sex_bias_param_names`` is the remaining subset of ``model_param_names``,
          excluding sex-bias parameters.
    """
    model_param_names = list(demographic_model.model_base_params.keys())
    sex_bias_param_names = [
        name for name, info in demographic_model.model_base_params.items()
        if info.type == ParamType.SEX_BIAS
    ]
    non_sex_bias_param_names = [
        name for name in model_param_names
        if name not in sex_bias_param_names
    ]
    return model_param_names, sex_bias_param_names, non_sex_bias_param_names


@dataclass
class ModelReloadContext:
    """
    Bundles the file-location and ancestry-proportion context needed to reload a demographic
    model from its driver/model YAML files (e.g. when the implicit population changes and the
    founder-event structure has to be re-parsed), so that functions needing this context take
    one parameter instead of five.

    Attributes
    ----------
    script_dir: str
        The directory containing the script.
    driver_path: str
        The path to the driver file.
    allosome_label: str | None
        The label used for allosomal data, as returned by ``get_admixture_models``.
    autosome_proportions: dict
        Observed autosomal ancestry proportions, as returned by ``get_ancestry_proportions``.
    allosome_proportions: dict
        Observed allosomal ancestry proportions, as returned by ``get_ancestry_proportions``.
    """
    script_dir: str
    driver_path: str
    allosome_label: str | None
    autosome_proportions: dict
    allosome_proportions: dict


def load_demographic_model_from_driver(driver_spec: InferenceConfig, script_dir: str | Path | None, driver_path: str, allosome_label: str | None=None):
    """
    Loads the demographic model based on the specifications in the driver file. The model is expected to be defined in a separate yaml file, 
    whose path is specified in the driver file under "models.model_filename". See online documentation for details on how to specify the model yaml file and its contents.

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
    tuple[ParametrizedDemography | ParametrizedDemographySexBiased, list[str], list[str], list[str]]
        A tuple ``(demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names)``, where:

        * ``demographic_model`` is the loaded demographic model, which can be either a ``ParametrizedDemography`` or a ``ParametrizedDemographySexBiased`` depending on whether allosomal admixture is modelled.
        * ``model_param_names`` is the list of all parameter names of ``demographic_model``, in the order given by ``demographic_model.model_base_params``.
        * ``sex_bias_param_names`` is the subset of ``model_param_names`` corresponding to sex-bias parameters.
        * ``non_sex_bias_param_names`` is the remaining subset of ``model_param_names``, excluding sex-bias parameters.
    """

    model_path = locate_file_path(filename=driver_spec.models.model_filename,
                                  script_dir=script_dir,
                                  absolute_driver_yaml_path=driver_path)
    if model_path is None:
        raise FileNotFoundError(f'Model yaml file {driver_spec.models.model_filename} could not be found. {filepath_error_additional_message}')
    if allosome_label:
        demographic_model = ParametrizedDemographySexBiased.load_from_YAML(source=str(model_path.resolve()),
                                                               implicit_population=driver_spec.models.implicit_population)
        demographic_model.allosome_label=allosome_label
    else:    
        demographic_model = ParametrizedDemography.load_from_YAML(source=str(model_path.resolve()),
                                                      implicit_population=driver_spec.models.implicit_population)

    model_param_names, sex_bias_param_names, non_sex_bias_param_names = get_param_names_by_type(demographic_model)

    return demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names

def parse_start_params(start_param_bounds, demographic_model: ParametrizedDemography, repetitions: int=1, seed: float | None = None,
                       sample_param_names: set[str] | None = None, fixed_param_values: dict[str, float] | None = None):
    """
    Produces starting parameters for optimization in physical units. Only produces starting parameters that are compatible with well-defined migration matrices.
    
    Parameters
    ----------
    start_param_bounds
        An object containing attributes corresponding to each parameter in demographic_model.model_base_parameters, where the value of each attribute is either a single number (if the starting value for that parameter should be fixed) or a string of the form "min:max" specifying the range from which to sample starting values for that parameter. The parameters specified in start_param_bounds must match those in demographic_model.model_base_parameters, and an error will be raised if any parameters are missing or if any extra parameters are included.
    demographic_model: ParametrizedDemography
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
    list[np.ndarray]: A list of arrays of starting parameters in physical units, where each array corresponds to a set of starting parameters for one repetition of the optimization. The parameters are ordered according to their order in demographic_model.model_base_parameters.
    
    Notes
    -----
    Starting-parameter specifications are parsed once per parameter and stored as either
    ``("fixed", value)`` or ``("range", (min, max))``. For each candidate vector,
    independent Uniform(0,1) draws are generated and then transformed per parameter:
    fixed parameters are assigned directly, while ranged parameters are mapped to
    Uniform(min, max) via an affine transform. Parameters fixed by ancestry are not
    sampled from user input and are initialized from the configured ancestry-fixed
    behavior.

    Feasibility is checked by evaluating ``demographic_model.get_violation_score(candidate)``.
    Candidates are accepted only when the returned score is non-negative. Any
    ``ValueError`` raised during validation is treated as infeasible, and candidate
    generation continues until the requested number of feasible starts is collected
    or the attempt limit is reached.    
    """ 
    
    num_params = len(demographic_model.model_base_params)
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
    for param_name, param_info in demographic_model.model_base_params.items():
        if param_name in fixed_param_values:
            parsed_specs[param_name] = ("fixed", float(fixed_param_values[param_name]))
            continue

        if sampled_param_names is not None and param_name not in sampled_param_names and param_name not in demographic_model.params_fixed_by_ancestry:
            raise KeyError(
                f"Parameter '{param_name}' must be provided in fixed_param_values when sampling only a subset of parameters."
            )

        if param_name in demographic_model.params_fixed_by_ancestry: # Ancestry-fixed parameters do not need to be present in start_param_bounds and default to model lower bound.
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
        for param_name, param_info in demographic_model.model_base_params.items():
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
            return demographic_model.get_violation_score(start_param_set) >= -_tol
        except ValueError:
            return False
        finally:
            demography_logger.setLevel(original_level)

    start_params = []
    max_attempts = max(1000, 100*repetitions)
    attempts = 0

    while len(start_params) < repetitions and attempts < max_attempts:
        attempts += 1
        candidate = _draw_candidate()

        if len(demographic_model.params_fixed_by_ancestry) > 0:
            demography_logger = logging.getLogger("tracts.demography.base_parametrized_demography")
            original_level = demography_logger.level
            demography_logger.setLevel(logging.ERROR)
            try:
                candidate = demographic_model.parameter_handler.compute_params_fixed_by_ancestry(candidate)
            except (ValueError, AssertionError):
                demography_logger.setLevel(original_level)
                continue
            finally:
                demography_logger.setLevel(original_level)
            # Re-apply any values from fixed_param_values that compute_params_fixed_by_ancestry
            # may have overridden (e.g. sex-bias params held at 0 during step-1 of a two-step
            # optimisation are still in params_fixed_by_ancestry on the shared demographic_model object).
            for param_name, value in fixed_param_values.items():
                if param_name in demographic_model.params_fixed_by_ancestry:
                    candidate[demographic_model.model_base_params[param_name].index] = value

            # Re-apply only values that were explicitly supplied via fixed_param_values.
            # Ancestry-fixed params absent from fixed_param_values also appear in parsed_specs
            # as ("fixed", lower_bound) due to the default fallback; restoring those here would
            # overwrite the correctly ancestry-solved value with the lower bound.
            for anc_param_name in demographic_model.params_fixed_by_ancestry:
                if anc_param_name in fixed_param_values:
                    anc_param_info = demographic_model.model_base_params[anc_param_name]
                    candidate[anc_param_info.index] = fixed_param_values[anc_param_name]

        if _is_feasible(candidate):
            start_params.append(candidate)

    if len(start_params) < repetitions:
        raise ValueError(f"Could not generate {repetitions} feasible starting parameter sets after {attempts} attempts. Try widening valid start ranges.")
        
    return start_params


def parse_param_bounds(param_bounds, demographic_model: ParametrizedDemography) -> None:
    """
    Narrows each model parameter's admissible bounds according to the ``"min:max"`` intervals
    given in ``param_bounds`` (typically ``driver_spec.bounds``), mutating
    ``demographic_model.model_base_params[name].bounds`` in place.

    Only parameters explicitly present in ``param_bounds`` are affected; any parameter not
    mentioned keeps its default, type-determined bounds (see ``tracts.demography.parameter.ParamType``).
    A given bound is intersected with  (not substituted for) the parameter's current bounds, so
    it can only narrow the admissible region, never widen it beyond what the parameter's type
    already allows (e.g. a RATE parameter can be narrowed within (0, 1), but not widened past it).

    Parameters
    ----------
    param_bounds
        An object with an entry per parameter to narrow, each value a ``"min:max"`` string (same
        format as ``start_params`` interval bounds). Accepts a ``Mapping`` or a pydantic model
        (e.g. ``ParamBoundsConfig``). For either, any key that isn't a parameter of
        ``demographic_model`` raises, to catch typos. Parameters absent from ``param_bounds``
        are left unchanged.
    demographic_model: ParametrizedDemography
        The demographic model whose parameter bounds are narrowed in place.

    Raises
    ------
    KeyError
        If ``param_bounds`` (given as a ``Mapping`` or pydantic model) specifies a name that is
        not a parameter of ``demographic_model``.
    ValueError
        If a bound is not of the form ``"min:max"``, or if the requested interval does not
        overlap the parameter's current (type-determined) bounds.

    Notes
    -----
    ``param_bounds`` also accepts any other attribute-style object (mirroring
    ``parse_start_params``'s flexibility, mainly for testing): in that case
    only names that match an actual model parameter are read, and other attributes present on
    the object are ignored, since there is no reliable way to enumerate "all" attributes of an
    arbitrary object.
    """
    if isinstance(param_bounds, Mapping):
        bound_values = dict(param_bounds)
    else:
        model_dump = getattr(param_bounds, "model_dump", None)
        bound_values = dict(model_dump()) if callable(model_dump) else None

    if bound_values is not None:
        unknown_names = [name for name in bound_values if name not in demographic_model.model_base_params]
        if unknown_names:
            raise KeyError(
                f"bounds specifies parameter(s) {', '.join(unknown_names)}, which are not parameters "
                f"of this model. Model parameters are: {', '.join(demographic_model.model_base_params.keys())}."
            )
    else:
        # Attribute-style object: only look up names that are actual model parameters (mirroring
        # parse_start_params's get_start_param), since arbitrary objects (e.g. Mock, in tests)
        # cannot be safely enumerated for "unknown" extra attributes.
        missing_attr = object()
        bound_values = {}
        for param_name in demographic_model.model_base_params:
            value = inspect.getattr_static(param_bounds, param_name, missing_attr)
            if value is not missing_attr:
                bound_values[param_name] = value

    for param_name, user_value in bound_values.items():
        try:
            lower, upper = (float(bound) for bound in user_value.split(':'))
            assert lower < upper
        except Exception as e:
            raise ValueError(
                f"bounds for parameter '{param_name}' must be specified as \"min:max\" with min < max, "
                f"got {user_value!r}."
            ) from e

        param_object = demographic_model.model_base_params[param_name]
        default_lower, default_upper = param_object.bounds
        new_lower, new_upper = max(lower, default_lower), min(upper, default_upper)
        if new_lower >= new_upper:
            raise ValueError(
                f"bounds for parameter '{param_name}' ({lower}:{upper}) do not overlap its "
                f"admissible range ({default_lower}:{default_upper})."
            )
        param_object.bounds = (new_lower, new_upper)


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

def check_start_params(physical_start_params: list[np.ndarray], model_param_names: list[str]):
    """
    Checks that the number of starting parameters matches the number of model parameters and prints a message about the starting parameters setup.

    Parameters
    ----------
    physical_start_params: list[np.ndarray]
        A list of arrays of starting parameters in physical units, where each array corresponds to a set of starting parameters for one repetition of the optimization. The parameters are ordered according to their order in demographic_model.model_base_parameters.
    model_param_names: list[str]
        A list of all parameter names of the demographic model, in the order given by demographic_model.model_base_params.
    """

    # ------ Message about starting parameters setup ------ 
    if len(physical_start_params) > 1: # Multiple runs with different starting parameters
        mult_params_message = "Multiple starting parameters will be generated and used for multiple optimization runs."
        logger.info(mult_params_message)
        print(mult_params_message+"\n")

    else: # Single run with one set of starting parameters
        single_params_message = "A single set of starting parameters was generated. It will be converted to optimizer units and used for optimization."
        logger.info(single_params_message)
        print(single_params_message+"\n")

    # ------ Print starting parameters in physical units ------
    n_start_params = len(physical_start_params[0]) if len(physical_start_params) > 0 else 0
    assert len(model_param_names) == n_start_params

def get_starting_ancestry_proportions(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, model_func: Callable[[np.ndarray], dict[str, np.ndarray]], optimizer_start_params: list[np.ndarray]):
    """
    Computes and logs the starting ancestry proportions for each set of starting parameters.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to compute the starting ancestry proportions.
    model_func: Callable[[np.ndarray], dict[str, np.ndarray]]
        A function that takes in optimizer parameters and returns the migration matrices for those parameters.
    optimizer_start_params: list[np.ndarray]
        A list of arrays of starting parameters in optimizer units, where each array corresponds to a set of starting parameters for one repetition of the optimization. The parameters are ordered according to their order in demographic_model.model_base_parameters.
    """

    first_props = demographic_model.proportions_from_matrices(model_func(optimizer_start_params[0]))
    tract_types = list(first_props.keys())
    start_ancestry_props_message = "Starting ancestry proportions for the starting parameters"
    header = f"{'Run':>3} | " + " | ".join(f"{k:<35}" for k in tract_types)
    line = "-" * len(header)
    logger.info(start_ancestry_props_message)
    
    for l in (line, header, line):
        logger.info(l)

    for i, opt in enumerate(optimizer_start_params):
        try: 
            props = demographic_model.proportions_from_matrices(model_func(opt))

        except ValueError:
            print("Could not compute starting ancestry proportions - likely due to out of bounds starting parameters.")

        row_values = []
        for k in tract_types:
            arr = props[k]
            arr_str = ", ".join(f"{x:.4g}" for x in arr)
            row_values.append(f"[{arr_str:<33}]")

        anc_line = f"{1+i:>3} | " + " | ".join(row_values)
        logger.info(anc_line)


def get_predicted_ancestry_proportions(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, model_func: Callable[[np.ndarray], dict[str, np.ndarray]], optimal_params: np.ndarray):
    """
    Computes and logs the predicted ancestry proportions for the optimal parameters.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to compute the predicted ancestry proportions.
    model_func: Callable[[np.ndarray], dict[str, np.ndarray]]
        A function that takes in optimizer parameters and returns the migration matrices for those parameters.
    optimal_params: np.ndarray
        The final optimal parameters in optimizer units, as returned by the optimization process.
    
    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        A tuple containing the predicted autosome proportions and predicted allosome proportions, respectively. Each is an array of proportions corresponding to the populations in the model. If no autosomal or allosomal proportions are predicted, the corresponding value in the tuple will be None.
    """

    predicted_props = demographic_model.proportions_from_matrices(model_func(demographic_model.parameter_handler.convert_to_optimizer_params(optimal_params)))
    predicted_autosome_props = {k: v for k, v in predicted_props.items() if "autosomal" in k.lower()}
    predicted_allosome_props = {k: v for k, v in predicted_props.items() if "autosomal" not in k.lower()}

    autosome_values = None
    allosome_values = None

    if predicted_autosome_props:
        autosome_key = sorted(predicted_autosome_props.keys())[0]
        autosome_values = np.asarray(predicted_autosome_props[autosome_key])
        predicted_autosome_message = f"Predicted autosome proportions: {np.array2string(autosome_values, separator=' ')}"
        print(predicted_autosome_message)
        logger.info(predicted_autosome_message)

    if predicted_allosome_props:
        allosome_key = sorted(predicted_allosome_props.keys())[0]
        allosome_values = np.asarray(predicted_allosome_props[allosome_key])
        predicted_allosome_message = f"Predicted allosome proportions: {np.array2string(allosome_values, separator=' ')}"
        print(predicted_allosome_message)
        logger.info(predicted_allosome_message)
    
    return autosome_values, allosome_values


def check_final_parameters(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, optimal_params: np.ndarray):
    """
    Checks that the final optimal parameters are compatible will well-defined migration matrices.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to check the final optimal parameters.
    optimal_params: np.ndarray
        The final optimal parameters in physical units, as returned by the optimization process.
    """
    class _WarnCapture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[str] = []
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                self.records.append(record.getMessage())

    _dem_logger_check = logging.getLogger("tracts.demography.base_parametrized_demography")
    _capture_handler = _WarnCapture()
    _dem_logger_check.addHandler(_capture_handler)
    try:
        _ = demographic_model.get_violation_score(optimal_params, verbose=True)
        # get_violation_score calls get_migration_matrices internally, which logs the warning.
        # We capture it here so it can be shown as a user-visible printed message.
    except Exception as e:
        logger.warning(f"Could not compute post-optimization diagnostics: {e}")
    finally:
        _dem_logger_check.removeHandler(_capture_handler)

    if any("Founding migration rates add up to more than 1" in msg for msg in _capture_handler.records):
        founding_rate_msg = (
            "Warning: the final optimal parameters have founding migration rates that add up "
            "to more than 1. This means that no valid combination of migration rates exists "
            "for these parameter values, and the model result may be unreliable."
        )
        print(founding_rate_msg)
        logger.warning(founding_rate_msg)


def _default_param_bounds(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, param) -> tuple[float, float]:
    """
    Returns the default (pre-narrowing) admissible bounds for ``param``, i.e. the bounds it would
    have without any user-specified narrowing from the driver's ``bounds`` section. These come
    from the parameter's type (see ``tracts.demography.parameter.ParamType``), except TIME
    parameters, whose default bounds are the model's ``(min_time, max_time)`` (see
    ``BaseParametrizedDemography.add_parameter``).
    """
    if param.type == ParamType.TIME:
        return (demographic_model.min_time, demographic_model.max_time)
    return param.type.bounds


def check_optimal_params_near_bounds(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased,
                                     optimal_params: np.ndarray, tol: float) -> list[str]:
    """
    Checks whether any of the final optimal parameters is close to a *user-narrowed* admissible
    bound (see ``bounds``/``parse_param_bounds``), which may indicate that the true optimum lies
    outside the range the user specified. If so, prints/logs a message recommending a re-run
    starting from these optimal values (see ``start_params``) with the admissible range widened
    for the affected parameter(s) (see ``bounds``).

    Only bounds that the user explicitly narrowed below their default (type-determined) value are
    checked, and only on the narrowed side: a parameter sitting at its natural type boundary (e.g.
    a sex-bias parameter landing at +-1, or a time parameter at its default lower bound) is *not*
    flagged, since that boundary was not something the user restricted. (A sex-bias parameter at a
    +-1 boundary is instead handled by the boundary re-optimization, see
    ``check_optimal_sex_bias_parameters_at_boundaries``.) Closeness is relative to the parameter's
    (narrowed) admissible range (``upper - lower``), not absolute, since parameters can differ by
    orders of magnitude in scale.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model whose parameters' bounds are checked against.
    optimal_params: np.ndarray
        The final optimal parameters in physical units, as returned by the optimization process,
        in the order given by ``demographic_model.model_base_params``.
    tol: float
        Relative tolerance, as a fraction of a parameter's admissible range, within which a value
        counts as "close" to a bound. See ``OptimizationConfig.bounds_proximity_tol``.

    Returns
    -------
    list[str]
        The names of the parameters found close to a user-narrowed bound (possibly empty).
    """
    near_bound_params = []
    for param_name, value in zip(demographic_model.model_base_params, optimal_params):
        param = demographic_model.model_base_params[param_name]
        lower, upper = param.bounds
        default_lower, default_upper = _default_param_bounds(demographic_model, param)

        # Only consider a side the user actually narrowed below the default (and hence made finite).
        lower_narrowed = np.isfinite(lower) and lower > default_lower
        upper_narrowed = np.isfinite(upper) and upper < default_upper
        if not (lower_narrowed or upper_narrowed):
            continue

        # A user-narrowed side implies both bounds are finite (bounds are given as ``min:max``),
        # so the admissible range is finite here.
        margin = tol * (upper - lower)
        near_lower = lower_narrowed and (value - lower < margin)
        near_upper = upper_narrowed and (upper - value < margin)
        if near_lower or near_upper:
            near_bound_params.append(param_name)

    if near_bound_params:
        near_bound_msg = (
            f"The optimal value(s) of parameter(s) {', '.join(near_bound_params)} are close to "
            "the admissible bounds specified by the user. This may mean the true optimum lies outside the "
            "specified range. Consider re-running the optimization starting from these optimal "
            "values (see start_params) after widening the admissible bounds (see bounds) for these "
            "parameter(s).\n"
        )
        _print_and_log(near_bound_msg)

    return near_bound_params


def _print_optimal_values_and_likelihood(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, 
                                        optimal_params: np.ndarray, optimal_likelihood: float, 
                                        remainder_parameters: dict[str, float], ad_model_allosomes: bool | None):
    
    """
    Prints the final optimal parameter values and the corresponding likelihood, along with any derived parameters for the remainder (dependent) ancestry.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to print the optimal parameter values and likelihood.
    optimal_params: np.ndarray
        The final optimal parameters in physical units, as returned by the optimization process.
    optimal_likelihood: float
        The likelihood corresponding to the final optimal parameters.
    remainder_parameters: dict[str, float]
        A dictionary of derived parameters for the 'remainder' (dependent) ancestry in each parametrized population, as computed by ``compute_remainder_params``.
    ad_model_allosomes: bool | None
        A boolean indicating whether allosomal admixture is modelled. If None, only autosomal admixture is modelled. This affects the message printed about the data used for computing the likelihood.
    """

    final_data = "autosomal + allosomal" if ad_model_allosomes is not None else "autosomal"
    final_message = f"Final parameters and corresponding likelihood computed on {final_data} data:"
    param_names = list(demographic_model.model_base_params.keys())

    all_param_names = param_names + list(remainder_parameters.keys())
    all_param_values = list(optimal_params) + list(remainder_parameters.values())
    param_col_widths = [max(len(name), 12) for name in all_param_names]
    header = f"{'LogLik':>12} | " + " | ".join(
        f"{name:>{w}}" for name, w in zip(all_param_names, param_col_widths)
    )
    line = "-" * len(header)
    print("\n" + final_message)
    for l in (line, header, line):
        print(l)
        logger.info(l)

    values_str = " | ".join(
        f"{x:>{w}.4g}" for x, w in zip(all_param_values, param_col_widths)
    )
    loglik_message = f"{float(optimal_likelihood):>12.6g} | {values_str}"
    logger.info(loglik_message)
    print(loglik_message)
    print(line)

    # Report derived parameters for the remainder (dependent) ancestry.
    if remainder_parameters:
        dep_msg = f"Parameters {', '.join(remainder_parameters.keys())} correspond to the dependent ancestry and were not free in the optimization."
        print(dep_msg)
        logger.info(dep_msg)

def has_free_sex_bias_parameters(parameter_handler: FixedParametersHandler, sex_bias_param_names: list[str]):
    """
    Checks whether any sex-bias parameters are free (not fixed by ancestry proportions or value) in the demographic model.

    Parameters
    ----------
    parameter_handler: FixedParametersHandler
        The parameter handler for the demographic model.
    sex_bias_param_names: list[str]
        A list of parameter names corresponding to sex-bias parameters
    
    Returns
    -------
    bool
        True if any sex-bias parameters are free, False otherwise.
    """

    fixed_sex_bias_params = (
        set(parameter_handler.params_fixed_by_ancestry)
        | set(parameter_handler.user_params_fixed_by_value.keys())
    )
    return any(name not in fixed_sex_bias_params for name in sex_bias_param_names)


def _get_driver_for_reoptimization(driver_spec: InferenceConfig, model_param_names: list[str], optimal_params: np.ndarray):

    """
    Creates a new driver specification for re-optimization, using the optimal parameters from a previous optimization as the starting parameters for the new optimization.
    The new driver specification is a copy of the original driver specification, with the starting parameters updated to the optimal parameters and the number of repetitions set to 1,
    so that no resampling of starting parameters is peformed.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file.
    model_param_names: list[str]
        A list of all parameter names of the demographic model, in the order given by demographic_model.model_base_params.
    optimal_params: np.ndarray
        The final optimal parameters in physical units, as returned by the optimization process.
    
    Returns
    -------
    InferenceConfig
        A new driver specification for re-optimization, with the starting parameters updated to the optimal parameters and the number of repetitions set to 1.
    """
    reopt_start_params_config = driver_spec.start_params.model_copy(update={
                                    name: float(value) for name, value in zip(model_param_names, optimal_params)
                                    })
            
    reopt_driver_spec = driver_spec.model_copy(update={
                                "start_params": reopt_start_params_config,
                                "optim": driver_spec.optim.model_copy(update={"repetitions": 1}),
                                })

    return reopt_driver_spec


def check_optimal_sex_bias_parameters_at_boundaries(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, driver_spec: InferenceConfig,
                                           sex_bias_param_names: list[str], remainder_params: dict[str, float], optimal_params: np.ndarray):

    """
    Checks whether the optimal sex-bias parameters have values at the border of the feasible region, up to a pre-specified tolerance.
    If any sex-bias parameter has an optimal value at the border, the function raises a warning to suggest a re-run with 
    fixed values.

    Parameters
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model for which to check the optimal parameters.
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file.
    sex_bias_param_names: list[str]
        A list of parameter names corresponding to sex-bias parameters in the demographic model.
    remainder_params: dict[str, float]
        A dictionary of derived parameters for the 'remainder' (dependent) ancestry in each parametrized population, as computed by ``compute_remainder_params``.
    optimal_params: np.ndarray
        The final optimal parameters in physical units, as returned by the optimization process.

    Returns
    -------
    list[str]
        A list of parameter names that have optimal values at the boundary of the feasible region.
    """    
    param_names = list(demographic_model.model_base_params.keys())    
    all_param_names = param_names + list(remainder_params.keys())
    all_param_values = list(optimal_params) + list(remainder_params.values())
     
    # Detect optimal sex-bias parameters at boundaries (explicitely optimized)
    unfixed_sex_bias_params_at_boundaries = {_param_name: _param_value for _param_name, _param_value in zip(all_param_names, all_param_values)
                            if _param_name in sex_bias_param_names and 
                            _param_name not in demographic_model.parameter_handler.params_fixed_by_ancestry and
                            _param_name not in demographic_model.parameter_handler.user_params_fixed_by_value.keys() and
                            (np.isclose(_param_value, 1.0, atol = driver_spec.optim.boundary_tol) or np.isclose(_param_value, -1.0, atol = driver_spec.optim.boundary_tol))}
    
    # Detect derived remainder sex-bias parameters at boundaries
    remainder_sex_bias_at_boundaries = {_param_name: _param_value for _param_name, _param_value in remainder_params.items()
                            if _param_name.endswith("_sex_bias") and
                            (np.isclose(_param_value, 1.0, atol = driver_spec.optim.boundary_tol) or np.isclose(_param_value, -1.0, atol = driver_spec.optim.boundary_tol))}

    all_at_boundary = list(unfixed_sex_bias_params_at_boundaries) + list(remainder_sex_bias_at_boundaries)
    if all_at_boundary:
        _boundary_msg = (
            f"The optimal solution has sex-bias parameter(s) "
            f"{', '.join(all_at_boundary)} near their \u00b11 boundary. "
            )
        if not driver_spec.optim.rerun_optimization_on_boundaries:
            _boundary_msg+="Re-running the optimization fixing these parameters near their boundary values may yield a better solution. Consider setting driver_spec.optim.rerun_optimization_on_boundaries to TRUE."
        print(_boundary_msg)
        logger.info(_boundary_msg)
    
    return all_at_boundary


def _get_founder_event_with_remainder(demographic_model: ParametrizedDemographySexBiased, population: str):
    """
    Returns the male-suffixed founder event for ``population`` (e.g. ``"X_male"``), or None if
    there is no such founder event or it has no remainder population (i.e. it is a continuous
    founder event, or every source population's proportion is explicitly parametrized).
    """
    founder_event = demographic_model.founder_events.get(f"{population}{SexType.MALE.suffix}")
    if founder_event is None or founder_event.remainder_population is None:
        return None
    return founder_event


def get_alternate_implicit_population(demographic_model: ParametrizedDemographySexBiased,
                                      optimal_sex_bias_at_boundaries: list[str]) -> str | None:
    """
    Checks whether any of the boundary-violating sex-bias parameter names in
    ``optimal_sex_bias_at_boundaries`` is a derived remainder parameter (i.e. corresponds to
    ``demographic_model``'s current implicit population, rather than a directly-optimized
    sex-bias parameter). If so, returns the name of a different source population from the
    same founder event, whose own sex-bias parameter is neither itself at a boundary nor
    already fixed by value, that could be used as the implicit population instead.

    A candidate already fixed by value is excluded even though it is not "at a boundary" (fixed
    parameters are skipped by ``check_optimal_sex_bias_parameters_at_boundaries``, so they never
    appear in ``optimal_sex_bias_at_boundaries``): switching the implicit population to it would
    silently discard its fixed value (fixed parameters become non-optimizable once a population
    is implicit; see ``base_parametrized_demography.set_up_fixed_parameters``) and, if it was
    fixed by a previous boundary re-optimization iteration, would simply undo that iteration's
    fix rather than making progress, risking an infinite switch-back-and-forth cycle.

    Parameters
    ----------
    demographic_model: ParametrizedDemographySexBiased
        The demographic model whose founder events are inspected.
    optimal_sex_bias_at_boundaries: list[str]
        Parameter names near their +-1 boundary, as returned by
        ``check_optimal_sex_bias_parameters_at_boundaries``.

    Returns
    -------
    str | None
        The name of an alternate source population to use as the implicit population, or
        None if no boundary-violating parameter corresponds to the current implicit
        population, or if every other source population in that founder event is either
        at a boundary or already fixed by value.
    """
    for population in demographic_model.parametrized_populations:
        founder_event = _get_founder_event_with_remainder(demographic_model, population)
        if founder_event is None:
            continue

        remainder_key = f"{population}_{founder_event.remainder_population}_sex_bias"
        if remainder_key not in optimal_sex_bias_at_boundaries:
            continue

        user_params_fixed_by_value = demographic_model.parameter_handler.user_params_fixed_by_value
        for source_population, rate_param in founder_event.source_populations.items():
            # rate_param is sex-suffixed here (e.g. "REUR_male", from the male founder event's
            # own source_populations); strip the suffix to match the plain sex-bias parameter
            # names in optimal_sex_bias_at_boundaries (e.g. "REUR_sex_bias").
            base_rate_param = rate_param.removesuffix(SexType.MALE.suffix)
            candidate_sex_bias = f"{base_rate_param}_sex_bias"
            if candidate_sex_bias not in optimal_sex_bias_at_boundaries and candidate_sex_bias not in user_params_fixed_by_value:
                return source_population

        _boundary_msg = (
            f"The implicit population's sex-bias parameter "
            f"({remainder_key}) is near the +-1 boundary,\nbut every other source population in the same "
            "founder event either also has its sex-bias parameter \nat a boundary or is already fixed "
            "by value: no alternate implicit population can be chosen."
        )
        print(_boundary_msg)
        logger.info(_boundary_msg)
        return None

    return None


def _get_params_for_newly_explicit_population(old_demographic_model: ParametrizedDemographySexBiased,
                                              new_demographic_model: ParametrizedDemographySexBiased,
                                              remainder_params: dict[str, float], near_one: float) -> tuple[dict[str, float], dict[str, float]]:
    """
    When switching the implicit population, the population that was previously implicit becomes an
    explicit, directly-optimized parameter in the new demographic model, with no starting value
    specified in the driver file (it was never optimized before, so the user never had to provide
    one), taken from the optimal remainder values computed at the end of the previous optimization
    (see ``compute_remainder_params``).

    The rate is only used as a starting value (it was not itself flagged as a boundary hit, so it
    remains free to be optimized). The sex-bias parameter, however, is the very one whose derived
    (remainder) value was at the +-1 boundary that triggered this implicit-population switch in
    the first place: per ``run_boundary_reoptimization``'s "all boundary-hitting parameters must be
    fixed by value" rule, it must be fixed in the new model rather than left free to be
    resampled/re-optimized from scratch, at ``+-near_one`` rather than its actual (possibly less
    extreme) boundary value.

    Parameters
    ----------
    old_demographic_model: ParametrizedDemographySexBiased
        The demographic model before the implicit-population switch.
    new_demographic_model: ParametrizedDemographySexBiased
        The demographic model after the implicit-population switch (freshly reloaded).
    remainder_params: dict[str, float]
        The remainder (derived) parameters computed from the previous optimization's optimal
        parameters, as returned by ``compute_remainder_params``.
    near_one: float
        The value (just under 1) to fix the newly-explicit sex-bias parameter at, signed to match
        which boundary (+1 or -1) it was at. See ``InferenceConfig.optim.near_one``.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        A pair ``(start_param_updates, fixed_param_values)``: a dict mapping the newly-explicit
        rate parameter name to its starting value, and a dict fixing the newly-explicit sex-bias
        parameter name at ``+-near_one``.
    """
    start_param_updates = {}
    fixed_param_values = {}
    for population in old_demographic_model.parametrized_populations:
        old_founder_event = _get_founder_event_with_remainder(old_demographic_model, population)
        if old_founder_event is None:
            continue

        old_remainder_population = old_founder_event.remainder_population
        new_founder_event = new_demographic_model.founder_events.get(f"{population}{SexType.MALE.suffix}")
        if new_founder_event is None:
            continue

        new_rate_param = new_founder_event.source_populations.get(old_remainder_population)
        if new_rate_param is None:
            continue
        new_rate_param = new_rate_param.removesuffix(SexType.MALE.suffix)

        rate_value = remainder_params.get(f"{population}_{old_remainder_population}_rate")
        if rate_value is not None:
            start_param_updates[new_rate_param] = float(rate_value)

        sex_bias_value = remainder_params.get(f"{population}_{old_remainder_population}_sex_bias")
        if sex_bias_value is not None and not np.isnan(sex_bias_value):
            fixed_param_values[f"{new_rate_param}_sex_bias"] = float(np.sign(sex_bias_value)) * near_one

    return start_param_updates, fixed_param_values


def compute_physical_start_params(driver_spec: InferenceConfig, demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased,
                                  sex_bias_param_names: list[str], non_sex_bias_param_names: list[str],
                                  step_label: str = "step 1") -> list[np.ndarray]:
    """
    Computes physical starting parameters to optimize from. When
    ``driver_spec.optim.two_steps_optimization`` is True, only non-sex-bias parameters are
    sampled (sex-bias parameters are fixed at the midpoint of their admissible bounds -- 0 with
    the default ``(-1, 1)`` bounds, or the midpoint of a user-narrowed range that may exclude 0,
    see ``bounds``; or at any user-provided fixed value), matching step 1's parameter subset, and
    identical starting points are collapsed into one (see ``collapse_identical_start_params``).
    Otherwise, all parameters are sampled/fixed according to
    ``driver_spec.optim.fix_parameters_by_value`` alone.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The driver-file configuration controlling repetitions/seed/two_steps_optimization/
        fix_parameters_by_value.
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model to sample starting parameters for.
    sex_bias_param_names: list[str]
        Names of the sex-bias parameters among the demographic model's free base parameters.
    non_sex_bias_param_names: list[str]
        Names of the non-sex-bias parameters among the demographic model's free base parameters.
    step_label: str
        Label used when collapsing identical two-step starting parameters into one. Defaults to
        "step 1".

    Returns
    -------
    list[np.ndarray]
        The physical starting parameters to optimize from.
    """
    if driver_spec.optim.two_steps_optimization:
        # Fix free sex-bias parameters at the midpoint of their admissible bounds for step 1's
        # non-sex-bias optimization. With the default sex-bias bounds (-1, 1) the midpoint is 0
        # (the historical default); with user-narrowed bounds (see ``bounds``) that exclude 0, the
        # midpoint keeps the fixed value inside the admissible range, so feasible starting
        # parameters can still be sampled.
        midpoint_sex_bias = {
            name: float(np.mean(demographic_model.model_base_params[name].bounds))
            for name in sex_bias_param_names
            if name not in driver_spec.optim.fix_parameters_by_value.keys()
        }
        physical_start_params = parse_start_params(
            start_param_bounds=driver_spec.start_params,
            repetitions=driver_spec.optim.repetitions,
            seed=driver_spec.optim.seed,
            demographic_model=demographic_model,
            sample_param_names=set(non_sex_bias_param_names),
            fixed_param_values=midpoint_sex_bias | driver_spec.optim.fix_parameters_by_value,
        )
        return collapse_identical_start_params(physical_start_params, step_label)

    return parse_start_params(
        start_param_bounds=driver_spec.start_params,
        repetitions=driver_spec.optim.repetitions,
        seed=driver_spec.optim.seed,
        demographic_model=demographic_model,
        fixed_param_values=driver_spec.optim.fix_parameters_by_value,
    )


def build_boundary_reoptimization_model(driver_spec: InferenceConfig, reload_context: ModelReloadContext,
                                        boundary_fixed_param_values: dict[str, float], genetic_model: GeneticModel,
                                        optimal_params: np.ndarray, remainder_params: dict[str, float],
                                        alternate_implicit_population: str | None = None):
    """
    Builds a genetic model and driver specification identical to the given ones, except with
    the sex-bias parameters in ``boundary_fixed_param_values`` now fixed by value, and (if given)
    ``alternate_implicit_population`` as the implicit population instead of the current one. Fixed
    parameters and starting parameters are set up accordingly. Used to retry the optimization when
    one or more sex-bias parameters have an optimal value at a +-1 boundary.

    As with ``_get_driver_for_reoptimization``, all starting parameters are pinned to their
    previous optimal value (no resampling) and repetitions are set to 1: this re-optimization
    should resume from where the previous one left off, not restart from a fresh random point.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The original driver-file configuration.
    reload_context: ModelReloadContext
        File-location and ancestry-proportion context needed to reload the demographic model from
        the model YAML file, used only when ``alternate_implicit_population`` is given.
    boundary_fixed_param_values: dict[str, float]
        Directly-optimized sex-bias parameter names (and their boundary value) to fix by value,
        merged into ``driver_spec.optim.fix_parameters_by_value``.
    genetic_model: GeneticModel
        The current genetic model. When ``alternate_implicit_population`` is None, a deep copy of
        this genetic model is reused instead of reloading the demographic model from the model
        YAML file (its parameter names are derived directly from
        ``genetic_model.demographic_model``, which is unchanged in that case). When
        ``alternate_implicit_population`` is given, only its ``phase_type_config`` is reused (the
        demographic model must be reloaded, since which population is implicit is baked into the
        founder events at YAML-parse time).
    optimal_params: np.ndarray
        The current optimal parameters (physical units), before this re-optimization, in the order
        given by ``genetic_model.demographic_model``'s free base parameters. Used to pin the
        starting parameters of the re-optimization (see ``_get_driver_for_reoptimization``).
    remainder_params: dict[str, float]
        The remainder (derived) parameters computed from the previous optimization's optimal
        parameters, as returned by ``compute_remainder_params``. Used, when
        ``alternate_implicit_population`` is given, to set a starting value for the rate, and fix
        by value the sex-bias parameter, of the population that was previously implicit and is now
        explicit (see ``_get_params_for_newly_explicit_population``).
    alternate_implicit_population: str | None
        The name of the population to use as the implicit population instead of the current one.
        If None, the current implicit population is kept. Defaults to None.

    Returns
    -------
    tuple[InferenceConfig, GeneticModel, list[str], list[str], list[str], list[np.ndarray]]
        The new driver spec; the new genetic model built from it; the model, sex-bias, and
        non-sex-bias parameter names; and the physical starting parameters to optimize from.
    """

    all_fixed_param_values = dict(boundary_fixed_param_values)

    model_param_names_before_reload, _, _ = get_param_names_by_type(genetic_model.demographic_model)
    optimal_param_values = dict(zip(model_param_names_before_reload, (float(v) for v in optimal_params)))

    reopt_driver_spec = _get_driver_for_reoptimization(driver_spec=driver_spec,
                                                       model_param_names=model_param_names_before_reload,
                                                       optimal_params=optimal_params)

    models_update = {}
    if alternate_implicit_population is not None:
        models_update["implicit_population"] = alternate_implicit_population

    reopt_driver_spec = reopt_driver_spec.model_copy(update={
        "models": reopt_driver_spec.models.model_copy(update=models_update),
        "optim": reopt_driver_spec.optim.model_copy(update={
            "fix_parameters_by_value": reopt_driver_spec.optim.fix_parameters_by_value | boundary_fixed_param_values,
        }),
    })

    if alternate_implicit_population is not None:
        # Changing the implicit population changes which parameters are explicit vs. derived, so
        # the demographic model must be reloaded from the model YAML file, and its fixed
        # parameters (ancestry- and value-based) set up from scratch.
        demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names = load_demographic_model_from_driver(driver_spec=reopt_driver_spec,
                                                                                                                                  script_dir=reload_context.script_dir,
                                                                                                                                  driver_path=reload_context.driver_path,
                                                                                                                                  allosome_label=reload_context.allosome_label)
        reopt_genetic_model = GeneticModel(demographic_model=demographic_model,
                                           phase_type_config=genetic_model.phase_type_config)

        # Switching the implicit population can turn a parameter that was previously explicit and
        # fixed by value (e.g. by an earlier boundary re-optimization iteration) into a derived
        # remainder parameter of the newly-implicit population, which can no longer be fixed by
        # value (it is computed automatically). Drop any such now-stale entries before setting up
        # the reloaded model's fixed parameters, to avoid a spurious KeyError.
        stale_fixed_by_value = set(reopt_driver_spec.optim.fix_parameters_by_value) - set(demographic_model.model_base_params)
        if stale_fixed_by_value:
            reopt_driver_spec = reopt_driver_spec.model_copy(update={
                "optim": reopt_driver_spec.optim.model_copy(update={
                    "fix_parameters_by_value": {
                        name: value for name, value in reopt_driver_spec.optim.fix_parameters_by_value.items()
                        if name not in stale_fixed_by_value
                    },
                }),
            })

        # The population that was previously implicit is now explicit and has no starting value in
        # the driver file (it was never optimized before). Seed its rate from the remainder values
        # computed at the end of the previous optimization, and fix its sex-bias parameter by value
        # at +-near_one: it was the derived sex-bias hitting the +-1 boundary that triggered this
        # implicit-population switch, so it must be fixed, not left free to be resampled/re-optimized
        # from scratch.
        new_explicit_start_params, new_explicit_fixed_param_values = _get_params_for_newly_explicit_population(
            old_demographic_model=genetic_model.demographic_model,
            new_demographic_model=demographic_model,
            remainder_params=remainder_params,
            near_one=driver_spec.optim.near_one,
        )
        if new_explicit_start_params:
            reopt_driver_spec = reopt_driver_spec.model_copy(update={
                "start_params": reopt_driver_spec.start_params.model_copy(update=new_explicit_start_params),
            })
        if new_explicit_fixed_param_values:
            reopt_driver_spec = reopt_driver_spec.model_copy(update={
                "optim": reopt_driver_spec.optim.model_copy(update={
                    "fix_parameters_by_value": reopt_driver_spec.optim.fix_parameters_by_value | new_explicit_fixed_param_values,
                }),
            })
            all_fixed_param_values.update(new_explicit_fixed_param_values)

        optimal_param_values.update(new_explicit_start_params)
        optimal_param_values.update(new_explicit_fixed_param_values)

        # reload_context.autosome_proportions/allosome_proportions were computed against the old
        # demographic model's population order; realign them to the new model's order.
        reload_autosome_proportions, reload_allosome_proportions = _reorder_ancestry_proportions(
            old_ancestor_labels=list(genetic_model.demographic_model.population_indices.keys()),
            new_ancestor_labels=list(demographic_model.population_indices.keys()),
            autosome_proportions=reload_context.autosome_proportions,
            allosome_proportions=reload_context.allosome_proportions)

        setup_fixed_parameters(driver_spec=reopt_driver_spec,
                               demographic_model=demographic_model,
                               allosome_label=reload_context.allosome_label,
                               autosome_proportions=reload_autosome_proportions,
                               allosome_proportions=reload_allosome_proportions,
                               print_details=False)
    else:
        # No structural change: reuse a copy of the current genetic model (which already has the
        # original ancestry- and value-based fixed parameters set up) instead of reloading, and
        # just add the new boundary fixes on top. Population order is unchanged, so
        # reload_context's proportions (kept in sync with genetic_model's order by the caller)
        # can be used as-is, without reordering.
        #
        # Uses setup_fixed_parameters (which re-derives user_params_fixed_by_value from
        # reopt_driver_spec.optim.fix_parameters_by_value, already merged with
        # boundary_fixed_param_values above) rather than parameter_handler.add_fixed_parameters:
        # the latter only updates current_fixed_parameters, which has_free_sex_bias_parameters
        # and check_optimal_sex_bias_parameters_at_boundaries do not consult, and this fix must
        # be permanent (unlike the transient step-1/step-2 fixing done by add_fixed_parameters in
        # core.py, which is later released again within the same optimization call).
        reopt_genetic_model = genetic_model.copy()
        demographic_model = reopt_genetic_model.demographic_model
        setup_fixed_parameters(driver_spec=reopt_driver_spec,
                               demographic_model=demographic_model,
                               allosome_label=reload_context.allosome_label,
                               autosome_proportions=reload_context.autosome_proportions,
                               allosome_proportions=reload_context.allosome_proportions,
                               print_details=False)
        model_param_names, sex_bias_param_names, non_sex_bias_param_names = get_param_names_by_type(demographic_model)

    _print_and_log("\nRe-optimizing with sex-bias parameter(s) "
                    f"{', '.join(all_fixed_param_values) or '(none)'} fixed at their boundary value"
                    + (f", and '{alternate_implicit_population}' as the implicit population" if alternate_implicit_population is not None else "")
                    + "."
                   )

    # Unlike compute_physical_start_params (used for the initial optimization, where there is no
    # previous run to carry values from), this re-optimization must start from the previous
    # optimization's result as-is: no sex-bias parameter is reset to 0, and no parameter is
    # resampled.
    physical_start_params = [np.array([optimal_param_values[name] for name in model_param_names])]

    return reopt_driver_spec, reopt_genetic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names, physical_start_params


# ---------- Conversion between optimizer and physical parameters ---------

def get_time_scaled_model_func(demographic_model: ParametrizedDemography) -> Callable[[np.ndarray], dict[str, np.ndarray]]:
    """
    Computes a function that takes in optimizer parameters, converts them to physical parameters using the model's parameter handler, and returns the migration matrices for those parameters.
    This is necessary because some optimizers may require parameters to be on a different scale (e.g. log scale) than the physical parameters used in the model, so this function serves as a wrapper to apply the necessary transformations before passing parameters to the model.
    
    Parameters
    ----------
    demographic_model: ParametrizedDemography
        The demographic model for which to compute the migration matrices.

    Returns
    -------
    Callable[[np.ndarray], dict[str, np.ndarray]]
        A function that takes in optimizer parameters, converts them to physical parameters, and returns the migration matrices for those parameters.
    """
    return lambda params: demographic_model.get_migration_matrices(demographic_model.parameter_handler.convert_to_physical_params(params))


def get_time_scaled_model_bounds(demographic_model: ParametrizedDemography, verbose = False):
    """
    Computes a function that takes in optimizer parameters, converts them to physical parameters using the model's parameter handler, and returns the violation score for those parameters.
    This is necessary because some optimizers may require parameters to be on a different scale (e.g. log scale) than the physical parameters used in the model, so this function serves as a wrapper to apply the necessary transformations before passing parameters to the model.
    
    Parameters
    ----------
    demographic_model: ParametrizedDemography
        The demographic model for which to compute the violation score.
    verbose: bool
        Whether to print detailed information about the violation score. Defaults to False.

    Returns
    -------
    Callable[[np.ndarray], float]
        A function that takes in optimizer parameters, converts them to physical parameters, and returns the violation score for those parameters.
    """
    return lambda params: demographic_model.get_violation_score(demographic_model.parameter_handler.convert_to_physical_params(params), verbose = verbose)


def scale_select_indices(arr, indices_to_scale, scaling_factor=1):
    if len(indices_to_scale) != len(arr):
        raise ValueError(
            f'Length of array ({len(arr)}) was not equal to length of indices_to_scale ({len(indices_to_scale)}).')
    return (np.multiply(indices_to_scale, scaling_factor - 1) + 1) * arr



# --------------- Output production ---------------

def compute_remainder_params(demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased, migration_matrices: dict) -> dict:
    r"""
    Compute derived parameters for the 'remainder' (dependent) ancestry in
    each parametrized population.

    For a demographic model with *n* free rate parameters ``R_1, ..., R_{n-1}`` and a
    remainder ancestry whose rate is ``1 - R_1 - ... - R_{n-1}``, the
    remainder rate is read directly from the founding row of the migration
    matrix.  For :class:`~tracts.demography.parametrized_demography_sex_biased.ParametrizedDemographySexBiased`
    models, the sex bias of the remainder ancestry is additionally derived from
    the constraint that male and female founding rates must each sum to 1:

    .. math::

        r_k^{\\text{male/female}} = 1 - \\sum_{i \\neq k} r_i^{\\text{male/female}}

        s_k = \\frac{r_k^{\\text{female}} - r_k^{\\text{male}}}
                     {2\\,\\min(r_k,\\,1-r_k)}

    Parameters
    ----------
    demographic_model : ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model.  Only these two types are accepted; any other
        type raises a :class:`TypeError`.
    migration_matrices : dict
        Migration matrices as returned by ``demographic_model.get_migration_matrices()``.
        For :class:`~tracts.demography.parametrized_demography_sex_biased.ParametrizedDemographySexBiased`
        models the keys are ``'{population}_male'`` / ``'{population}_female'``;
        for :class:`~tracts.demography.parametrized_demography.ParametrizedDemography`
        models the key is the population name directly.

    Returns
    -------
    dict[str, float]
        For each parametrized population that has a remainder ancestry:

        * ``'{dest_pop}_{remainder_pop}_rate'`` — always present; the mean founding rate
          (average of male and female for sex-biased models, direct value
          otherwise).
        * ``'{dest_pop}_{remainder_pop}_sex_bias'`` — only present for
          :class:`~tracts.demography.parametrized_demography_sex_biased.ParametrizedDemographySexBiased`
          models; ``nan`` when the remainder rate is 0 or 1.

        Returns an empty dict when the model has no remainder population or when
        *demographic_model* is not a recognised demography type (e.g. a test stub).
    """
    if not isinstance(demographic_model, (ParametrizedDemographySexBiased, ParametrizedDemography)):
        return {}
    is_sex_biased = isinstance(demographic_model, ParametrizedDemographySexBiased)

    result = {}
    seen = set()
    for population in demographic_model.parametrized_populations:
        if population in seen:
            continue
        seen.add(population)

        if is_sex_biased:
            event_key = f'{population}{SexType.MALE.suffix}'   # e.g. 'X_male'
        else:
            event_key = population                              # e.g. 'X'

        founder_event = demographic_model.founder_events.get(event_key)
        if founder_event is None or founder_event.remainder_population is None:
            continue

        remainder_pop = founder_event.remainder_population
        if remainder_pop not in demographic_model.population_indices:
            continue

        remainder_col = demographic_model.population_indices[remainder_pop]

        if is_sex_biased:
            male_matrix  = migration_matrices[f'{population}{SexType.MALE.suffix}']
            female_matrix = migration_matrices[f'{population}{SexType.FEMALE.suffix}']
            # Founding row is the last row of each migration matrix.
            r_male   = float(male_matrix[-1, remainder_col])
            r_female = float(female_matrix[-1, remainder_col])
            r_mean   = (r_male + r_female) / 2.0
            result[f'{population}_{remainder_pop}_rate'] = r_mean
            denom    = 2.0 * min(r_mean, 1.0 - r_mean)
            sex_bias = (r_female - r_male) / denom if abs(denom) > 1e-10 else float('nan')
            result[f'{population}_{remainder_pop}_sex_bias'] = sex_bias
        else:
            matrix = migration_matrices[population]
            result[f'{population}_{remainder_pop}_rate'] = float(matrix[-1, remainder_col])

    return result


def _plot_migration_matrices(migration_matrix_f: np.ndarray, migration_matrix_m: np.ndarray, pop_labels: list, output_path: str):

    mean_matrix = (migration_matrix_f[1:,:] + migration_matrix_m[1:,:]) / 2
    denom_matrix = 2 * np.minimum(mean_matrix, 1 - mean_matrix)
    safe_denom = np.where(denom_matrix > 1e-10, denom_matrix, np.nan)
    sex_bias_matrix = (migration_matrix_f[1:,:] - migration_matrix_m[1:,:]) / safe_denom

    # Add migration rate and sex-bias for the admixed population
    admixed_rate = 1 - np.sum(mean_matrix, axis = 1) # 1 - \sum_{i}R_i
    admixed_denom = 2 * np.minimum(admixed_rate, 1 - admixed_rate)
    safe_admixed_denom = np.where(admixed_denom > 1e-10, admixed_denom, np.nan)
    admixed_sex_bias = -np.nansum(denom_matrix * sex_bias_matrix, axis = 1) / safe_admixed_denom # (R^f_x - R^m_x) = -\sum_{i != x}(R^f_i - R^m_i)
    mean_matrix = np.concatenate([mean_matrix, admixed_rate[:, np.newaxis]], axis = 1)
    sex_bias_matrix = np.concatenate([sex_bias_matrix, admixed_sex_bias[:, np.newaxis]], axis = 1)
    pop_labels = pop_labels + ['Admixed']

    n_rows, n_cols = mean_matrix.shape

    fig = Figure()
    ax1, ax2 = fig.subplots(1, 2)

    # Adaptive fonts
    font_scale = max(7, min(12, 12 - 0.4 * n_cols))
    tick_font = max(6, min(10, 10 - 0.3 * n_cols))
    annot_font = max(5, min(9, 9 - 0.3 * max(n_rows, n_cols)))

    x_ticks = np.arange(n_cols)
    y_ticks = np.arange(n_rows)

    # Colormaps
    cmap_mean = LinearSegmentedColormap.from_list("white_green", ["white", "green"])
    cmap_bias = LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])
    norm_bias = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # Panel 1: Mean matrix
    im1 = ax1.imshow(mean_matrix, cmap=cmap_mean, vmin=0, vmax=1, aspect="auto")
    ax1.set_box_aspect(mean_matrix.shape[0] / mean_matrix.shape[1])

    for i in range(n_rows):
        for j in range(n_cols):
            if mean_matrix[i, j] > 1e-12:
                ax1.text(
                    j, i,
                    f"{mean_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=annot_font
                )

    ax1.set_title("Mean migration matrix", fontsize=font_scale, pad=10)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(pop_labels, fontsize=max(4, tick_font - 2))
    ax1.set_xlabel("Ancestral population", fontsize=font_scale)
    ax1.set_ylabel("Generation", fontsize=font_scale)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks + 1)
    ax1.tick_params(axis='y', labelsize=tick_font, pad=6)
    ax1.tick_params(axis='x', labelsize=max(4, tick_font - 2), pad=6)

    cbar1 = fig.colorbar(
        im1,
        ax=ax1,
        orientation="horizontal",
        pad=0.18,
        fraction=0.05
    )
    cbar1.set_label("Migration rate", fontsize=font_scale, labelpad=8)
    cbar1.ax.tick_params(labelsize=tick_font)

    # Panel 2: Sex bias matrix
    im2 = ax2.imshow(sex_bias_matrix, cmap=cmap_bias, norm=norm_bias, aspect="auto")
    ax2.set_box_aspect(sex_bias_matrix.shape[0] / sex_bias_matrix.shape[1])

    for i in range(n_rows):
        for j in range(n_cols):
            if abs(sex_bias_matrix[i, j]) > 1e-12:
                ax2.text(
                    j, i,
                    f"{sex_bias_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=annot_font
                )

    ax2.set_title("Sex bias in migration", fontsize=font_scale, pad=10)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(pop_labels, fontsize=max(4, tick_font - 2))
    ax2.set_xlabel("Ancestral population", fontsize=font_scale)
    ax2.set_ylabel("Generation", fontsize=font_scale)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_ticks + 1)
    ax2.tick_params(axis='y', labelsize=tick_font, pad=6)
    ax2.tick_params(axis='x', labelsize=max(4, tick_font - 2), pad=6)

    cbar2 = fig.colorbar(
        im2,
        ax=ax2,
        orientation="horizontal",
        pad=0.18,
        fraction=0.05
    )

    cbar2.set_label("Sex bias", fontsize=font_scale, labelpad=8)
    cbar2.set_ticks([-1, 0, 1])
    cbar2.set_ticklabels([
        "-1 (male-only)",
        "0 (unbiased)",
        "+1 (female-only)"
    ])
    cbar2.ax.tick_params(labelsize=tick_font)

    # Layout
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    # Save plot
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    png_path = os.path.splitext(output_path)[0] + ".png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


def output_simulation_data_sex_biased(sample_population: Population,
                                    optimal_params: np.ndarray,
                                    optimal_likelihood:float,
                                    genetic_model: GeneticModel,
                                    driver_spec: InferenceConfig,
                                    output_dir: Path,
                                    driver_path: str|None = None
                                    ):
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
    genetic_model: GeneticModel
        Bundles the demographic model for which to output simulation data with the admixture
        model configuration (``ad_model_autosomes``/``ad_model_allosomes``) used to compute it.
    driver_spec: InferenceConfig
        The driver specification containing output configuration.
    output_dir: Path
        The directory to which output files will be written.
    driver_path: str | None
        The path to the driver yaml file. If None, no driver file will be copied to the output directory. Defaults to None.
    """
    demographic_model = genetic_model.demographic_model
    ad_model_autosomes = genetic_model.phase_type_config.ad_model_autosomes
    ad_model_allosomes = genetic_model.phase_type_config.ad_model_allosomes

    # ------ Create output directory if it doesn't exist ------

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if driver_path is not None:
        shutil.copy2(driver_path, output_dir)

    # ------- Set up output filename format and load required parameters for output production ------
    output_filename_format = driver_spec.output.output_filename_format
    exclude_tracts_below_cM = driver_spec.optim.exclude_tracts_below_cm
    npts = driver_spec.optim.npts
    log_scale = driver_spec.output.log_scale
    N_cores = driver_spec.optim.N_cores

    matrices = demographic_model.get_migration_matrices(optimal_params)
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
    
    pop_names = list(demographic_model.population_indices.keys())

    ancestry_per_individual = {ind:ind.ancestryProps(pop_names, cutoff=0) for ind in sample_population.indivs}
    
    with open(output_dir / output_filename_format.format(label='ancestry_per_individual'), 'w') as fbins:
        fbins.write("individual\t" + "\t".join(pop_names)+"\n")
        for ind, proportions in ancestry_per_individual.items():
            fbins.write(ind.name + "\t" + "\t".join(map(str,proportions))+"\n")
    
    
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
                                                                                                                chrom_lengths=Ls) for pop, pop_num in demographic_model.population_indices.items()}
    elif ad_model_autosomes == 'M':
        autosome_predicted={pop:PhTMonoecious(migration_matrix=0.5*(female_matrix+male_matrix),
                                            rho=1).tract_length_histogram_multi_windowed(population_number=pop_num,
                                                                                        bins=autosome_bins,
                                                                                        chrom_lengths=Ls) for pop, pop_num in demographic_model.population_indices.items()}
    elif ad_model_autosomes == 'H-DC':
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DC',
                                                                            rho_f=1,
                                                                            rho_m=1,
                                                                            X_chr=False,
                                                                            X_chr_male=False,
                                                                            N_cores=N_cores,
                                                                            population_number=pop_num,
                                                                            bins=autosome_bins,
                                                                            chrom_lengths=Ls) for pop, pop_num in demographic_model.population_indices.items()}
    else:
        autosome_predicted={pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DF',
                                                                            rho_f=1,
                                                                            rho_m=1,
                                                                            X_chr=False,
                                                                            X_chr_male=False,
                                                                            N_cores=N_cores,
                                                                            population_number=pop_num,
                                                                            bins=autosome_bins,
                                                                            chrom_lengths=Ls) for pop, pop_num in demographic_model.population_indices.items()}
    
    # Save autosome results
    with open(output_dir / output_filename_format.format(label='tract_length_autosome_bins'), 'w') as fbins:
        fbins.write("\t".join(map(str, autosome_bins)))
    with open(output_dir / output_filename_format.format(label='autosome_sample_tract_distribution'), 'w') as fdat:
        for population in demographic_model.population_indices.keys():
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
        for pop, pop_num in demographic_model.population_indices.items():
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
                                                                                                        chrom_lengths=[allosome_length]) for pop, pop_num in demographic_model.population_indices.items()}
            male_predicted = {pop: PhTDioecious(migration_matrix_f=female_matrix,
                                                migration_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                sex_model=ad_model_allosomes,
                                                X_chromosome=True,
                                                X_chromosome_male=True).tract_length_histogram_multi_windowed(population_number=pop_num,
                                                                                                            bins=allosome_bins,
                                                                                                            chrom_lengths=[allosome_length]) for pop, pop_num in demographic_model.population_indices.items()}
        elif ad_model_allosomes == 'H-DC':
            female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                                mig_matrix_m=male_matrix,
                                                                                TP=2,
                                                                                D_model='DC',
                                                                                rho_f=1,
                                                                                rho_m=1,
                                                                                X_chr=True,
                                                                                X_chr_male=False,
                                                                                N_cores=N_cores,
                                                                                population_number=pop_num,
                                                                                bins=allosome_bins,
                                                                                chrom_lengths=[allosome_length]) for pop, pop_num in demographic_model.population_indices.items()}
            male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DC',
                                                                            rho_f=1,
                                                                            rho_m=1,
                                                                            X_chr=True,
                                                                            X_chr_male=True,
                                                                            N_cores=N_cores,
                                                                            population_number=pop_num,
                                                                            bins=allosome_bins,
                                                                            chrom_lengths=[allosome_length]) for pop, pop_num in demographic_model.population_indices.items()}
        else:
            female_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                                mig_matrix_m=male_matrix,
                                                                                TP=2,
                                                                                D_model='DF',
                                                                                rho_f=1,
                                                                                rho_m=1,
                                                                                X_chr=True,
                                                                                X_chr_male=False,
                                                                                N_cores=N_cores,
                                                                                population_number=pop_num,
                                                                                bins=allosome_bins,
                                                                                chrom_lengths=[allosome_length]) for pop, pop_num in demographic_model.population_indices.items()}
            male_predicted = {pop:HP.HP_tract_length_histogram_multi_windowed(mig_matrix_f=female_matrix,
                                                                            mig_matrix_m=male_matrix,
                                                                            TP=2,
                                                                            D_model='DF',
                                                                            rho_f=1, rho_m=1,
                                                                            X_chr=True,
                                                                            X_chr_male=True,
                                                                            N_cores=N_cores,
                                                                            population_number=pop_num,
                                                                            bins=allosome_bins,
                                                                            chrom_lengths=[allosome_length]) for pop, pop_num in demographic_model.population_indices.items()}
    
        # Save allosome results
        with open(output_dir / output_filename_format.format(label='tract_length_allosome_bins'), 'w') as fbins:
            fbins.write("\t".join(map(str, allosome_bins)))
        with open(output_dir / output_filename_format.format(label='female_allosome_sample_tract_distribution'), 'w') as fdat:
            for population in demographic_model.population_indices.keys():
                try:
                    fdat.write("\t".join(map(str, female_data[population])) + "\n")
                except KeyError:
                    female_data[population] = np.zeros(len(allosome_bins)).tolist()
                    print(f'Population {population} not found in female allosome data.')
        with open(output_dir / output_filename_format.format(label='male_allosome_sample_tract_distribution'), 'w') as fdat:
            for population in demographic_model.population_indices.keys():
                try:
                    fdat.write("\t".join(map(str, male_data[population])) + "\n")
                except KeyError:
                    male_data[population] = np.zeros(len(allosome_bins)).tolist()
                    print(f'Population {population} not found in male allosome data.')           
        with open(output_dir / output_filename_format.format(label='female_allosome_predicted_tract_distribution'), 'w') as fpred2:
            for pop, pop_num in demographic_model.population_indices.items():
                fpred2.write("\t".join(map(
                    str,
                    [num_females * num_tracts for num_tracts in female_predicted[pop]]))
                            + "\n")
        with open(output_dir / output_filename_format.format(label='male_allosome_predicted_tract_distribution'), 'w') as fpred2:
            for pop, pop_num in demographic_model.population_indices.items():
                fpred2.write("\t".join(map(
                    str,
                    [num_males * num_tracts for num_tracts in male_predicted[pop]]))
                            + "\n")

    # ------ Save optimal parameters -------
    param_names = list(demographic_model.model_base_params.keys())
    params_file_path = output_dir / output_filename_format.format(label="optimal_parameters.txt")
    remainder_params = compute_remainder_params(demographic_model, matrices)
    with open(params_file_path, "w") as f:

        f.write("parameter\tvalue\n")
        for name, value in zip(param_names, optimal_params):
            f.write(f"{name}\t{value}\n")
        for name, value in remainder_params.items():
            f.write(f"{name}\t{value}\n")
        f.write(f"likelihood {optimal_likelihood:>12.6g}\n")

    # ------ Produce and display plots -------
    
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

    
    fig, ax = plot_admixture(ancestry_per_individual, pop_names, colors, ax=None)
    admixture_file_path = output_dir / output_filename_format.format(label="admixture_plot.pdf")
    fig.savefig(admixture_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        subtitle: str | None = None):

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
                fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
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
        # Also save a PNG version
        png_path = os.path.splitext(output_path)[0] + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Produce migration matrices plot
    if driver_spec.output.plot_migration_matrices:
        _plot_migration_matrices(migration_matrix_f=female_matrix,
                                migration_matrix_m=male_matrix,
                                pop_labels=pop_names,
                                output_path=os.path.join(
                                    output_dir,
                                    output_filename_format.format(label="migration_matrices.pdf")
                                ))

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
            output_filename_format.format(label="autosomes_all_populations.pdf")
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
                output_filename_format.format(label="male_allosomes_all_populations.pdf")
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
                output_filename_format.format(label="female_allosomes_all_populations.pdf")
            ),
            subtitle=f"Log-likelihood: {optimal_likelihood:.6g}"
        )

    
    # Final message
    print('Results saved to : ' + str(output_dir))
    logger.info('Results saved to : ' + str(output_dir))



def plot_admixture(ancestry_per_individual, labels, colors, ax=None):
    """
    Stacked bar chart of ancestry proportions in ADMIXTURE style. AI-generated with review.

    Parameters
    ----------
    ancestry_per_individual : dict
        {individual: [prop_1, ..., prop_n]}
    labels : list[str]
        Population name for each proportion column.
    colors : list[str]
        Hex colour for each population.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if None.

    Returns
    -------
    fig, ax
    """
    individuals = list(ancestry_per_individual.keys())
    props = np.array([ancestry_per_individual[ind] for ind in individuals])  # shape (n_ind, n_pop)
    sort_pop = int(np.argmax(props.mean(axis=0)))
    order = np.argsort(props[:, sort_pop])[::-1]
    individuals = [individuals[i] for i in order]
    props = props[order]
    ind_names = [ind.name for ind in individuals]
    
    x = np.arange(len(individuals))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(individuals) * 0.15), 3))
    else:
        fig = ax.get_figure()

    bottoms = np.zeros(len(individuals))
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x, props[:, i], bottom=bottoms, color=color, width=1.0, label=label)
        bottoms += props[:, i]

    ax.set_xlim(-0.5, len(individuals) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(ind_names, rotation=90, fontsize=6)
    ax.set_ylabel("Ancestry proportion")
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1), frameon=False)
    fig.tight_layout()
    return fig, ax




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
        The parameter handler for the demographic model.
    param_names: list[str]
        A list of parameter names corresponding to the parameters in the demographic model, used for printing results.
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


def _print_param_bounds_table(demographic_model: ParametrizedDemography) -> None:
    """
    Prints and logs a table of each model parameter's effective admissibility bounds (i.e. after
    any narrowing from the driver file's ``bounds`` section via ``parse_param_bounds`` has already
    been applied to ``demographic_model.model_base_params[...].bounds``). Shown once per run,
    before the starting-parameters table.

    Parameters
    ----------
    demographic_model: ParametrizedDemography
        The demographic model whose parameters' current bounds are reported.
    """
    param_names = list(demographic_model.model_base_params.keys())
    name_col_width = max((len(name) for name in param_names), default=5)
    bound_col_width = 12

    title_message = "Model parameters and bounds:"
    print(title_message)
    logger.info(title_message)

    table_header = f"{'Parameter':<{name_col_width}} | {'Lower bound':>{bound_col_width}} | {'Upper bound':>{bound_col_width}}"
    table_line = "-" * len(table_header)

    for l in (table_line, table_header, table_line):
        print(l)
        logger.info(l)

    def _format_bound(value: float) -> str:
        return "inf" if np.isinf(value) else f"{value:.4g}"

    for name in param_names:
        lower, upper = demographic_model.model_base_params[name].bounds
        row = f"{name:<{name_col_width}} | {_format_bound(lower):>{bound_col_width}} | {_format_bound(upper):>{bound_col_width}}"
        print(row)
        logger.info(row)

    print(table_line)
    logger.info(table_line)


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
                               demographic_model,
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
    demographic_model
        Demographic model used as a fallback source for parameter metadata when ``parameter_handler.demography`` is unavailable.
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
        return list(range(len(demographic_model.model_base_params)))

    step_2_only = bool(steps) and all(step in (2, "step2") for step in steps)
    model_base_params = (
        parameter_handler.demography.model_base_params
        if hasattr(parameter_handler, "demography")
        else demographic_model.model_base_params
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


def _print_and_log(*messages: str) -> None:
    """
    Prints and logs each message, unconditionally.
    """
    for message in messages:
        print(message)
        logger.info(message)


def _build_reoptimization_intro_message(n_reoptimizations: int) -> str:
    """
    Builds the message announcing that ``n_reoptimizations`` (greater than 0) triggers an
    iterating re-optimization starting at the current optimal parameters, where sex-bias
    parameters are fixed at their optimal values, changing how step 1 is optimized.
    """

    reopt_intro_message = f"Launching re-optimization until convergence is achieved or {n_reoptimizations} re-optimizations have been performed."
    separator = "-" * len(reopt_intro_message)
    return (
        f"\n{separator}\n"
        f"{reopt_intro_message}\n"
        f"{separator}"
    )


def _build_step2_skip_message(sex_bias_param_names: list[str], parameter_handler: FixedParametersHandler) -> str:
    """
    Builds the message announcing that step 2 has no free sex-bias parameters to
    optimize, listing which fixing mechanism (ancestry proportions vs. user-provided
    values) accounts for each fixed sex-bias parameter.
    """
    fixed_by_ancestry = [n for n in sex_bias_param_names if n in set(parameter_handler.params_fixed_by_ancestry)]
    fixed_by_value = [n for n in sex_bias_param_names if n in set(parameter_handler.user_params_fixed_by_value.keys())]
    fix_parts = []
    if fixed_by_ancestry:
        fix_parts.append(f"{', '.join(fixed_by_ancestry)} by ancestry proportions")
    if fixed_by_value:
        fix_parts.append(f"{', '.join(fixed_by_value)} by user-provided values")
    return (
        "All sex-bias parameters are fixed"
        + (f" ({'; '.join(fix_parts)})" if fix_parts else "")
        + ". Step 2 has no free parameters to optimize and will be skipped."
    )


def _select_full_data_likelihood(likelihoods_step_2: list[float], full_likelihoods_step_2: list,
                                optimal_likelihood: float, use_autosomes_for_sex_bias: bool,
                                announce: bool = False) -> float:
    """
    When step 2 used allosomal data only (``use_autosomes_for_sex_bias`` is False), selects
    the full-data (autosomal + allosomal) likelihood computed at the best run's parameters, if
    one was computed, and returns it in place of ``optimal_likelihood``. Otherwise returns
    ``optimal_likelihood`` unchanged. If ``announce``, reports the substitution.
    """
    if use_autosomes_for_sex_bias:
        return optimal_likelihood
    best_run_index = int(np.argmax([float(x) for x in likelihoods_step_2]))
    full_data_likelihood = full_likelihoods_step_2[best_run_index]
    if full_data_likelihood is None:
        return optimal_likelihood
    if announce:
        _print_and_log(
            "Step 2 used allosomal data only. Final likelihood is evaluated on "
            "autosomal + allosomal data at the selected optimal parameters."
        )
    return float(full_data_likelihood)


def _print_run_intro(parameter_handler: FixedParametersHandler,
                     demographic_model,
                     start_params_list: list[np.ndarray],
                     bound_func: Callable[[np.ndarray], float],
                     title_message: str,
                     two_steps_optimization: bool,
                     autosomes_in_step_2: bool,
                     steps: list[int | str] | None = None,
                     print_start_params_table: bool = True) -> None:
    """
    Print the optimization subtitle (always) and starting-parameter table (optional) for a run.

    This helper centralizes the pre-run console/log output shown before each optimization phase. Time-parameter transition logging is temporarily
    disabled while printing the starting-parameter table to avoid emitting admissibility transition warnings during display-only conversions.

    Parameters
    ----------
    parameter_handler: FixedParametersHandler
        Parameter handler used for subtitle generation and parameter conversion.
    demographic_model
        Demographic model used as fallback metadata source when needed.
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
    print_start_params_table: bool
        Whether to print the starting-parameters table below the subtitle. The dash-bordered
        subtitle itself is always printed. Defaults to True.
    """
    if hasattr(parameter_handler, "demography"):
        subtitle_message = _get_optimization_subtitle(
            parameter_handler=parameter_handler,
            two_steps_optimization=two_steps_optimization,
            autosomes_in_step_2=autosomes_in_step_2,
            steps=steps,
        )
    else:
        all_params = demographic_model.model_base_params
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
        demographic_model=demographic_model,
        two_steps_optimization=two_steps_optimization,
        steps=steps,
    )

    if not print_start_params_table:
        return

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


def _save_ancestry_proportions_table(ancestor_labels, observed_autosome_proportions: np.ndarray, predicted_autosome_proportions: np.ndarray | None,
                                    output_dir, output_filename_format: str, observed_allosome_proportions: np.ndarray | None = None,
                                    predicted_allosome_proportions: np.ndarray | None = None, allosome_label: str | None = None) -> None:
    """
    Writes a fixed-width text table of observed and predicted ancestry proportions
    (for autosomes and, optionally, allosomes) to the output directory.

    Parameters
    ----------
    ancestor_labels:
        Ordered iterable of source-population names (columns of the table).
    observed_autosome_proportions:
        Observed autosomal ancestry proportions, one value per source population.
    predicted_autosome_proportions:
        Predicted autosomal ancestry proportions from the optimal model parameters,
        or ``None`` if not available.
    output_dir:
        Path to the directory where the file will be written.
    output_filename_format:
        The ``output_filename_format`` string from the driver file (must contain
        a ``{label}`` placeholder).
    observed_allosome_proportions:
        Observed allosomal ancestry proportions, or ``None`` if no allosomes are
        present in the sample.
    predicted_allosome_proportions:
        Predicted allosomal ancestry proportions from the optimal model parameters,
        or ``None`` if not available.
    allosome_label:
        The allosome identifier (e.g. ``'X'``), used as the row label suffix.
        Required when ``observed_allosome_proportions`` or
        ``predicted_allosome_proportions`` is provided.
    """
    pop_labels = list(ancestor_labels)
    col_w = max(max(len(lbl) for lbl in pop_labels), 12)
    row_label_w = 30

    rows = [("Observed (autosomes)", observed_autosome_proportions)]
    if predicted_autosome_proportions is not None:
        rows.append(("Predicted (autosomes)", predicted_autosome_proportions))
    if observed_allosome_proportions is not None:
        rows.append((f"Observed ({allosome_label})", observed_allosome_proportions))
    if predicted_allosome_proportions is not None:
        rows.append((f"Predicted ({allosome_label})", predicted_allosome_proportions))

    header = f"{'':>{row_label_w}} " + " ".join(f"{lbl:>{col_w}}" for lbl in pop_labels)
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for row_label, values in rows:
        lines.append(
            f"{row_label:>{row_label_w}} " + " ".join(f"{v:>{col_w}.6f}" for v in values)
        )
    lines.append(sep)

    out_path = Path(output_dir) / output_filename_format.format(label="ancestry_proportions.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    logger.info(f"Ancestry proportions table saved to {output_dir / output_filename_format.format(label='ancestry_proportions.txt')}")
