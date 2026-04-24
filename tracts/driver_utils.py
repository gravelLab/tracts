import logging
from logging.handlers import MemoryHandler
import numbers
import os
import sys
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

import ruamel.yaml as yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)

# ---------- Driver file setup ----------

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

# ---------- Models ----------

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
    male_names: List[str] | str
        A list of individual names corresponding to male individuals, or "auto" to automatically determine based on the presence of allosomes.
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
    male_names: List[str] | str = "auto" 
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
    output_directory: str
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
    """

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
    fix_parameters_from_ancestry_proportions: List[str] = []
    output_directory: str = ""
    output_filename_format: str
    log_filename: Optional[str] = "tracts.log"
    ad_model_autosomes: str = "DC"
    ad_model_allosomes: str = "DC"
    verbose_log: int = 1
    verbose_screen: int = 30
    log_scale: bool = True
    two_steps_optimization: bool = True
    run_optimize_cob: bool = False


# ---------- Driver file setup ----------

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

# ---------- Loader ----------

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
    
    male_list = driver_spec.samples.male_names
    chromosome_list = parse_chromosomes(driver_spec.samples.chromosomes)
    logger.info(f'Chromosomes: {chromosome_list}')
    logger.info(f'Allosomes: {allosome_labels if allosome_labels else []}')
    pop = Population(filenames_by_individual=individual_filenames,
                    selectchrom=chromosome_list,
                    allosomes=allosome_labels if allosome_labels else [],
                    male_list = male_list)
    if len(allosome_labels)>=1 and allosome_labels is not None:
        assert(allosome_labels[0] == 'X'), "Currently only X allosome is supported for male determination. Should be first allosome."
    if len(allosome_labels) >0:
        pop.set_males(male_list = male_list, allosome_label = allosome_labels[0]) 
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

def parse_start_params(start_param_bounds,
                    repetitions: int=1, 
                    seed: float=None,
                    model: ParametrizedDemography = None):
    """
    Produces a 1-dimensional array of starting parameters for optimization in physical units, for every parameter in base_model_parameters.
    
    Parameters
    ----------
    start_param_bounds
        An object containing attributes corresponding to each parameter in model.model_base_parameters, where the value of each attribute is either a single number (if the starting value for that parameter should be fixed) or a string of the form "min:max" specifying the range from which to sample starting values for that parameter. The parameters specified in start_param_bounds must match those in model.model_base_parameters, and an error will be raised if any parameters are missing or if any extra parameters are included.
    repetitions: int
        The number of sets of starting parameters to produce. Defaults to 1.
    seed: float
        The random seed to use for sampling starting parameters. Defaults to None.
    model: ParametrizedDemography
        The demographic model for which to produce starting parameters. 

    Returns
    -------
    list[np.ndarray]: A list of arrays of starting parameters in physical units, where each array corresponds to a set of starting parameters for one repetition of the optimization. The parameters are ordered according to their order in model.model_base_parameters.

    """ 
    
    num_params = len(model.model_base_params)
    rng = np.random.default_rng(seed=seed)
    start_params = rng.random((repetitions, num_params))
    for param_name, param_info in model.model_base_params.items():
        if param_name in model.params_fixed_by_ancestry:
            start_params[:, param_info.index] = param_info.bounds[0] # TODO: This will be replaced, set to arbitrary feasible value.
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
        
    if len(model.params_fixed_by_ancestry) > 0:
        start_params = [model.parameter_handler.compute_params_fixed_by_ancestry(start_param_set)
         for start_param_set in start_params]
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




# ----- Output production -----

def output_simulation_data_sex_biased(sample_population: Population,
                                    optimal_params: np.ndarray, 
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
    output_dir = driver_spec.output_directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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

    # ------ Save optimal parameters -------
    param_names = list(model.model_base_params.keys())
    params_file_path = output_dir + output_filename_format.format(label="optimal_parameters") + ".txt"
    with open(params_file_path, "w") as f:
        f.write("parameter\tvalue\n")
        for name, value in zip(param_names, optimal_params):
            f.write(f"{name}\t{value}\n")

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
        alpha_ci: float=0.05):

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
        )
    
    # Final message
    print('Results saved to : ' + output_dir)
    logger.info('Results saved to : ' + output_dir)