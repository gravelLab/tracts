import io
import contextlib
import logging
from typing import Callable
import numpy as np

from tracts.population import Population
from tracts.core import optimize_cob_sex_biased_single_step, optimize_cob_sex_biased_two_steps
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.driver_utils import *
from tracts.driver_utils import (
    _print_run_intro,
    _normalize_multi_init_result,
    _summarize_step_results,
    _print_optimal_values_and_likelihood,
    _save_ancestry_proportions_table,
)
from tracts.logs import initialize_tracts, close_log_file

logger = logging.getLogger(__name__)


def run_tracts(driver_filename: str, script_dir: str):
    """
    Main function to run tracts with a specified driver file. This function runs the inference pipeline based on the information provided in the driver file, and produces output files with the results.
    For details on how to specify the driver file, see the online documentation and user guide.

    Parameters
    ----------
    driver_filename: str
        The name of the driver file to use.
    script_dir: str
        The directory containing the script.
    """

    # ------- Locate and load the driver file -------
    driver_path = locate_file_path(filename=driver_filename,
                                   script_dir=script_dir)
    driver_spec = load_driver_file(driver_path)

    # ------ Initialize tracts: set up logging and output directory using filename from driver-------
    logger, log_full_path, output_dir = initialize_tracts(driver_spec=driver_spec, driver_filename=driver_filename)

    try:        
        # ----- Extract admixture models and allosomal configuration from the driver file -------
        ad_model_autosomes, ad_model_allosomes, allosome_label = get_admixture_models(driver_spec=driver_spec)

        # ------ Load the population -------
        pop = load_population(driver_path=driver_path,
                            driver_spec=driver_spec,
                            script_dir=script_dir,
                            allosome_labels = driver_spec.samples.allosomes) 
        pop.unknown_labels = driver_spec.optim.unknown_labels_for_smoothing
        pop.smooth_unknowns(allosome_labels=driver_spec.samples.allosomes)
        _bins, _data = pop.get_global_tractlengths(npts=driver_spec.optim.npts, # Get the population labels and validate that these correspond to to model population labels.
                                                   exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm) 
        
        # ------ Load the demographic model -------
        demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names  = load_demographic_model_from_driver(driver_spec=driver_spec,
                                                                                                                                    script_dir=script_dir,
                                                                                                                                    driver_path=driver_path,
                                                                                                                                    allosome_label=allosome_label)
        
        #----- Validate that the population labels in the data correspond to the model population labels -------
        check_population_labels(demographic_model=demographic_model,
                                population=pop,
                                data=_data)
        
        # ------ Calculate ancestry proportions -------
        autosome_proportions, allosome_proportions = get_ancestry_proportions(driver_spec=driver_spec,
                                                                            population=pop,
                                                                            ancestor_labels=demographic_model.population_indices.keys(),
                                                                            allosome_label=allosome_label)
        
        # ------ Set up fixed parameters if specified in the driver ------
        setup_fixed_parameters(driver_spec=driver_spec,
                               demographic_model=demographic_model,
                               allosome_label=allosome_label,
                               autosome_proportions=autosome_proportions,
                               allosome_proportions=allosome_proportions)
        
        # ------ Optimizer setup -------
        func = get_time_scaled_model_func(demographic_model) # Time parameters need to be rescaled for some optimizers, so we create a wrapper function that applies the necessary rescaling before passing parameters to the model.
        bound = get_time_scaled_model_bounds(demographic_model) # The same rescaling needs to be applied to the bounds function.
    
        # Show time-admissibility warnings only during optimization and final reporting.
        demographic_model.parameter_handler.enable_time_param_logging = False

        # ------ Compute starting parameters in physical units ------
        if driver_spec.optim.two_steps_optimization:
            physical_start_params = parse_start_params(start_param_bounds=driver_spec.start_params,
                                                        repetitions=driver_spec.optim.repetitions,
                                                        seed=driver_spec.optim.seed,
                                                        demographic_model=demographic_model,
                                                        sample_param_names=set(non_sex_bias_param_names),
                                                        fixed_param_values={name: 0.0 for name in sex_bias_param_names if name not in driver_spec.optim.fix_parameters_by_value.keys()} | driver_spec.optim.fix_parameters_by_value, # Dictionary union
                                                        )
            physical_start_params = collapse_identical_start_params(physical_start_params, "step 1")
        else:
            physical_start_params = parse_start_params(start_param_bounds=driver_spec.start_params,
                                                        repetitions=driver_spec.optim.repetitions,
                                                        seed=driver_spec.optim.seed,
                                                        demographic_model=demographic_model,
                                                        fixed_param_values=driver_spec.optim.fix_parameters_by_value
                                                        )
        
        check_start_params(physical_start_params=physical_start_params,
                          model_param_names=model_param_names) # Checks compatibility with demographic model and prints message

        # ------ Convert starting parameters to optimizer units ------
        optimizer_start_params = [demographic_model.parameter_handler.convert_to_optimizer_params(params) for params in physical_start_params]   
      
        # ------ Get starting ancestry proportions for the starting parameters ------ 
        get_starting_ancestry_proportions(demographic_model=demographic_model,
                                            model_func=func,
                                            optimizer_start_params=optimizer_start_params) # The computed proportions are only logged.

        demographic_model.parameter_handler.enable_time_param_logging = True

        # ------ Run the model with (multiple) starting parameters ------
        step_1_start_params_title = "Starting parameters for step 1 optimization" if driver_spec.optim.two_steps_optimization else "Starting parameters for single-step optimization"

        if driver_spec.optim.two_steps_optimization is False: # Single-step optimization using optimize_cob_sex_biased_single_step

            _print_run_intro(demographic_model.parameter_handler, demographic_model, optimizer_start_params, bound, step_1_start_params_title, False, True)

            params_found, likelihoods, _full_likelihoods = _normalize_multi_init_result(run_model_multi_init(model_func=func,
                                                            bound_func=bound,
                                                            population=pop, 
                                                            start_params_list=optimizer_start_params,
                                                            population_dict=demographic_model.population_indices.items(),
                                                            parameter_handler=demographic_model.parameter_handler,
                                                            max_iter=driver_spec.optim.maximum_iterations,
                                                            exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm,
                                                            ad_model_autosomes = ad_model_autosomes, 
                                                            ad_model_allosomes=ad_model_allosomes,
                                                            npts=driver_spec.optim.npts, 
                                                            verbose_log=driver_spec.output.verbose_log,
                                                            verbose_screen=driver_spec.output.verbose_screen, 
                                                            two_steps_optimization=False,
                                                            start_params_title=step_1_start_params_title,
                                                            print_start_params_table=False,
                                                            N_cores=driver_spec.optim.N_cores,
                                                            rho_f=driver_spec.models.rho_f,
                                                            rho_m=driver_spec.models.rho_m,
                                                            TP=driver_spec.models.TP))

            optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found,
                                                                        likelihoods=likelihoods,
                                                                        parameter_handler=demographic_model.parameter_handler,
                                                                        param_names=model_param_names)
        
        else: # Performs two-steps optimization with multiple starting parameters            

            # ------ Step 1: optimize non-sex-bias parameters on autosomal data ------

            _print_run_intro(demographic_model.parameter_handler, demographic_model, optimizer_start_params, bound, step_1_start_params_title, True, driver_spec.optim.use_autosomes_for_sex_bias, [1])

            params_found_step_1, likelihoods_step_1, _full_likelihoods_step_1 = _normalize_multi_init_result(run_model_multi_init(model_func=func,
                                                            bound_func=bound,
                                                            population=pop, 
                                                            start_params_list=optimizer_start_params,
                                                            population_dict=demographic_model.population_indices.items(),
                                                            parameter_handler=demographic_model.parameter_handler,
                                                            max_iter=driver_spec.optim.maximum_iterations,
                                                            exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm,
                                                            ad_model_autosomes = ad_model_autosomes, 
                                                            ad_model_allosomes=ad_model_allosomes,
                                                            npts=driver_spec.optim.npts, 
                                                            verbose_log=driver_spec.output.verbose_log,
                                                            verbose_screen=driver_spec.output.verbose_screen, 
                                                            two_steps_optimization=True,
                                                            autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias, # This parameter is ignored in step 1
                                                            steps=[1],
                                                            start_params_title=step_1_start_params_title,
                                                            print_start_params_table=False,
                                                            N_cores=driver_spec.optim.N_cores,
                                                            rho_f=driver_spec.models.rho_f,
                                                            rho_m=driver_spec.models.rho_m,
                                                            TP=driver_spec.models.TP))
                                    
            #  Process and print results
            optimal_params_step_1, _optimal_likelihood_step_1 = _summarize_step_results(params_found=params_found_step_1,
                                                                        likelihoods=likelihoods_step_1,
                                                                        parameter_handler=demographic_model.parameter_handler,
                                                                        param_names=model_param_names,
                                                                        step_label="Step 1")

            if ad_model_allosomes is None:
                logger.info("No allosomal data provided. Skipping Step 2 and using Step 1 results.")
                optimal_params, optimal_likelihood = optimal_params_step_1, _optimal_likelihood_step_1
            else:
                # ------ Step 2: optimize sex-bias parameters on (autosomal and) allosomal data ------

                # Detect whether there are any sex-bias parameters that are free (not fixed by ancestry
                # or by value). When all sex-bias params are fixed, step 2 has no free variables: skip
                # all the verbose setup/results output and go straight to the final table.
                _all_fixed_in_step_2 = (
                    set(demographic_model.parameter_handler.params_fixed_by_ancestry)
                    | set(demographic_model.parameter_handler.user_params_fixed_by_value.keys())
                )
                _has_free_sex_bias = any(name not in _all_fixed_in_step_2 for name in sex_bias_param_names)

                # Draw fresh sex-bias starts for step 2, while keeping the best step 1 non-sex-bias values fixed.
                step_2_fixed_param_values = {
                    name: float(value)
                    for name, value in zip(model_param_names, optimal_params_step_1)
                    if name not in sex_bias_param_names
                }
                # Sex-bias parameters fixed by the user at a specific value must not be resampled.
                for _sbv_name, _sbv_value in demographic_model.parameter_handler.user_params_fixed_by_value.items():
                    if _sbv_name in sex_bias_param_names:
                        step_2_fixed_param_values[_sbv_name] = _sbv_value

                if _has_free_sex_bias:
                    end_step_1_message = "Selecting best parameters from step 1 and proceeding to step 2 optimization.\n"
                    print(end_step_1_message)
                    logger.info(end_step_1_message)

                    step_2_physical_start_params = parse_start_params(
                        start_param_bounds=driver_spec.start_params,
                        repetitions=driver_spec.optim.repetitions,
                        seed=driver_spec.optim.seed,
                        demographic_model=demographic_model,
                        sample_param_names=set(sex_bias_param_names),
                        fixed_param_values=step_2_fixed_param_values,
                    )
                    step_2_physical_start_params = collapse_identical_start_params(step_2_physical_start_params, "step 2")
                    step_2_start_params = [
                        demographic_model.parameter_handler.convert_to_optimizer_params(params)
                        for params in step_2_physical_start_params
                    ]

                    step_2_start_params_title = (
                        "Starting parameters for step 2 optimization "
                        "(non-sex-bias parameters are fixed to the best step 1 estimates)."
                    )

                    _print_run_intro(demographic_model.parameter_handler, demographic_model, step_2_start_params, bound, step_2_start_params_title, True, driver_spec.optim.use_autosomes_for_sex_bias, [2])

                    params_found_step_2, likelihoods_step_2, full_likelihoods_step_2 = _normalize_multi_init_result(run_model_multi_init(model_func=func,
                                                                                bound_func=bound,
                                                                                population=pop,
                                                                                start_params_list=step_2_start_params,
                                                                                population_dict=demographic_model.population_indices.items(),
                                                                                parameter_handler=demographic_model.parameter_handler,
                                                                                max_iter=driver_spec.optim.maximum_iterations,
                                                                                exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm,
                                                                                ad_model_autosomes=ad_model_autosomes,
                                                                                ad_model_allosomes=ad_model_allosomes,
                                                                                npts=driver_spec.optim.npts,
                                                                                verbose_log=driver_spec.output.verbose_log,
                                                                                verbose_screen=driver_spec.output.verbose_screen,
                                                                                two_steps_optimization=True,
                                                                                autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias,
                                                                                steps=[2],
                                                                                start_params_title=step_2_start_params_title,
                                                                                print_start_params_table=False,
                                                                                N_cores=driver_spec.optim.N_cores,
                                                                                rho_f=driver_spec.models.rho_f,
                                                                                rho_m=driver_spec.models.rho_m,
                                                                                TP=driver_spec.models.TP))

                    #  Process and print results
                    optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found_step_2,
                                                                                likelihoods=likelihoods_step_2,
                                                                                parameter_handler=demographic_model.parameter_handler,
                                                                                param_names=model_param_names,
                                                                                step_label="Step 2")

                    end_step_2_message = "Selecting best parameters from step 2."
                    print(end_step_2_message)
                    logger.info(end_step_2_message)

                    if not driver_spec.optim.use_autosomes_for_sex_bias:
                        best_run_index = int(np.argmax([float(x) for x in likelihoods_step_2]))
                        full_data_likelihood = full_likelihoods_step_2[best_run_index]
                        if full_data_likelihood is not None:
                            optimal_likelihood = float(full_data_likelihood)
                            full_lik_message = "Step 2 used allosomal data only. Final likelihood is evaluated on autosomal + allosomal data at the selected optimal parameters."
                            print(full_lik_message)
                            logger.info(full_lik_message)

                else:
                    # No free sex-bias parameters: run step 2 silently (only to compute the
                    # full-data likelihood at the step-1 optimal params) then skip to final table.
                    _fixed_by_ancestry = [n for n in sex_bias_param_names if n in set(demographic_model.parameter_handler.params_fixed_by_ancestry)]
                    _fixed_by_value = [n for n in sex_bias_param_names if n in set(demographic_model.parameter_handler.user_params_fixed_by_value.keys())]
                    _fix_parts = []
                    if _fixed_by_ancestry:
                        _fix_parts.append(f"{', '.join(_fixed_by_ancestry)} by ancestry proportions")
                    if _fixed_by_value:
                        _fix_parts.append(f"{', '.join(_fixed_by_value)} by user-provided values")
                    _skip_msg = (
                        "All sex-bias parameters are fixed"
                        + (f" ({'; '.join(_fix_parts)})" if _fix_parts else "")
                        + ". Step 2 has no free parameters to optimize and will be skipped."
                    )
                    print(_skip_msg)
                    logger.info(_skip_msg)
                    _silent_start = [demographic_model.parameter_handler.convert_to_optimizer_params(optimal_params_step_1)]
                    _tracts_logger = logging.getLogger("tracts")
                    _saved_tracts_level = _tracts_logger.level
                    _tracts_logger.setLevel(logging.CRITICAL)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            params_found_step_2, likelihoods_step_2, full_likelihoods_step_2 = _normalize_multi_init_result(
                                run_model_multi_init(
                                    model_func=func,
                                    bound_func=bound,
                                    population=pop,
                                    start_params_list=_silent_start,
                                    population_dict=demographic_model.population_indices.items(),
                                    parameter_handler=demographic_model.parameter_handler,
                                    max_iter=driver_spec.optim.maximum_iterations,
                                    exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm,
                                    ad_model_autosomes=ad_model_autosomes,
                                    ad_model_allosomes=ad_model_allosomes,
                                    npts=driver_spec.optim.npts,
                                    verbose_log=0,
                                    verbose_screen=0,
                                    two_steps_optimization=True,
                                    autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias,
                                    steps=[2],
                                    print_start_params_table=False,
                                    N_cores=driver_spec.optim.N_cores,
                                    rho_f=driver_spec.models.rho_f,
                                    rho_m=driver_spec.models.rho_m,
                                    TP=driver_spec.models.TP
                                )
                            )
                    finally:
                        _tracts_logger.setLevel(_saved_tracts_level)

                    optimal_params, optimal_likelihood = _summarize_step_results(
                        params_found=params_found_step_2,
                        likelihoods=likelihoods_step_2,
                        parameter_handler=demographic_model.parameter_handler,
                        param_names=model_param_names,
                        step_label="Step 2",
                    )
                    if not driver_spec.optim.use_autosomes_for_sex_bias:
                        best_run_index = int(np.argmax([float(x) for x in likelihoods_step_2]))
                        full_data_likelihood = full_likelihoods_step_2[best_run_index]
                        if full_data_likelihood is not None:
                            optimal_likelihood = float(full_data_likelihood)


        # ------ Compute remainder parameters (i.e. dependent parameters that have not been explicitely optimized) ------
        remainder_params = compute_remainder_params(demographic_model=demographic_model,
                                                migration_matrices=demographic_model.get_migration_matrices(optimal_params))
     
        # ------ Print final optimal parameters and likelihood ------
        _print_optimal_values_and_likelihood(demographic_model=demographic_model,
                                             optimal_params=optimal_params,
                                             optimal_likelihood=optimal_likelihood,
                                             remainder_parameters=remainder_params,
                                             ad_model_allosomes=ad_model_allosomes)

        # ------ Detect optimal parameters at boundaries ------
        check_optimal_parameters_at_boundaries(demographic_model=demographic_model,
                                               driver_spec=driver_spec,
                                               sex_bias_param_names=sex_bias_param_names,
                                               remainder_params=remainder_params,
                                               optimal_params=optimal_params)

        # ------ Check for founding migration rates > 1 in the final parameters ------
        check_final_parameters(demographic_model=demographic_model,
                               optimal_params=optimal_params)

        # ------ Compute and print ancestry proportions predicted by the model ------
        autosomal_predicted_ancestries, allosomal_predicted_ancestries = get_predicted_ancestry_proportions(demographic_model=demographic_model,
                                                                                                            model_func=func,
                                                                                                            optimal_params=optimal_params)

        # ------ Save ancestry proportions table -------
        _save_ancestry_proportions_table(ancestor_labels=demographic_model.population_indices.keys(),
                                        observed_autosome_proportions=autosome_proportions,
                                        predicted_autosome_proportions=autosomal_predicted_ancestries,
                                        output_dir=output_dir,
                                        output_filename_format=driver_spec.output.output_filename_format,
                                        observed_allosome_proportions=allosome_proportions if len(driver_spec.samples.allosomes) >= 1 else None,
                                        predicted_allosome_proportions=allosomal_predicted_ancestries,
                                        allosome_label=allosome_label)
        
        # ------ Produce output -------
        output_simulation_data_sex_biased(sample_population=pop,
                                        optimal_params=optimal_params,
                                        demographic_model=demographic_model,
                                        driver_spec=driver_spec,
                                        output_dir=output_dir,
                                        ad_model_autosomes=ad_model_autosomes,
                                        ad_model_allosomes=ad_model_allosomes,
                                        optimal_likelihood=optimal_likelihood,
                                        driver_path = driver_path
                                        )
    finally:
        close_log_file(log_filename=log_full_path)


# ----- Runner functions -----

def run_model_multi_init(model_func: Callable, bound_func: Callable, population: Population, 
                        start_params_list: list[np.ndarray], population_dict : dict, parameter_handler: FixedParametersHandler, 
                        max_iter: int=None, exclude_tracts_below_cM: int = 0, ad_model_autosomes = 'DC', 
                        ad_model_allosomes = 'DC', npts: int = 50, verbose_log: int = 0, verbose_screen:int = 0, 
                        two_steps_optimization: bool = True, autosomes_in_step_2: bool = True,
                        steps: list[int | str] | None = None, start_params_title: str | None = None,
                        print_start_params_table: bool = True, N_cores: int = 1,
                        rho_f: float = 1, rho_m: float = 1, TP: int = 2) -> tuple[list[np.ndarray], list[float], list[float | None]]:
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
    start_params_list: list[np.ndarray]
    	A list of initial parameter arrays to start the optimization.
    population_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    parameter_handler: FixedParametersHandler
        An object that handles parameter transformations and fixed parameters.
    max_iter: int, optional
        Maximum number of iterations for the optimization algorithm. Default is None, which means no limit.
    exclude_tracts_below_cM: int, optional
    	Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    ad_model_autosomes: str, optional
        The model to use for autosomal admixture. Must be one of 'DC', 'DF', 'M', 'H-DC' or 'H-DF'. Default is 'DC'.
    ad_model_allosomes: str or None, optional
        The model to use for allosomal admixture. Must be one of 'DC', 'DF', 'H-DC' or 'H-DF', or None if allosomal admixture is not to be modeled. Default is 'DC'.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
    verbose_log: int, optional
        Verbosity level for logging. Default is 0 (no verbose output). If greater than 0, iterations are logged every ``verbose_log`` steps.
    verbose_screen: int, optional
        Verbosity level for screen prints. Default is 0 (no verbose output). If greater than 0, iterations are printed every ``verbose_screen`` steps.
    two_steps_optimization: bool, optional
        Whether to use a two-step optimization procedure for sex-biased models. Default is True.
    autosomes_in_step_2: bool, optional
        If two_steps_optimization is True, whether both autosomal and allosomal data will be used in the second optimization step. If True, both types of data will be used. If False, only allosomal data will be used in the second step. Default is True.
    steps: list[int | str] | None, optional
        If two_steps_optimization is True, a list specifying which steps to run. Step 1 (non-sex-bias parameter optimization) can be denoted as 1 or 'step1', and step 2 (sex-bias parameter optimization)
        can be denoted as 2 or 'step2'. The only allowed combinations are step 1 only, step 2 only, or both steps.
        Examples of valid values are [1], ['step1'], [2], ['step2'], [1, 2], or ['step1', 'step2'].
        Mixed types are allowed, but duplicate references to the same step such as [1, 'step1'] are not. Default is None (both steps will be run).
    start_params_title: str or None, optional
        For internal use only. An optional title to display above the starting parameters table. If None, a default title will be generated based on the steps being run. Default is None.     
    print_start_params_table: bool, optional
        For internal use only. Whether to print the starting parameters table. Default is True.
    N_cores: int, optional
        The number of CPU cores to use for parallel processing, when the hybrid-pedigree refinements of the DF or DC models
        are used. Ignored if the hybrid-pedigree refinements are not used. Default is 1.
    rho_f: float, optional
        The female-specific recombination rate. Default is 1.
    rho_m: float, optional
        The male-specific recombination rate. Default is 1.
    TP: int, optional
        The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models.
        Default is 2. Ignored if not applicable. 
        
    Returns
    ----------
    tuple[list[np.ndarray], list[float], list[float | None]]
        A tuple containing three lists: (i) optimal parameters for each run, (ii) optimization likelihoods for each run, and (iii) optional
        full-data likelihoods (only populated when step 2 is run with allosomal data only).
    """
    if len(start_params_list) == 0:
        raise ValueError("start_params_list cannot be empty. Provide at least one starting-parameter set.")

    if print_start_params_table:
        _print_run_intro(
            parameter_handler=parameter_handler,
            demographic_model=parameter_handler.demography if hasattr(parameter_handler, "demography") else None,
            start_params_list=start_params_list,
            bound_func=bound_func,
            title_message=start_params_title,
            two_steps_optimization=two_steps_optimization,
            autosomes_in_step_2=autosomes_in_step_2,
            steps=steps,
        )

    optimal_params = []
    likelihoods = []
    full_likelihoods = []
    
    for start_params in start_params_list:
        opt_run_message = f"Optimization run #{len(optimal_params)+1}"
        print("\n" + opt_run_message + "\n")
        logger.info(opt_run_message)
        
        logger.debug(f'Starting parameters in optimizer units: {start_params}')
        params_found, likelihood_found, full_likelihood_found = run_model(model_func=model_func,
                                                                        bound_func=bound_func, 
                                                                        population=population, 
                                                                        startparams=start_params,
                                                                        population_dict=population_dict,
                                                                        parameter_handler=parameter_handler,
                                                                        max_iter=max_iter,
                                                                        exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                                        ad_model_autosomes=ad_model_autosomes,
                                                                        ad_model_allosomes=ad_model_allosomes,
                                                                        npts=npts,
                                                                        verbose_log=verbose_log,
                                                                        verbose_screen=verbose_screen,
                                                                        two_steps_optimization=two_steps_optimization,
                                                                        autosomes_in_step_2=autosomes_in_step_2,
                                                                        steps=steps,
                                                                        print_step_header=False,
                                                                        N_cores=N_cores,
                                                                        rho_f=rho_f,
                                                                        rho_m=rho_m,
                                                                        TP=TP)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
        full_likelihoods.append(full_likelihood_found)
    return optimal_params, likelihoods, full_likelihoods

def run_model(model_func: callable, bound_func: callable, population: Population, 
                        startparams: list, population_dict: dict, parameter_handler: FixedParametersHandler, max_iter: int | None = None, 
                        exclude_tracts_below_cM: float = 0, ad_model_autosomes: str = 'DC', ad_model_allosomes: str = 'DC',
                        npts: int = 0, verbose_log: int = 0, verbose_screen: int = 0, two_steps_optimization: bool = True,
                        autosomes_in_step_2: bool = True, steps: list[int | str] | None = None, print_step_header: bool = True, N_cores: int = 1,
                        rho_f: float = 1, rho_m: float = 1, TP: int = 2) -> tuple[np.ndarray, float, float | None]:
    
    """
    Runs the optimization for any demographic model, including sex-biased models. Works with only autosomal admixture or with both autosomal and allosomal admixture.
    
    Parameters
    ----------
    model_func: callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    bound_func: callable
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    startparams: list
        An array of initial parameters to start the optimization.
    population_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    parameter_handler: FixedParametersHandler
        An object that handles parameter transformations and fixed parameters.
    max_iter: int, optional
        Maximum number of iterations for the optimization algorithm. Default is None, which means no limit.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    ad_model_autosomes: str, optional
        The model to use for autosomal admixture. Must be one of 'DC', 'DF', 'M', 'H-DC' or 'H-DF'. Default is 'DC'.
    ad_model_allosomes: str, optional
        The model to use for allosomal admixture. Must be one of 'DC', 'DF', 'H-DC' or 'H-DF'. Default is 'DC'. If None, allosomal admixture will not be modeled.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
    verbose_log: int, optional
        Verbosity level for logging. Default is 0 (no verbose output). If greater than 0, iterations are logged every ``verbose_log`` steps.
    verbose_screen: int, optional
        Verbosity level for screen prints. Default is 0. If greater than 0, iterations are printed every ``verbose_screen`` steps.
    two_steps_optimization: bool, optional
        Whether to use a two-step optimization procedure for sex-biased models. If True, the optimization will first be run on non-sex bias parameters using only autosomal data. Then, a second optimization will be run with sex-bias parameters using both autosomal and allosomal data, starting from the results of the first optimization. Default is True.  
    autosomes_in_step_2: bool, optional
        If two_steps_optimization is True, whether both autosomal and allosomal data will be used in the second optimization step. If True, both types of data will be used. If False, only allosomal data will be used in the second step. Default is True.
    steps: list[int | str] | None, optional
        If two_steps_optimization is True, a list specifying which steps to run. Step 1 (non-sex-bias parameter optimization) can be denoted as 1 or 'step1', and step 2 (sex-bias parameter optimization)
        can be denoted as 2 or 'step2'. The only allowed combinations are step 1 only, step 2 only, or both steps.
        Examples of valid values are [1], ['step1'], [2], ['step2'], [1, 2], or ['step1', 'step2'].
        Mixed types are allowed, but duplicate references to the same step such as [1, 'step1'] are not. Default is None (both steps will be run).
    print_step_header: bool, optional
        If True, print the admixture-model title and step subtitle at the start of the optimization.
        If False, only the iteration table header is printed. For internal use only; set
        automatically by :func:`~tracts.driver.run_model_multi_init` to suppress repeated headers
        across multiple runs within the same step. Default is True.
    N_cores: int, optional
        The number of CPU cores to use for parallel processing, when the hybrid-pedigree refinements of the DF or DC models
        are used. Ignored if the hybrid-pedigree refinements are not used. Default is 1.
    rho_f: float, optional
        The female-specific recombination rate. Default is 1.
    rho_m: float, optional
        The male-specific recombination rate. Default is 1.
    TP: int, optional
        The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models.
        Default is 2. Ignored if not applicable. 
        
    Returns
    ----------
    tuple [np.ndarray, float, float | None]
        A tuple containing the optimal parameters found, the corresponding
        optimization likelihood, and an optional full-data likelihood.
    """
    if not two_steps_optimization:
        optimal_params, optimal_likelihood = optimize_cob_sex_biased_single_step(p0=startparams, 
                                                                    population=population,
                                                                    model_func=model_func, 
                                                                    parameter_handler=parameter_handler,
                                                                    outofbounds_fun=bound_func,
                                                                    p_dict = population_dict,
                                                                    exclude_tracts_below_cM=exclude_tracts_below_cM, 
                                                                    maxiter=max_iter,
                                                                    verbose_log=verbose_log,
                                                                    verbose_screen=verbose_screen,
                                                                    ad_model_autosomes=ad_model_autosomes, 
                                                                    ad_model_allosomes=ad_model_allosomes,
                                                                    npts=npts,
                                                                    print_step_header=print_step_header,
                                                                    N_cores=N_cores,
                                                                    rho_f=rho_f,
                                                                    rho_m=rho_m,
                                                                    TP=TP)
        full_data_likelihood = None
    else:
        optimal_params, optimal_likelihood, full_data_likelihood = optimize_cob_sex_biased_two_steps(p0=startparams, 
                                                                                                       population=population, 
                                                                                                       model_func=model_func, 
                                                                                                       parameter_handler= parameter_handler, 
                                                                                                       outofbounds_fun = bound_func, 
                                                                                                       p_dict = population_dict, 
                                                                                                       exclude_tracts_below_cM=exclude_tracts_below_cM, 
                                                                                                       maxiter=max_iter,
                                                                                                       verbose_log=verbose_log,
                                                                                                       verbose_screen=verbose_screen,
                                                                                                       ad_model_autosomes=ad_model_autosomes, 
                                                                                                       ad_model_allosomes=ad_model_allosomes,
                                                                                                       autosomes_in_step_2=autosomes_in_step_2,
                                                                                                       steps=steps,
                                                                                                       npts=npts,
                                                                                                       print_step_header=print_step_header,
                                                                                                       return_full_likelihood=True,
                                                                                                       N_cores=N_cores,
                                                                                                       rho_f=rho_f,
                                                                                                       rho_m=rho_m,
                                                                                                       TP=TP)
    
    return optimal_params, optimal_likelihood, full_data_likelihood
       
