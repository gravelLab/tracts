import io
import contextlib
import logging
from dataclasses import replace
from functools import partial
from typing import Callable
import numpy as np

from tracts.population import Population
from tracts.core import optimize_cob_sex_biased_single_step, optimize_cob_sex_biased_two_steps
from tracts.genetic_model import GeneticModel
from tracts.likelihood_options import LikelihoodOptions
from tracts.driver_utils import *
from tracts.driver_utils import (
    _print_run_intro,
    _normalize_multi_init_result,
    _summarize_step_results,
    _print_optimal_values_and_likelihood,
    _save_ancestry_proportions_table,
    _print_and_log,
    _build_step2_skip_message,
    _build_reoptimization_intro_message,
    _select_full_data_likelihood,
    _get_driver_for_reoptimization,
    _reorder_ancestry_proportions,
    _print_param_bounds_table
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
                            allosome_labels=driver_spec.samples.allosomes)
        pop.unknown_labels = driver_spec.optim.unknown_labels_for_smoothing
        pop.smooth_unknowns(allosome_labels=driver_spec.samples.allosomes)
        _bins, _data = pop.get_global_tractlengths(npts=driver_spec.optim.npts, # Get the population labels and validate that these correspond to to model population labels.
                                                   exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm)

        # ------ Load the demographic model -------
        demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names = load_demographic_model_from_driver(driver_spec=driver_spec,
                                                                                                                                script_dir=script_dir,
                                                                                                                                driver_path=driver_path,
                                                                                                                                allosome_label=allosome_label)
        parameter_handler = demographic_model.parameter_handler

        # ------ Narrow parameter bounds if specified in the driver -------
        parse_param_bounds(driver_spec.bounds, demographic_model)

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

        # ------ Optimization setup -------
        genetic_model = GeneticModel(
            demographic_model=demographic_model,
            ad_model_autosomes=ad_model_autosomes,
            ad_model_allosomes=ad_model_allosomes,
            rho_f=driver_spec.models.rho_f,
            rho_m=driver_spec.models.rho_m,
            TP=driver_spec.models.TP,
            N_cores=driver_spec.optim.N_cores,
        )
        likelihood_options = LikelihoodOptions(
            verbose_log=driver_spec.output.verbose_log,
            verbose_screen=driver_spec.output.verbose_screen,
        )
        # Time parameters need to be rescaled for some optimizers; this always rescales via
        # parameter_handler before computing migration matrices. Also needed after optimization,
        # to compute predicted ancestry proportions at the optimal parameters.
        model_func = genetic_model.model_func

        # Show time-admissibility warnings only during optimization and final reporting.
        parameter_handler.enable_time_param_logging = False

        # ------ Print parameter bounds table ------
        _print_param_bounds_table(demographic_model=demographic_model)

        # ------ Compute starting parameters in physical units ------
        physical_start_params = compute_physical_start_params(driver_spec=driver_spec,
                                                            demographic_model=demographic_model,
                                                            sex_bias_param_names=sex_bias_param_names,
                                                            non_sex_bias_param_names=non_sex_bias_param_names,
                                                            step_label="step 1")

        check_start_params(physical_start_params=physical_start_params,
                          model_param_names=model_param_names) # Checks compatibility with demographic model and prints message


        # ------ Run the optimization ------
        run_optimization_fixed_options = partial(run_optimization, 
                                                genetic_model=genetic_model,
                                                population=pop,
                                                likelihood_options=likelihood_options,
                                                model_param_names=model_param_names,
                                                sex_bias_param_names=sex_bias_param_names)


        optimal_params, optimal_likelihood = run_optimization_fixed_options(physical_start_params=physical_start_params,
                                                                            driver_spec=driver_spec)


        # ------ Perform sex-bias-fixing based re-optimization -------
        _has_free_sex_bias = has_free_sex_bias_parameters(parameter_handler, sex_bias_param_names)

        if ad_model_allosomes is not None and _has_free_sex_bias and driver_spec.optim.n_reoptimizations > 0:

            optimal_params, optimal_likelihood = run_sex_bias_fixing_reoptimizations(driver_spec=driver_spec,
                                                                                    model_param_names=model_param_names,
                                                                                    optimal_params=optimal_params,
                                                                                    optimal_likelihood=optimal_likelihood,
                                                                                    run_optimization_fixed_options=run_optimization_fixed_options)

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
        optimal_sex_bias_at_boundaries = check_optimal_sex_bias_parameters_at_boundaries(demographic_model=demographic_model,
                                                                                        driver_spec=driver_spec,
                                                                                        sex_bias_param_names=sex_bias_param_names,
                                                                                        remainder_params=remainder_params,
                                                                                        optimal_params=optimal_params)

        # ------ Re-optimize if any sex-bias parameters are at boundaries ------
        if len(optimal_sex_bias_at_boundaries) > 0 and driver_spec.optim.rerun_optimization_on_boundaries:

            reload_context = ModelReloadContext(script_dir=script_dir,
                                                driver_path=driver_path,
                                                allosome_label=allosome_label,
                                                autosome_proportions=autosome_proportions,
                                                allosome_proportions=allosome_proportions)

            (driver_spec, genetic_model, optimal_params, optimal_likelihood,
             autosome_proportions, allosome_proportions) = run_boundary_reoptimization(driver_spec=driver_spec,
                                                                                        reload_context=reload_context,
                                                                                        optimal_sex_bias_at_boundaries=optimal_sex_bias_at_boundaries,
                                                                                        genetic_model=genetic_model,
                                                                                        optimal_params=optimal_params,
                                                                                        optimal_likelihood=optimal_likelihood,
                                                                                        remainder_params=remainder_params,
                                                                                        population=pop,
                                                                                        likelihood_options=likelihood_options)

            demographic_model = genetic_model.demographic_model
            model_func = genetic_model.model_func

        # ------ Check for founding migration rates > 1 in the final parameters ------
        check_final_parameters(demographic_model=demographic_model,
                               optimal_params=optimal_params)

        # ------ Check whether any optimal parameter is close to its admissible bounds ------
        check_optimal_params_near_bounds(demographic_model=demographic_model,
                                         optimal_params=optimal_params,
                                         tol=driver_spec.optim.bounds_proximity_tol)

        # ------ Compute and print ancestry proportions predicted by the model ------
        autosomal_predicted_ancestries, allosomal_predicted_ancestries = get_predicted_ancestry_proportions(demographic_model=demographic_model,
                                                                                                            model_func=model_func,
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
                                        optimal_likelihood=optimal_likelihood,
                                        genetic_model=genetic_model,
                                        driver_spec=driver_spec,
                                        output_dir=output_dir,
                                        driver_path=driver_path
                                        )
    finally:
        close_log_file(log_filename=log_full_path)




# ----- Runner functions -----

def run_optimization(physical_start_params: list, genetic_model: GeneticModel,
                                    population: Population, driver_spec, likelihood_options: LikelihoodOptions,
                                    model_param_names: list, sex_bias_param_names: list,
                                    print_run_details: bool = True) -> tuple:
    """
    Runs the optimization (single-step, or two-step with the step 1 / step 2 / step-2-skipped
    branches) given a list of starting parameters in physical units, returning the optimal
    parameters and the corresponding likelihood.

    Parameters
    ----------
    physical_start_params: list[np.ndarray]
        Starting parameter sets, in physical units, to start the optimization from (one
        optimization run per entry).
    genetic_model: GeneticModel
        Bundles the demographic model being fit (accessible as ``genetic_model.demographic_model``,
        which also provides ``parameter_handler`` for unit conversions and fixed-parameter state)
        with the admixture and phase-type model configuration used to compute the likelihood.
    population: :class:`tracts.population.Population`
        The population object containing the data to fit.
    driver_spec: InferenceConfig
        The parsed driver-file configuration; controls whether a single-step or two-step
        optimization is run, and its iteration/verbosity settings.
    likelihood_options: LikelihoodOptions
        Logging verbosity for this run. Its ``include_autosomes``/``include_allosomes`` flags
        are ignored here: which data are included is determined per step, as in
        :func:`~tracts.core.optimize_cob_sex_biased_two_steps`.
    model_param_names: list[str]
        Names of all of the demographic model's free base parameters, in order.
    sex_bias_param_names: list[str]
        Names of the sex-bias parameters among ``model_param_names``.
    print_run_details: bool
        Whether to print the starting-parameters table under each step's header, the
        "Optimization run #N" line, and the step-transition messages ("Selecting best parameters
        from step X..."). The step headers themselves and the optimizer's own iteration table are
        always printed. Set to False for quick, low-noise repeated re-optimizations (e.g.
        ``run_sex_bias_fixing_reoptimizations``) where only one run per step is performed and
        these details add clutter rather than information. Defaults to True.

    Returns
    -------
    tuple [np.ndarray, float]
        The optimal parameters found (in physical units) and the corresponding likelihood.
    """
    demographic_model = genetic_model.demographic_model
    parameter_handler = demographic_model.parameter_handler
    ad_model_allosomes = genetic_model.phase_type_config.ad_model_allosomes
    model_func = genetic_model.model_func
    bound_func = genetic_model.outofbounds_fun

    def _run_stage(start_params: list, two_steps_optimization: bool, start_params_title: str = None,
                  steps: list = None, autosomes_in_step_2: bool = True, stage_likelihood_options: LikelihoodOptions = likelihood_options):
        """
        Runs one optimization stage (single-step, or step 1 / step 2 of a two-step run)
        over all of ``start_params``, sharing the setup common to every stage of this run
        (``genetic_model``, ``population``, iteration/verbosity settings).
        """
        return _normalize_multi_init_result(run_model_multi_init(genetic_model=genetic_model,
                                                                population=population,
                                                                start_params_list=start_params,
                                                                population_dict=demographic_model.population_indices.items(),
                                                                likelihood_options=stage_likelihood_options,
                                                                max_iter=driver_spec.optim.maximum_iterations,
                                                                exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm,
                                                                npts=driver_spec.optim.npts,
                                                                two_steps_optimization=two_steps_optimization,
                                                                autosomes_in_step_2=autosomes_in_step_2,
                                                                steps=steps,
                                                                start_params_title=start_params_title,
                                                                print_start_params_table=False,
                                                                print_run_number=print_run_details)
        )

    # ------ Convert starting parameters to optimizer units ------
    optimizer_start_params = [parameter_handler.convert_to_optimizer_params(params) for params in physical_start_params]

    # ------ Get starting ancestry proportions for the starting parameters ------
    get_starting_ancestry_proportions(demographic_model=demographic_model,
                                        model_func=model_func,
                                        optimizer_start_params=optimizer_start_params) # The computed proportions are only logged.

    parameter_handler.enable_time_param_logging = True

    # ------ Run the model with (multiple) starting parameters ------
    step_1_start_params_title = "Starting parameters for step 1 optimization" if driver_spec.optim.two_steps_optimization else "Starting parameters for single-step optimization"

    if driver_spec.optim.two_steps_optimization is False: # Single-step optimization using optimize_cob_sex_biased_single_step

        _print_run_intro(parameter_handler=parameter_handler,
                        demographic_model=demographic_model,
                        start_params_list=optimizer_start_params,
                        bound_func=bound_func,
                        title_message=step_1_start_params_title,
                        two_steps_optimization=False,
                        autosomes_in_step_2=True,
                        print_start_params_table=print_run_details)

        params_found, likelihoods, _full_likelihoods = _run_stage(start_params=optimizer_start_params,
                                                                two_steps_optimization=False,
                                                                start_params_title=step_1_start_params_title)

        optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found,
                                                                    likelihoods=likelihoods,
                                                                    parameter_handler=parameter_handler,
                                                                    param_names=model_param_names,
                                                                    likelihood_tolerance=driver_spec.optim.repetitions_likelihood_tolerance)

    else: # Performs two-steps optimization with multiple starting parameters

        # ------ Step 1: optimize non-sex-bias parameters on autosomal data ------

        _print_run_intro(parameter_handler=parameter_handler,
                        demographic_model=demographic_model,
                        start_params_list=optimizer_start_params,
                        bound_func=bound_func,
                        title_message=step_1_start_params_title,
                        two_steps_optimization=True,
                        autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias,
                        steps=[1],
                        print_start_params_table=print_run_details)

        params_found_step_1, likelihoods_step_1, _full_likelihoods_step_1 = _run_stage(start_params=optimizer_start_params,
                                                                                    two_steps_optimization=True,
                                                                                    start_params_title=step_1_start_params_title,
                                                                                    steps=[1],
                                                                                    autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias) # This parameter is ignored in step 1

        #  Process and print results
        optimal_params_step_1, _optimal_likelihood_step_1 = _summarize_step_results(params_found=params_found_step_1,
                                                                                    likelihoods=likelihoods_step_1,
                                                                                    parameter_handler=parameter_handler,
                                                                                    param_names=model_param_names,
                                                                                    step_label="Step 1",
                                                                                    likelihood_tolerance=driver_spec.optim.repetitions_likelihood_tolerance)

        if ad_model_allosomes is None:
            logger.info("No allosomal data provided. Skipping Step 2 and using Step 1 results.")
            optimal_params, optimal_likelihood = optimal_params_step_1, _optimal_likelihood_step_1
        else:
            # ------ Step 2: optimize sex-bias parameters on (autosomal and) allosomal data ------

            # Detect whether there are any sex-bias parameters that are free (not fixed by ancestry
            # or by value). When all sex-bias params are fixed, step 2 has no free variables: skip
            # all the verbose setup/results output and go straight to the final table.
            _has_free_sex_bias = has_free_sex_bias_parameters(parameter_handler, sex_bias_param_names)

            # Draw fresh sex-bias starts for step 2, while keeping the best step 1 non-sex-bias values fixed.
            step_2_fixed_param_values = {
                name: float(value)
                for name, value in zip(model_param_names, optimal_params_step_1)
                if name not in sex_bias_param_names
            }
            # Sex-bias parameters fixed by the user at a specific value must not be resampled.
            for _sbv_name, _sbv_value in parameter_handler.user_params_fixed_by_value.items():
                if _sbv_name in sex_bias_param_names:
                    step_2_fixed_param_values[_sbv_name] = _sbv_value

            if _has_free_sex_bias:
                if print_run_details:
                    _print_and_log("Selecting best parameters from step 1 and proceeding to step 2 optimization.\n")

                step_2_physical_start_params = parse_start_params(start_param_bounds=driver_spec.start_params,
                                                                repetitions=driver_spec.optim.repetitions,
                                                                seed=driver_spec.optim.seed,
                                                                demographic_model=demographic_model,
                                                                sample_param_names=set(sex_bias_param_names),
                                                                fixed_param_values=step_2_fixed_param_values)

                step_2_physical_start_params = collapse_identical_start_params(step_2_physical_start_params, "step 2")
                step_2_start_params = [
                    parameter_handler.convert_to_optimizer_params(params)
                    for params in step_2_physical_start_params
                ]

                step_2_start_params_title = (
                    "Starting parameters for step 2 optimization "
                    "(non-sex-bias parameters are fixed to the best step 1 estimates)."
                )

                _print_run_intro(parameter_handler=parameter_handler,
                                demographic_model=demographic_model,
                                start_params_list=step_2_start_params,
                                bound_func=bound_func,
                                title_message=step_2_start_params_title,
                                two_steps_optimization=True,
                                autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias,
                                steps=[2],
                                print_start_params_table=print_run_details)

                params_found_step_2, likelihoods_step_2, full_likelihoods_step_2 = _run_stage(start_params=step_2_start_params,
                                                                                            two_steps_optimization=True,
                                                                                            start_params_title=step_2_start_params_title,
                                                                                            steps=[2],
                                                                                            autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias)

                #  Process and print results
                optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found_step_2,
                                                                            likelihoods=likelihoods_step_2,
                                                                            parameter_handler=parameter_handler,
                                                                            param_names=model_param_names,
                                                                            step_label="Step 2",
                                                                            likelihood_tolerance=driver_spec.optim.repetitions_likelihood_tolerance)

                if print_run_details:
                    _print_and_log("Selecting best parameters from step 2.")

                optimal_likelihood = _select_full_data_likelihood(likelihoods_step_2=likelihoods_step_2,
                                                                full_likelihoods_step_2=full_likelihoods_step_2,
                                                                optimal_likelihood=optimal_likelihood,
                                                                use_autosomes_for_sex_bias=driver_spec.optim.use_autosomes_for_sex_bias,
                                                                announce=print_run_details)

            else:
                # No free sex-bias parameters: run step 2 silently (only to compute the
                # full-data likelihood at the step-1 optimal params) then skip to final table.
                _print_and_log(_build_step2_skip_message(sex_bias_param_names, parameter_handler))
                _silent_start = [parameter_handler.convert_to_optimizer_params(optimal_params_step_1)]
                _tracts_logger = logging.getLogger("tracts")
                _saved_tracts_level = _tracts_logger.level
                _tracts_logger.setLevel(logging.CRITICAL)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        params_found_step_2, likelihoods_step_2, full_likelihoods_step_2 = _run_stage(start_params=_silent_start,
                                                                                                    two_steps_optimization=True,
                                                                                                    steps=[2],
                                                                                                    autosomes_in_step_2=driver_spec.optim.use_autosomes_for_sex_bias,
                                                                                                    stage_likelihood_options=likelihood_options.with_overrides(verbose_log=0, verbose_screen=0))
                finally:
                    _tracts_logger.setLevel(_saved_tracts_level)

                optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found_step_2,
                                                                            likelihoods=likelihoods_step_2,
                                                                            parameter_handler=parameter_handler,
                                                                            param_names=model_param_names,
                                                                            step_label="Step 2",
                                                                            likelihood_tolerance=driver_spec.optim.repetitions_likelihood_tolerance)

                optimal_likelihood = _select_full_data_likelihood(likelihoods_step_2=likelihoods_step_2,
                                                                  full_likelihoods_step_2=full_likelihoods_step_2,
                                                                  optimal_likelihood=optimal_likelihood,
                                                                  use_autosomes_for_sex_bias=driver_spec.optim.use_autosomes_for_sex_bias,
                                                                  announce=False)

    return optimal_params, optimal_likelihood


def run_sex_bias_fixing_reoptimizations(driver_spec, model_param_names: list[str], optimal_params: np.ndarray,
                                        optimal_likelihood: float, run_optimization_fixed_options: Callable) -> tuple[np.ndarray, float]:
    """
    Repeats ``driver_spec.optim.n_reoptimizations`` times: fixing the sex-bias parameters at
    their most recently optimized values and re-running the optimization starting from the
    current optimal parameters (using a copy of ``driver_spec`` with ``repetitions=1`` and
    starting parameters set to the current optimum, see ``_get_driver_for_reoptimization``; this
    also means, per ``core.py``, that sex-bias parameters end up fixed only during step 1,
    remaining free to be optimized again in step 2). Stops early if the likelihood no longer
    improves between repetitions. Since only one run per step is performed, each repetition runs
    with ``print_run_details=False`` (see ``run_optimization``): step headers and the optimizer's
    own iteration table are still shown, but the starting-parameters table, "Optimization run #N"
    line, and step-transition messages are suppressed.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The current driver-file configuration; controls the number of repetitions
        (``driver_spec.optim.n_reoptimizations``).
    model_param_names: list[str]
        Names of all of the current demographic model's free base parameters, in order.
    optimal_params: np.ndarray
        The current optimal parameters (physical units), before these re-optimizations.
    optimal_likelihood: float
        The current optimal likelihood, before these re-optimizations.
    run_optimization_fixed_options: Callable
        A partially-applied ``run_optimization`` (with ``genetic_model``, ``population``,
        ``likelihood_options``, ``model_param_names``, and ``sex_bias_param_names`` already
        bound), taking ``physical_start_params`` and ``driver_spec`` as its remaining arguments.

    Returns
    -------
    tuple[np.ndarray, float]
        The updated optimal parameters and likelihood after the re-optimization repetitions.
    """
    _print_and_log(_build_reoptimization_intro_message(driver_spec.optim.n_reoptimizations))

    reopt_driver_spec = _get_driver_for_reoptimization(driver_spec=driver_spec,
                                                       model_param_names=model_param_names,
                                                       optimal_params=optimal_params)

    # Quiet mode: only one run per step is performed here, so the starting-parameters table,
    # "Optimization run #N" line, and step-transition messages are just clutter. Step headers and
    # the optimizer's own iteration table are always shown regardless.
    run_reoptimization = partial(run_optimization_fixed_options,
                                driver_spec=reopt_driver_spec,
                                print_run_details=False)

    convergence = False
    for _i in range(driver_spec.optim.n_reoptimizations):

        # Run the optimization again, starting from the just-optimized parameters
        # This makes that during the first step, sex-bias parameters are fixed at their optimal values from the previous optimization
        _print_and_log(f"\nRe-optimization {_i + 1}/{driver_spec.optim.n_reoptimizations}: "
                       f"re-optimizing starting from the current optimal parameters (likelihood = {optimal_likelihood:.6f}).")

        optimal_params, optimal_likelihood_new = run_reoptimization([optimal_params])

        if np.isclose(optimal_likelihood_new, optimal_likelihood, atol=driver_spec.optim.reoptimization_likelihood_tolerance):
            _print_and_log(f"No further improvement in likelihood after { _i + 1 } repetitions. Re-optimization completed.")
            optimal_likelihood = optimal_likelihood_new
            convergence = True
            break
        else:
            _print_and_log(f"Change in likelihood from {optimal_likelihood:.6f} to {optimal_likelihood_new:.6f} after re-optimizing.")
            optimal_likelihood = optimal_likelihood_new

    if not convergence:
        _print_and_log(f"Convergence not achieved after {driver_spec.optim.n_reoptimizations} repetitions. Stopping re-optimization.")

    return optimal_params, optimal_likelihood


def run_boundary_reoptimization(driver_spec, reload_context: ModelReloadContext, optimal_sex_bias_at_boundaries: list[str],
                                genetic_model: GeneticModel, optimal_params: np.ndarray, optimal_likelihood: float,
                                remainder_params: dict, population: Population, likelihood_options: LikelihoodOptions):
    """
    Repeatedly re-optimizes while one or more sex-bias parameters have an optimal value at a +-1
    boundary. Each iteration: all boundary-hitting sex-bias parameters that are directly optimized
    (present in ``model_base_params``) are fixed by value (at +-``near_one``) for the
    re-optimization. A boundary hit on the implicit population's derived sex-bias parameter cannot
    be fixed this way; instead, this tries to switch to a different explicit source population
    (from the same founder event) whose sex-bias parameter is not itself at a boundary, and use
    that as the implicit population instead (see ``get_alternate_implicit_population``).

    After each re-optimization, if the likelihood has not improved over the state before that
    iteration, the loop stops immediately, printing/logging a message stating so, and the state
    from before that iteration (the original input, if this was the first iteration) is returned.
    Otherwise, the improvement is kept and, unless no sex-bias parameter remains free to fix (all
    have already been fixed by value), the newly-updated optimal parameters are checked again for
    boundary hits: if any (previously free) sex-bias parameter is now at a boundary too, the loop
    repeats to fix and re-optimize those as well. The loop otherwise stops once no sex-bias
    parameter is at a boundary any more, or if neither of the two fixing mechanisms above applies
    (e.g. the boundary hit is on the implicit population and no alternate is available).

    Each re-optimization resumes from the previous optimization's result: all starting parameters
    are pinned to their previous optimal value (no resampling), so there is a single repetition,
    and (as in ``run_sex_bias_fixing_reoptimizations``) it runs with ``print_run_details=False``
    to keep its console output brief.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The current driver-file configuration.
    reload_context: ModelReloadContext
        File-location and ancestry-proportion context needed to reload the demographic model from
        the model YAML file, used only if the implicit population is changed.
    optimal_sex_bias_at_boundaries: list[str]
        Parameter names near their +-1 boundary, as returned by
        ``check_optimal_sex_bias_parameters_at_boundaries``.
    genetic_model: GeneticModel
        The current genetic model. Its demographic model's parameter names are derived directly
        (see ``get_param_names_by_type``) rather than passed in separately.
    optimal_params: np.ndarray
        The current optimal parameters (physical units), before this re-optimization.
    optimal_likelihood: float
        The current optimal likelihood, before this re-optimization.
    remainder_params: dict
        The remainder (derived) parameters computed from ``optimal_params``, as returned by
        ``compute_remainder_params``. Used, if the implicit population is changed, to set starting
        values for the population that was previously implicit and is now explicit.
    population: :class:`tracts.population.Population`
        The population object containing the data to fit.
    likelihood_options: LikelihoodOptions
        Logging verbosity for this run.

    Returns
    -------
    tuple[InferenceConfig, GeneticModel, np.ndarray, float, np.ndarray, np.ndarray | list]
        The driver spec and genetic model, the optimal parameters and likelihood, and
        ``reload_context.autosome_proportions``/``allosome_proportions`` realigned to the
        population order of the returned genetic model's demographic model (see
        ``_reorder_ancestry_proportions``): switching the implicit population reorders
        ``demographic_model.population_indices`` (the implicit population is always placed last;
        see ``ParametrizedDemography(SexBiased).load_from_YAML``) -- all reflecting the last
        iteration whose re-optimization improved the likelihood, or the original input if none did
        (including if no iteration was ever attempted).
    """
    autosome_proportions, allosome_proportions = reload_context.autosome_proportions, reload_context.allosome_proportions

    demographic_model = genetic_model.demographic_model
    model_param_names, sex_bias_param_names, _ = get_param_names_by_type(demographic_model)

    while True:
        # Only directly-optimized parameters (present in model_base_params) can be fixed by value; a
        # boundary hit on the implicit population's derived sex-bias parameter is instead addressed by
        # changing which population is implicit (if possible). Fixed at +-near_one rather than the
        # actual optimal value (which can be less extreme, e.g. 1 - boundary_tol).
        boundary_fixed_param_values = {
            name: float(np.sign(value)) * driver_spec.optim.near_one
            for name, value in zip(model_param_names, optimal_params)
            if name in optimal_sex_bias_at_boundaries
        }

        # If any boundary-hitting parameter corresponds to the implicit population, try to switch to a
        # different explicit source population (from the same founder event) whose sex-bias parameter
        # is not itself at a boundary. If none is available, get_alternate_implicit_population already
        # prints/logs an informative message and the implicit population is left unchanged.
        alternate_implicit_population = get_alternate_implicit_population(
            demographic_model=demographic_model, optimal_sex_bias_at_boundaries=optimal_sex_bias_at_boundaries)

        if not boundary_fixed_param_values and alternate_implicit_population is None:
            _print_and_log(
                "None of the boundary-hitting sex-bias "
                f"parameter(s) ({', '.join(optimal_sex_bias_at_boundaries)}) can be fixed by value, "
                "and no alternate implicit population is available to resolve the boundary hit. "
                "Stopping boundary re-optimization."
            )
            break

        ancestor_labels_before_reopt = list(demographic_model.population_indices.keys())

        (new_driver_spec, new_genetic_model, new_model_param_names, new_sex_bias_param_names, _new_non_sex_bias_param_names,
         reopt_physical_start_params) = build_boundary_reoptimization_model(driver_spec=driver_spec,
                                                                            reload_context=reload_context,
                                                                            boundary_fixed_param_values=boundary_fixed_param_values,
                                                                            genetic_model=genetic_model,
                                                                            optimal_params=optimal_params,
                                                                            remainder_params=remainder_params,
                                                                            alternate_implicit_population=alternate_implicit_population)

        new_optimal_params, new_optimal_likelihood = run_optimization(physical_start_params=reopt_physical_start_params,
                                                            genetic_model=new_genetic_model,
                                                            population=population,
                                                            driver_spec=new_driver_spec,
                                                            likelihood_options=likelihood_options,
                                                            model_param_names=new_model_param_names,
                                                            sex_bias_param_names=new_sex_bias_param_names,
                                                            print_run_details=False)

        if new_optimal_likelihood <= optimal_likelihood:
            _print_and_log(
                f"Boundary re-optimization did not improve the likelihood "
                f"({new_optimal_likelihood:.6f} after vs. {optimal_likelihood:.6f} before this re-optimization step). "
                "Keeping the parameters and likelihood from before this re-optimization step."
            )
            break

        driver_spec, genetic_model = new_driver_spec, new_genetic_model
        optimal_params, optimal_likelihood = new_optimal_params, new_optimal_likelihood
        model_param_names, sex_bias_param_names = new_model_param_names, new_sex_bias_param_names
        demographic_model = genetic_model.demographic_model

        remainder_params = compute_remainder_params(demographic_model=demographic_model,
                                                    migration_matrices=demographic_model.get_migration_matrices(optimal_params))

        _print_optimal_values_and_likelihood(demographic_model=demographic_model,
                                            optimal_params=optimal_params,
                                            optimal_likelihood=optimal_likelihood,
                                            remainder_parameters=remainder_params,
                                            ad_model_allosomes=genetic_model.phase_type_config.ad_model_allosomes)

        autosome_proportions, allosome_proportions = _reorder_ancestry_proportions(
            old_ancestor_labels=ancestor_labels_before_reopt,
            new_ancestor_labels=list(demographic_model.population_indices.keys()),
            autosome_proportions=autosome_proportions,
            allosome_proportions=allosome_proportions)

        # Keep reload_context's proportions in sync with the current population order: if a later
        # iteration also switches the implicit population, build_boundary_reoptimization_model
        # reorders reload_context.autosome_proportions/allosome_proportions assuming they still
        # match the (just-accepted) genetic_model's population order (which is only true if this
        # is kept up to date across iterations).
        reload_context = replace(reload_context, autosome_proportions=autosome_proportions,
                                 allosome_proportions=allosome_proportions)

        # Stop once no sex-bias parameter remains free to fix (all have hit the boundary and been
        # fixed already, over this and/or previous iterations): nothing is left to check or
        # re-optimize further.
        if not has_free_sex_bias_parameters(demographic_model.parameter_handler, sex_bias_param_names):
            _print_and_log("All sex-bias parameters have been fixed by value. Boundary re-optimization completed.")
            break

        # Check whether fixing the previous boundary-hitters revealed new ones among the remaining
        # free sex-bias parameters; if so, loop again to fix and re-optimize those too.
        optimal_sex_bias_at_boundaries = check_optimal_sex_bias_parameters_at_boundaries(
            demographic_model=demographic_model,
            driver_spec=driver_spec,
            sex_bias_param_names=sex_bias_param_names,
            remainder_params=remainder_params,
            optimal_params=optimal_params)

        if len(optimal_sex_bias_at_boundaries) == 0:
            _print_and_log("No free sex-bias parameters remain at a boundary. Boundary re-optimization completed.")
            break

    return driver_spec, genetic_model, optimal_params, optimal_likelihood, autosome_proportions, allosome_proportions


def run_model_multi_init(genetic_model: GeneticModel, population: Population,
                        start_params_list: list[np.ndarray], population_dict : dict,
                        likelihood_options: LikelihoodOptions | None = None,
                        max_iter: int=None, exclude_tracts_below_cM: int = 0, npts: int = 50,
                        two_steps_optimization: bool = True, autosomes_in_step_2: bool = True,
                        steps: list[int | str] | None = None, start_params_title: str | None = None,
                        print_start_params_table: bool = True, print_run_number: bool = True) -> tuple[list[np.ndarray], list[float], list[float | None]]:
    """
    Runs the model multiple times with different initial parameters.

    Parameters
    ----------
    genetic_model: GeneticModel
        Bundles the demographic model (whose ``parameter_handler`` handles parameter
        transformations and fixed parameters, and whose ``model_func``/``outofbounds_fun``
        methods compute migration matrices and violation scores) with the admixture and
        phase-type model configuration (``ad_model_autosomes``, ``ad_model_allosomes``,
        ``rho_f``, ``rho_m``, ``TP``, ``N_cores``) used to compute the likelihood.
    population: :class:`tracts.population.Population`
        The population object containing individual data.
    start_params_list: list[np.ndarray]
    	A list of initial parameter arrays to start the optimization.
    population_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    likelihood_options: LikelihoodOptions | None
        Logging verbosity (``verbose_log``, ``verbose_screen``) for this run. If None,
        defaults to ``LikelihoodOptions()``.
    max_iter: int, optional
        Maximum number of iterations for the optimization algorithm. Default is None, which means no limit.
    exclude_tracts_below_cM: int, optional
    	Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
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
    print_run_number: bool, optional
        For internal use only. Whether to print the "Optimization run #N" line before each run.
        Default is True.

    Returns
    ----------
    tuple[list[np.ndarray], list[float], list[float | None]]
        A tuple containing three lists: (i) optimal parameters for each run, (ii) optimization likelihoods for each run, and (iii) optional
        full-data likelihoods (only populated when step 2 is run with allosomal data only).
    """
    if len(start_params_list) == 0:
        raise ValueError("start_params_list cannot be empty. Provide at least one starting-parameter set.")

    parameter_handler = genetic_model.parameter_handler

    if print_start_params_table:
        _print_run_intro(
            parameter_handler=parameter_handler,
            demographic_model=genetic_model.demographic_model,
            start_params_list=start_params_list,
            bound_func=genetic_model.outofbounds_fun,
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
        if print_run_number:
            print("\n" + opt_run_message + "\n")
        logger.info(opt_run_message)

        logger.debug(f'Starting parameters in optimizer units: {start_params}')
        params_found, likelihood_found, full_likelihood_found = run_model(genetic_model=genetic_model,
                                                                        population=population,
                                                                        startparams=start_params,
                                                                        population_dict=population_dict,
                                                                        likelihood_options=likelihood_options,
                                                                        max_iter=max_iter,
                                                                        exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                                        npts=npts,
                                                                        two_steps_optimization=two_steps_optimization,
                                                                        autosomes_in_step_2=autosomes_in_step_2,
                                                                        steps=steps,
                                                                        print_step_header=False)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
        full_likelihoods.append(full_likelihood_found)
    return optimal_params, likelihoods, full_likelihoods

def run_model(genetic_model: GeneticModel, population: Population,
                        startparams: list, population_dict: dict,
                        likelihood_options: LikelihoodOptions | None = None, max_iter: int | None = None,
                        exclude_tracts_below_cM: float = 0,
                        npts: int = 0, two_steps_optimization: bool = True,
                        autosomes_in_step_2: bool = True, steps: list[int | str] | None = None, print_step_header: bool = True
                        ) -> tuple[np.ndarray, float, float | None]:

    """
    Runs the optimization for any demographic model, including sex-biased models. Works with only autosomal admixture or with both autosomal and allosomal admixture.

    Parameters
    ----------
    genetic_model: GeneticModel
        Bundles the demographic model (whose ``parameter_handler`` handles parameter
        transformations and fixed parameters, and whose ``model_func``/``outofbounds_fun``
        methods compute migration matrices and violation scores) with the admixture and
        phase-type model configuration (``ad_model_autosomes``, ``ad_model_allosomes``,
        ``rho_f``, ``rho_m``, ``TP``, ``N_cores``) used to compute the likelihood.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    startparams: list
        An array of initial parameters to start the optimization.
    population_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    likelihood_options: LikelihoodOptions | None
        Logging verbosity (``verbose_log``, ``verbose_screen``) for this run. If None,
        defaults to ``LikelihoodOptions()``.
    max_iter: int, optional
        Maximum number of iterations for the optimization algorithm. Default is None, which means no limit.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
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

    Returns
    ----------
    tuple [np.ndarray, float, float | None]
        A tuple containing the optimal parameters found, the corresponding
        optimization likelihood, and an optional full-data likelihood.
    """
    if not two_steps_optimization:
        optimal_params, optimal_likelihood = optimize_cob_sex_biased_single_step(p0=startparams,
                                                                                population=population,
                                                                                genetic_model=genetic_model,
                                                                                likelihood_options=likelihood_options,
                                                                                p_dict = population_dict,
                                                                                exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                                                maxiter=max_iter,
                                                                                npts=npts,
                                                                                print_step_header=print_step_header)
        full_data_likelihood = None
    else:
        optimal_params, optimal_likelihood, full_data_likelihood = optimize_cob_sex_biased_two_steps(p0=startparams,
                                                                                                    population=population,
                                                                                                    genetic_model=genetic_model,
                                                                                                    likelihood_options=likelihood_options,
                                                                                                    p_dict = population_dict,
                                                                                                    exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                                                                    maxiter=max_iter,
                                                                                                    autosomes_in_step_2=autosomes_in_step_2,
                                                                                                    steps=steps,
                                                                                                    npts=npts,
                                                                                                    print_step_header=print_step_header,
                                                                                                    return_full_likelihood=True)

    return optimal_params, optimal_likelihood, full_data_likelihood
