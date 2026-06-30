import io
import contextlib
import logging
from pathlib import Path
from typing import Callable
import numpy as np
import os

from tracts.population import Population
from tracts.core import optimize_cob_sex_biased_single_step, optimize_cob_sex_biased_two_steps
from tracts.util import time_to_physical_function, rate_to_physical_function, sex_bias_to_physical_function, time_to_optimizer_function, rate_to_optimizer_function, sex_bias_to_optimizer_function
from tracts.demography.parameter import ParamType
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.driver_utils import locate_file_path, load_driver_file, load_population, load_model_from_driver, get_time_scaled_model_func, get_time_scaled_model_bounds, parse_start_params, collapse_identical_start_params, output_simulation_data_sex_biased, _summarize_step_results, _normalize_multi_init_result, _print_run_intro, _compute_remainder_params, _save_ancestry_proportions_table
from tracts.logs import setup_logger, set_log_file, close_log_file
from datetime import datetime

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

    # ------ Set up logging using filename from driver-------
    logger, memory_handler = setup_logger()
    if driver_spec.output.log_filename:
        breakpoint()
        log_filename = Path(driver_spec.output.log_filename)
    else:
        log_filename = Path("tracts.log")
        logger.warning(f"No log filename specified in driver file. Defaulting to {log_filename} in the working directory.")
    
    if not driver_spec.output.output_directory:
        logger.warning("No output directory specified in driver file. Defaulting to current working directory.")
        output_dir = Path.cwd()
        driver_spec.output.output_directory = str(output_dir)
    else:
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_output_directory =  driver_spec.output.output_directory.format(date=timestamp)
        output_dir = Path(formatted_output_directory)
 
 
    # ------ Set up logging using filename from driver-------
    logger, memory_handler = setup_logger()
 
    if hasattr(driver_spec, "log_filename") and driver_spec.log_filename:
        log_path = Path(driver_spec.log_filename)
        if log_path.suffix == "":
            log_path = log_path.with_suffix(".log")
        log_filename = output_dir / log_path.name
    else:
        log_filename =  output_dir / "tracts.log"
        logger.warning(f"No log filename specified in driver file. Defaulting to {log_filename} in the working directory.")
    
    if not os.path.exists(output_dir): # Create output directory if it doesn't exist 
        os.makedirs(output_dir)
    
    log_full_path = Path(log_filename)
    if not log_full_path.is_absolute() and log_full_path.parent == Path("."): # If log_filename is a relative path without directories, save it in the output directory. Otherwise, save it in the specified path (which may be absolute or relative with directories).
        log_full_path = Path(output_dir) / log_full_path

    set_log_file(log_filename=log_full_path,
                memory_handler=memory_handler)

    try:
        logger.info(f"Running tracts 2.0 with driver file: {driver_filename}")
        output_message = f"Results will be written to: {output_dir}."
        logger_message = f"Using log file: {log_full_path}."
        tracts_below_cm_message = f'excluding_tracts_below set to {driver_spec.optim.exclude_tracts_below_cm} cM.'

        # ------ Print initial information -------
        print('------------------------------------------------------------------------------------------------\n')
        print('Running tracts 2.0 with driver file:', driver_filename,'\n')
        print('------------------------------------------------------------------------------------------------\n')   
        for message in (output_message, logger_message, tracts_below_cm_message):
            print(message)
            logger.info(message)
        
        # ----- Extract specifications from the driver file and do necessary checks -------
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

        # ------ Load the population -------
        pop = load_population(driver_path=driver_path,
                            driver_spec=driver_spec,
                            script_dir=script_dir,
                            allosome_labels = allosome_labels) 
        pop.unknown_labels = driver_spec.optim.unknown_labels_for_smoothing
        pop.smooth_unknowns(allosome_labels=allosome_labels)
        _bins, _data = pop.get_global_tractlengths(npts=driver_spec.optim.npts, # Get the population labels and validate that these correspond to to model population labels.
                                                   exclude_tracts_below_cM=driver_spec.optim.exclude_tracts_below_cm) 
        
        # ------ Load the model -------
        model = load_model_from_driver(driver_spec=driver_spec,
                                    script_dir=script_dir,
                                    driver_path=driver_path,
                                    allosome_label=allosome_label)
        ancestor_labels = model.population_indices.keys()
        data_labels =  _data.keys()
           
        for label in data_labels:
            if label not in ancestor_labels and label not in pop.unknown_labels:
                raise ValueError(f"Population label '{label}' found in data but not in model or labels to be smoothed over. data labels: {data_labels}, model labels: {ancestor_labels}, " \
                f"unknown labels: {pop.unknown_labels}")

        # ------ Calculate ancestry proportions and set up fixed parameters if specified in the driver -------
        ancestry_proportions = pop.calculate_ancestry_proportions(ancestor_labels)
        
        print(f"Ancestries: {', '.join(ancestor_labels)}")
        autosomal_ancestry_message = f"Data autosome proportions: {np.array2string(ancestry_proportions, separator=' ')}"
        print(autosomal_ancestry_message)
        logger.info(autosomal_ancestry_message)
        if len(allosome_labels)>=1:
            allosome_proportions = pop.calculate_allosome_proportions(population_labels=ancestor_labels,
                                                                    allosome_label=allosome_label)
            allosomal_ancestry_message = f"Data allosome proportions: {np.array2string(allosome_proportions, separator=' ')}"
            print(allosomal_ancestry_message)
            logger.info(allosomal_ancestry_message)

        if len(driver_spec.optim.fix_parameters_from_ancestry_proportions) > 0: # Set up fixed parameters if specified in the driver
            
            if allosome_label:
                model.parameter_handler.set_up_fixed_parameters(demography=model,
                                                                params_to_fix_by_ancestry=driver_spec.optim.fix_parameters_from_ancestry_proportions,
                                                                proportions={
                                                                f'{model.parametrized_populations[0]}_autosomal':ancestry_proportions,
                                                                f'{model.parametrized_populations[0]}_{allosome_label}': allosome_proportions
                                                                } # Here, the option params_to_fix_by_value can be added in future development
                                                                )
            else:
                model.set_up_fixed_parameters(params_to_fix_by_ancestry=driver_spec.optim.fix_parameters_from_ancestry_proportions,
                                            proportions= {model.parametrized_populations[0]:ancestry_proportions}) # Here, the option params_to_fix_by_value can be added in future development
        else: # No parameters to fix 
            model.set_up_fixed_parameters([],{})
        print(f"Model parameters: {', '.join(model.model_base_params.keys())}") # Print model parameters
        if len(driver_spec.optim.fix_parameters_from_ancestry_proportions) > 0:
            fixed_params = ", ".join(driver_spec.optim.fix_parameters_from_ancestry_proportions)
            print(f"The following parameters have been fixed from ancestry proportions: {fixed_params}")

        if ad_model_allosomes is not None:
            admixture_model_message = (
                f"Admixture is modelled with the {ad_model_autosomes} model for autosomes "
                f"and with the {ad_model_allosomes} model for allosomes."
            )
        else:
            admixture_model_message = f"Admixture is modelled with the {ad_model_autosomes} model for autosomes."
        print(admixture_model_message)
        logger.info(admixture_model_message)

        # ------ Optimizer setup -------
        func = get_time_scaled_model_func(model) # Time parameters need to be rescaled for some optimizers, so we create a wrapper function that applies the necessary rescaling before passing parameters to the model.
        bound = get_time_scaled_model_bounds(model) # The same rescaling needs to be applied to the bounds function.
        
        # ------ Set up conversion to physical and optimizer units ------ 
        to_physical_params_functions = {ParamType.TIME: time_to_physical_function, 
                                    ParamType.RATE: rate_to_physical_function, 
                                    ParamType.SEX_BIAS: sex_bias_to_physical_function} 
        to_optimizer_params_functions  = {ParamType.TIME: time_to_optimizer_function, 
                                        ParamType.RATE: rate_to_optimizer_function, 
                                        ParamType.SEX_BIAS: sex_bias_to_optimizer_function}
        model.parameter_handler.to_physical_params_functions = to_physical_params_functions
        model.parameter_handler.to_optimizer_params_functions = to_optimizer_params_functions

        model_param_names = list(model.model_base_params.keys())
        sex_bias_param_names = [
            name for name, info in model.model_base_params.items()
            if info.type == ParamType.SEX_BIAS
        ]
        non_sex_bias_param_names = [
            name for name in model_param_names
            if name not in sex_bias_param_names
        ]

        # Show time-admissibility warnings only during optimization and final reporting.
        model.parameter_handler.enable_time_param_logging = False

        # ------ Compute starting parameters in physical units ------
        if driver_spec.optim.two_steps_optimization:
            physical_start_params = parse_start_params(
                start_param_bounds=driver_spec.start_params,
                repetitions=driver_spec.optim.repetitions,
                seed=driver_spec.optim.seed,
                model=model,
                sample_param_names=set(non_sex_bias_param_names),
                fixed_param_values={name: 0.0 for name in sex_bias_param_names},
            )
            physical_start_params = collapse_identical_start_params(physical_start_params, "step 1")
        else:
            physical_start_params = parse_start_params(
                start_param_bounds=driver_spec.start_params,
                repetitions=driver_spec.optim.repetitions,
                seed=driver_spec.optim.seed,
                model=model,
            )
        
        # ------ Convert starting parameters to optimizer units ------
        optimizer_start_params = [model.parameter_handler.convert_to_optimizer_params(params) for params in physical_start_params]   

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
        start_param_names = model_param_names

        if driver_spec.optim.two_steps_optimization:
            step_1_start_params_title = "Starting parameters for step 1 optimization"
        else:
            step_1_start_params_title = "Starting parameters for single-step optimization"
        
        # ------ Get starting ancestry proportions for the starting parameters ------ 
        # Check that the starting parameters produce reasonable ancestry proportions before optimization. Only logged for now.
        first_props = model.proportions_from_matrices(func(optimizer_start_params[0]))
        tract_types = list(first_props.keys())
        start_ancestry_props_message = "Starting ancestry proportions for the starting parameters"
        header = f"{'Run':>3} | " + " | ".join(f"{k:<35}" for k in tract_types)
        line = "-" * len(header)
        logger.info(start_ancestry_props_message)
        for l in (line, header, line):
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
            logger.info(anc_line)

        model.parameter_handler.enable_time_param_logging = True

        # ------ Run the model with (multiple) starting parameters ------

        if driver_spec.optim.two_steps_optimization is False: # Single-step optimization using optimize_cob_sex_biased_single_step

            _print_run_intro(model.parameter_handler, model, optimizer_start_params, bound, step_1_start_params_title, False, True)

            params_found, likelihoods, _full_likelihoods = _normalize_multi_init_result(run_model_multi_init(model_func=func,
                                                            bound_func=bound,
                                                            population=pop, 
                                                            start_params_list=optimizer_start_params,
                                                            population_dict=model.population_indices.items(),
                                                            parameter_handler=model.parameter_handler,
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
                                                            print_subtitle=False))

            optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found,
                                                                        likelihoods=likelihoods,
                                                                        parameter_handler=model.parameter_handler,
                                                                        param_names=start_param_names)
        
        else: # Performs two-steps optimization with multiple starting parameters            

            # ------ Step 1: optimize non-sex-bias parameters on autosomal data ------

            _print_run_intro(model.parameter_handler, model, optimizer_start_params, bound, step_1_start_params_title, True, driver_spec.optim.use_autosomes_for_sex_bias, [1])

            params_found_step_1, likelihoods_step_1, _full_likelihoods_step_1 = _normalize_multi_init_result(run_model_multi_init(model_func=func,
                                                            bound_func=bound,
                                                            population=pop, 
                                                            start_params_list=optimizer_start_params,
                                                            population_dict=model.population_indices.items(),
                                                            parameter_handler=model.parameter_handler,
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
                                                            print_subtitle=False))
            
            #  Process and print results
            optimal_params_step_1, _optimal_likelihood_step_1 = _summarize_step_results(params_found=params_found_step_1,
                                                                        likelihoods=likelihoods_step_1,
                                                                        parameter_handler=model.parameter_handler,
                                                                        param_names=start_param_names,
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
                    set(model.parameter_handler.params_fixed_by_ancestry)
                    | set(model.parameter_handler.user_params_fixed_by_value.keys())
                )
                _has_free_sex_bias = any(name not in _all_fixed_in_step_2 for name in sex_bias_param_names)

                # Draw fresh sex-bias starts for step 2, while keeping the best step 1 non-sex-bias values fixed.
                step_2_fixed_param_values = {
                    name: float(value)
                    for name, value in zip(model_param_names, optimal_params_step_1)
                    if name not in sex_bias_param_names
                }

                if _has_free_sex_bias:
                    end_step_1_message = "Selecting best parameters from step 1 and proceeding to step 2 optimization.\n"
                    print(end_step_1_message)
                    logger.info(end_step_1_message)

                    step_2_physical_start_params = parse_start_params(
                        start_param_bounds=driver_spec.start_params,
                        repetitions=driver_spec.optim.repetitions,
                        seed=driver_spec.optim.seed,
                        model=model,
                        sample_param_names=set(sex_bias_param_names),
                        fixed_param_values=step_2_fixed_param_values,
                    )
                    step_2_physical_start_params = collapse_identical_start_params(step_2_physical_start_params, "step 2")
                    step_2_start_params = [
                        model.parameter_handler.convert_to_optimizer_params(params)
                        for params in step_2_physical_start_params
                    ]

                    step_2_start_params_title = (
                        "Starting parameters for step 2 optimization "
                        "(non-sex-bias parameters are fixed to the best step 1 estimates)."
                    )

                    _print_run_intro(model.parameter_handler, model, step_2_start_params, bound, step_2_start_params_title, True, driver_spec.optim.use_autosomes_for_sex_bias, [2])

                    params_found_step_2, likelihoods_step_2, full_likelihoods_step_2 = _normalize_multi_init_result(run_model_multi_init(model_func=func,
                                                                                bound_func=bound,
                                                                                population=pop,
                                                                                start_params_list=step_2_start_params,
                                                                                population_dict=model.population_indices.items(),
                                                                                parameter_handler=model.parameter_handler,
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
                                                                                print_subtitle=False))

                    #  Process and print results
                    optimal_params, optimal_likelihood = _summarize_step_results(params_found=params_found_step_2,
                                                                                likelihoods=likelihoods_step_2,
                                                                                parameter_handler=model.parameter_handler,
                                                                                param_names=start_param_names,
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
                    _fixed_by_ancestry = [n for n in sex_bias_param_names if n in set(model.parameter_handler.params_fixed_by_ancestry)]
                    _fixed_by_value = [n for n in sex_bias_param_names if n in set(model.parameter_handler.user_params_fixed_by_value.keys())]
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
                    _silent_start = [model.parameter_handler.convert_to_optimizer_params(optimal_params_step_1)]
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
                                    population_dict=model.population_indices.items(),
                                    parameter_handler=model.parameter_handler,
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
                                    print_subtitle=False,
                                )
                            )
                    finally:
                        _tracts_logger.setLevel(_saved_tracts_level)

                    optimal_params, optimal_likelihood = _summarize_step_results(
                        params_found=params_found_step_2,
                        likelihoods=likelihoods_step_2,
                        parameter_handler=model.parameter_handler,
                        param_names=start_param_names,
                        step_label="Step 2",
                    )
                    if not driver_spec.optim.use_autosomes_for_sex_bias:
                        best_run_index = int(np.argmax([float(x) for x in likelihoods_step_2]))
                        full_data_likelihood = full_likelihoods_step_2[best_run_index]
                        if full_data_likelihood is not None:
                            optimal_likelihood = float(full_data_likelihood)

        # Print final optimal parameters and likelihood.
        final_data = "autosomal + allosomal" if ad_model_allosomes is not None else "autosomal"
        final_message = f"Final parameters and corresponding likelihood computed on {final_data} data:"
        param_names = list(model.model_base_params.keys())
        # Append derived (remainder) quantities to the table
        remainder_params = _compute_remainder_params(
            model, model.get_migration_matrices(optimal_params)
        )
        all_param_names = param_names + list(remainder_params.keys())
        all_param_values = list(optimal_params) + list(remainder_params.values())
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
        if remainder_params:
            dep_msg = f"Parameters {', '.join(remainder_params.keys())} correspond to the dependent ancestry and were not free in the optimization."
            print(dep_msg)
            logger.info(dep_msg)

        # Check for "founding migration rates > 1" in the final parameters.
        # get_violation_score calls get_migration_matrices internally, which logs the warning.
        # we capture it here so it can be shown as a user-visible printed message.
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
            _ = model.get_violation_score(optimal_params, verbose=True)
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

        # Print ancestry proportions predicted by the model
        predicted_props = model.proportions_from_matrices(func(model.parameter_handler.convert_to_optimizer_params(optimal_params)))
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

        # ------ Save ancestry proportions table -------
        _save_ancestry_proportions_table(
            ancestor_labels=ancestor_labels,
            observed_autosome_proportions=ancestry_proportions,
            predicted_autosome_proportions=autosome_values,
            output_dir=output_dir,
            output_filename_format=driver_spec.output.output_filename_format,
            observed_allosome_proportions=allosome_proportions if len(allosome_labels) >= 1 else None,
            predicted_allosome_proportions=allosome_values,
            allosome_label=allosome_label,
        )
        logger.info(f"Ancestry proportions table saved to {output_dir / driver_spec.output.output_filename_format.format(label='ancestry_proportions.txt')}")

        # ------ Produce output -------
        output_simulation_data_sex_biased(sample_population=pop,
                                        optimal_params=optimal_params,
                                        model=model,
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
                        print_start_params_table: bool = True,
                        print_subtitle: bool = True) -> tuple[list[np.ndarray], list[float], list[float | None]]:
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
    print_subtitle: bool, optional
        For internal use only. Whether to print a subtitle message describing the optimization run. Default is True.
        
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
            model=parameter_handler.demography if hasattr(parameter_handler, "demography") else None,
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
                                                                        print_step_header=False)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
        full_likelihoods.append(full_likelihood_found)
    return optimal_params, likelihoods, full_likelihoods

def run_model(model_func: callable, bound_func: callable, population: Population, 
                        startparams: list, population_dict: dict, parameter_handler: FixedParametersHandler, max_iter: int | None = None, 
                        exclude_tracts_below_cM: float = 0, ad_model_autosomes: str = 'DC', ad_model_allosomes: str = 'DC',
                        npts: int = 0, verbose_log: int = 0, verbose_screen: int = 0, two_steps_optimization: bool = True,
                        autosomes_in_step_2: bool = True, steps: list[int | str] | None = None, print_step_header: bool = True) -> tuple[np.ndarray, float, float | None]:
    
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
                                                                    print_step_header=print_step_header)
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
                                                                                                       return_full_likelihood=True)
    
    return optimal_params, optimal_likelihood, full_data_likelihood
       
