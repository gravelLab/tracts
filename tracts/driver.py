import logging
from pathlib import Path
from typing import Callable
import numpy as np
from tracts.population import Population
from tracts.core import optimize_cob, optimize_cob_sex_biased, optimize_cob_sex_biased_fixed_values
from tracts.util import time_to_physical_function, rate_to_physical_function, sex_bias_to_physical_function, time_to_optimizer_function, rate_to_optimizer_function, sex_bias_to_optimizer_function
from tracts.phase_type_distribution import PhTMonoecious
from tracts.demography.parameter import ParamType
from tracts.driver_utils import locate_file_path, load_driver_file, load_population, load_model_from_driver, get_time_scaled_model_func, get_time_scaled_model_bounds, parse_start_params, output_simulation_data_sex_biased
from tracts.logs import setup_logger, set_log_file
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
    if hasattr(driver_spec, "log_filename") and driver_spec.log_filename:
        log_path = Path(driver_spec.log_filename)
        if log_path.suffix == "":
            log_path = log_path.with_suffix(".log")
        log_filename = log_path
    else:
        log_filename = "tracts.log"
        logger.warning(f"No log filename specified in driver file. Defaulting to {log_filename} in the working directory.")
    set_log_file(log_filename, memory_handler)
    logger.info(f"Using log file: {log_filename}")
    logger.info(f"Running tracts 2.0 with driver file: {driver_filename}")

    # ------ Print initial information -------
    print('------------------------------------------------------------------------------------------------\n')
    print('Running tracts 2.0 with driver file:', driver_filename,'\n')
    print('Reading data, demographic model and driver specifications...\n')
    print('------------------------------------------------------------------------------------------------\n')   
    print(f'excluding_tracts_below set to {driver_spec.exclude_tracts_below_cm} cM.')
    
    # ----- Extract specifications from the driver file and do necessary checks -------
    # Autosomal admixture model is correctly specified
    ad_model_autosomes = driver_spec.ad_model_autosomes
    if not driver_spec.ad_model_autosomes in ['DC','DF','M','H-DC','H-DF']:
        print('The model for autosomal admixture must be either DC (for Dioecious-Coarse), DF (for Dioecious-Fine), M (for Monoecious), H-DC or H-DF (for the hybrid pedigree refinements of DC and DF, resp.). Setting ad_model_autosomes = DC by default.')
        ad_model_autosomes = 'DC'

    
    # Check whether allosomes are present in the sample
    allosome_labels = driver_spec.samples.allosomes
    allosome_label = allosome_labels[0] if len(allosome_labels) > 0 else None  # Currently assumes allosomes is a single label. May change in the future

    # Allosomal admixture model is correctly specified
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
        ad_model_allosomes = None # This will trigger the code to not model allosomal admixture.

    # ------ Load the population -------
    pop = load_population(driver_path=driver_path,
                        driver_spec=driver_spec,
                        script_dir=script_dir,
                        allosome_labels = allosome_labels) 
    pop.unknown_labels = driver_spec.unknown_labels_for_smoothing
    pop.smooth_unknowns(allosome_labels=allosome_labels)
    _bins, _data = pop.get_global_tractlengths(npts=driver_spec.npts, # Get the population labels and validate that these correspond to to model population labels.
                                               exclude_tracts_below_cM=driver_spec.exclude_tracts_below_cm) 
    
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
            "unknown labels: {pop.unknown_labels}")

    # ------ Calculate ancestry proportions and set up fixed parameters if specified in the driver -------
    ancestry_proportions = pop.calculate_ancestry_proportions(ancestor_labels)
    
    print("Ancestries:", [ancestry for ancestry in ancestor_labels] )
    print("Data autosome proportions:", ancestry_proportions )
    if len(allosome_labels)>=1:
        allosome_proportions = pop.calculate_allosome_proportions(population_labels=ancestor_labels,
                                                                allosome_label=allosome_label)
        print("Data allosome proportions:", allosome_proportions )

    if len(driver_spec.fix_parameters_from_ancestry_proportions) > 0: # Set up fixed parameters if specified in the driver
        
        if allosome_label:
            model.parameter_handler.set_up_fixed_parameters(demography=model,
                                                            params_to_fix_by_ancestry=driver_spec.fix_parameters_from_ancestry_proportions,
                                                            proportions={
                                                            f'{model.parametrized_populations[0]}_autosomal':ancestry_proportions,
                                                            f'{model.parametrized_populations[0]}_{allosome_label}': allosome_proportions
                                                            } # Here, the option params_to_fix_by_value can be added in future development
                                                            )
        else:
            model.set_up_fixed_parameters(params_to_fix_by_ancestry=driver_spec.fix_parameters_from_ancestry_proportions,
                                        proportions= {model.parametrized_populations[0]:ancestry_proportions}) # Here, the option params_to_fix_by_value can be added in future development
    else: # No parameters to fix 
        model.set_up_fixed_parameters([],{})
    print("Model parameters :",[param_name for param_name in model.model_base_params.keys()]) # Print model parameters

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
    
    # ------ Compute starting parameters in physical units ------
    physical_start_params = parse_start_params(start_param_bounds=driver_spec.start_params,
                                            repetitions=driver_spec.repetitions, 
                                            seed=driver_spec.seed, 
                                            model=model)
    # ------ Convert starting parameters to optimizer units ------
    optimizer_start_params = [model.parameter_handler.convert_to_optimizer_params(params) for params in physical_start_params]   

    # ------ Message about starting parameters setup ------ 
    if len(physical_start_params) > 1: # Multiple runs with different starting parameters
        mult_params_message = "Multiple starting parameters were generated. These will be converted to optimizer units and used for multiple optimization runs."
        logger.info(mult_params_message)
        print("\n"+mult_params_message+"\n")

    else: # Single run with one set of starting parameters
        single_params_message = "A single set of starting parameters was generated. It will be converted to optimizer units and used for optimization."
        logger.info(single_params_message)
        print("\n"+single_params_message+"\n")

    # ------ Print starting parameters in physical units ------
    header = f"{'Run':>3} | {'Starting parameters':<45}"
    line = "-" * len(header) 

    for l in (header, line):
        print(l)
        logger.info(l)

    # ------ Check that starting parameters are correctly converted to optimizer units and within bounds ------
    for i, (phys, opt) in enumerate(zip(physical_start_params, optimizer_start_params)):
        assert np.isclose(phys, model.parameter_handler.convert_to_physical_params(opt)).all()
        if bound(opt)<0:
            print("Warning, starting parameters are out of bounds.")
        phys_str = ", ".join(f"{x:.4g}" for x in phys)
        start_param_message = f"{1+i:>3} | [{phys_str:<43}]"
        print(start_param_message)
        logger.info(start_param_message)
    print(line)

    # ------ Get starting ancestry proportions for the starting parameters ------ 
    # Check that the starting parameters produce reasonable ancestry proportions before optimization.
    # Only logged for now.
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
        logger.info(anc_line)
    

    # ------ Run the model with (multiple) starting parameters ------
    params_found, likelihoods = run_model_multi_init(model_func=func,
                                                    bound_func=bound,
                                                    population=pop, 
                                                    population_labels=ancestor_labels,
                                                    start_params_list=optimizer_start_params,
                                                    population_dict=model.population_indices.items(),
                                                    parameter_handler=model.parameter_handler,
                                                    max_iter=driver_spec.maximum_iterations,
                                                    exclude_tracts_below_cM=driver_spec.exclude_tracts_below_cm,
                                                    ad_model_autosomes = ad_model_autosomes, 
                                                    ad_model_allosomes=ad_model_allosomes,
                                                    npts=driver_spec.npts, 
                                                    verbose_log=driver_spec.verbose_log,
                                                    verbose_screen=driver_spec.verbose_screen, 
                                                    two_steps_optimization=driver_spec.two_steps_optimization, 
                                                    run_optimize_cob=driver_spec.run_optimize_cob)

    # ------ Process and print results ------
    formatted_likelihoods = [float(x) for x in likelihoods] # One likelihood per optimization run. If multiple runs were done, these will be printed in a table with the corresponding parameters. The best likelihood and parameters across runs will be selected as the final result.
    physical_found_params = [model.parameter_handler.convert_to_physical_params(f_param) for f_param in params_found] # One set of parameters per optimization run, converted to physical units. If multiple runs were done, these will be printed in a table with the corresponding likelihoods. 
    
    if len(formatted_likelihoods) > 1: # Print optimal parameters and likelihoods for multiple runs with different starting parameters.

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
    
    # Choose optimal run across multiple runs with different starting parameters, if applicable. This will be the run with the highest likelihood (lowest negative log-likelihood).
    optimal_params, optimal_likelihood = max(zip(physical_found_params, formatted_likelihoods), key=lambda x: x[1])
    
    # Print final optimal parameters and likelihood.
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

    # ------ Produce output -------
    output_simulation_data_sex_biased(sample_population=pop,
                                    optimal_params=optimal_params,
                                    model=model,
                                    driver_spec=driver_spec,
                                    ad_model_autosomes=ad_model_autosomes,
                                    ad_model_allosomes=ad_model_allosomes)


# ----- Runner functions -----

def run_model_multi_init(model_func: Callable, bound_func: Callable, population: Population, population_labels: list[str], 
                          start_params_list: list[np.ndarray], population_dict : dict, parameter_handler = None , 
                          max_iter: int=None, exclude_tracts_below_cM: int = 0, ad_model_autosomes = 'DC', 
                          ad_model_allosomes = 'DC', npts: int = 50, verbose_log: int = 0, verbose_screen:int = 0, two_steps_optimization: bool = True, run_optimize_cob: bool = False) -> tuple[list[np.ndarray], list[float]]:
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
        population_dict: dict
            A dictionary mapping population labels to their corresponding indices in the model.
        parameter_handler: ParameterHandler, optional
            An object that handles parameter transformations and fixed parameters. Default is None.
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

    Returns
    ----------
    tuple[list[np.ndarray], list[float]]
    	A tuple containing two lists: (i) a list of optimal parameters found for each set of starting parameters and (ii) a list of likelihoods corresponding to the optimal parameters.
    """
    
    optimal_params = []
    likelihoods = []
    for start_params in start_params_list:
        opt_run_message = f"Optimization run #{len(optimal_params)+1}"
        print("\n" + opt_run_message + "\n")
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
                                                   verbose_log=verbose_log,
                                                   verbose_screen=verbose_screen,
                                                   two_steps_optimization=two_steps_optimization, 
                                                   run_optimize_cob=run_optimize_cob)
        optimal_params.append(params_found)
        likelihoods.append(likelihood_found)
    return optimal_params, likelihoods

def run_model(model_func: callable, bound_func: callable, population: Population, population_labels: list[str], startparams: list,
            population_dict: dict, parameter_handler=None, max_iter: int | None = None, 
            exclude_tracts_below_cM: float =0, ad_model_autosomes:str ='DC', ad_model_allosomes:str ='DC', 
            npts:int =0, verbose_log:int=0, verbose_screen:int =0, two_steps_optimization: bool = True, run_optimize_cob: bool = False):
    
    """
    Runs the optimization for any demographic model, including sex-biased models. This function allows to run the old optimization function optimize_cob.

    Parameters
    ----------
    model_func: callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    bound_func: callable
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    population_labels: list[str]
    	A list of labels corresponding to the populations.	
    startparams: list
        An array of initial parameters to start the optimization.
    population_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    parameter_handler: ParameterHandler, optional
        An object that handles parameter transformations and fixed parameters. Default is None.
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
        Verbosity level for screen prints. Default is 0 (no verbose output). If greater than 0, iterations are printed every ``verbose_screen`` steps.
    two_steps_optimization: bool, optional
        Whether to use a two-step optimization procedure for sex-biased models. If True, the optimization will first be run on non-sex bias parameters using only autosomal data. Then, a second optimization will be run with sex-bias parameters using both autosomal and allosomal data, starting from the results of the first optimization. Default is True.  
    run_optimize_cob: bool, optional
        Whether to run the optimize_cob function. Default is False.
        
    Returns
    -------
    tuple [np.ndarray, float]
        A tuple containing the optimal parameters found and the corresponding likelihood.

    Notes
    -----
    The optimize_cob function implements the PhTMonoecious model on autosomal data. Corresponds to the previous version of tracts. 

    """
    # NOTE: If optimize_cob is no longer used, this function can be removed and replaced by run_model_sex_bias
    

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
            verbose_log=verbose_log,
            verbose_screen=verbose_screen,
            two_steps_optimization=two_steps_optimization,
        )
    elif ad_model_allosomes is None:
        Ls = population.Ls
        nind = population.nind
        bins, data = population.get_global_tractlengths(npts=npts,
                                                        exclude_tracts_below_cM=exclude_tracts_below_cM)
        data = [data[poplab] for poplab in population_labels]
        model_func_sample_pop = lambda params:list(model_func(params).values())[0]
        xopt = optimize_cob(p0=startparams,
                            bins=bins,
                            Ls=Ls,
                            data=data,
                            nsamp=nind, 
                            model_func=model_func_sample_pop,
                            outofbounds_fun=bound_func,
                            verbose=verbose_screen)
        optmod = PhTMonoecious(model_func_sample_pop(xopt))
        optlik = optmod.loglik(bins=bins,
                               Ls=Ls,
                               data=data,
                               num_samples=nind)
        return xopt, optlik
    else:
        raise Exception("The optimize_cob method does not accept allosomal admixture.")

def run_model_sex_biased(model_func: callable, bound_func: callable, population: Population, 
                        startparams: list, population_dict: dict, parameter_handler = None, max_iter: int | None = None, 
                        exclude_tracts_below_cM: float = 0, ad_model_autosomes: str = 'DC', ad_model_allosomes: str = 'DC',
                        npts: int = 0, verbose_log: int = 0, verbose_screen: int = 0, two_steps_optimization: bool = True):
    
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
    parameter_handler: ParameterHandler, optional
        An object that handles parameter transformations and fixed parameters. Default is None.
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
    
    Returns
    ----------
    tuple [np.ndarray, float]
        A tuple containing the optimal parameters found and the corresponding likelihood.
    """
    if not two_steps_optimization:
        optimal_params, optimal_likelihood = optimize_cob_sex_biased(p0=startparams, 
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
                                                                                verbose_log=verbose_log,
                                                                                verbose_screen=verbose_screen,
                                                                                ad_model_autosomes=ad_model_autosomes, 
                                                                                ad_model_allosomes=ad_model_allosomes,
                                                                                npts=npts)    
    
    return optimal_params, optimal_likelihood
       



   
