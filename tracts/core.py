import logging
import numpy as np
import scipy.optimize
import copy

from tracts.phase_type import hybrid_pedigree as HP
from tracts.phase_type import PhTMonoecious, PhTDioecious
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.population import Population
from tracts.util import eprint
from tracts.demography.parameter import ParamType
logger = logging.getLogger(__name__)

_counter = 0
_out_of_bounds_val = -1e32
_min_out_of_bounds_val = -1e-10
_ignore_oob_above = -1e-14

_TABLE_HEADER = "Iter.\t Log-likelihood\t Model parameters\t Transmission"
_LINE_HEADER = "-" * len(_TABLE_HEADER.expandtabs())

# ------------------ Objective function ------------------

def _compute_objective(
    parameters,
    *,
    local_parameter_handler,
    best_state,
    model_func,
    outofbounds_fun,
    ad_model_autosomes,
    ad_model_allosomes,
    N_cores,
    verbose_log,
    verbose_screen,
    population,
    autosome_bins,
    autosome_data_mapped,
    allosome_bins=None,
    allosome_length=None,
    female_data_mapped=None,
    male_data_mapped=None,
    num_females=None,
    num_males=None,
    include_autosomes=True,
    include_allosomes=True,
    rho_f=1,
    rho_m=1,
    TP=2
):
    """
    Evaluate the optimization objective (negative log-likelihood) for a given parameter vector.

    This is the shared implementation called by both
    :func:`optimize_cob_sex_biased_single_step` and
    :func:`optimize_cob_sex_biased_two_steps` via their thin ``objective_function``
    wrappers. It increments the global iteration counter, logs/prints the current
    iterate, handles out-of-bounds and singular-matrix penalties, computes the
    autosomal and/or allosomal log-likelihoods, and updates ``best_state`` whenever
    a new best finite objective is found.

    Parameters
    ----------
    parameters : np.ndarray
        Full parameter vector in optimizer space.
    local_parameter_handler : FixedParametersHandler
        Deep copy of the original handler; used to convert optimizer-space
        parameters to physical parameters for logging.
    best_state : dict
        Mutable dict with keys ``'objective'`` (float, lowest objective seen so
        far, initialised to ``np.inf``) and ``'params'`` (np.ndarray or None,
        the corresponding parameter vector). Updated in place whenever a new best
        finite objective is found.
    model_func : callable
        Returns a dict of migration matrices keyed by population label.
    outofbounds_fun : callable or None
        Returns a negative violation score when parameters are out of bounds, or
        a value close to zero when they are admissible. If None, a warning is
        printed but execution continues.
    ad_model_autosomes : str
        Admixture model for autosomes. One of ``'DC'``, ``'DF'``, ``'M'``,
        ``'H-DC'``, or ``'H-DF'``.
    ad_model_allosomes : str or None
        Admixture model for allosomes. One of ``'DC'``, ``'DF'``, ``'H-DC'``,
        or ``'H-DF'``. Only used when ``include_allosomes=True``.
    N_cores : int
        Number of CPU cores for parallel hybrid-pedigree computations.
    verbose_log : int
        Log every ``verbose_log`` iterations (0 = never).
    verbose_screen : int
        Print every ``verbose_screen`` iterations (0 = never).
    population : Population
        Population object providing chromosome lengths and sample sizes.
    autosome_bins : np.ndarray
        Bin edges for the autosomal tract-length histogram.
    autosome_data_mapped : list of list
        Observed autosomal tract counts per population, indexed to match the
        model's population ordering.
    allosome_bins : np.ndarray, optional
        Bin edges for the allosomal tract-length histogram. Required when
        ``include_allosomes=True``.
    allosome_length : float, optional
        Length of the X chromosome in Morgans. Required when
        ``include_allosomes=True``.
    female_data_mapped : list of list, optional
        Observed X-chromosome tract counts for females per population. Required
        when ``include_allosomes=True``.
    male_data_mapped : list of list, optional
        Observed X-chromosome tract counts for males per population. Required
        when ``include_allosomes=True``.
    num_females : int, optional
        Number of female samples. Required when ``include_allosomes=True``.
    num_males : int, optional
        Number of male samples. Required when ``include_allosomes=True``.
    include_autosomes : bool, default True
        Whether to include the autosomal log-likelihood in the objective.
    include_allosomes : bool, default True
        Whether to include the allosomal log-likelihood in the objective.
    rho_f : float, default 1
        The female-specific recombination rate.
    rho_m : float, default 1
        The male-specific recombination rate.
    TP : int, default 2
        The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models.

    Returns
    -------
    float
        The objective value (negative total log-likelihood), or a large positive
        penalty when parameters are out of bounds or produce a singular matrix.
    """
    global _counter
    global _out_of_bounds_val
    global _min_out_of_bounds_val
    _counter += 1

    if not include_autosomes and not include_allosomes:
        raise ValueError("At least one of include_autosomes or include_allosomes must be True.")

    def flush_result(result, note=str()):
        prev_time_param_logging = local_parameter_handler.enable_time_param_logging
        local_parameter_handler.enable_time_param_logging = False
        try:
            param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in local_parameter_handler.convert_to_physical_params(parameters, report_non_admissible=False)]))
        finally:
            local_parameter_handler.enable_time_param_logging = prev_time_param_logging
        _did_log = (verbose_log > 0) and (_counter % verbose_log == 0)
        _did_screen = (verbose_screen > 0) and (_counter % verbose_screen == 0)
        if _did_log:
            logger.info("iter=%-6d | obj=%-12g | params=%s %s", _counter, result, param_str, note)
        if _did_screen:
            eprint('%-8i, %-12g, %s, %s' % (_counter, result, param_str, note))

    if outofbounds_fun is not None:
        oob = outofbounds_fun(parameters)
        if oob < _ignore_oob_above:
            out = oob * _out_of_bounds_val - _min_out_of_bounds_val
            flush_result(out, f'OOB (oob={oob})')
            if np.isfinite(out) and out < best_state['objective']:
                best_state['objective'] = out
                best_state['params'] = parameters.copy()
            return out
    else:
        eprint("No bound function defined")

    try:
        matrices = model_func(parameters)
        matrix_list = [matrix for matrix in matrices.values()]
        if include_allosomes:
            [male_matrix, female_matrix] = matrix_list
        else:
            avg_matrix = np.mean(matrix_list, axis=0)
            male_matrix = avg_matrix
            female_matrix = avg_matrix
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        out = -_out_of_bounds_val - _min_out_of_bounds_val  # large positive penalty, consistent with OOB
        flush_result(out, 'Singular matrix (infeasible params)')
        if np.isfinite(out) and out < best_state['objective']:
            best_state['objective'] = out
            best_state['params'] = parameters.copy()
        return out

    # ----------------- Compute model likelihood for autosomes -----------------

    if include_autosomes:

        if ad_model_autosomes == 'H-DC' or ad_model_autosomes == 'H-DF':

            result_autosomes = HP.HP_loglik(mig_matrix_f=female_matrix,
                                            mig_matrix_m=male_matrix,
                                            rho_f=rho_f,
                                            rho_m=rho_m,
                                            TP=TP,
                                            Dioecious_model=ad_model_autosomes[2:],
                                            X_chr=False,
                                            X_chr_male=False,
                                            N_cores=N_cores,
                                            bins=autosome_bins,
                                            Ls=population.Ls,
                                            data=[mat for mat in autosome_data_mapped],
                                            num_samples=len(population.indivs),
                                            cutoff=0)

        else:

            try:
                if ad_model_autosomes == 'M':
                    model = PhTMonoecious(migration_matrix=0.5*(female_matrix+male_matrix), rho=1)
                else:
                    assert male_matrix.shape[0] < 20, "PhTDioecious currently only supports less than 20 generations for autosomes."
                    model = PhTDioecious(migration_matrix_f=female_matrix,
                                        migration_matrix_m=male_matrix,
                                        rho_f=rho_f,
                                        rho_m=rho_m,
                                        sex_model=ad_model_autosomes)
            except (np.linalg.LinAlgError, ValueError):
                out = -_out_of_bounds_val - _min_out_of_bounds_val  # large positive penalty
                flush_result(out, 'Singular matrix (infeasible params)')
                return out

            result_autosomes = model.loglik(bins=autosome_bins,
                                            Ls=population.Ls,
                                            data=[mat for mat in autosome_data_mapped],
                                            num_samples=len(population.indivs))

        flush_result(result_autosomes, 'Autosomes')

    # ----------------- Compute model likelihood for allosomes -----------------

    if include_allosomes:

        if ad_model_allosomes == 'H-DC' or ad_model_allosomes == 'H-DF':

            result_X_females = HP.HP_loglik(mig_matrix_f=female_matrix,
                                            mig_matrix_m=male_matrix,
                                            rho_f=rho_f,
                                            rho_m=rho_m,
                                            TP=TP,
                                            Dioecious_model=ad_model_allosomes[2:],
                                            X_chr=True,
                                            X_chr_male=False,
                                            N_cores=N_cores,
                                            bins=allosome_bins,
                                            Ls=[allosome_length],
                                            data=[mat for mat in female_data_mapped],
                                            num_samples=num_females, cutoff=0)

            result_X_males = HP.HP_loglik(mig_matrix_f=female_matrix,
                                          mig_matrix_m=male_matrix,
                                          rho_f=rho_f,
                                          rho_m=rho_m,
                                          TP=TP,
                                          Dioecious_model=ad_model_allosomes[2:],
                                          X_chr=True,
                                          X_chr_male=True,
                                          N_cores=N_cores,
                                          bins=allosome_bins,
                                          Ls=[allosome_length],
                                          data=[mat for mat in male_data_mapped],
                                          num_samples=num_males, cutoff=0)

        else:

            result_X_females = PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=rho_f,
                                            rho_m=rho_m,
                                            sex_model=ad_model_allosomes,
                                            X_chromosome=True).loglik(bins=allosome_bins,
                                                                      Ls=[allosome_length],
                                                                      data=[mat for mat in female_data_mapped],
                                                                      num_samples=num_females)

            result_X_males = PhTDioecious(migration_matrix_f=female_matrix,
                                          migration_matrix_m=male_matrix,
                                          rho_f=rho_f,
                                          rho_m=rho_m,
                                          sex_model=ad_model_allosomes,
                                          X_chromosome=True,
                                          X_chromosome_male=True).loglik(bins=allosome_bins,
                                                                         Ls=[allosome_length],
                                                                         data=[mat for mat in male_data_mapped],
                                                                         num_samples=num_males)

        flush_result(result_X_females, 'Female allosomes')
        flush_result(result_X_males, 'Male allosomes')

    if include_autosomes and include_allosomes:
        result = result_autosomes + result_X_females + result_X_males
    elif include_autosomes and not include_allosomes:
        result = result_autosomes
    elif not include_autosomes and include_allosomes:
        result = result_X_females + result_X_males
    else:
        raise ValueError("At least one of include_autosomes or include_allosomes must be True.")

    obj = -result

    if np.isfinite(obj) and obj < best_state['objective']:
        best_state['objective'] = obj
        best_state['params'] = parameters.copy()
    return obj


def _flush_final(note, *, best_state, verbose_log, verbose_screen, local_parameter_handler):
    """Emit the best-seen result if it wasn't already printed on the last regular iteration."""
    if best_state['params'] is None:
        return
    _needs_log = verbose_log > 0 and (_counter % verbose_log != 0)
    _needs_screen = verbose_screen > 0 and (_counter % verbose_screen != 0)
    if not (_needs_log or _needs_screen):
        return
    _prev_tpl = local_parameter_handler.enable_time_param_logging
    local_parameter_handler.enable_time_param_logging = False
    try:
        _final_param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in local_parameter_handler.convert_to_physical_params(best_state['params'], report_non_admissible=False)]))
    finally:
        local_parameter_handler.enable_time_param_logging = _prev_tpl
    if _needs_log:
        logger.info("iter=%-6d | obj=%-12g | params=%s %s", _counter, -best_state['objective'], _final_param_str, note)
    if _needs_screen:
        eprint('%-8i, %-12g, %s, %s' % (_counter, -best_state['objective'], _final_param_str, note))


# ------------------ Single-step optimization ------------------


def optimize_cob_sex_biased_single_step(p0:list, population: Population, model_func: callable, parameter_handler: FixedParametersHandler, outofbounds_fun:callable=None, 
                            verbose_log:int=0, verbose_screen:int=10, p_dict:dict=None, exclude_tracts_below_cM:float=0, 
                            maxiter:int=None, reset_counter:bool=True, ad_model_autosomes:str='DC',
                            ad_model_allosomes:str='DC', npts:int=50, print_step_header:bool=True, N_cores:int=1,
                            rho_f:float=1, rho_m:float=1, TP:int=2) -> tuple[np.ndarray, float]:
    """
    Optimizes the log-likelihood over all parameters defined by the demographic model, given a specified pair of admixture models for autosomes and allosomes.
    The optimization is carried out jointly in a single step, estimating all parameters simultaneously using both autosomal and allosomal data.

    Parameters
    ----------    
    p0: list
            An array of initial parameters to start the optimization.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    model_func: callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    parameter_handler: FixedParametersHandler
        An object that handles parameter transformations and fixed parameters.
    outofbounds_fun: callable, Optional
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    verbose_log: int, default: 0
        If greater than zero, logs optimization status every ``verbose`` iterations.
    verbose_screen: int, default: 0
        If greater than zero, prints optimization status every ``verbose`` iterations.
    p_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    maxiter: int, default: None
        Maximum iterations to run for.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).
    ad_model_autosomes: str, optional
        The model to use for autosomal admixture. Must be one of 'DC', 'DF', 'M', 'H-DC' or 'H-DF'. Default is 'DC'.
    ad_model_allosomes: str, optional
        The model to use for allosomal admixture. Must be one of 'DC', 'DF', 'H-DC' or 'H-DF'. Default is 'DC'. If None, allosomal admixture will not be modeled.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
    print_step_header: bool, optional
        If True, print the admixture-model title and parameter-set subtitle at the start of the
        optimization. If False, only the iteration table header is printed. For internal use only;
        set automatically by :func:`~tracts.driver.run_model_multi_init` to suppress repeated
        headers across multiple runs within the same step. Default is True.
    N_cores: int, optional
        The number of CPU cores to use for parallel processing, when the hybrid-pedigree refinements of the DF or DC models
        are used. Ignored if the hybrid-pedigree refinements are not used. Default is 1. 
    rho_f: float, default 1
        The female-specific recombination rate.
    rho_m: float, default 1
        The male-specific recombination rate.
    TP: int, default 2
        The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models.

    Returns
    -------
    tuple [np.ndarray, float]
        A tuple containing the optimal parameters found and the corresponding likelihood.
    """
    
    if reset_counter:
        global _counter
        _counter = 0

    autosome_bins, autosome_data = population.get_global_tractlengths(
        npts=npts,
        exclude_tracts_below_cM=exclude_tracts_below_cM,
    )
    n_autosome_bins = len(autosome_bins)
    autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
    for k, v in autosome_data.items():
        autosome_data_mapped[dict(p_dict)[k]] = v

    if ad_model_allosomes is not None:
        allosome_bins, allosome_data = population.get_global_allosome_tractlengths(
            allosome='X',
            npts=npts,
            exclude_tracts_below_cM=exclude_tracts_below_cM,
        )
        n_allosome_bins = len(allosome_bins)
        allosome_length = population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
        num_males = population.num_males
        num_females = population.num_females

        female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
        for k, v in female_data.items():
            female_data_mapped[dict(p_dict)[k]] = v

        male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
        for k, v in male_data.items():
            male_data_mapped[dict(p_dict)[k]] = v

    local_parameter_handler = copy.deepcopy(parameter_handler)
    _best_state = {'objective': np.inf, 'params': None}

    def objective_function(parameters):
        return _compute_objective(
            parameters,
            local_parameter_handler=local_parameter_handler,
            best_state=_best_state,
            model_func=model_func,
            outofbounds_fun=outofbounds_fun,
            ad_model_autosomes=ad_model_autosomes,
            ad_model_allosomes=ad_model_allosomes,
            N_cores=N_cores,
            verbose_log=verbose_log,
            verbose_screen=verbose_screen,
            population=population,
            autosome_bins=autosome_bins,
            autosome_data_mapped=autosome_data_mapped,
            allosome_bins=allosome_bins if ad_model_allosomes is not None else None,
            allosome_length=allosome_length if ad_model_allosomes is not None else None,
            female_data_mapped=female_data_mapped if ad_model_allosomes is not None else None,
            male_data_mapped=male_data_mapped if ad_model_allosomes is not None else None,
            num_females=num_females if ad_model_allosomes is not None else None,
            num_males=num_males if ad_model_allosomes is not None else None,
            include_allosomes=ad_model_allosomes is not None,
            rho_f=rho_f,
            rho_m=rho_m,
            TP=TP
        )
    # ------------ Define reduced objective function and out-of-bounds function for optimization ------------

    def reduced_objective_function(free_parameters_opt):
        extended_parameters = local_parameter_handler.extend_parameters(free_parameters=free_parameters_opt,
                                                                        units="opt",
                                                                        counter=_counter,
                                                                        verbose_warning_log=verbose_log) # NOTE: Add verbose_warning_screen=verbose_screen if RuntimeWarnings should be printed on screen.
        
        return objective_function(extended_parameters) #Full parameters in optimizer space
  
    def reduced_outofbounds_fun(free_parameters_opt):
        return outofbounds_fun(local_parameter_handler.extend_parameters(free_parameters=free_parameters_opt,
                                                                        units="opt")) #Full parameters in optimizer space

    reduced_p0 = local_parameter_handler.reduce_parameters(p0) # Initial parameters

    # ------------ Run single-step optimization ------------

    subtitle_message = (
        "Optimizing model likelihood over parameters "
        f"{str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."
    )
    if print_step_header:
        print(subtitle_message)
        logger.info(subtitle_message)

    if (verbose_log > 0) and (_counter % verbose_log == 0):
        for l in [_TABLE_HEADER, _LINE_HEADER]:
            logger.info(l)
    if (verbose_screen > 0) and (_counter % verbose_screen == 0):
        for l in [_TABLE_HEADER, _LINE_HEADER]:
            print(l)

    reduced_objective_to_optimize = lambda x: reduced_objective_function(x)

    outputs = scipy.optimize.fmin_cobyla(func=reduced_objective_to_optimize,
                                        x0=reduced_p0,
                                        cons=reduced_outofbounds_fun,
                                        rhobeg=.01,
                                        rhoend=.0001,
                                        maxfun=maxiter)
    
    # Final flush: always show the last result at the end of the optimization run
    _flush_final('Total', best_state=_best_state, verbose_log=verbose_log, verbose_screen=verbose_screen, local_parameter_handler=local_parameter_handler)

    current_best_parameters = local_parameter_handler.extend_parameters(free_parameters=outputs,
                                                                    units="opt",
                                                                    show_ancestry_warning=True)

    # ------------ Return optimal parameters corresponding to best likelihood ------------

    if _best_state['params'] is None:
        try:
            fallback_likelihood = -objective_function(current_best_parameters)
            return current_best_parameters, fallback_likelihood
        except Exception:
            return current_best_parameters, -1e32

    return _best_state['params'], -_best_state['objective']


# ------------------ Two-steps optimization ------------------


def _assign_step_indicators(steps):
    """Validate the `steps` argument and return (step_1, step_2) booleans indicating 
    which optimization steps to run."""
    if steps is not None:
        if not isinstance(steps, list):
            raise TypeError("steps must be a list of integers or strings, or None.")
        valid_step_values = {1, 2, 'step1', 'step2'}
        for step in steps:
            if step not in valid_step_values:
                raise ValueError(f"Invalid step value: {step}. Must be one of {valid_step_values}.")
        if len(steps) == 0:
            raise ValueError("steps list cannot be empty.")

    if steps is None:
        normalized_steps = (1, 2)
    else:
        normalized_steps = tuple(sorted({1 if step in (1, 'step1') else 2 for step in steps}))
        if len(normalized_steps) != len(steps):
            raise ValueError("steps cannot contain duplicate references to the same optimization step.")
        if normalized_steps not in ((1,), (2,), (1, 2)):
            raise ValueError("Only step 1 only, step 2 only, or the combined step 1 + step 2 optimization are allowed.")

    return 1 in normalized_steps, 2 in normalized_steps


def optimize_cob_sex_biased_two_steps(p0:list, population: Population, model_func:callable, parameter_handler: FixedParametersHandler,
                                    outofbounds_fun:callable=None, verbose_log:int=0, verbose_screen:int=10,
                                    p_dict:dict=None, exclude_tracts_below_cM:float=0, maxiter:int=None, reset_counter:bool=True, 
                                    ad_model_autosomes:str='DC', ad_model_allosomes:str='DC', autosomes_in_step_2:bool=True,
                                    steps: list[int | str] | None = None, npts:int=50, print_step_header:bool=True,
                                    return_full_likelihood: bool = False, N_cores:int=1,
                                    rho_f:float=1, rho_m:float=1, TP:int=2) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float | None]:
    """
    Optimizes the log-likelihood over all parameters defined by the demographic model, for a specified admixture model applied to both autosomes and allosomes.
    The procedure supports exactly three modes.

    1. Step 1 only: optimize only non-sex-bias parameters using autosomal data.
    2. Step 2 only: fix non-sex-bias parameters at their ``p0`` values and optimize only sex-bias parameters using allosomal data, optionally together with autosomal data if ``autosomes_in_step_2`` is True.
    3. Step 1 + Step 2: first run step 1, then fix non-sex-bias parameters at their step-1 estimates and optimize only sex-bias parameters as in step 2.

    No other optimization-step combinations are allowed.

    Parameters
    ----------    
    p0: list
            An array of initial parameters to start the optimization. 
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    model_func: callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    parameter_handler: FixedParametersHandler
        An object that handles parameter transformations and fixed parameters.
    outofbounds_fun: callable, Optional
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    cutoff: int, default:0 
        The number of bins to drop at the beginning of the array. This could be achieved with masks.
    verbose_log: int, default: 0
        If greater than zero, logs optimization status every ``verbose`` iterations.
    verbose_screen: int, default: 0
        If greater than zero, prints optimization status every ``verbose`` iterations.
    p_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    maxiter: int, default: None
        Maximum iterations to run for.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).
    ad_model_autosomes: str, optional
        The model to use for autosomal admixture. Must be one of 'DC', 'DF', 'M', 'H-DC' or 'H-DF'. Default is 'DC'.
    ad_model_allosomes: str, optional
        The model to use for allosomal admixture. Must be one of 'DC', 'DF', 'H-DC' or 'H-DF'. Default is 'DC'.
        If None (no allosomal data provided), step 2 cannot be run. If only step 2 is requested (steps=[2]), an error is raised.
        If both steps are requested (steps=None or steps=[1,2]), step 2 is automatically disabled with a log message.
    autosomes_in_step_2: bool, optional
        If True, both autosomal and allosomal data will be used in the second optimization step. If False, only allosomal data will be used.
        This option is only relevant when step 2 is run. Default is True.
    steps: list[int | str] | None, optional
        A list specifying which steps to run. Step 1 (non-sex-bias parameter optimization) can be denoted as 1 or 'step1', and step 2 (sex-bias parameter optimization)
        can be denoted as 2 or 'step2'. The only allowed combinations are step 1 only, step 2 only, or both steps.
        Examples of valid values are [1], ['step1'], [2], ['step2'], [1, 2], or ['step1', 'step2'].
        Mixed types are allowed, but duplicate references to the same step such as [1, 'step1'] are not. Default is None (both steps will be run).
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
    print_step_header: bool, optional
        If True, print the admixture-model title and step subtitle at the start of each optimization
        step. If False, only the iteration table header is printed. For internal use only; set
        automatically by :func:`~tracts.driver.run_model_multi_init` to suppress repeated headers
        across multiple runs within the same step. Default is True.
    N_cores: int, optional
        The number of CPU cores to use for parallel processing, when the hybrid-pedigree refinements of the DF or DC models
        are used. Ignored if the hybrid-pedigree refinements are not used. Default is 1. 
    rho_f: float, default 1
        The female-specific recombination rate.
    rho_m: float, default 1
        The male-specific recombination rate.
    TP: int, default 2
        The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models.

    Returns
    -------
    tuple [np.ndarray, float] or tuple [np.ndarray, float, float | None]
        By default, returns the optimal parameters found and the optimization likelihood.
        If ``return_full_likelihood`` is True, also returns an additional full-data
        likelihood. The additional likelihood is only non-None when step 2 is run
        with allosomal data only (``autosomes_in_step_2=False``), and corresponds
        to evaluating the final parameters on autosomal + allosomal data.
    """

    def _format_return(parameters: np.ndarray, likelihood: float, full_likelihood: float | None = None):
        if return_full_likelihood:
            return parameters, likelihood, full_likelihood
        return parameters, likelihood

    if reset_counter:
        global _counter
        _counter = 0

    # ----------- Specify which steps are to be run in the optimization procedure ------------

    step_1, step_2 = _assign_step_indicators(steps)

    if ad_model_allosomes is None and step_2:
        if step_1:
            # Both steps were requested, but allosomes unavailable - downgrade to step 1 only
            logger.info("ad_model_allosomes is None (no allosomal data provided). Forcing step 2 to False and running only step 1.")
            step_2 = False
        else:
            # Step 2 only was explicitly requested, but allosomes unavailable - error
            raise ValueError("ad_model_allosomes is None but step 2 only was explicitly requested. Step 2 requires allosomal data. Please specify steps=[1] or steps=None to run step 1 only or both steps respectively.")

    # ----------- Load data and map to model populations ------------

    # Include autosomal data for inference
    autosome_bins, autosome_data = population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM) 
    n_autosome_bins = len(autosome_bins)

    autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
    for k, v in autosome_data.items():
        autosome_data_mapped[dict(p_dict)[k]] = v
    
    if ad_model_allosomes is not None and step_2: # Include allosomal data for inference at step 2

        allosome_bins, allosome_data = population.get_global_allosome_tractlengths('X', npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
        n_allosome_bins = len(allosome_bins)
        allosome_length = population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
        num_males = population.num_males
        num_females = population.num_females  
        
        female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in female_data.items():
            female_data_mapped[dict(p_dict)[k]] = v
        
        male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in male_data.items():
            male_data_mapped[dict(p_dict)[k]] = v

    # ------------ Set up fixed parameters for the upcoming optimization step ------------

    _best_state = {'objective': np.inf, 'params': None}

    local_parameter_handler = copy.deepcopy(parameter_handler)

    # Optimizer-space overrides applied after extend_parameters() in step 2 to keep
    # ancestry-fixed non-sex-bias parameters frozen at their step-1 / p0 values.
    # Without this, compute_params_fixed_by_ancestry() would re-solve them at every
    # optimizer call given the current sex-bias candidate, letting them drift.
    _ancestry_overrides: dict = {}  # maps param index -> optimizer-space value

    # Identify free sex-bias parameters and fix them at their p0 values (optimizer space) during step 1
    free_sex_bias_parameters_to_fix = {
        param: p0[idx]
        for idx, (param, value) in enumerate(local_parameter_handler.demography.model_base_params.items())
        if (value.type == ParamType.SEX_BIAS)
        and (param not in local_parameter_handler.user_params_fixed_by_value)
        and (param not in local_parameter_handler.params_fixed_by_ancestry)
    }
    
    if step_1:
        # Fix free sex-bias parameters for step 1 optimization (optimize non-sex-bias parameters)
        local_parameter_handler.add_fixed_parameters(free_sex_bias_parameters_to_fix)
    else:
        # Step 2 only: fix non-sex-bias parameters at p0 values (optimize only sex-bias parameters)
        param_names_ordered = list(local_parameter_handler.demography.model_base_params.keys())
        fixed_non_sex_bias = {}
        for idx, param_name in enumerate(param_names_ordered):
            param_info = local_parameter_handler.demography.model_base_params[param_name]
            if (param_info.type != ParamType.SEX_BIAS and 
                param_name not in local_parameter_handler.user_params_fixed_by_value and
                param_name not in local_parameter_handler.params_fixed_by_ancestry):
                if idx < len(p0):
                    fixed_non_sex_bias[param_name] = p0[idx]
        local_parameter_handler.add_fixed_parameters(fixed_non_sex_bias)
        
    # ----------- Define objective function for optimization ------------

    def objective_function(model_base_parameters, include_autosomes=True, include_allosomes=True):
        return _compute_objective(
            model_base_parameters,
            local_parameter_handler=local_parameter_handler,
            best_state=_best_state,
            model_func=model_func,
            outofbounds_fun=outofbounds_fun,
            ad_model_autosomes=ad_model_autosomes,
            ad_model_allosomes=ad_model_allosomes,
            N_cores=N_cores,
            verbose_log=verbose_log,
            verbose_screen=verbose_screen,
            population=population,
            autosome_bins=autosome_bins,
            autosome_data_mapped=autosome_data_mapped,
            allosome_bins=allosome_bins if (ad_model_allosomes is not None and step_2) else None,
            allosome_length=allosome_length if (ad_model_allosomes is not None and step_2) else None,
            female_data_mapped=female_data_mapped if (ad_model_allosomes is not None and step_2) else None,
            male_data_mapped=male_data_mapped if (ad_model_allosomes is not None and step_2) else None,
            num_females=num_females if (ad_model_allosomes is not None and step_2) else None,
            num_males=num_males if (ad_model_allosomes is not None and step_2) else None,
            include_autosomes=include_autosomes,
            include_allosomes=include_allosomes,
            rho_f=rho_f,
            rho_m=rho_m,
            TP=TP
        )

    # Reduced functions are shared by both optimization steps

    def reduced_objective_function(free_parameters_opt, include_autosomes=True, include_allosomes=True):
        extended_parameters = local_parameter_handler.extend_parameters(free_parameters=free_parameters_opt,  # Full parameters in optimizer space
                                                                        units="opt",
                                                                        counter=_counter,
                                                                        verbose_warning_log=verbose_log) # NOTE: Add verbose_warning_screen=verbose_screen if RuntimeWarnings should be printed on screen.
        for _idx, _val in _ancestry_overrides.items():
            extended_parameters[_idx] = _val

        return objective_function(model_base_parameters=extended_parameters,
                                  include_autosomes=include_autosomes,
                                  include_allosomes=include_allosomes) 

    def reduced_outofbounds_fun(free_parameters_opt):
        _extended_oob = local_parameter_handler.extend_parameters(free_parameters=free_parameters_opt, # Full parameters in optimizer space
                                                                   units="opt")
        for _idx, _val in _ancestry_overrides.items():
            _extended_oob[_idx] = _val
        return outofbounds_fun(_extended_oob)

    current_best_parameters = np.array(p0)  # default; overwritten by step 1 if it runs

    if step_1:
        p0_adjusted = p0
        
        """
        if local_parameter_handler.has_known_proportions:
            
            p0_adjusted = local_parameter_handler.optimize_rates_to_match_ancestry(np.array(p0), units="opt")
            try:
                p0_adjusted_phys = local_parameter_handler.convert_to_physical_params(p0_adjusted)
                _check_matrices = local_parameter_handler.demography.get_migration_matrices(p0_adjusted_phys)
                if not all(np.all(np.isfinite(m)) for m in _check_matrices.values()):
                    raise ValueError("Migration matrices contain non-finite values.")
            except Exception:
                logger.warning("optimize_rates_to_match_ancestry produced an infeasible starting point; falling back to p0.")
                p0_adjusted = p0
        """
        
        reduced_p0 = local_parameter_handler.reduce_parameters(np.array(p0_adjusted))
        
        # ------------ Run first optimization step on autosomal data across non-sex-bias parameters ------------

        if ad_model_allosomes is not None and step_2:
            subtitle_message = f"Optimization is performed in two steps.\nStep 1 : Optimizing autosomal likelihood over parameters {str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."
        else:
            subtitle_message = f"Step 1 : Optimizing autosomal likelihood over parameters {str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."      

        if print_step_header:
            print(subtitle_message)
            logger.info(subtitle_message)

        for l in [_TABLE_HEADER, _LINE_HEADER]:
            if verbose_log>0:
                logger.info(l)
            if verbose_screen>0:
                print(l)
            
        reduced_objective_autosomes = lambda x: reduced_objective_function(x, include_allosomes=False)
    
        outputs = scipy.optimize.fmin_cobyla(func=reduced_objective_autosomes,
                                            x0=reduced_p0,
                                            cons=reduced_outofbounds_fun,
                                            rhobeg=.01,
                                            rhoend=.0001,
                                            maxfun=maxiter)
    
        # Final flush: always show the last result at the end of step 1
        _flush_final('Autosomes', best_state=_best_state, verbose_log=verbose_log, verbose_screen=verbose_screen, local_parameter_handler=local_parameter_handler)
        current_best_parameters = local_parameter_handler.extend_parameters(free_parameters=outputs,
                                                                        units="opt",
                                                                        show_ancestry_warning=True)
        final_message = f"Optimization completed"

    if step_2:
    
        # ------------ Run second optimization step on sex-bias parameters ------------
        
        if step_1:
            # Step 1 ran: release sex-bias parameters and fix optimized non-sex-bias parameters
            new_fixed_parameters_names = local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices)
            new_fixed_values = current_best_parameters[local_parameter_handler.free_parameters_indices]
            new_fixed_parameters = dict(zip(new_fixed_parameters_names, new_fixed_values))
            local_parameter_handler.release_fixed_parameters(free_sex_bias_parameters_to_fix.keys())
            local_parameter_handler.add_fixed_parameters(new_fixed_parameters)
        # If step 2 only: current_best_parameters already holds p0, non-sex-bias already fixed, sex-bias already free

        # Freeze ancestry-fixed non-sex-bias parameters at their step-1 / p0 values for
        # the entirety of step 2. compute_params_fixed_by_ancestry() would otherwise
        # re-solve them at each call to extend_parameters(), causing them to drift with
        # the sex-bias candidate instead of remaining fixed.
        _step2_param_names = list(local_parameter_handler.demography.model_base_params.keys())
        for _idx, _pname in enumerate(_step2_param_names):
            _pinfo = local_parameter_handler.demography.model_base_params[_pname]
            if (_pinfo.type != ParamType.SEX_BIAS
                    and _pname in local_parameter_handler.params_fixed_by_ancestry
                    and _idx < len(current_best_parameters)):
                _ancestry_overrides[_idx] = current_best_parameters[_idx]

        reduced_params = local_parameter_handler.reduce_parameters(current_best_parameters)

        step_2_data = "autosomal + allosomal" if autosomes_in_step_2 else "allosomal"            
        step_2_message_1 = f"Step 2 : Optimizing {step_2_data} likelihood over parameters : {str(list(free_sex_bias_parameters_to_fix.keys()))}."
        step_2_message = f"{step_2_message_1}\nNon-sex-bias parameters fixed at initial values." if not step_1 else f"{step_2_message_1}\nNon-sex-bias parameters fixed at values from previous optimization step."    
        line = "-" * len(step_2_message_1)
    
        if len(reduced_params)>0 and verbose_log>0:
            if print_step_header:
                logger.info(line)
                logger.info(step_2_message)
            if ad_model_allosomes is not None:    
                logger.info(_TABLE_HEADER)
                logger.info(_LINE_HEADER)
        if len(reduced_params)>0 and verbose_screen>0:
            if print_step_header:
                print(line)
                print(step_2_message)
            if ad_model_allosomes is not None:
                print(_TABLE_HEADER)
                print(_LINE_HEADER)

        _best_state['objective'] = np.inf
        _best_state['params'] = None

        reduced_objective_allosomes = lambda x: reduced_objective_function(x, include_autosomes=autosomes_in_step_2, include_allosomes=True)
        
        if len(reduced_params)>0:
            outputs = scipy.optimize.fmin_cobyla(func=reduced_objective_allosomes,
                                                x0=reduced_params,
                                                cons=reduced_outofbounds_fun,
                                                rhobeg=.01,
                                                rhoend=.0001,
                                                maxfun=maxiter)
            
            # Final flush: always show the last result at the end of step 2
            _flush_final('Total' if autosomes_in_step_2 else 'Allosomes', best_state=_best_state, verbose_log=verbose_log, verbose_screen=verbose_screen, local_parameter_handler=local_parameter_handler)

            step2_full_params_opt = local_parameter_handler.extend_parameters(free_parameters=outputs,
                                                                               units="opt",
                                                                               show_ancestry_warning=True) # Checks for the ancestry warning at the end of step 2.
            final_message = f"Optimization completed."
            line = "-" * len(final_message)
            for l in [final_message, line]:
                print(l)
                logger.info(l)

            # ------------ Return optimal parameters corresponding to best likelihood ------------
            
            if _best_state['params'] is None:
                try:
                    fallback_likelihood = -objective_function(step2_full_params_opt, include_autosomes=autosomes_in_step_2, include_allosomes=True)
                    full_data_likelihood = None
                    if not autosomes_in_step_2:
                        prev_best_objective = _best_state['objective']
                        prev_best_params = _best_state['params']
                        full_data_likelihood = -objective_function(step2_full_params_opt, include_autosomes=True, include_allosomes=True)
                        _best_state['objective'] = prev_best_objective
                        _best_state['params'] = prev_best_params
                    return _format_return(step2_full_params_opt, fallback_likelihood, full_data_likelihood)
                except Exception:
                    return _format_return(step2_full_params_opt, -1e32, None)

            full_data_likelihood = None
            if not autosomes_in_step_2:
                prev_best_objective = _best_state['objective']
                prev_best_params = _best_state['params']
                full_data_likelihood = -objective_function(_best_state['params'], include_autosomes=True, include_allosomes=True)
                _best_state['objective'] = prev_best_objective
                _best_state['params'] = prev_best_params
            return _format_return(_best_state['params'], -_best_state['objective'], full_data_likelihood)
        
        else:
            final_message = f"No free parameters to optimize at step 2. Optimization completed."

    # ------------ Return optimal parameters corresponding to best likelihood ------------
    
    line = "-" * len(final_message)
    for l in [final_message, line]:
        print(l)
        logger.info(l)
            
    if _best_state['params'] is None:
        try:
            if step_2:
                fallback_likelihood = -objective_function(current_best_parameters,
                                                          include_autosomes=autosomes_in_step_2,
                                                          include_allosomes=True)
                full_data_likelihood = None
                if not autosomes_in_step_2:
                    prev_best_objective = _best_state['objective']
                    prev_best_params = _best_state['params']
                    full_data_likelihood = -objective_function(current_best_parameters,
                                                               include_autosomes=True,
                                                               include_allosomes=True)
                    _best_state['objective'] = prev_best_objective
                    _best_state['params'] = prev_best_params
                return _format_return(current_best_parameters, fallback_likelihood, full_data_likelihood)
            else:
                fallback_likelihood = -objective_function(current_best_parameters, include_allosomes=False)
            return _format_return(current_best_parameters, fallback_likelihood, None)
        except Exception:
            return _format_return(current_best_parameters, -1e32, None)
    return _format_return(_best_state['params'], -_best_state['objective'], None)
