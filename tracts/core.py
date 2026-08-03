import logging
import numpy as np
import scipy.optimize
from tracts.population import Population
from tracts.util import eprint
from tracts.demography.parameter import ParamType
from tracts.genetic_model import GeneticModel
from tracts.tracts_data import TractsData
from tracts.likelihood_options import LikelihoodOptions
from tracts.core_utils import (
    _print_and_log,
    _print_verbose,
    _print_single_step_header,
    _get_steps,
    _flush_final_result,
    _print_step2_header,
)
logger = logging.getLogger(__name__)

_counter = 0
_out_of_bounds_val = -1e32
_min_out_of_bounds_val = -1e-10
_ignore_oob_above = -1e-14

# ------------------ Objective function ------------------

def _compute_objective(
    parameters,
    *,
    best_state,
    local_genetic_model: GeneticModel,
    tracts_data: TractsData,
    likelihood_options: LikelihoodOptions,
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
    best_state : dict
        Mutable dict with keys ``'objective'`` (float, lowest objective seen so
        far, initialised to ``np.inf``) and ``'params'`` (np.ndarray or None,
        the corresponding parameter vector). Updated in place whenever a new best
        finite objective is found.
    local_genetic_model : GeneticModel
        Deep copy of the original genetic model (demographic model + admixture/
        phase-type configuration), local to this optimization run. Its
        ``parameter_handler`` is used to convert optimizer-space parameters to
        physical parameters for logging, its ``model_func``/``outofbounds_fun``
        methods compute the migration matrices and violation score for
        ``parameters``, and its ``phase_type_config`` supplies the admixture
        models and recombination/pedigree settings used to compute the
        likelihood.
    tracts_data : TractsData
        The population and mapped autosomal/allosomal tract-length histogram data
        used to compute the likelihood. Its allosome-related fields
        (``allosome_bins``, ``allosome_length``, ``female_data_mapped``,
        ``male_data_mapped``, ``num_females``, ``num_males``) are required when
        ``likelihood_options.include_allosomes=True``.
    likelihood_options : LikelihoodOptions
        Logging verbosity (``verbose_log``, ``verbose_screen``) and autosome/allosome
        inclusion flags (``include_autosomes``, ``include_allosomes``) for this
        evaluation.

    Returns
    -------
    float
        The objective value (negative total log-likelihood), or a large positive
        penalty when parameters are out of bounds or produce a singular matrix.
    """
    local_parameter_handler = local_genetic_model.parameter_handler
    include_allosomes = likelihood_options.include_allosomes
    verbose_log = likelihood_options.verbose_log
    verbose_screen = likelihood_options.verbose_screen
    global _counter
    global _out_of_bounds_val
    global _min_out_of_bounds_val
    _counter += 1

    def flush_result(result, note=''):
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

    oob = local_genetic_model.outofbounds_fun(parameters)
    if oob < _ignore_oob_above:
        out = oob * _out_of_bounds_val - _min_out_of_bounds_val
        flush_result(out, f'OOB (oob={oob})')
        if np.isfinite(out) and out < best_state['objective']:
            best_state['objective'] = out
            best_state['params'] = parameters.copy()
        return out

    try:
        matrices = local_genetic_model.model_func(parameters)
        matrix_list = [matrix for matrix in matrices.values()]
        if include_allosomes:
            [male_matrix, female_matrix] = matrix_list
        else:
            avg_matrix = np.mean(matrix_list, axis=0)
            male_matrix = avg_matrix
            female_matrix = avg_matrix

        loglik = local_genetic_model.loglik(
            male_matrix=male_matrix,
            female_matrix=female_matrix,
            tracts_data=tracts_data,
            likelihood_options=likelihood_options,
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # Covers both a singular/infeasible model_func(parameters) call and a singular
        # migration matrix rejected during phase-type model construction inside loglik().
        out = -_out_of_bounds_val - _min_out_of_bounds_val  # large positive penalty, consistent with OOB
        flush_result(out, 'Singular matrix (infeasible params)')
        if np.isfinite(out) and out < best_state['objective']:
            best_state['objective'] = out
            best_state['params'] = parameters.copy()
        return out

    if loglik.autosomes is not None:
        flush_result(loglik.autosomes, 'Autosomes')
    if loglik.female_allosomes is not None:
        flush_result(loglik.female_allosomes, 'Female allosomes')
    if loglik.male_allosomes is not None:
        flush_result(loglik.male_allosomes, 'Male allosomes')

    obj = -loglik.total

    if np.isfinite(obj) and obj < best_state['objective']:
        best_state['objective'] = obj
        best_state['params'] = parameters.copy()
    return obj


# ------------------ Single-step optimization ------------------


def optimize_cob_sex_biased_single_step(p0:list, population: Population, genetic_model: GeneticModel,
                            likelihood_options: LikelihoodOptions | None = None, p_dict:dict=None, exclude_tracts_below_cM:float=0,
                            maxiter:int=None, reset_counter:bool=True,
                            npts:int=50, print_step_header:bool=True) -> tuple[np.ndarray, float]:
    """
    Optimizes the log-likelihood over all parameters defined by the demographic model, given a specified pair of admixture models for autosomes and allosomes.
    The optimization is carried out jointly in a single step, estimating all parameters simultaneously using both autosomal and allosomal data.

    Parameters
    ----------
    p0: list
            An array of initial parameters to start the optimization.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    genetic_model: GeneticModel
        Bundles the demographic model (whose ``parameter_handler`` handles parameter
        transformations and fixed parameters, and whose ``model_func``/``outofbounds_fun``
        methods compute migration matrices and violation scores) with the admixture and
        phase-type model configuration (``ad_model_autosomes``, ``ad_model_allosomes``,
        ``rho_f``, ``rho_m``, ``TP``, ``N_cores``) used to compute the likelihood.
    likelihood_options: LikelihoodOptions | None
        Logging verbosity (``verbose_log``, ``verbose_screen``) for this optimization run.
        Its ``include_autosomes``/``include_allosomes`` flags are ignored here: allosomes
        are included whenever ``genetic_model.phase_type_config.ad_model_allosomes`` is not
        None. If None, defaults to ``LikelihoodOptions()``.
    p_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    maxiter: int, default: None
        Maximum iterations to run for.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.
    print_step_header: bool, optional
        If True, print the admixture-model title and parameter-set subtitle at the start of the
        optimization. If False, only the iteration table header is printed. For internal use only;
        set automatically by :func:`~tracts.driver.run_model_multi_init` to suppress repeated
        headers across multiple runs within the same step. Default is True.

    Returns
    -------
    tuple [np.ndarray, float]
        A tuple containing the optimal parameters found and the corresponding likelihood.
    """

    if reset_counter:
        global _counter
        _counter = 0

    likelihood_options = likelihood_options if likelihood_options is not None else LikelihoodOptions()
    verbose_log = likelihood_options.verbose_log
    verbose_screen = likelihood_options.verbose_screen
    ad_model_allosomes = genetic_model.phase_type_config.ad_model_allosomes

    tracts_data = TractsData.from_population(
        population=population,
        p_dict=p_dict,
        npts=npts,
        exclude_tracts_below_cM=exclude_tracts_below_cM,
        include_allosomes=ad_model_allosomes is not None,
    )

    local_genetic_model = genetic_model.copy()
    local_parameter_handler = local_genetic_model.parameter_handler
    _best_state = {'objective': np.inf, 'params': None}
    _likelihood_options = likelihood_options.with_overrides(include_allosomes=ad_model_allosomes is not None)

    def objective_function(parameters):
        return _compute_objective(
            parameters,
            local_genetic_model=local_genetic_model,
            best_state=_best_state,
            tracts_data=tracts_data,
            likelihood_options=_likelihood_options,
        )
    # ------------ Define reduced objective function and out-of-bounds function for optimization ------------

    def reduced_objective_function(free_parameters_opt):
        extended_parameters = local_parameter_handler.extend_parameters(free_parameters=free_parameters_opt,
                                                                        units="opt",
                                                                        counter=_counter,
                                                                        verbose_warning_log=verbose_log) # NOTE: Add verbose_warning_screen=verbose_screen if RuntimeWarnings should be printed on screen.

        return objective_function(extended_parameters) #Full parameters in optimizer space

    def reduced_outofbounds_fun(free_parameters_opt):
        return local_genetic_model.outofbounds_fun(local_parameter_handler.extend_parameters(free_parameters=free_parameters_opt,
                                                                        units="opt")) #Full parameters in optimizer space

    reduced_p0 = local_parameter_handler.reduce_parameters(p0) # Initial parameters

    # ------------ Run single-step optimization ------------

    _print_single_step_header(local_parameter_handler, print_step_header, verbose_log, verbose_screen, _counter)

    outputs = scipy.optimize.fmin_cobyla(func=reduced_objective_function,
                                        x0=reduced_p0,
                                        cons=reduced_outofbounds_fun,
                                        rhobeg=.01,
                                        rhoend=.0001,
                                        maxfun=maxiter)

    optimized_parameters = local_parameter_handler.extend_parameters(free_parameters=outputs,
                                                                    units="opt",
                                                                    show_ancestry_warning=True)

    _flush_final_result(_best_state, local_parameter_handler, verbose_log, verbose_screen, _counter) # Final flush: always show the last result at the end of the optimization run

    # ------------ Return optimal parameters corresponding to best likelihood ------------

    if _best_state['params'] is None:
        try:
            fallback_likelihood = -objective_function(optimized_parameters)
            return optimized_parameters, fallback_likelihood
        except Exception:
            return optimized_parameters, -1e32

    return _best_state['params'], -_best_state['objective']


# ------------------ Two-steps optimization ------------------

def optimize_cob_sex_biased_two_steps(p0:list, population: Population, genetic_model: GeneticModel,
                                    likelihood_options: LikelihoodOptions | None = None,
                                    p_dict:dict=None, exclude_tracts_below_cM:float=0, maxiter:int=None, reset_counter:bool=True,
                                    autosomes_in_step_2:bool=True,
                                    steps: list[int | str] | None = None, npts:int=50, print_step_header:bool=True,
                                    return_full_likelihood: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float | None]:
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
    genetic_model: GeneticModel
        Bundles the demographic model (whose ``parameter_handler`` handles parameter
        transformations and fixed parameters, and whose ``model_func``/``outofbounds_fun``
        methods compute migration matrices and violation scores) with the admixture and
        phase-type model configuration (``ad_model_autosomes``, ``ad_model_allosomes``,
        ``rho_f``, ``rho_m``, ``TP``, ``N_cores``) used to compute the likelihood. If
        ``ad_model_allosomes`` is None (no allosomal data provided), step 2 cannot be
        run. If only step 2 is requested (steps=[2]), an error is raised. If both
        steps are requested (steps=None or steps=[1,2]), step 2 is automatically
        disabled with a log message.
    likelihood_options: LikelihoodOptions | None
        Logging verbosity (``verbose_log``, ``verbose_screen``) for this optimization run.
        Its ``include_autosomes``/``include_allosomes`` flags are ignored here: which data
        are included is determined per step (autosomes only in step 1; allosomes, and
        optionally autosomes per ``autosomes_in_step_2``, in step 2). If None, defaults to
        ``LikelihoodOptions()``.
    p_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    maxiter: int, default: None
        Maximum iterations to run for.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).
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

    Returns
    -------
    tuple [np.ndarray, float] or tuple [np.ndarray, float, float | None]
        By default, returns the optimal parameters found and the optimization likelihood.
        If ``return_full_likelihood`` is True, also returns an additional full-data
        likelihood. The additional likelihood is only non-None when step 2 is run
        with allosomal data only (``autosomes_in_step_2=False``), and corresponds
        to evaluating the final parameters on autosomal + allosomal data.
    """
    likelihood_options = likelihood_options if likelihood_options is not None else LikelihoodOptions()
    verbose_log = likelihood_options.verbose_log
    verbose_screen = likelihood_options.verbose_screen
    ad_model_allosomes = genetic_model.phase_type_config.ad_model_allosomes

    def _format_return(parameters: np.ndarray, likelihood: float, full_likelihood: float | None = None):
        if return_full_likelihood:
            return parameters, likelihood, full_likelihood
        return parameters, likelihood

    if reset_counter:
        global _counter
        _counter = 0

    # ----------- Specify which steps are to be run in the optimization procedure ------------

    step_1, step_2 = _get_steps(steps, ad_model_allosomes)

    # ----------- Load data and map to model populations ------------

    tracts_data = TractsData.from_population(
        population=population,
        p_dict=p_dict,
        npts=npts,
        exclude_tracts_below_cM=exclude_tracts_below_cM,
        include_allosomes=ad_model_allosomes is not None and step_2,
    )

    # ------------ Set up fixed parameters for the upcoming optimization step ------------

    _best_state = {'objective': np.inf, 'params': None}

    local_genetic_model = genetic_model.copy()
    local_parameter_handler = local_genetic_model.parameter_handler

    # Optimizer-space overrides applied after extend_parameters() in step 2 to keep
    # ancestry-fixed non-sex-bias parameters frozen at their step-1 / p0 values.
    # Without this, compute_params_fixed_by_ancestry() would re-solve them at every
    # optimizer call given the current sex-bias candidate, letting them drift.
    _ancestry_overrides: dict = {}  # maps param index -> optimizer-space value

    # p0 is in optimizer space (converted by the driver); convert to physical so that
    # add_fixed_parameters always stores physical values (required by extend_parameters).
    p0_phys = local_parameter_handler.convert_to_physical_params(np.array(p0))
    param_names_ordered = list(local_parameter_handler.demography.model_base_params.keys())

    # Identify free sex-bias parameters, fixed at their p0 (starting-parameter) values for step 1
    # optimization: this lets a caller carry over previously-optimized sex-bias values (e.g. when
    # re-optimizing) by simply passing them in p0, without permanently fixing them by value, so
    # they remain free to be optimized again in step 2.
    free_sex_bias_parameters = {param: p0_phys[param_names_ordered.index(param)]
                                for param, value in local_parameter_handler.demography.model_base_params.items() if
                                (value.type == ParamType.SEX_BIAS) and
                                (param not in local_parameter_handler.user_params_fixed_by_value) and
                                (param not in local_parameter_handler.params_fixed_by_ancestry)}

    if step_1:
        # Fix free sex-bias parameters for step 1 optimization (optimize non-sex-bias parameters)
        local_parameter_handler.add_fixed_parameters(free_sex_bias_parameters)
    else:
        # Step 2 only: fix non-sex-bias parameters at p0 values (optimize only sex-bias parameters)
        fixed_non_sex_bias = {}
        for idx, param_name in enumerate(param_names_ordered):
            param_info = local_parameter_handler.demography.model_base_params[param_name]
            if (param_info.type != ParamType.SEX_BIAS and
                param_name not in local_parameter_handler.user_params_fixed_by_value and
                param_name not in local_parameter_handler.params_fixed_by_ancestry):
                if idx < len(p0):
                    fixed_non_sex_bias[param_name] = p0_phys[idx]
        local_parameter_handler.add_fixed_parameters(fixed_non_sex_bias)

    # ----------- Define objective function for optimization ------------

    def objective_function(model_base_parameters, include_autosomes=True, include_allosomes=True):
        return _compute_objective(
            model_base_parameters,
            local_genetic_model=local_genetic_model,
            best_state=_best_state,
            tracts_data=tracts_data,
            likelihood_options=likelihood_options.with_overrides(
                include_autosomes=include_autosomes, include_allosomes=include_allosomes
            ),
        )

    # ----------- Reduced functions (shared by both optimization steps) -----------

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
        return local_genetic_model.outofbounds_fun(_extended_oob)

    table_header = "Iter.\t Log-likelihood\t Model parameters\t Transmission"
    line_header = "-" * len(table_header.expandtabs())

    if step_1:

        reduced_p0 = local_parameter_handler.reduce_parameters(p0) # Initial parameters

        # ------------ Run first optimization step on autosomal data across non-sex-bias parameters ------------

        if ad_model_allosomes is not None and step_2:
            subtitle_message = f"Optimization is performed in two steps.\nStep 1 : Optimizing autosomal likelihood over parameters {str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."
        else:
            subtitle_message = f"Step 1 : Optimizing autosomal likelihood over parameters {str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."

        if print_step_header:
            _print_and_log(subtitle_message)

        _print_verbose([table_header, line_header], verbose_log, verbose_screen)

        reduced_objective_autosomes = lambda x: reduced_objective_function(x, include_allosomes=False)

        outputs = scipy.optimize.fmin_cobyla(func=reduced_objective_autosomes,
                                            x0=reduced_p0,
                                            cons=reduced_outofbounds_fun,
                                            rhobeg=.01,
                                            rhoend=.0001,
                                            maxfun=maxiter)

        optimized_parameters = local_parameter_handler.extend_parameters(free_parameters=outputs,
                                                                        units="opt",
                                                                        show_ancestry_warning=True)

        _flush_final_result(_best_state, local_parameter_handler, verbose_log, verbose_screen, _counter, note='Autosomes') # Final flush: always show the last result at the end of step 1
        final_message = "Optimization completed."

    if step_2:

        # ------------ Run second optimization step on sex-bias parameters ------------

        if not step_1:
            # Step 2 only: optimized_parameters starts at p0, non-sex-bias already fixed, sex-bias already free
            optimized_parameters = np.array(p0)
        else:
            # Step 1 ran: release sex-bias parameters and fix optimized non-sex-bias parameters
            new_fixed_parameters_names = local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices)
            optimized_parameters_phys = local_parameter_handler.convert_to_physical_params(optimized_parameters)
            new_fixed_values = optimized_parameters_phys[local_parameter_handler.free_parameters_indices]
            new_fixed_parameters = dict(zip(new_fixed_parameters_names, new_fixed_values))
            local_parameter_handler.release_fixed_parameters(free_sex_bias_parameters.keys())
            local_parameter_handler.add_fixed_parameters(new_fixed_parameters)

        # Freeze ancestry-fixed non-sex-bias parameters at their step-1 / p0 values for
        # the entirety of step 2. compute_params_fixed_by_ancestry() would otherwise
        # re-solve them at each call to extend_parameters(), causing them to drift with
        # the sex-bias candidate instead of remaining fixed.
        _step2_param_names = list(local_parameter_handler.demography.model_base_params.keys())
        for _idx, _pname in enumerate(_step2_param_names):
            _pinfo = local_parameter_handler.demography.model_base_params[_pname]
            if (_pinfo.type != ParamType.SEX_BIAS
                    and _pname in local_parameter_handler.params_fixed_by_ancestry
                    and _idx < len(optimized_parameters)):
                _ancestry_overrides[_idx] = optimized_parameters[_idx]

        reduced_params = local_parameter_handler.reduce_parameters(optimized_parameters)

        _print_step2_header(
            step_1, autosomes_in_step_2, free_sex_bias_parameters,
            table_header, line_header, print_step_header, ad_model_allosomes,
            has_free_params=len(reduced_params) > 0,
            verbose_log=verbose_log, verbose_screen=verbose_screen,
        )

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

            step2_full_params_opt = local_parameter_handler.extend_parameters(free_parameters=outputs,
                                                                               units="opt",
                                                                               show_ancestry_warning=True) # Checks for the ancestry warning at the end of step 2.

            # Final flush: always show the last result at the end of step 2
            _flush_final_result(
                _best_state, local_parameter_handler, verbose_log, verbose_screen, _counter,
                note='Allosomes' if not autosomes_in_step_2 else 'Autosomes + Allosomes',
            )
            final_message = "Optimization completed."
            line = "-" * len(final_message)
            _print_and_log(final_message, line)

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
            final_message = "No free parameters to optimize at step 2. Optimization completed."

    # ------------ Return optimal parameters corresponding to best likelihood ------------

    line = "-" * len(final_message)
    _print_and_log(final_message, line)

    if _best_state['params'] is None:
        try:
            if step_2:
                fallback_likelihood = -objective_function(optimized_parameters,
                                                          include_autosomes=autosomes_in_step_2,
                                                          include_allosomes=True)
                full_data_likelihood = None
                if not autosomes_in_step_2:
                    prev_best_objective = _best_state['objective']
                    prev_best_params = _best_state['params']
                    full_data_likelihood = -objective_function(optimized_parameters,
                                                               include_autosomes=True,
                                                               include_allosomes=True)
                    _best_state['objective'] = prev_best_objective
                    _best_state['params'] = prev_best_params
                return _format_return(optimized_parameters, fallback_likelihood, full_data_likelihood)
            else:
                fallback_likelihood = -objective_function(optimized_parameters, include_allosomes=False)
            return _format_return(optimized_parameters, fallback_likelihood, None)
        except Exception:
            return _format_return(optimized_parameters, -1e32, None)
    return _format_return(_best_state['params'], -_best_state['objective'], None)
