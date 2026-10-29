"""
Console/logging helper functions used by :mod:`tracts.core`'s optimization
functions (:func:`~tracts.core.optimize_cob_sex_biased_single_step`,
:func:`~tracts.core.optimize_cob_sex_biased_two_steps`), factored out to keep
the optimization loops themselves focused on control flow rather than
print/logging boilerplate. Also holds the checks that step 2 of a two-step
optimization only changes sex-bias parameters
(:func:`check_step_2_kept_non_sex_bias_parameters`) and, where the Monoecious
autosomal model is guaranteed to ignore sex bias, that it leaves the autosomal
log-likelihood unchanged (:func:`check_monoecious_autosomal_loglik_unchanged`).
"""
import logging
import numpy as np
from tracts.util import eprint
from tracts.demography.parameter import ParamType
from tracts.demography.parametrized_demography import PulseEvent
from tracts.genetic_model import GeneticModel
from tracts.likelihood_options import LikelihoodOptions
from tracts.tracts_data import TractsData
logger = logging.getLogger(__name__)

def _print_and_log(*messages: str) -> None:
    """
    Prints and logs each message, unconditionally.
    """
    for message in messages:
        print(message)
        logger.info(message)


def _print_verbose(lines: list, verbose_log: int, verbose_screen: int) -> None:
    """
    Logs `lines` if verbose_log > 0, and prints them if verbose_screen > 0.
    """
    if verbose_log > 0:
        for line in lines:
            logger.info(line)
    if verbose_screen > 0:
        for line in lines:
            print(line)


def _print_periodic(lines: list, verbose_log: int, verbose_screen: int, counter: int) -> None:
    """
    Logs/prints `lines`, gated on `counter` landing on a verbose_log/verbose_screen checkpoint.
    """
    if (verbose_log > 0) and (counter % verbose_log == 0):
        for line in lines:
            logger.info(line)
    if (verbose_screen > 0) and (counter % verbose_screen == 0):
        for line in lines:
            print(line)


def _print_single_step_header(parameter_handler, print_step_header: bool, verbose_log: int,
                            verbose_screen: int, counter: int) -> None:
    """
    Builds and reports the single-step optimization header (title + iteration table header).
    """
    subtitle_message = (
        "Optimizing model likelihood over parameters "
        f"{str(parameter_handler.indices_to_labels(parameter_handler.free_parameters_indices))}."
    )
    subsubtitle_message = "Iter.\t Log-likelihood\t Model parameters\t Transmission"
    line = "-" * len(subsubtitle_message)

    if print_step_header:
        _print_and_log(subtitle_message)

    _print_periodic([subsubtitle_message, line], verbose_log, verbose_screen, counter)


def _get_steps(steps: list, ad_model_allosomes) -> tuple:
    """
    Validates and normalizes the `steps` argument of optimize_cob_sex_biased_two_steps,
    returning which of step 1 / step 2 should be run.

    Downgrades step 2 to False (with a log message) if both steps were requested but
    ad_model_allosomes is None (no allosomal data available); raises if step 2 only was
    explicitly requested in that case.
    """
    if steps is not None: # Validate steps argument
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

    step_1 = 1 in normalized_steps
    step_2 = 2 in normalized_steps

    if ad_model_allosomes is None and step_2:
        if step_1:
            # Both steps were requested, but allosomes unavailable - downgrade to step 1 only
            logger.info("ad_model_allosomes is None (no allosomal data provided). Forcing step 2 to False and running only step 1.")
            step_2 = False
        else:
            # Step 2 only was explicitly requested, but allosomes unavailable - error
            raise ValueError("ad_model_allosomes is None but step 2 only was explicitly requested. Step 2 requires allosomal data. Please specify steps=[1] or steps=None to run step 1 only or both steps respectively.")

    return step_1, step_2


def _build_cobyla_stop_message(result, maxfun: int | None, step_label: str | None = None) -> str:
    """
    Builds the message reporting why a COBYLA run stopped, from the ``scipy.optimize.OptimizeResult``
    returned by ``scipy.optimize.minimize(method='COBYLA')``: it used up its budget of objective
    evaluations (so it may have been cut off before converging, and restarting it from its result
    can still improve the likelihood), it converged (trust-region radius reached ``rhoend``), or it
    stopped early for another reason reported by scipy (e.g. rounding errors or a NaN result).
    """
    prefix = f"{step_label}: " if step_label else ""
    limit_reached = ((maxfun is not None and result.nfev >= maxfun)
                     or (not result.success and "function evaluations" in str(result.message).lower()))
    if limit_reached:
        limit = f" (maximum_iterations = {maxfun})" if maxfun is not None else ""
        return (f"{prefix}COBYLA stopped after {result.nfev} objective evaluations: the evaluation limit{limit} "
                "was reached, so the optimizer may not have converged.")
    if result.success:
        return (f"{prefix}COBYLA converged after {result.nfev} objective evaluations: the trust-region radius "
                "reached its final value (rhoend).")
    return (f"{prefix}COBYLA stopped after {result.nfev} objective evaluations without converging: "
            f"{result.message} (status {result.status}).")


def _flush_final_result(best_state: dict, parameter_handler, verbose_log: int, verbose_screen: int,
                        counter: int, note: str = '') -> None:
    """
    Reports the best result found so far, if the last periodic report (gated by
    verbose_log/verbose_screen every `counter`-th iteration) would have missed it.
    Called once at the end of an optimization run/step so the final best iterate
    is always shown even when it doesn't land on a reporting checkpoint.
    """
    if best_state['params'] is None:
        return
    needs_log = verbose_log > 0 and (counter % verbose_log != 0)
    needs_screen = verbose_screen > 0 and (counter % verbose_screen != 0)
    if not (needs_log or needs_screen):
        return
    prev_time_param_logging = parameter_handler.enable_time_param_logging
    parameter_handler.enable_time_param_logging = False
    try:
        param_str = 'array([%s])' % (', '.join(
            ['%- 12g' % v for v in parameter_handler.convert_to_physical_params(best_state['params'], report_non_admissible=False)]
        ))
    finally:
        parameter_handler.enable_time_param_logging = prev_time_param_logging

    loglik = best_state.get('loglik')
    if loglik is not None:
        # One row per computed component, matching the per-iteration reporting.
        rows = [
            (value, component_note)
            for value, component_note in (
                (loglik.autosomes, 'Autosomes'),
                (loglik.female_allosomes, 'Female allosomes'),
                (loglik.male_allosomes, 'Male allosomes'),
            )
            if value is not None
        ]
    else:
        # No per-component breakdown (e.g. penalty state): fall back to a single summed row.
        rows = [(-best_state['objective'], note)]

    for value, row_note in rows:
        if needs_log:
            logger.info("iter=%-6d | obj=%-12g | params=%s %s", counter, value, param_str, row_note)
        if needs_screen:
            eprint('%-8i, %-12g, %s, %s' % (counter, value, param_str, row_note))


def _print_step2_header(step_1: bool, autosomes_in_step_2: bool, free_sex_bias_parameters,
                        table_header: str, line_header: str, print_step_header: bool,
                        ad_model_allosomes, has_free_params: bool,
                        verbose_log: int, verbose_screen: int) -> None:
    """
    Builds and reports the step-2 optimization header (title + iteration table header),
    if there are free sex-bias parameters left to optimize (`has_free_params`).
    """
    if not has_free_params:
        return

    step_2_data = "autosomal + allosomal" if autosomes_in_step_2 else "allosomal"
    step_2_message_1 = f"Step 2 : Optimizing {step_2_data} likelihood over parameters : {str(list(free_sex_bias_parameters.keys()))}."
    step_2_message = (
        f"{step_2_message_1}\nNon-sex-bias parameters fixed at initial values." if not step_1
        else f"{step_2_message_1}\nNon-sex-bias parameters fixed at values from previous optimization step."
    )
    line = "-" * len(step_2_message_1)

    if print_step_header:
        _print_verbose([line, step_2_message], verbose_log, verbose_screen)
    if ad_model_allosomes is not None:
        _print_verbose([table_header, line_header], verbose_log, verbose_screen)


# ------------------ Step-2 parameter-freezing check ------------------

def check_step_2_kept_non_sex_bias_parameters(model_base_params: dict, step_1_params: np.ndarray, final_params: np.ndarray,
                                              rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """
    Checks that step 2 of a two-step optimization kept every non-sex-bias parameter at the value
    it was fixed at for step 2 (the step-1 optimum, or ``p0`` when only step 2 is run), i.e. that
    step 2 only changed sex-bias parameters. A change is reported as a warning rather than
    raised, so that it cannot abort a long inference run.

    Comparing the parameters themselves, rather than e.g. the autosomal log-likelihood, makes the
    check valid for every demography and autosomal model: even under the Monoecious model, the
    autosomal log-likelihood can depend on sex bias (for pulses at fractional times, the averaged
    female/male migration matrix is nonlinear in the sex-specific rates).

    Parameters
    ----------
    model_base_params: dict
        The demographic model's ``model_base_params`` (parameter name -> parameter info, with its
        ``type``), in the order of the parameter vectors.
    step_1_params: np.ndarray
        The full parameters that step 2 started from, in physical units.
    final_params: np.ndarray
        The full final (step-2) parameters, in physical units.
    rtol, atol: float
        Relative and absolute tolerances of the comparison. Default to 1e-8 and 1e-10.

    Returns
    -------
    bool
        True if all non-sex-bias parameters are unchanged, False otherwise (a warning is logged).
    """
    changed_params = [
        f"{name}: {step_1_value:.10g} -> {final_value:.10g}"
        for (name, param_info), step_1_value, final_value in zip(model_base_params.items(),
                                                                 step_1_params, final_params)
        if param_info.type != ParamType.SEX_BIAS
        and not np.isclose(step_1_value, final_value, rtol=rtol, atol=atol, equal_nan=True)
    ]

    if not changed_params:
        logger.info("Check passed: step 2 kept all non-sex-bias parameters at the values it started from.")
        return True

    warning_message = ("WARNING: Step 2 changed non-sex-bias parameters, although it should only optimize "
                       f"sex-bias parameters ({'; '.join(changed_params)}). This is a bug.")
    print(warning_message)
    logger.warning(warning_message)
    return False


# ------------------ Monoecious autosomal log-likelihood check ------------------

def _monoecious_sex_bias_invariance_violation(demography, params_list: list[np.ndarray]) -> str | None:
    """
    Returns why the averaged female/male migration matrix seen by the Monoecious autosomal model
    may depend on the sex-bias parameters for ``demography`` at the (physical) parameter sets in
    ``params_list``, or None if it cannot.

    Each sex-specific rate is ``rate +- sex_bias * c(rate)``, so the average over sexes of any
    migration-matrix entry that is linear in the rate does not depend on sex bias. This holds for
    founder events and continuous migration, but not for:

    - pulses at fractional times, spread over two generations with a rate nonlinear in the pulse rate;
    - continuous founder events, which divide each rate by the sum of the founding rates;
    - founder events whose founding rates for one sex sum to slightly more than 1 (within the
      tolerance below which ``FounderEvent.execute`` clips the remainder rate at 0, leaving the
      founding row summing to more than 1).
    """
    for founder_event in demography.founder_events.values():
        if getattr(founder_event, "end_time", None) is not None:
            return "the demography has a continuous founder event"

    for params in params_list:
        for events in demography.events.values():
            for event in events:
                if isinstance(event, PulseEvent):
                    pulse_time = demography.get_param_value(param_name=event.time_parameter, params=params)
                    if not np.isclose(pulse_time, np.round(pulse_time), rtol=0, atol=1e-12):
                        return f"the pulse at time {event.time_parameter} = {pulse_time:.6g} is at a fractional time"
        for matrix in demography.get_migration_matrices(params).values():
            if np.any(matrix.sum(axis=1) > 1 + 1e-12):
                return "the founding rates of one sex sum to more than 1, so the remainder rate is clipped at 0"
    return None


def _compute_autosomal_loglik(params: np.ndarray, genetic_model: GeneticModel, tracts_data: TractsData) -> float:
    """
    Computes the autosomal log-likelihood at the (physical) parameters ``params``, with the
    female and male migration matrices passed separately to the autosomal model, so that the
    result reflects how the autosomal model itself depends on the sex-bias parameters.
    """
    matrices = genetic_model.demographic_model.get_migration_matrices(params)
    male_matrix, female_matrix = genetic_model.split_migration_matrices(matrices)
    loglik = genetic_model.loglik(male_matrix=male_matrix,
                                  female_matrix=female_matrix,
                                  tracts_data=tracts_data,
                                  likelihood_options=LikelihoodOptions(include_autosomes=True,
                                                                       include_allosomes=False,
                                                                       verbose_log=0,
                                                                       verbose_screen=0))
    return loglik.autosomes


def check_monoecious_autosomal_loglik_unchanged(genetic_model: GeneticModel, tracts_data: TractsData,
                                                step_1_autosomal_loglik: float, step_1_params: np.ndarray,
                                                final_params: np.ndarray, rtol: float = 1e-8,
                                                atol: float = 1e-8) -> bool | None:
    """
    Checks that step 2 of a two-step optimization did not change the autosomal log-likelihood
    found at the end of step 1, when autosomes are modelled with the Monoecious (M) model and the
    demography guarantees that M ignores the sex-bias parameters (see
    ``_monoecious_sex_bias_invariance_violation``). Step 2 only changes sex-bias parameters, so
    the autosomal log-likelihood at the final step-2 parameters must then equal the step-1 one.
    A mismatch is reported as a warning rather than raised, so that it cannot abort a long run.

    Parameters
    ----------
    genetic_model: GeneticModel
        The genetic model used in the optimization (with a sex-biased demographic model).
    tracts_data: TractsData
        The tract-length data used in the optimization (allosomal data is not needed).
    step_1_autosomal_loglik: float
        The autosomal log-likelihood at the best step-1 parameters.
    step_1_params: np.ndarray
        The full parameters that step 2 started from, in physical units.
    final_params: np.ndarray
        The full final (step-2) parameters, in physical units.
    rtol, atol: float
        Relative and absolute tolerances of the comparison. Default to 1e-8.

    Returns
    -------
    bool | None
        True if the autosomal log-likelihood is unchanged, False if it changed (a warning is
        logged), and None if the check does not apply (autosomal model is not M, or M may depend
        on sex bias for this demography and parameters; logged) or cannot be performed
        (penalty-only step-1 or step-2 result, or the final likelihood cannot be evaluated; logged).
    """
    if genetic_model.phase_type_config.ad_model_autosomes != 'M':
        return None

    violation = _monoecious_sex_bias_invariance_violation(genetic_model.demographic_model, [step_1_params, final_params])
    if violation is not None:
        logger.info(f"Skipping the check that step 2 leaves the Monoecious autosomal log-likelihood unchanged: "
                    f"{violation}, so this log-likelihood may depend on sex bias.")
        return None

    # Out-of-bounds or infeasible parameters only have a penalty objective (see compute_objective
    # in tracts.core, whose out-of-bounds threshold -1e-14 is mirrored here), not a likelihood to compare.
    skip_reason = None
    if not np.isfinite(step_1_autosomal_loglik):
        skip_reason = "the step-1 log-likelihood is not finite"
    elif genetic_model.demographic_model.get_violation_score(final_params) < -1e-14:
        skip_reason = "the final step-2 parameters are out of bounds"
    else:
        try:
            final_autosomal_loglik = _compute_autosomal_loglik(params=final_params,
                                                              genetic_model=genetic_model,
                                                              tracts_data=tracts_data)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            skip_reason = f"the final autosomal log-likelihood could not be evaluated ({exc})"
        else:
            if not np.isfinite(final_autosomal_loglik):
                skip_reason = "the final autosomal log-likelihood is not finite"

    if skip_reason is not None:
        logger.warning(f"Skipping the check that step 2 leaves the Monoecious autosomal log-likelihood unchanged: "
                       f"{skip_reason}.")
        return None

    if np.isclose(final_autosomal_loglik, step_1_autosomal_loglik, rtol=rtol, atol=atol):
        logger.info(f"Check passed: the autosomal log-likelihood ({final_autosomal_loglik:.6f}) is unchanged by "
                    "step 2, as expected for the Monoecious (M) autosomal model.")
        return True

    warning_message = (f"WARNING: The autosomal log-likelihood changed from {step_1_autosomal_loglik:.6f} (end of step 1) "
                       f"to {final_autosomal_loglik:.6f} (final step-2 parameters), although the Monoecious (M) "
                       "autosomal model does not depend on sex-bias parameters for this demography and step 2 only "
                       "optimizes sex-bias parameters. This is a bug.")
    print(warning_message)
    logger.warning(warning_message)
    return False
