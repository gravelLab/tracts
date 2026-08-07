"""
Console/logging helper functions used by :mod:`tracts.core`'s optimization
functions (:func:`~tracts.core.optimize_cob_sex_biased_single_step`,
:func:`~tracts.core.optimize_cob_sex_biased_two_steps`), factored out to keep
the optimization loops themselves focused on control flow rather than
print/logging boilerplate.
"""
import logging
from tracts.util import eprint
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
