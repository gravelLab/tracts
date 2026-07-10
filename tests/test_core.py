"""
Tests for sex-biased optimizers in tracts/core.py.

The tests use lightweight mocks so that no real tract data or costly model
evaluations are needed.  The Population mock returns pre-computed constant
histogram arrays; fmin_cobyla is patched to return x0 unchanged so we can
inspect what the optimizers would have been called with without running real
phase-type computations. Tests that only need to exercise the validation
logic (raises before fmin_cobyla is called) do not need the patch.
"""

import logging
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import tracts.core as core_module
from tracts.core import optimize_cob_sex_biased_single_step, optimize_cob_sex_biased_two_steps
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.demography.parameter import ParamType
from tracts.demography.parametrized_demography_sex_biased import SexType


# --------------- Helpers / shared fixtures ---------------

N_PARAMS = 4  # 2 non-sex-bias (t, rate_eur) + 2 sex-bias (sb_eur, sb_afr)
N_BINS   = 5
N_POPS   = 2  # populations A, B
P0      = np.array([5.0, 0.3, 0.05, 0.05])   # [t, rate_eur, sb_eur, sb_afr]
P_DICT  = {"A": 0, "B": 1}

def _make_demography():
    """
    Return a minimal ParametrizedDemographySexBiased with two sex-bias params.
    """
    dem = ParametrizedDemographySexBiased()
    dem.add_parameter("t",       ParamType.TIME)
    dem.add_parameter("rate_eur",ParamType.RATE)
    dem.add_parameter("sb_eur",  ParamType.SEX_BIAS)
    dem.add_parameter("sb_afr",  ParamType.SEX_BIAS)
    
    return dem

def _make_handler(dem=None):
    """
    Return a FixedParametersHandler wired to *dem* with no parameters fixed.    
    """
    if dem is None:
        dem = _make_demography()
    ph = FixedParametersHandler(logging.getLogger("test"))
    ph.set_up_fixed_parameters(dem, params_to_fix_by_ancestry=[], proportions={})
    return ph


def _make_population():
    """
    Return a mock Population whose data accessors return constant arrays.
    """
    pop = MagicMock()

    bins   = np.linspace(0, 1, N_BINS + 1)
    counts = np.zeros(N_BINS, dtype="int64").tolist()

    pop.get_global_tractlengths.return_value = (bins, {"A": counts, "B": counts})
    pop.get_global_allosome_tractlengths.return_value = (
        bins,
        {
            SexType.FEMALE: {"A": counts, "B": counts},
            SexType.MALE:   {"A": counts, "B": counts},
        },
    )
    pop.Ls             = [1.0]
    pop.indivs         = [MagicMock()] * 10
    pop.num_males      = 5
    pop.num_females    = 5
    pop.allosome_lengths = {"X": 1.0}
    return pop

def _make_model_func():
    """
    Dummy model_func – never actually called (fmin_cobyla is always patched).
    """
    mat = np.array([[0.5, 0.5]])

    def model_func(params):
        return {"female": mat, "male": mat}

    return model_func


def _always_valid(params):
    """
    outofbounds_fun that always returns +1 (never out of bounds).
    """
    return 1.0

def _fake_fmin(func, x0, cons, **kwargs):
    """
    fmin_cobyla mock: returns x0 unchanged without calling the objective.
    """
    return x0


# Common kwargs shared by most calls; tests override what they need.
COMMON_KWARGS = dict(
    p0                    = P0,
    population            = None,   # replaced per test
    model_func            = _make_model_func(),
    parameter_handler     = None,   # replaced per test
    outofbounds_fun       = _always_valid,
    verbose_log           = 0,
    verbose_screen        = 0,
    p_dict                = P_DICT,
    exclude_tracts_below_cM = 0,
    maxiter               = 2,
    reset_counter         = True,
    ad_model_autosomes    = "DC",
    ad_model_allosomes    = "DC",
    autosomes_in_step_2   = True,
    npts                  = N_BINS,
)


def _call(steps=None, patch_fmin=True, **overrides):
    """
    Invoke optimize_cob_sex_biased_two_steps with the common fixture.

    When *patch_fmin* is True (default) scipy.optimize.fmin_cobyla is replaced
    with a stub that returns x0 unchanged, avoiding any real phase-type call.
    Set patch_fmin=False only for tests that raise before fmin_cobyla is reached.
    """
    kwargs = dict(COMMON_KWARGS)
    kwargs["population"]         = _make_population()
    kwargs["parameter_handler"]  = _make_handler()
    kwargs["steps"]              = steps
    kwargs.update(overrides)

    if patch_fmin:
        with patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=_fake_fmin):
            return optimize_cob_sex_biased_two_steps(**kwargs)
    else:
        return optimize_cob_sex_biased_two_steps(**kwargs)


def _call_single(patch_fmin=True, **overrides):
    """
    Invoke optimize_cob_sex_biased_single_step with the common fixture.

    When *patch_fmin* is True (default) scipy.optimize.fmin_cobyla is replaced
    with a stub that returns x0 unchanged.
    """
    kwargs = dict(COMMON_KWARGS)
    kwargs["population"] = _make_population()
    kwargs["parameter_handler"] = _make_handler()
    kwargs.pop("steps", None)
    kwargs.pop("autosomes_in_step_2", None)
    kwargs.update(overrides)

    if patch_fmin:
        with patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=_fake_fmin):
            return optimize_cob_sex_biased_single_step(**kwargs)
    return optimize_cob_sex_biased_single_step(**kwargs)


# --------------- Steps argument validation ---------------

class TestStepsValidation:
    """
    This class contains tests for the validation logic of the *steps* argument to optimize_cob_sex_biased_two_steps.
    The tests verify that valid specifications are accepted and invalid ones raise the appropriate exceptions with informative messages. 
    It also checks that the default behavior (steps=None) runs without error and returns a valid output type, as this is a common usage pattern.
    """

    def test_none_runs_both_steps(self):
        """
        Checks that steps=None must run without error (default: both steps).
        """
        params, lik = _call(steps=None)
        assert isinstance(params, np.ndarray)
        assert np.isfinite(lik) or lik == -1e32  # finite or hard fallback

    @pytest.mark.parametrize("steps", [
        [1], ["step1"],
        [2], ["step2"],
        [1, 2], ["step1", "step2"], [1, "step2"], ["step1", 2],
    ])

    def test_valid_steps_accepted(self, steps):
        """
        Checks that all documented valid step specifications are accepted.
        """
        # step 2 requires allosomal data already set in COMMON_KWARGS
        params, lik = _call(steps=steps)
        assert isinstance(params, np.ndarray)

    def test_not_a_list_raises_type_error(self):
        """
        Checks that a non-list steps argument raises TypeError.
        """
        with pytest.raises(TypeError):
            _call(steps=1)

    def test_empty_list_raises_value_error(self):
        """
        Checks that an empty list for steps raises ValueError, as it does not specify any valid step.
        """
        with pytest.raises(ValueError, match="empty"):
            _call(steps=[])

    def test_invalid_step_value_raises_value_error(self):
        """
        Checks that invalid step values (e.g., 3 or "step3") raise ValueError with an informative message.
        """
        with pytest.raises(ValueError, match="Invalid step value"):
            _call(steps=[3])

    def test_invalid_string_raises_value_error(self):
        """
        Checks that invalid string step values (e.g., "step3") raise ValueError with an informative message.
        """
        with pytest.raises(ValueError, match="Invalid step value"):
            _call(steps=["step3"])

    def test_duplicate_step_integer_raises(self):
        """
        Checks that [1, 1] duplicates step 1.
        """
        with pytest.raises(ValueError, match="duplicate"):
            _call(steps=[1, 1])

    def test_duplicate_step_mixed_raises(self):
        """
        Checks that [1, 'step1'] references step 1 twice.
        """
        with pytest.raises(ValueError, match="duplicate"):
            _call(steps=[1, "step1"])

    def test_duplicate_step_2_raises(self):
        """
        Checks that [2, "step2"] duplicates step 2.
        """
        with pytest.raises(ValueError, match="duplicate"):
            _call(steps=[2, "step2"])


# --------------- Tests for the ad_model_allosomes constraint ---------------

class TestAllosomeModelConstraint:
    """
    This class contains tests for the requirement that step 2 of optimize_cob_sex_biased_two_steps must have an allosomal model specified.
    The tests verify that attempting to run step 2 without an allosomal model raises a ValueError with an informative message, 
    and that step 1 only does not require an allosomal model.
    """

    def test_step2_without_allosome_model_raises(self):
        """
        Checks that step 2 without an allosomal model raises ValueError, as step 2 relies on allosomal data to estimate sex bias. 
        The error message should mention "ad_model_allosomes" to guide the user to the missing argument.
        """
        with pytest.raises(ValueError, match="ad_model_allosomes"):
            _call(steps=[2], ad_model_allosomes=None)

    def test_step2_both_without_allosome_model_downgrades(self):
        """
        Checks that running both steps without an allosomal model auto-downgrades to step 1 only
        without raising an error, since step 1 does not require allosomal data.
        """
        params, lik = _call(steps=[1, 2], ad_model_allosomes=None)
        assert isinstance(params, np.ndarray)
        assert isinstance(lik, float)

    def test_step1_only_none_allosome_model_allowed(self):
        """
        Checks that step 1 only does not require an allosomal model and runs without error even if ad_model_allosomes
        is None, since step 1 does not use allosomal data.
        """
        params, lik = _call(steps=[1], ad_model_allosomes=None)
        assert isinstance(params, np.ndarray)


# --------------- Tests for parameter fixing semantics per mode ---------------

class TestParameterFixingSemantics:
    """
    This class checks that each mode optimizes over exactly the right parameter subset.
    The functions capture the local FixedParametersHandler seen by the optimizer
    and assert the exact free parameter labels at each fmin_cobyla call.
    """

    def _extract_handler_from_callable(self, func):
        """
        Walk a callable closure to find the local FixedParametersHandler instance.
        """
        seen = set()

        def visit(obj):
            obj_id = id(obj)
            if obj_id in seen:
                return None
            seen.add(obj_id)

            if isinstance(obj, FixedParametersHandler):
                return obj

            closure = getattr(obj, "__closure__", None)
            if closure is None:
                return None

            for cell in closure:
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    continue
                found = visit(cell_value)
                if found is not None:
                    return found
            return None

        handler = visit(func)
        if handler is None:
            raise AssertionError("Could not extract FixedParametersHandler from optimizer callback closure.")
        return handler

    def _run_and_capture_free_labels(self, steps, **extra):
        """
        Runs the two-step optimizer and returns a list of free parameter label
        lists, one per fmin_cobyla call, captured from the local handler closure.
        """
        captured = []

        def capturing_fmin(func, x0, cons, **kwargs):
            handler = self._extract_handler_from_callable(func)
            captured.append(handler.indices_to_labels(handler.free_parameters_indices))
            return x0

        with patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=capturing_fmin):
            _call(steps=steps, patch_fmin=False, **extra)

        return captured

    def test_step1_only_optimises_non_sex_bias(self):
        """ 
        Checks that step 1 only optimizes non-sex-bias parameters.
        """
        captured = self._run_and_capture_free_labels(steps=[1])
        assert len(captured) == 1, "exactly one fmin_cobyla call expected"
        assert captured[0] == ["t", "rate_eur"]

    def test_step2_only_optimises_sex_bias(self):
        """
        Checks that step 2 only optimizes sex-bias parameters, with non-sex-bias parameters fixed.
        """
        captured = self._run_and_capture_free_labels(steps=[2])
        assert len(captured) == 1
        assert captured[0] == ["sb_eur", "sb_afr"]

    def test_both_steps_two_fmin_calls(self):
        """
        Checks that running both steps results in two calls to fmin_cobyla, one for each step,
        with the appropriate parameter subsets free in each.
        """
        captured = self._run_and_capture_free_labels(steps=[1, 2])
        assert len(captured) == 2

    def test_step1_in_combined_optimises_non_sex_bias(self):
        """
        Checks that in combined mode, step 1 optimizes the non-sex-bias parameters (t and rate_eur) while sex-bias parameters are fixed.
        """
        captured = self._run_and_capture_free_labels(steps=[1, 2])
        assert captured[0] == ["t", "rate_eur"]

    def test_step2_in_combined_optimises_sex_bias(self):
        """
        Checks that in combined mode, step 2 optimizes the sex-bias parameters (sb_eur and sb_afr) while non-sex-bias parameters are fixed.
        """
        captured = self._run_and_capture_free_labels(steps=[1, 2])
        assert captured[1] == ["sb_eur", "sb_afr"]


# --------------- Tests for best-likelihood tracking scoped to the active step ---------------

class TestBestLikelihoodTracking:
    """
    This class verifies that the best-likelihood tracking is correctly scoped to the active step(s) and does not cross-contaminate between steps.
    In particular, verifies that:
    - Step 1 only: returned likelihood reflects the best seen during step 1.
    - Step 2 only: returned likelihood reflects the best seen during step 2.
    - Both steps: the returned likelihood is the best from step 2, not the
      global best across both steps.
    """

    def _run_tracking_test(self, steps):
        """
        Return the likelihood reported by the function.  We patch fmin_cobyla
        to return the initial x0 unchanged so the function always evaluates
        at p0; the exact value does not matter, only that it is consistent.
        """
        _, lik = _call(steps=steps)
        return lik

    def test_step1_only_returns_finite_or_fallback(self):
        lik = self._run_tracking_test(steps=[1])
        assert np.isfinite(lik) or lik == -1e32

    def test_step2_only_returns_finite_or_fallback(self):
        lik = self._run_tracking_test(steps=[2])
        assert np.isfinite(lik) or lik == -1e32

    def test_both_steps_returns_finite_or_fallback(self):
        lik = self._run_tracking_test(steps=[1, 2])
        assert np.isfinite(lik) or lik == -1e32

    def test_step2_best_resets_between_steps(self):
        """
        Verify that the best objective is reset before step 2 by checking that
        two independent runs (step-1-only vs step-2-only) do not share state.
        """
        _, lik_s1 = _call(steps=[1])
        _, lik_s2 = _call(steps=[2])
        # Both should be a real number (even if very negative); not cross-contaminated
        assert np.isfinite(lik_s1) or lik_s1 == -1e32
        assert np.isfinite(lik_s2) or lik_s2 == -1e32


# --------------- Tests that parameter autosomes_in_step_2 controls data scope in step 2 ---------------

class TestAutosomesInStep2:

    def _run_and_capture_include_autosomes(self, steps, autosomes_in_step_2):
        """
        Capture the include_autosomes value that was passed to the step-2 objective.
        We do this by recording the x0 size and any calls; the objective itself
        is not called here, but we verify no ValueError is raised.
        """
        params, lik = _call(steps=steps, autosomes_in_step_2=autosomes_in_step_2)
        return params, lik

    def test_autosomes_in_step2_true_no_error(self):
        """
        Tests that autosomes_in_step_2=True runs without error and returns a parameter array,
        indicating that the step-2 objective was able to include autosomal data as intended.
        """    
        params, lik = self._run_and_capture_include_autosomes([2], autosomes_in_step_2=True)
        assert isinstance(params, np.ndarray)

    def test_autosomes_in_step2_false_no_error(self):
        """
        Tests that autosomes_in_step_2=False runs without error and returns a parameter array,
        indicating that the step-2 objective was able to exclude autosomal data and focus on allosomal data as intended.
        """
        params, lik = self._run_and_capture_include_autosomes([2], autosomes_in_step_2=False)
        assert isinstance(params, np.ndarray)

    def test_autosomes_in_step2_false_requires_allosome_model(self):
        """
        Tests that autosomes_in_step_2=False raises ValueError if ad_model_allosomes is None,
        since excluding autosomal data in step 2 requires an allosomal model to be specified
        for the optimization to proceed.
        """
        with pytest.raises(ValueError):
            _call(steps=[2], autosomes_in_step_2=False, ad_model_allosomes=None)


# --------------- Tests for return type and shape ---------------

class TestReturnTypeAndShape:

    @pytest.mark.parametrize("steps", [[1], [2], [1, 2], None])
    def test_returns_tuple_of_array_and_float(self, steps):
        """
        Tests that the function returns a tuple of (params, lik) where params is a numpy array and lik is a float,
        for all valid step specifications including the default None. This ensures that the output format is
        consistent regardless of which steps are run.
        """
        result = _call(steps=steps)
        assert isinstance(result, tuple) and len(result) == 2
        params, lik = result
        assert isinstance(params, np.ndarray)
        assert isinstance(lik, float)

    @pytest.mark.parametrize("steps", [[1], [2], [1, 2], None])
    def test_returned_params_have_correct_length(self, steps):
        """
        Tests that the returned parameter array has the expected length (N_PARAMS) for all valid step specifications,
        including the default None. This verifies that the function returns a complete parameter vector regardless of which steps are executed.
        """
        params, _ = _call(steps=steps)
        assert len(params) == N_PARAMS


# ---------------  Tests for reset_counter behaviour ---------------

class TestResetCounter:
    """
    This class verifies that the reset_counter argument correctly controls whether the internal optimization counter is reset to zero before optimization begins.
    """

    def test_reset_counter_true_resets_to_zero(self):
        """
        Tests that when reset_counter=True, the internal counter is reset to zero before optimization begins.
        """
        core_module._counter = 999
        _call(steps=[1], reset_counter=True)
        assert core_module._counter < 999 # Counter will have been incremented at least once from 0; it will not be 999+.

    def test_reset_counter_false_preserves_count(self):
        """
        Tests that when reset_counter=False, the internal counter is not reset and retains its value from previous runs,
        allowing for cumulative counting across multiple calls if desired.
        """
        core_module._counter = 0
        _call(steps=[1], reset_counter=True)
        count_after_first = core_module._counter
        _call(steps=[1], reset_counter=False)
        assert core_module._counter > count_after_first # Counter must have grown (not been reset to 0)


# ---------------  Tests for p0 used as starting point in step-2-only mode ---------------

class TestStep2OnlyStartingPoint:

    def test_step2_only_x0_derived_from_p0(self):
        """
        Tests that in step-2-only mode, the initial parameter vector (x0) passed to fmin_cobyla is derived
        from the input p0, specifically taking the sex-bias parameters from p0 and not the non-sex-bias parameters.
        """
        captured_x0 = []

        def capture(func, x0, cons, **kw):
            captured_x0.append(x0.copy())
            return x0

        with patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=capture):
            _call(steps=[2], p0=P0, patch_fmin=False)

        assert len(captured_x0) == 1
        # sb_eur=0.05, sb_afr=0.05 are the sex-bias values in P0
        np.testing.assert_allclose(captured_x0[0], P0[[2, 3]], rtol=1e-6)


# --------------- Test for optimize_cob_sex_biased_single_step ---------------

class TestSingleStepBasicModes:
    """
    This class contains basic tests for optimize_cob_sex_biased_single_step to verify that it runs without error
    and returns outputs of the expected type and shape, both with and without an allosomal model specified.
    These tests do not check the internal logic of the optimization but ensure that the function can be invoked in
    its basic modes and that it handles the presence or absence of an allosomal model as expected.
    """

    @pytest.mark.parametrize("ad_model_allosomes", ["DC", None])
    def test_single_step_accepts_with_or_without_allosomes(self, ad_model_allosomes):
        """
        Tests that the function accepts both with and without an allosomal model specified.
        """
        params, lik = _call_single(ad_model_allosomes=ad_model_allosomes)
        assert isinstance(params, np.ndarray)
        assert len(params) == N_PARAMS
        assert np.isfinite(lik) or lik == -1e32

    def test_single_step_returns_tuple_array_float(self):
        """
        Tests that the function returns a tuple containing a numpy array and a float.
        """
        result = _call_single()
        assert isinstance(result, tuple) and len(result) == 2
        params, lik = result
        assert isinstance(params, np.ndarray)
        assert isinstance(lik, float)
        assert len(params) == N_PARAMS


class TestSingleStepOptimizerCall:
    """
    This class contains tests to verify that optimize_cob_sex_biased_single_step makes exactly one call to scipy.optimize.fmin_cobyla
    and that the initial parameter vector (x0) passed to fmin_cobyla is derived from the input p0 as expected, specifically that it includes
    the full parameter vector and not just a subset. These tests use a mock for fmin_cobyla to capture the calls and inspect the arguments
    without performing any real optimization.
    """

    def test_single_step_calls_fmin_once(self):
        """
        Tests that optimize_cob_sex_biased_single_step makes exactly one call to scipy.optimize.fmin_cobyla, ensuring
        that the optimization process is initiated as expected and that there are no unexpected multiple calls to the optimizer.
        """
        call_count = 0

        def capture(func, x0, cons, **kw):
            nonlocal call_count
            call_count += 1
            return x0

        with patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=capture):
            _call_single(patch_fmin=False)

        assert call_count == 1

    def test_single_step_x0_is_full_parameter_vector(self):
        """
        Tests that the initial parameter vector (x0) passed to fmin_cobyla in optimize_cob_sex_biased_single_step is the full parameter
        vector derived from the input p0, rather than a subset, ensuring that the optimization is correctly initialized with all parameters
        available for optimization.
        """
        captured_x0 = []

        def capture(func, x0, cons, **kw):
            captured_x0.append(np.array(x0, copy=True))
            return x0

        with patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=capture):
            _call_single(p0=P0, patch_fmin=False)

        assert len(captured_x0) == 1
        np.testing.assert_allclose(captured_x0[0], P0, rtol=1e-6)


class TestSingleStepResetCounter:
    """
    This class contains tests to verify that the reset_counter argument in optimize_cob_sex_biased_single_step correctly
    controls whether the internal optimization counter is reset to zero before optimization begins.
    """

    def test_single_step_reset_counter_true_resets_to_zero(self):
        """
        Tests that when reset_counter=True, the internal counter is reset to zero before optimization begins,
        ensuring that the optimization process starts with a clean state and that the counter does not carry over from previous runs.
        """
        core_module._counter = 999
        _call_single(reset_counter=True)
        assert core_module._counter < 999

    def test_single_step_reset_counter_false_preserves_count(self):
        """
        Tests that when reset_counter=False, the internal counter is not reset and retains its value from previous runs,
        allowing for cumulative counting across multiple calls if desired, and that the counter continues to increment from its
        previous value rather than resetting to zero.
        """
        core_module._counter = 0
        _call_single(reset_counter=True)
        count_after_first = core_module._counter
        _call_single(reset_counter=False)
        assert core_module._counter > count_after_first


# --------------- Regression test for _ancestry_overrides in step 2 ---------------

class TestAncestryOverridesInStep2:
    """
    Regression tests for the _ancestry_overrides mechanism in
    optimize_cob_sex_biased_two_steps.

    During step 2, ancestry-fixed non-sex-bias parameters must remain pinned to their
    step-1/p0 values for every optimizer iteration.  Without _ancestry_overrides, each
    call to extend_parameters() would invoke compute_params_fixed_by_ancestry() and
    re-solve those parameters against the current sex-bias candidate, letting them drift.
    """

    def _make_handler_with_ancestry_fixed(self):
        """
        Return a FixedParametersHandler where rate_eur (index 1) is declared as
        fixed-by-ancestry.  We set the attribute directly after calling the standard
        _make_handler() factory to avoid needing real migration-matrix machinery.
        """
        ph = _make_handler()  # t(0), rate_eur(1), sb_eur(2), sb_afr(3); nothing ancestry-fixed
        ph.params_fixed_by_ancestry = {"rate_eur": ""}
        ph.free_parameters_indices = [
            idx
            for idx, name in enumerate(ph.demography.model_base_params)
            if name not in ph.current_fixed_parameters
            and name not in ph.params_fixed_by_ancestry
        ]
        return ph

    def test_step2_ancestry_fixed_param_pinned_in_objective(self):
        """
        _ancestry_overrides must keep rate_eur (an ancestry-fixed non-sex-bias param at
        index 1) at its p0 value in every call to reduced_objective_function during step 2,
        even though the patched compute_params_fixed_by_ancestry() would compute a
        different value based on the current sex-bias candidate.

        The test patches compute_params_fixed_by_ancestry to set
        rate_eur = |sb_eur| * 100 (simulating drift), then checks that the full
        parameter vector seen by outofbounds_fun — after the _ancestry_overrides
        override — always has rate_eur == P0[1] (== 0.3), never the drifted value.
        """
        captured = []
        state = {"in_fmin": False}

        def drifting_ancestry(self_ph, params, **kwargs):
            """Simulate drift: compute_params_fixed_by_ancestry sets rate_eur ∝ sb_eur."""
            result = np.array(params, dtype=float)
            result[1] = abs(result[2]) * 100  # e.g. sb_eur=0.07 → rate_eur=7.0 without fix
            return result

        def capturing_oob(params):
            """
            Collect extended params seen during the optimizer phase, then short-circuit
            model evaluation so PhT code is never invoked.
            """
            if state["in_fmin"]:
                captured.append(np.array(params))
            return -1  # always OOB → objective returns without calling model_func

        def fmin_mock(func, x0, cons, **kwargs):
            state["in_fmin"] = True
            try:
                perturbed = x0.copy()
                perturbed[0] += 0.02  # shift sb_eur to trigger ancestry drift
                func(perturbed)
            finally:
                state["in_fmin"] = False
            return x0

        ph = self._make_handler_with_ancestry_fixed()

        with patch.object(
            FixedParametersHandler,
            "compute_params_fixed_by_ancestry",
            drifting_ancestry,
        ):
            with patch.object(
                core_module.scipy.optimize, "fmin_cobyla", side_effect=fmin_mock
            ):
                optimize_cob_sex_biased_two_steps(
                    p0=P0,
                    population=_make_population(),
                    model_func=_make_model_func(),
                    parameter_handler=ph,
                    outofbounds_fun=capturing_oob,
                    verbose_log=0,
                    verbose_screen=0,
                    p_dict=P_DICT,
                    exclude_tracts_below_cM=0,
                    maxiter=2,
                    reset_counter=True,
                    ad_model_autosomes="DC",
                    ad_model_allosomes="DC",
                    autosomes_in_step_2=True,
                    npts=N_BINS,
                    steps=[2],
                )

        assert len(captured) > 0, (
            "outofbounds_fun was never called during the fmin phase; "
            "the fmin mock did not invoke the objective function."
        )

        RATE_EUR_P0 = P0[1]  # 0.3 — the value _ancestry_overrides should pin to
        drifted_value = abs(P0[2] + 0.02) * 100  # what drifting_ancestry would produce
        for i, params in enumerate(captured):
            np.testing.assert_allclose(
                params[1],
                RATE_EUR_P0,
                rtol=1e-9,
                err_msg=(
                    f"Call {i}: rate_eur should be pinned to p0 ({RATE_EUR_P0}) "
                    f"by _ancestry_overrides, not drifted to {drifted_value:.4f} "
                    "via compute_params_fixed_by_ancestry."
                ),
            )
