"""
Tests of when the Monoecious (M) autosomal model ignores sex-bias parameters, and of the
consequences for the two-step optimization and the sex-bias re-optimizations:

- For founder events and integer-time pulses, ``GeneticModel.loglik`` with
  ``ad_model_autosomes='M'`` gives the same autosomal log-likelihood for any sex-bias values (and
  the Dioecious-Coarse model, as a control, does not). For pulses at fractional times it does not:
  ``PulseEvent.execute`` is nonlinear in the sex-specific rates, so the averaged female/male
  migration matrix seen by M depends on sex bias.
- In ``core.optimize_cob_sex_biased_two_steps``, for a founder-only model, step 1 under M follows
  exactly the same path whatever values the (fixed) sex-bias parameters take, so that a
  re-optimization round is equivalent to restarting step 1 from its previous optimum: any
  improvement of the autosomal log-likelihood across rounds comes from restarting COBYLA, not from
  the updated sex bias. With a fractional-time pulse, the carried-over sex bias does change step 1.
- ``core_utils.check_step_2_kept_non_sex_bias_parameters`` detects step 2 changing a
  non-sex-bias parameter, and gives no false warning for fractional-time pulses.
- ``core_utils.check_monoecious_autosomal_loglik_unchanged`` compares the M autosomal
  log-likelihood before and after step 2 only where M is guaranteed to ignore sex bias
  (``core_utils._monoecious_sex_bias_invariance_violation``), and skips it otherwise.
- ``core._run_cobyla`` reports why COBYLA stopped.

The demographic models are real, in-memory ``ParametrizedDemographySexBiased`` instances: one
founder event (EUR, AMR explicit; AFR implicit), as in ``tests/test_reoptimization.py``, and a
founder event followed by an AFR pulse. The data are small synthetic tract-length histograms, so
that real likelihoods are computed quickly.
"""
import logging
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import scipy.optimize

import tracts.core as core_module
from tracts.core import optimize_cob_sex_biased_two_steps, _run_cobyla
from tracts.core_utils import (
    check_step_2_kept_non_sex_bias_parameters,
    check_monoecious_autosomal_loglik_unchanged,
    _monoecious_sex_bias_invariance_violation,
    _compute_autosomal_loglik,
    _build_cobyla_stop_message,
)
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased, SexType
from tracts.genetic_model import GeneticModel
from tracts.likelihood_options import LikelihoodOptions
from tracts.tracts_data import TractsData


# Parameter order of the model below: REUR, REUR_sex_bias, RAMR, RAMR_sex_bias, t.
NON_SEX_BIAS_INDICES = [0, 2, 4]
SEX_BIAS_INDICES = [1, 3]
SEX_BIAS_VALUES = [(0.0, 0.0), (0.5, -0.3), (-1.0, 1.0), (0.9, 0.9), (-0.25, -0.75)]


# --------------- Shared fixtures ---------------

def _make_model():
    model = ParametrizedDemographySexBiased(name="ThreePop")
    model.add_founder_event(
        dest_population="X",
        source_populations={"EUR": "REUR", "AMR": "RAMR"},
        remainder_population="AFR",
        found_time="t",
    )
    model.finalize()
    model.set_up_fixed_parameters(params_to_fix_by_ancestry=[], proportions={}, user_params_to_fix_by_value={})
    return model


def _make_pulse_model():
    """
    A founder event (EUR explicit, AMR implicit) at ``t1`` followed by an AFR pulse at ``t2``.
    Parameter order: REUR, REUR_sex_bias, t1, RAFR, RAFR_sex_bias, t2.
    """
    model = ParametrizedDemographySexBiased(name="FounderPlusPulse")
    model.add_founder_event(
        dest_population="X",
        source_populations={"EUR": "REUR"},
        remainder_population="AMR",
        found_time="t1",
    )
    model.add_pulse_migration(dest_population="X", source_population="AFR", rate_param="RAFR", time_param="t2")
    model.finalize()
    model.set_up_fixed_parameters(params_to_fix_by_ancestry=[], proportions={}, user_params_to_fix_by_value={})
    return model


PULSE_NON_SEX_BIAS_INDICES = [0, 2, 3, 5]
PULSE_SEX_BIAS_INDICES = [1, 4]


def _pulse_physical_params(sex_bias, t2, reur=0.5, t1=10.0, rafr=0.3):
    return np.array([reur, sex_bias[0], t1, rafr, sex_bias[1], t2])


def _make_population(population_indices, npts=10, seed=0):
    """
    A Population stand-in exposing the methods and attributes used by ``TractsData.from_population``,
    with random (but fixed) autosomal and allosomal tract-length histograms.
    """
    rng = np.random.default_rng(seed)
    bins = np.linspace(0, 3, npts)
    autosome_counts = {label: rng.integers(1, 50, npts).tolist() for label in population_indices}
    female_counts = {label: rng.integers(1, 20, npts).tolist() for label in population_indices}
    male_counts = {label: rng.integers(1, 20, npts).tolist() for label in population_indices}
    return SimpleNamespace(
        Ls=[1.5, 2.0],
        indivs=[object()] * 10,
        num_males=5,
        num_females=5,
        allosome_lengths={"X": 1.8},
        get_global_tractlengths=lambda npts, exclude_tracts_below_cM=0: (bins, autosome_counts),
        get_global_allosome_tractlengths=lambda allosome, npts, exclude_tracts_below_cM=0: (
            bins, {SexType.FEMALE: female_counts, SexType.MALE: male_counts}
        ),
    )


def _physical_params(sex_bias, reur=0.3, ramr=0.2, t=6.0):
    return np.array([reur, sex_bias[0], ramr, sex_bias[1], t])


@pytest.fixture
def setup():
    model = _make_model()
    population = _make_population(model.population_indices)
    tracts_data = TractsData.from_population(population=population, p_dict=model.population_indices, npts=10)
    return model, population, tracts_data


@pytest.fixture
def core_utils_caplog(caplog):
    """
    ``caplog`` capturing INFO records of ``tracts.core_utils``. The handler is attached to that logger
    directly because ``tracts.logs.initialize_tracts`` (run by driver tests) sets ``propagate = False``
    on the ``tracts`` logger, so records would otherwise never reach caplog's root handler.
    """
    core_utils_logger = logging.getLogger("tracts.core_utils")
    previous_level = core_utils_logger.level
    core_utils_logger.addHandler(caplog.handler)
    core_utils_logger.setLevel(logging.INFO)
    try:
        yield caplog
    finally:
        core_utils_logger.removeHandler(caplog.handler)
        core_utils_logger.setLevel(previous_level)


@pytest.fixture
def pulse_setup():
    model = _make_pulse_model()
    population = _make_population(model.population_indices)
    tracts_data = TractsData.from_population(population=population, p_dict=model.population_indices, npts=10)
    return model, population, tracts_data


def _genetic_model(model, ad_model_autosomes):
    return GeneticModel(model, ad_model_autosomes=ad_model_autosomes, ad_model_allosomes="DC")


def _to_opt(model, physical_params):
    return model.parameter_handler.convert_to_optimizer_params(np.array(physical_params))


# --------------- A. The M autosomal model ignores sex bias ---------------

class TestMonoeciousAutosomalLoglikIgnoresSexBias:

    def test_m_autosomal_loglik_is_identical_for_any_sex_bias(self, setup):
        model, _, tracts_data = setup
        genetic_model = _genetic_model(model, "M")
        logliks = [_compute_autosomal_loglik(_physical_params(sex_bias), genetic_model, tracts_data)
                   for sex_bias in SEX_BIAS_VALUES]
        assert np.all(np.isfinite(logliks))
        np.testing.assert_allclose(logliks, logliks[0], rtol=1e-12)

    def test_average_of_female_and_male_matrices_does_not_depend_on_sex_bias(self, setup):
        model, _, _ = setup
        unbiased = model.get_migration_matrices(_physical_params((0.0, 0.0)))
        for sex_bias in SEX_BIAS_VALUES[1:]:
            matrices = model.get_migration_matrices(_physical_params(sex_bias))
            np.testing.assert_allclose((matrices["X_female"] + matrices["X_male"]) / 2, unbiased["X_female"],
                                       rtol=1e-12, atol=1e-15)

    def test_dc_autosomal_loglik_depends_on_sex_bias(self, setup):
        # Control: without it, the M test above could pass because the sex bias never reaches the
        # autosomal likelihood at all (e.g. if the matrices were averaged before being passed in).
        model, _, tracts_data = setup
        genetic_model = _genetic_model(model, "DC")
        unbiased = _compute_autosomal_loglik(_physical_params((0.0, 0.0)), genetic_model, tracts_data)
        biased = _compute_autosomal_loglik(_physical_params((0.5, -0.3)), genetic_model, tracts_data)
        assert not np.isclose(unbiased, biased, rtol=1e-6)


class TestMonoeciousAutosomalLoglikWithPulses:
    """
    ``PulseEvent.execute`` spreads a pulse at fractional time t over generations floor(t) and
    floor(t)+1, the latter with rate a*(t-floor(t))/(1-a*(floor(t)+1-t)), nonlinear in the rate a.
    Applied separately to the female and male rates, its average over sexes depends on sex bias.
    """

    SEX_BIASES = [(0.0, 0.0), (0.0, 0.5), (0.3, -0.8)]

    def _m_logliks(self, model, tracts_data, t2):
        genetic_model = _genetic_model(model, "M")
        return [_compute_autosomal_loglik(_pulse_physical_params(sex_bias, t2), genetic_model, tracts_data)
                for sex_bias in self.SEX_BIASES]

    def test_m_autosomal_loglik_ignores_sex_bias_for_integer_time_pulse(self, pulse_setup):
        model, _, tracts_data = pulse_setup
        logliks = self._m_logliks(model, tracts_data, t2=4.0)
        np.testing.assert_allclose(logliks, logliks[0], rtol=1e-12)

    def test_m_autosomal_loglik_depends_on_sex_bias_for_fractional_time_pulse(self, pulse_setup):
        model, _, tracts_data = pulse_setup
        logliks = self._m_logliks(model, tracts_data, t2=4.5)
        assert len(set(np.round(logliks, 6))) == len(self.SEX_BIASES)

    def test_averaged_matrix_depends_on_sex_bias_only_in_the_second_pulse_generation(self, pulse_setup):
        model, _, _ = pulse_setup
        afr = model.population_indices["AFR"]
        averages = []
        for sex_bias in self.SEX_BIASES:
            matrices = model.get_migration_matrices(_pulse_physical_params(sex_bias, t2=4.5))
            averages.append((matrices["X_female"] + matrices["X_male"]) / 2)
        for average in averages[1:]:
            np.testing.assert_allclose(average[4, afr], averages[0][4, afr], rtol=1e-12)
            assert not np.isclose(average[5, afr], averages[0][5, afr], rtol=1e-6)


# --------------- B. Step 1 under M, and re-optimization rounds ---------------

def _run_step(model, population, steps, p0_opt, maxiter, ad_model_autosomes="M"):
    """
    Runs ``optimize_cob_sex_biased_two_steps`` for ``steps`` with the real COBYLA, returning the
    optimal parameters (optimizer space), the optimal likelihood, and the list of objective values
    evaluated by COBYLA, in order.
    """
    evaluated_objectives = []
    real_minimize = core_module.scipy.optimize.minimize

    def recording_minimize(fun, x0, **kwargs):
        def recording_fun(x):
            value = fun(x)
            evaluated_objectives.append(value)
            return value
        return real_minimize(fun=recording_fun, x0=x0, **kwargs)

    with patch.object(core_module.scipy.optimize, "minimize", side_effect=recording_minimize):
        params, likelihood = optimize_cob_sex_biased_two_steps(
            p0=p0_opt,
            population=population,
            genetic_model=_genetic_model(model, ad_model_autosomes),
            likelihood_options=LikelihoodOptions(verbose_log=0, verbose_screen=0),
            p_dict=model.population_indices.items(),
            maxiter=maxiter,
            autosomes_in_step_2=False,
            npts=10,
            steps=steps,
        )
    return np.array(params), likelihood, evaluated_objectives


class TestStepOneUnderMonoeciousIgnoresSexBias:

    def test_step1_path_is_identical_for_different_fixed_sex_bias(self, setup):
        model, population, _ = setup
        runs = [_run_step(model, population, steps=[1], maxiter=60,
                          p0_opt=_to_opt(model, _physical_params(sex_bias, reur=0.25, ramr=0.25, t=5.0)))
                for sex_bias in [(0.0, 0.0), (0.6, -0.4)]]
        (params_a, likelihood_a, objectives_a), (params_b, likelihood_b, objectives_b) = runs

        assert len(objectives_a) == len(objectives_b)
        np.testing.assert_allclose(objectives_a, objectives_b, rtol=1e-10)
        np.testing.assert_allclose(params_a[NON_SEX_BIAS_INDICES], params_b[NON_SEX_BIAS_INDICES], rtol=1e-10)
        assert likelihood_a == pytest.approx(likelihood_b, rel=1e-10)

    def test_reoptimization_round_equals_plain_restart_of_step1(self, setup):
        model, population, _ = setup
        p0_opt = _to_opt(model, _physical_params((0.0, 0.0), reur=0.25, ramr=0.25, t=5.0))

        # Initial optimization: step 1, then step 2 (which only changes the sex-bias parameters).
        step1_params, step1_likelihood, _ = _run_step(model, population, steps=[1], p0_opt=p0_opt, maxiter=40)
        step2_params, _, _ = _run_step(model, population, steps=[2], p0_opt=step1_params, maxiter=15)
        np.testing.assert_allclose(step2_params[NON_SEX_BIAS_INDICES], step1_params[NON_SEX_BIAS_INDICES])
        assert not np.allclose(step2_params[SEX_BIAS_INDICES], step1_params[SEX_BIAS_INDICES])

        # Re-optimization round: step 1 restarts from the step-2 optimum (sex bias fixed at its step-2 values),
        # compared with simply restarting step 1 from its own optimum (no step 2 in between).
        reopt_params, reopt_likelihood, reopt_objectives = _run_step(model, population, steps=[1],
                                                                     p0_opt=step2_params, maxiter=40)
        restart_params, restart_likelihood, restart_objectives = _run_step(model, population, steps=[1],
                                                                           p0_opt=step1_params, maxiter=40)

        np.testing.assert_allclose(reopt_objectives, restart_objectives, rtol=1e-10)
        np.testing.assert_allclose(reopt_params[NON_SEX_BIAS_INDICES], restart_params[NON_SEX_BIAS_INDICES], rtol=1e-10)
        assert reopt_likelihood == pytest.approx(restart_likelihood, rel=1e-10)

        # The round starts exactly where the previous step 1 ended, and cannot end lower.
        assert -reopt_objectives[0] == pytest.approx(step1_likelihood, rel=1e-10)
        assert reopt_likelihood >= step1_likelihood - 1e-9

    def test_autosomal_loglik_never_decreases_across_rounds(self, setup):
        model, population, _ = setup
        params = _to_opt(model, _physical_params((0.0, 0.0), reur=0.25, ramr=0.25, t=5.0))
        previous_likelihood = None
        for _round in range(3):
            params, likelihood, objectives = _run_step(model, population, steps=[1], p0_opt=params, maxiter=25)
            if previous_likelihood is not None:
                assert -objectives[0] == pytest.approx(previous_likelihood, rel=1e-10)
                assert likelihood >= previous_likelihood - 1e-9
            previous_likelihood = likelihood
            params, _, _ = _run_step(model, population, steps=[2], p0_opt=params, maxiter=8)

    def test_step1_depends_on_fixed_sex_bias_for_fractional_time_pulse(self, pulse_setup):
        # With a fractional-time pulse, the sex-bias values carried into step 1 (at which they are
        # fixed) do change the step-1 objective, so a re-optimization round is not a plain restart.
        model, population, _ = pulse_setup
        runs = [_run_step(model, population, steps=[1], maxiter=10,
                          p0_opt=_to_opt(model, _pulse_physical_params(sex_bias, t2=4.5)))
                for sex_bias in [(0.0, 0.0), (0.0, 0.6)]]
        (_, _, objectives_a), (_, _, objectives_b) = runs
        assert not np.isclose(objectives_a[0], objectives_b[0], rtol=1e-8)


# --------------- C. Runtime check that step 2 only changes sex-bias parameters ---------------

class TestCheckStep2KeptNonSexBiasParameters:

    def _check(self, model, step_1_params, final_params):
        return check_step_2_kept_non_sex_bias_parameters(model_base_params=model.model_base_params,
                                                         step_1_params=step_1_params,
                                                         final_params=final_params)

    def test_passes_when_only_sex_bias_changed(self, setup, core_utils_caplog, capsys):
        model, _, _ = setup
        result = self._check(model, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5)))

        assert result is True
        assert "WARNING" not in capsys.readouterr().out
        assert not [record for record in core_utils_caplog.records if record.levelno >= logging.WARNING]
        assert any("Check passed" in record.getMessage() for record in core_utils_caplog.records)

    def test_passes_within_unit_conversion_round_off(self, setup, core_utils_caplog, capsys):
        model, _, _ = setup
        step_1_params = _physical_params((0.0, 0.0), t=6.37)
        round_tripped = model.parameter_handler.convert_to_physical_params(_to_opt(model, step_1_params))
        assert self._check(model, step_1_params, round_tripped) is True

    def test_warns_and_names_the_changed_non_sex_bias_parameters(self, setup, core_utils_caplog, capsys):
        model, _, _ = setup
        result = self._check(model, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5), reur=0.4, t=7.0))

        assert result is False
        out = capsys.readouterr().out
        assert "Step 2 changed non-sex-bias parameters" in out
        assert "REUR: 0.3 -> 0.4" in out
        assert "t: 6 -> 7" in out
        assert "RAMR:" not in out and "sex_bias:" not in out
        assert any(record.levelno == logging.WARNING for record in core_utils_caplog.records)

    @pytest.mark.parametrize("ad_model_autosomes", ["M", "DC"])
    def test_two_steps_optimization_runs_the_check_and_it_passes(self, setup, core_utils_caplog, capsys,
                                                                 ad_model_autosomes):
        model, population, _ = setup
        p0_opt = _to_opt(model, _physical_params((0.0, 0.0), reur=0.25, ramr=0.25, t=5.0))
        results = []

        def recording_check(**kwargs):
            results.append(check_step_2_kept_non_sex_bias_parameters(**kwargs))
            return results[-1]

        with patch.object(core_module, "check_step_2_kept_non_sex_bias_parameters", side_effect=recording_check):
            _run_step(model, population, steps=None, p0_opt=p0_opt, maxiter=6, ad_model_autosomes=ad_model_autosomes)

        assert results == [True]
        assert "Step 2 changed non-sex-bias parameters" not in capsys.readouterr().out

    def test_no_false_warning_for_fractional_time_pulse(self, pulse_setup, core_utils_caplog, capsys):
        # Regression test: comparing autosomal log-likelihoods before and after step 2 would warn
        # here, because under M the autosomal log-likelihood depends on sex bias for this pulse.
        model, population, tracts_data = pulse_setup
        p0_opt = _to_opt(model, _pulse_physical_params((0.0, 0.0), t2=4.5))

        step1_params, _, _ = _run_step(model, population, steps=[1], p0_opt=p0_opt, maxiter=10)
        step2_params, _, _ = _run_step(model, population, steps=[2], p0_opt=step1_params, maxiter=10)

        assert not np.isclose(model.parameter_handler.convert_to_physical_params(step2_params)[5] % 1, 0)
        genetic_model = _genetic_model(model, "M")
        to_physical = model.parameter_handler.convert_to_physical_params
        assert not np.isclose(_compute_autosomal_loglik(to_physical(step1_params), genetic_model, tracts_data),
                              _compute_autosomal_loglik(to_physical(step2_params), genetic_model, tracts_data), rtol=1e-8)

        out = capsys.readouterr().out
        assert "Step 2 changed non-sex-bias parameters" not in out
        assert any("Check passed" in record.getMessage() for record in core_utils_caplog.records)


# --------------- C2. Runtime check of the M autosomal likelihood, where M ignores sex bias ---------------

def _make_continuous_founder_model():
    model = ParametrizedDemographySexBiased(name="ContinuousFounder")
    model.add_founder_event(dest_population="X", source_populations={"EUR": "REUR", "AMR": "RAMR"},
                            remainder_population=None, found_time="t", end_time="tend")
    model.finalize()
    model.set_up_fixed_parameters(params_to_fix_by_ancestry=[], proportions={}, user_params_to_fix_by_value={})
    return model


class TestMonoeciousSexBiasInvarianceViolation:

    def test_founder_only_model_is_invariant(self, setup):
        model, _, _ = setup
        params = [_physical_params(sex_bias) for sex_bias in SEX_BIAS_VALUES]
        assert _monoecious_sex_bias_invariance_violation(model, params) is None

    def test_integer_time_pulse_is_invariant(self, pulse_setup):
        model, _, _ = pulse_setup
        params = [_pulse_physical_params((0.0, 0.0), t2=4.0), _pulse_physical_params((0.3, -0.8), t2=4.0)]
        assert _monoecious_sex_bias_invariance_violation(model, params) is None

    def test_fractional_time_pulse_is_not_invariant(self, pulse_setup):
        model, _, _ = pulse_setup
        violation = _monoecious_sex_bias_invariance_violation(model, [_pulse_physical_params((0.0, 0.0), t2=4.5)])
        assert "fractional time" in violation and "t2" in violation

    def test_continuous_founder_event_is_not_invariant(self):
        model = _make_continuous_founder_model()
        violation = _monoecious_sex_bias_invariance_violation(model, [np.zeros(len(model.model_base_params))])
        assert "continuous founder event" in violation

    def test_clipped_remainder_rate_is_not_invariant(self, setup):
        # Female founding rates summing to 1 + 5e-5: the remainder rate is clipped at 0, which makes the
        # averaged matrix depend on sex bias.
        model, _, _ = setup
        unbiased = np.array([0.5, 0.0, 0.5, 0.0, 6.0])
        clipped = np.array([0.5, 1e-4, 0.5, 0.0, 6.0])
        assert _monoecious_sex_bias_invariance_violation(model, [unbiased]) is None
        assert "clipped" in _monoecious_sex_bias_invariance_violation(model, [unbiased, clipped])
        averages = [(m["X_female"] + m["X_male"]) / 2 for m in map(model.get_migration_matrices, [unbiased, clipped])]
        assert not np.allclose(averages[0], averages[1], rtol=0, atol=1e-6)


class TestCheckMonoeciousAutosomalLoglikUnchanged:

    def _check(self, model, tracts_data, step_1_params, final_params, ad_model_autosomes="M",
               step_1_autosomal_loglik=None):
        genetic_model = _genetic_model(model, ad_model_autosomes)
        if step_1_autosomal_loglik is None:
            step_1_autosomal_loglik = _compute_autosomal_loglik(step_1_params, genetic_model, tracts_data)
        return check_monoecious_autosomal_loglik_unchanged(genetic_model=genetic_model,
                                                           tracts_data=tracts_data,
                                                           step_1_autosomal_loglik=step_1_autosomal_loglik,
                                                           step_1_params=step_1_params,
                                                           final_params=final_params)

    def _warnings(self, core_utils_caplog):
        return [record.getMessage() for record in core_utils_caplog.records if record.levelno >= logging.WARNING]

    def test_passes_when_only_sex_bias_changed(self, setup, core_utils_caplog, capsys):
        model, _, tracts_data = setup
        result = self._check(model, tracts_data, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5)))
        assert result is True
        assert "WARNING" not in capsys.readouterr().out
        assert not self._warnings(core_utils_caplog)

    def test_passes_for_integer_time_pulse(self, pulse_setup, core_utils_caplog, capsys):
        model, _, tracts_data = pulse_setup
        result = self._check(model, tracts_data, _pulse_physical_params((0.0, 0.0), t2=4.0),
                             _pulse_physical_params((0.3, -0.8), t2=4.0))
        assert result is True

    def test_warns_when_the_autosomal_loglik_changed(self, setup, core_utils_caplog, capsys):
        model, _, tracts_data = setup
        result = self._check(model, tracts_data, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5), reur=0.4))
        assert result is False
        assert "The autosomal log-likelihood changed" in capsys.readouterr().out
        assert self._warnings(core_utils_caplog)

    def test_returns_none_for_non_monoecious_autosomal_model(self, setup):
        model, _, tracts_data = setup
        result = self._check(model, tracts_data, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5)),
                             ad_model_autosomes="DC", step_1_autosomal_loglik=-1.0)
        assert result is None

    def test_skipped_without_warning_for_fractional_time_pulse(self, pulse_setup, core_utils_caplog, capsys):
        # The M autosomal log-likelihood does change with sex bias here, so comparing it would be a
        # false alarm: the check must be skipped (logged at INFO level), not warn.
        model, _, tracts_data = pulse_setup
        step_1_params = _pulse_physical_params((0.0, 0.0), t2=4.5)
        final_params = _pulse_physical_params((0.3, -0.8), t2=4.5)
        genetic_model = _genetic_model(model, "M")
        assert not np.isclose(_compute_autosomal_loglik(step_1_params, genetic_model, tracts_data),
                              _compute_autosomal_loglik(final_params, genetic_model, tracts_data), rtol=1e-8)

        assert self._check(model, tracts_data, step_1_params, final_params) is None
        assert "WARNING" not in capsys.readouterr().out
        assert not self._warnings(core_utils_caplog)
        assert any("fractional time" in record.getMessage() for record in core_utils_caplog.records)

    def test_skips_when_step_1_loglik_is_not_finite(self, setup, core_utils_caplog):
        model, _, tracts_data = setup
        result = self._check(model, tracts_data, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5)),
                             step_1_autosomal_loglik=-np.inf)
        assert result is None
        assert any("step-1 log-likelihood is not finite" in message for message in self._warnings(core_utils_caplog))

    def test_skips_when_final_params_out_of_bounds(self, setup, core_utils_caplog):
        model, _, tracts_data = setup
        with patch.object(ParametrizedDemographySexBiased, "get_violation_score", return_value=-1.0):
            result = self._check(model, tracts_data, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5)),
                                 step_1_autosomal_loglik=-100.0)
        assert result is None
        assert any("out of bounds" in message for message in self._warnings(core_utils_caplog))

    @pytest.mark.parametrize("error", [np.linalg.LinAlgError("singular"), ValueError("bad matrix"),
                                       FloatingPointError("overflow")])
    def test_skips_when_final_loglik_cannot_be_evaluated(self, setup, core_utils_caplog, error):
        model, _, tracts_data = setup
        with patch.object(GeneticModel, "loglik", side_effect=error):
            result = self._check(model, tracts_data, _physical_params((0.0, 0.0)), _physical_params((0.7, -0.5)),
                                 step_1_autosomal_loglik=-100.0)
        assert result is None
        assert any("could not be evaluated" in message for message in self._warnings(core_utils_caplog))

    @pytest.mark.parametrize("use_pulse_model, expected", [(False, [True]), (True, [None])])
    def test_two_steps_optimization_runs_the_check(self, setup, pulse_setup, core_utils_caplog, capsys,
                                                   use_pulse_model, expected):
        if use_pulse_model:
            model, population, _ = pulse_setup
            p0_opt = _to_opt(model, _pulse_physical_params((0.0, 0.0), t2=4.5))
        else:
            model, population, _ = setup
            p0_opt = _to_opt(model, _physical_params((0.0, 0.0), reur=0.25, ramr=0.25, t=5.0))
        results = []

        def recording_check(**kwargs):
            results.append(check_monoecious_autosomal_loglik_unchanged(**kwargs))
            return results[-1]

        with patch.object(core_module, "check_monoecious_autosomal_loglik_unchanged", side_effect=recording_check):
            _run_step(model, population, steps=None, p0_opt=p0_opt, maxiter=6)

        assert results == expected
        assert "WARNING" not in capsys.readouterr().out


# --------------- D. COBYLA stop-reason reporting ---------------

def _optimize_result(nfev, success, status, message):
    return scipy.optimize.OptimizeResult(x=np.zeros(2), nfev=nfev, success=success, status=status, message=message)


class TestCobylaStopReason:

    def test_message_when_evaluation_limit_reached(self):
        result = _optimize_result(300, False, 2, "Maximum number of function evaluations has been exceeded.")
        message = _build_cobyla_stop_message(result, maxfun=300, step_label="Step 1")
        assert message.startswith("Step 1: ")
        assert "evaluation limit (maximum_iterations = 300) was reached" in message

    def test_message_when_limit_reached_without_explicit_maxfun(self):
        # With maxfun=None, scipy applies its own default limit: detected from the result's message.
        result = _optimize_result(100, False, 2, "Maximum number of function evaluations has been exceeded.")
        message = _build_cobyla_stop_message(result, maxfun=None)
        assert "the evaluation limit was reached" in message

    def test_message_when_converged(self):
        result = _optimize_result(26, True, 1, "Optimization terminated successfully.")
        message = _build_cobyla_stop_message(result, maxfun=300)
        assert "COBYLA converged after 26 objective evaluations" in message
        assert "rhoend" in message

    @pytest.mark.parametrize("status, scipy_message", [
        (3, "Rounding errors are becoming damaging in COBYLA subroutine."),
        (5, "NaN result encountered."),
    ])
    def test_message_when_stopped_early_without_converging(self, status, scipy_message):
        # Stopping before the evaluation limit does not mean convergence: the scipy status decides.
        result = _optimize_result(40, False, status, scipy_message)
        message = _build_cobyla_stop_message(result, maxfun=300)
        assert "without converging" in message
        assert scipy_message in message
        assert f"(status {status})" in message
        assert "rhoend" not in message

    def test_run_cobyla_reports_limit_reached(self, capsys):
        _run_cobyla(func=lambda x: float(np.sum((x - 3) ** 2)), x0=np.zeros(2), cons=lambda x: 1.0,
                    maxfun=5, step_label="Step 1")
        assert "Step 1: COBYLA stopped after 5 objective evaluations: the evaluation limit" in capsys.readouterr().out

    def test_run_cobyla_reports_convergence(self, capsys):
        outputs = _run_cobyla(func=lambda x: float(np.sum((x - 0.02) ** 2)), x0=np.zeros(2), cons=lambda x: 1.0,
                              maxfun=1000)
        np.testing.assert_allclose(outputs, [0.02, 0.02], atol=1e-3)
        assert "COBYLA converged after" in capsys.readouterr().out

    def test_run_cobyla_reports_nan_result(self, capsys):
        _run_cobyla(func=lambda x: np.nan, x0=np.zeros(2), cons=lambda x: 1.0, maxfun=50)
        assert "without converging: NaN result encountered." in capsys.readouterr().out

    def test_run_cobyla_matches_fmin_cobyla(self):
        func = lambda x: float(np.sum((x - np.array([0.3, -0.2])) ** 2) + np.sin(5 * x[0]))
        cons = lambda x: 1.0 - np.sum(x ** 2)
        expected = scipy.optimize.fmin_cobyla(func, np.zeros(2), cons, rhobeg=.01, rhoend=.0001, maxfun=200)
        np.testing.assert_array_equal(_run_cobyla(func=func, x0=np.zeros(2), cons=cons, maxfun=200), expected)
