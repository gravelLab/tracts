"""
Tests for the re-optimization features in tracts.driver / tracts.driver_utils / tracts.core:

- ``n_reoptimizations``: repeatedly fixing sex-bias parameters at their most recently optimized
  values and re-running the optimization from the current optimum
  (``driver.run_sex_bias_fixing_reoptimizations``).
- ``rerun_optimization_on_boundaries``: re-running the optimization when a sex-bias parameter's
  optimal value is at its +-1 boundary, optionally switching the implicit population
  (``driver.run_boundary_reoptimization``, ``driver_utils.build_boundary_reoptimization_model``,
  ``driver_utils.get_alternate_implicit_population``).
- ``core.py``'s step-1 sex-bias fixing, which derives fixed values from ``p0`` instead of
  hardcoding 0, so that carrying sex-bias values forward (as done by the features above) actually
  has an effect on step 1.

Model-dependent tests use a real, in-memory ``ParametrizedDemographySexBiased`` with a single
founder event and two explicit source populations (``EUR``, ``NAT``) plus one implicit/remainder
population (``AFR``), built directly via ``add_founder_event`` (no YAML/file I/O, no real
optimization) -- mirroring the pattern already used in ``tests/test_ancestry_fixing.py`` and
``tests/test_driver_utils.py::TestComputeRemainderParams``. Driver-orchestration tests
(``run_sex_bias_fixing_reoptimizations``, ``run_boundary_reoptimization``) stub out the actual
optimizer calls, following ``tests/test_driver.py``'s pattern of monkeypatching
``run_model_multi_init``/callables rather than running a real likelihood computation.
"""
import logging
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

import tracts.driver as driver_module
import tracts.driver_utils as driver_utils_module
import tracts.core as core_module
from tracts.driver import run_sex_bias_fixing_reoptimizations, run_boundary_reoptimization
from tracts.driver_utils import (
    has_free_sex_bias_parameters,
    _build_reoptimization_intro_message,
    get_alternate_implicit_population,
    _get_driver_for_reoptimization,
    build_boundary_reoptimization_model,
    check_optimal_sex_bias_parameters_at_boundaries,
    compute_remainder_params,
    get_param_names_by_type,
    ModelReloadContext,
    InferenceConfig,
    OptimizationConfig,
    StartParamsConfig,
    ModelsConfig,
    SamplesConfig,
    OutputConfig,
)
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parameter import ParamType
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.genetic_model import GeneticModel


# --------------- Shared fixtures ---------------

def _make_three_pop_sex_biased_model():
    """
    A ParametrizedDemographySexBiased with one founder event: two explicit source populations
    (EUR -> REUR, NAT -> RNAT) and one implicit/remainder population (AFR). Parameter order is
    REUR, REUR_sex_bias, RNAT, RNAT_sex_bias, t.
    """
    model = ParametrizedDemographySexBiased(name="ThreePop")
    model.add_founder_event(
        dest_population="X",
        source_populations={"EUR": "REUR", "NAT": "RNAT"},
        remainder_population="AFR",
        found_time="t",
    )
    model.finalize()
    # Mirror driver_utils.setup_fixed_parameters's no-op call, which real runs always perform
    # once at startup: initializes current_fixed_parameters, required by add_fixed_parameters.
    model.set_up_fixed_parameters(params_to_fix_by_ancestry=[], proportions={}, user_params_to_fix_by_value={})
    return model


def _make_three_pop_model_with_implicit(remainder_population: str, source_populations: dict):
    """
    Like ``_make_three_pop_sex_biased_model``, but with a caller-chosen implicit/remainder
    population and explicit ``source_populations`` mapping (population name -> rate parameter
    name), for testing an implicit-population switch against a genuinely different model
    structure (rather than a same-structure stand-in).
    """
    model = ParametrizedDemographySexBiased(name="ThreePop")
    model.add_founder_event(
        dest_population="X",
        source_populations=source_populations,
        remainder_population=remainder_population,
        found_time="t",
    )
    model.finalize()
    model.set_up_fixed_parameters(params_to_fix_by_ancestry=[], proportions={}, user_params_to_fix_by_value={})
    return model


def _three_pop_param_names():
    return (
        ["REUR", "REUR_sex_bias", "RNAT", "RNAT_sex_bias", "t"],
        ["REUR_sex_bias", "RNAT_sex_bias"],
        ["REUR", "RNAT", "t"],
    )


def _make_real_driver_spec(**optim_overrides):
    """
    A minimal, real (pydantic) InferenceConfig for the three-pop sex-biased model, matching the
    parameter names/order of ``_make_three_pop_sex_biased_model``. Real pydantic instances are
    required (rather than SimpleNamespace mocks) because ``_get_driver_for_reoptimization`` and
    ``build_boundary_reoptimization_model`` call ``.model_copy(update=...)``.
    """
    optim_kwargs = dict(seed=1, repetitions=2, maximum_iterations=2, npts=5)
    optim_kwargs.update(optim_overrides)
    return InferenceConfig(
        samples=SamplesConfig(
            directory=".",
            individual_names=["indiv1"],
            filename_format="{individual_name}_{label}.bed",
            chromosomes="1",
            allosomes=["X"],
        ),
        models=ModelsConfig(model_filename="dummy_model.yaml", implicit_population="AFR"),
        start_params=StartParamsConfig(REUR=0.3, RNAT=0.3, t=10),
        optim=OptimizationConfig(**optim_kwargs),
        output=OutputConfig(output_filename_format="out_{label}"),
    )


# --------------- Pure helper functions ---------------

class TestHasFreeSexBiasParameters:

    def test_all_free(self):
        handler = SimpleNamespace(params_fixed_by_ancestry=[], user_params_fixed_by_value={})
        assert has_free_sex_bias_parameters(handler, ["sb_eur", "sb_afr"]) is True

    def test_all_fixed_by_ancestry(self):
        handler = SimpleNamespace(params_fixed_by_ancestry=["sb_eur", "sb_afr"], user_params_fixed_by_value={})
        assert has_free_sex_bias_parameters(handler, ["sb_eur", "sb_afr"]) is False

    def test_all_fixed_by_value(self):
        handler = SimpleNamespace(params_fixed_by_ancestry=[], user_params_fixed_by_value={"sb_eur": 0.1, "sb_afr": -0.2})
        assert has_free_sex_bias_parameters(handler, ["sb_eur", "sb_afr"]) is False

    def test_mixed_one_free(self):
        handler = SimpleNamespace(params_fixed_by_ancestry=["sb_eur"], user_params_fixed_by_value={})
        assert has_free_sex_bias_parameters(handler, ["sb_eur", "sb_afr"]) is True

    def test_no_sex_bias_params(self):
        handler = SimpleNamespace(params_fixed_by_ancestry=[], user_params_fixed_by_value={})
        assert has_free_sex_bias_parameters(handler, []) is False


class TestBuildReoptimizationIntroMessage:

    def test_contains_n_reoptimizations_value(self):
        message = _build_reoptimization_intro_message(5)
        assert "5" in message
        assert "re-optimization" in message.lower()

    def test_bordered_by_dashes(self):
        message = _build_reoptimization_intro_message(2)
        lines = [line for line in message.splitlines() if line]
        assert lines[0].startswith("---")
        assert lines[-1].startswith("---")


# --------------- Boundary detection / alternate implicit population ---------------

class TestCheckOptimalSexBiasParametersAtBoundaries:

    def _boundaries(self, optimal_params, boundary_tol=0.05):
        model = _make_three_pop_sex_biased_model()
        model_param_names, sex_bias_param_names, _ = _three_pop_param_names()
        matrices = model.get_migration_matrices(optimal_params)
        remainder_params = compute_remainder_params(model, matrices)
        driver_spec = SimpleNamespace(optim=SimpleNamespace(boundary_tol=boundary_tol, rerun_optimization_on_boundaries=True))
        return check_optimal_sex_bias_parameters_at_boundaries(
            demographic_model=model,
            driver_spec=driver_spec,
            sex_bias_param_names=sex_bias_param_names,
            remainder_params=remainder_params,
            optimal_params=optimal_params,
        )

    def test_no_parameter_at_boundary(self):
        boundaries = self._boundaries(np.array([0.3, 0.0, 0.3, 0.0, 10.0]))
        assert boundaries == []

    def test_explicit_sex_bias_at_positive_boundary(self):
        boundaries = self._boundaries(np.array([0.3, 1.0, 0.3, 0.0, 10.0]))
        assert "REUR_sex_bias" in boundaries
        assert "RNAT_sex_bias" not in boundaries

    def test_explicit_sex_bias_at_negative_boundary(self):
        boundaries = self._boundaries(np.array([0.3, -1.0, 0.3, 0.0, 10.0]))
        assert "REUR_sex_bias" in boundaries

    def test_implicit_remainder_sex_bias_at_boundary(self):
        # Push REUR and RNAT proportions so that the AFR remainder's sex-bias is close to a boundary.
        boundaries = self._boundaries(np.array([0.1, 0.0, 0.1, 1.0, 10.0]), boundary_tol=0.5)
        assert "X_AFR_sex_bias" in boundaries


class TestGetAlternateImplicitPopulation:

    def test_no_implicit_boundary_hit_returns_none(self):
        # Only an explicit sex-bias parameter is at boundary; the implicit population is fine.
        model = _make_three_pop_sex_biased_model()
        result = get_alternate_implicit_population(model, ["REUR_sex_bias"])
        assert result is None

    def test_implicit_boundary_hit_returns_first_free_alternate(self):
        model = _make_three_pop_sex_biased_model()
        result = get_alternate_implicit_population(model, ["X_AFR_sex_bias"])
        assert result == "EUR"

    def test_implicit_boundary_hit_skips_also_boundary_hit_alternate(self):
        model = _make_three_pop_sex_biased_model()
        result = get_alternate_implicit_population(model, ["X_AFR_sex_bias", "REUR_sex_bias"])
        assert result == "NAT"

    def test_implicit_boundary_hit_no_valid_alternate_returns_none(self, capsys):
        model = _make_three_pop_sex_biased_model()
        result = get_alternate_implicit_population(
            model, ["X_AFR_sex_bias", "REUR_sex_bias", "RNAT_sex_bias"]
        )
        assert result is None
        captured = capsys.readouterr()
        assert "AFR" in captured.out


# --------------- _get_driver_for_reoptimization ---------------

class TestGetDriverForReoptimization:

    def test_repetitions_set_to_one_and_start_params_updated(self):
        driver_spec = _make_real_driver_spec(repetitions=3)
        model_param_names, _, _ = _three_pop_param_names()
        optimal_params = np.array([0.4, 0.2, 0.5, -0.1, 12.0])

        reopt_spec = _get_driver_for_reoptimization(driver_spec, model_param_names, optimal_params)

        assert reopt_spec.optim.repetitions == 1
        assert reopt_spec.start_params.REUR == pytest.approx(0.4)
        assert reopt_spec.start_params.REUR_sex_bias == pytest.approx(0.2)
        assert reopt_spec.start_params.RNAT == pytest.approx(0.5)
        assert reopt_spec.start_params.RNAT_sex_bias == pytest.approx(-0.1)
        assert reopt_spec.start_params.t == pytest.approx(12.0)

    def test_original_driver_spec_is_unmodified(self):
        driver_spec = _make_real_driver_spec(repetitions=3)
        model_param_names, _, _ = _three_pop_param_names()
        optimal_params = np.array([0.4, 0.2, 0.5, -0.1, 12.0])

        _get_driver_for_reoptimization(driver_spec, model_param_names, optimal_params)

        assert driver_spec.optim.repetitions == 3
        assert driver_spec.start_params.REUR == pytest.approx(0.3)


# --------------- build_boundary_reoptimization_model ---------------

class TestBuildBoundaryReoptimizationModel:

    def test_no_implicit_population_change_reuses_genetic_model_copy(self):
        driver_spec = _make_real_driver_spec()
        model = _make_three_pop_sex_biased_model()
        genetic_model = GeneticModel(model, ad_model_autosomes="DC", ad_model_allosomes="DC")
        model_param_names, sex_bias_param_names, non_sex_bias_param_names = _three_pop_param_names()
        reload_context = ModelReloadContext(script_dir=".", driver_path="dummy_driver.yaml", allosome_label="X",
                                            autosome_proportions={}, allosome_proportions={})

        # A feasible parameter set for this model, with REUR_sex_bias at its +-1 boundary (the
        # value it will be fixed at) -- see also the module-level note on choosing feasible values.
        optimal_params = np.array([0.2, 1.0, 0.2, -0.2, 10.0])

        (reopt_driver_spec, reopt_genetic_model, out_model_param_names, out_sex_bias_names,
         out_non_sex_bias_names, physical_start_params) = build_boundary_reoptimization_model(
            driver_spec=driver_spec,
            reload_context=reload_context,
            boundary_fixed_param_values={"REUR_sex_bias": 1.0},
            genetic_model=genetic_model,
            optimal_params=optimal_params,
            remainder_params={},
            alternate_implicit_population=None,
        )

        assert reopt_driver_spec.optim.fix_parameters_by_value == {"REUR_sex_bias": 1.0}
        assert reopt_driver_spec.optim.repetitions == 1
        assert reopt_driver_spec.start_params.REUR == pytest.approx(0.2)
        assert reopt_driver_spec.start_params.t == pytest.approx(10.0)
        assert reopt_genetic_model.demographic_model is not model
        assert reopt_genetic_model.demographic_model.parameter_handler.current_fixed_parameters == {"REUR_sex_bias": 1.0}
        assert out_model_param_names == model_param_names
        assert out_sex_bias_names == sex_bias_param_names
        assert out_non_sex_bias_names == non_sex_bias_param_names
        assert len(physical_start_params) == 1
        # RNAT_sex_bias is free (not fixed by value here): it must start from its previous optimal
        # value (-0.2), not be reset to 0 as the very first (non-reoptimization) start would be.
        np.testing.assert_allclose(physical_start_params[0], [0.2, 1.0, 0.2, -0.2, 10.0])

    def test_alternate_implicit_population_reloads_model(self, monkeypatch, capsys):
        driver_spec = _make_real_driver_spec()
        model = _make_three_pop_sex_biased_model()
        genetic_model = GeneticModel(model, ad_model_autosomes="DC", ad_model_allosomes="DC")
        model_param_names, sex_bias_param_names, non_sex_bias_param_names = _three_pop_param_names()

        alt_model = _make_three_pop_sex_biased_model()
        alt_model_param_names, alt_sex_bias_names, alt_non_sex_bias_names = _three_pop_param_names()
        captured_load_calls = []

        def fake_load_demographic_model_from_driver(*, driver_spec, script_dir, driver_path, allosome_label):
            captured_load_calls.append(driver_spec)
            return alt_model, alt_model_param_names, alt_sex_bias_names, alt_non_sex_bias_names

        monkeypatch.setattr(driver_utils_module, "load_demographic_model_from_driver", fake_load_demographic_model_from_driver)

        reload_context = ModelReloadContext(script_dir=".", driver_path="dummy_driver.yaml", allosome_label="X",
                                            autosome_proportions={}, allosome_proportions={})

        optimal_params = np.array([0.2, 1.0, 0.2, -0.2, 10.0])

        (reopt_driver_spec, reopt_genetic_model, out_model_param_names, out_sex_bias_names,
         out_non_sex_bias_names, physical_start_params) = build_boundary_reoptimization_model(
            driver_spec=driver_spec,
            reload_context=reload_context,
            boundary_fixed_param_values={},
            genetic_model=genetic_model,
            optimal_params=optimal_params,
            remainder_params={},
            alternate_implicit_population="EUR",
        )

        assert len(captured_load_calls) == 1
        assert captured_load_calls[0].models.implicit_population == "EUR"
        assert reopt_driver_spec.models.implicit_population == "EUR"
        assert reopt_genetic_model.demographic_model is alt_model
        assert reopt_genetic_model.phase_type_config is genetic_model.phase_type_config
        # The reload branch calls setup_fixed_parameters(print_details=False): the model has
        # already been reported once for the original run, so this shouldn't be repeated here.
        captured_out = capsys.readouterr().out
        assert "Model parameters:" not in captured_out
        assert "have been fixed by value" not in captured_out

    def test_newly_explicit_population_sex_bias_is_fixed_at_its_boundary_value(self, monkeypatch):
        # The previously-implicit population's (AFR) derived sex-bias was at the +-1 boundary
        # (that's what triggered the implicit-population switch to EUR in the first place); once
        # AFR becomes explicit in the new model, its sex-bias parameter must be fixed by value at
        # that same boundary value, not left free to be resampled/re-optimized from scratch.
        old_model = _make_three_pop_sex_biased_model()  # EUR, NAT explicit; AFR implicit
        genetic_model = GeneticModel(old_model, ad_model_autosomes="DC", ad_model_allosomes="DC")
        driver_spec = _make_real_driver_spec()

        new_model = _make_three_pop_model_with_implicit("EUR", {"NAT": "RNAT", "AFR": "RAFR"})
        new_model_param_names, new_sex_bias_names, new_non_sex_bias_names = get_param_names_by_type(new_model)

        def fake_load_demographic_model_from_driver(*, driver_spec, script_dir, driver_path, allosome_label):
            return new_model, new_model_param_names, new_sex_bias_names, new_non_sex_bias_names

        monkeypatch.setattr(driver_utils_module, "load_demographic_model_from_driver", fake_load_demographic_model_from_driver)

        reload_context = ModelReloadContext(script_dir=".", driver_path="dummy_driver.yaml", allosome_label="X",
                                            autosome_proportions=[0.5, 0.3, 0.2], allosome_proportions=[0.5, 0.3, 0.2])

        remainder_params = {"X_AFR_rate": 0.25, "X_AFR_sex_bias": 0.97}
        optimal_params = np.array([0.2, 1.0, 0.2, -0.2, 10.0])

        (reopt_driver_spec, reopt_genetic_model, out_model_param_names, out_sex_bias_names,
         out_non_sex_bias_names, physical_start_params) = build_boundary_reoptimization_model(
            driver_spec=driver_spec,
            reload_context=reload_context,
            boundary_fixed_param_values={},
            genetic_model=genetic_model,
            optimal_params=optimal_params,
            remainder_params=remainder_params,
            alternate_implicit_population="EUR",
        )

        assert reopt_driver_spec.optim.fix_parameters_by_value["RAFR_sex_bias"] == pytest.approx(0.97)
        assert reopt_driver_spec.start_params.RAFR == pytest.approx(0.25)
        assert reopt_genetic_model.demographic_model.parameter_handler.user_params_fixed_by_value["RAFR_sex_bias"] == pytest.approx(0.97)
        # The rate is only seeded as a starting value, not fixed: it should remain free to optimize.
        assert "RAFR" not in reopt_genetic_model.demographic_model.parameter_handler.user_params_fixed_by_value
        # RNAT_sex_bias is retained from the old model and free (not fixed): it must carry forward
        # its previous optimal value (-0.2), not reset to 0.
        assert out_model_param_names == ["RNAT", "RNAT_sex_bias", "RAFR", "RAFR_sex_bias", "t"]
        np.testing.assert_allclose(physical_start_params[0], [0.2, -0.2, 0.25, 0.97, 10.0])


# --------------- run_sex_bias_fixing_reoptimizations ---------------

class TestRunSexBiasFixingReoptimizations:

    def test_stops_early_when_likelihood_stops_improving(self):
        model_param_names, _, _ = _three_pop_param_names()
        driver_spec = _make_real_driver_spec(n_reoptimizations=5)
        calls = []

        def fake_run_optimization_fixed_options(physical_start_params, driver_spec, **kwargs):
            calls.append((physical_start_params, driver_spec))
            return np.array([0.3, 0.1, 0.3, 0.1, 10.0]), -100.0

        optimal_params, optimal_likelihood = run_sex_bias_fixing_reoptimizations(
            driver_spec=driver_spec,
            model_param_names=model_param_names,
            optimal_params=np.array([0.3, 0.0, 0.3, 0.0, 10.0]),
            optimal_likelihood=-150.0,
            run_optimization_fixed_options=fake_run_optimization_fixed_options,
        )

        # First call "improves" (-150 -> -100, not close), second call has no further
        # improvement (-100 -> -100, close) and triggers the early stop.
        assert len(calls) == 2
        assert optimal_likelihood == pytest.approx(-100.0)
        np.testing.assert_allclose(optimal_params, [0.3, 0.1, 0.3, 0.1, 10.0])

    def test_runs_full_repetitions_without_early_stop(self):
        model_param_names, _, _ = _three_pop_param_names()
        driver_spec = _make_real_driver_spec(n_reoptimizations=3)
        likelihoods = iter([-140.0, -130.0, -120.0])
        calls = []

        def fake_run_optimization_fixed_options(physical_start_params, driver_spec, **kwargs):
            calls.append(physical_start_params)
            return np.array([0.3, 0.0, 0.3, 0.0, 10.0]), next(likelihoods)

        optimal_params, optimal_likelihood = run_sex_bias_fixing_reoptimizations(
            driver_spec=driver_spec,
            model_param_names=model_param_names,
            optimal_params=np.array([0.3, 0.0, 0.3, 0.0, 10.0]),
            optimal_likelihood=-150.0,
            run_optimization_fixed_options=fake_run_optimization_fixed_options,
        )

        assert len(calls) == 3
        assert optimal_likelihood == pytest.approx(-120.0)

    def test_reoptimization_driver_spec_built_once_and_reused_across_iterations(self):
        model_param_names, _, _ = _three_pop_param_names()
        driver_spec = _make_real_driver_spec(n_reoptimizations=3)
        seen_driver_specs = []
        likelihoods = iter([-140.0, -130.0, -120.0])

        def fake_run_optimization_fixed_options(physical_start_params, driver_spec, **kwargs):
            seen_driver_specs.append(driver_spec)
            return np.array([0.3, 0.0, 0.3, 0.0, 10.0]), next(likelihoods)

        run_sex_bias_fixing_reoptimizations(
            driver_spec=driver_spec,
            model_param_names=model_param_names,
            optimal_params=np.array([0.3, 0.0, 0.3, 0.0, 10.0]),
            optimal_likelihood=-150.0,
            run_optimization_fixed_options=fake_run_optimization_fixed_options,
        )

        assert len(seen_driver_specs) == 3
        assert all(spec is seen_driver_specs[0] for spec in seen_driver_specs)

    def test_start_params_carry_the_previous_call_optimum(self):
        model_param_names, _, _ = _three_pop_param_names()
        driver_spec = _make_real_driver_spec(n_reoptimizations=2)
        results = iter([
            (np.array([0.1, 0.1, 0.1, 0.1, 10.0]), -140.0),
            (np.array([0.2, 0.2, 0.2, 0.2, 10.0]), -120.0),
        ])
        seen_start_params = []

        def fake_run_optimization_fixed_options(physical_start_params, driver_spec, **kwargs):
            seen_start_params.append(physical_start_params)
            return next(results)

        run_sex_bias_fixing_reoptimizations(
            driver_spec=driver_spec,
            model_param_names=model_param_names,
            optimal_params=np.array([0.3, 0.0, 0.3, 0.0, 10.0]),
            optimal_likelihood=-150.0,
            run_optimization_fixed_options=fake_run_optimization_fixed_options,
        )

        np.testing.assert_allclose(seen_start_params[0][0], [0.3, 0.0, 0.3, 0.0, 10.0])
        np.testing.assert_allclose(seen_start_params[1][0], [0.1, 0.1, 0.1, 0.1, 10.0])


# --------------- run_boundary_reoptimization ---------------

class TestRunBoundaryReoptimization:

    def test_returns_unchanged_when_no_fix_and_no_alternate_population(self, monkeypatch):
        model = _make_three_pop_sex_biased_model()
        genetic_model = GeneticModel(model, ad_model_autosomes="DC", ad_model_allosomes="DC")
        model_param_names, sex_bias_param_names, non_sex_bias_param_names = _three_pop_param_names()
        driver_spec = _make_real_driver_spec()

        monkeypatch.setattr(driver_module, "get_alternate_implicit_population", lambda **kwargs: None)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("build_boundary_reoptimization_model should not be called")

        monkeypatch.setattr(driver_module, "build_boundary_reoptimization_model", fail_if_called)

        reload_context = ModelReloadContext(script_dir=".", driver_path="dummy_driver.yaml", allosome_label="X",
                                            autosome_proportions={}, allosome_proportions={})

        result_driver_spec, result_genetic_model, result_optimal_params, result_optimal_likelihood = run_boundary_reoptimization(
            driver_spec=driver_spec,
            reload_context=reload_context,
            optimal_sex_bias_at_boundaries=["X_AFR_sex_bias"],
            genetic_model=genetic_model,
            optimal_params=np.array([0.3, 0.0, 0.3, 0.0, 10.0]),
            optimal_likelihood=-150.0,
            remainder_params={},
            population=SimpleNamespace(),
            likelihood_options=SimpleNamespace(),
        )

        assert result_driver_spec is driver_spec
        assert result_genetic_model is genetic_model
        assert result_optimal_likelihood == pytest.approx(-150.0)

    def test_reoptimizes_when_directly_optimized_parameter_at_boundary(self, monkeypatch):
        model = _make_three_pop_sex_biased_model()
        genetic_model = GeneticModel(model, ad_model_autosomes="DC", ad_model_allosomes="DC")
        model_param_names, sex_bias_param_names, non_sex_bias_param_names = _three_pop_param_names()
        driver_spec = _make_real_driver_spec()

        rebuilt_genetic_model = GeneticModel(_make_three_pop_sex_biased_model(), ad_model_autosomes="DC", ad_model_allosomes="DC")
        rebuilt_driver_spec = _make_real_driver_spec()
        captured_build_calls = []
        captured_run_optimization_calls = []

        def fake_build_boundary_reoptimization_model(**kwargs):
            captured_build_calls.append(kwargs)
            return (rebuilt_driver_spec, rebuilt_genetic_model, model_param_names,
                    sex_bias_param_names, non_sex_bias_param_names, [np.array([0.3, 1.0, 0.3, 0.0, 10.0])])

        def fake_run_optimization(**kwargs):
            captured_run_optimization_calls.append(kwargs)
            return np.array([0.35, 0.9, 0.3, 0.0, 10.0]), -90.0

        monkeypatch.setattr(driver_module, "get_alternate_implicit_population", lambda **kwargs: None)
        monkeypatch.setattr(driver_module, "build_boundary_reoptimization_model", fake_build_boundary_reoptimization_model)
        monkeypatch.setattr(driver_module, "run_optimization", fake_run_optimization)
        monkeypatch.setattr(driver_module, "_print_optimal_values_and_likelihood", lambda **kwargs: None)

        reload_context = ModelReloadContext(script_dir=".", driver_path="dummy_driver.yaml", allosome_label="X",
                                            autosome_proportions={}, allosome_proportions={})

        result_driver_spec, result_genetic_model, result_optimal_params, result_optimal_likelihood = run_boundary_reoptimization(
            driver_spec=driver_spec,
            reload_context=reload_context,
            optimal_sex_bias_at_boundaries=["REUR_sex_bias"],
            genetic_model=genetic_model,
            optimal_params=np.array([0.3, 1.0, 0.3, 0.0, 10.0]),
            optimal_likelihood=-150.0,
            remainder_params={},
            population=SimpleNamespace(),
            likelihood_options=SimpleNamespace(),
        )

        assert len(captured_build_calls) == 1
        assert captured_build_calls[0]["boundary_fixed_param_values"] == {"REUR_sex_bias": 1.0}
        assert captured_build_calls[0]["alternate_implicit_population"] is None
        np.testing.assert_allclose(captured_build_calls[0]["optimal_params"], [0.3, 1.0, 0.3, 0.0, 10.0])
        assert len(captured_run_optimization_calls) == 1
        assert captured_run_optimization_calls[0]["print_run_details"] is False
        assert result_driver_spec is rebuilt_driver_spec
        assert result_genetic_model is rebuilt_genetic_model
        assert result_optimal_likelihood == pytest.approx(-90.0)
        np.testing.assert_allclose(result_optimal_params, [0.35, 0.9, 0.3, 0.0, 10.0])


# --------------- core.py: step-1 sex-bias fixing derives values from p0, not 0 ---------------

class TestCoreStepOneFixesSexBiasFromP0:
    """
    Confirms that ``optimize_cob_sex_biased_two_steps`` fixes free sex-bias parameters at their
    ``p0`` values during step 1, rather than hardcoding 0 -- the mechanism that lets
    ``run_sex_bias_fixing_reoptimizations`` carry sex-bias values forward across repetitions by
    simply passing them in ``p0``.
    """

    def _capture_fixed_sex_bias_values(self, p0):
        from tracts.likelihood_options import LikelihoodOptions
        from tracts.demography.parametrized_demography_sex_biased import SexType

        dem = ParametrizedDemographySexBiased()
        dem.add_parameter("t", ParamType.TIME)
        dem.add_parameter("rate_eur", ParamType.RATE)
        dem.add_parameter("sb_eur", ParamType.SEX_BIAS)
        dem.add_parameter("sb_afr", ParamType.SEX_BIAS)
        handler = FixedParametersHandler(logging.getLogger("test"))
        handler.set_up_fixed_parameters(dem, params_to_fix_by_ancestry=[], proportions={})
        dem.parameter_handler = handler
        genetic_model = GeneticModel(dem, ad_model_autosomes="DC", ad_model_allosomes="DC")

        pop = SimpleNamespace()
        bins = np.linspace(0, 1, 6)
        counts = np.zeros(5, dtype="int64").tolist()
        pop.get_global_tractlengths = lambda npts, exclude_tracts_below_cM=0: (bins, {"A": counts, "B": counts})
        pop.get_global_allosome_tractlengths = lambda npts, exclude_tracts_below_cM=0: (
            bins, {SexType.FEMALE: {"A": counts, "B": counts}, SexType.MALE: {"A": counts, "B": counts}}
        )
        pop.Ls = [1.0]
        pop.indivs = [object()] * 10
        pop.num_males = 5
        pop.num_females = 5
        pop.allosome_lengths = {"X": 1.0}

        captured = {}

        def capturing_fmin(func, x0, cons, **kwargs):
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

            handler_seen = visit(func)
            captured["current_fixed_parameters"] = dict(handler_seen.current_fixed_parameters)
            return x0

        from tracts.genetic_model import LoglikBreakdown
        dummy_matrix = np.array([[0.5, 0.5]])
        fake_loglik_result = LoglikBreakdown(autosomes=-1.0, female_allosomes=-1.0, male_allosomes=-1.0)

        with patch.object(GeneticModel, "model_func", return_value={"female": dummy_matrix, "male": dummy_matrix}), \
             patch.object(GeneticModel, "outofbounds_fun", return_value=1.0), \
             patch.object(GeneticModel, "loglik", return_value=fake_loglik_result), \
             patch.object(core_module.scipy.optimize, "fmin_cobyla", side_effect=capturing_fmin):
            core_module.optimize_cob_sex_biased_two_steps(
                p0=p0,
                population=pop,
                genetic_model=genetic_model,
                likelihood_options=LikelihoodOptions(verbose_log=0, verbose_screen=0),
                p_dict={"A": 0, "B": 1},
                exclude_tracts_below_cM=0,
                maxiter=2,
                reset_counter=True,
                autosomes_in_step_2=True,
                npts=5,
                steps=[1],
            )

        return captured["current_fixed_parameters"]

    def test_step1_fixes_sex_bias_at_nonzero_p0_values(self):
        # p0 is in optimizer space; sex-bias parameters are mapped to physical space ((-1, 1))
        # via sex_bias_to_physical_function(x) = 2*expit(x) - 1 before being fixed.
        from tracts.util import sex_bias_to_physical_function

        p0 = np.array([5.0, 0.3, 0.12, -0.34])
        fixed = self._capture_fixed_sex_bias_values(p0)
        assert fixed["sb_eur"] == pytest.approx(sex_bias_to_physical_function(0.12))
        assert fixed["sb_afr"] == pytest.approx(sex_bias_to_physical_function(-0.34))

    def test_step1_fixes_sex_bias_at_different_p0_values(self):
        # A different p0 must produce different fixed values, proving they are derived from p0
        # rather than coincidentally matching a hardcoded constant.
        from tracts.util import sex_bias_to_physical_function

        p0 = np.array([5.0, 0.3, -0.5, 0.7])
        fixed = self._capture_fixed_sex_bias_values(p0)
        assert fixed["sb_eur"] == pytest.approx(sex_bias_to_physical_function(-0.5))
        assert fixed["sb_afr"] == pytest.approx(sex_bias_to_physical_function(0.7))
        # Distinct from the other test's p0, confirming the fixed values track p0 rather than
        # being a coincidental/hardcoded constant.
        assert fixed["sb_eur"] != pytest.approx(0.0)
