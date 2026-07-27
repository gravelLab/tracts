"""
Tests for tracts/genetic_model.py: PhaseTypeModelConfig, LoglikBreakdown, and GeneticModel.
"""

import copy
import logging
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tracts.genetic_model import GeneticModel, PhaseTypeModelConfig, LoglikBreakdown
from tracts.likelihood_options import LikelihoodOptions
from tracts.tracts_data import TractsData
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.base_parametrized_demography import FixedParametersHandler
from tracts.demography.parameter import ParamType


# --------------- Helpers ---------------

def _make_sex_biased_demography():
    dem = ParametrizedDemographySexBiased()
    dem.add_parameter("t", ParamType.TIME)
    dem.add_parameter("rate_eur", ParamType.RATE)
    dem.add_parameter("sb_eur", ParamType.SEX_BIAS)
    ph = FixedParametersHandler(logging.getLogger("test"))
    ph.set_up_fixed_parameters(dem, params_to_fix_by_ancestry=[], proportions={})
    dem.parameter_handler = ph
    return dem


def _make_demography():
    dem = ParametrizedDemography()
    dem.add_parameter("t", ParamType.TIME)
    dem.add_parameter("rate_eur", ParamType.RATE)
    ph = FixedParametersHandler(logging.getLogger("test"))
    ph.set_up_fixed_parameters(dem, params_to_fix_by_ancestry=[], proportions={})
    dem.parameter_handler = ph
    return dem


def _make_tracts_data(with_allosomes=True):
    population = MagicMock()
    population.Ls = [1.0]
    population.indivs = [object()] * 4
    population.num_females = 2
    population.num_males = 2
    kwargs = dict(
        population=population,
        autosome_bins=np.linspace(0, 1, 6),
        autosome_data_mapped=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )
    if with_allosomes:
        kwargs.update(
            allosome_bins=np.linspace(0, 1, 6),
            allosome_length=1.0,
            female_data_mapped=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            male_data_mapped=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            num_females=2,
            num_males=2,
        )
    return TractsData(**kwargs)


# --------------- LoglikBreakdown ---------------

class TestLoglikBreakdown:

    def test_total_sums_non_none_components(self):
        result = LoglikBreakdown(autosomes=-1.0, female_allosomes=-2.0, male_allosomes=-3.0)
        assert result.total == -6.0

    def test_total_ignores_none_components(self):
        result = LoglikBreakdown(autosomes=-1.0)
        assert result.total == -1.0

    def test_total_zero_when_all_none(self):
        result = LoglikBreakdown()
        assert result.total == 0


# --------------- PhaseTypeModelConfig ---------------

class TestPhaseTypeModelConfig:

    def test_defaults(self):
        config = PhaseTypeModelConfig()
        assert config.ad_model_autosomes == "DC"
        assert config.ad_model_allosomes == "DC"
        assert config.rho_f == 1
        assert config.rho_m == 1
        assert config.TP == 2
        assert config.N_cores == 1

    @pytest.mark.parametrize("bad_value", ["invalid", "dc", ""])
    def test_invalid_ad_model_autosomes_raises(self, bad_value):
        with pytest.raises(ValueError, match="ad_model_autosomes"):
            PhaseTypeModelConfig(ad_model_autosomes=bad_value)

    @pytest.mark.parametrize("bad_value", ["invalid", "M"])  # 'M' is valid for autosomes only
    def test_invalid_ad_model_allosomes_raises(self, bad_value):
        with pytest.raises(ValueError, match="ad_model_allosomes"):
            PhaseTypeModelConfig(ad_model_allosomes=bad_value)

    def test_ad_model_allosomes_none_is_allowed(self):
        config = PhaseTypeModelConfig(ad_model_allosomes=None)
        assert config.ad_model_allosomes is None
        assert config.models_allosomes is False

    def test_n_cores_below_one_raises(self):
        with pytest.raises(ValueError, match="N_cores"):
            PhaseTypeModelConfig(N_cores=0)

    def test_tp_below_one_raises(self):
        with pytest.raises(ValueError, match="TP"):
            PhaseTypeModelConfig(TP=0)

    @pytest.mark.parametrize("ad_model_autosomes,expected", [("DC", False), ("DF", False), ("M", False), ("H-DC", True), ("H-DF", True)])
    def test_uses_hybrid_pedigree_autosomes(self, ad_model_autosomes, expected):
        config = PhaseTypeModelConfig(ad_model_autosomes=ad_model_autosomes)
        assert config.uses_hybrid_pedigree_autosomes is expected

    @pytest.mark.parametrize("ad_model_allosomes,expected", [("DC", False), ("DF", False), ("H-DC", True), ("H-DF", True), (None, False)])
    def test_uses_hybrid_pedigree_allosomes(self, ad_model_allosomes, expected):
        config = PhaseTypeModelConfig(ad_model_allosomes=ad_model_allosomes)
        assert config.uses_hybrid_pedigree_allosomes is expected


# --------------- GeneticModel construction ---------------

class TestGeneticModelConstruction:

    def test_rejects_non_demography_object(self):
        with pytest.raises(TypeError, match="demographic_model"):
            GeneticModel(demographic_model=object())

    def test_accepts_parametrized_demography(self):
        dem = _make_demography()
        model = GeneticModel(dem)
        assert model.demographic_model is dem
        assert model.is_sex_biased is False

    def test_accepts_parametrized_demography_sex_biased(self):
        dem = _make_sex_biased_demography()
        model = GeneticModel(dem)
        assert model.is_sex_biased is True

    def test_builds_phase_type_config_from_kwargs(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DF", rho_f=2.0)
        assert model.phase_type_config.ad_model_autosomes == "DF"
        assert model.phase_type_config.rho_f == 2.0

    def test_accepts_explicit_phase_type_config(self):
        dem = _make_demography()
        config = PhaseTypeModelConfig(ad_model_autosomes="DF")
        model = GeneticModel(dem, phase_type_config=config)
        assert model.phase_type_config is config

    def test_rejects_both_phase_type_config_and_kwargs(self):
        dem = _make_demography()
        config = PhaseTypeModelConfig()
        with pytest.raises(ValueError, match="not both"):
            GeneticModel(dem, phase_type_config=config, ad_model_autosomes="DF")


# --------------- GeneticModel passthrough properties ---------------

class TestGeneticModelPassthroughs:

    def test_parameter_handler_passthrough(self):
        dem = _make_demography()
        model = GeneticModel(dem)
        assert model.parameter_handler is dem.parameter_handler

    def test_model_base_params_passthrough(self):
        dem = _make_demography()
        model = GeneticModel(dem)
        assert model.model_base_params is dem.model_base_params

    def test_population_indices_passthrough(self):
        dem = _make_demography()
        model = GeneticModel(dem)
        assert model.population_indices is dem.population_indices

    def test_get_migration_matrices_delegates(self):
        dem = _make_demography()
        dem.get_migration_matrices = MagicMock(return_value={"pop": np.zeros((1, 1))})
        model = GeneticModel(dem)
        result = model.get_migration_matrices([1.0, 2.0])
        dem.get_migration_matrices.assert_called_once_with([1.0, 2.0])
        assert result == {"pop": np.zeros((1, 1))}


# --------------- GeneticModel.model_func / outofbounds_fun ---------------

class TestGeneticModelModelFuncAndBounds:

    def test_model_func_converts_then_computes_matrices(self):
        dem = _make_demography()
        dem.parameter_handler.convert_to_physical_params = MagicMock(return_value=np.array([9.0, 9.0]))
        dem.get_migration_matrices = MagicMock(return_value={"female": np.zeros((1, 1)), "male": np.zeros((1, 1))})
        model = GeneticModel(dem)

        params = np.array([0.1, 0.2])
        result = model.model_func(params)

        dem.parameter_handler.convert_to_physical_params.assert_called_once()
        np.testing.assert_array_equal(dem.parameter_handler.convert_to_physical_params.call_args.args[0], params)
        dem.get_migration_matrices.assert_called_once()
        np.testing.assert_array_equal(dem.get_migration_matrices.call_args.args[0], np.array([9.0, 9.0]))
        assert set(result.keys()) == {"female", "male"}

    def test_outofbounds_fun_converts_then_scores(self):
        dem = _make_demography()
        dem.parameter_handler.convert_to_physical_params = MagicMock(return_value=np.array([9.0, 9.0]))
        dem.get_violation_score = MagicMock(return_value=0.5)
        model = GeneticModel(dem)

        result = model.outofbounds_fun(np.array([0.1, 0.2]), verbose=True)

        dem.get_violation_score.assert_called_once()
        np.testing.assert_array_equal(dem.get_violation_score.call_args.args[0], np.array([9.0, 9.0]))
        assert dem.get_violation_score.call_args.kwargs == {"verbose": True}
        assert result == 0.5

    def test_model_func_survives_copy(self):
        """
        model_func/outofbounds_fun are implemented as methods (not stored closures)
        specifically so they keep reading self.demographic_model fresh after .copy() —
        this is a regression test for that design choice.
        """
        dem = _make_demography()
        dem.get_migration_matrices = MagicMock(return_value={"marker": "original"})
        model = GeneticModel(dem)
        model_copy = model.copy()
        model_copy.demographic_model.get_migration_matrices = MagicMock(return_value={"marker": "copy"})

        assert model_copy.model_func(np.array([0.1, 0.2])) == {"marker": "copy"}


# --------------- GeneticModel.loglik dispatch ---------------

class TestGeneticModelLoglik:

    def _tracts_data(self, with_allosomes=True):
        return _make_tracts_data(with_allosomes=with_allosomes)

    def test_dispatches_to_phtmonoecious_for_m_model(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="M", ad_model_allosomes=None)
        male_matrix = np.zeros((2, 2))
        female_matrix = np.zeros((2, 2))

        with patch("tracts.genetic_model.PhTMonoecious") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.loglik.return_value = -5.0
            result = model.loglik(
                male_matrix=male_matrix, female_matrix=female_matrix,
                tracts_data=self._tracts_data(with_allosomes=False),
                likelihood_options=LikelihoodOptions(include_autosomes=True, include_allosomes=False),
            )

        mock_cls.assert_called_once()
        assert result.autosomes == -5.0
        assert result.female_allosomes is None
        assert result.male_allosomes is None

    def test_dispatches_to_phtdioecious_for_dc_model(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DC", ad_model_allosomes=None)

        with patch("tracts.genetic_model.PhTDioecious") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.loglik.return_value = -3.0
            result = model.loglik(
                male_matrix=np.zeros((2, 2)), female_matrix=np.zeros((2, 2)),
                tracts_data=self._tracts_data(with_allosomes=False),
                likelihood_options=LikelihoodOptions(include_autosomes=True, include_allosomes=False),
            )

        mock_cls.assert_called_once()
        assert result.autosomes == -3.0

    def test_dispatches_to_hybrid_pedigree_for_h_model(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="H-DC", ad_model_allosomes=None)

        with patch("tracts.genetic_model.HP.HP_loglik", return_value=-7.0) as mock_hp:
            result = model.loglik(
                male_matrix=np.zeros((2, 2)), female_matrix=np.zeros((2, 2)),
                tracts_data=self._tracts_data(with_allosomes=False),
                likelihood_options=LikelihoodOptions(include_autosomes=True, include_allosomes=False),
            )

        mock_hp.assert_called_once()
        assert mock_hp.call_args.kwargs["Dioecious_model"] == "DC"
        assert result.autosomes == -7.0

    def test_include_autosomes_false_skips_autosome_computation(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DC", ad_model_allosomes="DC")

        with patch("tracts.genetic_model.PhTDioecious") as mock_cls:
            mock_cls.return_value.loglik.return_value = -1.0
            result = model.loglik(
                male_matrix=np.zeros((2, 2)), female_matrix=np.zeros((2, 2)),
                tracts_data=self._tracts_data(with_allosomes=True),
                likelihood_options=LikelihoodOptions(include_autosomes=False, include_allosomes=True),
            )

        assert result.autosomes is None
        assert result.female_allosomes == -1.0
        assert result.male_allosomes == -1.0

    def test_include_allosomes_false_skips_allosome_computation(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DC", ad_model_allosomes="DC")

        with patch("tracts.genetic_model.PhTDioecious") as mock_cls:
            mock_cls.return_value.loglik.return_value = -2.0
            result = model.loglik(
                male_matrix=np.zeros((2, 2)), female_matrix=np.zeros((2, 2)),
                tracts_data=self._tracts_data(with_allosomes=True),
                likelihood_options=LikelihoodOptions(include_autosomes=True, include_allosomes=False),
            )

        assert result.autosomes == -2.0
        assert result.female_allosomes is None
        assert result.male_allosomes is None

    def test_allosome_computation_uses_hybrid_pedigree_when_configured(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DC", ad_model_allosomes="H-DF")

        with patch("tracts.genetic_model.HP.HP_loglik", side_effect=[-1.0, -2.0]) as mock_hp, \
             patch("tracts.genetic_model.PhTDioecious") as mock_pht:
            mock_pht.return_value.loglik.return_value = -9.0
            result = model.loglik(
                male_matrix=np.zeros((2, 2)), female_matrix=np.zeros((2, 2)),
                tracts_data=self._tracts_data(with_allosomes=True),
                likelihood_options=LikelihoodOptions(include_autosomes=False, include_allosomes=True),
            )

        assert mock_hp.call_count == 2
        assert result.female_allosomes == -1.0
        assert result.male_allosomes == -2.0


# --------------- GeneticModel.copy ---------------

class TestGeneticModelCopy:

    def test_copy_produces_distinct_demographic_model_instance(self):
        dem = _make_demography()
        model = GeneticModel(dem)
        model_copy = model.copy()

        assert model_copy is not model
        assert model_copy.demographic_model is not dem
        assert isinstance(model_copy.demographic_model, ParametrizedDemography)

    def test_copy_produces_distinct_phase_type_config(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DF")
        model_copy = model.copy()

        assert model_copy.phase_type_config is not model.phase_type_config
        assert model_copy.phase_type_config.ad_model_autosomes == "DF"

    def test_mutating_copy_does_not_affect_original(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DF")
        model_copy = model.copy()

        model_copy.phase_type_config.ad_model_autosomes = "DC"

        assert model.phase_type_config.ad_model_autosomes == "DF"


class TestGeneticModelRepr:

    def test_repr_contains_demographic_model_type_and_config(self):
        dem = _make_demography()
        model = GeneticModel(dem, ad_model_autosomes="DF")
        text = repr(model)
        assert "GeneticModel" in text
        assert "ParametrizedDemography" in text
        assert "DF" in text
