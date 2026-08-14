from tracts.driver_utils import locate_file_path
from tracts.driver_utils import parse_chromosomes, parse_start_params, scale_select_indices, get_time_scaled_model_func, get_time_scaled_model_bounds
from tracts.driver_utils import parse_param_bounds, _print_param_bounds_table, check_optimal_params_near_bounds
from tracts.driver_utils import SamplesConfig, InferenceConfig, ParamBoundsConfig
from tracts.driver_utils import load_demographic_model_from_driver
from tracts.driver_utils import load_population
from tracts.driver_utils import compute_remainder_params
from tracts.driver_utils import _fill_missing_populations_with_zeros
from tracts.driver_utils import _run_with_generation_zero_warning_reporting
from tracts.driver_utils import _report_generation_zero_warning_for_optimal_params
from tracts.phase_type.base_phase_type import _GenerationZeroContributionWarning
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parameter import ParamType
from pathlib import Path
from types import SimpleNamespace
import warnings
import pytest
from unittest.mock import Mock, patch
import numpy as np
import tracts.driver_utils as driver_utils

"""
This test suite is designed to validate the functionality of the utility functions and configuration models used in the driver script of the tracts package, 
defined in the driver_utils module. The tests cover a range of functionalities including file path location, chromosome parsing, parameter parsing,
scaling of indices, and loading of models and populations based on driver specifications.
"""

@pytest.fixture
def mock_locate():
    with patch("tracts.driver_utils.locate_file_path") as mock:
        yield mock

@pytest.fixture
def mock_model_class():
    with patch.object(driver_utils, "ParametrizedDemography") as mock:
        yield mock

@pytest.fixture
def mock_model_class_sexbiased():
    with patch.object(driver_utils, "ParametrizedDemographySexBiased") as mock:
        yield mock

@pytest.fixture
def mock_parse_chrom():
    with patch("tracts.driver_utils.parse_chromosomes") as mock:
        yield mock

@pytest.fixture
def mock_parse_files():
    with patch("tracts.driver_utils.parse_individual_filenames") as mock:
        yield mock

@pytest.fixture
def mock_pop_class():
    with patch("tracts.driver_utils.Population") as mock:
        yield mock

class TestLocateFilePath:
    """
    A class for testing the locate_file_path function, which is responsible for finding the path to a specified file in various locations such as the working directory and script directory.
    The tests cover scenarios including successful file location, handling of non-existent files, and searching within the script directory. Each test ensures that the function behaves as
    expected under different conditions, providing confidence in its reliability for locating files needed by the driver script.
    """

    def test_locate_file_in_working_directory(self, tmp_path, monkeypatch):
        """
        Test finding file in working directory.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        monkeypatch.chdir(tmp_path)

        result = locate_file_path(
            filename="test.txt",
            script_dir=None,
            absolute_driver_yaml_path=None,
            verbose=False
        )
        assert result is not None
        assert result.resolve() == test_file.resolve()

    def test_locate_file_returns_none_when_not_found(self, tmp_path):
        """
        Test that None is returned when file doesn't exist.
        """
        result = locate_file_path(
            filename="nonexistent.txt",
            script_dir=tmp_path,
            absolute_driver_yaml_path=None,
            verbose=False
        )
        assert result is None

    def test_locate_file_in_script_directory(self, tmp_path):
        """
        Test finding file in script directory.
        """
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        test_file = script_dir / "test.txt"
        test_file.write_text("content")

        result = locate_file_path(
            filename="test.txt",
            script_dir=script_dir,
            absolute_driver_yaml_path=None,
            verbose=False
        )
        assert result is not None
        assert result.name == "test.txt"

class TestParseChromosomes:
    """
    A class for testing the parse_chromosomes function, which is designed to interpret various formats of chromosome specifications and return a standardized list of chromosome numbers.
    The tests cover different input formats, including single integers, range strings, lists of integers, and mixed lists containing both integers and range strings. Additionally,
    the tests ensure that invalid input formats raise appropriate errors, confirming the robustness of the function in handling diverse chromosome specifications.
    """

    def test_parse_single_integer(self):
        """
        Test parsing a single chromosome number.
        """
        result = parse_chromosomes(1)
        assert result == [1]

    def test_parse_range_string(self):
        """
        Test parsing chromosome range as string.
        """
        result = parse_chromosomes("1-5")
        assert result == [1, 2, 3, 4, 5]

    def test_parse_list_of_integers(self):
        """
        Test parsing list of integers.
        """
        result = parse_chromosomes([1, 2, 3])
        assert result == [1, 2, 3]

    def test_parse_mixed_list(self):
        """
        Test parsing list with mixed types.
        """
        assert parse_chromosomes([1, "2-4", 5]) == [1, 2, 3, 4, 5]
        assert parse_chromosomes(['1-5', 10, '13-18']) == [1, 2, 3, 4, 5, 10, 13, 14, 15, 16, 17, 18]


    def test_parse_invalid_range_raises_error(self):
        """
        Test that invalid range raises ValueError.
        """
        with pytest.raises(ValueError):
            parse_chromosomes("invalid")

class TestParseStartParams:
    """
    A class for testing the parse_start_params function, which is responsible for interpreting the starting parameter bounds for model optimization.
    The tests cover scenarios including fixed parameter values, range specifications, and error handling for missing parameters.
    """

    def test_parse_start_params_fixed_values(self):
        """
        Test parsing start params with fixed values.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.1, 1.0]),
            'param2': Mock(index=1, bounds=[0.1, 1.0])
        }
        mock_model.params_fixed_by_ancestry = []
        mock_model.parameter_handler = Mock()
        mock_model.get_violation_score.return_value = 1

        start_bounds = Mock(param1=0.5, param2=0.7)

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=2,
            seed=42,
            demographic_model=mock_model
        )

        assert len(result) == 2
        assert len(result[0]) == 2

    def test_parse_start_params_retries_after_negative_violation_score(self):
        """
        Test that infeasible candidates are rejected and resampled until feasible.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.1, 1.0])
        }
        mock_model.params_fixed_by_ancestry = []
        mock_model.parameter_handler = Mock()
        mock_model.get_violation_score.side_effect = [-1, 1]

        start_bounds = Mock(param1="0.5:0.9")

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=1,
            seed=42,
            demographic_model=mock_model
        )

        assert len(result) == 1
        assert mock_model.get_violation_score.call_count == 2
        assert result[0][0] != 0.5

    def test_parse_start_params_skips_value_error_candidates(self):
        """
        Test that candidates raising ValueError during feasibility checks are resampled.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.1, 1.0])
        }
        mock_model.params_fixed_by_ancestry = []
        mock_model.parameter_handler = Mock()
        mock_model.get_violation_score.side_effect = [ValueError("bad candidate"), 1]

        start_bounds = Mock(param1="0.5:0.9")

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=1,
            seed=42,
            demographic_model=mock_model
        )

        assert len(result) == 1
        assert mock_model.get_violation_score.call_count == 2
        assert result[0][0] != 0.5

    def test_parse_start_params_raises_when_all_candidates_are_infeasible(self):
        """
        Test that parse_start_params fails after exhausting the attempt limit.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.1, 1.0])
        }
        mock_model.params_fixed_by_ancestry = []
        mock_model.parameter_handler = Mock()
        mock_model.get_violation_score.return_value = -1

        start_bounds = Mock(param1=0.5)

        with pytest.raises(
            ValueError,
            match=r"Could not generate 1 feasible starting parameter sets after 1000 attempts"
        ):
            parse_start_params(
                start_param_bounds=start_bounds,
                repetitions=1,
                seed=42,
                demographic_model=mock_model
            )

        assert mock_model.get_violation_score.call_count == 1000

    def test_parse_start_params_range_values(self):
        """
        Test parsing start params with range values.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.1, 1.0])
        }
        mock_model.params_fixed_by_ancestry = []
        mock_model.parameter_handler = Mock()
        mock_model.get_violation_score.return_value = 1

        start_bounds = Mock(param1="0.5:0.9")

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=1,
            seed=42,
            demographic_model=mock_model
        )

        assert len(result) == 1
        assert 0.5 <= result[0][0] <= 0.9

    def test_parse_start_params_multiple_repetitions_use_user_lower_bounds_for_ancestry_fixed_params(self):
        """
        Test that ancestry-fixed parameters start from the user-provided lower bounds.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.05, 1.0]),
            'param2': Mock(index=1, bounds=[0.1, 1.0]),
            'param3': Mock(index=2, bounds=[0.2, 1.0])
        }
        mock_model.params_fixed_by_ancestry = {'param1': '', 'param2': ''}
        mock_model.parameter_handler = Mock()
        mock_model.parameter_handler.compute_params_fixed_by_ancestry.side_effect = lambda candidate: candidate
        mock_model.get_violation_score.return_value = 1

        start_bounds = Mock(param1="0.3:0.8", param2="0.4:0.9", param3="0.5:0.7")

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=3,
            seed=42,
            demographic_model=mock_model
        )

        assert len(result) == 3
        for start_params in result:
            assert start_params[0] == pytest.approx(0.3)
            assert start_params[1] == pytest.approx(0.4)
            assert 0.5 <= start_params[2] <= 0.7

    def test_parse_start_params_ancestry_fixed_param_can_be_missing(self):
        """
        Test that ancestry-fixed parameters can be omitted and default to model lower bound.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.05, 1.0]),
            'param2': Mock(index=1, bounds=[0.1, 1.0]),
        }
        mock_model.params_fixed_by_ancestry = {'param1': ''}
        mock_model.parameter_handler = Mock()
        mock_model.parameter_handler.compute_params_fixed_by_ancestry.side_effect = lambda candidate: candidate
        mock_model.get_violation_score.return_value = 1

        start_bounds = Mock(param2="0.4:0.9")

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=2,
            seed=42,
            demographic_model=mock_model
        )

        assert len(result) == 2
        for start_params in result:
            assert start_params[0] == pytest.approx(0.05)
            assert 0.4 <= start_params[1] <= 0.9

    def test_parse_start_params_missing_parameter_raises_error(self):
        """
        Test that missing parameter raises KeyError.
        """
        mock_model = Mock()
        mock_model.model_base_params = {
            'param1': Mock(index=0, bounds=[0.1, 1.0])
        }
        mock_model.params_fixed_by_ancestry = []

        start_bounds = Mock(spec=[])  # No attributes

        with pytest.raises(KeyError):
            parse_start_params(
                start_param_bounds=start_bounds,
                repetitions=1,
                seed=42,
                demographic_model=mock_model
            )


class TestParseParamBounds:
    """
    Tests for parse_param_bounds, which narrows demographic_model.model_base_params[...].bounds
    in place from a "min:max"-per-parameter bounds config (mirroring parse_start_params's
    "min:max" parsing), intersecting rather than replacing each parameter's existing bounds.
    """

    def test_narrows_specified_params_and_leaves_others_untouched(self):
        model = Mock()
        model.model_base_params = {
            'REUR': Mock(bounds=(1e-9, 1 - 1e-9)),
            't': Mock(bounds=(1, np.inf)),
        }

        # 't' is not mentioned in param_bounds and must keep its original bounds.
        param_bounds = Mock(spec=["REUR"], REUR="0.1:0.6")

        parse_param_bounds(param_bounds, model)

        assert model.model_base_params['REUR'].bounds == (0.1, 0.6)
        assert model.model_base_params['t'].bounds == (1, np.inf)

    def test_accepts_mapping_input(self):
        model = Mock()
        model.model_base_params = {'t': Mock(bounds=(1, np.inf))}

        parse_param_bounds({'t': '2:20'}, model)

        assert model.model_base_params['t'].bounds == (2.0, 20.0)

    def test_narrowing_is_intersection_not_replacement(self):
        """
        A requested bound wider than the parameter's current bounds does not widen it: only the
        overlap is kept.
        """
        model = Mock()
        model.model_base_params = {'REUR': Mock(bounds=(0.2, 0.8))}

        parse_param_bounds({'REUR': '0:1'}, model)

        assert model.model_base_params['REUR'].bounds == (0.2, 0.8)

    def test_unknown_parameter_name_raises_key_error(self):
        model = Mock()
        model.model_base_params = {'t': Mock(bounds=(1, np.inf))}

        with pytest.raises(KeyError, match="not_a_param"):
            parse_param_bounds({'not_a_param': '1:2'}, model)

    def test_malformed_bound_raises_value_error(self):
        model = Mock()
        model.model_base_params = {'t': Mock(bounds=(1, np.inf))}

        with pytest.raises(ValueError, match="min:max"):
            parse_param_bounds({'t': 'not-a-range'}, model)

    def test_min_greater_than_or_equal_to_max_raises_value_error(self):
        model = Mock()
        model.model_base_params = {'t': Mock(bounds=(1, np.inf))}

        with pytest.raises(ValueError, match="min:max"):
            parse_param_bounds({'t': '20:2'}, model)

    def test_non_overlapping_bound_raises_value_error(self):
        model = Mock()
        model.model_base_params = {'REUR': Mock(bounds=(1e-9, 1 - 1e-9))}

        with pytest.raises(ValueError, match="do not overlap"):
            parse_param_bounds({'REUR': '2:3'}, model)

    def test_enforced_by_real_model_check_bounds(self):
        """
        End-to-end check (real model, not a mock): narrowed bounds are actually read by
        check_bounds, the bounds component of the violation score the optimizer's constraint
        function (GeneticModel.outofbounds_fun) is built from.
        """
        model = ParametrizedDemography(name="Bounds")
        model.add_parameter("t", ParamType.TIME)
        model.add_parameter("REUR", ParamType.RATE)
        model.finalize()

        parse_param_bounds({'t': '2:20'}, model)

        t_index = model.model_base_params['t'].index
        reur_index = model.model_base_params['REUR'].index
        feasible = [0.0, 0.0]
        feasible[t_index], feasible[reur_index] = 10.0, 0.5
        assert model.check_bounds(feasible) >= 0

        infeasible = list(feasible)
        infeasible[t_index] = 25.0  # outside the narrowed (2, 20), inside the original (1, inf)
        assert model.check_bounds(infeasible) < 0


class TestPrintParamBoundsTable:
    """
    Tests for _print_param_bounds_table, the table printed/logged once at the start of a run
    showing each model parameter's effective (post-narrowing) bounds.
    """

    def test_prints_one_row_per_parameter_in_order(self, capsys):
        model = Mock()
        model.model_base_params = {
            'REUR': Mock(bounds=(0.1, 0.6)),
            't': Mock(bounds=(2.0, np.inf)),
        }

        _print_param_bounds_table(demographic_model=model)

        lines = capsys.readouterr().out.splitlines()
        assert "Model parameters and bounds:" in lines
        reur_line_index = next(i for i, l in enumerate(lines) if l.startswith("REUR"))
        t_line_index = next(i for i, l in enumerate(lines) if l.startswith("t "))
        assert "0.1" in lines[reur_line_index] and "0.6" in lines[reur_line_index]
        # np.inf must be displayed as "inf", not a raw float representation.
        assert "inf" in lines[t_line_index]
        # REUR's row must appear before t's row (model_base_params insertion order).
        assert reur_line_index < t_line_index


class TestCheckOptimalParamsNearBounds:
    """
    Tests for check_optimal_params_near_bounds, which flags (and warns about) a final optimal
    parameter only when it is close to a bound the user *narrowed* below the default (type-
    determined) one -- and only on the narrowed side. A parameter sitting at its natural type
    boundary (e.g. a sex-bias parameter at +-1) is not flagged.
    """

    # Default (pre-narrowing) bounds, mirroring tracts.demography.parameter.ParamType.
    _RATE_DEFAULT = ParamType.RATE.bounds        # (1e-9, 1 - 1e-9)
    _SEX_BIAS_DEFAULT = ParamType.SEX_BIAS.bounds  # (-1, 1)

    def _model(self, params: dict, min_time: float = 1.0, max_time: float = np.inf):
        """params maps name -> (ParamType, bounds tuple)."""
        model = Mock()
        model.min_time = min_time
        model.max_time = max_time
        model.model_base_params = {
            name: SimpleNamespace(type=param_type, bounds=bounds)
            for name, (param_type, bounds) in params.items()
        }
        return model

    def test_param_at_default_type_boundary_is_not_flagged(self, capsys):
        # Sex-bias parameter landing exactly at its natural +1 boundary (not user-narrowed).
        model = self._model({'sb': (ParamType.SEX_BIAS, self._SEX_BIAS_DEFAULT)})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([1.0]), tol=0.05)

        assert result == []
        assert capsys.readouterr().out == ""

    def test_rate_at_default_bounds_is_not_flagged(self):
        model = self._model({'REUR': (ParamType.RATE, self._RATE_DEFAULT)})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([0.9999]), tol=0.05)

        assert result == []

    def test_time_at_default_lower_bound_is_not_flagged(self):
        # TIME default (min_time, inf): neither side narrowed, so a value at min_time is not flagged.
        model = self._model({'t': (ParamType.TIME, (1.0, np.inf))})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([1.0]), tol=0.05)

        assert result == []

    def test_value_near_user_narrowed_lower_bound_is_flagged(self, capsys):
        # RATE narrowed to (0.1, 0.6): margin = 0.05 * 0.5 = 0.025.
        model = self._model({'REUR': (ParamType.RATE, (0.1, 0.6))})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([0.11]), tol=0.05)

        assert result == ['REUR']
        out = capsys.readouterr().out
        assert "REUR" in out and "close to the admissible bounds specified by the user" in out

    def test_value_near_user_narrowed_upper_bound_is_flagged(self):
        # TIME narrowed to (2, 20): margin = 0.05 * 18 = 0.9.
        model = self._model({'t': (ParamType.TIME, (2.0, 20.0))})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([19.5]), tol=0.05)

        assert result == ['t']

    def test_one_sided_narrowing_with_infinite_span_is_not_flagged(self):
        # TIME narrowed on the lower side only, leaving the upper unbounded (e.g. "2:inf"): the
        # admissible range is infinite, so the relative margin is undefined and nothing is flagged,
        # even for a value sitting right at the narrowed lower bound.
        model = self._model({'t': (ParamType.TIME, (2.0, np.inf))})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([2.0]), tol=0.05)

        assert result == []

    def test_value_within_user_narrowed_bounds_is_not_flagged(self):
        model = self._model({'REUR': (ParamType.RATE, (0.1, 0.6))})

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([0.35]), tol=0.05)

        assert result == []

    def test_only_the_narrowed_side_is_checked(self):
        # Lower narrowed (0.1 > default ~0), upper left at default (~1): a value near the default
        # upper must not be flagged; a value near the narrowed lower must be.
        model = self._model({'REUR': (ParamType.RATE, (0.1, self._RATE_DEFAULT[1]))})

        near_default_upper = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([0.999]), tol=0.05)
        assert near_default_upper == []

        near_narrowed_lower = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([0.12]), tol=0.05)
        assert near_narrowed_lower == ['REUR']

    def test_only_affected_parameters_are_reported(self):
        model = self._model({
            'REUR': (ParamType.RATE, (0.1, 0.6)),          # narrowed; value near lower -> flagged
            'RAMR': (ParamType.RATE, (0.1, 0.6)),          # narrowed; value mid-range -> not flagged
            'sb':   (ParamType.SEX_BIAS, self._SEX_BIAS_DEFAULT),  # default; at +1 -> not flagged
            't':    (ParamType.TIME, (1.0, np.inf)),       # default; not flagged
        })

        result = check_optimal_params_near_bounds(
            demographic_model=model, optimal_params=np.array([0.11, 0.35, 1.0, 50.0]), tol=0.05)

        assert result == ['REUR']


class TestComputePhysicalStartParamsSexBiasMidpoint:
    """
    In two-step optimization, compute_physical_start_params fixes free sex-bias parameters for
    step 1 at the midpoint of their admissible bounds (0 for the default (-1, 1) bounds), so that
    a user-narrowed bound excluding 0 still yields feasible starting parameters.
    """

    def _driver_spec(self, fix_by_value=None):
        return SimpleNamespace(
            start_params=SimpleNamespace(),
            optim=SimpleNamespace(
                two_steps_optimization=True,
                repetitions=1,
                seed=1,
                fix_parameters_by_value=fix_by_value or {},
            ),
        )

    def _model(self, sex_bias_bounds: dict):
        model = Mock()
        model.model_base_params = {
            name: SimpleNamespace(type=ParamType.SEX_BIAS, bounds=bounds)
            for name, bounds in sex_bias_bounds.items()
        }
        return model

    def test_narrowed_sex_bias_uses_midpoint_default_uses_zero(self):
        model = self._model({"sb_narrow": (0.2, 0.8), "sb_default": (-1, 1)})
        captured = {}

        def fake_parse_start_params(**kwargs):
            captured.update(kwargs)
            return [np.array([0.0])]

        with patch.object(driver_utils, "parse_start_params", side_effect=fake_parse_start_params), \
             patch.object(driver_utils, "collapse_identical_start_params", side_effect=lambda sp, label: sp):
            driver_utils.compute_physical_start_params(
                driver_spec=self._driver_spec(),
                demographic_model=model,
                sex_bias_param_names=["sb_narrow", "sb_default"],
                non_sex_bias_param_names=[],
            )

        assert captured["fixed_param_values"] == {"sb_narrow": 0.5, "sb_default": 0.0}

    def test_user_fixed_sex_bias_is_not_overridden_by_midpoint(self):
        model = self._model({"sb_narrow": (0.2, 0.8), "sb_fixed": (-1, 1)})
        captured = {}

        def fake_parse_start_params(**kwargs):
            captured.update(kwargs)
            return [np.array([0.0])]

        with patch.object(driver_utils, "parse_start_params", side_effect=fake_parse_start_params), \
             patch.object(driver_utils, "collapse_identical_start_params", side_effect=lambda sp, label: sp):
            driver_utils.compute_physical_start_params(
                driver_spec=self._driver_spec(fix_by_value={"sb_fixed": 0.9}),
                demographic_model=model,
                sex_bias_param_names=["sb_narrow", "sb_fixed"],
                non_sex_bias_param_names=[],
            )

        # sb_fixed keeps its user-specified value; only sb_narrow gets the midpoint default.
        assert captured["fixed_param_values"] == {"sb_narrow": 0.5, "sb_fixed": 0.9}


class TestScaleSelectIndices:
    """
    A class for testing the scale_select_indices function, which is designed to apply a scaling factor to selected indices of an array based on a provided boolean mask.
    The tests cover scenarios including basic scaling of selected indices, ensuring that non-selected indices remain unchanged, and error handling for mismatched lengths between the input array and the indices mask.
    """

    def test_scale_select_indices_basic(self):
        """
        Test basic scaling of selected indices.
        """
        arr = np.array([1.0, 2.0, 3.0])
        indices = np.array([1, 0, 1])

        result = scale_select_indices(arr, indices, scaling_factor=2)

        expected = np.array([2.0, 2.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_select_indices_no_scaling(self):
        """
        Test with scaling factor = 1 (no scaling).
        """
        arr = np.array([1.0, 2.0, 3.0])
        indices = np.array([1, 1, 1])
        result = scale_select_indices(arr, indices, scaling_factor=1)
        np.testing.assert_array_almost_equal(result, arr)

    def test_scale_select_indices_mismatched_length_raises_error(self):
        """
        Test that mismatched lengths raise ValueError.
        """
        arr = np.array([1.0, 2.0])
        indices = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            scale_select_indices(arr, indices, scaling_factor=2)

class TestGetTimeScaledModelFunc:
    """
    A class for testing the get_time_scaled_model_func function, which generates a callable function that applies time scaling to a model's parameters and retrieves the corresponding migration matrices.
    The tests cover scenarios including verifying that the returned object is callable and ensuring that the function correctly applies parameter conversion and retrieves migration matrices from the model.
    """

    def test_get_time_scaled_model_func_returns_callable(self):
        """
        Test that function returns a callable.
        """
        mock_model = Mock()
        mock_model.parameter_handler = Mock()
        mock_model.get_migration_matrices = Mock(return_value={"matrix": np.array([[1, 0], [0, 1]])})

        func = get_time_scaled_model_func(mock_model)
        assert callable(func)

    def test_get_time_scaled_model_func_applies_conversion(self):
        """
        Test that returned function applies parameter conversion and retrieves migration matrices.
        """
        mock_model = Mock()
        mock_model.parameter_handler.convert_to_physical_params = Mock(
            return_value=np.array([0.5, 0.5])
        )
        mock_model.get_migration_matrices = Mock(
            return_value={"matrix": np.array([[1, 0], [0, 1]])}
        )

        params = np.array([0.1, 0.2])
        func = get_time_scaled_model_func(mock_model)
        result = func(params)
        
        mock_model.parameter_handler.convert_to_physical_params.assert_called_once_with(params)
        mock_model.get_migration_matrices.assert_called_once()

        expected = {"matrix": np.array([[1, 0], [0, 1]])}
        assert result.keys() == expected.keys()
        np.testing.assert_array_equal(result["matrix"], expected["matrix"])

class TestGetTimeScaledModelBounds:
    """
    A class for testing the get_time_scaled_model_bounds function, which generates a callable function that applies time scaling to a model's parameters and retrieves the violation score for parameter bounds.
    The tests cover scenarios including verifying that the returned object is callable and ensuring that the function correctly applies parameter conversion and retrieves the violation score from the model. 
    """

    def test_get_time_scaled_model_bounds_returns_callable(self):
        """
        Test that function returns a callable.
        """
        mock_model = Mock()
        mock_model.parameter_handler = Mock()
        mock_model.get_violation_score = Mock(return_value=0.5)

        func = get_time_scaled_model_bounds(mock_model)
        assert callable(func)

    def test_get_time_scaled_model_bounds_applies_conversion(self):
        """
        Test that returned function applies parameter conversion and retrieves violation score.
        """
        mock_model = Mock()
        mock_model.parameter_handler.convert_to_physical_params = Mock(
            return_value=np.array([0.5, 0.5])
        )
        mock_model.get_violation_score = Mock(return_value=0.1)

        func = get_time_scaled_model_bounds(mock_model)
        params = np.array([0.1, 0.2])
        result = func(params)

        mock_model.parameter_handler.convert_to_physical_params.assert_called_once_with(params)
        mock_model.get_violation_score.assert_called_once()
        assert result == 0.1

class TestConfigModels:
    """
    A class for testing the configuration models used in the driver script, including SamplesConfig and InferenceConfig.
    The tests cover scenarios such as validating the creation of configuration instances, ensuring that default values are set correctly,
    and verifying that extra fields are not allowed in the InferenceConfig model.
    """

    def test_samples_config_validation(self):
        """
        Test SamplesConfig model validation.
        """
        config = SamplesConfig(
            directory="./samples/",
            individual_names=["ind1", "ind2"],
            filename_format="{individual}_{label}.txt",
            labels=["A", "B"],
            chromosomes="1-22",
            allosomes=[]
        )
        assert config.directory == "./samples/"
        assert len(config.individual_names) == 2

    def test_samples_config_default_labels(self):
        """
        Test SamplesConfig default labels.
        """
        config = SamplesConfig(
            directory="./samples/",
            individual_names=["ind1"],
            filename_format="{individual}_{label}.txt",
            chromosomes="1-22"
        )
        assert config.labels == ["A", "B"]

    def test_inference_config_defaults(self):
        """
        Test InferenceConfig default values.
        """
        mock_samples = Mock(spec=SamplesConfig)
        config_dict = {
            'samples': mock_samples,
            'models': {'model_filename': 'model.yaml'},
            'start_params': {},
            'optim': {'seed': 42},
            'output': {'output_filename_format': 'output_{label}.txt'}
        }
        config = InferenceConfig(**config_dict)
        assert config.models.model_filename == 'model.yaml'
        assert config.optim.seed == 42
        assert config.output.output_filename_format == 'output_{label}.txt'
        assert config.optim.npts == 50
        assert config.optim.exclude_tracts_below_cm == 1
        assert config.output.log_scale is True
        assert config.optim.repetitions == 1

        
    def test_inference_config_forbids_extra_fields(self):
        """
        Test that InferenceConfig forbids extra fields.
        """
        with pytest.raises(Exception):
            InferenceConfig(
                samples=Mock(),
                models={'model_filename': 'model.yaml'},
                start_params={},
                optim={'seed': 42},
                output={'output_filename_format': 'output_{label}.txt'},
                extra_field="should_fail"
            )
    
    def test_samples_config_missing_required_field(self):
        """
        Test SamplesConfig with missing required field.
        """
        with pytest.raises(Exception):  # ValidationError
            SamplesConfig(individual_names=["ind1"])

    def test_models_config_implicit_population_default(self):
        """
        Test that ModelsConfig.implicit_population defaults to None when not specified.
        """
        from tracts.driver_utils import ModelsConfig
        config = ModelsConfig(model_filename='model.yaml')
        assert config.implicit_population is None

    def test_models_config_implicit_population_explicit(self):
        """
        Test that ModelsConfig.implicit_population can be set explicitly.
        """
        from tracts.driver_utils import ModelsConfig
        config = ModelsConfig(model_filename='model.yaml', implicit_population='AFR')
        assert config.implicit_population == 'AFR'

    def test_inference_config_bounds_defaults_to_empty(self):
        """
        Test that InferenceConfig.bounds defaults to an empty ParamBoundsConfig when the driver
        file omits a "bounds:" section, so existing driver files without one keep working.
        """
        mock_samples = Mock(spec=SamplesConfig)
        config = InferenceConfig(
            samples=mock_samples,
            models={'model_filename': 'model.yaml'},
            start_params={},
            optim={'seed': 42},
            output={'output_filename_format': 'output_{label}.txt'},
        )
        assert isinstance(config.bounds, ParamBoundsConfig)
        assert config.bounds.model_dump() == {}

    def test_inference_config_bounds_explicit(self):
        """
        Test that InferenceConfig.bounds accepts a "min:max"-per-parameter mapping, mirroring
        start_params's format.
        """
        mock_samples = Mock(spec=SamplesConfig)
        config = InferenceConfig(
            samples=mock_samples,
            models={'model_filename': 'model.yaml'},
            start_params={},
            bounds={'t': '2:20', 'REUR': '0.1:0.6'},
            optim={'seed': 42},
            output={'output_filename_format': 'output_{label}.txt'},
        )
        assert config.bounds.t == '2:20'
        assert config.bounds.REUR == '0.1:0.6'


class TestLoadModelFromDriver:
    """
    A class for testing the load_demographic_model_from_driver function, which is responsible for loading a demographic model based on specifications provided in a driver file.
    The tests cover scenarios such as successful model loading, handling of missing model filename in the driver specifications, and error handling for cases where the specified model file cannot be found.
    """

    def test_load_model_basic(self, mock_locate, mock_model_class):
        """
        Test basic model loading without sex-bias.
        """
        model_path = Path("/path/to/model.yaml")
        mock_locate.return_value = model_path

        mock_model = Mock()
        mock_model.model_base_params = {
            "rate1": Mock(type=ParamType.RATE),
            "sb1": Mock(type=ParamType.SEX_BIAS),
        }
        mock_model_class.load_from_YAML.return_value = mock_model

        driver_spec = Mock()
        driver_spec.models.model_filename = "model.yaml"
        driver_spec.models.implicit_population = None
        driver_spec.samples.allosomes = None

        demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names = load_demographic_model_from_driver(
            driver_spec,
            script_dir=None,
            driver_path="/path/driver.yaml",
        )

        assert demographic_model is mock_model
        assert model_param_names == ["rate1", "sb1"]
        assert sex_bias_param_names == ["sb1"]
        assert non_sex_bias_param_names == ["rate1"]
        mock_model_class.load_from_YAML.assert_called_once_with(
            source=str(model_path.resolve()),
            implicit_population=None,
        )

    def test_load_model_basic_sexbiased(self, mock_locate, mock_model_class, mock_model_class_sexbiased):
        """
        Test basic sex-biased model loading.
        """
        model_path = Path("/path/to/model.yaml")
        mock_locate.return_value = model_path
        mock_model = Mock()
        mock_model.model_base_params = {
            "rate1": Mock(type=ParamType.RATE),
            "sb1": Mock(type=ParamType.SEX_BIAS),
        }
        mock_model_class_sexbiased.load_from_YAML.return_value = mock_model

        driver_spec = Mock()
        driver_spec.models.model_filename = "model.yaml"
        driver_spec.models.implicit_population = None
        driver_spec.samples = Mock()
        driver_spec.samples.allosomes = ["X"]
        driver_spec.samples.male_names = ["ind1"]

        demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names = load_demographic_model_from_driver(
            driver_spec, script_dir=None,
            driver_path="/path/driver.yaml",
            allosome_label="X")

        assert demographic_model == mock_model
        assert model_param_names == ["rate1", "sb1"]
        assert sex_bias_param_names == ["sb1"]
        assert non_sex_bias_param_names == ["rate1"]
        mock_model_class_sexbiased.load_from_YAML.assert_called_once_with(
            source=str(model_path.resolve()),
            implicit_population=None,
        )
        mock_model_class.load_from_YAML.assert_not_called()

    def test_load_model_forwards_implicit_population(self, mock_locate, mock_model_class):
        """
        Test that a non-None implicit_population set in the driver spec is forwarded
        to ParametrizedDemography.load_from_YAML.
        """
        model_path = Path("/path/to/model.yaml")
        mock_locate.return_value = model_path

        mock_model = Mock()
        mock_model.model_base_params = {
            "rate1": Mock(type=ParamType.RATE),
        }
        mock_model_class.load_from_YAML.return_value = mock_model

        driver_spec = Mock()
        driver_spec.models.model_filename = "model.yaml"
        driver_spec.models.implicit_population = "AFR"
        driver_spec.samples.allosomes = None

        demographic_model, model_param_names, sex_bias_param_names, non_sex_bias_param_names = load_demographic_model_from_driver(
            driver_spec,
            script_dir=None,
            driver_path="/path/driver.yaml",
        )

        assert demographic_model is mock_model
        assert model_param_names == ["rate1"]
        assert sex_bias_param_names == []
        assert non_sex_bias_param_names == ["rate1"]
        mock_model_class.load_from_YAML.assert_called_once_with(
            source=str(model_path.resolve()),
            implicit_population="AFR",
        )

    def test_load_model_missing_filename_raises_error(self):
        """
        Test that an error is raised when model_filename is not accessible.
        Validation of models.model_filename presence now happens at load_driver_file
        time (InferenceConfig requires it), so accessing it on a bare Mock(spec=[])
        raises AttributeError.
        """
        driver_spec = Mock(spec=[])  # No attributes at all
        
        with pytest.raises((ValueError, AttributeError)):
            load_demographic_model_from_driver(driver_spec, script_dir=None, 
                                  driver_path="/path/driver.yaml")
    

    def test_load_model_file_not_found_raises_error(self, mock_locate):
        """
        Test FileNotFoundError when model file doesn't exist.
        """
        mock_locate.return_value = None
        
        driver_spec = Mock()
        driver_spec.models.model_filename = "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_demographic_model_from_driver(driver_spec, script_dir=None, 
                                  driver_path="/path/driver.yaml")


class TestLoadPopulation:
    """
    A class for testing the load_population function, which is responsible for loading population data based on specifications provided in a driver file.
    The tests cover scenarios such as successful population loading, handling of allosome specifications, and error handling for cases where the specified population data cannot be found or loaded correctly.
    """
    
    def test_load_population_basic(self, mock_parse_chrom, mock_parse_files, mock_pop_class):
        """
        Test basic population loading.
        """
        mock_parse_files.return_value = {"ind1": ["file1_A", "file1_B"]}
        mock_parse_chrom.return_value = [1, 2, 3]
        mock_pop = Mock()
        mock_pop_class.return_value = mock_pop
        
        driver_spec = Mock()
        driver_spec.samples.individual_names = ["ind1"]
        driver_spec.samples.filename_format = "{name}_{label}.txt"
        driver_spec.samples.labels = ["A", "B"]
        driver_spec.samples.directory = ""
        driver_spec.samples.male_names = None
        driver_spec.samples.chromosomes = "1-3"
        
        result = load_population("/path/driver.yaml", driver_spec)
        
        assert result == mock_pop
        mock_pop_class.assert_called_once()
    
 
    def test_load_population_with_allosomes(self, mock_parse_chrom, mock_parse_files, mock_pop_class):
        """
        Test population loading with allosomes.
        """
        mock_parse_files.return_value = {"ind1": ["file1_A", "file1_B"]}
        mock_parse_chrom.return_value = [1, 2, 3, "X"]
        mock_pop = Mock()
        mock_pop_class.return_value = mock_pop
        
        driver_spec = Mock()
        driver_spec.samples.individual_names = ["ind1"]
        driver_spec.samples.filename_format = "{name}_{label}.txt"
        driver_spec.samples.labels = ["A", "B"]
        driver_spec.samples.directory = ""
        driver_spec.samples.male_names = ["ind1"]
        driver_spec.samples.chromosomes = "1-3"
        
        result = load_population("/path/driver.yaml", driver_spec, 
                               allosome_labels=["X"])
        
        assert result == mock_pop
        mock_pop.set_males.assert_called_once_with(male_list=["ind1"], 
                                                   allosome_label="X")          


class TestComputeRemainderParams:
    """
    Tests for compute_remainder_params, which extracts the founding rate (and,
    for sex-biased models, the derived sex bias) of the remainder/dependent
    ancestry from the final migration matrices.
    """

    # ------------------------------------------------------------------
    # Helpers to build minimal models
    # ------------------------------------------------------------------

    def _plain_model(self, founder_rate=0.3, found_time=5):
        """ParametrizedDemography with one source + one remainder ancestry."""
        model = ParametrizedDemography()
        # Parameters added: founder_rate (idx 0), found_time (idx 1)
        model.add_founder_event(
            "dest_pop",
            {"source_pop": "founder_rate"},
            "remainder_pop",
            "found_time",
        )
        # parametrized_populations is only set via YAML in ParametrizedDemography;
        # set it manually to mirror real runtime behaviour.
        model.parametrized_populations = ["dest_pop"]
        matrices = model.get_migration_matrices([founder_rate, found_time])
        return model, matrices

    def _sex_biased_model(self, founder_rate=0.3, sex_bias=0.5, found_time=5):
        """ParametrizedDemographySexBiased with one source + one remainder ancestry."""
        model = ParametrizedDemographySexBiased()
        # Parameters added: founder_rate (idx 0), founder_rate_sex_bias (idx 1),
        #                   found_time (idx 2).
        # parametrized_populations is populated automatically by add_founder_event.
        model.add_founder_event(
            "dest_pop",
            {"source_pop": "founder_rate"},
            "remainder_pop",
            "found_time",
        )
        matrices = model.get_migration_matrices([founder_rate, sex_bias, found_time])
        return model, matrices

    # ------------------------------------------------------------------
    # Non-demography type
    # ------------------------------------------------------------------

    def test_unsupported_model_type_returns_empty_dict(self):
        """Any object that is not a recognised demography type returns {}."""
        from types import SimpleNamespace
        result = compute_remainder_params(SimpleNamespace(), {})
        assert result == {}

    # ------------------------------------------------------------------
    # Plain (non-sex-biased) model
    # ------------------------------------------------------------------

    def test_plain_basic_rate(self):
        """Remainder rate = 1 - source_rate is read from the founding row."""
        model, matrices = self._plain_model(founder_rate=0.3, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 0.7)

    def test_plain_rate_zero(self):
        """When source occupies 100 %, remainder rate = 0."""
        model, matrices = self._plain_model(founder_rate=1.0, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 0.0)

    def test_plain_rate_one(self):
        """When source contributes 0 %, remainder rate = 1."""
        model, matrices = self._plain_model(founder_rate=0.0, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 1.0)

    def test_plain_no_sex_bias_key(self):
        """Non-sex-biased models must not produce a sex_bias key."""
        model, matrices = self._plain_model()
        result = compute_remainder_params(model, matrices)
        assert not any("sex_bias" in k for k in result)

    def test_plain_key_includes_dest_pop(self):
        """Key must be '{dest_pop}_{remainder_pop}_rate', not just '{remainder_pop}_rate'."""
        model, matrices = self._plain_model()
        result = compute_remainder_params(model, matrices)
        assert "dest_pop_remainder_pop_rate" in result
        assert "remainder_pop_rate" not in result

    def test_plain_duplicate_in_parametrized_populations(self):
        """A population listed twice is processed only once (no duplicate keys)."""
        model, matrices = self._plain_model()
        model.parametrized_populations = ["dest_pop", "dest_pop"]
        result = compute_remainder_params(model, matrices)
        assert list(result.keys()).count("dest_pop_remainder_pop_rate") == 1

    def test_plain_empty_parametrized_populations(self):
        """Empty parametrized_populations → empty result."""
        model, matrices = self._plain_model()
        model.parametrized_populations = []
        result = compute_remainder_params(model, matrices)
        assert result == {}

    def test_plain_no_remainder_continuous_founder(self):
        """Continuous founder event has no remainder_population → empty result."""
        model = ParametrizedDemography()
        # Parameters: rate1 (0), rate2 (1), found_time (2), end_time (3)
        model.add_founder_event(
            "dest_pop",
            {"source_pop1": "rate1", "source_pop2": "rate2"},
            None,
            "found_time",
            end_time="end_time",
        )
        model.parametrized_populations = ["dest_pop"]
        matrices = model.get_migration_matrices([0.4, 0.3, 8, 5])
        result = compute_remainder_params(model, matrices)
        assert result == {}

    # ------------------------------------------------------------------
    # Sex-biased model
    # ------------------------------------------------------------------

    def test_sex_biased_rate_value(self):
        """Remainder rate = mean of male and female founding rates = 1 - source_rate."""
        model, matrices = self._sex_biased_model(founder_rate=0.3, sex_bias=0.0, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 0.7)

    def test_sex_biased_sex_bias_opposite_sign(self):
        """
        Remainder sex bias is the negative of the source sex bias.

        For source rate r and sex bias s:
          r_male_source   = r - s * min(r, 1-r)
          r_female_source = r + s * min(r, 1-r)
          r_male_rem   = 1 - r_male_source
          r_female_rem = 1 - r_female_source
          r_mean_rem   = 1 - r
          sex_bias_rem = (r_female_rem - r_male_rem) / (2 * min(r_mean_rem, 1 - r_mean_rem))
                       = -2 s * min(r, 1-r) / (2 * min(1-r, r))
                       = -s
        """
        model, matrices = self._sex_biased_model(founder_rate=0.3, sex_bias=0.5, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_sex_bias"], -0.5)

    def test_sex_biased_zero_sex_bias(self):
        """Zero source sex bias → remainder sex bias is also 0."""
        model, matrices = self._sex_biased_model(founder_rate=0.4, sex_bias=0.0, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_sex_bias"], 0.0)

    def test_sex_biased_nan_when_remainder_rate_zero(self):
        """When remainder rate = 0 the sex bias denominator collapses → NaN."""
        # source_rate = 1 → remainder_rate = 0
        model, matrices = self._sex_biased_model(founder_rate=1.0, sex_bias=0.0, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isnan(result["dest_pop_remainder_pop_sex_bias"])

    def test_sex_biased_nan_when_remainder_rate_one(self):
        """When remainder rate = 1 the sex bias denominator collapses → NaN."""
        # source_rate = 0 → remainder_rate = 1
        model, matrices = self._sex_biased_model(founder_rate=0.0, sex_bias=0.0, found_time=5)
        result = compute_remainder_params(model, matrices)
        assert np.isnan(result["dest_pop_remainder_pop_sex_bias"])

    def test_sex_biased_keys_include_dest_pop(self):
        """Both keys must be prefixed with the destination population name."""
        model, matrices = self._sex_biased_model()
        result = compute_remainder_params(model, matrices)
        assert "dest_pop_remainder_pop_rate" in result
        assert "dest_pop_remainder_pop_sex_bias" in result
        assert "remainder_pop_rate" not in result


class TestFillMissingPopulationsWithZeros:
    """
    Tests for _fill_missing_populations_with_zeros, used to backfill populations with no observed
    tracts before saving/plotting tract length distributions in output_simulation_data_sex_biased.
    The zero-count arrays it inserts must have length n_counts (== len(bins) - 1, matching
    Population.tractlength_histogram's output), not len(bins): a past off-by-one bug here produced
    arrays one element too long, which broke both plotting (mismatched against bin centers) and
    combining female/male allosome data (mismatched array lengths).
    """

    def test_missing_population_filled_with_correct_length(self):
        data = {"EUR": [1.0, 2.0, 3.0]}
        _fill_missing_populations_with_zeros(data, ["EUR", "AFR"], n_counts=3, data_label="autosome data")

        assert data["AFR"] == [0.0, 0.0, 0.0]
        assert len(data["AFR"]) == len(data["EUR"])

    def test_present_population_left_untouched(self):
        data = {"EUR": [1.0, 2.0, 3.0]}
        _fill_missing_populations_with_zeros(data, ["EUR"], n_counts=3, data_label="autosome data")
        assert data["EUR"] == [1.0, 2.0, 3.0]

    def test_prints_message_naming_missing_population_and_label(self, capsys):
        _fill_missing_populations_with_zeros({}, ["AFR"], n_counts=5, data_label="female allosome data")
        captured = capsys.readouterr()
        assert "AFR" in captured.out
        assert "female allosome data" in captured.out

    def test_no_message_when_nothing_missing(self, capsys):
        _fill_missing_populations_with_zeros({"EUR": [0.0]}, ["EUR"], n_counts=1, data_label="autosome data")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestRunWithGenerationZeroWarningReporting:
    """
    Tests for _run_with_generation_zero_warning_reporting, used to catch
    _GenerationZeroContributionWarning raised on individual objective-function evaluations during an
    optimization stage (which would otherwise print once per evaluation) and report it once, as a
    single consolidated message, after the stage completes.
    """

    def test_returns_run_fn_result(self):
        assert _run_with_generation_zero_warning_reporting(lambda: 42) == 42

    def test_prints_one_consolidated_message_with_count(self, capsys):
        def run_fn():
            for _ in range(3):
                warnings.warn("gen0", category=_GenerationZeroContributionWarning)
            return "done"

        result = _run_with_generation_zero_warning_reporting(run_fn)

        assert result == "done"
        captured = capsys.readouterr()
        assert captured.out.count("generation 0") == 1
        assert "3" in captured.out

    def test_no_message_when_warning_never_raised(self, capsys):
        _run_with_generation_zero_warning_reporting(lambda: None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warning_does_not_propagate_past_the_wrapper(self):
        # It must be genuinely caught (not just printed once): callers wrapping this in their own
        # catch_warnings should see nothing escape.
        with warnings.catch_warnings(record=True) as outer:
            warnings.simplefilter("always")
            _run_with_generation_zero_warning_reporting(
                lambda: warnings.warn("gen0", category=_GenerationZeroContributionWarning)
            )
        assert len(outer) == 0

    def test_other_warnings_are_forwarded_to_the_logger(self):
        def run_fn():
            warnings.warn("something unrelated", category=UserWarning)
            return None
        
        with patch.object(driver_utils.logger, "warning") as mock_warning:
            _run_with_generation_zero_warning_reporting(run_fn)

        mock_warning.assert_called_once()
        assert "something unrelated" in mock_warning.call_args.args[0]

    def test_other_warnings_do_not_trigger_generation_zero_message(self, capsys):
        _run_with_generation_zero_warning_reporting(
            lambda: warnings.warn("something unrelated", category=UserWarning)
        )
        captured = capsys.readouterr()
        assert "generation 0" not in captured.out


class TestReportGenerationZeroWarningForOptimalParams:
    """
    Tests for _report_generation_zero_warning_for_optimal_params, used to additionally flag when a
    step's final optimal parameters (not just some evaluations seen during optimization) have source
    populations contributing to the admixed population at generation 0.
    """

    def _make_genetic_model_mock(self, check_result: bool):
        genetic_model = Mock()
        genetic_model.demographic_model.get_migration_matrices.return_value = {"a": np.zeros((2, 2))}
        genetic_model.split_migration_matrices.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
        genetic_model.check_generation_zero_migration_warning.return_value = check_result
        return genetic_model

    def test_prints_message_when_warning_would_fire(self, capsys):
        genetic_model = self._make_genetic_model_mock(check_result=True)
        _report_generation_zero_warning_for_optimal_params(
            genetic_model=genetic_model, optimal_params=np.array([1.0]),
            include_autosomes=True, include_allosomes=False, step_label="Step 1",
        )
        captured = capsys.readouterr()
        assert "Step 1" in captured.out
        assert "generation 0" in captured.out

    def test_no_message_when_warning_would_not_fire(self, capsys):
        genetic_model = self._make_genetic_model_mock(check_result=False)
        _report_generation_zero_warning_for_optimal_params(
            genetic_model=genetic_model, optimal_params=np.array([1.0]),
            include_autosomes=True, include_allosomes=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_message_omits_step_label_when_none(self, capsys):
        genetic_model = self._make_genetic_model_mock(check_result=True)
        _report_generation_zero_warning_for_optimal_params(
            genetic_model=genetic_model, optimal_params=np.array([1.0]),
            include_autosomes=True, include_allosomes=False, step_label=None,
        )
        captured = capsys.readouterr()
        assert "None" not in captured.out

    def test_passes_optimal_params_and_flags_through(self):
        genetic_model = self._make_genetic_model_mock(check_result=False)
        optimal_params = np.array([1.0, 2.0])

        _report_generation_zero_warning_for_optimal_params(
            genetic_model=genetic_model, optimal_params=optimal_params,
            include_autosomes=True, include_allosomes=True,
        )

        genetic_model.demographic_model.get_migration_matrices.assert_called_once_with(optimal_params)
        genetic_model.split_migration_matrices.assert_called_once()
        _, kwargs = genetic_model.check_generation_zero_migration_warning.call_args
        assert kwargs["include_autosomes"] is True
        assert kwargs["include_allosomes"] is True

    def test_swallows_exceptions_instead_of_raising(self, capsys):
        # A stubbed/mocked demographic model (as used in some driver-orchestration tests) may not
        # construct a migration matrix valid enough for real PhT model validation; this check must
        # not be allowed to break the optimization run it is merely reporting on.
        genetic_model = Mock()
        genetic_model.demographic_model.get_migration_matrices.side_effect = Exception("invalid matrix")

        _report_generation_zero_warning_for_optimal_params(
            genetic_model=genetic_model, optimal_params=np.array([1.0]),
            include_autosomes=True, include_allosomes=False,
        )  # must not raise

        captured = capsys.readouterr()
        assert captured.out == ""
