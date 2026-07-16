from tracts.driver_utils import locate_file_path
from tracts.driver_utils import parse_chromosomes, parse_start_params, scale_select_indices, get_time_scaled_model_func, get_time_scaled_model_bounds
from tracts.driver_utils import SamplesConfig, InferenceConfig
from tracts.driver_utils import load_model_from_driver
from tracts.driver_utils import load_population
from tracts.driver_utils import _compute_remainder_params
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from pathlib import Path
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
            model=mock_model
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
            model=mock_model
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
            model=mock_model
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
                model=mock_model
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
            model=mock_model
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
            model=mock_model
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
            model=mock_model
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
                model=mock_model
            )

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


class TestLoadModelFromDriver:
    """
    A class for testing the load_model_from_driver function, which is responsible for loading a demographic model based on specifications provided in a driver file.
    The tests cover scenarios such as successful model loading, handling of missing model filename in the driver specifications, and error handling for cases where the specified model file cannot be found.
    """

    def test_load_model_basic(self, mock_locate, mock_model_class):
        """
        Test basic model loading without sex-bias.
        """
        model_path = Path("/path/to/model.yaml")
        mock_locate.return_value = model_path

        mock_model = Mock()
        mock_model_class.load_from_YAML.return_value = mock_model

        driver_spec = Mock()
        driver_spec.models.model_filename = "model.yaml"
        driver_spec.models.implicit_population = None
        driver_spec.samples.allosomes = None

        result = load_model_from_driver(
            driver_spec,
            script_dir=None,
            driver_path="/path/driver.yaml",
        )

        assert result is mock_model
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
        mock_model_class_sexbiased.load_from_YAML.return_value = mock_model

        driver_spec = Mock()
        driver_spec.models.model_filename = "model.yaml"
        driver_spec.models.implicit_population = None
        driver_spec.samples = Mock()
        driver_spec.samples.allosomes = ["X"]
        driver_spec.samples.male_names = ["ind1"]

        result = load_model_from_driver(driver_spec, script_dir=None,
                                       driver_path="/path/driver.yaml",
                                       allosome_label="X")

        assert result == mock_model
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
        mock_model_class.load_from_YAML.return_value = mock_model

        driver_spec = Mock()
        driver_spec.models.model_filename = "model.yaml"
        driver_spec.models.implicit_population = "AFR"
        driver_spec.samples.allosomes = None

        result = load_model_from_driver(
            driver_spec,
            script_dir=None,
            driver_path="/path/driver.yaml",
        )

        assert result is mock_model
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
            load_model_from_driver(driver_spec, script_dir=None, 
                                  driver_path="/path/driver.yaml")
    

    def test_load_model_file_not_found_raises_error(self, mock_locate):
        """
        Test FileNotFoundError when model file doesn't exist.
        """
        mock_locate.return_value = None
        
        driver_spec = Mock()
        driver_spec.models.model_filename = "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_model_from_driver(driver_spec, script_dir=None, 
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
    Tests for _compute_remainder_params, which extracts the founding rate (and,
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
        result = _compute_remainder_params(SimpleNamespace(), {})
        assert result == {}

    # ------------------------------------------------------------------
    # Plain (non-sex-biased) model
    # ------------------------------------------------------------------

    def test_plain_basic_rate(self):
        """Remainder rate = 1 - source_rate is read from the founding row."""
        model, matrices = self._plain_model(founder_rate=0.3, found_time=5)
        result = _compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 0.7)

    def test_plain_rate_zero(self):
        """When source occupies 100 %, remainder rate = 0."""
        model, matrices = self._plain_model(founder_rate=1.0, found_time=5)
        result = _compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 0.0)

    def test_plain_rate_one(self):
        """When source contributes 0 %, remainder rate = 1."""
        model, matrices = self._plain_model(founder_rate=0.0, found_time=5)
        result = _compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_rate"], 1.0)

    def test_plain_no_sex_bias_key(self):
        """Non-sex-biased models must not produce a sex_bias key."""
        model, matrices = self._plain_model()
        result = _compute_remainder_params(model, matrices)
        assert not any("sex_bias" in k for k in result)

    def test_plain_key_includes_dest_pop(self):
        """Key must be '{dest_pop}_{remainder_pop}_rate', not just '{remainder_pop}_rate'."""
        model, matrices = self._plain_model()
        result = _compute_remainder_params(model, matrices)
        assert "dest_pop_remainder_pop_rate" in result
        assert "remainder_pop_rate" not in result

    def test_plain_duplicate_in_parametrized_populations(self):
        """A population listed twice is processed only once (no duplicate keys)."""
        model, matrices = self._plain_model()
        model.parametrized_populations = ["dest_pop", "dest_pop"]
        result = _compute_remainder_params(model, matrices)
        assert list(result.keys()).count("dest_pop_remainder_pop_rate") == 1

    def test_plain_empty_parametrized_populations(self):
        """Empty parametrized_populations → empty result."""
        model, matrices = self._plain_model()
        model.parametrized_populations = []
        result = _compute_remainder_params(model, matrices)
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
        result = _compute_remainder_params(model, matrices)
        assert result == {}

    # ------------------------------------------------------------------
    # Sex-biased model
    # ------------------------------------------------------------------

    def test_sex_biased_rate_value(self):
        """Remainder rate = mean of male and female founding rates = 1 - source_rate."""
        model, matrices = self._sex_biased_model(founder_rate=0.3, sex_bias=0.0, found_time=5)
        result = _compute_remainder_params(model, matrices)
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
        result = _compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_sex_bias"], -0.5)

    def test_sex_biased_zero_sex_bias(self):
        """Zero source sex bias → remainder sex bias is also 0."""
        model, matrices = self._sex_biased_model(founder_rate=0.4, sex_bias=0.0, found_time=5)
        result = _compute_remainder_params(model, matrices)
        assert np.isclose(result["dest_pop_remainder_pop_sex_bias"], 0.0)

    def test_sex_biased_nan_when_remainder_rate_zero(self):
        """When remainder rate = 0 the sex bias denominator collapses → NaN."""
        # source_rate = 1 → remainder_rate = 0
        model, matrices = self._sex_biased_model(founder_rate=1.0, sex_bias=0.0, found_time=5)
        result = _compute_remainder_params(model, matrices)
        assert np.isnan(result["dest_pop_remainder_pop_sex_bias"])

    def test_sex_biased_nan_when_remainder_rate_one(self):
        """When remainder rate = 1 the sex bias denominator collapses → NaN."""
        # source_rate = 0 → remainder_rate = 1
        model, matrices = self._sex_biased_model(founder_rate=0.0, sex_bias=0.0, found_time=5)
        result = _compute_remainder_params(model, matrices)
        assert np.isnan(result["dest_pop_remainder_pop_sex_bias"])

    def test_sex_biased_keys_include_dest_pop(self):
        """Both keys must be prefixed with the destination population name."""
        model, matrices = self._sex_biased_model()
        result = _compute_remainder_params(model, matrices)
        assert "dest_pop_remainder_pop_rate" in result
        assert "dest_pop_remainder_pop_sex_bias" in result
        assert "remainder_pop_rate" not in result
        assert "remainder_pop_sex_bias" not in result                        