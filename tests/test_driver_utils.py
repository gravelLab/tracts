from tracts.driver_utils import locate_file_path
from tracts.driver_utils import parse_chromosomes, parse_start_params, scale_select_indices, get_time_scaled_model_func, get_time_scaled_model_bounds
from tracts.driver_utils import SamplesConfig, InferenceConfig
from tracts.driver_utils import load_model_from_driver
from tracts.driver_utils import load_population
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

        start_bounds = Mock(param1=0.5)

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=1,
            seed=42,
            model=mock_model
        )

        assert len(result) == 1
        assert mock_model.get_violation_score.call_count == 2

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

        start_bounds = Mock(param1=0.5)

        result = parse_start_params(
            start_param_bounds=start_bounds,
            repetitions=1,
            seed=42,
            model=mock_model
        )

        assert len(result) == 1
        assert mock_model.get_violation_score.call_count == 2

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
            'model_filename': 'model.yaml',
            'start_params': {},
            'seed': 42,
            'output_filename_format': 'output_{label}.txt'
        }
        config = InferenceConfig(**config_dict)
        assert config.model_filename == 'model.yaml'
        assert config.seed == 42
        assert config.output_filename_format == 'output_{label}.txt'
        assert config.npts == 50
        assert config.exclude_tracts_below_cm == 1
        assert config.log_scale is True
        assert config.repetitions == 1

        
    def test_inference_config_forbids_extra_fields(self):
        """
        Test that InferenceConfig forbids extra fields.
        """
        with pytest.raises(Exception):
            InferenceConfig(
                samples=Mock(),
                model_filename='model.yaml',
                start_params={},
                seed=42,
                output_filename_format='output_{label}.txt',
                extra_field="should_fail"
            )
    
    def test_samples_config_missing_required_field(self):
        """
        Test SamplesConfig with missing required field.
        """
        with pytest.raises(Exception):  # ValidationError
            SamplesConfig(individual_names=["ind1"])


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
        driver_spec.model_filename = "model.yaml"
        driver_spec.samples.allosomes = None

        result = load_model_from_driver(
            driver_spec,
            script_dir=None,
            driver_path="/path/driver.yaml",
        )

        assert result is mock_model
        mock_model_class.load_from_YAML.assert_called_once_with(
            str(model_path.resolve())
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
        driver_spec.model_filename = "model.yaml"
        driver_spec.samples = Mock()
        driver_spec.samples.allosomes = ["X"]
        driver_spec.samples.male_names = ["ind1"]

        result = load_model_from_driver(driver_spec, script_dir=None,
                                       driver_path="/path/driver.yaml",
                                       allosome_label="X")

        assert result == mock_model
        mock_model_class_sexbiased.load_from_YAML.assert_called_once_with(
            str(model_path.resolve())
        )
        mock_model_class.load_from_YAML.assert_not_called()
    

    def test_load_model_missing_filename_raises_error(self):
        """
        Test ValueError when model_filename not specified.
        """
        driver_spec = Mock(spec=[])  # No model_filename attribute
        
        with pytest.raises(ValueError, match="model_filename"):
            load_model_from_driver(driver_spec, script_dir=None, 
                                  driver_path="/path/driver.yaml")
    

    def test_load_model_file_not_found_raises_error(self, mock_locate):
        """
        Test FileNotFoundError when model file doesn't exist.
        """
        mock_locate.return_value = None
        
        driver_spec = Mock()
        driver_spec.model_filename = "nonexistent.yaml"
        
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