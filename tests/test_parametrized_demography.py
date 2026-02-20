import os
import tempfile
import numpy as np

import pytest

from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parameter import ParamType


@pytest.fixture
def basic_model():
    """Fixture that provides a basic ParametrizedDemography model."""
    return ParametrizedDemography()


@pytest.fixture
def custom_time_model():
    """Fixture that provides a model with custom time bounds."""
    return ParametrizedDemography(min_time=5, max_time=100)


@pytest.fixture
def named_model():
    """Fixture that provides a model with a custom name."""
    return ParametrizedDemography(name="TestModel")


@pytest.fixture
def complete_model():
    """Fixture that provides a model with all initialization parameters specified."""
    return ParametrizedDemography(
        name="CompleteModel",
        min_time=10,
        max_time=200
    )


@pytest.fixture
def model_with_founder_event(basic_model):
    """Fixture that provides a model with a founder event."""
    basic_model.add_founder_event("destination_pop", {"source_pop1": "founder_rate1"}, "source_pop2", "found_time")
    return basic_model

@pytest.fixture
def model_with_continuous_founder_event(basic_model):
    """Fixture that provides a model with a continuous founder event."""
    basic_model.add_founder_event("destination_pop", {"source_pop1": "founder_rate1", "source_pop2": "founder_rate2"}, None, "found_time", end_time="end_time")
    return basic_model


@pytest.fixture
def model_with_pulse_migration(model_with_founder_event):
    """Fixture that provides a model with a founder event and pulse migration."""
    model_with_founder_event.add_pulse_migration("destination_pop", "source_pop1", "rate1", "time1")
    return model_with_founder_event


@pytest.fixture
def model_with_continuous_migration(model_with_founder_event):
    """Fixture that provides a model with a founder event and continuous migration."""
    model_with_founder_event.add_continuous_migration("destination_pop", "source_pop1", "rate1", "start1", "end1")
    return model_with_founder_event


@pytest.fixture
def model_with_both_migrations(model_with_founder_event):
    """Fixture that provides a model with a founder event and both types of migrations."""
    model_with_founder_event.add_pulse_migration("destination_pop", "source_pop1", "rate1", "time1")
    model_with_founder_event.add_continuous_migration("destination_pop", "source_pop2", "rate2", "start1", "end1")
    return model_with_founder_event


@pytest.fixture
def model_with_multiple_populations(basic_model):
    """Fixture that provides a model with multiple populations and migrations."""
    # Add founder events for destination populations
    basic_model.add_founder_event("dest_pop1", {"source_pop1": "founder_rate1"}, "source_pop2", "found_time1")
    basic_model.add_founder_event("dest_pop2", {"source_pop2": "founder_rate2"}, "source_pop1", "found_time2")
    
    # Add migrations between populations
    basic_model.add_pulse_migration("dest_pop1", "source_pop2", "rate1", "time1")
    basic_model.add_continuous_migration("dest_pop2", "source_pop1", "rate2", "start1", "end1")
    
    return basic_model


def test_initialization(basic_model):
    """Test that a new instance initializes correctly with default parameters"""
    assert basic_model.name == ""
    assert basic_model.min_time == 2
    assert basic_model.max_time == np.inf
    assert len(basic_model.free_params) == 0
    assert len(basic_model.dependent_params) == 0
    assert len(basic_model.population_indices) == 0
    assert basic_model.finalized is False
    assert basic_model.founder_events == {}


def test_custom_time_bounds(custom_time_model):
    """Test initialization with custom min_time and max_time values"""
    assert custom_time_model.min_time == 5
    assert custom_time_model.max_time == 100


def test_name_assignment(named_model):
    """Test that the model name is correctly assigned"""
    assert named_model.name == "TestModel"




def test_initialization_with_all_params(complete_model):
    """Test initialization with all parameters specified"""
    assert complete_model.name == "CompleteModel"
    assert complete_model.min_time == 10
    assert complete_model.max_time == 200
    assert len(complete_model.free_params) == 0
    assert len(complete_model.dependent_params) == 0
    assert len(complete_model.population_indices) == 0
    assert complete_model.finalized is False
    assert complete_model.founder_events == {}


def test_add_parameter(basic_model):
    """Test adding different types of parameters"""
    # Test adding rate parameter
    basic_model.add_parameter("rate1", ParamType.RATE)
    assert "rate1" in basic_model.free_params
    assert basic_model.free_params["rate1"].type == ParamType.RATE
    assert basic_model.free_params["rate1"].bounds == (0, 1)
    
    # Test adding time parameter
    basic_model.add_parameter("time1", ParamType.TIME)
    assert "time1" in basic_model.free_params
    assert basic_model.free_params["time1"].type == ParamType.TIME
    assert basic_model.free_params["time1"].bounds == (2, np.inf)


def test_parameter_bounds(custom_time_model):
    """Test that parameters are created with correct bounds"""
    # Test custom bounds for rate parameter
    custom_time_model.add_parameter("rate1", ParamType.RATE, bounds=(0.1, 0.5))
    assert custom_time_model.free_params["rate1"].bounds == (0.1, 0.5)
    
    # Test custom bounds for time parameter
    custom_time_model.add_parameter("time1", ParamType.TIME, bounds=(10, 50))
    assert custom_time_model.free_params["time1"].bounds == (10, 50)


def test_add_population(basic_model):
    """Test adding populations to the model"""
    # Add a single population
    basic_model.add_population("pop1")
    assert "pop1" in basic_model.population_indices
    assert basic_model.population_indices["pop1"] is None  # Index should be None before finalization
    
    # Add multiple populations
    populations = ["pop2", "pop3", "pop4"]
    for pop in populations:
        basic_model.add_population(pop)
        assert pop in basic_model.population_indices
        assert basic_model.population_indices[pop] is None
    
    # Verify finalization assigns indices
    basic_model.finalize()
    for pop in populations:
        assert basic_model.population_indices[pop] is not None
    
    # Verify indices are unique
    indices = list(basic_model.population_indices.values())
    assert len(indices) == len(set(indices))


#def test_population_after_fix(basic_model):
#    """Test that adding populations after fixing proportions raises appropriate errors"""
#    """IMPORTANT: This test does not reflect ideal behaviour.
#    In the future, the model should require that each key in the proportions dict has an associated founder event.
#    This is not the case in this test."""
#    # Add initial populations
#    basic_model.add_population("pop1")
#    basic_model.add_population("pop2")
#
#    basic_model.add_parameter("rate1", ParamType.RATE)
#    
#    # Fix proportions
#    basic_model.fixed_proportions_handler.fix_ancestry_proportions(basic_model, ["rate1"], {"pop1": [0.7, 0.3]})
#    
#    # Test adding population after fixing proportions
#    with pytest.raises(ValueError):
#        basic_model.add_population("pop3")

def test_continuous_founder_event(model_with_continuous_founder_event):
    assert "found_time" in model_with_continuous_founder_event.free_params
    assert "end_time" in model_with_continuous_founder_event.free_params
    assert "founder_rate1" in model_with_continuous_founder_event.free_params
    assert "founder_rate2" in model_with_continuous_founder_event.free_params


def test_add_pulse_migration(model_with_pulse_migration):
    """Test adding pulse migrations"""
    # Verify parameters were added
    assert "rate1" in model_with_pulse_migration.free_params
    assert "time1" in model_with_pulse_migration.free_params
    assert "founder_rate1" in model_with_pulse_migration.free_params
    assert "found_time" in model_with_pulse_migration.free_params
    
    # Verify events were added
    model_with_pulse_migration.finalize()
    assert "destination_pop" in model_with_pulse_migration.events
    assert len(model_with_pulse_migration.events["destination_pop"]) == 1


def test_add_continuous_migration(model_with_continuous_migration):
    """Test adding continuous migrations"""
    # Verify parameters were added
    assert "rate1" in model_with_continuous_migration.free_params
    assert "start1" in model_with_continuous_migration.free_params
    assert "end1" in model_with_continuous_migration.free_params
    assert "founder_rate1" in model_with_continuous_migration.free_params
    assert "found_time" in model_with_continuous_migration.free_params
    
    # Verify events were added
    model_with_continuous_migration.finalize()
    assert "destination_pop" in model_with_continuous_migration.events
    assert len(model_with_continuous_migration.events["destination_pop"]) == 1


def test_migration_matrix_basic(model_with_both_migrations):
    """Test that migration matrices are generated correctly for a basic model"""
    # Finalize the model
    model_with_both_migrations.finalize()
    
    # Test parameter evaluation
    test_params = [0.4, 10, 0.3, 3, 0.2, 7, 5]  # founder_rate1, found_time, rate1, time1, rate2, start1, end1
    
    # Get migration matrices
    migration_matrices = model_with_both_migrations.get_migration_matrices(test_params)
    
    # Verify matrices were created
    assert "destination_pop" in migration_matrices
    
    # Verify matrix dimensions
    assert migration_matrices["destination_pop"].shape[0] == 11  # max(time1, end1) + 1
    assert migration_matrices["destination_pop"].shape[1] == 2  # number of populations
    
    # Verify founder rates are applied correctly
    founder_rate = model_with_both_migrations.get_param_value("founder_rate1", test_params)
    
    # Check that founder rates are applied correctly at the founding time
    assert np.isclose(migration_matrices["destination_pop"][10, 0], founder_rate)
    
    # Verify that rates sum to 1 at the founding time
    assert np.isclose(migration_matrices["destination_pop"][10, 0] + migration_matrices["destination_pop"][10, 1], 1)
    
    # Verify pulse migration is applied correctly
    pulse_rate = model_with_both_migrations.get_param_value("rate1", test_params)
    
    # Check that pulse migration is applied correctly at the pulse time
    assert np.isclose(migration_matrices["destination_pop"][3, 0], pulse_rate)
    
    # Verify continuous migration is applied correctly
    continuous_rate = model_with_both_migrations.get_param_value("rate2", test_params)
    
    # Check that continuous migration is applied correctly during the migration period
    for t in range(5, 7):  # From start1 to end1
        assert np.isclose(migration_matrices["destination_pop"][t, 1], continuous_rate)


def test_migration_matrix_multiple_populations(model_with_multiple_populations):
    """Test that migration matrices are generated correctly for multiple populations"""
    # Finalize the model
    model_with_multiple_populations.finalize()
    
    # Test parameter evaluation
    test_params = [0.4, 10, 0.5, 12, 0.3, 8, 0.6, 7, 5]  # founder_rate1, found_time1, founder_rate2, found_time2, rate1, time1, rate2, start1, end1
    
    # Get migration matrices
    migration_matrices = model_with_multiple_populations.get_migration_matrices(test_params)
    
    # Verify matrices were created for both populations
    assert "dest_pop1" in migration_matrices
    assert "dest_pop2" in migration_matrices
    
    # Verify matrix dimensions
    assert migration_matrices["dest_pop1"].shape[0] == 11  # max(found_time1, found_time2, time1, end1) + 1
    assert migration_matrices["dest_pop1"].shape[1] == 2  # number of populations
    assert migration_matrices["dest_pop2"].shape[0] == 13
    assert migration_matrices["dest_pop2"].shape[1] == 2
    
    # Verify founder rates are applied correctly for dest_pop1
    founder_rate1 = model_with_multiple_populations.get_param_value("founder_rate1", test_params)
    
    # Check that founder rates are applied correctly at the founding time for dest_pop1
    assert np.isclose(migration_matrices["dest_pop1"][10, 0], founder_rate1)
    
    # Verify founder rates are applied correctly for dest_pop2
    founder_rate2 = model_with_multiple_populations.get_param_value("founder_rate2", test_params)
    
    # Check that founder rates are applied correctly at the founding time for dest_pop2
    assert np.isclose(migration_matrices["dest_pop2"][12, 1], founder_rate2)
    
    # Verify pulse migration is applied correctly
    pulse_rate = model_with_multiple_populations.get_param_value("rate1", test_params)
    
    # Check that pulse migration is applied correctly at the pulse time
    assert np.isclose(migration_matrices["dest_pop1"][8, 1], pulse_rate)
    
    # Verify continuous migration is applied correctly
    continuous_rate = model_with_multiple_populations.get_param_value("rate2", test_params)
    
    # Check that continuous migration is applied correctly during the migration period
    for t in range(5, 7):  # From start1 to end1
        assert np.isclose(migration_matrices["dest_pop2"][t, 0], continuous_rate)


def test_error_migration_without_founder(basic_model):
    """Test that adding migrations without a founder event raises appropriate errors"""
    # Try to add a pulse migration without a founder event
    with pytest.raises(ValueError):
        basic_model.add_pulse_migration("dest_pop", "source_pop", "rate1", "time1")
    
    # Try to add a continuous migration without a founder event
    with pytest.raises(ValueError):
        basic_model.add_continuous_migration("dest_pop", "source_pop", "rate1", "start1", "end1")


def test_error_duplicate_founder_event(basic_model):
    """Test that adding duplicate founder events raises appropriate errors"""
    # Add a founder event
    basic_model.add_founder_event("dest_pop", {"source_pop1": "founder_rate1"}, "source_pop2", "found_time")
    
    # Try to add another founder event for the same population
    with pytest.raises(ValueError):
        basic_model.add_founder_event("dest_pop", {"source_pop3": "founder_rate2"}, "source_pop4", "found_time2")


def test_error_invalid_founder_event(basic_model):
    """Test that adding founder events with invalid parameters raises appropriate errors"""
    # Try to add a founder event with an empty source populations dict
    with pytest.raises(ValueError):
        basic_model.add_founder_event("dest_pop", {}, "source_pop", "found_time")    


def test_error_invalid_parameter_values(model_with_both_migrations):
    """Test that evaluating parameters with invalid values raises appropriate errors"""
    # Finalize the model
    model_with_both_migrations.finalize()
    
    # Try to evaluate with too few parameters
    with pytest.raises(ValueError):
        model_with_both_migrations.get_migration_matrices([0.4, 10])
    
    # Try to evaluate with too many parameters
    with pytest.raises(ValueError):
        model_with_both_migrations.get_migration_matrices([0.4, 10, 0.3, 5, 0.2, 3, 7, 0.5])


def test_error_invalid_yaml_file():
    """Test that loading non-existent YAML files raises appropriate errors"""
    # Try to load a non-existent YAML file
    with pytest.raises(FileNotFoundError):
        ParametrizedDemography.load_from_YAML("non_existent_file.yaml")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to load a YAML file from the temporary directory
        with pytest.raises(FileNotFoundError):
            ParametrizedDemography.load_from_YAML(os.path.join(temp_dir, "non_existent_file.yaml"))


def test_non_integer_founding_time():
    """
    Test that non-integer founding times are handled correctly.
    
    This test:
    1. Creates a model with a founding event
    2. Tests it with both integer and non-integer founding times
    3. Verifies that the non-integer model has non-zero rates at both the nearest integers to the founding time
    4. Verifies that the final proportions match those of the integer model
    """
    # Create a model
    model = ParametrizedDemography(name="TestModel")
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    model.finalize()
    
    # Define parameters for integer founding time (10)
    params_int = [0.7, 10]  # [founder_rate1, found_time]
    
    # Define parameters for non-integer founding time (10.5)
    params_non_int = [0.7, 10.5]  # [founder_rate1, found_time]
    
    # Get migration matrices for both parameter sets
    migration_matrices_int = model.get_migration_matrices(params_int)
    migration_matrices_non_int = model.get_migration_matrices(params_non_int)
    
    # Get the matrices for the target population
    matrix_int = migration_matrices_int["target_pop"]
    matrix_non_int = migration_matrices_non_int["target_pop"]
    
    # Verify matrix dimensions
    assert matrix_non_int.shape[0] == 12  # ceil(found_time) + 1
    assert matrix_non_int.shape[1] == 2  # two source populations
    
    # Verify that the non-integer model has non-zero rates at both the nearest integers to the founding time
    # At time 10 (floor(10.5))
    assert matrix_non_int[10, 0]>0  # source_pop1 proportion at floor(founding time)
    assert matrix_non_int[10, 1]>0  # source_pop2 proportion at floor(founding time)
    # At time 11 (ceil(10.5))
    assert matrix_non_int[11, 0]>0  # source_pop1 proportion at ceil(founding time)
    assert matrix_non_int[11, 1]>0  # source_pop2 proportion at ceil(founding time)
    
    # Verify that the final proportions match
    final_proportions_int = model.proportions_from_matrix(matrix_int)
    final_proportions_non_int = model.proportions_from_matrix(matrix_non_int)
    
    assert np.isclose(final_proportions_int[0], final_proportions_non_int[0])  # source_pop1 proportion for integer model
    assert np.isclose(final_proportions_int[1], final_proportions_non_int[1])  # source_pop2 proportion for integer model
    
    # Verify that the sum of proportions is 1 for each model
    assert np.isclose(final_proportions_int.sum(), 1.0)
    assert np.isclose(final_proportions_non_int.sum(), 1.0)


def test_non_integer_founding_time_continuous_founder():
    """
    Test that non-integer founding times are handled correctly.
    
    This test:
    1. Creates a model with a founding event
    2. Tests it with both integer and non-integer founding times
    3. Verifies that the non-integer model has non-zero rates at both the nearest integers to the founding time
    4. Verifies that the final proportions match those of the integer model
    """
    # Create a model
    model = ParametrizedDemography(name="TestModel")
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1", "source_pop2": "founder_rate2"},
        remainder_population=None,
        found_time="found_time",
        end_time = "end_time"
    )
    model.finalize()
    small = 1e-9
    # Define parameters for integer founding time (10)
    params_int = [0.1, 0.2, 10-small, 5]  # [founder_rate1, founder_rate2, found_time, end_time]
    
    # Define parameters for non-integer founding time (10.5)
    params_non_int = [0.1, 0.2, 10+small, 5]  # [founder_rate1, found_time, end_time]
    
    # Get migration matrices for both parameter sets
    migration_matrices_int = model.get_migration_matrices(params_int)
    migration_matrices_non_int = model.get_migration_matrices(params_non_int)

    # Get the matrices for the target population
    matrix_int = migration_matrices_int["target_pop"]
    matrix_non_int = migration_matrices_non_int["target_pop"]
    
    # Verify matrix dimensions
    assert matrix_non_int.shape[0] == 12  # ceil(found_time) + 1
    assert matrix_non_int.shape[1] == 2  # two source populations
    
    # Verify that the non-integer model has non-zero rates at both the nearest integers to the founding time
    # At time 10 (floor(10.5))
    assert matrix_non_int[10, 0]>0  # source_pop1 proportion at floor(founding time)
    assert matrix_non_int[10, 1]>0  # source_pop2 proportion at floor(founding time)
    # At time 11 (ceil(10.5))
    assert matrix_non_int[11, 0]>0  # source_pop1 proportion at ceil(founding time)
    assert matrix_non_int[11, 1]>0  # source_pop2 proportion at ceil(founding time)
    
    assert np.isclose(matrix_non_int[11,:].sum(), 1) # founding generation should sum up to one. 
    assert np.isclose(matrix_int[10,:].sum(), 1) # founding generation should sum up to one. 

    assert np.sum(np.linalg.norm(matrix_non_int[4:11,:]- matrix_int[4:11,:])**2)<0.0001

    # Verify that the final proportions match
    final_proportions_int = model.proportions_from_matrix(matrix_int)
    final_proportions_non_int = model.proportions_from_matrix(matrix_non_int)
    
    assert np.isclose(final_proportions_int[0], final_proportions_non_int[0])  # source_pop1 proportion for integer model
    assert np.isclose(final_proportions_int[1], final_proportions_non_int[1])  # source_pop2 proportion for integer model
    
    # Verify that the sum of proportions is 1 for each model
    assert np.isclose(final_proportions_int.sum(), 1.0)
    assert np.isclose(final_proportions_non_int.sum(), 1.0)


def test_non_integer_pulse_time(model_with_pulse_migration):
    """
    Test that non-integer pulse times are handled correctly.
    
    This test:
    1. Uses a model with a founding event and a pulse migration
    2. Tests it with both integer and non-integer pulse times
    3. Verifies that the non-integer model has non-zero rates at both the nearest integers to the pulse time
    4. Verifies that the final proportions match those of the integer model
    """
    # Finalize the model
    model_with_pulse_migration.finalize()
    
    # Define parameters for integer pulse time (15)
    params_int = [0.7, 10, 0.3, 5]  # [founder_rate1, found_time, pulse_rate, pulse_time]
    
    # Define parameters for non-integer pulse time (15.5)
    params_non_int = [0.7, 10, 0.3, 5.5]  # [founder_rate1, found_time, pulse_rate, pulse_time]
    
    # Get migration matrices for both parameter sets
    migration_matrices_int = model_with_pulse_migration.get_migration_matrices(params_int)
    migration_matrices_non_int = model_with_pulse_migration.get_migration_matrices(params_non_int)
    
    # Get the matrices for the target population
    matrix_int = migration_matrices_int["destination_pop"]
    matrix_non_int = migration_matrices_non_int["destination_pop"]

    # Verify that the non-integer model has non-zero rates at both the nearest integers to the pulse time
    assert matrix_non_int[5, 0]>0  # source_pop1 proportion at floor(pulse time)
    assert matrix_non_int[6, 0]>0  # source_pop2 proportion at floor(pulse time)

    # Verify that the final proportions match
    final_proportions_int = model_with_pulse_migration.proportions_from_matrix(matrix_int)
    final_proportions_non_int = model_with_pulse_migration.proportions_from_matrix(matrix_non_int)
    
    # The final proportions should be the same for both models
    assert np.isclose(final_proportions_int[0], final_proportions_non_int[0])  # source_pop1 proportion for integer model
    assert np.isclose(final_proportions_int[1], final_proportions_non_int[1])  # source_pop2 proportion for integer model
    
    # Verify that the sum of proportions is 1 for each model
    assert np.isclose(final_proportions_int.sum(), 1.0)
    assert np.isclose(final_proportions_non_int.sum(), 1.0)


def test_non_integer_continuous_migration_time(model_with_continuous_migration):
    """
    Test that non-integer continuous migration times are handled correctly.
    
    This test:
    1. Uses a model with a founding event and a continuous migration
    2. Tests it with both integer and non-integer continuous migration times
    3. Verifies that the non-integer model has non-zero rates at both the nearest integers to the migration times
    4. Verifies that the final proportions match those of the integer model
    """
    # Finalize the model
    model_with_continuous_migration.finalize()
    
    # Define parameters for integer migration times (start=12, end=15)
    params_int = [0.7, 10, 0.1, 6, 3]  # [founder_rate1, found_time, migration_rate, start_time, end_time]
    
    # Define parameters for non-integer migration times (start=12.5, end=15.5)
    params_non_int = [0.7, 10, 0.1, 6.01, 3.01]  # [founder_rate1, found_time, migration_rate, start_time, end_time]
    
    # Get migration matrices for both parameter sets
    migration_matrices_int = model_with_continuous_migration.get_migration_matrices(params_int)
    migration_matrices_non_int = model_with_continuous_migration.get_migration_matrices(params_non_int)
    
    # Get the matrices for the target population
    matrix_int = migration_matrices_int["destination_pop"]
    matrix_non_int = migration_matrices_non_int["destination_pop"]

    
    # Verify that the non-integer model has non-zero rates at both the nearest integers to the migration times
    assert matrix_non_int[4, 0]>0  # source_pop1 proportion at floor(start time)
    assert matrix_non_int[7, 0]>0  # source_pop2 proportion at floor(start time)
    
    # Verify that the final proportions match
    final_proportions_int = model_with_continuous_migration.proportions_from_matrix(matrix_int)
    final_proportions_non_int = model_with_continuous_migration.proportions_from_matrix(matrix_non_int)
    
    # The final proportions should be the same for both models
    assert np.isclose(final_proportions_int[0], final_proportions_non_int[0], atol=1e-4)  # source_pop1 proportion for integer model
    assert np.isclose(final_proportions_int[1], final_proportions_non_int[1], atol=1e-4)  # source_pop2 proportion for integer model
    
    # Verify that the sum of proportions is 1 for each model
    assert np.isclose(final_proportions_int.sum(), 1.0)
    assert np.isclose(final_proportions_non_int.sum(), 1.0)


