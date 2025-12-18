import pytest
import numpy as np
import os
import tempfile
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parameter import ParamType

# Basic model fixtures
@pytest.fixture
def basic_model():
    """Fixture that provides a basic ParametrizedDemographySexBiased model."""
    return ParametrizedDemographySexBiased()

@pytest.fixture
def custom_time_model():
    """Fixture that provides a model with custom time bounds."""
    return ParametrizedDemographySexBiased(min_time=5, max_time=100)

@pytest.fixture
def named_model():
    """Fixture that provides a model with a custom name."""
    return ParametrizedDemographySexBiased(name="TestModel")

@pytest.fixture
def complete_model():
    """Fixture that provides a model with all initialization parameters specified."""
    return ParametrizedDemographySexBiased(
        name="CompleteModel",
        min_time=10,
        max_time=200
    )

# Sex bias parameter fixtures
@pytest.fixture
def model_with_sex_bias_param(basic_model):
    """Fixture that provides a model with a sex bias parameter."""
    basic_model.add_parameter("bias1", ParamType.SEX_BIAS)
    return basic_model

@pytest.fixture
def model_with_custom_bounds_sex_bias_param(custom_time_model):
    """Fixture that provides a model with a sex bias parameter with custom bounds."""
    custom_time_model.add_parameter("bias1", ParamType.SEX_BIAS, bounds=(-0.5, 0.5))
    return custom_time_model

@pytest.fixture
def model_with_dependent_params(basic_model):
    """Fixture that provides a model with sex-biased dependent parameters."""
    # Add base rate parameter
    basic_model.add_parameter("rate1", ParamType.RATE)
    
    # Add sex bias parameter
    basic_model.add_parameter("rate1_sex_bias", ParamType.SEX_BIAS)
    
    # Add dependent parameters for male and female
    def male_expression(demography, params):
        return demography.get_param_value("rate1", params) * (1 - demography.get_param_value("rate1_sex_bias", params))
    
    def female_expression(demography, params):
        return demography.get_param_value("rate1", params) * (1 + demography.get_param_value("rate1_sex_bias", params))
    
    basic_model.add_dependent_parameter("rate1_male", male_expression, ParamType.RATE)
    basic_model.add_dependent_parameter("rate1_female", female_expression, ParamType.RATE)
    
    return basic_model

@pytest.fixture
def model_with_population_and_sex_bias(basic_model):
    """Fixture that provides a model with a population and sex-specific parameters."""
    # Add base population
    basic_model.add_population("pop1")
    
    # Add sex-specific parameters
    basic_model.add_parameter("rate1", ParamType.RATE)
    basic_model.add_parameter("rate1_sex_bias", ParamType.SEX_BIAS)
    
    # Add sex-specific dependent parameters
    def male_expression(demography, params):
        return demography.get_param_value("rate1", params) * (1 - demography.get_param_value("rate1_sex_bias", params))
    
    def female_expression(demography, params):
        return demography.get_param_value("rate1", params) * (1 + demography.get_param_value("rate1_sex_bias", params))
    
    basic_model.add_dependent_parameter("rate1_male", male_expression, ParamType.RATE)
    basic_model.add_dependent_parameter("rate1_female", female_expression, ParamType.RATE)
    
    return basic_model

# Founder event fixtures
@pytest.fixture
def model_with_founder_event(basic_model):
    """Fixture that provides a model with a founder event."""
    basic_model.add_founder_event("destination_pop", {"source_pop1": "founder_rate1"}, "source_pop2", "found_time")
    return basic_model



@pytest.fixture
def model_with_founder_event_and_params(model_with_founder_event):
    """Fixture that provides a model with a founder event and its parameters."""
    model_with_founder_event.add_parameter("founder_rate1", ParamType.RATE)
    model_with_founder_event.add_parameter("founder_rate1_sex_bias", ParamType.SEX_BIAS)
    return model_with_founder_event

@pytest.fixture
def model_with_continuous_founder_event(basic_model):
    """Fixture that provides a model with a founder event."""
    basic_model.add_founder_event("destination_pop", {"source_pop1": "founder_rate1","source_pop2":"founder_rate2"}, None, "found_time", end_time = "end_time")
    basic_model.add_parameter("founder_rate1", ParamType.RATE)
    basic_model.add_parameter("founder_rate2", ParamType.RATE)
    basic_model.add_parameter("founder_rate1_sex_bias", ParamType.SEX_BIAS)
    basic_model.add_parameter("founder_rate2_sex_bias", ParamType.SEX_BIAS)
    return basic_model

# Migration fixtures
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

# Test functions using fixtures
def test_add_sex_bias_parameter(model_with_sex_bias_param):
    """Test adding different types of parameters"""
    # Test adding sex bias parameter
    assert "bias1" in model_with_sex_bias_param.free_params
    assert model_with_sex_bias_param.free_params["bias1"].type == ParamType.SEX_BIAS
    assert model_with_sex_bias_param.free_params["bias1"].bounds == (-1, 1)

def test_sex_bias_parameter_bounds(model_with_custom_bounds_sex_bias_param):
    """Test that parameters are created with correct bounds"""
    # Test custom bounds for sex bias parameter
    assert model_with_custom_bounds_sex_bias_param.free_params["bias1"].bounds == (-0.5, 0.5)

def test_dependent_parameter_creation(model_with_dependent_params):
    """Test creation of sex-biased dependent parameters"""
    # Test dependent parameters
    assert "rate1_male" in model_with_dependent_params.dependent_params
    assert "rate1_female" in model_with_dependent_params.dependent_params

    model_with_dependent_params.finalize()

    # Test parameter evaluation
    test_params = [0.5, 0.3]  # rate1 = 0.5, sex_bias = 0.3
    male_rate = model_with_dependent_params.dependent_params["rate1_male"](model_with_dependent_params, test_params)
    female_rate = model_with_dependent_params.dependent_params["rate1_female"](model_with_dependent_params, test_params)
    
    assert np.isclose(male_rate, 0.5 * (1 - 0.3))
    assert np.isclose(female_rate, 0.5 * (1 + 0.3))

def test_population_sex_specific(model_with_population_and_sex_bias):
    """Test that populations are correctly handled for sex-specific operations"""
    # Finalize the model
    model_with_population_and_sex_bias.finalize()
    
    # Verify population indices are assigned
    assert model_with_population_and_sex_bias.population_indices["pop1"] is not None
    
    # Verify sex-specific parameters are correctly associated with the population
    test_params = [0.5, 0.3]  # rate1 = 0.5, sex_bias = 0.3
    male_rate = model_with_population_and_sex_bias.dependent_params["rate1_male"](model_with_population_and_sex_bias, test_params)
    female_rate = model_with_population_and_sex_bias.dependent_params["rate1_female"](model_with_population_and_sex_bias, test_params)
    
    assert np.isclose(male_rate, 0.5 * (1 - 0.3))
    assert np.isclose(female_rate, 0.5 * (1 + 0.3))

def test_add_pulse_migration(model_with_pulse_migration):
    """Test adding pulse migrations with sex bias"""
    # Verify parameters were added
    assert "rate1" in model_with_pulse_migration.free_params
    assert "rate1_sex_bias" in model_with_pulse_migration.free_params
    assert "time1" in model_with_pulse_migration.free_params
    assert "founder_rate1" in model_with_pulse_migration.free_params
    assert "founder_rate1_sex_bias" in model_with_pulse_migration.free_params
    assert "found_time" in model_with_pulse_migration.free_params
    
    # Verify dependent parameters were created
    assert "rate1_male" in model_with_pulse_migration.dependent_params
    assert "rate1_female" in model_with_pulse_migration.dependent_params
    assert "founder_rate1_male" in model_with_pulse_migration.dependent_params
    assert "founder_rate1_female" in model_with_pulse_migration.dependent_params
    
    # Verify events were added
    model_with_pulse_migration.finalize()
    assert "destination_pop_male" in model_with_pulse_migration.events
    assert "destination_pop_female" in model_with_pulse_migration.events
    assert len(model_with_pulse_migration.events["destination_pop_male"]) == 1
    assert len(model_with_pulse_migration.events["destination_pop_female"]) == 1

def test_add_continuous_migration(model_with_continuous_migration):
    """Test adding continuous migrations with sex bias"""
    # Verify parameters were added
    assert "rate1" in model_with_continuous_migration.free_params
    assert "rate1_sex_bias" in model_with_continuous_migration.free_params
    assert "start1" in model_with_continuous_migration.free_params
    assert "end1" in model_with_continuous_migration.free_params
    assert "founder_rate1" in model_with_continuous_migration.free_params
    assert "founder_rate1_sex_bias" in model_with_continuous_migration.free_params
    assert "found_time" in model_with_continuous_migration.free_params
    
    # Verify dependent parameters were created
    assert "rate1_male" in model_with_continuous_migration.dependent_params
    assert "rate1_female" in model_with_continuous_migration.dependent_params
    assert "founder_rate1_male" in model_with_continuous_migration.dependent_params
    assert "founder_rate1_female" in model_with_continuous_migration.dependent_params
    
    # Verify events were added
    model_with_continuous_migration.finalize()
    assert "destination_pop_male" in model_with_continuous_migration.events
    assert "destination_pop_female" in model_with_continuous_migration.events
    assert len(model_with_continuous_migration.events["destination_pop_male"]) == 1
    assert len(model_with_continuous_migration.events["destination_pop_female"]) == 1

def test_migration_parameter_creation(model_with_pulse_migration):
    """Test that sex bias parameters are created correctly for migrations"""
    # Verify sex bias parameter
    assert "rate1_sex_bias" in model_with_pulse_migration.free_params
    assert model_with_pulse_migration.free_params["rate1_sex_bias"].type == ParamType.SEX_BIAS
    assert model_with_pulse_migration.free_params["rate1_sex_bias"].bounds == (-1, 1)
    
    # Add continuous migration
    model_with_pulse_migration.add_continuous_migration("destination_pop", "source_pop1", "rate2", "start1", "end1")
    
    # Verify sex bias parameter
    assert "rate2_sex_bias" in model_with_pulse_migration.free_params
    assert model_with_pulse_migration.free_params["rate2_sex_bias"].type == ParamType.SEX_BIAS
    assert model_with_pulse_migration.free_params["rate2_sex_bias"].bounds == (-1, 1)

def test_migration_execution(model_with_pulse_migration):
    """Test that migrations are executed correctly for both sexes"""
    # Finalize the model
    model_with_pulse_migration.finalize()
    
    # Test parameter evaluation
    test_params = [0.5, 0.3, 10, 0.4, 0.2, 5]  # founder_rate1, founder_rate1_sex_bias, found_time, rate1, sex_bias, time1
    
    # Get migration matrices
    migration_matrices = model_with_pulse_migration.get_migration_matrices(test_params)
    
    # Verify matrices were created for both sexes
    assert "destination_pop_male" in migration_matrices
    assert "destination_pop_female" in migration_matrices
    
    # Verify matrix dimensions
    assert migration_matrices["destination_pop_male"].shape[0] == 11  # time1 + 1
    assert migration_matrices["destination_pop_male"].shape[1] == 2  # number of populations
    assert migration_matrices["destination_pop_female"].shape[0] == 11
    assert migration_matrices["destination_pop_female"].shape[1] == 2
    
    # Verify migration rates
    male_rate = model_with_pulse_migration.dependent_params["rate1_male"](model_with_pulse_migration, test_params)
    female_rate = model_with_pulse_migration.dependent_params["rate1_female"](model_with_pulse_migration, test_params)
    
    # Check that migration rates are applied correctly
    assert np.isclose(migration_matrices["destination_pop_male"][5, 0], male_rate)
    assert np.isclose(migration_matrices["destination_pop_female"][5, 0], female_rate)

def test_founder_event_parameter_creation(model_with_founder_event):
    """Test that founder event parameters are created correctly with sex bias"""
    # Verify parameters were added
    assert "founder_rate1" in model_with_founder_event.free_params
    assert "founder_rate1_sex_bias" in model_with_founder_event.free_params
    assert "found_time" in model_with_founder_event.free_params
    
    # Verify parameter types
    assert model_with_founder_event.free_params["founder_rate1"].type == ParamType.RATE
    assert model_with_founder_event.free_params["founder_rate1_sex_bias"].type == ParamType.SEX_BIAS
    assert model_with_founder_event.free_params["found_time"].type == ParamType.TIME
    
    # Verify parameter bounds
    assert model_with_founder_event.free_params["founder_rate1"].bounds == (0, 1)
    assert model_with_founder_event.free_params["founder_rate1_sex_bias"].bounds == (-1, 1)
    assert model_with_founder_event.free_params["found_time"].bounds == (2, np.inf)
    
    # Verify dependent parameters were created
    assert "founder_rate1_male" in model_with_founder_event.dependent_params
    assert "founder_rate1_female" in model_with_founder_event.dependent_params

def test_continuous_founder_event_parameter_creation(model_with_continuous_founder_event):
    """Test that founder event parameters are created correctly with sex bias"""
    # Verify parameters were added
    assert "founder_rate1" in model_with_continuous_founder_event.free_params
    assert "founder_rate1_sex_bias" in model_with_continuous_founder_event.free_params
    assert "found_time" in model_with_continuous_founder_event.free_params
    assert "end_time" in model_with_continuous_founder_event.free_params
    
    # Verify parameter types
    assert model_with_continuous_founder_event.free_params["founder_rate1"].type == ParamType.RATE
    assert model_with_continuous_founder_event.free_params["founder_rate1_sex_bias"].type == ParamType.SEX_BIAS
    assert model_with_continuous_founder_event.free_params["founder_rate1"].type == ParamType.RATE
    assert model_with_continuous_founder_event.free_params["founder_rate1_sex_bias"].type == ParamType.SEX_BIAS
    assert model_with_continuous_founder_event.free_params["found_time"].type == ParamType.TIME
    assert model_with_continuous_founder_event.free_params["end_time"].type == ParamType.TIME
    
    # Verify parameter bounds
    assert model_with_continuous_founder_event.free_params["founder_rate1"].bounds == (0, 1)
    assert model_with_continuous_founder_event.free_params["founder_rate1_sex_bias"].bounds == (-1, 1)
    assert model_with_continuous_founder_event.free_params["found_time"].bounds == (2, np.inf)
    assert model_with_continuous_founder_event.free_params["founder_rate2_sex_bias"].bounds == (-1, 1)
    assert model_with_continuous_founder_event.free_params["end_time"].bounds == (2, np.inf)
    # Verify dependent parameters were created
    assert "founder_rate1_male" in model_with_continuous_founder_event.dependent_params
    assert "founder_rate1_female" in model_with_continuous_founder_event.dependent_params


def test_founder_event_population_addition(model_with_founder_event):
    """Test that founder event adds populations correctly"""
    # Verify populations were added
    assert "source_pop1" in model_with_founder_event.population_indices
    assert "source_pop2" in model_with_founder_event.population_indices
    
    # Verify founder event was recorded
    assert "destination_pop_male" in model_with_founder_event.founder_events
    assert "destination_pop_female" in model_with_founder_event.founder_events
    assert model_with_founder_event.founder_events["destination_pop_male"].source_populations["source_pop1"] == "founder_rate1_male"
    assert "source_pop2" not in model_with_founder_event.founder_events["destination_pop_male"].source_populations
    assert model_with_founder_event.founder_events["destination_pop_male"].remainder_population == "source_pop2"
    assert model_with_founder_event.founder_events["destination_pop_female"].source_populations["source_pop1"] == "founder_rate1_female"
    assert "source_pop2" not in model_with_founder_event.founder_events["destination_pop_female"].source_populations
    assert model_with_founder_event.founder_events["destination_pop_female"].remainder_population == "source_pop2"

def test_founder_event_sex_bias_calculation(model_with_founder_event_and_params):
    """Test that founder event sex bias is calculated correctly"""
    # Finalize the model
    model_with_founder_event_and_params.finalize()
    
    # Test parameter evaluation
    test_params = [0.4, 0.2, 5]  # founder_rate1, founder_rate1_sex_bias, found_time
    
    # Calculate expected rates
    expected_male_rate = 0.4-0.2 * (1/2-(1/2-0.4))  # rate * (1 - sex_bias)
    expected_female_rate = 0.4+0.2 * (1/2-(1/2-0.4))  # rate * (1 + sex_bias)
    
    # Get actual rates from dependent parameters
    male_rate = model_with_founder_event_and_params.dependent_params["founder_rate1_male"](model_with_founder_event_and_params, test_params)
    female_rate = model_with_founder_event_and_params.dependent_params["founder_rate1_female"](model_with_founder_event_and_params, test_params)
    
    # Verify rates
    assert np.isclose(male_rate, expected_male_rate)
    assert np.isclose(female_rate, expected_female_rate)

def test_continuous_founder_event_sex_bias_calculation(model_with_continuous_founder_event):
    """Test that founder event sex bias is calculated correctly"""
    # Finalize the model
    model_with_continuous_founder_event.finalize()
    
    # Test parameter evaluation
    founder_rate_1 = 0.2
    bias1 = 0.1
    founder_rate_2 = 0.2
    bias2 = -0.7
    found_time = 5
    end_time=2
    test_params = [founder_rate_1, bias1, founder_rate_2, bias2, found_time,end_time]  
    
    # Calculate expected rates
    expected_male_rate1 = founder_rate_1-bias1 * (1/2-np.abs((1/2-founder_rate_1)))  # rate * (1 - sex_bias)
    expected_female_rate1 = founder_rate_1+bias1 * (1/2-np.abs((1/2-founder_rate_1)))  # rate * (1 + sex_bias)
    expected_male_rate2 = founder_rate_2-bias2 * (1/2-np.abs((1/2-founder_rate_2)))  # rate * (1 - sex_bias)
    expected_female_rate2 = founder_rate_2+bias2 * (1/2-np.abs((1/2-founder_rate_2)))  # rate * (1 + sex_bias)

    # Get actual rates from dependent parameters
    male_rate1 = model_with_continuous_founder_event.dependent_params["founder_rate1_male"](model_with_continuous_founder_event, test_params)
    female_rate1 = model_with_continuous_founder_event.dependent_params["founder_rate1_female"](model_with_continuous_founder_event, test_params)
    male_rate2 = model_with_continuous_founder_event.dependent_params["founder_rate2_male"](model_with_continuous_founder_event, test_params)
    female_rate2 = model_with_continuous_founder_event.dependent_params["founder_rate2_female"](model_with_continuous_founder_event, test_params)
    # Verify rates
    assert np.isclose(male_rate1, expected_male_rate1)
    assert np.isclose(female_rate1, expected_female_rate1)
    assert np.isclose(male_rate2, expected_male_rate2)
    assert np.isclose(female_rate2, expected_female_rate2)


def test_yaml_loading_basic():
    """Test loading a basic sex-biased demography from YAML"""
    # Create a temporary YAML file
    yaml_content = """
model_name: TestModel
demes:
  - name: dest_pop1
    ancestors: [source_pop2, source_pop3]
    proportions: [founding_rate_1, 1-founding_rate_1]
    start_time: found_time
pulses:
  - dest: dest_pop1
    sources: [source_pop2]
    proportions: [rate1]
    time: time1
migrations:
  - dest: dest_pop1
    source: source_pop3
    rate: rate2
    start_time: start1
    end_time: end1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = temp_file.name
    
    try:
        # Load the model from YAML
        model = ParametrizedDemographySexBiased.load_from_YAML(temp_file_path)
        
        # Verify model name
        assert model.name == "TestModel"
        
        # Verify founder event
        assert "dest_pop1_male" in model.founder_events
        assert "dest_pop1_female" in model.founder_events
        
        # Verify parameters
        assert "found_time" in model.free_params
        assert "rate1" in model.free_params
        assert "rate1_sex_bias" in model.free_params
        assert "time1" in model.free_params
        assert "rate2" in model.free_params
        assert "rate2_sex_bias" in model.free_params
        assert "start1" in model.free_params
        assert "end1" in model.free_params
        
        # Verify dependent parameters
        assert "rate1_male" in model.dependent_params
        assert "rate1_female" in model.dependent_params
        assert "rate2_male" in model.dependent_params
        assert "rate2_female" in model.dependent_params
        
        # Verify events
        model.finalize()
        assert "dest_pop1_male" in model.events
        assert "dest_pop1_female" in model.events
        assert len(model.events["dest_pop1_male"]) == 2  # One pulse and one continuous migration
        assert len(model.events["dest_pop1_female"]) == 2
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def test_yaml_loading_multiple_populations():
    """Test loading a sex-biased demography with multiple populations from YAML"""
    # Create a temporary YAML file
    yaml_content = """
model_name: MultiPopModel
demes:
  - name: dest_pop1
    ancestors: [source_pop1, source_pop2]
    proportions: [founding_rate_1, 1-founding_rate_1]
    start_time: found_time1
  - name: dest_pop2
    ancestors: [source_pop1, source_pop2]
    proportions: [founding_rate_2, 1-founding_rate_2]
    start_time: found_time2
pulses:
  - dest: dest_pop1
    sources: [source_pop1]
    proportions: [rate1]
    time: time1
  - dest: dest_pop2
    sources: [source_pop1]
    proportions: [rate2]
    time: time2
migrations:
  - dest: dest_pop1
    source: source_pop1
    rate: rate3
    start_time: start1
    end_time: end1
  - dest: dest_pop2
    source: source_pop2
    rate: rate4
    start_time: start2
    end_time: end2
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = temp_file.name
    
    try:
        # Load the model from YAML
        model = ParametrizedDemographySexBiased.load_from_YAML(temp_file_path)
        
        # Verify model name
        assert model.name == "MultiPopModel"
        
        # Verify founder events
        assert "dest_pop1_male" in model.founder_events
        assert "dest_pop1_female" in model.founder_events
        assert "dest_pop2_male" in model.founder_events
        assert "dest_pop2_female" in model.founder_events
        
        # Verify parameters
        assert "found_time1" in model.free_params
        assert "found_time2" in model.free_params
        assert "rate1" in model.free_params
        assert "rate1_sex_bias" in model.free_params
        assert "time1" in model.free_params
        assert "rate2" in model.free_params
        assert "rate2_sex_bias" in model.free_params
        assert "time2" in model.free_params
        assert "rate3" in model.free_params
        assert "rate3_sex_bias" in model.free_params
        assert "start1" in model.free_params
        assert "end1" in model.free_params
        assert "rate4" in model.free_params
        assert "rate4_sex_bias" in model.free_params
        assert "start2" in model.free_params
        assert "end2" in model.free_params
        
        # Verify dependent parameters
        assert "rate1_male" in model.dependent_params
        assert "rate1_female" in model.dependent_params
        assert "rate2_male" in model.dependent_params
        assert "rate2_female" in model.dependent_params
        assert "rate3_male" in model.dependent_params
        assert "rate3_female" in model.dependent_params
        assert "rate4_male" in model.dependent_params
        assert "rate4_female" in model.dependent_params
        
        # Verify events
        model.finalize()
        assert "dest_pop1_male" in model.events
        assert "dest_pop1_female" in model.events
        assert "dest_pop2_male" in model.events
        assert "dest_pop2_female" in model.events
        assert len(model.events["dest_pop1_male"]) == 2  # One pulse and one continuous migration
        assert len(model.events["dest_pop1_female"]) == 2
        assert len(model.events["dest_pop2_male"]) == 2  # One pulse and one continuous migration
        assert len(model.events["dest_pop2_female"]) == 2
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def test_yaml_loading_missing_required():
    """Test loading a YAML file with missing required fields raises appropriate errors"""
    # Create a temporary YAML file with missing required fields
    yaml_content = """
model_name: MissingRequiredModel
demes:
  - name: dest_pop1
    ancestors: [source_pop1, source_pop2]
    proportions: [founding_rate_1, 1-founding_rate_1]
    # Missing start_time
pulses:
  - dest: dest_pop1
    sources: [source_pop1]
    proportions: [rate1]
    time: time1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = temp_file.name
    
    try:
        # Load the model from YAML should raise an error
        with pytest.raises(KeyError):
            ParametrizedDemographySexBiased.load_from_YAML(temp_file_path)
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def test_migration_matrix_basic(model_with_both_migrations):
    """Test that migration matrices are generated correctly for a basic model"""
    # Finalize the model
    model_with_both_migrations.finalize()
    
    # Test parameter evaluation
    test_params = [0.4, 0.2, 10,# founder_rate1, founder_rate1_sex_bias, found_time,
        0.3, 0.1, 3,# rate1, rate1_sex_bias, time1,
        0.2, 0.05, 7, 5] #rate2, rate2_sex_bias, start1, end1
    
    # Get migration matrices
    migration_matrices = model_with_both_migrations.get_migration_matrices(test_params)
    
    # Verify matrices were created for both sexes
    assert "destination_pop_male" in migration_matrices
    assert "destination_pop_female" in migration_matrices
    
    # Verify matrix dimensions
    assert migration_matrices["destination_pop_male"].shape[0] == 11  # max(time1, end1) + 1
    assert migration_matrices["destination_pop_male"].shape[1] == 2  # number of populations
    assert migration_matrices["destination_pop_female"].shape[0] == 11
    assert migration_matrices["destination_pop_female"].shape[1] == 2
    
    # Verify founder rates are applied correctly
    male_founder_rate = model_with_both_migrations.dependent_params["founder_rate1_male"](model_with_both_migrations, test_params)
    female_founder_rate = model_with_both_migrations.dependent_params["founder_rate1_female"](model_with_both_migrations, test_params)
    
    # Check that founder rates are applied correctly at the founding time
    assert np.isclose(migration_matrices["destination_pop_male"][10, 0], male_founder_rate)
    assert np.isclose(migration_matrices["destination_pop_female"][10, 0], female_founder_rate)
    
    # Verify that rates sum to 1 for each sex at the founding time
    assert np.isclose(migration_matrices["destination_pop_male"][10, 0] + migration_matrices["destination_pop_male"][10, 1], 1)
    assert np.isclose(migration_matrices["destination_pop_female"][10, 0] + migration_matrices["destination_pop_female"][10, 1], 1)
    
    # Verify pulse migration is applied correctly
    male_pulse_rate = model_with_both_migrations.dependent_params["rate1_male"](model_with_both_migrations, test_params)
    female_pulse_rate = model_with_both_migrations.dependent_params["rate1_female"](model_with_both_migrations, test_params)
    
    # Check that pulse migration is applied correctly at the pulse time
    assert np.isclose(migration_matrices["destination_pop_male"][3, 0], male_pulse_rate)
    assert np.isclose(migration_matrices["destination_pop_female"][3, 0], female_pulse_rate)
    
    # Verify continuous migration is applied correctly
    male_continuous_rate = model_with_both_migrations.dependent_params["rate2_male"](model_with_both_migrations, test_params)
    female_continuous_rate = model_with_both_migrations.dependent_params["rate2_female"](model_with_both_migrations, test_params)
    
    # Check that continuous migration is applied correctly during the migration period
    for t in range(5, 7):  # From start1 to end1
        assert np.isclose(migration_matrices["destination_pop_male"][t, 1], male_continuous_rate)
        assert np.isclose(migration_matrices["destination_pop_female"][t, 1], female_continuous_rate)



def test_migration_matrix_continuous_founder(model_with_continuous_founder_event):
    """Test that migration matrices are generated correctly for a basic model"""
    # Finalize the model
    model_with_continuous_founder_event.finalize()
    

    # Test parameter evaluation
    founder_rate_1 = 0.2
    bias1 = 0.1
    founder_rate_2 = 0.2
    bias2 = -0.7
    found_time = 8
    end_time=2
    test_params = [founder_rate_1, bias1, founder_rate_2, bias2, found_time,end_time]


    
    # Get migration matrices
    migration_matrices = model_with_continuous_founder_event.get_migration_matrices(test_params)
    
    # Verify matrices were created for both sexes
    assert "destination_pop_male" in migration_matrices
    assert "destination_pop_female" in migration_matrices
    
    # Verify matrix dimensions
    maxgen = np.ceil(found_time) + 1
    assert migration_matrices["destination_pop_male"].shape[0] == maxgen
    assert migration_matrices["destination_pop_male"].shape[1] == 2  # number of populations
    assert migration_matrices["destination_pop_female"].shape[0] == maxgen
    assert migration_matrices["destination_pop_female"].shape[1] == 2
    
    # Verify founder rates are applied correctly
    male_founder_rate1 = model_with_continuous_founder_event.dependent_params["founder_rate1_male"](model_with_continuous_founder_event, test_params)
    female_founder_rate1 = model_with_continuous_founder_event.dependent_params["founder_rate1_female"](model_with_continuous_founder_event, test_params)
    male_founder_rate2 = model_with_continuous_founder_event.dependent_params["founder_rate2_male"](model_with_continuous_founder_event, test_params)
    female_founder_rate2 = model_with_continuous_founder_event.dependent_params["founder_rate2_female"](model_with_continuous_founder_event, test_params)

    # Check that founder rates are applied correctly at the founding time
    assert np.isclose(migration_matrices["destination_pop_male"][6, 0], male_founder_rate1)
    assert np.isclose(migration_matrices["destination_pop_female"][6, 0], female_founder_rate1)
    assert np.isclose(migration_matrices["destination_pop_male"][6, 1], male_founder_rate2)
    assert np.isclose(migration_matrices["destination_pop_female"][6, 1], female_founder_rate2)


    # Verify that rates sum to 1 for each sex at the founding time
    assert np.isclose(migration_matrices["destination_pop_male"][maxgen-1, 0] + migration_matrices["destination_pop_male"][maxgen-1, 1], 1)
    assert np.isclose(migration_matrices["destination_pop_female"][maxgen-1, 0] + migration_matrices["destination_pop_female"][maxgen-1, 1], 1)
    
    # Test parameter evaluation, no bias
    founder_rate_1 = 0.2
    bias1 = 0.
    founder_rate_2 = 0.2
    bias2 = -0.
    found_time = 8
    end_time=2
    test_params = [founder_rate_1, bias1, founder_rate_2, bias2, found_time,end_time]


    
    # Get migration matrices
    migration_matrices = model_with_continuous_founder_event.get_migration_matrices(test_params)
    diff =migration_matrices["destination_pop_male"] - migration_matrices["destination_pop_female"]
    assert np.isclose( np.sum(diff**2),0), f"males and females should be equal if no sex bias. "


    # Test parameter evaluation, strong bias
    founder_rate_1 = 0.5
    bias1 = 1
    founder_rate_2 = 0.5
    bias2 = -1
    found_time = 8
    end_time=2
    test_params = [founder_rate_1, bias1, founder_rate_2, bias2, found_time,end_time]


    
    # Get migration matrices
    migration_matrices = model_with_continuous_founder_event.get_migration_matrices(test_params)
    assert np.isclose(migration_matrices["destination_pop_male"].sum(axis=1)[0] ,0), "should have no males from pop 1"
    assert np.isclose(migration_matrices["destination_pop_female"].sum(axis=1)[1] ,0), "should have no females from pop 2"



def test_migration_matrix_multiple_populations(model_with_multiple_populations):
    """Test that migration matrices are generated correctly for multiple populations"""
    # Finalize the model
    model_with_multiple_populations.finalize()
    
    # Test parameter evaluation
    test_params = [0.4, 0.2, 10, # founder_rate1, founder_rate1_sex_bias, found_time1
        0.5, 0.1, 12, # founder_rate2, founder_rate2_sex_bias, found_time2,
        0.3, 0.05, 8,  # rate1, rate1_sex_bias, time1
        0.6, 0.15, 7, 5] #rate2, rate2_sex_bias, start1, end1
    
    # Get migration matrices
    migration_matrices = model_with_multiple_populations.get_migration_matrices(test_params)
    
    # Verify matrices were created for both sexes and both populations
    assert "dest_pop1_male" in migration_matrices
    assert "dest_pop1_female" in migration_matrices
    assert "dest_pop2_male" in migration_matrices
    assert "dest_pop2_female" in migration_matrices
    
    # Verify matrix dimensions
    assert migration_matrices["dest_pop1_male"].shape[0] == 11  # max(found_time1, found_time2, time1, end1) + 1
    assert migration_matrices["dest_pop1_male"].shape[1] == 2  # number of populations
    assert migration_matrices["dest_pop1_female"].shape[0] == 11
    assert migration_matrices["dest_pop1_female"].shape[1] == 2
    assert migration_matrices["dest_pop2_male"].shape[0] == 13
    assert migration_matrices["dest_pop2_male"].shape[1] == 2
    assert migration_matrices["dest_pop2_female"].shape[0] == 13
    assert migration_matrices["dest_pop2_female"].shape[1] == 2
    
    # Verify founder rates are applied correctly for dest_pop1
    male_founder_rate1 = model_with_multiple_populations.dependent_params["founder_rate1_male"](model_with_multiple_populations, test_params)
    female_founder_rate1 = model_with_multiple_populations.dependent_params["founder_rate1_female"](model_with_multiple_populations, test_params)
    
    # Check that founder rates are applied correctly at the founding time for dest_pop1
    assert np.isclose(migration_matrices["dest_pop1_male"][10, 0], male_founder_rate1)
    assert np.isclose(migration_matrices["dest_pop1_female"][10, 0], female_founder_rate1)
    
    # Verify founder rates are applied correctly for dest_pop2
    male_founder_rate2 = model_with_multiple_populations.dependent_params["founder_rate2_male"](model_with_multiple_populations, test_params)
    female_founder_rate2 = model_with_multiple_populations.dependent_params["founder_rate2_female"](model_with_multiple_populations, test_params)
    
    # Check that founder rates are applied correctly at the founding time for dest_pop2
    assert np.isclose(migration_matrices["dest_pop2_male"][12, 1], male_founder_rate2)
    assert np.isclose(migration_matrices["dest_pop2_female"][12, 1], female_founder_rate2)
    
    # Verify pulse migration is applied correctly
    male_pulse_rate = model_with_multiple_populations.dependent_params["rate1_male"](model_with_multiple_populations, test_params)
    female_pulse_rate = model_with_multiple_populations.dependent_params["rate1_female"](model_with_multiple_populations, test_params)
    
    # Check that pulse migration is applied correctly at the pulse time
    assert np.isclose(migration_matrices["dest_pop1_male"][8, 1], male_pulse_rate)
    assert np.isclose(migration_matrices["dest_pop1_female"][8, 1], female_pulse_rate)
    
    # Verify continuous migration is applied correctly
    male_continuous_rate = model_with_multiple_populations.dependent_params["rate2_male"](model_with_multiple_populations, test_params)
    female_continuous_rate = model_with_multiple_populations.dependent_params["rate2_female"](model_with_multiple_populations, test_params)
    
    # Check that continuous migration is applied correctly during the migration period
    for t in range(5, 7):  # From start1 to end1
        assert np.isclose(migration_matrices["dest_pop2_male"][t, 0], male_continuous_rate)
        assert np.isclose(migration_matrices["dest_pop2_female"][t, 0], female_continuous_rate)









def test_migration_matrix_sex_bias(model_with_both_migrations):
    """Test that sex bias is correctly applied in migration matrices"""
    # Finalize the model
    model_with_both_migrations.finalize()
    
    # Test parameter evaluation with different sex bias values
    # Case 1: No sex bias (sex_bias = 0)
    test_params1 = [0.4, 0.0, 10, 0.3, 0.0, 5, 0.2, 0.0, 7, 3]
    
    # Get migration matrices
    migration_matrices1 = model_with_both_migrations.get_migration_matrices(test_params1)
    
    # Verify that male and female rates are equal when sex_bias = 0
    male_founder_rate1 = model_with_both_migrations.dependent_params["founder_rate1_male"](model_with_both_migrations, test_params1)
    female_founder_rate1 = model_with_both_migrations.dependent_params["founder_rate1_female"](model_with_both_migrations, test_params1)
    assert np.isclose(male_founder_rate1, female_founder_rate1)
    assert np.isclose(migration_matrices1["destination_pop_male"][10, 0], migration_matrices1["destination_pop_female"][10, 0])
    
    male_pulse_rate1 = model_with_both_migrations.dependent_params["rate1_male"](model_with_both_migrations, test_params1)
    female_pulse_rate1 = model_with_both_migrations.dependent_params["rate1_female"](model_with_both_migrations, test_params1)
    assert np.isclose(male_pulse_rate1, female_pulse_rate1)
    assert np.isclose(migration_matrices1["destination_pop_male"][5, 0], migration_matrices1["destination_pop_female"][5, 0])
    
    male_continuous_rate1 = model_with_both_migrations.dependent_params["rate2_male"](model_with_both_migrations, test_params1)
    female_continuous_rate1 = model_with_both_migrations.dependent_params["rate2_female"](model_with_both_migrations, test_params1)
    assert np.isclose(male_continuous_rate1, female_continuous_rate1)
    assert np.isclose(migration_matrices1["destination_pop_male"][3, 1], migration_matrices1["destination_pop_female"][3, 1])
    
    # Case 2: Positive sex bias (sex_bias > 0)
    test_params2 = [0.4, 0.2, 10, 0.3, 0.1, 5, 0.2, 0.05, 7, 3]
    
    # Get migration matrices
    migration_matrices2 = model_with_both_migrations.get_migration_matrices(test_params2)
    
    # Verify that female rates are higher than male rates when sex_bias > 0
    male_founder_rate2 = model_with_both_migrations.dependent_params["founder_rate1_male"](model_with_both_migrations, test_params2)
    female_founder_rate2 = model_with_both_migrations.dependent_params["founder_rate1_female"](model_with_both_migrations, test_params2)
    assert female_founder_rate2 > male_founder_rate2
    assert migration_matrices2["destination_pop_female"][10, 0] > migration_matrices2["destination_pop_male"][10, 0]
    
    male_pulse_rate2 = model_with_both_migrations.dependent_params["rate1_male"](model_with_both_migrations, test_params2)
    female_pulse_rate2 = model_with_both_migrations.dependent_params["rate1_female"](model_with_both_migrations, test_params2)
    assert female_pulse_rate2 > male_pulse_rate2
    assert migration_matrices2["destination_pop_female"][5, 0] > migration_matrices2["destination_pop_male"][5, 0]
    
    male_continuous_rate2 = model_with_both_migrations.dependent_params["rate2_male"](model_with_both_migrations, test_params2)
    female_continuous_rate2 = model_with_both_migrations.dependent_params["rate2_female"](model_with_both_migrations, test_params2)
    assert female_continuous_rate2 > male_continuous_rate2
    assert migration_matrices2["destination_pop_female"][3, 1] > migration_matrices2["destination_pop_male"][3, 1]
    
    # Case 3: Negative sex bias (sex_bias < 0)
    test_params3 = [0.4, -0.2, 10, 0.3, -0.1, 5, 0.2, -0.05, 7, 3]
    
    # Get migration matrices
    migration_matrices3 = model_with_both_migrations.get_migration_matrices(test_params3)
    
    # Verify that male rates are higher than female rates when sex_bias < 0
    male_founder_rate3 = model_with_both_migrations.dependent_params["founder_rate1_male"](model_with_both_migrations, test_params3)
    female_founder_rate3 = model_with_both_migrations.dependent_params["founder_rate1_female"](model_with_both_migrations, test_params3)
    assert male_founder_rate3 > female_founder_rate3
    assert migration_matrices3["destination_pop_male"][10, 0] > migration_matrices3["destination_pop_female"][10, 0]
    
    male_pulse_rate3 = model_with_both_migrations.dependent_params["rate1_male"](model_with_both_migrations, test_params3)
    female_pulse_rate3 = model_with_both_migrations.dependent_params["rate1_female"](model_with_both_migrations, test_params3)
    assert male_pulse_rate3 > female_pulse_rate3
    assert migration_matrices3["destination_pop_male"][5, 0] > migration_matrices3["destination_pop_female"][5, 0]
    
    male_continuous_rate3 = model_with_both_migrations.dependent_params["rate2_male"](model_with_both_migrations, test_params3)
    female_continuous_rate3 = model_with_both_migrations.dependent_params["rate2_female"](model_with_both_migrations, test_params3)
    assert male_continuous_rate3 > female_continuous_rate3
    assert migration_matrices3["destination_pop_male"][3, 1] > migration_matrices3["destination_pop_female"][3, 1]

def test_error_invalid_yaml_file():
    """Test that loading non-existent YAML files raises appropriate errors"""
    # Try to load a non-existent YAML file
    with pytest.raises(FileNotFoundError):
        ParametrizedDemographySexBiased.load_from_YAML("non_existent_file.yaml")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to load a YAML file from the temporary directory
        with pytest.raises(FileNotFoundError):
            ParametrizedDemographySexBiased.load_from_YAML(os.path.join(temp_dir, "non_existent_file.yaml"))