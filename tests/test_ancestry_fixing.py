import os
import sys
import numpy as np
import math
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+'\\..')
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased

"""
This test suite checks the functionality of the ancestry fixing feature in the `ParametrizedDemography` class, which allows users to fix certain parameters based on sample proportions.
"""

def test_ancestry_fixing_single_population():
    """
    Test the ancestry fixing functionality for a single population with two founders.
    
    This test:
    1. Creates a model with two founders and no other events,
    2. Sets up sample proportions,
    3. Fixes the founding rate parameter using the sample proportions,
    4. Verifies that the model can take in only a founding time value and output a migration matrix,
    5. Verifies that the resulting rate parameter matches the sample proportions,
    6. Verifies that the final proportions of the matrix match the sample proportions.
    """

    # Create a model with two founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions (70% from source_pop1, 30% from source_pop2)
    sample_proportions = {
        "target_pop": [0.7, 0.3]  # [source_pop1, source_pop2]
    }
    
    # Fix the founding rate parameter using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding time (since the rate is fixed)
    found_time = 10-1e-9
    found_time_ceil = math.ceil(found_time)
    test_free_params = [found_time]  # Only the founding time
    test_params = [0,found_time] #the first parameter should be rewritten
    # Get the migration matrices

    assert (model.parameter_handler.params_fixed_by_ancestry_indices == [0])

    test_params = model.parameter_handler.compute_params_fixed_by_ancestry(test_params)
    test_params2 = model.parameter_handler.extend_parameters(test_free_params)
    assert np.isclose(test_params, test_params2).all()
    
    migration_matrices = model.get_migration_matrices(test_params)

    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert np.isclose(matrix.shape[0], found_time + 1)  # found_time + 1
    assert matrix.shape[1] == 2  # two source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix[found_time_ceil, 0], 0.7)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[found_time_ceil, 1], 0.3)  # source_pop2 proportion at founding time
    
    # Verify that the final proportions match the sample proportions
    final_proportions = model.proportions_from_matrix(matrix)
    assert np.isclose(final_proportions[0], 0.7)  # source_pop1 proportion
    assert np.isclose(final_proportions[1], 0.3)  # source_pop2 proportion
    
    # Verify that the sum of proportions is 1
    assert np.isclose(final_proportions.sum(), 1.0)


def test_ancestry_fixing_single_population_with_fixed_param():
    """
    Test the ancestry fixing functionality for a single population with two founders.
    
    This test:
    1. Creates a model with two founders and no other events,
    2. Sets up sample proportions,
    3. Fixes the founding rate parameter using the sample proportions,
    4. Verifies that the model can take in only a founding time value and output a migration matrix,
    5. Verifies that the resulting rate parameter matches the sample proportions,
    6. Verifies that the final proportions of the matrix match the sample proportions.
    """
    # Create a model with two founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions (70% from source_pop1, 30% from source_pop2)
    sample_proportions = {
        "target_pop": [0.7, 0.3]  # [source_pop1, source_pop2]
    }
    
    # Fix the start time
    fixed_time =10-1e-9
    fixed_time_ceil = math.ceil(fixed_time)

    
    fixed_params = {"found_time": fixed_time}
    # Fix the founding rate parameter using the sample proportions
    
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        user_params_to_fix_by_value=fixed_params,
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # There are no free parameters !
    test_free_params = []  # Only the founding time
    test_params = model.parameter_handler.extend_parameters(test_free_params)
    
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == 11  # found_time + 1
    assert matrix.shape[1] == 2  # two source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix[fixed_time_ceil, 0], 0.7)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[fixed_time_ceil, 1], 0.3)  # source_pop2 proportion at founding time
    
    # Verify that the final proportions match the sample proportions
    final_proportions = model.proportions_from_matrix(matrix)
    assert np.isclose(final_proportions[0], 0.7)  # source_pop1 proportion
    assert np.isclose(final_proportions[1], 0.3)  # source_pop2 proportion
    
    # Verify that the sum of proportions is 1
    assert np.isclose(final_proportions.sum(), 1.0)

def test_ancestry_fixing_multiple_populations():
    """
    Test the ancestry fixing functionality for multiple populations.
    
    This test:
    1. Creates a model with two populations, each with two founders,
    2. Sets up sample proportions for each population,
    3. Fixes the founding rate parameters using the sample proportions,
    4. Verifies that the model can take in only founding time values and output migration matrices,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrices match the sample proportions.
    """

    # Create a model with two populations, each with two founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add founder events for two populations
    model.add_founder_event(
        dest_population="target_pop1",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time1"
    )
    
    model.add_founder_event(
        dest_population="target_pop2",
        source_populations={"source_pop2": "founder_rate2"},
        remainder_population="source_pop1",
        found_time="found_time2"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions
    sample_proportions = {
        "target_pop1": [0.7, 0.3],  # [source_pop1, source_pop2]
        "target_pop2": [0.4, 0.6]   # [source_pop1, source_pop2]
    }
    
    # Fix the founding rate parameters using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding times (since the rates are fixed)
    
    found_time1 = 10-1e-9
    found_time2 = 15-1e-9
    int_found_time1 = math.ceil(found_time1)
    int_found_time2 = math.ceil(found_time2)
    test_free_params = [found_time1, found_time2]  # [found_time1, found_time2]
    test_parameters = model.parameter_handler.extend_parameters(test_free_params)
 
    assert len(test_parameters) == 4
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_parameters)
    
    # Verify that we got migration matrices for both target populations
    assert "target_pop1" in migration_matrices
    assert "target_pop2" in migration_matrices
    
    # Get the matrices for the target populations
    matrix1 = migration_matrices["target_pop1"]
    matrix2 = migration_matrices["target_pop2"]
    
    # Verify the matrix dimensions
    assert matrix1.shape[0] == int_found_time1 + 1   
    assert matrix1.shape[1] == 2  # two source populations
    assert matrix2.shape[0] == int_found_time2 + 1   
    assert matrix2.shape[1] == 2  # two source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix1[10, 0], 0.7)  # source_pop1 proportion at founding time for target_pop1
    assert np.isclose(matrix1[10, 1], 0.3)  # source_pop2 proportion at founding time for target_pop1
    assert np.isclose(matrix2[15, 0], 0.4)  # source_pop1 proportion at founding time for target_pop2
    assert np.isclose(matrix2[15, 1], 0.6)  # source_pop2 proportion at founding time for target_pop2
    
    # Verify that the final proportions match the sample proportions
    final_proportions1 = model.proportions_from_matrix(matrix1)
    final_proportions2 = model.proportions_from_matrix(matrix2)
    
    assert np.isclose(final_proportions1[0], 0.7)  # source_pop1 proportion for target_pop1
    assert np.isclose(final_proportions1[1], 0.3)  # source_pop2 proportion for target_pop1
    assert np.isclose(final_proportions2[0], 0.4)  # source_pop1 proportion for target_pop2
    assert np.isclose(final_proportions2[1], 0.6)  # source_pop2 proportion for target_pop2
    
    # Verify that the sum of proportions is 1 for each population
    assert np.isclose(final_proportions1.sum(), 1.0)
    assert np.isclose(final_proportions2.sum(), 1.0)


def test_ancestry_fixing_three_founders():
    """
    Test the ancestry fixing functionality for a population with three founders.
    
    This test:
    1. Creates a model with one population and three founders,
    2. Sets up sample proportions,
    3. Fixes the founding rate parameters using the sample proportions,
    4. Verifies that the model can take in only a founding time value and output a migration matrix,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrix match the sample proportions.
    """
    # Create a model with three founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add a founder event with three source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={
            "source_pop1": "founder_rate1",
            "source_pop2": "founder_rate2"
        },
        remainder_population="source_pop3",
        found_time="found_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions (40% from source_pop1, 30% from source_pop2, 30% from source_pop3)
    sample_proportions = {
        "target_pop": [0.4, 0.3, 0.3]  # [source_pop1, source_pop2, source_pop3]
    }
    
    # Fix the founding rate parameters using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding time (since the rates are fixed)
    founding_time = 10 - 1e-9
    int_found_time = math.ceil(founding_time) 
    
    test_free_params = [founding_time]  # Only the founding time
    test_params = model.parameter_handler.extend_parameters(test_free_params)
 
    assert len(test_params) == 3

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == int_found_time + 1 
    assert matrix.shape[1] == 3  # three source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix[10, 0], 0.4)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[10, 1], 0.3)  # source_pop2 proportion at founding time
    assert np.isclose(matrix[10, 2], 0.3)  # source_pop3 proportion at founding time
    
    # Verify that the final proportions match the sample proportions
    final_proportions = model.proportions_from_matrix(matrix)
    assert np.isclose(final_proportions[0], 0.4)  # source_pop1 proportion
    assert np.isclose(final_proportions[1], 0.3)  # source_pop2 proportion
    assert np.isclose(final_proportions[2], 0.3)  # source_pop3 proportion
    
    # Verify that the sum of proportions is 1
    assert np.isclose(final_proportions.sum(), 1.0)


def test_ancestry_fixing_two_samples_three_founders():
    """
    Test the ancestry fixing functionality for two populations, each with three founders.
    
    This test:
    1. Creates a model with two populations, each with three founders,
    2. Sets up sample proportions for each population,
    3. Fixes the founding rate parameters using the sample proportions,
    4. Verifies that the model can take in only founding time values and output migration matrices,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrices match the sample proportions.
    """

    # Create a model with two populations, each with three founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add founder events for two populations
    model.add_founder_event(
        dest_population="target_pop1",
        source_populations={
            "source_pop1": "founder_rate1_pop1",
            "source_pop2": "founder_rate2_pop1"
        },
        remainder_population="source_pop3",
        found_time="found_time1"
    )
    
    model.add_founder_event(
        dest_population="target_pop2",
        source_populations={
            "source_pop2": "founder_rate1_pop2",
            "source_pop3": "founder_rate2_pop2"
        },
        remainder_population="source_pop1",
        found_time="found_time2"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions
    sample_proportions = {
        "target_pop1": [0.4, 0.3, 0.3],  # [source_pop1, source_pop2, source_pop3]
        "target_pop2": [0.2, 0.3, 0.5]   # [source_pop1, source_pop2, source_pop3]
    }
    
    # Fix the founding rate parameters using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1_pop1", "founder_rate2_pop1", "founder_rate1_pop2", "founder_rate2_pop2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding times (since the rates are fixed)
    
    found_time1 = 10-1e-9
    found_time2 = 15-1e-9
    int_found_time1 = math.ceil(found_time1)
    int_found_time2 = math.ceil(found_time2)
    
    test_free_params = [found_time1, found_time2]
    test_params = model.parameter_handler.extend_parameters(test_free_params)
 
    assert len(test_params) == 6

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both target populations
    assert "target_pop1" in migration_matrices
    assert "target_pop2" in migration_matrices
    
    # Get the matrices for the target populations
    matrix1 = migration_matrices["target_pop1"]
    matrix2 = migration_matrices["target_pop2"]
    
    # Verify the matrix dimensions
    assert matrix1.shape[0] == int_found_time1 + 1
    assert matrix1.shape[1] == 3  # three source populations
    assert matrix2.shape[0] == int_found_time2 + 1
    assert matrix2.shape[1] == 3  # three source populations
    
    # Verify that the founder rates match the sample proportions for target_pop1
    assert np.isclose(matrix1[10, 0], 0.4)  # source_pop1 proportion at founding time for target_pop1
    assert np.isclose(matrix1[10, 1], 0.3)  # source_pop2 proportion at founding time for target_pop1
    assert np.isclose(matrix1[10, 2], 0.3)  # source_pop3 proportion at founding time for target_pop1
    
    # Verify that the founder rates match the sample proportions for target_pop2
    assert np.isclose(matrix2[15, 0], 0.2)  # source_pop1 proportion at founding time for target_pop2
    assert np.isclose(matrix2[15, 1], 0.3)  # source_pop2 proportion at founding time for target_pop2
    assert np.isclose(matrix2[15, 2], 0.5)  # source_pop3 proportion at founding time for target_pop2
    
    # Verify that the final proportions match the sample proportions
    final_proportions1 = model.proportions_from_matrix(matrix1)
    final_proportions2 = model.proportions_from_matrix(matrix2)
    
    assert np.isclose(final_proportions1[0], 0.4)  # source_pop1 proportion for target_pop1
    assert np.isclose(final_proportions1[1], 0.3)  # source_pop2 proportion for target_pop1
    assert np.isclose(final_proportions1[2], 0.3)  # source_pop3 proportion for target_pop1
    
    assert np.isclose(final_proportions2[0], 0.2)  # source_pop1 proportion for target_pop2
    assert np.isclose(final_proportions2[1], 0.3)  # source_pop2 proportion for target_pop2
    assert np.isclose(final_proportions2[2], 0.5)  # source_pop3 proportion for target_pop2
    
    # Verify that the sum of proportions is 1 for each population
    assert np.isclose(final_proportions1.sum(), 1.0)
    assert np.isclose(final_proportions2.sum(), 1.0)

def test_ancestry_fixing_with_pulse_migration():
    """
    Test the ancestry fixing functionality for a model with two founders and one pulse migration.
    
    This test:
    1. Creates a model with two founders and one pulse migration,
    2. Sets up sample proportions,
    3. Fixes the founding rate parameter using the sample proportions,
    4. Verifies that the model can take in founding time, pulse time, and pulse rate values and output a migration matrix,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrix match the sample proportions.
    """
    # Create a model with two founders and one pulse migration
    model = ParametrizedDemography(name="TestModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Add a pulse migration
    model.add_pulse_migration(
        dest_population="target_pop",
        source_population="source_pop1",
        rate_param="pulse_rate",
        time_param="pulse_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions (60% from source_pop1, 40% from source_pop2)
    sample_proportions = {
        "target_pop": [0.6, 0.4]  # [source_pop1, source_pop2]
    }
    
    # Fix the founding rate parameter using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with founding time, pulse time, and pulse rate
    
    founding_time = 10 - 1e-9
    int_found_time = math.ceil(founding_time) 
    test_free_params = [founding_time, 0.2, 5]  # [found_time, pulse_rate, pulse_time]
    test_params = model.parameter_handler.extend_parameters(test_free_params)
 
    assert len(test_params) == 4

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == int_found_time + 1  
    assert matrix.shape[1] == 2  # two source populations
        
    # Verify that the founder rates are greater than 0
    assert matrix[10, 0] > 0
    assert matrix[10, 1] > 0

    # Verify that the pulse migration is applied correctly
    assert np.isclose(matrix[5, 0], 0.2)  # pulse migration at pulse time
    
    # Verify that the final proportions match the sample proportions
    final_proportions = model.proportions_from_matrix(matrix)
    assert np.isclose(final_proportions[0], 0.6)  # source_pop1 proportion
    assert np.isclose(final_proportions[1], 0.4)  # source_pop2 proportion
    
    # Verify that the sum of proportions is 1
    assert np.isclose(final_proportions.sum(), 1.0)


def test_ancestry_fixing_with_pulse_migration_fixed_rate():
    """
    Test the ancestry fixing functionality for a model with two founders and one pulse migration,
    where the pulse migration rate is fixed by the final proportions.
    
    This test:
    1. Creates a model with two founders and one pulse migration,
    2. Sets up sample proportions,
    3. Fixes the founding rate parameter using the sample proportions,
    4. Verifies that the model can take in founding time and pulse time values and output a migration matrix,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrix match the sample proportions.
    """
    # Create a model with two founders and one pulse migration
    model = ParametrizedDemography(name="TestModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Add a pulse migration
    model.add_pulse_migration(
        dest_population="target_pop",
        source_population="source_pop1",
        rate_param="pulse_rate",
        time_param="pulse_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions (60% from source_pop1, 40% from source_pop2)
    sample_proportions = {
        "target_pop": [0.6, 0.4]  # [source_pop1, source_pop2]
    }
    
    # Fix the founding rate parameter and pulse rate using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["pulse_rate"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with founding time and pulse time (since the rates are fixed)
    founding_time = 10 - 1e-9
    int_found_time = math.ceil(founding_time) 
    test_free_params = [0.2, founding_time, 5]  # [found_time, founding_rate, pulse_time]

    test_params = model.parameter_handler.extend_parameters(test_free_params)
 
    assert len(test_params) == 4

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == int_found_time + 1
    assert matrix.shape[1] == 2  # two source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix[10, 0], 0.2)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[10, 1], 0.8)  # source_pop2 proportion at founding time
    
    # Verify that the pulse migration is applied correctly
    # The pulse rate should be calculated to achieve the final proportions
    # We don't know the exact value, but we can verify that it's applied
    assert matrix[5, 0] > 0  # pulse migration at pulse time
    
    # Verify that the final proportions match the sample proportions
    final_proportions = model.proportions_from_matrix(matrix)
    assert np.isclose(final_proportions[0], 0.6)  # source_pop1 proportion
    assert np.isclose(final_proportions[1], 0.4)  # source_pop2 proportion
    
    # Verify that the sum of proportions is 1
    assert np.isclose(final_proportions.sum(), 1.0)


def test_ancestry_fixing_sex_biased():
    """
    Test the ancestry fixing functionality for a sex-biased demography with two founders.
    
    This test:
    1. Creates a sex-biased model with two founders,
    2. Sets up sample proportions for both male and female populations,
    3. Fixes the founding rate and sex-bias parameters using the sample proportions,
    4. Verifies that the model can take in only a founding time value and output migration matrices,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrices match the sample proportions.
    """
    # Create a sex-biased model with two founders
    model = ParametrizedDemographySexBiased(name="SexBiasedModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions
    sample_proportions = {
        "target_pop_autosomal": [0.6, 0.4],  # [source_pop1, source_pop2] for autosomes
        "target_pop_X": [0.51, 0.49]  # [source_pop1, source_pop2] for X chromosomes
    }
    
    # Fix the founding rate and sex-bias parameters using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate1_sex_bias"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding time (since the rates are fixed)
    
    founding_time = 10-1e-15
    int_found_time = math.ceil(founding_time) 
    
    test_free_params = [founding_time]  # Only the founding time
    test_params = model.parameter_handler.extend_parameters(test_free_params)
 
    assert len(test_params) == 3

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both male and female populations
    assert "target_pop_male" in migration_matrices
    assert "target_pop_female" in migration_matrices
    
    # Get the matrices for the male and female populations
    matrix_male = migration_matrices["target_pop_male"]
    matrix_female = migration_matrices["target_pop_female"]
    
    # Verify the matrix dimensions
    assert matrix_male.shape[0] == int_found_time + 1
    assert matrix_male.shape[1] == 2  # two source populations
    assert matrix_female.shape[0] == int_found_time + 1
    assert matrix_female.shape[1] == 2  # two source populations
     

    final_proportions=model.proportions_from_matrices(migration_matrices)
    # Verify that the final proportions match the sample proportions
    for key in final_proportions.keys():
        assert np.allclose(final_proportions[key], sample_proportions[key])

def test_ancestry_fixing_sex_biased_continuous_founder():
    """
    Test the ancestry fixing functionality for a sex-biased demography with two founders.
    
    This test:
    1. Creates a sex-biased model with two founders,
    2. Sets up sample proportions with full parameters for both male and female populations,
    3. Fixes the founding rate and sex-bias parameters using the computed proportions,
    4. Verifies that the model can take in only a founding time value and output migration matrices,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrices match the sample proportions.
    """
    # Create a sex-biased model with two founders
    model = ParametrizedDemographySexBiased(name="SexBiasedModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1","source_pop2":"founder_rate2"},
        remainder_population=None,
        found_time="found_time",
        end_time="end_time"
    )
    
    model_full = ParametrizedDemographySexBiased(name="SexBiasedModel")
    
    # Add a founder event with two source populations
    model_full.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1","source_pop2":"founder_rate2"},
        remainder_population=None,
        found_time="found_time",
        end_time="end_time"
    )

    # Finalize the model
    model.finalize()
    model_full.finalize()

    # Create a parameter list
    rate1=0.4
    bias1 = 1
    rate2=0.4
    bias2=-1
    foundt=10-1e-9
    int_found_time = math.ceil(foundt)
    endt=5
    params_full = [rate1, bias1, rate2, bias2, foundt, endt] 

    migration_matrices = model_full.get_migration_matrices(params_full)

    calculated_proportions = model_full.proportions_from_matrices(migration_matrices)
    
    # Define sample proportions
    sample_proportions = {
        "target_pop_autosomal": calculated_proportions['target_pop_autosomal'] ,  # [source_pop1, source_pop2] for autosomes
        "target_pop_X": calculated_proportions['target_pop_None']   # [source_pop1, source_pop2] for X chromosomes
    }
    
    # Fix the founding rate and sex-bias parameters using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate1_sex_bias"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding time (since the rates are fixed)
    test_free_params = [rate2,bias2,foundt,endt]  
    
    computed_params = model.parameter_handler.compute_params_fixed_by_ancestry(params_full)
    assert np.allclose(computed_params, params_full)
    
    assert (model.parameter_handler.free_parameters_indices == [2,3,4,5])
    assert (model.parameter_handler.params_fixed_by_value_indices.size == 0)
    assert (model.parameter_handler.params_fixed_by_ancestry_indices.tolist() == [0,1])

    raw_test_params = params_full.copy()
    raw_test_params[0]= 0.0  # messing values to be filled in
    raw_test_params[1]=0.  # messing values to be filled in  
    test_params = model.parameter_handler.compute_params_fixed_by_ancestry(raw_test_params)
    
    assert len(test_params) == 6
    assert np.allclose(test_params, params_full)

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both male and female populations
    assert "target_pop_male" in migration_matrices
    assert "target_pop_female" in migration_matrices
    
    # Get the matrices for the male and female populations
    matrix_male = migration_matrices["target_pop_male"]
    matrix_female = migration_matrices["target_pop_female"]
    
    # Verify the matrix dimensions
    assert matrix_male.shape[0] == int_found_time + 1
    assert matrix_male.shape[1] == 2  # two source populations
    assert matrix_female.shape[0] == int_found_time + 1
    assert matrix_female.shape[1] == 2  # two source populations

    final_proportions=model.proportions_from_matrices(migration_matrices)
    # Verify that the final proportions match the sample proportions
    for key in final_proportions.keys():
        #if not np.allclose(final_proportions[key], sample_proportions[key]):
        #    
        assert np.allclose(final_proportions[key], sample_proportions[key])



def test_ancestry_fixing_sex_biased_with_pulse():
    """
    Test the ancestry fixing functionality for a sex-biased demography with two founders and one pulse migration.
    
    This test:
    1. Creates a sex-biased model with two founders and one pulse migration,
    2. Sets up sample proportions for both male and female populations,
    3. Fixes the pulse rate and sex-bias parameters using the sample proportions,
    4. Verifies that the model can take in founding time and pulse time values and output migration matrices,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrices match the sample proportions.
    """
    # Create a sex-biased model with two founders and one pulse migration
    model = ParametrizedDemographySexBiased(name="SexBiasedPulseModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Add a pulse migration
    model.add_pulse_migration(
        dest_population="target_pop",
        source_population="source_pop1",
        rate_param="pulse_rate",
        time_param="pulse_time"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions for X and autosomes
    sample_proportions = {
        "target_pop_autosomal": [0.7, 0.3],  
        "target_pop_X": [0.7, 0.3]  
    }
    
    # Fix the pulse rate and sex-bias parameters using the sample proportions
    model.set_up_fixed_parameters(
        params_to_fix_by_ancestry=["pulse_rate", "pulse_rate_sex_bias"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with founding time, founding rate, and pulse time (since the pulse rate is fixed)
    found_time = 10-1e-9
    int_found_time = math.ceil(found_time)
    
    test_free_params = [0.5, 0, found_time, 5]  # [founder_rate, founder_rate_sex_bias, found_time, pulse_time]
    
    test_params = model.parameter_handler.extend_parameters(test_free_params)
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both male and female populations
    assert "target_pop_male" in migration_matrices
    assert "target_pop_female" in migration_matrices
    
    # Get the matrices for the male and female populations
    matrix_male = migration_matrices["target_pop_male"]
    matrix_female = migration_matrices["target_pop_female"]
    
    # Verify the matrix dimensions
    assert matrix_male.shape[0] == int_found_time + 1
    assert matrix_male.shape[1] == 2  # two source populations
    assert matrix_female.shape[0] == int_found_time + 1
    assert matrix_female.shape[1] == 2  # two source populations
    
    # Verify that the founder rates are greater than 0
    assert matrix_male[10, 0] > 0
    assert matrix_male[10, 1] > 0
    assert matrix_female[10, 0] > 0
    assert matrix_female[10, 1] > 0
    
    # Verify that the pulse migration is applied correctly
    # The pulse rate should be calculated to achieve the final proportions
    assert matrix_male[5, 0] > 0  # pulse migration at pulse time for males
    assert matrix_female[5, 0] > 0  # pulse migration at pulse time for females

    # Verify that the final proportions match the sample proportions 
    final_proportions=model.proportions_from_matrices(migration_matrices)
    for key in final_proportions.keys():
        assert np.allclose(final_proportions[key], sample_proportions[key])


def test_parameter_fixing_single_population():
    """
    Test the parameter fixing functionality for a single population with two founders.
    
    This test:
    1. Creates a model with two founders and no other events,
    2. Sets up sample proportions,
    3. Fixes the founding rate parameter using the sample proportions,
    4. Verifies that the model can take in only a founding time value and output a migration matrix,
    5. Verifies that the resulting rate parameter matches the sample proportions,
    6. Verifies that the final proportions of the matrix match the sample proportions.
    """
    # Create a model with two founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add a founder event with two source populations
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time"
    )
    
    # Finalize the model
    model.finalize()

    # Define sample proportions (70% from source_pop1, 30% from source_pop2)
    sample_proportions = {
        "target_pop": [0.7, 0.3]  # [source_pop1, source_pop2]
    }
    
    found_time = 10-1e-9
    int_found_time = math.ceil(found_time)
    params_to_fix_by_value = {"found_time":found_time}

    # Fix the founding rate parameter using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions, user_params_to_fix_by_value = params_to_fix_by_value
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with the remaining parameters (since there are no free paramaters left)
    test_free_params = []
    
    test_params = model.parameter_handler.extend_parameters(test_free_params)

    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == int_found_time + 1
    assert matrix.shape[1] == 2  # two source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix[10, 0], 0.7)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[10, 1], 0.3)  # source_pop2 proportion at founding time
    
    # Verify that the final proportions match the sample proportions
    final_proportions = model.proportions_from_matrix(matrix)
    assert np.isclose(final_proportions[0], 0.7)  # source_pop1 proportion
    assert np.isclose(final_proportions[1], 0.3)  # source_pop2 proportion
    
    # Verify that the sum of proportions is 1
    assert np.isclose(final_proportions.sum(), 1.0)


def test_ancestry_fixing_multiple_populations_v2():
    """
    Test the ancestry fixing functionality for multiple populations.
    
    This test:
    1. Creates a model with two populations, each with two founders,
    2. Sets up sample proportions for each population,
    3. Fixes the founding rate parameters using the sample proportions,
    4. Verifies that the model can take in only founding time values and output migration matrices,
    5. Verifies that the resulting rate parameters match the sample proportions,
    6. Verifies that the final proportions of the matrices match the sample proportions.
    """
    # Create a model with two populations, each with two founders
    model = ParametrizedDemography(name="TestModel")
    
    # Add founder events for two populations
    model.add_founder_event(
        dest_population="target_pop1",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time1"
    )
    
    model.add_founder_event(
        dest_population="target_pop2",
        source_populations={"source_pop2": "founder_rate2"},
        remainder_population="source_pop1",
        found_time="found_time2"
    )
    
    # Finalize the model
    model.finalize()
    
    # Define sample proportions
    sample_proportions = {
        "target_pop1": [0.7, 0.3],  # [source_pop1, source_pop2]
        "target_pop2": [0.4, 0.6]   # [source_pop1, source_pop2]
    }
    
    # Fix the founding rate parameters using the sample proportions
    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.parameter_handler.has_known_proportions
    
    # Create a parameter list with only the founding times (since the rates are fixed)
    
    found_time1 = 10-1e-9
    found_time2 = 15-1e-9
    int_found_time1 = math.ceil(found_time1)
    int_found_time2 = math.ceil(found_time2)
    
    test_free_params = [found_time1, found_time2]  # [found_time1, found_time2]
    test_params = model.parameter_handler.extend_parameters(test_free_params)
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both target populations
    assert "target_pop1" in migration_matrices
    assert "target_pop2" in migration_matrices
    
    # Get the matrices for the target populations
    matrix1 = migration_matrices["target_pop1"]
    matrix2 = migration_matrices["target_pop2"]
    
    # Verify the matrix dimensions
    assert matrix1.shape[0] == int_found_time1 + 1
    assert matrix1.shape[1] == 2  # two source populations
    assert matrix2.shape[0] == int_found_time2 + 1
    assert matrix2.shape[1] == 2  # two source populations
    
    # Verify that the founder rates match the sample proportions
    assert np.isclose(matrix1[10, 0], 0.7)  # source_pop1 proportion at founding time for target_pop1
    assert np.isclose(matrix1[10, 1], 0.3)  # source_pop2 proportion at founding time for target_pop1
    assert np.isclose(matrix2[15, 0], 0.4)  # source_pop1 proportion at founding time for target_pop2
    assert np.isclose(matrix2[15, 1], 0.6)  # source_pop2 proportion at founding time for target_pop2
    
    # Verify that the final proportions match the sample proportions
    final_proportions1 = model.proportions_from_matrix(matrix1)
    final_proportions2 = model.proportions_from_matrix(matrix2)
    
    assert np.isclose(final_proportions1[0], 0.7)  # source_pop1 proportion for target_pop1
    assert np.isclose(final_proportions1[1], 0.3)  # source_pop2 proportion for target_pop1
    assert np.isclose(final_proportions2[0], 0.4)  # source_pop1 proportion for target_pop2
    assert np.isclose(final_proportions2[1], 0.6)  # source_pop2 proportion for target_pop2
    
    # Verify that the sum of proportions is 1 for each population
    assert np.isclose(final_proportions1.sum(), 1.0)
    assert np.isclose(final_proportions2.sum(), 1.0)


# -------- Tests for optimize_rates_to_match_ancestry --------

def _make_single_pop_model(sample_proportions=None, params_to_fix_by_ancestry=("founder_rate1",)):
    """Helper: single target_pop with two source populations."""
    model = ParametrizedDemography(name="TestModel")
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time",
    )
    model.finalize()
    if sample_proportions is not None:
        model.parameter_handler.set_up_fixed_parameters(
            demography=model,
            params_to_fix_by_ancestry=list(params_to_fix_by_ancestry),
            proportions=sample_proportions,
        )
    return model


def test_optimize_rates_identity_single_pop():
    """Starting from parameters that already match the target proportions produces no change."""
    found_time = 10 - 1e-9
    true_params = [0.7, found_time]  # [founder_rate1, found_time]

    # Derive target proportions from the true parameters so the starting point is exact.
    model_ref = _make_single_pop_model()
    matrices = model_ref.get_migration_matrices(true_params)
    sample_proportions = {
        pop: list(prop) for pop, prop in model_ref.proportions_from_matrices(matrices).items()
    }

    model = _make_single_pop_model(sample_proportions=sample_proportions)
    result = model.parameter_handler.optimize_rates_to_match_ancestry(true_params)

    assert np.allclose(result, true_params, atol=1e-4)


def test_optimize_rates_convergence_single_pop():
    """Starting from a wrong rate converges to the rate that matches the target proportions."""
    found_time = 10 - 1e-9
    true_params = [0.7, found_time]
    wrong_params = [0.3, found_time]  # wrong rate, correct time

    model_ref = _make_single_pop_model()
    matrices = model_ref.get_migration_matrices(true_params)
    sample_proportions = {
        pop: list(prop) for pop, prop in model_ref.proportions_from_matrices(matrices).items()
    }

    model = _make_single_pop_model(sample_proportions=sample_proportions)
    result = model.parameter_handler.optimize_rates_to_match_ancestry(wrong_params)

    assert np.isclose(result[0], true_params[0], atol=1e-3), \
        f"Rate did not converge: expected {true_params[0]}, got {result[0]}"


def test_optimize_rates_time_params_unchanged():
    """TIME parameters are never modified by optimize_rates_to_match_ancestry."""
    found_time = 10 - 1e-9
    true_params = [0.7, found_time]

    model_ref = _make_single_pop_model()
    matrices = model_ref.get_migration_matrices(true_params)
    sample_proportions = {
        pop: list(prop) for pop, prop in model_ref.proportions_from_matrices(matrices).items()
    }

    model = _make_single_pop_model(sample_proportions=sample_proportions)
    # Start from deliberately wrong rate; time should be preserved regardless.
    result = model.parameter_handler.optimize_rates_to_match_ancestry([0.2, found_time])

    assert np.isclose(result[1], found_time), \
        f"TIME parameter was modified: expected {found_time}, got {result[1]}"


def test_optimize_rates_identity_sex_biased():
    """Identity holds for a sex-biased model: starting from parameters that already match the
    target proportions produces no change.

    We hardcode the proportions (matching the convention used by the existing sex-biased tests,
    where the non-autosomal key is 'target_pop_X'), then recover the corresponding true parameters
    via compute_params_fixed_by_ancestry and verify that optimising from those params is a no-op.
    """
    model = ParametrizedDemographySexBiased(name="SexBiasedModel")
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time",
    )
    model.finalize()

    # Use the same key convention as the existing passing sex-biased tests.
    sample_proportions = {
        "target_pop_autosomal": [0.6, 0.4],
        "target_pop_X": [0.55, 0.45],
    }

    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate1_sex_bias"],
        proportions=sample_proportions,
    )

    found_time = 10 - 1e-15
    # params order: [founder_rate1 (RATE), founder_rate1_sex_bias (SEX_BIAS), found_time (TIME)]
    # Use compute_params_fixed_by_ancestry to obtain the exact params that reproduce the target
    # proportions, so the identity test starts from a truly consistent point.
    seed_params = [0.6, 0.0, found_time]
    true_params = model.parameter_handler.compute_params_fixed_by_ancestry(seed_params)

    result = model.parameter_handler.optimize_rates_to_match_ancestry(true_params)

    assert np.allclose(result, true_params, atol=1e-4)


def test_optimize_rates_identity_with_pulse_migration():
    """Identity holds for a model with a pulse migration: starting from the true parameters produces no change."""
    model = ParametrizedDemography(name="TestModel")
    model.add_founder_event(
        dest_population="target_pop",
        source_populations={"source_pop1": "founder_rate1"},
        remainder_population="source_pop2",
        found_time="found_time",
    )
    model.add_pulse_migration(
        dest_population="target_pop",
        source_population="source_pop1",
        rate_param="pulse_rate",
        time_param="pulse_time",
    )
    model.finalize()

    found_time = 10 - 1e-9
    # params order: [founder_rate1 (RATE), found_time (TIME), pulse_rate (RATE), pulse_time (TIME)]
    true_params = [0.5, found_time, 0.1, 5.0]

    matrices = model.get_migration_matrices(true_params)
    sample_proportions = {
        pop: list(prop) for pop, prop in model.proportions_from_matrices(matrices).items()
    }

    model.parameter_handler.set_up_fixed_parameters(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions,
    )

    result = model.parameter_handler.optimize_rates_to_match_ancestry(true_params)

    assert np.allclose(result, true_params, atol=1e-3)