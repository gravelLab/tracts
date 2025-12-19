import os
import sys
import numpy as np
import pytest

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+'\\..')
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.demography.parameter import ParamType

def test_ancestry_fixing_single_population():
    """
    Test the ancestry fixing functionality for a single population with two founders.
    
    This test:
    1. Creates a model with two founders and no other events
    2. Sets up sample proportions
    3. Fixes the founding rate parameter using the sample proportions
    4. Verifies that the model can take in only a founding time value and output a migration matrix
    5. Verifies that the resulting rate parameter matches the sample proportions
    6. Verifies that the final proportions of the matrix match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding time (since the rate is fixed)
    test_params = [10]  # Only the founding time
    
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
    assert np.isclose(matrix[10, 0], 0.7)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[10, 1], 0.3)  # source_pop2 proportion at founding time
    
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
    1. Creates a model with two populations, each with two founders
    2. Sets up sample proportions for each population
    3. Fixes the founding rate parameters using the sample proportions
    4. Verifies that the model can take in only founding time values and output migration matrices
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrices match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding times (since the rates are fixed)
    test_params = [10, 15]  # [found_time1, found_time2]
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both target populations
    assert "target_pop1" in migration_matrices
    assert "target_pop2" in migration_matrices
    
    # Get the matrices for the target populations
    matrix1 = migration_matrices["target_pop1"]
    matrix2 = migration_matrices["target_pop2"]
    
    # Verify the matrix dimensions
    assert matrix1.shape[0] == 11  # found_time1 + 1
    assert matrix1.shape[1] == 2  # two source populations
    assert matrix2.shape[0] == 16  # found_time2 + 1
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
    1. Creates a model with one population and three founders
    2. Sets up sample proportions
    3. Fixes the founding rate parameters using the sample proportions
    4. Verifies that the model can take in only a founding time value and output a migration matrix
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrix match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding time (since the rates are fixed)
    test_params = [10]  # Only the founding time
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == 11  # found_time + 1
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
    1. Creates a model with two populations, each with three founders
    2. Sets up sample proportions for each population
    3. Fixes the founding rate parameters using the sample proportions
    4. Verifies that the model can take in only founding time values and output migration matrices
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrices match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1_pop1", "founder_rate2_pop1", "founder_rate1_pop2", "founder_rate2_pop2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding times (since the rates are fixed)
    test_params = [10, 15]  # [found_time1, found_time2]
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both target populations
    assert "target_pop1" in migration_matrices
    assert "target_pop2" in migration_matrices
    
    # Get the matrices for the target populations
    matrix1 = migration_matrices["target_pop1"]
    matrix2 = migration_matrices["target_pop2"]
    
    # Verify the matrix dimensions
    assert matrix1.shape[0] == 11  # found_time1 + 1
    assert matrix1.shape[1] == 3  # three source populations
    assert matrix2.shape[0] == 16  # found_time2 + 1
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
    1. Creates a model with two founders and one pulse migration
    2. Sets up sample proportions
    3. Fixes the founding rate parameter using the sample proportions
    4. Verifies that the model can take in founding time, pulse time, and pulse rate values and output a migration matrix
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrix match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with founding time, pulse time, and pulse rate
    test_params = [10, 0.2, 5]  # [found_time, pulse_rate, pulse_time]
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got a migration matrix for the target population
    assert "target_pop" in migration_matrices
    
    # Get the matrix for the target population
    matrix = migration_matrices["target_pop"]
    
    # Verify the matrix dimensions
    assert matrix.shape[0] == 11  # found_time + 1
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
    1. Creates a model with two founders and one pulse migration
    2. Sets up sample proportions
    3. Fixes the founding rate parameter using the sample proportions
    4. Verifies that the model can take in founding time and pulse time values and output a migration matrix
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrix match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["pulse_rate"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with founding time and pulse time (since the rates are fixed)
    test_params = [0.2, 10, 5]  # [found_time, founding_rate, pulse_time]
    
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
    1. Creates a sex-biased model with two founders
    2. Sets up sample proportions for both male and female populations
    3. Fixes the founding rate and sex-bias parameters using the sample proportions
    4. Verifies that the model can take in only a founding time value and output migration matrices
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrices match the sample proportions
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
        "target_pop_X": [0.4, 0.6]  # [source_pop1, source_pop2] for X chromosomes
    }
    
    # Fix the founding rate and sex-bias parameters using the sample proportions
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate1_sex_bias"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding time (since the rates are fixed)
    test_params = [10]  # Only the founding time
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both male and female populations
    assert "target_pop_male" in migration_matrices
    assert "target_pop_female" in migration_matrices
    
    # Get the matrices for the male and female populations
    matrix_male = migration_matrices["target_pop_male"]
    matrix_female = migration_matrices["target_pop_female"]
    
    # Verify the matrix dimensions
    assert matrix_male.shape[0] == 11  # found_time + 1
    assert matrix_male.shape[1] == 2  # two source populations
    assert matrix_female.shape[0] == 11  # found_time + 1
    assert matrix_female.shape[1] == 2  # two source populations
     

    final_proportions=model.proportions_from_matrices(migration_matrices)
    # Verify that the final proportions match the sample proportions
    for key in final_proportions.keys():
        assert np.allclose(final_proportions[key], sample_proportions[key])

def test_ancestry_fixing_sex_biased_continuous_founder():
    """
    Test the ancestry fixing functionality for a sex-biased demography with two founders.
    
    This test:
    1. Creates a sex-biased model with two founders
    2. Sets up sample proportions with full parameters for both male and female populations
    3. Fixes the founding rate and sex-bias parameters using the computed
    4. Verifies that the model can take in only a founding time value and output migration matrices
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrices match the sample proportions
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
    foundt=10
    endt=5
    params_full = [rate1,bias1, rate2,bias2,foundt,endt] 

    migration_matrices = model_full.get_migration_matrices(params_full)

    calculated_proportions = model_full.proportions_from_matrices(migration_matrices)
    
    # Define sample proportions
    sample_proportions = {
        "target_pop_autosomal": calculated_proportions['target_pop_autosomal'] ,  # [source_pop1, source_pop2] for autosomes
        "target_pop_X": calculated_proportions['target_pop_None']   # [source_pop1, source_pop2] for X chromosomes
    }
    
    # Fix the founding rate and sex-bias parameters using the sample proportions
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate1_sex_bias"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding time (since the rates are fixed)

    test_params = [rate2,bias2,foundt,endt]  
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both male and female populations
    assert "target_pop_male" in migration_matrices
    assert "target_pop_female" in migration_matrices
    
    # Get the matrices for the male and female populations
    matrix_male = migration_matrices["target_pop_male"]
    matrix_female = migration_matrices["target_pop_female"]
    
    # Verify the matrix dimensions
    assert matrix_male.shape[0] == 11  # found_time + 1
    assert matrix_male.shape[1] == 2  # two source populations
    assert matrix_female.shape[0] == 11  # found_time + 1
    assert matrix_female.shape[1] == 2  # two source populations
     

    final_proportions=model.proportions_from_matrices(migration_matrices)
    # Verify that the final proportions match the sample proportions
    for key in final_proportions.keys():
        #if not np.allclose(final_proportions[key], sample_proportions[key]):
        #    breakpoint()
        assert np.allclose(final_proportions[key], sample_proportions[key])



def test_ancestry_fixing_sex_biased_with_pulse():
    """
    Test the ancestry fixing functionality for a sex-biased demography with two founders and one pulse migration.
    
    This test:
    1. Creates a sex-biased model with two founders and one pulse migration
    2. Sets up sample proportions for both male and female populations
    3. Fixes the pulse rate and sex-bias parameters using the sample proportions
    4. Verifies that the model can take in founding time and pulse time values and output migration matrices
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrices match the sample proportions
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
    model.set_up_fixed_ancestry_proportions(
        params_to_fix=["pulse_rate", "pulse_rate_sex_bias"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with founding time, founding rate, and pulse time (since the pulse rate is fixed)
    test_params = [0.5, 0, 10, 5]  # [founder_rate, founder_rate_sex_bias, found_time, pulse_time]
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both male and female populations
    assert "target_pop_male" in migration_matrices
    assert "target_pop_female" in migration_matrices
    
    # Get the matrices for the male and female populations
    matrix_male = migration_matrices["target_pop_male"]
    matrix_female = migration_matrices["target_pop_female"]
    
 


    # Verify the matrix dimensions
    assert matrix_male.shape[0] == 11  # found_time + 1
    assert matrix_male.shape[1] == 2  # two source populations
    assert matrix_female.shape[0] == 11  # found_time + 1
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



## Test parameter fixing
# 
# 
def test_parameter_fixing_single_population():
    """
    Test the parameter fixing functionality for a single population with two founders.
    
    This test:
    1. Creates a model with two founders and no other events
    2. Sets up sample proportions
    3. Fixes the founding rate parameter using the sample proportions
    4. Verifies that the model can take in only a founding time value and output a migration matrix
    5. Verifies that the resulting rate parameter matches the sample proportions
    6. Verifies that the final proportions of the matrix match the sample proportions
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
    
    params_to_fix_by_value = {"found_time":10}

    # Fix the founding rate parameter using the sample proportions
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1"],
        proportions=sample_proportions, params_to_fix_by_value = params_to_fix_by_value
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with the remaining parameters (since there are no free paramaters left)
    test_params = []  
    
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
    assert np.isclose(matrix[10, 0], 0.7)  # source_pop1 proportion at founding time
    assert np.isclose(matrix[10, 1], 0.3)  # source_pop2 proportion at founding time
    
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
    1. Creates a model with two populations, each with two founders
    2. Sets up sample proportions for each population
    3. Fixes the founding rate parameters using the sample proportions
    4. Verifies that the model can take in only founding time values and output migration matrices
    5. Verifies that the resulting rate parameters match the sample proportions
    6. Verifies that the final proportions of the matrices match the sample proportions
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
    model.fixed_proportions_handler.set_up_fixed_ancestry_proportions(
        demography=model,
        params_to_fix_by_ancestry=["founder_rate1", "founder_rate2"],
        proportions=sample_proportions
    )
    
    # Verify that the model has been fixed
    assert model.fixed_proportions_handler.has_been_fixed
    
    # Create a parameter list with only the founding times (since the rates are fixed)
    test_params = [10, 15]  # [found_time1, found_time2]
    
    # Get the migration matrices
    migration_matrices = model.get_migration_matrices(test_params)
    
    # Verify that we got migration matrices for both target populations
    assert "target_pop1" in migration_matrices
    assert "target_pop2" in migration_matrices
    
    # Get the matrices for the target populations
    matrix1 = migration_matrices["target_pop1"]
    matrix2 = migration_matrices["target_pop2"]
    
    # Verify the matrix dimensions
    assert matrix1.shape[0] == 11  # found_time1 + 1
    assert matrix1.shape[1] == 2  # two source populations
    assert matrix2.shape[0] == 16  # found_time2 + 1
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