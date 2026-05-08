import os
import pytest
from tracts.driver import run_tracts
from pathlib import Path
import numpy as np

def _run_tracts_test(driver_file: str, script_dir: Path, output_dir: Path, log_name: str, expected_files: list[str]):
    """
    Helper method to test run_tracts with specific driver file and expected outputs.
    """

    # Run tracts
    run_tracts(driver_file, script_dir=script_dir)

    # ------------ log file checks ------------

    # Verify output directory exists
    assert output_dir.exists(), f"Output directory not created. Current output_dir is: {output_dir}"

    # Verify output files exist (check for expected outputs)
    output_files = os.listdir(output_dir)
    assert len(output_files) > 0, "No output files created."

    # Verify there is one .log file
    log_files = [f for f in output_files if f.endswith(".log")]
    assert len(log_files) == 1, (
        f"One .log file should be created in output directory. "
        f"There are currently {len(log_files)} .log files."
    )

    # Verify log file name matches log_name
    assert log_files[0] == log_name, f"Log file name {log_files[0]} does not match expected {log_name}"

    # Verify log file contains expected content
    log_path = output_dir / log_name
    with open(log_path, "r") as f:
        log_content = f.read()
        assert len(log_content) > 0, "Log file is empty."

    # Delete log file after checks
    os.remove(log_path)

    # ------------ output files checks ------------

    # Verify expected output files exist (if any expected)
    for expected_file in expected_files:
        file_path = output_dir / expected_file
        assert file_path.exists(), f"Expected file '{expected_file}' not found in output directory."

    # Clean up output directory
    for entry in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, entry))
    os.rmdir(output_dir)


def test_run_tracts():
    """
    Test that run_tracts creates all outputs and log file in specified directory, for a set of driver files covering all possible
    configurations regarding optimization.
    """

    script_dir = Path(__file__).resolve().parent / "drivers"
    driver_files_autosomes = sorted(
        [f.name for f in script_dir.iterdir() if "autosomes" in f.name]
    )
    driver_files_allosomes = sorted(
        [f.name for f in script_dir.iterdir() if "allosomes" in f.name]
    )

    output_dir = Path(__file__).resolve().parent / "drivers" / "test_output"
    log_name = "test_logfile.log"

    expected_files_with_allosomes = [
        "test_output_optimal_parameters.txt",
        "test_output_autosomes_all_populations.png",
        "test_output_female_allosomes_all_populations.png",
        "test_output_male_allosomes_all_populations.png",
        "test_output_male_allosome_predicted_tract_distribution",
        "test_output_female_allosome_predicted_tract_distribution",
        "test_output_male_allosome_sample_tract_distribution",
        "test_output_female_allosome_sample_tract_distribution",
        "test_output_tract_length_allosome_bins",
        "test_output_autosome_predicted_tract_distribution",
        "test_output_male_migration_matrix",
        "test_output_female_migration_matrix",
        "test_output_autosome_sample_tract_distribution",
        "test_output_tract_length_autosome_bins",
    ]

    expected_files_without_allosomes = [
        "test_output_optimal_parameters.txt",
        "test_output_autosomes_all_populations.png",
        "test_output_autosome_predicted_tract_distribution",
        "test_output_male_migration_matrix",
        "test_output_female_migration_matrix",
        "test_output_autosome_sample_tract_distribution",
        "test_output_tract_length_autosome_bins",
    ]

    for driver_file in driver_files_autosomes:
        _run_tracts_test(driver_file, script_dir, output_dir, log_name, expected_files_without_allosomes)

    for driver_file in driver_files_allosomes:
        _run_tracts_test(driver_file, script_dir, output_dir, log_name, expected_files_with_allosomes)


def _compare_driver_results(driver_files: list[str], script_dir: Path, output_dir: Path, log_name: str, tolerance: float = 0.01):
    """
    Helper method to compare results from two driver files.
    """
    results = {}

    # Run both driver files and collect results
    for driver_file in driver_files:
        run_tracts(driver_file, script_dir=script_dir)
        
        # Collect output files
        results[driver_file] = {}
        
        # Read optimal parameters: second column only, ignoring header
        params_file = output_dir / "test_output_optimal_parameters.txt"
        results[driver_file]["params"] = np.atleast_1d(
            np.loadtxt(params_file, skiprows=1, usecols=1)
        )
        
        # Read migration matrices
        male_mig_file = output_dir / "test_output_male_migration_matrix"
        female_mig_file = output_dir / "test_output_female_migration_matrix"
        with open(male_mig_file, "r") as f:
            results[driver_file]["male_mig"] = np.loadtxt(f)
        with open(female_mig_file, "r") as f:
            results[driver_file]["female_mig"] = np.loadtxt(f)
        
        # Read tract distribution
        tract_file = output_dir / "test_output_autosome_predicted_tract_distribution"
        with open(tract_file, "r") as f:
            results[driver_file]["tract_dist"] = np.loadtxt(f)
        
        # Clean up
        if (output_dir / log_name).exists():
            os.remove(output_dir / log_name)
        for entry in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, entry))
        os.rmdir(output_dir)

    # Compare optimal parameters
    params_0 = results[driver_files[0]]["params"]
    params_1 = results[driver_files[1]]["params"]
    assert params_0.shape == params_1.shape, (
        "Optimal parameter arrays have different shapes: "
        f"{params_0.shape} vs {params_1.shape}"
    )
    params_diff = np.abs(params_0 - params_1)
    params_rel_diff = params_diff / (np.abs(params_0) + 1e-10)
    assert np.max(params_rel_diff) < tolerance, (
        f"Optimal parameters differ by more than {tolerance*100}%. "
        f"Max relative difference: {np.max(params_rel_diff)*100:.2f}%"
    )

    # Compare male migration matrices
    male_mig_diff = np.abs(results[driver_files[0]]["male_mig"] - results[driver_files[1]]["male_mig"])
    male_mig_rel_diff = male_mig_diff / (np.abs(results[driver_files[0]]["male_mig"]) + 1e-10)
    assert np.max(male_mig_rel_diff) < tolerance, (
        f"Male migration matrices differ by more than {tolerance*100}%. "
        f"Max relative difference: {np.max(male_mig_rel_diff)*100:.2f}%"
    )

    # Compare female migration matrices
    female_mig_diff = np.abs(results[driver_files[0]]["female_mig"] - results[driver_files[1]]["female_mig"])
    female_mig_rel_diff = female_mig_diff / (np.abs(results[driver_files[0]]["female_mig"]) + 1e-10)
    assert np.max(female_mig_rel_diff) < tolerance, (
        f"Female migration matrices differ by more than {tolerance*100}%. "
        f"Max relative difference: {np.max(female_mig_rel_diff)*100:.2f}%"
    )

    # Compare tract distributions
    tract_diff = np.abs(results[driver_files[0]]["tract_dist"] - results[driver_files[1]]["tract_dist"])
    tract_rel_diff = tract_diff / (np.abs(results[driver_files[0]]["tract_dist"]) + 1e-10)
    assert np.max(tract_rel_diff) < tolerance, (
        f"Tract distributions differ by more than {tolerance*100}%. "
        f"Max relative difference: {np.max(tract_rel_diff)*100:.2f}%"
    )


def test_compare_only_autosomal_one_step_vs_two_steps():
    """
    Test that one_step and two_steps optimizations produce very similar results when only autosomes are present in the sample.
    Optimizations are expected to be equivalent in this context: the two-steps optimization is expected to stop after the first step,
    optimizing only over autosomal data. The test compares migration matrices, optimal parameters and tract distributions.
    Performs the comparison with and without parameters fixed by ancestry.
    """
    script_dir = Path(__file__).resolve().parent / "drivers"
    output_dir = Path(__file__).resolve().parent / "drivers" / "test_output"
    log_name = "test_logfile.log"

    # No parameters fixed by ancestry
    driver_files = ["test_one_step_only_autosomes.yaml", "test_two_steps_only_autosomes.yaml"]
    _compare_driver_results(driver_files, script_dir, output_dir, log_name)

    # Parameters fixed by ancestry
    driver_files_fix = ["test_one_step_only_autosomes_fix.yaml", "test_two_steps_only_autosomes_fix.yaml"]
    _compare_driver_results(driver_files_fix, script_dir, output_dir, log_name)

        



