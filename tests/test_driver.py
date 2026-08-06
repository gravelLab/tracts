import os
import pytest
import shutil
from collections import OrderedDict
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

import tracts.driver as driver_module
import tracts.driver_utils as driver_utils_module
from tracts.driver import run_tracts
from tracts.demography.parameter import ParamType
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.driver_utils import _print_run_intro as real_print_run_intro

# ------------ Helper functions for test setup and checks ----------

def _copy_tests_to_tmp(tmp_path: Path) -> Path:

    source_tests = Path(__file__).resolve().parent
    tmp_tests = tmp_path / "tests"
    tmp_tests.mkdir(parents=True, exist_ok=True)

    required_subdirs = ("drivers", "models", "data")
    ignore = shutil.ignore_patterns("test_output", "__pycache__")

    for subdir in required_subdirs:
        source_subdir = source_tests / subdir
        if source_subdir.exists():
            shutil.copytree(
                source_subdir,
                tmp_tests / subdir,
                ignore=ignore,
            )

    return tmp_tests / "drivers"


def _prepare_driver(driver_path: Path, output_dir: Path) -> str:

    text = driver_path.read_text()

    lines = text.splitlines()
    new_lines = []
    found_output_directory = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("output_directory:"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}output_directory: '{output_dir}'")
            found_output_directory = True
        else:
            new_lines.append(line)

    if not found_output_directory:
        new_lines.append(f"output_directory: '{output_dir}'")

    driver_path.write_text("\n".join(new_lines) + "\n")

    return driver_path.name


def _clean_output_dir(output_dir: Path):

    if output_dir.exists():
        shutil.rmtree(output_dir)


def _make_mock_driver_spec(tmp_path: Path, two_steps_optimization: bool, autosomes_in_step_2: bool):
    """
    Return a minimal driver-spec SimpleNamespace that covers the two-step and allosome flags.
    The specific values are not important, but they should be plausible and consistent with the expected types.
    """
    return SimpleNamespace(
        samples=SimpleNamespace(allosomes=["X"]),
        models=SimpleNamespace(
            model_filename="test_model.yaml",
            ad_model_autosomes="DC",
            ad_model_allosomes="DC",
            rho_f=1,
            rho_m=1,
            TP=2,
        ),
        start_params=SimpleNamespace(),
        bounds=SimpleNamespace(),
        optim=SimpleNamespace(
            seed=1,
            repetitions=2,
            maximum_iterations=2,
            npts=5,
            exclude_tracts_below_cm=0,
            fix_parameters_from_ancestry_proportions=[],
            fix_parameters_by_value={},
            unknown_labels_for_smoothing=[],
            two_steps_optimization=two_steps_optimization,
            autosomes_in_step_2=autosomes_in_step_2,
            use_autosomes_for_sex_bias=autosomes_in_step_2,
            N_cores=5,
            boundary_tol=0.3,
            n_reoptimizations=0,
            rerun_optimization_on_boundaries=True,
            reoptimization_likelihood_tolerance=1e-3,
            repetitions_likelihood_tolerance=0.5,
            bounds_proximity_tol=0.05,
        ),
        output=SimpleNamespace(
            output_filename_format="test_output_{label}",
            log_filename="test_logfile.log",
            output_directory=str(tmp_path / "test_output"),
            verbose_log=0,
            verbose_screen=0,
            log_scale=False,
        ),
    )


def _make_mock_model():
    """
    Return a minimal mock model (spec'd to ParametrizedDemographySexBiased so it
    satisfies GeneticModel's isinstance check) with four parameters:
      - t        (TIME,     index 0)
      - rate_eur (RATE,     index 1)
      - sb_eur   (SEX_BIAS, index 2)
      - sb_afr   (SEX_BIAS, index 3)

    Indices 0–1 are non-sex-bias (replaced by Step 1 best in two-step runs).
    Indices 2–3 are sex-bias (kept run-specific in Step 2).
    """
    model = MagicMock(spec=ParametrizedDemographySexBiased)
    model.model_base_params = OrderedDict([
        ("t", SimpleNamespace(index=0, type=ParamType.TIME, bounds=ParamType.TIME.bounds)),
        ("rate_eur", SimpleNamespace(index=1, type=ParamType.RATE, bounds=ParamType.RATE.bounds)),
        ("sb_eur", SimpleNamespace(index=2, type=ParamType.SEX_BIAS, bounds=ParamType.SEX_BIAS.bounds)),
        ("sb_afr", SimpleNamespace(index=3, type=ParamType.SEX_BIAS, bounds=ParamType.SEX_BIAS.bounds)),
    ])
    model.population_indices = OrderedDict([("A", 0), ("B", 1)])
    model.parametrized_populations = ["pop"]
    model.founder_events = {}
    model.parameter_handler = SimpleNamespace(
        to_physical_params_functions={},
        to_optimizer_params_functions={},
        enable_time_param_logging=True,
        convert_to_optimizer_params=lambda params: np.array(params, dtype=float),
        convert_to_physical_params=lambda params, report_non_admissible=False: np.array(params, dtype=float),
        set_up_fixed_parameters=lambda *args, **kwargs: None,
        release_fixed_parameters=lambda *args, **kwargs: None,
        add_fixed_parameters=lambda *args, **kwargs: None,
        params_fixed_by_ancestry=[],
        user_params_fixed_by_value={},
    )
    model.proportions_from_matrices = lambda matrices: {"A": np.array([1.0])}
    model.get_violation_score = lambda params, verbose=False: 1.0
    model.get_migration_matrices = lambda params: {"female": np.zeros((1, 1)), "male": np.zeros((1, 1))}
    model.set_up_fixed_parameters = lambda *args, **kwargs: None
    return model


def _load_demographic_model_from_driver_result(model):
    """
    Mirrors the derived-name computation performed by ``load_demographic_model_from_driver``,
    so that mocked ``model`` instances can stand in for its 4-tuple return value.
    """
    model_param_names = list(model.model_base_params.keys())
    sex_bias_param_names = [
        name for name, info in model.model_base_params.items()
        if info.type == ParamType.SEX_BIAS
    ]
    non_sex_bias_param_names = [
        name for name in model_param_names
        if name not in sex_bias_param_names
    ]
    return model, model_param_names, sex_bias_param_names, non_sex_bias_param_names


def _make_mock_population():
    """
    Return a minimal population SimpleNamespace with stub data sufficient for driver book-keeping.
    The specific values are not important, but they should be plausible and consistent with the expected types.
    """
    population = SimpleNamespace()
    population.get_global_tractlengths = lambda npts, exclude_tracts_below_cM: (
        np.linspace(0, 1, 6),
        {"A": [0, 0, 0, 0, 0], "B": [0, 0, 0, 0, 0]},
    )
    population.calculate_ancestry_proportions = lambda ancestor_labels: np.array([0.6, 0.4])
    population.calculate_allosome_proportions = lambda population_labels, allosome_label: np.array([0.6, 0.4])
    population.smooth_unknowns = lambda allosome_labels: None
    population.unknown_labels = []
    population.Ls = [1.0]
    population.indivs = [object()]
    population.nind = 1
    population.num_males = 1
    population.num_females = 1
    population.allosome_lengths = {"X": 1.0}
    return population


def _install_driver_run_mocks(monkeypatch, tmp_path: Path, two_steps_optimization: bool, autosomes_in_step_2: bool):
    """
    Patch all I/O and optimizer calls in ``tracts.driver`` so that ``run_tracts`` can be
    exercised without touching the file system or running a real optimization. ``fake_run_model_multi_init`` 
    records every call into ``recorded_runs`` and returns deterministic dummy results:
      - Step 1 best is [11, 22, 101, 201] / [12, 23, 102, 202] with likelihoods [2.0, 1.0]
        (best → run 0: [11, 22, 101, 201]).
      - Step 2 returns plausible values that are not checked here.
    ``recording_print_run_intro`` wraps the real implementation and appends
    every ``title_message`` to ``recorded_titles`` so tests can inspect table headers.
    Returns (driver_spec, model, population, recorded_titles, recorded_runs,
    recorded_parse_calls, recorded_collapse_calls)
    """
    driver_spec = _make_mock_driver_spec(tmp_path, two_steps_optimization, autosomes_in_step_2)
    model = _make_mock_model()
    population = _make_mock_population()
    recorded_titles = []
    recorded_runs = []
    recorded_parse_calls = []
    recorded_collapse_calls = []

    def recording_print_run_intro(*args, **kwargs):
        if "title_message" in kwargs:
            recorded_titles.append(kwargs["title_message"])
        else:
            recorded_titles.append(args[4])
        return real_print_run_intro(*args, **kwargs)

    def fake_run_model_multi_init(*, start_params_list, steps=None, autosomes_in_step_2=None, **kwargs):
        normalized_start_params = [np.array(params, dtype=float) for params in start_params_list]
        recorded_runs.append(
            {
                "steps": steps,
                "autosomes_in_step_2": autosomes_in_step_2,
                "start_params_list": normalized_start_params,
            }
        )

        if steps == [1]:
            return (
                [
                    np.array([11.0, 22.0, 101.0, 201.0]),
                    np.array([12.0, 23.0, 102.0, 202.0]),
                ],
                [2.0, 1.0],
            )

        if steps == [2]:
            return (
                [
                    np.array([11.0, 22.0, 301.0, 401.0]),
                    np.array([11.0, 22.0, 302.0, 402.0]),
                ],
                [5.0, 4.0],
            )

        return ([np.array([1.0, 2.0, 3.0, 4.0])], [1.0])

    def fake_parse_start_params(*, sample_param_names=None, fixed_param_values=None, **kwargs):
        sample_param_names = set(sample_param_names) if sample_param_names is not None else None
        fixed_param_values = {} if fixed_param_values is None else dict(fixed_param_values)

        recorded_parse_calls.append(
            {
                "sample_param_names": sample_param_names,
                "fixed_param_values": fixed_param_values,
            }
        )

        if sample_param_names == {"t", "rate_eur"}:
            return [
                np.array([1.0, 2.0, 0.0, 0.0]),
                np.array([10.0, 20.0, 0.0, 0.0]),
            ]

        if sample_param_names == {"sb_eur", "sb_afr"}:
            return [
                np.array([fixed_param_values["t"], fixed_param_values["rate_eur"], 3.0, 4.0]),
                np.array([fixed_param_values["t"], fixed_param_values["rate_eur"], 30.0, 40.0]),
            ]

        return [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]

    def fake_collapse_identical_start_params(start_params, step_label):
        recorded_collapse_calls.append(
            {
                "step_label": step_label,
                "start_params": [np.array(params, dtype=float) for params in start_params],
            }
        )
        return start_params

    monkeypatch.setattr(driver_module, "locate_file_path", lambda filename, script_dir: Path("/tmp/test_driver.yaml"))
    monkeypatch.setattr(driver_module, "load_driver_file", lambda driver_path: driver_spec)
    monkeypatch.setattr(driver_module, "load_population", lambda **kwargs: population)
    monkeypatch.setattr(driver_module, "load_demographic_model_from_driver", lambda **kwargs: _load_demographic_model_from_driver_result(model))
    monkeypatch.setattr(driver_module, "parse_start_params", fake_parse_start_params)
    monkeypatch.setattr(driver_module, "collapse_identical_start_params", fake_collapse_identical_start_params)
    # compute_physical_start_params (used by run_tracts for the step-1/single-step starting
    # parameters) lives in driver_utils.py and calls parse_start_params/collapse_identical_start_params
    # via that module's own namespace, so those must be patched there too.
    monkeypatch.setattr(driver_utils_module, "parse_start_params", fake_parse_start_params)
    monkeypatch.setattr(driver_utils_module, "collapse_identical_start_params", fake_collapse_identical_start_params)
    monkeypatch.setattr(driver_module, "get_time_scaled_model_func", lambda model: (lambda params: params))
    monkeypatch.setattr(driver_module, "get_time_scaled_model_bounds", lambda model: (lambda params: 1.0))
    monkeypatch.setattr(driver_module, "run_model_multi_init", fake_run_model_multi_init)
    monkeypatch.setattr(driver_module, "output_simulation_data_sex_biased", lambda **kwargs: None)
    mock_output_dir = tmp_path / "test_output"
    mock_output_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(driver_module, "initialize_tracts", lambda **kwargs: (driver_module.logger, mock_output_dir / "test.log", mock_output_dir))
    monkeypatch.setattr(driver_module, "close_log_file", lambda **kwargs: None)
    monkeypatch.setattr(driver_module, "_print_run_intro", recording_print_run_intro)

    return (
        driver_spec,
        model,
        population,
        recorded_titles,
        recorded_runs,
        recorded_parse_calls,
        recorded_collapse_calls,
    )


def _assert_arrays_list_equal(actual, expected):
    """
    Assert two lists of numpy arrays are element-wise equal (via ``assert_allclose``).
    """
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item, expected_item)

# ------------ Test functions ----------

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
    _clean_output_dir(output_dir)



def test_run_tracts(tmp_path):
    """
    Test that run_tracts creates all outputs and log file in specified directory, for a set of driver files covering all possible
    configurations regarding optimization.
    """

    script_dir =  _copy_tests_to_tmp(tmp_path)
    driver_files_autosomes = sorted(
        [f.name for f in script_dir.iterdir() if "autosomes" in f.name]
    )
    driver_files_allosomes = sorted(
        [f.name for f in script_dir.iterdir() if "allosomes" in f.name]
    )

    output_dir = tmp_path / "test_output"
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
        prepared_driver = _prepare_driver(script_dir / driver_file, output_dir)
        _run_tracts_test(prepared_driver, script_dir, output_dir, log_name, expected_files_without_allosomes)

    for driver_file in driver_files_allosomes:
        prepared_driver = _prepare_driver(script_dir / driver_file, output_dir)
        _run_tracts_test(prepared_driver, script_dir, output_dir, log_name, expected_files_with_allosomes)


def _compare_driver_results(driver_files: list[str], script_dir: Path, output_dir: Path, tolerance: float = 0.01):
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
        _clean_output_dir(output_dir)

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


def test_compare_only_autosomal_one_step_vs_two_steps(tmp_path):
    """
    Test that one_step and two_steps optimizations produce very similar results when only autosomes are present in the sample.
    Optimizations are expected to be equivalent in this context: the two-steps optimization is expected to stop after the first step,
    optimizing only over autosomal data. The test compares migration matrices, optimal parameters and tract distributions.
    Performs the comparison with and without parameters fixed by ancestry.
    """
    script_dir =  _copy_tests_to_tmp(tmp_path)
    output_dir = tmp_path / "test_output"

    # No parameters fixed by ancestry
    driver_files = [
        _prepare_driver(script_dir / "test_one_step_only_autosomes.yaml", output_dir),
        _prepare_driver(script_dir / "test_two_steps_only_autosomes.yaml", output_dir),
    ]
    _compare_driver_results(driver_files, script_dir, output_dir)

    # Parameters fixed by ancestry
    driver_files_fix = [
        _prepare_driver(script_dir / "test_one_step_only_autosomes_fix.yaml", output_dir),
        _prepare_driver(script_dir / "test_two_steps_only_autosomes_fix.yaml", output_dir),
    ]
    _compare_driver_results(driver_files_fix, script_dir, output_dir)


def test_run_tracts_two_steps_reuses_step1_values_for_step2_start_params(tmp_path, monkeypatch, capsys):
    """
    Verify that two-step runs print separate Step 1 and Step 2 starting-parameter tables and that Step 2
    starts from Step 1's best non-sex-bias parameters while preserving run-specific sex-bias starts.
    """
    _, _, _, recorded_titles, recorded_runs, recorded_parse_calls, recorded_collapse_calls = _install_driver_run_mocks(
        monkeypatch,
        tmp_path,
        two_steps_optimization=True,
        autosomes_in_step_2=True,
    )

    run_tracts("driver.yaml", script_dir=tmp_path)
    captured = capsys.readouterr()

    assert "Starting parameters for step 1 optimization" in captured.out
    assert "non-sex-bias parameters are fixed to the best step 1 estimates" in captured.out
    assert recorded_titles == [
        "Starting parameters for step 1 optimization",
        "Starting parameters for step 2 optimization (non-sex-bias parameters are fixed to the best step 1 estimates)."
        ]

    assert [call["steps"] for call in recorded_runs] == [[1], [2]]
    assert recorded_parse_calls == [
        {
            "sample_param_names": {"t", "rate_eur"},
            "fixed_param_values": {"sb_eur": 0.0, "sb_afr": 0.0},
        },
        {
            "sample_param_names": {"sb_eur", "sb_afr"},
            "fixed_param_values": {"t": 11.0, "rate_eur": 22.0},
        },
    ]
    assert [call["step_label"] for call in recorded_collapse_calls] == ["step 1", "step 2"]
    _assert_arrays_list_equal(
        recorded_runs[0]["start_params_list"],
        [
            np.array([1.0, 2.0, 0.0, 0.0]),
            np.array([10.0, 20.0, 0.0, 0.0]),
        ],
    )
    _assert_arrays_list_equal(
        recorded_runs[1]["start_params_list"],
        [
            np.array([11.0, 22.0, 3.0, 4.0]),
            np.array([11.0, 22.0, 30.0, 40.0]),
        ],
    )


def test_run_tracts_forwards_autosomes_in_step_2_flag(tmp_path, monkeypatch):
    """
    Verify that the driver forwards autosomes_in_step_2 to the Step 2 optimization call.
    """
    _, _, _, _, recorded_runs, _, _ = _install_driver_run_mocks(
        monkeypatch,
        tmp_path,
        two_steps_optimization=True,
        autosomes_in_step_2=False,
    )

    run_tracts("driver.yaml", script_dir=tmp_path)

    assert recorded_runs[0]["steps"] == [1]
    assert recorded_runs[1]["steps"] == [2]
    assert recorded_runs[1]["autosomes_in_step_2"] is False


# ------------ Helper and tests for ancestry-fixed parameters in two-step optimisation ----------

def _make_mock_model_with_ancestry_fixed():
    """
    Like _make_mock_model but with an ancestry-fixed rate parameter (``rate_afr``) at index 2.
    Parameters:
      - t        (TIME,     index 0)
      - rate_eur (RATE,     index 1)
      - rate_afr (RATE,     index 2) ← fixed by ancestry; never directly optimised
      - sb_eur   (SEX_BIAS, index 3)
      - sb_afr   (SEX_BIAS, index 4)
    """
    model = MagicMock(spec=ParametrizedDemographySexBiased)
    model.model_base_params = OrderedDict([
        ("t",        SimpleNamespace(index=0, type=ParamType.TIME, bounds=ParamType.TIME.bounds)),
        ("rate_eur", SimpleNamespace(index=1, type=ParamType.RATE, bounds=ParamType.RATE.bounds)),
        ("rate_afr", SimpleNamespace(index=2, type=ParamType.RATE, bounds=ParamType.RATE.bounds)),
        ("sb_eur",   SimpleNamespace(index=3, type=ParamType.SEX_BIAS, bounds=ParamType.SEX_BIAS.bounds)),
        ("sb_afr",   SimpleNamespace(index=4, type=ParamType.SEX_BIAS, bounds=ParamType.SEX_BIAS.bounds)),
    ])
    model.params_fixed_by_ancestry = {"rate_afr"}
    model.population_indices = OrderedDict([("A", 0), ("B", 1)])
    model.parametrized_populations = ["pop"]
    model.founder_events = {}
    model.parameter_handler = SimpleNamespace(
        to_physical_params_functions={},
        to_optimizer_params_functions={},
        enable_time_param_logging=True,
        convert_to_optimizer_params=lambda params: np.array(params, dtype=float),
        convert_to_physical_params=lambda params, report_non_admissible=False: np.array(params, dtype=float),
        set_up_fixed_parameters=lambda *args, **kwargs: None,
        release_fixed_parameters=lambda *args, **kwargs: None,
        add_fixed_parameters=lambda *args, **kwargs: None,
    )
    model.proportions_from_matrices = lambda matrices: {"A": np.array([1.0])}
    model.get_violation_score = lambda params, verbose=False: 1.0
    model.get_migration_matrices = lambda params: {"female": np.zeros((1, 1)), "male": np.zeros((1, 1))}
    model.set_up_fixed_parameters = lambda *args, **kwargs: None
    return model


def test_parse_start_params_preserves_fixed_values_for_ancestry_fixed_params():
    """
    Regression test for the bug where ``parse_start_params`` called
    ``compute_params_fixed_by_ancestry`` unconditionally and let it overwrite values that
    were explicitly provided in ``fixed_param_values``.

    In the two-step optimisation workflow, ``driver.py`` passes the step-1 optimal values
    for ancestry-fixed non-sex-bias parameters through ``fixed_param_values`` when building
    step-2 starting parameters.  Before the fix, those values were silently discarded:
    ``compute_params_fixed_by_ancestry`` was called unconditionally after the candidate was
    drawn and would re-solve ancestry-fixed parameters given the freshly sampled sex-bias
    starting values, producing different (wrong) values.
    """
    from tracts.driver_utils import parse_start_params as real_parse_start_params

    # Minimal 3-param model: t (TIME, 0), rate_afr (RATE, 1, ancestry-fixed), sb (SEX_BIAS, 2)
    mock_model = SimpleNamespace()
    mock_model.model_base_params = OrderedDict([
        ("t",        SimpleNamespace(index=0, type=ParamType.TIME,     bounds=(0.0, 1.0))),
        ("rate_afr", SimpleNamespace(index=1, type=ParamType.RATE,     bounds=(0.0, 1.0))),
        ("sb",       SimpleNamespace(index=2, type=ParamType.SEX_BIAS, bounds=(-1.0, 1.0))),
    ])
    mock_model.params_fixed_by_ancestry = {"rate_afr"}
    mock_model.get_violation_score = lambda params, verbose=False: 1.0  # always feasible

    STEP1_OPTIMAL    = 0.7   # the step-1 best value carried in fixed_param_values
    WRONG_RECOMPUTED = 0.1   # what ancestry would compute given new sex-bias starts

    def fake_compute_ancestry(candidate):
        """Simulate ancestry re-solving rate_afr to a value different from the step-1 optimal."""
        result = candidate.copy()
        result[1] = WRONG_RECOMPUTED
        return result

    mock_model.parameter_handler = SimpleNamespace(
        compute_params_fixed_by_ancestry=fake_compute_ancestry,
    )

    candidates = real_parse_start_params(
        start_param_bounds=SimpleNamespace(sb="0.0:0.5"),
        repetitions=3,
        seed=42,
        demographic_model=mock_model,
        sample_param_names={"sb"},
        fixed_param_values={
            "t": 0.5,
            "rate_afr": STEP1_OPTIMAL,  # step-1 best; must survive compute_params_fixed_by_ancestry
        },
    )

    assert len(candidates) == 3
    for candidate in candidates:
        np.testing.assert_allclose(
            candidate[1], STEP1_OPTIMAL,
            rtol=1e-10,
            err_msg=(
                f"rate_afr should be fixed at the step-1 optimal value ({STEP1_OPTIMAL}), "
                f"not the ancestry-recomputed value ({WRONG_RECOMPUTED}). "
                "compute_params_fixed_by_ancestry must not overwrite values "
                "explicitly provided in fixed_param_values."
            ),
        )
        np.testing.assert_allclose(
            candidate[0], 0.5, rtol=1e-10,
            err_msg="t should remain fixed at 0.5",
        )


def test_two_steps_ancestry_fixed_params_included_in_step2_fixed_values(tmp_path, monkeypatch):
    """
    Regression test ensuring that ``driver.py`` includes ancestry-fixed non-sex-bias
    parameters in the ``fixed_param_values`` dict that is forwarded to ``parse_start_params``
    for step 2.  Those values must be the step-1 optimal values so that the downstream
    ``parse_start_params`` fix (and the ``core.py`` step-2 freeze) have the correct input
    to work with.
    """
    model = _make_mock_model_with_ancestry_fixed()
    model.parameter_handler.params_fixed_by_ancestry = {"rate_afr"}
    model.parameter_handler.user_params_fixed_by_value = {}
    driver_spec = _make_mock_driver_spec(tmp_path, two_steps_optimization=True, autosomes_in_step_2=True)
    population = _make_mock_population()

    recorded_parse_calls = []

    def fake_parse_start_params(*, sample_param_names=None, fixed_param_values=None, **kwargs):
        sample_param_names = set(sample_param_names) if sample_param_names is not None else None
        fixed_param_values = {} if fixed_param_values is None else dict(fixed_param_values)
        recorded_parse_calls.append({
            "sample_param_names": sample_param_names,
            "fixed_param_values": fixed_param_values,
        })
        if sample_param_names == {"t", "rate_eur", "rate_afr"}:
            # Step 1: sample all non-sex-bias params (including ancestry-fixed rate_afr)
            return [
                np.array([1.0,  2.0,  3.0,  0.0,  0.0]),
                np.array([10.0, 20.0, 30.0, 0.0,  0.0]),
            ]
        if sample_param_names == {"sb_eur", "sb_afr"}:
            # Step 2: sex-bias params sampled; non-sex-bias come from fixed_param_values
            rate_afr_used = fixed_param_values.get("rate_afr", float("nan"))
            return [
                np.array([fixed_param_values["t"], fixed_param_values["rate_eur"], rate_afr_used, 3.0,  4.0]),
                np.array([fixed_param_values["t"], fixed_param_values["rate_eur"], rate_afr_used, 30.0, 40.0]),
            ]
        return [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]

    def fake_run_model_multi_init(*, start_params_list, steps=None, **kwargs):
        if steps == [1]:
            # Best result (highest likelihood): run 0 → rate_afr = 33.0
            return (
                [np.array([11.0, 22.0, 33.0, 101.0, 201.0]),
                 np.array([12.0, 23.0, 34.0, 102.0, 202.0])],
                [2.0, 1.0],
            )
        if steps == [2]:
            return (
                [np.array([11.0, 22.0, 33.0, 301.0, 401.0]),
                 np.array([11.0, 22.0, 33.0, 302.0, 402.0])],
                [5.0, 4.0],
            )
        return ([np.array([1.0, 2.0, 3.0, 4.0, 5.0])], [1.0])

    monkeypatch.setattr(driver_module, "locate_file_path",    lambda filename, script_dir: Path("/tmp/test_driver.yaml"))
    monkeypatch.setattr(driver_module, "load_driver_file",    lambda driver_path: driver_spec)
    monkeypatch.setattr(driver_module, "load_population",     lambda **kwargs: population)
    monkeypatch.setattr(driver_module, "load_demographic_model_from_driver", lambda **kwargs: _load_demographic_model_from_driver_result(model))
    monkeypatch.setattr(driver_module, "parse_start_params",  fake_parse_start_params)
    monkeypatch.setattr(driver_module, "collapse_identical_start_params", lambda sp, label: sp)
    # compute_physical_start_params (used by run_tracts for the step-1/single-step starting
    # parameters) lives in driver_utils.py and calls parse_start_params/collapse_identical_start_params
    # via that module's own namespace, so those must be patched there too.
    monkeypatch.setattr(driver_utils_module, "parse_start_params", fake_parse_start_params)
    monkeypatch.setattr(driver_utils_module, "collapse_identical_start_params", lambda sp, label: sp)
    monkeypatch.setattr(driver_module, "get_time_scaled_model_func",   lambda m: (lambda params: params))
    monkeypatch.setattr(driver_module, "get_time_scaled_model_bounds", lambda m: (lambda params: 1.0))
    monkeypatch.setattr(driver_module, "run_model_multi_init", fake_run_model_multi_init)
    monkeypatch.setattr(driver_module, "output_simulation_data_sex_biased", lambda **kwargs: None)
    mock_output_dir = tmp_path / "test_output"
    mock_output_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(driver_module, "initialize_tracts", lambda **kwargs: (driver_module.logger, mock_output_dir / "test.log", mock_output_dir))
    monkeypatch.setattr(driver_module, "close_log_file", lambda **kwargs: None)
    monkeypatch.setattr(driver_module, "_print_run_intro", lambda *args, **kwargs: None)

    run_tracts("driver.yaml", script_dir=tmp_path)

    # Exactly two parse_start_params calls: one for step 1, one for step 2.
    assert len(recorded_parse_calls) == 2, (
        f"Expected two parse_start_params calls; got {len(recorded_parse_calls)}."
    )

    step_2_fixed = recorded_parse_calls[1]["fixed_param_values"]

    # The ancestry-fixed non-sex-bias parameter must be forwarded with its step-1 optimal value.
    assert "rate_afr" in step_2_fixed, (
        "rate_afr (ancestry-fixed non-sex-bias parameter) must appear in fixed_param_values "
        "for the step-2 parse_start_params call so it can be frozen at its step-1 optimal value."
    )
    np.testing.assert_allclose(
        step_2_fixed["rate_afr"], 33.0,
        rtol=1e-10,
        err_msg=(
            "rate_afr in step-2 fixed_param_values must equal the step-1 optimal value (33.0). "
            "The driver must extract it from optimal_params_step_1, not recompute or default it."
        ),
    )

    # Sanity-check: other non-sex-bias params are also correctly forwarded.
    np.testing.assert_allclose(step_2_fixed["t"],        11.0, rtol=1e-10)
    np.testing.assert_allclose(step_2_fixed["rate_eur"], 22.0, rtol=1e-10)

    # Sex-bias params must NOT appear in step-2 fixed_param_values (they are being optimised).
    assert "sb_eur" not in step_2_fixed, "sb_eur must not appear in step-2 fixed_param_values."
    assert "sb_afr" not in step_2_fixed, "sb_afr must not appear in step-2 fixed_param_values."
