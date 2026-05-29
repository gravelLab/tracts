import os
import pytest
import shutil
from collections import OrderedDict
from types import SimpleNamespace
from pathlib import Path
import numpy as np

import tracts.driver as driver_module
from tracts.driver import run_tracts
from tracts.demography.parameter import ParamType
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
        log_filename="test_logfile.log",
        output_directory=str(tmp_path / "test_output"),
        exclude_tracts_below_cm=0,
        npts=5,
        ad_model_autosomes="DC",
        ad_model_allosomes="DC",
        samples=SimpleNamespace(allosomes=["X"]),
        unknown_labels_for_smoothing=[],
        model_filename="test_model.yaml",
        start_params=SimpleNamespace(),
        repetitions=2,
        seed=1,
        maximum_iterations=2,
        verbose_log=0,
        verbose_screen=0,
        fix_parameters_from_ancestry_proportions=[],
        output_filename_format="test_output_{label}",
        two_steps_optimization=two_steps_optimization,
        autosomes_in_step_2=autosomes_in_step_2,
        use_autosomes_for_sex_bias=autosomes_in_step_2,
        log_scale=False,
    )


def _make_mock_model():
    """
    Return a minimal model SimpleNamespace with four parameters:
      - t        (TIME,     index 0)
      - rate_eur (RATE,     index 1)
      - sb_eur   (SEX_BIAS, index 2)
      - sb_afr   (SEX_BIAS, index 3)

    Indices 0–1 are non-sex-bias (replaced by Step 1 best in two-step runs).
    Indices 2–3 are sex-bias (kept run-specific in Step 2).
    """
    model = SimpleNamespace()
    model.model_base_params = OrderedDict([
        ("t", SimpleNamespace(index=0, type=ParamType.TIME)),
        ("rate_eur", SimpleNamespace(index=1, type=ParamType.RATE)),
        ("sb_eur", SimpleNamespace(index=2, type=ParamType.SEX_BIAS)),
        ("sb_afr", SimpleNamespace(index=3, type=ParamType.SEX_BIAS)),
    ])
    model.population_indices = OrderedDict([("A", 0), ("B", 1)])
    model.parametrized_populations = ["pop"]
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
    monkeypatch.setattr(driver_module, "load_model_from_driver", lambda **kwargs: model)
    monkeypatch.setattr(driver_module, "parse_start_params", fake_parse_start_params)
    monkeypatch.setattr(driver_module, "collapse_identical_start_params", fake_collapse_identical_start_params)
    monkeypatch.setattr(driver_module, "get_time_scaled_model_func", lambda model: (lambda params: params))
    monkeypatch.setattr(driver_module, "get_time_scaled_model_bounds", lambda model: (lambda params: 1.0))
    monkeypatch.setattr(driver_module, "run_model_multi_init", fake_run_model_multi_init)
    monkeypatch.setattr(driver_module, "output_simulation_data_sex_biased", lambda **kwargs: None)
    monkeypatch.setattr(driver_module, "setup_logger", lambda: (driver_module.logger, SimpleNamespace()))
    monkeypatch.setattr(driver_module, "set_log_file", lambda **kwargs: None)
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