import shutil
from pathlib import Path

import pytest

from tracts.driver import run_tracts
from tracts.driver_utils import _OUTPUT_SUBDIRS
from tracts.plot import (
    _detect_output_filename_format,
    _read_ancestry_per_individual,
    _read_population_names,
    plot_admixture_from_output,
    plot_migration_matrices_from_output,
    plot_tract_length_distributions_from_output,
    plot_all_from_output_directories,
)

# ------------ Helper functions for test setup ----------


def _copy_tests_to_tmp(tmp_path: Path) -> Path:

    source_tests = Path(__file__).resolve().parent
    tmp_tests = tmp_path / "tests"
    tmp_tests.mkdir(parents=True, exist_ok=True)

    required_subdirs = ("drivers", "models", "data")
    ignore = shutil.ignore_patterns("test_output", "__pycache__")

    for subdir in required_subdirs:
        source_subdir = source_tests / subdir
        if source_subdir.exists():
            shutil.copytree(source_subdir, tmp_tests / subdir, ignore=ignore)

    return tmp_tests / "drivers"


def _prepare_driver(driver_path: Path, output_dir: Path, extra_output_lines: list[str] | None = None,
                    output_filename_format: str | None = None) -> str:

    lines = driver_path.read_text().splitlines()
    new_lines = []
    found_output_directory = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("output_directory:"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}output_directory: '{output_dir}'")
            found_output_directory = True
            if extra_output_lines:
                new_lines.extend(f"{indent}{extra_line}" for extra_line in extra_output_lines)
        elif output_filename_format is not None and stripped.startswith("output_filename_format:"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}output_filename_format: "{output_filename_format}"')
        else:
            new_lines.append(line)

    if not found_output_directory:
        new_lines.append(f"output_directory: '{output_dir}'")

    driver_path.write_text("\n".join(new_lines) + "\n")
    return driver_path.name


def _run_driver(tmp_path_factory, driver_filename: str, tag: str, extra_output_lines: list[str] | None = None,
                output_filename_format: str | None = None) -> Path:
    tmp_path = tmp_path_factory.mktemp(tag)
    drivers_dir = _copy_tests_to_tmp(tmp_path)
    output_dir = tmp_path / "test_output"
    prepared = _prepare_driver(drivers_dir / driver_filename, output_dir, extra_output_lines=extra_output_lines,
                                output_filename_format=output_filename_format)
    run_tracts(prepared, script_dir=str(drivers_dir))
    return output_dir


def _delete_plots(output_dir: Path) -> None:
    for f in list(output_dir.rglob("*.pdf")) + list(output_dir.rglob("*.png")):
        f.unlink()


def _out(output_dir: Path, prefix: str, label: str) -> Path:
    """Path to the output file for ``label`` (e.g. ``"admixture_plot.pdf"``), given the ``output_filename_format``
    ``prefix`` (e.g. ``"test_output_"``), within the structured output directory (see ``_OUTPUT_SUBDIRS``).
    ``.png`` companions are not registered under their own label in ``_OUTPUT_SUBDIRS`` (they are always
    written alongside the corresponding ``.pdf`` file), so their subdirectory is looked up via that ``.pdf``
    label instead."""
    lookup_label = label[:-len('.png')] + '.pdf' if label.endswith('.png') else label
    return output_dir / _OUTPUT_SUBDIRS.get(lookup_label, '') / f"{prefix}{label}"


# ------------ Fixtures: real run_tracts output directories, reused across tests ----------


@pytest.fixture(scope="module")
def combined_allosome_output_dir(tmp_path_factory) -> Path:
    """Output directory for a run with allosomes. Female, male and summed allosomal tract distributions are all
    saved to output files, regardless of how they are later plotted."""
    return _run_driver(tmp_path_factory, "test_allosomes_one_step.yaml", "plot_combined")


@pytest.fixture(scope="module")
def autosome_only_output_dir(tmp_path_factory) -> Path:
    """Output directory for a run without allosomes."""
    return _run_driver(tmp_path_factory, "test_autosomes.yaml", "plot_autosome_only")


@pytest.fixture(scope="module")
def autosome_only_output_dir_distinct_format(tmp_path_factory) -> Path:
    """Same as autosome_only_output_dir, but with a distinct output_filename_format, to test that plots
    from multiple runs can be collected into a common save_dir without colliding."""
    return _run_driver(tmp_path_factory, "test_autosomes.yaml", "plot_autosome_only_distinct",
                        output_filename_format="autosome_only_{label}")


# ------------ Tests ----------


class TestPlotFromCombinedAllosomeOutput:

    def test_admixture_plot(self, combined_allosome_output_dir):
        _delete_plots(combined_allosome_output_dir)
        plot_admixture_from_output(combined_allosome_output_dir)
        assert _out(combined_allosome_output_dir, "test_output_", "admixture_plot.pdf").exists()

    def test_migration_matrices_plot(self, combined_allosome_output_dir):
        _delete_plots(combined_allosome_output_dir)
        plot_migration_matrices_from_output(combined_allosome_output_dir)
        assert _out(combined_allosome_output_dir, "test_output_", "migration_matrices.pdf").exists()
        assert _out(combined_allosome_output_dir, "test_output_", "migration_matrices.png").exists()

    def test_tract_length_distribution_plots(self, combined_allosome_output_dir):
        _delete_plots(combined_allosome_output_dir)
        plot_tract_length_distributions_from_output(combined_allosome_output_dir)

        assert _out(combined_allosome_output_dir, "test_output_", "autosomes_all_populations.pdf").exists()
        assert _out(combined_allosome_output_dir, "test_output_", "autosomes_all_populations.png").exists()
        assert _out(combined_allosome_output_dir, "test_output_", "allosomes_all_populations.pdf").exists()
        assert _out(combined_allosome_output_dir, "test_output_", "allosomes_all_populations.png").exists()

        # Combined mode: no per-sex allosome plots should be produced.
        assert not _out(combined_allosome_output_dir, "test_output_", "female_allosomes_all_populations.pdf").exists()
        assert not _out(combined_allosome_output_dir, "test_output_", "male_allosomes_all_populations.pdf").exists()

    def test_output_filename_format_is_auto_detected_correctly(self, combined_allosome_output_dir):
        assert _detect_output_filename_format(combined_allosome_output_dir) == "test_output_{label}"

    def test_explicit_output_filename_format_matches_auto_detected(self, combined_allosome_output_dir):
        _delete_plots(combined_allosome_output_dir)
        plot_admixture_from_output(combined_allosome_output_dir, output_filename_format="test_output_{label}")
        assert _out(combined_allosome_output_dir, "test_output_", "admixture_plot.pdf").exists()

    def test_log_scale_can_be_overridden(self, combined_allosome_output_dir):
        # Should not raise, regardless of the log_scale used in the original run.
        plot_tract_length_distributions_from_output(combined_allosome_output_dir, log_scale=False)


class TestPlotFromSeparateAllosomeOutput:

    def test_tract_length_distribution_plots(self, combined_allosome_output_dir):
        _delete_plots(combined_allosome_output_dir)
        plot_tract_length_distributions_from_output(combined_allosome_output_dir,
                                                    sum_female_and_male_allosome_tracts=False)

        assert _out(combined_allosome_output_dir, "test_output_", "autosomes_all_populations.pdf").exists()
        assert _out(combined_allosome_output_dir, "test_output_", "female_allosomes_all_populations.pdf").exists()
        assert _out(combined_allosome_output_dir, "test_output_", "male_allosomes_all_populations.pdf").exists()

        # sum_female_and_male_allosome_tracts=False: no combined allosome plot should be produced.
        assert not _out(combined_allosome_output_dir, "test_output_", "allosomes_all_populations.pdf").exists()


class TestPlotFromAutosomeOnlyOutput:

    def test_tract_length_distribution_plots(self, autosome_only_output_dir):
        _delete_plots(autosome_only_output_dir)
        plot_tract_length_distributions_from_output(autosome_only_output_dir)

        assert _out(autosome_only_output_dir, "test_output_", "autosomes_all_populations.pdf").exists()
        assert not _out(autosome_only_output_dir, "test_output_", "allosomes_all_populations.pdf").exists()
        assert not _out(autosome_only_output_dir, "test_output_", "female_allosomes_all_populations.pdf").exists()
        assert not _out(autosome_only_output_dir, "test_output_", "male_allosomes_all_populations.pdf").exists()

    def test_admixture_and_migration_matrices_plots(self, autosome_only_output_dir):
        _delete_plots(autosome_only_output_dir)
        plot_admixture_from_output(autosome_only_output_dir)
        plot_migration_matrices_from_output(autosome_only_output_dir)
        assert _out(autosome_only_output_dir, "test_output_", "admixture_plot.pdf").exists()
        assert _out(autosome_only_output_dir, "test_output_", "migration_matrices.pdf").exists()


class TestSaveDir:

    def test_plots_written_to_save_dir_not_output_dir(self, combined_allosome_output_dir, tmp_path):
        _delete_plots(combined_allosome_output_dir)
        save_dir = tmp_path / "elsewhere"

        plot_admixture_from_output(combined_allosome_output_dir, save_dir=save_dir)
        plot_migration_matrices_from_output(combined_allosome_output_dir, save_dir=save_dir)
        plot_tract_length_distributions_from_output(combined_allosome_output_dir, save_dir=save_dir)

        # Nothing should have been written back into the original output directory.
        assert not list(combined_allosome_output_dir.rglob("*.pdf"))
        assert not list(combined_allosome_output_dir.rglob("*.png"))

        assert _out(save_dir, "test_output_", "admixture_plot.pdf").exists()
        assert _out(save_dir, "test_output_", "migration_matrices.pdf").exists()
        assert _out(save_dir, "test_output_", "autosomes_all_populations.pdf").exists()
        assert _out(save_dir, "test_output_", "allosomes_all_populations.pdf").exists()

    def test_save_dir_is_created_if_missing(self, combined_allosome_output_dir, tmp_path):
        save_dir = tmp_path / "does" / "not" / "exist" / "yet"
        assert not save_dir.exists()
        plot_admixture_from_output(combined_allosome_output_dir, save_dir=save_dir)
        assert _out(save_dir, "test_output_", "admixture_plot.pdf").exists()


class TestPlotAllFromOutputDirectories:

    def test_produces_all_three_plot_sets_for_every_directory_without_collision(
            self, combined_allosome_output_dir, autosome_only_output_dir_distinct_format, tmp_path):
        _delete_plots(combined_allosome_output_dir)
        _delete_plots(autosome_only_output_dir_distinct_format)
        save_dir = tmp_path / "batch"

        # output_filename_format is left as None (default): each directory's format ("test_output_{label}"
        # and "autosome_only_{label}", respectively) is auto-detected independently, so results from both
        # runs land in save_dir without overwriting each other.
        plot_all_from_output_directories(
            [combined_allosome_output_dir, autosome_only_output_dir_distinct_format],
            save_dir=save_dir,
        )

        for prefix in ("test_output_", "autosome_only_"):
            assert _out(save_dir, prefix, "admixture_plot.pdf").exists()
            assert _out(save_dir, prefix, "migration_matrices.pdf").exists()
            assert _out(save_dir, prefix, "autosomes_all_populations.pdf").exists()

        # The combined run has allosomal outputs, the distinct-format autosome-only run does not.
        assert _out(save_dir, "test_output_", "allosomes_all_populations.pdf").exists()
        assert not _out(save_dir, "autosome_only_", "allosomes_all_populations.pdf").exists()

        # Original output directories are untouched.
        assert not list(combined_allosome_output_dir.rglob("*.pdf"))
        assert not list(autosome_only_output_dir_distinct_format.rglob("*.pdf"))


class TestReadHelpers:

    def test_read_ancestry_per_individual(self, combined_allosome_output_dir):
        path = _out(combined_allosome_output_dir, "test_output_", "ancestry_per_individual")
        pop_names, ancestry_per_individual = _read_ancestry_per_individual(path)

        assert set(pop_names) == {"EUR", "AMR", "AFR"}
        assert len(ancestry_per_individual) > 0
        for name, proportions in ancestry_per_individual.items():
            assert isinstance(name, str)
            assert len(proportions) == len(pop_names)

    def test_read_population_names_matches_full_read(self, combined_allosome_output_dir):
        path = _out(combined_allosome_output_dir, "test_output_", "ancestry_per_individual")
        assert _read_population_names(path) == _read_ancestry_per_individual(path)[0]


class TestDetectOutputFilenameFormat:

    def test_raises_when_no_candidate_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _detect_output_filename_format(tmp_path)

    def test_raises_when_multiple_candidates_found(self, tmp_path):
        optimal_model_dir = tmp_path / _OUTPUT_SUBDIRS["optimal_parameters.txt"]
        optimal_model_dir.mkdir(parents=True)
        (optimal_model_dir / "runA_optimal_parameters.txt").write_text("parameter\tvalue\nlikelihood -1\n")
        (optimal_model_dir / "runB_optimal_parameters.txt").write_text("parameter\tvalue\nlikelihood -2\n")
        with pytest.raises(ValueError):
            _detect_output_filename_format(tmp_path)

    def test_detects_single_candidate(self, tmp_path):
        optimal_model_dir = tmp_path / _OUTPUT_SUBDIRS["optimal_parameters.txt"]
        optimal_model_dir.mkdir(parents=True)
        (optimal_model_dir / "myrun_optimal_parameters.txt").write_text("parameter\tvalue\nlikelihood -1\n")
        assert _detect_output_filename_format(tmp_path) == "myrun_{label}"
