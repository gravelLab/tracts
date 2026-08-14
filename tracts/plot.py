"""
Re-plots the output figures produced by :func:`~tracts.driver_utils.output_simulation_data_sex_biased`
(i.e. by :func:`~tracts.driver.run_tracts`) directly from a previously produced output directory, without
re-running the optimization. This is useful to reformat plots for a given analysis, or to batch-produce plots for a list of different runs (different models, populations, etc.),
by calling the relevant function below once per output directory.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tracts.driver_utils import (
    get_population_colors,
    plot_admixture,
    _plot_panel,
    _plot_migration_matrices,
    _get_output_path,
    _OUTPUT_SUBDIRS,
)

import logging
logger = logging.getLogger(__name__)

# -------------- Helper functions --------------

def _detect_output_filename_format(output_dir: Path) -> str:
    """
    Infers the ``output_filename_format`` used to produce the files in ``output_dir``, by looking for a single
    file ending in ``optimal_parameters.txt`` (always produced by :func:`~tracts.driver_utils.output_simulation_data_sex_biased`).
    Only works if ``{label}`` is the last element of the format string, which is the case for every format
    produced by :func:`~tracts.driver_utils.load_driver_file` (either the user-specified value, or its default,
    see :class:`~tracts.driver_utils.OutputConfig`).

    Parameters
    ----------
    output_dir: Path
        The output directory in which to look for output files.

    Returns
    -------
    str
        The inferred ``output_filename_format``.
    """
    suffix = "optimal_parameters.txt"
    candidates = sorted((Path(output_dir) / _OUTPUT_SUBDIRS[suffix]).glob(f"*{suffix}"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not find any file ending in '{suffix}' in {output_dir}, so the output_filename_format "
            "could not be automatically inferred. Please specify output_filename_format explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple candidate output file sets in {output_dir} ({[c.name for c in candidates]}), so "
            "the output_filename_format could not be automatically inferred. Please specify "
            "output_filename_format explicitly."
        )

    prefix = candidates[0].name[: -len(suffix)]
    return f"{prefix}{{label}}"


def _resolve_output_paths(output_dir: str | Path, output_filename_format: str | None, save_dir: str | Path | None):
    """
    Resolves ``output_dir``/``output_filename_format`` (auto-detecting the latter if not given) and returns
    ``(output_dir, save_dir, read_path_fn, write_path_fn)``, where ``read_path_fn(label)`` builds the path to
    the existing output file for ``label`` (in ``output_dir``), and ``write_path_fn(label)`` builds the path at
    which a re-produced plot for ``label`` should be saved (in ``save_dir``, defaulting to ``output_dir`` if
    ``save_dir`` is None, creating it if it does not already exist).
    """
    output_dir = Path(output_dir)
    if output_filename_format is None:
        output_filename_format = _detect_output_filename_format(output_dir)

    save_dir = Path(save_dir) if save_dir is not None else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    def read_path_fn(label: str) -> Path:
        return output_dir / _OUTPUT_SUBDIRS.get(label, '') / output_filename_format.format(label=label)

    def write_path_fn(label: str) -> Path:
        return _get_output_path(save_dir, output_filename_format, label)

    return output_dir, save_dir, read_path_fn, write_path_fn


def _read_bins(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        return np.array([float(x) for x in f.read().split("\t")])


def _read_ancestry_per_individual(path: Path) -> tuple[list, dict]:
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        pop_names = header[1:]
        ancestry_per_individual = {}
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            ancestry_per_individual[fields[0]] = [float(x) for x in fields[1:]]
    return pop_names, ancestry_per_individual


def _read_population_names(path: Path) -> list:
    """Reads only the population names (the header) from an ``ancestry_per_individual`` output file."""
    with open(path, "r") as f:
        return f.readline().rstrip("\n").split("\t")[1:]


def _read_population_rows(path: Path, pop_names: list) -> dict:
    with open(path, "r") as f:
        rows = [[float(x) for x in line.rstrip("\n").split("\t")] for line in f if line.strip()]
    if len(rows) != len(pop_names):
        raise ValueError(
            f"Expected {len(pop_names)} rows (one per population) in {path}, found {len(rows)}."
        )
    return {pop: rows[i] for i, pop in enumerate(pop_names)}


def _read_optimal_likelihood(path: Path) -> float:
    with open(path, "r") as f:
        for line in f:
            if line.startswith("likelihood"):
                return float(line.split()[1])
    raise ValueError(f"Could not find a 'likelihood' line in {path}.")



# -------------- Plotting functions --------------

def plot_admixture_from_output(output_dir: str | Path, output_filename_format: str | None = None,
                                save_dir: str | Path | None = None, title: str | None = None,
                                title_fontsize: float = 14, label_fontsize: float = 10,
                                tick_fontsize: float = 6, legend_fontsize: float = 10) -> None:
    """
    Re-produces the ``_admixture_plot`` directly from the ``_ancestry_per_individual`` output file saved in ``output_dir`` by a previous
    :func:`~tracts.driver.run_tracts` run, without re-running the inference.

    Parameters
    ----------
    output_dir: str | Path
        The output directory (as produced by :func:`~tracts.driver.run_tracts`) from which to read the ancestry
        proportions.
    output_filename_format: str | None
        The output filename format used to produce the files in ``output_dir``, as specified in the driver file
        used for the original run (or its default value, see :class:`~tracts.driver_utils.OutputConfig`). If
        None, it is automatically inferred from the files present in ``output_dir``.
    save_dir: str | Path | None
        The directory in which to save the re-produced plot (created if it does not already exist). If None,
        defaults to ``output_dir``, overwriting the original plot in place.
    title: str | None
        An optional title for the plot. Defaults to None (no title, matching the plot produced during
        :func:`~tracts.driver.run_tracts`).
    title_fontsize: float
        The font size of the title, if given. Defaults to 14.
    label_fontsize: float
        The font size of the y-axis label. Defaults to 10.
    tick_fontsize: float
        The font size of the per-individual x-axis tick labels. Defaults to 6.
    legend_fontsize: float
        The font size of the legend. Defaults to 10.
    """
    output_dir, save_dir, read_path, write_path = _resolve_output_paths(output_dir, output_filename_format, save_dir)

    pop_names, ancestry_per_individual = _read_ancestry_per_individual(read_path("ancestry_per_individual"))
    pop_colors = get_population_colors(pop_names)

    fig, ax = plot_admixture(ancestry_per_individual, pop_names, [pop_colors[pop] for pop in pop_names], ax=None,
                            title=title, title_fontsize=title_fontsize, label_fontsize=label_fontsize,
                            tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize)
    fig.savefig(write_path("admixture_plot.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Admixture plot regenerated in: {save_dir}")
    logger.info(f"Admixture plot regenerated in: {save_dir}")


def plot_migration_matrices_from_output(output_dir: str | Path, output_filename_format: str | None = None,
                                        save_dir: str | Path | None = None,
                                        title_mean: str = "Mean migration matrix",
                                        title_sex_bias: str = "Sex-bias in migration",
                                        title_fontsize: float | None = None,
                                        tick_fontsize: float | None = None,
                                        annot_fontsize: float | None = None) -> None:
    """
    Re-produces the ``_migration_matrices`` plot (mean migration matrix and sex-bias values per generation),
    directly from the ``_female_migration_matrix``/``_male_migration_matrix`` output files saved in
    ``output_dir`` by a previous :func:`~tracts.driver.run_tracts` run, without re-running the inference.

    Parameters
    ----------
    output_dir: str | Path
        The output directory (as produced by :func:`~tracts.driver.run_tracts`) from which to read the migration
        matrices.
    output_filename_format: str | None
        The output filename format used to produce the files in ``output_dir``, as specified in the driver file
        used for the original run (or its default value, see :class:`~tracts.driver_utils.OutputConfig`). If
        None, it is automatically inferred from the files present in ``output_dir``.
    save_dir: str | Path | None
        The directory in which to save the re-produced plot (created if it does not already exist). If None,
        defaults to ``output_dir``, overwriting the original plot in place.
    title_mean: str
        The title of the left panel (mean migration matrix). Defaults to "Mean migration matrix".
    title_sex_bias: str
        The title of the right panel (sex bias matrix). Defaults to "Sex-bias in migration".
    title_fontsize: float | None
        The font size of both panel titles (and colorbar labels). If None (default), an adaptive font size is
        used, based on the number of populations/generations being plotted.
    tick_fontsize: float | None
        The font size of the tick labels. If None (default), an adaptive font size is used.
    annot_fontsize: float | None
        The font size of the value annotations inside each matrix cell. If None (default), an adaptive font
        size is used.
    """
    output_dir, save_dir, read_path, write_path = _resolve_output_paths(output_dir, output_filename_format, save_dir)

    pop_names = _read_population_names(read_path("ancestry_per_individual"))
    female_matrix = np.loadtxt(read_path("female_migration_matrix"), ndmin=2)
    male_matrix = np.loadtxt(read_path("male_migration_matrix"), ndmin=2)

    _plot_migration_matrices(migration_matrix_f=female_matrix,
                            migration_matrix_m=male_matrix,
                            pop_labels=list(pop_names),
                            output_path=str(write_path("migration_matrices.pdf")),
                            title_mean=title_mean,
                            title_sex_bias=title_sex_bias,
                            title_fontsize=title_fontsize,
                            tick_fontsize=tick_fontsize,
                            annot_fontsize=annot_fontsize)

    print(f"Migration matrices plot regenerated in: {save_dir}")
    logger.info(f"Migration matrices plot regenerated in: {save_dir}")


def plot_tract_length_distributions_from_output(output_dir: str | Path, output_filename_format: str | None = None,
                                                log_scale: bool = True, save_dir: str | Path | None = None,
                                                sum_female_and_male_allosome_tracts: bool = True,
                                                autosome_title: str = "Autosomal tract length distributions",
                                                allosome_title: str = "X-chromosome tract length distributions",
                                                female_allosome_title: str = "Female X-chromosome tract length distributions",
                                                male_allosome_title: str = "Male X-chromosome tract length distributions",
                                                subtitle: str | None = None,
                                                xlabel: str = "Tract Length (M)",
                                                ylabel: str = "Count",
                                                title_fontsize: float = 14,
                                                subtitle_fontsize: float = 10,
                                                label_fontsize: float = 12,
                                                tick_fontsize: float = 10,
                                                legend_fontsize: float = 10) -> None:
    """
    Re-produces the autosomal and (if present) allosomal tract length distribution plots (observed counts
    against the predicted distribution), directly from the tract length distribution output files saved in
    ``output_dir`` by a previous :func:`~tracts.driver.run_tracts` run, without re-running the inference.

    Whether allosomal plots are produced at all is inferred from whether allosomal output files are present in
    ``output_dir``. When they are, ``sum_female_and_male_allosome_tracts`` controls whether female and male
    allosomal tracts are combined into a single plot (default) or plotted separately.

    Parameters
    ----------
    output_dir: str | Path
        The output directory (as produced by :func:`~tracts.driver.run_tracts`) from which to read the data and
        predicted tract length distributions.
    output_filename_format: str | None
        The output filename format used to produce the files in ``output_dir``, as specified in the driver file
        used for the original run (or its default value, see :class:`~tracts.driver_utils.OutputConfig`). If
        None, it is automatically inferred from the files present in ``output_dir``.
    log_scale: bool
        Whether to use log scale for the y-axis. Defaults to True. Does not have to match the value used in the
        original run: this can be used to reformat plots.
    save_dir: str | Path | None
        The directory in which to save the re-produced plots (created if it does not already exist). If None,
        defaults to ``output_dir``, overwriting the original plots in place.
    sum_female_and_male_allosome_tracts: bool
        If allosomes are present in the sample, whether to plot the female and male allosomal tract length
        distributions summed into a single plot (default) or as two separate plots. Both the summed and
        per-sex output files are always saved by :func:`~tracts.driver_utils.output_simulation_data_sex_biased`,
        so either can be plotted from the same ``output_dir`` regardless of this setting. Defaults to True.
    autosome_title: str
        The title of the autosomal plot. Defaults to "Autosomal tract length distributions".
    allosome_title: str
        The title of the allosomal plot, when female and male tracts are combined into a single plot. Defaults
        to "X-chromosome tract length distributions".
    female_allosome_title: str
        The title of the female allosomal plot, when female and male tracts are plotted separately. Defaults to
        "Female X-chromosome tract length distributions".
    male_allosome_title: str
        The title of the male allosomal plot, when female and male tracts are plotted separately. Defaults to
        "Male X-chromosome tract length distributions".
    subtitle: str | None
        An optional subtitle, applied to every plot produced by this call. If None (default), it is computed
        from the saved likelihood, as "Log-likelihood: {value}".
    xlabel: str
        The label for the x-axis. Defaults to "Tract Length (M)".
    ylabel: str
        The label for the y-axis. Defaults to "Count".
    title_fontsize: float
        The font size of the title. Defaults to 14.
    subtitle_fontsize: float
        The font size of the subtitle. Defaults to 10.
    label_fontsize: float
        The font size of the x- and y-axis labels. Defaults to 12.
    tick_fontsize: float
        The font size of the tick labels. Defaults to 10.
    legend_fontsize: float
        The font size of the legend text and titles. Defaults to 10.
    """
    output_dir, save_dir, read_path, write_path = _resolve_output_paths(output_dir, output_filename_format, save_dir)

    pop_names = _read_population_names(read_path("ancestry_per_individual"))
    pop_colors = get_population_colors(pop_names)
    if subtitle is None:
        optimal_likelihood = _read_optimal_likelihood(read_path("optimal_parameters.txt"))
        subtitle = f"Log-likelihood: {optimal_likelihood:.6g}"

    common_kwargs = dict(
        scale_factor=1,  # Predicted counts saved to disk are already scaled.
        pop_names=pop_names,
        pop_colors=pop_colors,
        log_scale=log_scale,
        xlabel=xlabel,
        ylabel=ylabel,
        subtitle=subtitle,
        title_fontsize=title_fontsize,
        subtitle_fontsize=subtitle_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
    )

    # --- Autosomes ---
    autosome_bins = _read_bins(read_path("tract_length_autosome_bins"))
    autosome_data = _read_population_rows(read_path("autosome_sample_tract_distribution"), pop_names)
    autosome_predicted = _read_population_rows(read_path("autosome_predicted_tract_distribution"), pop_names)
    _plot_panel(
        xbins=autosome_bins,
        observed_dict=autosome_data,
        predicted_dict=autosome_predicted,
        title=autosome_title,
        output_path=str(write_path("autosomes_all_populations.pdf")),
        **common_kwargs,
    )

    # --- Allosomes (if present): combined or separate by sex, depending on sum_female_and_male_allosome_tracts ---
    if read_path("allosome_sample_tract_distribution").exists():
        allosome_bins = _read_bins(read_path("tract_length_allosome_bins"))
        if sum_female_and_male_allosome_tracts:
            allosome_data = _read_population_rows(read_path("allosome_sample_tract_distribution"), pop_names)
            allosome_predicted = _read_population_rows(read_path("allosome_predicted_tract_distribution"), pop_names)
            _plot_panel(
                xbins=allosome_bins,
                observed_dict=allosome_data,
                predicted_dict=allosome_predicted,
                title=allosome_title,
                output_path=str(write_path("allosomes_all_populations.pdf")),
                **common_kwargs,
            )
        else:
            female_data = _read_population_rows(read_path("female_allosome_sample_tract_distribution"), pop_names)
            female_predicted = _read_population_rows(read_path("female_allosome_predicted_tract_distribution"), pop_names)
            _plot_panel(
                xbins=allosome_bins,
                observed_dict=female_data,
                predicted_dict=female_predicted,
                title=female_allosome_title,
                output_path=str(write_path("female_allosomes_all_populations.pdf")),
                **common_kwargs,
            )
            male_data = _read_population_rows(read_path("male_allosome_sample_tract_distribution"), pop_names)
            male_predicted = _read_population_rows(read_path("male_allosome_predicted_tract_distribution"), pop_names)
            _plot_panel(
                xbins=allosome_bins,
                observed_dict=male_data,
                predicted_dict=male_predicted,
                title=male_allosome_title,
                output_path=str(write_path("male_allosomes_all_populations.pdf")),
                **common_kwargs,
            )

    print(f"Tract length distribution plots regenerated in: {save_dir}")
    logger.info(f"Tract length distribution plots regenerated in: {save_dir}")


def plot_all_from_output_directories(output_dirs, output_filename_format: str | None = None,
                                    log_scale: bool = True, save_dir: str | Path | None = None,
                                    sum_female_and_male_allosome_tracts: bool = True,
                                    title_fontsize: float | None = None, subtitle_fontsize: float | None = None,
                                    label_fontsize: float | None = None, tick_fontsize: float | None = None,
                                    legend_fontsize: float | None = None) -> None:
    """
    Convenience wrapper around :func:`~tracts.plot.plot_admixture_from_output`,
    :func:`~tracts.plot.plot_migration_matrices_from_output` and
    :func:`~tracts.plot.plot_tract_length_distributions_from_output`, to re-produce all three sets of plots for
    a list of output directories (e.g. corresponding to different models or populations) in a single call.

    Since titles are inherently specific to each run, they are not exposed here; call the individual plotting
    functions directly to customize titles or subtitles. Font sizes, being run-agnostic style choices, are
    exposed and applied uniformly across every directory in ``output_dirs``.

    Parameters
    ----------
    output_dirs: Iterable[str | Path]
        The output directories for which to (re-)produce plots.
    output_filename_format: str | None
        See :func:`~tracts.plot.plot_admixture_from_output`. If None (default), it is auto-detected separately
        for each directory in ``output_dirs``, so directories using different formats are supported. Pass an
        explicit value only if every directory in ``output_dirs`` shares that same format.
    log_scale: bool
        See :func:`~tracts.plot.plot_tract_length_distributions_from_output`.
    save_dir: str | Path | None
        See :func:`~tracts.plot.plot_admixture_from_output`. Applied to every directory in ``output_dirs``; since
        each output directory typically has its own ``output_filename_format`` prefix, plots from multiple runs
        can safely be collected into the same ``save_dir`` without overwriting each other.
    sum_female_and_male_allosome_tracts: bool
        See :func:`~tracts.plot.plot_tract_length_distributions_from_output`.
    title_fontsize: float | None
        The font size of plot titles, applied to every directory in ``output_dirs``. If None (default), each
        plotting function's own default is used (a fixed size for tract length/admixture plots, an adaptive
        size for the migration matrices plot).
    subtitle_fontsize: float | None
        The font size of plot subtitles (tract length distribution plots only). If None, the default of
        :func:`~tracts.plot.plot_tract_length_distributions_from_output` is used.
    label_fontsize: float | None
        The font size of axis labels (tract length distribution and admixture plots). If None, each plotting
        function's own default is used.
    tick_fontsize: float | None
        The font size of tick labels, applied to every directory in ``output_dirs``. If None (default), each
        plotting function's own default is used (a fixed size for tract length/admixture plots, an adaptive
        size for the migration matrices plot).
    legend_fontsize: float | None
        The font size of legends (tract length distribution and admixture plots). If None, each plotting
        function's own default is used.
    """
    admixture_kwargs = {}
    if title_fontsize is not None:
        admixture_kwargs["title_fontsize"] = title_fontsize
    if label_fontsize is not None:
        admixture_kwargs["label_fontsize"] = label_fontsize
    if tick_fontsize is not None:
        admixture_kwargs["tick_fontsize"] = tick_fontsize
    if legend_fontsize is not None:
        admixture_kwargs["legend_fontsize"] = legend_fontsize

    migration_matrices_kwargs = {}
    if title_fontsize is not None:
        migration_matrices_kwargs["title_fontsize"] = title_fontsize
    if tick_fontsize is not None:
        migration_matrices_kwargs["tick_fontsize"] = tick_fontsize

    tract_length_kwargs = dict(admixture_kwargs)
    if subtitle_fontsize is not None:
        tract_length_kwargs["subtitle_fontsize"] = subtitle_fontsize

    for output_dir in output_dirs:

        plot_admixture_from_output(output_dir=output_dir,
                                   output_filename_format=output_filename_format,
                                   save_dir=save_dir,
                                   **admixture_kwargs)
        
        plot_migration_matrices_from_output(output_dir=output_dir,
                                            output_filename_format=output_filename_format,
                                            save_dir=save_dir,
                                            **migration_matrices_kwargs)
        
        plot_tract_length_distributions_from_output(output_dir=output_dir,
                                                    output_filename_format=output_filename_format,
                                                    log_scale=log_scale,
                                                    save_dir=save_dir,
                                                    sum_female_and_male_allosome_tracts=sum_female_and_male_allosome_tracts,
                                                    **tract_length_kwargs)
