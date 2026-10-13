#!/usr/bin/env bash
# Regenerates admixture, migration-matrix and tract-length plots for the most recent completed
# tracts run of each selected population/model, without re-running inference.
# Uses the same RUN_<POP>/<POP>_MODELS configuration convention as run_all.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Per-population run flags ───────────────────────────────────────────────────
RUN_ACB=1
RUN_ASW=1
RUN_CLM=1
RUN_MXL=1
RUN_PEL=1
RUN_PUR=1

# ── Models per population (driver files must be <POP>/<POP>_<model>.yaml) ─────
ACB_MODELS=(ppp ppx_xxp_pxx)
ASW_MODELS=(ppp ppx_xxp_pxx)
CLM_MODELS=(ppp ccc)
MXL_MODELS=(ppp ccp)
PEL_MODELS=(ppp ccc)
PUR_MODELS=(ppp cpc)

# Directory where the regenerated plots are saved (created if it does not already exist).
SAVE_DIR="$SCRIPT_DIR/plots"

# If 1, an incomplete latest run (still running or crashed) falls back to the most recent *complete* run
# of that model instead of being skipped. Can be overridden from the environment.
USE_LATEST_COMPLETE_RUN="${USE_LATEST_COMPLETE_RUN:-0}"

# ── Build the (population, model) list to plot, from the flags/arrays above ───
PAIRS=()
add_pop() {
    local pop=$1; shift
    for model in "$@"; do
        PAIRS+=("$pop $model")
    done
}
[ "$RUN_ACB" -eq 1 ] && add_pop ACB "${ACB_MODELS[@]}"
[ "$RUN_ASW" -eq 1 ] && add_pop ASW "${ASW_MODELS[@]}"
[ "$RUN_CLM" -eq 1 ] && add_pop CLM "${CLM_MODELS[@]}"
[ "$RUN_MXL" -eq 1 ] && add_pop MXL "${MXL_MODELS[@]}"
[ "$RUN_PEL" -eq 1 ] && add_pop PEL "${PEL_MODELS[@]}"
[ "$RUN_PUR" -eq 1 ] && add_pop PUR "${PUR_MODELS[@]}"

if [ "${#PAIRS[@]}" -eq 0 ]; then
    echo "No populations selected (all RUN_<POP> flags are 0)."
    exit 1
fi

SCRIPT_DIR="$SCRIPT_DIR" SAVE_DIR="$SAVE_DIR" USE_LATEST_COMPLETE_RUN="$USE_LATEST_COMPLETE_RUN" PAIRS="$(printf '%s\n' "${PAIRS[@]}")" python3 - <<'PYEOF'
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from scipy.stats import poisson
from ruamel.yaml import YAML

from tracts.driver_utils import _OUTPUT_SUBDIRS, get_population_colors, _bin_centers, _draw_migration_matrices
from tracts.plot import (
    plot_admixture_from_output,
    plot_migration_matrices_from_output,
    _resolve_output_paths,
    _read_bins,
    _read_population_names,
    _read_population_rows,
    _read_optimal_likelihood,
)

# ════════════════════════════════════════════════════════════════════════════
# Plot arguments -- edit these to customize the regenerated plots. See the
# docstrings of plot_admixture_from_output, plot_migration_matrices_from_output
# and plot_tract_length_distributions_from_output (tracts/plot.py) for every
# available option.
# ════════════════════════════════════════════════════════════════════════════
LOG_SCALE = True
SUM_FEMALE_AND_MALE_ALLOSOME_TRACTS = True
OUTPUT_FILENAME_FORMAT = None  # auto-detected per directory if left as None

# Both the tract length comparison figures and the migration matrix figures have two panels in a single
# row and are meant to sit side by side in the same document -- FIGURE_WIDTH_INCHES is their shared target
# display width (e.g. a page's full text width), so that, once both are placed at that width, their text
# reads at the same size. Adjust to match your target page/column width.
FIGURE_WIDTH_INCHES = 7.5
TRACT_LENGTH_FIGURE_HEIGHT_INCHES = 3.5
# The migration matrix figure has much less surrounding whitespace than the tract length figure (no
# legend/caption row below it), so savefig(bbox_inches="tight") crops it down by roughly 22% in width
# relative to its nominal figsize. The nominal width below is calibrated so the *cropped* output lands at
# FIGURE_WIDTH_INCHES -- i.e. the same final display width as the tract length figure -- rather than
# ending up narrower and having its (same point-size) text read larger once both are placed at that width.
_MIGRATION_MATRICES_CROP_FACTOR = 0.772
MIGRATION_MATRICES_FIGSIZE = (FIGURE_WIDTH_INCHES / _MIGRATION_MATRICES_CROP_FACTOR,
                               4.0 / _MIGRATION_MATRICES_CROP_FACTOR)

ADMIXTURE_KWARGS = {
    # "title_fontsize": 14,
    # "label_fontsize": 10,
    # "tick_fontsize": 6,
    # "legend_fontsize": 10,
}
MIGRATION_MATRICES_KWARGS = {
    "figsize": MIGRATION_MATRICES_FIGSIZE,
    # "title_fontsize": 12,
    # "tick_fontsize": 8,
    # "annot_fontsize": 7,
}
TRACT_LENGTH_KWARGS = {
    "figure_width_inches": FIGURE_WIDTH_INCHES,
    "figure_height_inches": TRACT_LENGTH_FIGURE_HEIGHT_INCHES,
    "title_fontsize": 11,
    "subtitle_fontsize": 10.5,
    "label_fontsize": 12,
    "tick_fontsize": 10.5,
    "legend_fontsize": 11,
}
# Combined per-population figure: X-chromosome tract length distributions for every model (a, b) on top,
# migration matrices of the complex model -- the last model listed in <POP>_MODELS -- below (c, d).
COMBINED_FIGURE_HEIGHT_INCHES = 9.0
COMBINED_FIGURE_HEIGHT_RATIOS = (1.0, 1.0)  # (tract length row, migration matrix row)
# ════════════════════════════════════════════════════════════════════════════

script_dir = Path(os.environ["SCRIPT_DIR"])
save_dir = Path(os.environ["SAVE_DIR"])
use_latest_complete_run = os.environ.get("USE_LATEST_COMPLETE_RUN", "0") == "1"
pairs = [line.split() for line in os.environ["PAIRS"].splitlines() if line.strip()]

yaml_loader = YAML(typ="safe")


def most_recent_output_dir(pop: str, model: str) -> Path | None:
    """
    Finds the most recent output directory for ``pop``/``model``, by reading the
    ``output.output_directory`` field of its driver file (resolving the "{date}" placeholder to the
    directory it expands into at run time) and picking the lexicographically-last "YYYYMMDD_HHMMSS"
    subdirectory. By default the latest run is always used: if it is not complete (still running or
    crashed), the model is skipped with a message. If USE_LATEST_COMPLETE_RUN is 1, the most recent
    complete run is used instead.
    """
    driver_path = script_dir / pop / f"{pop}_{model}.yaml"
    if not driver_path.exists():
        print(f"  x {pop}_{model}: driver file not found ({driver_path})", file=sys.stderr)
        return None

    with open(driver_path, "r") as f:
        driver_spec = yaml_loader.load(f)
    output_directory = driver_spec.get("output", {}).get("output_directory")
    if output_directory is None:
        print(f"  x {pop}_{model}: no output.output_directory in {driver_path}", file=sys.stderr)
        return None

    root = (script_dir / pop / output_directory.replace("{date}", "")).resolve()
    if not root.is_dir():
        print(f"  x {pop}_{model}: no output directory found at {root}", file=sys.stderr)
        return None

    runs = [d for d in root.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}_\d{6}", d.name)]
    if not runs:
        print(f"  x {pop}_{model}: no run found under {root}", file=sys.stderr)
        return None

    def is_complete(run):
        return bool(list((run / _OUTPUT_SUBDIRS["optimal_parameters.txt"]).glob("*optimal_parameters.txt")))

    latest = max(runs, key=lambda d: d.name)
    if is_complete(latest):
        return latest
    if use_latest_complete_run:
        complete = [d for d in runs if is_complete(d)]
        if complete:
            fallback = max(complete, key=lambda d: d.name)
            print(f"  ! {pop}_{model}: latest run {latest.name} is not complete; using {fallback.name}",
                  file=sys.stderr)
            return fallback
        print(f"  x {pop}_{model}: no complete run found under {root}; skipped", file=sys.stderr)
        return None
    print(f"  x {pop}_{model}: latest run {latest.name} is not complete (still running or crashed); skipped",
          file=sys.stderr)
    return None


def _draw_tract_length_panel(ax, xbins, observed_dict, predicted_dict, pop_names, pop_colors, log_scale,
                              xlabel, ylabel, panel_title, subtitle, alpha_ci,
                              title_fontsize, subtitle_fontsize, label_fontsize, tick_fontsize,
                              panel_label=None):
    """
    Draws one model's tract length distribution onto ``ax`` (observed counts as points, predicted
    distribution as a step function with a shaded Poisson prediction interval), without creating a
    legend -- used to build the multi-panel, one-legend-for-the-whole-figure comparison plots below.
    Mirrors tracts.driver_utils._plot_panel's drawing logic (scale_factor=1, since the predicted
    counts saved to disk are already on the count scale).
    """
    x_centers = _bin_centers(xbins)
    population_handles = []
    for pop in pop_names:
        color = pop_colors[pop]
        y_obs = np.asarray(observed_dict[pop], dtype=float)
        ax.scatter(x_centers, y_obs, s=18, color=color, alpha=0.95, edgecolor="white", linewidth=0.5, zorder=3)

        y_pred_bin = np.asarray(predicted_dict[pop], dtype=float)
        y_low_bin = np.asarray(poisson.ppf(alpha_ci / 2, y_pred_bin), dtype=float)
        y_high_bin = np.asarray(poisson.ppf(1 - alpha_ci / 2, y_pred_bin), dtype=float)

        y_pred_step = np.r_[y_pred_bin, y_pred_bin[-1]]
        y_low_step = np.r_[y_low_bin, y_low_bin[-1]]
        y_high_step = np.r_[y_high_bin, y_high_bin[-1]]

        ax.step(xbins, y_pred_step, where="post", color=color, lw=1.5, alpha=0.95, zorder=2)
        ax.fill_between(xbins, y_low_step, y_high_step, step="post", color=color, alpha=0.18, linewidth=0, zorder=1)

        population_handles.append(
            Line2D([0], [0], color=color, lw=1.5, marker='o', markersize=5,
                   markerfacecolor=color, markeredgecolor="white", label=pop)
        )

    ax.text(0.5, 1.1, panel_title, transform=ax.transAxes, ha='center', va='bottom', clip_on=False,
            fontsize=title_fontsize, fontweight='bold', fontfamily='DejaVu Sans')
    if subtitle is not None:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', clip_on=False,
                fontsize=subtitle_fontsize, color='0.4')
    if panel_label is not None:
        # Bold "a"/"b" tag for referencing each panel in text (e.g. "Fig. 2a"), placed above the panel
        # title (rather than beside it) so it never collides with a long/centered title string.
        ax.text(0.0, 1.15, panel_label, transform=ax.transAxes, ha='left', va='bottom', clip_on=False,
                fontsize=title_fontsize + 2, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    return population_handles


def _save_tract_length_comparison_figure(population, model_output_dirs, tract_kind, suptitle_suffix,
                                          bins_label, sample_label, predicted_label,
                                          output_filename_format, log_scale, save_dir,
                                          xlabel, ylabel, alpha_ci, title_fontsize, subtitle_fontsize,
                                          label_fontsize, tick_fontsize, legend_fontsize,
                                          figure_width_inches, figure_height_inches):
    """
    Builds one figure for ``population`` with one panel per (model, output_dir) in ``model_output_dirs``,
    all sharing a single legend (population colors + observed/predicted glyphs) below the figure.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Total figure width is fixed at figure_width_inches regardless of panel count (rather than scaling
    # with it), so the figure always renders at the intended display width -- e.g. a page's full text
    # width -- with the chosen font sizes reading at their intended size once placed in a document.
    fig, axes = plt.subplots(1, len(model_output_dirs), figsize=(figure_width_inches, figure_height_inches),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]

    # Build a single population->color mapping shared across every panel, from the union of
    # population names across all panels in canonical (sorted) order. Each panel's output file
    # can list its populations in a different order, and get_population_colors() assigns colors
    # by list position, so computing it per-panel (from that panel's own local order) could map
    # the same population to a different color in each panel despite the figure sharing one legend.
    all_pop_names = []
    for _, output_dir in model_output_dirs:
        _, _, read_path, _ = _resolve_output_paths(output_dir, output_filename_format, save_dir)
        for name in _read_population_names(read_path("ancestry_per_individual")):
            if name not in all_pop_names:
                all_pop_names.append(name)
    pop_colors = get_population_colors(sorted(all_pop_names))

    population_handles = None
    for panel_index, (ax, (model, output_dir)) in enumerate(zip(axes, model_output_dirs)):
        _, _, read_path, _ = _resolve_output_paths(output_dir, output_filename_format, save_dir)
        pop_names = _read_population_names(read_path("ancestry_per_individual"))
        optimal_likelihood = _read_optimal_likelihood(read_path("optimal_parameters.txt"))

        bins = _read_bins(read_path(bins_label))
        observed = _read_population_rows(read_path(sample_label), pop_names)
        predicted = _read_population_rows(read_path(predicted_label), pop_names)

        handles = _draw_tract_length_panel(
            ax, bins, observed, predicted, pop_names, pop_colors, log_scale, xlabel, ylabel,
            panel_title=model, subtitle=f"Log-likelihood: {optimal_likelihood:.6g}", alpha_ci=alpha_ci,
            title_fontsize=title_fontsize, subtitle_fontsize=subtitle_fontsize,
            label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
            panel_label=chr(ord('a') + panel_index),
        )
        if population_handles is None:
            population_handles = handles

    fig.suptitle(f"{population} — {suptitle_suffix}", fontsize=title_fontsize + 2, fontweight='bold',
                 fontfamily='DejaVu Sans')

    glyph_handles = [
        Line2D([0], [0], linestyle="None", marker='o', color='0.35', markerfacecolor='0.35',
               markeredgecolor="white", markersize=5, label="Observed"),
        Line2D([0], [0], linestyle='-', color='0.35', lw=1.5, label="Predicted"),
    ]
    # Anchored with x in figure-fraction (centered across every panel) but y in the first axes' own
    # fraction, so the vertical gap below the x-axis label stays consistent regardless of how many
    # panels (and therefore how much suptitle/panel-title space) the figure has.
    blended = mtransforms.blended_transform_factory(fig.transFigure, axes[0].transAxes)
    legend_pop = fig.legend(
        handles=population_handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), bbox_transform=blended,
        frameon=False, fontsize=legend_fontsize, ncol=min(len(population_handles), 4),
        title="Source population", title_fontsize=legend_fontsize,
    )
    fig.add_artist(legend_pop)
    fig.legend(handles=glyph_handles, loc="upper center", bbox_to_anchor=(0.5, -0.46), bbox_transform=blended,
               frameon=False, fontsize=legend_fontsize, ncol=2)

    output_path = save_dir / f"{population}_{tract_kind}_model_comparison.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {population} {tract_kind} model-comparison plot: {output_path}")


def plot_tract_length_model_comparison(population, model_output_dirs, output_filename_format, log_scale,
                                        save_dir, sum_female_and_male_allosome_tracts,
                                        xlabel="Tract Length (M)", ylabel="Count", alpha_ci=0.05,
                                        title_fontsize=14, subtitle_fontsize=10, label_fontsize=12,
                                        tick_fontsize=10, legend_fontsize=10,
                                        figure_width_inches=7.5, figure_height_inches=4.4):
    """
    Produces, for a single population, one tract length distribution comparison figure per tract kind
    (autosomal, and allosomal if present) with one panel per entry of ``model_output_dirs``
    (a list of ``(model_label, output_dir)`` pairs for that population), replacing the single-model
    plots normally produced by tracts.plot.plot_tract_length_distributions_from_output.
    """
    _save_tract_length_comparison_figure(
        population, model_output_dirs, "autosomes", "Autosomal tract length distributions",
        "tract_length_autosome_bins", "autosome_sample_tract_distribution",
        "autosome_predicted_tract_distribution", output_filename_format, log_scale, save_dir,
        xlabel, ylabel, alpha_ci, title_fontsize, subtitle_fontsize, label_fontsize, tick_fontsize,
        legend_fontsize, figure_width_inches, figure_height_inches,
    )

    # Allosomes are only present for sex-biased models; detect from the first model's output directory.
    _, _, read_path0, _ = _resolve_output_paths(model_output_dirs[0][1], output_filename_format, save_dir)
    if read_path0("allosome_sample_tract_distribution").exists():
        sample_label = "allosome_sample_tract_distribution" if sum_female_and_male_allosome_tracts \
            else "female_allosome_sample_tract_distribution"
        predicted_label = "allosome_predicted_tract_distribution" if sum_female_and_male_allosome_tracts \
            else "female_allosome_predicted_tract_distribution"
        _save_tract_length_comparison_figure(
            population, model_output_dirs, "allosomes", "X-chromosome tract length distributions",
            "tract_length_allosome_bins", sample_label, predicted_label, output_filename_format,
            log_scale, save_dir, xlabel, ylabel, alpha_ci, title_fontsize, subtitle_fontsize,
            label_fontsize, tick_fontsize, legend_fontsize, figure_width_inches, figure_height_inches,
        )


def plot_allosome_tracts_and_migration_matrices(population, model_output_dirs, output_filename_format,
                                                log_scale, save_dir, sum_female_and_male_allosome_tracts,
                                                figure_width_inches, figure_height_inches, height_ratios,
                                                xlabel="Tract Length (M)", ylabel="Count", alpha_ci=0.05,
                                                title_fontsize=14, subtitle_fontsize=10, label_fontsize=12,
                                                tick_fontsize=10, legend_fontsize=10, matrix_kwargs=None):
    """
    Produces, for a single population, one 2x2 figure: the X-chromosome tract length distributions of the
    first two entries of ``model_output_dirs`` in the top row (panels a, b), and the mean and sex-bias
    migration matrices of the last (complex) model in the bottom row (panels c, d). Skipped if the
    population has no allosomal output.
    """
    if len(model_output_dirs) < 2:
        print(f"  x {population}: needs two completed models (simple and complex); "
              f"combined X/migration figure skipped", file=sys.stderr)
        return
    matrix_kwargs = matrix_kwargs or {}
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _, _, read_path0, _ = _resolve_output_paths(model_output_dirs[0][1], output_filename_format, save_dir)
    if not read_path0("allosome_sample_tract_distribution").exists():
        print(f"  x {population}: no allosomal output; combined X/migration figure skipped", file=sys.stderr)
        return
    sample_label = "allosome_sample_tract_distribution" if sum_female_and_male_allosome_tracts \
        else "female_allosome_sample_tract_distribution"
    predicted_label = "allosome_predicted_tract_distribution" if sum_female_and_male_allosome_tracts \
        else "female_allosome_predicted_tract_distribution"

    fig = plt.figure(figsize=(figure_width_inches, figure_height_inches), layout="constrained")
    fig_top, fig_bottom = fig.subfigures(2, 1, height_ratios=height_ratios, hspace=0.04)

    # Top row: X-chromosome tract length distributions, one panel per model, sharing one legend and one
    # population->color mapping (see _save_tract_length_comparison_figure).
    tract_axes = fig_top.subplots(1, 2)
    all_pop_names = []
    for _, output_dir in model_output_dirs:
        _, _, read_path, _ = _resolve_output_paths(output_dir, output_filename_format, save_dir)
        for name in _read_population_names(read_path("ancestry_per_individual")):
            if name not in all_pop_names:
                all_pop_names.append(name)
    pop_colors = get_population_colors(sorted(all_pop_names))

    population_handles = None
    for panel_index, (ax, (model, output_dir)) in enumerate(zip(tract_axes, model_output_dirs)):
        _, _, read_path, _ = _resolve_output_paths(output_dir, output_filename_format, save_dir)
        pop_names = _read_population_names(read_path("ancestry_per_individual"))
        optimal_likelihood = _read_optimal_likelihood(read_path("optimal_parameters.txt"))
        handles = _draw_tract_length_panel(
            ax, _read_bins(read_path("tract_length_allosome_bins")),
            _read_population_rows(read_path(sample_label), pop_names),
            _read_population_rows(read_path(predicted_label), pop_names),
            pop_names, pop_colors, log_scale, xlabel, ylabel,
            panel_title=model, subtitle=f"Log-likelihood: {optimal_likelihood:.6g}", alpha_ci=alpha_ci,
            title_fontsize=title_fontsize, subtitle_fontsize=subtitle_fontsize,
            label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
            panel_label=chr(ord('a') + panel_index),
        )
        if population_handles is None:
            population_handles = handles

    glyph_handles = [
        Line2D([0], [0], linestyle="None", marker='o', color='0.35', markerfacecolor='0.35',
               markeredgecolor="white", markersize=5, label="Observed"),
        Line2D([0], [0], linestyle='-', color='0.35', lw=1.5, label="Predicted"),
    ]
    legend_handles = population_handles + glyph_handles
    fig_top.legend(handles=legend_handles, loc="outside lower center", frameon=False,
                   fontsize=legend_fontsize, ncol=len(legend_handles))
    fig_top.suptitle("X-chromosome tract length distributions", fontsize=title_fontsize + 1,
                     fontweight='bold', fontfamily='DejaVu Sans')

    # Bottom row: migration matrices of the complex model.
    complex_model, complex_output_dir = model_output_dirs[-1]
    _, _, read_path, _ = _resolve_output_paths(complex_output_dir, output_filename_format, save_dir)
    ax_mean, ax_bias = fig_bottom.subplots(1, 2)
    _draw_migration_matrices(
        fig_bottom, ax_mean, ax_bias,
        np.loadtxt(read_path("female_migration_matrix"), ndmin=2),
        np.loadtxt(read_path("male_migration_matrix"), ndmin=2),
        list(_read_population_names(read_path("ancestry_per_individual"))),
        panel_labels=("c", "d"), **matrix_kwargs,
    )
    fig_bottom.suptitle(f"Migration matrices ({complex_model})", fontsize=title_fontsize + 1,
                        fontweight='bold', fontfamily='DejaVu Sans')

    fig.suptitle(population, fontsize=title_fontsize + 3, fontweight='bold', fontfamily='DejaVu Sans')

    # Center each matrix (together with its colorbar) horizontally under the tract length panel above it.
    # Constrained layout is frozen first, as it would otherwise recompute the positions on save.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax_top, ax_matrix in zip(tract_axes, (ax_mean, ax_bias)):
        cbar_ax = ax_matrix.images[0].colorbar.ax
        matrix_box, cbar_box = ax_matrix.get_position(), cbar_ax.get_position()
        top_box = ax_top.get_position()
        dx = (top_box.x0 + top_box.x1) / 2 - (matrix_box.x0 + cbar_box.x1) / 2
        for ax in (ax_matrix, cbar_ax):
            box = ax.get_position()
            ax.set_position([box.x0 + dx, box.y0, box.width, box.height])

    output_path = save_dir / f"{population}_allosomes_and_migration_matrices.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {population} X-chromosome tracts + migration matrices plot: {output_path}")


resolved = []  # list of (pop, model, output_dir)
for pop, model in pairs:
    latest = most_recent_output_dir(pop, model)
    if latest is not None:
        print(f"  -> {pop}_{model}: {latest}")
        resolved.append((pop, model, latest))

if not resolved:
    sys.exit("No completed output directories found for the selected populations/models.")

for pop, model, output_dir in resolved:
    plot_admixture_from_output(output_dir=output_dir, output_filename_format=OUTPUT_FILENAME_FORMAT,
                               save_dir=save_dir, **ADMIXTURE_KWARGS)
    plot_migration_matrices_from_output(output_dir=output_dir, output_filename_format=OUTPUT_FILENAME_FORMAT,
                                        save_dir=save_dir, **MIGRATION_MATRICES_KWARGS)

# One tract length distribution comparison figure per population (one panel per model), instead of the
# default single-model plots.
by_pop = defaultdict(list)
for pop, model, output_dir in resolved:
    by_pop[pop].append((model, output_dir))

for pop, model_output_dirs in by_pop.items():
    plot_tract_length_model_comparison(
        population=pop,
        model_output_dirs=model_output_dirs,
        output_filename_format=OUTPUT_FILENAME_FORMAT,
        log_scale=LOG_SCALE,
        save_dir=save_dir,
        sum_female_and_male_allosome_tracts=SUM_FEMALE_AND_MALE_ALLOSOME_TRACTS,
        **TRACT_LENGTH_KWARGS,
    )
    plot_allosome_tracts_and_migration_matrices(
        population=pop,
        model_output_dirs=model_output_dirs,
        output_filename_format=OUTPUT_FILENAME_FORMAT,
        log_scale=LOG_SCALE,
        save_dir=save_dir,
        sum_female_and_male_allosome_tracts=SUM_FEMALE_AND_MALE_ALLOSOME_TRACTS,
        figure_height_inches=COMBINED_FIGURE_HEIGHT_INCHES,
        height_ratios=COMBINED_FIGURE_HEIGHT_RATIOS,
        matrix_kwargs={k: v for k, v in MIGRATION_MATRICES_KWARGS.items() if k != "figsize"},
        **{k: v for k, v in TRACT_LENGTH_KWARGS.items() if k != "figure_height_inches"},
    )

# Flatten: plot_admixture_from_output/plot_migration_matrices_from_output organize their output into
# category subdirectories (diagnostics/, optimal_model/figures/) under save_dir (the tract-length
# model-comparison figures above are already written directly to save_dir). Move every produced file
# directly into save_dir instead (overwriting a previous run's flattened file of the same name,
# matching the rest of the codebase's "save_dir overwrites in place" convention), then remove the
# now-empty subdirectories. Only two *nested* files from this same run colliding on the same
# flattened name is treated as an error, since that would silently discard one of them.
nested_files = [p for p in sorted(save_dir.rglob("*")) if p.is_file() and p.parent != save_dir]
seen = {}
for path in nested_files:
    if path.name in seen:
        sys.exit(f"Cannot flatten plots into {save_dir}: filename collision for {path.name} "
                 f"(from {seen[path.name]} and {path}).")
    seen[path.name] = path
for path in nested_files:
    path.replace(save_dir / path.name)
for d in sorted((p for p in save_dir.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
    if not any(d.iterdir()):
        d.rmdir()

print(f"\nPlots saved to: {save_dir}")
PYEOF
