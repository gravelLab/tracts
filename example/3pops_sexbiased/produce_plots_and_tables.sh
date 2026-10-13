#!/usr/bin/env bash
# Regenerates admixture, migration-matrix and tract-length plots, and the paper's LaTeX result
# tables, for the most recent completed tracts run of each selected population/model, without
# re-running inference. Uses the same RUN_<POP>/<POP>_MODELS configuration convention as run_all.sh.

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
PEL_MODELS=(ppp pcc)
PUR_MODELS=(ppp cpc)

# Directory where the regenerated plots and LaTeX tables are saved (created if it does not already exist).
SAVE_DIR="$SCRIPT_DIR/plots"

# If 1, an incomplete latest run (still running or crashed) falls back to the most recent *complete* run
# of that model instead of being skipped. Can be overridden from the environment.
USE_LATEST_COMPLETE_RUN="${USE_LATEST_COMPLETE_RUN:-0}"

# ── LaTeX table options (paper format, as in summarize_output.sh) ─────────────
LATEX_SIG_FIGS=3              # significant figures for parameter values in the LaTeX tables
PRINT_ANCESTRY_PROPORTIONS=1  # ancestry proportions table (single-pulse and second model)
PRINT_TRACT_COUNTS=1          # observed vs predicted tract counts table (single-pulse and second model)
PRINT_PARAMETERS=1            # best-fit parameters table of the second model
PRINT_SINGLE_PULSE_PARAMETERS=1  # best-fit parameters table of the single-pulse model

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

SCRIPT_DIR="$SCRIPT_DIR" SAVE_DIR="$SAVE_DIR" USE_LATEST_COMPLETE_RUN="$USE_LATEST_COMPLETE_RUN" \
    LATEX_SIG_FIGS="$LATEX_SIG_FIGS" PRINT_ANCESTRY_PROPORTIONS="$PRINT_ANCESTRY_PROPORTIONS" \
    PRINT_TRACT_COUNTS="$PRINT_TRACT_COUNTS" \
    PRINT_PARAMETERS="$PRINT_PARAMETERS" PRINT_SINGLE_PULSE_PARAMETERS="$PRINT_SINGLE_PULSE_PARAMETERS" \
    PAIRS="$(printf '%s\n' "${PAIRS[@]}")" python3 - <<'PYEOF'
import os
import re
import sys
from collections import defaultdict
from decimal import Decimal
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

LATEX_SIG_FIGS = int(os.environ["LATEX_SIG_FIGS"])
PRINT_ANCESTRY_PROPORTIONS = os.environ["PRINT_ANCESTRY_PROPORTIONS"] == "1"
PRINT_TRACT_COUNTS = os.environ["PRINT_TRACT_COUNTS"] == "1"
PRINT_PARAMETERS = os.environ["PRINT_PARAMETERS"] == "1"
PRINT_SINGLE_PULSE_PARAMETERS = os.environ["PRINT_SINGLE_PULSE_PARAMETERS"] == "1"

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


# ════════════════════════════════════════════════════════════════════════════
# LaTeX tables (paper format): ported from summarize_output.sh's print_paper_tables, writing
# each table (ancestry proportions, tract counts, best-fit parameters) to its own file instead of
# printing a combined block to stdout.
# ════════════════════════════════════════════════════════════════════════════
ANCESTRIES = ["EUR", "AMR", "AFR"]
ALIASES = {"NAT": "AMR"}

PROP_ROW_LABEL = {
    "ppx_xxp_pxx": "Predicted (three pulses)",
    "ccc": "Predicted (continuous pulse)",
    "ccp": "Predicted (single and continuous pulse)",
    "cpc": "Predicted (continuous)",
}
PARAMS_LABEL = {"ACB": "table:params_ACB", "PUR": "tab:PURcpc"}
SINGLE_LABEL = {"PUR": "tab:PURparams_pulse"}
THREE_PULSE_CAPTION = ("inferred using a three-pulse model (ppx\\_xxp\\_pxx) with joint EUR and AMR "
                       "migration $t_1$ generations ago, an AFR pulse at $t_2$, and a second EUR pulse at $t_3$.")
PARAMS_CAPTION = {
    "ACB": THREE_PULSE_CAPTION,
    "ASW": THREE_PULSE_CAPTION,
    "CLM": "inferred using a continuous-pulse model (ccc) in which the three source populations "
           "contribute migrants between $t_1$ and $t_2$ generations ago.",
    "MXL": "inferred using a model with a discrete AFR pulse $t_1$ generations ago and continuous "
           "EUR and AMR migration between $t_1$ and $t_2$ generations ago.",
    "PEL": "inferred using a continuous-migration model in which the three source populations "
           "contribute migrants between $t_1$ and $t_2$ generations ago.",
    "PUR": "inferred using a model (cpc) with a discrete AMR pulse $t_1$ generations ago and continuous "
           "EUR and AFR migration between $t_1$ and $t_2$ generations ago.",
}


def find_file(run_dir, suffix):
    matches = sorted(Path(run_dir).glob(f"*{suffix}")) + sorted(Path(run_dir).glob(f"*/*{suffix}"))
    return matches[0] if matches else None


def read_params(run_dir):
    params = {}
    lines = find_file(run_dir, "optimal_parameters.txt").read_text().splitlines()
    for line in lines[1:]:
        tokens = line.split()
        if tokens:
            params[tokens[0]] = float(tokens[-1])
    return params


def read_ancestry_table(run_dir, suffix):
    """Reads one of the fixed-width "Observed/Predicted (autosomes/X)" tables written by
    tracts.driver_utils.save_ancestry_table (ancestry_proportions.txt or tract_counts.txt)."""
    path = find_file(run_dir, suffix)
    if path is None:
        return None
    content = [l for l in path.read_text().splitlines() if l.strip() and set(l.strip()) - {"-"}]
    header = [ALIASES.get(a, a) for a in content[0].split()]
    rows = {}
    for line in content[1:]:
        tokens = line.split()
        label, values = " ".join(tokens[:-len(header)]), tokens[-len(header):]
        rows[label] = dict(zip(header, values))
    return rows


def read_proportions(run_dir):
    return read_ancestry_table(run_dir, "ancestry_proportions.txt")


def read_tract_counts(run_dir):
    return read_ancestry_table(run_dir, "tract_counts.txt")


def rate(params, anc):
    for name in [f"R{anc}"] + [f"R{a}" for a, b in ALIASES.items() if b == anc] + [f"targetpop_{anc}_rate"]:
        if name in params:
            return params[name]
    raise KeyError(f"no rate parameter for {anc}")


def sex_bias(params, anc):
    for name in [f"R{anc}_sex_bias"] + [f"R{a}_sex_bias" for a, b in ALIASES.items() if b == anc] + [f"targetpop_{anc}_sex_bias"]:
        if name in params:
            return params[name]
    raise KeyError(f"no sex-bias parameter for {anc}")


def fmt(value, sig_figs):
    """Round to sig_figs significant figures, never in scientific notation."""
    return format(Decimal(f"{value:.{sig_figs}g}"), "f")


def int_digits(value, sig_figs):
    return len(str(int(abs(float(fmt(value, sig_figs))))))


def rate_header(anc, suffix=""):
    sub = f"\\text{{{anc}}},{suffix}" if suffix else f"\\text{{{anc}}}"
    return f"{{$R_{{{sub}}}$}} & {{$R_{{{sub}}}^{{\\text{{SB}}}}$}}"


def fit_colspec(colspec, values):
    """Widen each S[table-format=...] column so that its value fits (never narrows it)."""
    values = iter(values)

    def widen(match):
        spec, value = match.group(1), next(values)
        sign = "-" if spec.startswith("-") or value.startswith("-") else ""
        int_part, dec_part = spec.lstrip("-").split(".")
        v_int, _, v_dec = value.lstrip("-").partition(".")
        return f"S[table-format={sign}{max(int(int_part), len(v_int))}.{max(int(dec_part), len(v_dec))}]"

    return re.sub(r"S\[table-format=([-\d.]+)\]", widen, colspec)


def print_proportions_table(f, pop, model, single, other):
    observed = single["Observed (autosomes)"], single["Observed (X)"]
    lines = [("Observed", observed[0], observed[1]),
             ("Predicted (single pulse)", single["Predicted (autosomes)"], single["Predicted (X)"]),
             (PROP_ROW_LABEL[model], other["Predicted (autosomes)"], other["Predicted (X)"])]
    width = max(len(label) for label, _, _ in lines)
    print(r"\begin{table}", file=f)
    print(r"\centering", file=f)
    print(r"\sisetup{round-mode=places, round-precision=4}", file=f)
    print(r"\begin{tabular}{l", file=f)
    print(r"                S[table-format=1.4] S[table-format=1.4] S[table-format=1.4]", file=f)
    print(r"                S[table-format=1.4] S[table-format=1.4] S[table-format=1.4]}", file=f)
    print(r"\toprule", file=f)
    print(r" & \multicolumn{3}{c}{Autosomes} & \multicolumn{3}{c}{X chromosome} \\", file=f)
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}", file=f)
    print(r" & {EUR} & {AMR} & {AFR} & {EUR} & {AMR} & {AFR} \\", file=f)
    print(r"\midrule", file=f)
    for label, aut, x in lines:
        values = [aut[a] for a in ANCESTRIES] + [x[a] for a in ANCESTRIES]
        print(f"{label:<{width}} & " + " & ".join(values) + r" \\", file=f)
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)
    print(f"\\caption{{Observed and predicted ancestry proportions for the {pop} population, on the autosomes and the X chromosome.}}", file=f)
    print(f"\\label{{tab:{pop}prop}}", file=f)
    print(r"\end{table}", file=f)


# Decimal places tract counts are rounded to for display (via siunitx's round-mode=places), independent
# of LATEX_SIG_FIGS (which only applies to the best-fit parameter tables).
TRACT_COUNTS_ROUND_PRECISION = 1


def print_tract_counts_table(f, pop, model, single, other):
    observed = single["Observed (autosomes)"], single["Observed (X)"]
    lines = [("Observed", observed[0], observed[1]),
             ("Predicted (single pulse)", single["Predicted (autosomes)"], single["Predicted (X)"]),
             (PROP_ROW_LABEL[model], other["Predicted (autosomes)"], other["Predicted (X)"])]
    width = max(len(label) for label, _, _ in lines)
    # Unlike ancestry proportions (always in [0, 1], hence a fixed S[table-format=1.4] column), tract
    # counts range from single to several thousand, so each column's integer-part width is computed
    # from its own values (the decimal part stays fixed at TRACT_COUNTS_ROUND_PRECISION, matching the
    # round-precision siunitx rounds the (higher-precision) values down to for display).
    columns = [[aut[a] for _, aut, _ in lines] for a in ANCESTRIES] + [[x[a] for _, _, x in lines] for a in ANCESTRIES]
    colspecs = [
        f"S[table-format={max(len(v.lstrip('-').split('.')[0]) for v in col)}.{TRACT_COUNTS_ROUND_PRECISION}]"
        for col in columns
    ]
    print(r"\begin{table}", file=f)
    print(r"\centering", file=f)
    print(f"\\sisetup{{round-mode=places, round-precision={TRACT_COUNTS_ROUND_PRECISION}}}", file=f)
    print(r"\begin{tabular}{l", file=f)
    print("                " + " ".join(colspecs[:3]), file=f)
    print("                " + " ".join(colspecs[3:]) + "}", file=f)
    print(r"\toprule", file=f)
    print(r" & \multicolumn{3}{c}{Autosomes} & \multicolumn{3}{c}{X chromosome} \\", file=f)
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}", file=f)
    print(r" & {EUR} & {AMR} & {AFR} & {EUR} & {AMR} & {AFR} \\", file=f)
    print(r"\midrule", file=f)
    for label, aut, x in lines:
        values = [aut[a] for a in ANCESTRIES] + [x[a] for a in ANCESTRIES]
        print(f"{label:<{width}} & " + " & ".join(values) + r" \\", file=f)
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)
    print(f"\\caption{{Observed and predicted tract counts for the {pop} population, on the autosomes and the X chromosome.}}", file=f)
    print(f"\\label{{tab:{pop}counts}}", file=f)
    print(r"\end{table}", file=f)


def print_params_table(f, pop, colspec, group_rows, header, values, caption, label):
    colspec = fit_colspec(colspec, values)
    print(r"\begin{table}", file=f)
    print(r"\centering", file=f)
    print(r"\small", file=f)
    print(r"\setlength{\tabcolsep}{4pt}", file=f)
    print(r"\begin{tabular}{" + colspec + "}", file=f)
    print(r"\toprule", file=f)
    for row in group_rows:
        print(row, file=f)
    print(header, file=f)
    print(r"\midrule", file=f)
    print(" " + " & ".join(values) + r" \\", file=f)
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)
    print(f"\\caption{{Best-fit parameters for the {pop} population, {caption}}}", file=f)
    print(f"\\label{{{label}}}", file=f)
    print(r"\end{table}", file=f)


def print_single_pulse_table(f, pop, sig_figs, p):
    lik = p["likelihood"]
    colspec = ("S[table-format=1.4] S[table-format=-1.3]\n"
               "                S[table-format=1.4] S[table-format=-1.3]\n"
               "                S[table-format=1.4] S[table-format=-1.3]\n"
               f"                S[table-format={int_digits(p['t'], sig_figs)}.2]\n"
               f"                S[table-format=-{int_digits(lik, sig_figs)}.0]")
    groups = [r" \multicolumn{2}{c}{EUR} & \multicolumn{2}{c}{AMR} & \multicolumn{2}{c}{AFR} & & \\",
              r"\cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6}"]
    header = (" " + rate_header("EUR") + "\n & " + rate_header("AMR") + "\n & " + rate_header("AFR")
              + "\n & {$t$}\n & {$\\mathcal{L}_{\\mathcal{D}}(\\bm{\\theta})$} \\\\")
    values = []
    for anc in ANCESTRIES:
        values += [fmt(rate(p, anc), sig_figs), fmt(sex_bias(p, anc), sig_figs)]
    values += [fmt(p["t"], sig_figs), fmt(lik, sig_figs)]
    caption = ("inferred using a single-pulse model in which the three source populations "
               "contribute at a single admixture event $t$ generations ago.")
    print_params_table(f, pop, colspec, groups, header, values, caption, SINGLE_LABEL.get(pop, f"tab:{pop}params_single"))


def print_model_table(f, pop, model, sig_figs, p):
    lik = fmt(p["likelihood"], sig_figs)
    if model == "ppx_xxp_pxx":
        colspec = ("S[table-format=1.4] S[table-format=-1.3]\n"
                   "                S[table-format=1.4] S[table-format=-1.3] S[table-format=2.2]\n"
                   "                S[table-format=1.4] S[table-format=-1.3] S[table-format=2.2]\n"
                   "                S[table-format=1.4] S[table-format=-1.3] S[table-format=2.2]\n"
                   "                S[table-format=-4.0]")
        groups = [" \\multicolumn{5}{c}{Pulse 1}\n & \\multicolumn{3}{c}{Pulse 2}\n & \\multicolumn{3}{c}{Pulse 3}\n & \\\\",
                  r"\cmidrule(lr){1-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}"]
        header = (" " + rate_header("EUR") + "\n & " + rate_header("AMR") + " & {$t_1$}\n & "
                  + rate_header("AFR") + " & {$t_2$}\n & " + rate_header("EUR", "2") + " & {$t_3$}\n"
                  " & {$\\mathcal{L}_{\\mathcal{D}}(\\bm{\\theta})$} \\\\")
        values = [fmt(rate(p, "EUR"), sig_figs), fmt(sex_bias(p, "EUR"), sig_figs),
                  fmt(rate(p, "AMR"), sig_figs), fmt(sex_bias(p, "AMR"), sig_figs), fmt(p["t1"], sig_figs),
                  fmt(rate(p, "AFR"), sig_figs), fmt(sex_bias(p, "AFR"), sig_figs), fmt(p["t2"], sig_figs),
                  fmt(p["REUR2"], sig_figs), fmt(p["REUR2_sex_bias"], sig_figs), fmt(p["t3"], sig_figs), lik]
    elif model == "ccc":
        if pop == "CLM":
            colspec = ("S[table-format=1.4] S[table-format=-1.2]\n"
                       "                S[table-format=1.4] S[table-format=-1.2]\n"
                       "                S[table-format=1.4] S[table-format=-1.2]\n"
                       "                S[table-format=2.1] S[table-format=1.1]\n"
                       "                S[table-format=-3.1]")
        else:
            colspec = ("S[table-format=1.5] S[table-format=-1.3]\n"
                       "                S[table-format=1.5] S[table-format=-1.3]\n"
                       "                S[table-format=1.5] S[table-format=-1.3]\n"
                       "                S[table-format=1.2] S[table-format=1.2]\n"
                       "                S[table-format=-3.0]")
        groups = [" \\multicolumn{2}{c}{EUR} & \\multicolumn{2}{c}{AMR} & \\multicolumn{2}{c}{AFR}\n"
                  " & \\multicolumn{2}{c}{Time window} & \\\\",
                  r"\cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"]
        header = (" " + rate_header("EUR") + "\n & " + rate_header("AMR") + "\n & " + rate_header("AFR")
                  + "\n & {$t_1$} & {$t_2$}\n & {$\\mathcal{L}_{\\mathcal{D}}(\\bm{\\theta})$} \\\\")
        values = []
        for anc in ANCESTRIES:
            values += [fmt(rate(p, anc), sig_figs), fmt(sex_bias(p, anc), sig_figs)]
        values += [fmt(p["t1"], sig_figs), fmt(p["t2"], sig_figs), lik]
    elif model in ("ccp", "cpc"):
        pulse = "AFR" if model == "ccp" else "AMR"
        continuous = [a for a in ANCESTRIES if a != pulse]
        r_fmt = "1.4" if model == "ccp" else "1.5"
        colspec = (f"S[table-format={r_fmt}] S[table-format=-1.3]\n"
                   f"                S[table-format={r_fmt}] S[table-format=-1.3]\n"
                   f"                S[table-format={r_fmt}] S[table-format=-1.3]\n"
                   f"                S[table-format=2.2] S[table-format={'1.2'}]\n"
                   f"                S[table-format=-3.0]")
        groups = [" \\multicolumn{4}{c}{Continuous migration} & \\multicolumn{2}{c}{Discrete pulse}\n"
                  " & \\multicolumn{2}{c}{Time window} & \\\\",
                  r"\cmidrule(lr){1-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"]
        header = (" " + rate_header(continuous[0]) + "\n & " + rate_header(continuous[1]) + "\n & "
                  + rate_header(pulse) + "\n & {$t_1$} & {$t_2$}\n & {$\\mathcal{L}_{\\mathcal{D}}(\\bm{\\theta})$} \\\\")
        values = []
        for anc in continuous + [pulse]:
            values += [fmt(rate(p, anc), sig_figs), fmt(sex_bias(p, anc), sig_figs)]
        values += [fmt(p["t1"], sig_figs), fmt(p["t2"], sig_figs), lik]
    else:
        print(f"% No paper table template for model {model}.", file=f)
        return
    print_params_table(f, pop, colspec, groups, header, values, PARAMS_CAPTION[pop], PARAMS_LABEL.get(pop, f"tab:{pop}params"))


# ── Per-table-file writers ────────────────────────────────────────────────────
# Each writes exactly one \begin{table}...\end{table} (or, if the data is unavailable, just an
# explanatory LaTeX comment) to its own file, so it can be \input{} at the right place in the text.

def _write_observed_predicted_file(path, pop, single_model, single_run, model, model_run,
                                    reader, table_printer, quantity_label, missing_file_label):
    with open(path, "w") as f:
        print(f"% {pop} {quantity_label}: {single_model} from {single_run}, {model} from {model_run}", file=f)
        single_data, model_data = reader(single_run), reader(model_run)
        if single_data is None or model_data is None:
            print(f"% (missing {missing_file_label}: cannot build the {quantity_label} table)", file=f)
            return
        if any(
            [float(single_data[row][anc]) for anc in ANCESTRIES] !=
            [float(model_data[row][anc]) for anc in ANCESTRIES]
            for row in ("Observed (autosomes)", "Observed (X)")
        ):
            warning = (f"WARNING: {pop}: observed values differ between the {single_model} and {model} runs "
                       f"(different samples?); {quantity_label} table skipped.")
            print(f"% {warning}", file=f)
            print(f"*** {warning}", file=sys.stderr)
            return
        table_printer(f, pop, model, single_data, model_data)


def write_ancestry_proportions_file(path, pop, single_model, single_run, model, model_run):
    _write_observed_predicted_file(path, pop, single_model, single_run, model, model_run,
                                    read_proportions, print_proportions_table,
                                    "ancestry proportions", "ancestry_proportions.txt")


def write_tract_counts_file(path, pop, single_model, single_run, model, model_run):
    _write_observed_predicted_file(path, pop, single_model, single_run, model, model_run,
                                    read_tract_counts, print_tract_counts_table,
                                    "tract counts", "tract_counts.txt")


def write_single_pulse_params_file(path, pop, single_model, single_run, sig_figs):
    with open(path, "w") as f:
        print(f"% {pop} single-pulse parameters: {single_model} from {single_run}", file=f)
        print_single_pulse_table(f, pop, sig_figs, read_params(single_run))


def write_model_params_file(path, pop, model, model_run, sig_figs):
    with open(path, "w") as f:
        print(f"% {pop} {model} parameters: from {model_run}", file=f)
        print_model_table(f, pop, model, sig_figs, read_params(model_run))


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

# LaTeX tables (paper format): one file per table (ancestry proportions, tract counts, single-pulse
# parameters, second/complex-model parameters) per population, built from the single-pulse model
# (first entry of <POP>_MODELS) and the second/complex model (last entry) -- so each table can be
# \input{} independently at the right place in the paper text.
TABLE_KINDS = [
    (PRINT_ANCESTRY_PROPORTIONS, "ancestry_proportions", "ancestry proportions"),
    (PRINT_TRACT_COUNTS, "tract_counts", "tract counts"),
    (PRINT_SINGLE_PULSE_PARAMETERS, "params_single", "single-pulse parameters"),
    (PRINT_PARAMETERS, None, "second-model parameters"),  # filename suffix depends on the model name
]

for pop, model_output_dirs in by_pop.items():
    if len(model_output_dirs) < 2:
        msg = f"{pop}: skipped -- needs two completed models (single-pulse and second)"
        print(f"*** {msg}", file=sys.stderr)
        for enabled, suffix, _ in TABLE_KINDS:
            if enabled and suffix is not None:
                with open(save_dir / f"{pop}_{suffix}.tex", "w") as f:
                    print(f"% ===== {msg} =====", file=f)
        continue

    single_model, single_run = model_output_dirs[0]
    model, model_run = model_output_dirs[-1]

    if PRINT_ANCESTRY_PROPORTIONS:
        path = save_dir / f"{pop}_ancestry_proportions.tex"
        write_ancestry_proportions_file(path, pop, single_model, single_run, model, model_run)
        print(f"  -> {pop} ancestry proportions table: {path}")
    if PRINT_TRACT_COUNTS:
        path = save_dir / f"{pop}_tract_counts.tex"
        write_tract_counts_file(path, pop, single_model, single_run, model, model_run)
        print(f"  -> {pop} tract counts table: {path}")
    if PRINT_SINGLE_PULSE_PARAMETERS:
        path = save_dir / f"{pop}_params_single.tex"
        write_single_pulse_params_file(path, pop, single_model, single_run, LATEX_SIG_FIGS)
        print(f"  -> {pop} single-pulse parameters table: {path}")
    if PRINT_PARAMETERS:
        path = save_dir / f"{pop}_params_{model}.tex"
        write_model_params_file(path, pop, model, model_run, LATEX_SIG_FIGS)
        print(f"  -> {pop} {model} parameters table: {path}")

print(f"\nPlots and tables saved to: {save_dir}")
PYEOF
