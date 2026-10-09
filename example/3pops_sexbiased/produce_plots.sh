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

SCRIPT_DIR="$SCRIPT_DIR" SAVE_DIR="$SAVE_DIR" PAIRS="$(printf '%s\n' "${PAIRS[@]}")" python3 - <<'PYEOF'
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from scipy.stats import poisson
from ruamel.yaml import YAML

from tracts.driver_utils import _OUTPUT_SUBDIRS, get_population_colors, _bin_centers
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

ADMIXTURE_KWARGS = {
    # "title_fontsize": 14,
    # "label_fontsize": 10,
    # "tick_fontsize": 6,
    # "legend_fontsize": 10,
}
MIGRATION_MATRICES_KWARGS = {
    # "title_fontsize": 12,
    # "tick_fontsize": 8,
    # "annot_fontsize": 7,
}
TRACT_LENGTH_KWARGS = {
    # "title_fontsize": 14,
    # "subtitle_fontsize": 10,
    # "label_fontsize": 12,
    # "tick_fontsize": 10,
    # "legend_fontsize": 10,
}
# ════════════════════════════════════════════════════════════════════════════

script_dir = Path(os.environ["SCRIPT_DIR"])
save_dir = Path(os.environ["SAVE_DIR"])
pairs = [line.split() for line in os.environ["PAIRS"].splitlines() if line.strip()]

yaml_loader = YAML(typ="safe")


def most_recent_output_dir(pop: str, model: str) -> Path | None:
    """
    Finds the most recent *completed* output directory for ``pop``/``model``, by reading the
    ``output.output_directory`` field of its driver file (resolving the "{date}" placeholder to the
    directory it expands into at run time) and picking the lexicographically-last "YYYYMMDD_HHMMSS"
    subdirectory that actually contains a completed run (an interrupted/crashed run is skipped, even
    if it is the most recent by timestamp).
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

    complete_runs = [
        d for d in root.iterdir()
        if d.is_dir() and list((d / _OUTPUT_SUBDIRS["optimal_parameters.txt"]).glob("*optimal_parameters.txt"))
    ]
    if not complete_runs:
        print(f"  x {pop}_{model}: no completed run found under {root}", file=sys.stderr)
        return None

    return max(complete_runs, key=lambda d: d.name)


def _draw_tract_length_panel(ax, xbins, observed_dict, predicted_dict, pop_names, pop_colors, log_scale,
                              xlabel, ylabel, panel_title, subtitle, alpha_ci,
                              title_fontsize, subtitle_fontsize, label_fontsize, tick_fontsize):
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
        ax.scatter(x_centers, y_obs, s=30, color=color, alpha=0.95, edgecolor="white", linewidth=0.6, zorder=3)

        y_pred_bin = np.asarray(predicted_dict[pop], dtype=float)
        y_low_bin = np.asarray(poisson.ppf(alpha_ci / 2, y_pred_bin), dtype=float)
        y_high_bin = np.asarray(poisson.ppf(1 - alpha_ci / 2, y_pred_bin), dtype=float)

        y_pred_step = np.r_[y_pred_bin, y_pred_bin[-1]]
        y_low_step = np.r_[y_low_bin, y_low_bin[-1]]
        y_high_step = np.r_[y_high_bin, y_high_bin[-1]]

        ax.step(xbins, y_pred_step, where="post", color=color, lw=2.2, alpha=0.95, zorder=2)
        ax.fill_between(xbins, y_low_step, y_high_step, step="post", color=color, alpha=0.18, linewidth=0, zorder=1)

        population_handles.append(
            Line2D([0], [0], color=color, lw=2.2, marker='o', markersize=6,
                   markerfacecolor=color, markeredgecolor="white", label=pop)
        )

    ax.text(0.5, 1.1, panel_title, transform=ax.transAxes, ha='center', va='bottom', clip_on=False,
            fontsize=title_fontsize, fontweight='bold', fontfamily='DejaVu Sans')
    if subtitle is not None:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', clip_on=False,
                fontsize=subtitle_fontsize, color='0.4')
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
                                          label_fontsize, tick_fontsize, legend_fontsize):
    """
    Builds one figure for ``population`` with one panel per (model, output_dir) in ``model_output_dirs``,
    all sharing a single legend (population colors + observed/predicted glyphs) below the figure.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(model_output_dirs), figsize=(7.2 * len(model_output_dirs), 5.8),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]

    population_handles = None
    for ax, (model, output_dir) in zip(axes, model_output_dirs):
        _, _, read_path, _ = _resolve_output_paths(output_dir, output_filename_format, save_dir)
        pop_names = _read_population_names(read_path("ancestry_per_individual"))
        pop_colors = get_population_colors(pop_names)
        optimal_likelihood = _read_optimal_likelihood(read_path("optimal_parameters.txt"))

        bins = _read_bins(read_path(bins_label))
        observed = _read_population_rows(read_path(sample_label), pop_names)
        predicted = _read_population_rows(read_path(predicted_label), pop_names)

        handles = _draw_tract_length_panel(
            ax, bins, observed, predicted, pop_names, pop_colors, log_scale, xlabel, ylabel,
            panel_title=model, subtitle=f"Log-likelihood: {optimal_likelihood:.6g}", alpha_ci=alpha_ci,
            title_fontsize=title_fontsize, subtitle_fontsize=subtitle_fontsize,
            label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
        )
        if population_handles is None:
            population_handles = handles

    fig.suptitle(f"{population} — {suptitle_suffix}", fontsize=title_fontsize + 2, fontweight='bold',
                 fontfamily='DejaVu Sans')

    glyph_handles = [
        Line2D([0], [0], linestyle="None", marker='o', color='0.35', markerfacecolor='0.35',
               markeredgecolor="white", markersize=6, label="Observed"),
        Line2D([0], [0], linestyle='-', color='0.35', lw=2.2, label="Predicted"),
    ]
    # Anchored with x in figure-fraction (centered across every panel) but y in the first axes' own
    # fraction, so the vertical gap below the x-axis label stays consistent regardless of how many
    # panels (and therefore how much suptitle/panel-title space) the figure has.
    blended = mtransforms.blended_transform_factory(fig.transFigure, axes[0].transAxes)
    legend_pop = fig.legend(
        handles=population_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), bbox_transform=blended,
        frameon=False, fontsize=legend_fontsize, ncol=min(len(population_handles), 4),
        title="Source population", title_fontsize=legend_fontsize,
    )
    fig.add_artist(legend_pop)
    fig.legend(handles=glyph_handles, loc="upper center", bbox_to_anchor=(0.5, -0.29), bbox_transform=blended,
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
                                        tick_fontsize=10, legend_fontsize=10):
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
        legend_fontsize,
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
            label_fontsize, tick_fontsize, legend_fontsize,
        )


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
