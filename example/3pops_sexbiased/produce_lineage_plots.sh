#!/usr/bin/env bash
# Produces multi-population migration lineage figures (see tracts.lineage_flow_plots.plot_lineages_by_population)
# without re-running inference. For each fit type (M: fit_M.sh, DC_hotstart: fit_DC_hotstart.sh), the model with
# the highest likelihood among the <POP>_MODELS of each population is selected, and drawn in one figure with the
# autosomal panels of all populations side by side (all_pops_best_<fit>_autosome_lineages.{pdf,png}), and in a
# second one with the X chromosome panels on a second row (all_pops_best_<fit>_autosome_X_lineages.{pdf,png}; only
# if a selected model is sex-biased). Likelihoods are raw log-likelihoods, not penalized for the number of
# parameters. The single-run lineage plot is part of each run's own output (optimal_model/figures/).
# Uses the same RUN_<POP>/<POP>_MODELS convention as those scripts, and the same run lookup as
# produce_plots.sh: the output directory is read from each <POP>/<POP>_<model>.yaml driver file, and the
# latest YYYYMMDD_HHMMSS run is used.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Per-population flags ───────────────────────────────────────────────────────
RUN_ACB=1
RUN_ASW=1
RUN_CLM=1
RUN_MXL=1
RUN_PEL=1
RUN_PUR=1

# ── Which fits to plot ─────────────────────────────────────────────────────────
PLOT_M=1            # models of fit_M.sh            (<POP>_<model>.yaml)
PLOT_DC_HOTSTART=1  # models of fit_DC_hotstart.sh  (<POP>_<model>_DC_hotstart.yaml)

# ── Candidate models per population (same as fit_M.sh / fit_DC_hotstart.sh); the best one is plotted ──
ACB_MODELS=(ppp ppx_xxp_pxx)
ASW_MODELS=(ppp ppx_xxp_pxx)
CLM_MODELS=(ppp ccc)
MXL_MODELS=(ppp ccp)
PEL_MODELS=(ppp pcc)
PUR_MODELS=(ppp cpc)

# Directory where the figures are saved. Can be overridden from the environment.
SAVE_DIR="${SAVE_DIR:-$SCRIPT_DIR/plots}"

# If 1, an incomplete latest run (still running or crashed) falls back to the most recent *complete* run
# of that model instead of being skipped. Can be overridden from the environment.
USE_LATEST_COMPLETE_RUN="${USE_LATEST_COMPLETE_RUN:-0}"

# ── Build the (population, model) list ─────────────────────────────────────────
FITS=()
[ "$PLOT_M" -eq 1 ] && FITS+=("M")
[ "$PLOT_DC_HOTSTART" -eq 1 ] && FITS+=("DC_hotstart")

PAIRS=()
add_pop() {
    local pop=$1; shift
    for fit in "${FITS[@]}"; do
        for model in "$@"; do
            PAIRS+=("$pop $model $fit")
        done
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
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML

from tracts.driver_utils import _OUTPUT_SUBDIRS

from tracts.lineage_flow_plots import LINEAGE_CHROMOSOMES as CHROMOSOMES, plot_lineages_by_population
from tracts.plot import read_migration_matrices_from_output, _read_optimal_likelihood

script_dir = Path(os.environ["SCRIPT_DIR"])

# ════════════════════════════════════════════════════════════════════════════
# Plot arguments -- see plot_lineages_by_population / plot_lineage_flows for every option.
# ════════════════════════════════════════════════════════════════════════════
N_LINEAGES = None  # None: per panel, the fewest (100 to 1000) that show every source population
LINEAGE_KWARGS = {
    # "show_bands": "admixed",     # or "all", "none"
    # "source_gap": 0.25,          # admixed-to-sources gap, fraction of the number of lineages
    # "alpha": 0.6,
    # "panel_width": 3.2,          # inches per population column
}
# ════════════════════════════════════════════════════════════════════════════

save_dir = Path(os.environ["SAVE_DIR"])
use_latest_complete_run = os.environ.get("USE_LATEST_COMPLETE_RUN", "0") == "1"
pairs = [line.split() for line in os.environ["PAIRS"].splitlines() if line.strip()]
yaml_loader = YAML(typ="safe")


def most_recent_output_dir(pop: str, model: str) -> Path | None:
    """Same lookup as produce_plots.sh."""
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


FIT_SUFFIXES = {"M": "", "DC_hotstart": "_DC_hotstart"}  # driver file <POP>_<model><suffix>.yaml

# Best-likelihood model of each population, for each fit type.
best = {}  # fit -> {pop: (model, likelihood, output_dir)}
for pop, model, fit in pairs:
    output_dir = most_recent_output_dir(pop, model + FIT_SUFFIXES[fit])
    if output_dir is None:
        continue
    params_file = next((output_dir / _OUTPUT_SUBDIRS["optimal_parameters.txt"]).glob("*optimal_parameters.txt"))
    likelihood = _read_optimal_likelihood(params_file)
    print(f"  -> {pop} {model} ({fit}): log-likelihood {likelihood:.6g}  [{output_dir}]")
    current = best.setdefault(fit, {}).get(pop)
    if current is None or likelihood > current[1]:
        best[fit][pop] = (model, likelihood, output_dir)
if not best:
    sys.exit("No completed output directories found for the selected populations/models.")


def save(fig, name):
    output_path = save_dir / f"{name}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path.name}")


save_dir.mkdir(parents=True, exist_ok=True)
pop_order = list(dict.fromkeys(pop for pop, _, _ in pairs))

for fit, by_pop in best.items():
    pops = [pop for pop in pop_order if pop in by_pop]
    models = {pop: read_migration_matrices_from_output(by_pop[pop][2]) for pop in pops}
    panel_titles = {pop: f"{pop}: {by_pop[pop][0]}\nlog-likelihood {by_pop[pop][1]:.1f}" for pop in pops}
    print(f"\n{fit}: best models: " + ", ".join(f"{pop} {by_pop[pop][0]}" for pop in pops))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig, _ = plot_lineages_by_population(models, chromosomes="autosome", n_lineages=N_LINEAGES,
                                             panel_titles=panel_titles, **LINEAGE_KWARGS)
    save(fig, f"all_pops_best_{fit}_autosome_lineages")

    # Autosomes on the top row, X on the bottom row. Skipped when no selected model is sex-biased (the X rows
    # would be identical to the autosomal ones).
    if all(np.allclose(f, m) for f, m, _ in models.values()):
        print(f"  (all_pops_best_{fit}_autosome_X_lineages skipped: no sex bias in the selected models)")
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig, _ = plot_lineages_by_population(models, chromosomes=CHROMOSOMES, n_lineages=N_LINEAGES,
                                             panel_titles=panel_titles, **LINEAGE_KWARGS)
    save(fig, f"all_pops_best_{fit}_autosome_X_lineages")

print(f"\nPlots saved to: {save_dir}")
PYEOF
