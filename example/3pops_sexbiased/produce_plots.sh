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
from pathlib import Path

from ruamel.yaml import YAML

from tracts.driver_utils import _OUTPUT_SUBDIRS
from tracts.plot import plot_all_from_output_directories

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


output_dirs = []
for pop, model in pairs:
    latest = most_recent_output_dir(pop, model)
    if latest is not None:
        print(f"  -> {pop}_{model}: {latest}")
        output_dirs.append(latest)

if not output_dirs:
    sys.exit("No completed output directories found for the selected populations/models.")

plot_all_from_output_directories(
    output_dirs=output_dirs,
    output_filename_format=OUTPUT_FILENAME_FORMAT,
    log_scale=LOG_SCALE,
    save_dir=save_dir,
    sum_female_and_male_allosome_tracts=SUM_FEMALE_AND_MALE_ALLOSOME_TRACTS,
    admixture_kwargs=ADMIXTURE_KWARGS,
    migration_matrices_kwargs=MIGRATION_MATRICES_KWARGS,
    tract_length_kwargs=TRACT_LENGTH_KWARGS,
)

# Flatten: plot_all_from_output_directories organizes its output into category subdirectories
# (diagnostics/, length_distributions/figures/, optimal_model/figures/) under save_dir. Move every
# produced file directly into save_dir instead (overwriting a previous run's flattened file of the
# same name, matching the rest of the codebase's "save_dir overwrites in place" convention), then
# remove the now-empty subdirectories. Only two *nested* files from this same run colliding on the
# same flattened name is treated as an error, since that would silently discard one of them.
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
