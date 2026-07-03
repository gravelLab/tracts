#!/usr/bin/env bash
# Run tracts inference for all populations and print a likelihood summary.
# Set RUN_<POP>=1 to rerun a population, 0 to skip.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verbosity (0 = summary only, 1 = include likelihood table, 2 = also ancestry proportions) ──
VERBOSITY=1

# ── Per-population run flags ───────────────────────────────────────────────────
RUN_ACB=1
RUN_ASW=0
RUN_CLM=0
RUN_MXL=0
RUN_PEL=0
RUN_PUR=0

# ── Models per population (files must be <POP>/<POP>_<model>.py) ──────────────
ACB_MODELS=(ppp ppx_xxp_pxx)
ASW_MODELS=(ppp ppx_xxp_ppx ppx_xxp_pxx)
CLM_MODELS=(ppp ccc ccp)
MXL_MODELS=(ppp ccp)
PEL_MODELS=(ppp ccp)
PUR_MODELS=(ppp cpc cpp)

MAX_WORKERS=6

RERUN_POPS=()
_PIDS=()

_wait_for_slot() {
    while [ "${#_PIDS[@]}" -ge "$MAX_WORKERS" ]; do
        for i in "${!_PIDS[@]}"; do
            if ! kill -0 "${_PIDS[$i]}" 2>/dev/null; then
                unset '_PIDS[$i]'
            fi
        done
        _PIDS=("${_PIDS[@]}")
        sleep 0.2
    done
}

run_pop() {
    local pop=$1; shift
    local models=("$@")
    echo "──────────────────────────────────────────"
    echo " $pop"
    echo "──────────────────────────────────────────"
    for model in "${models[@]}"; do
        local script="$SCRIPT_DIR/$pop/${pop}_${model}.py"
        if [ -f "$script" ]; then
            echo "  → $model"
            _wait_for_slot
            (cd "$SCRIPT_DIR/$pop" && python "$script") &
            _PIDS+=($!)
        else
            echo "  ✗ not found: ${pop}_${model}.py"
        fi
    done
    RERUN_POPS+=("$pop")
}

[ "$RUN_ACB" -eq 1 ] && run_pop ACB "${ACB_MODELS[@]}"
[ "$RUN_ASW" -eq 1 ] && run_pop ASW "${ASW_MODELS[@]}"
[ "$RUN_CLM" -eq 1 ] && run_pop CLM "${CLM_MODELS[@]}"
[ "$RUN_MXL" -eq 1 ] && run_pop MXL "${MXL_MODELS[@]}"
[ "$RUN_PEL" -eq 1 ] && run_pop PEL "${PEL_MODELS[@]}"
[ "$RUN_PUR" -eq 1 ] && run_pop PUR "${PUR_MODELS[@]}"

wait  # wait for all background jobs before summarising

# ── Summary ────────────────────────────────────────────────────────────────────
# For each rerun population, scan every output_* directory, take the latest
# timestamped subdirectory (YYYYMMDD_HHMMSS sorts lexicographically), and
# extract the likelihood from *optimal_parameters.txt.

get_latest_run() {
    ls -d "$1"/[0-9]*/ 2>/dev/null | sort | tail -1
}

# Prints "<best_likelihood> <timestamp>" across all runs in out_dir.
# Returns empty if no runs have a likelihood line.
get_best_run_info() {
    local out_dir="$1"
    local best_lik="" best_ts=""
    for run_dir in $(ls -d "$out_dir"/[0-9]*/ 2>/dev/null | sort); do
        local params_file ts lik
        params_file=$(find "$run_dir" -maxdepth 1 -name "*optimal_parameters.txt" | head -1)
        [ -n "$params_file" ] || continue
        lik=$(grep "^likelihood" "$params_file" | awk '{print $2}')
        [ -n "$lik" ] || continue
        ts=$(basename "${run_dir%/}")
        if [ -z "$best_lik" ] || awk "BEGIN { exit !($lik > $best_lik) }"; then
            best_lik="$lik"
            best_ts="$ts"
        fi
    done
    [ -n "$best_lik" ] && echo "$best_lik $best_ts"
}

get_latest_likelihood() {
    local latest_run="$1"
    local params_file
    params_file=$(find "$latest_run" -maxdepth 1 -name "*optimal_parameters.txt" | head -1)
    [ -n "$params_file" ] || { echo "ERROR: optimal_parameters.txt missing in $(basename "$latest_run")"; return; }
    local lik
    lik=$(grep "^likelihood" "$params_file" | awk '{print $2}')
    [ -n "$lik" ] || { echo "ERROR: no likelihood line in $(basename "$params_file")"; return; }
    echo "$lik"
}

get_latest_yaml() {
    local yaml_file
    yaml_file=$(find "$1" -maxdepth 1 -name "*.yaml" | head -1)
    [ -n "$yaml_file" ] && echo "$yaml_file"
}

# query_yaml <yaml_file> <dotted.key.path> — returns a scalar value.
query_yaml() {
    python3 - "$1" "$2" <<'EOF'
import sys
from ruamel.yaml import YAML
yaml = YAML(typ="safe")
with open(sys.argv[1]) as f:
    d = yaml.load(f)
val = d
for k in sys.argv[2].split("."):
    val = val.get(k) if isinstance(val, dict) else None
print(val)
EOF
}

# query_yaml_len <yaml_file> <dotted.key.path> — returns the length of a list field.
query_yaml_len() {
    python3 - "$1" "$2" <<'EOF'
import sys
from ruamel.yaml import YAML
yaml = YAML(typ="safe")
with open(sys.argv[1]) as f:
    d = yaml.load(f)
val = d
for k in sys.argv[2].split("."):
    val = val.get(k) if isinstance(val, dict) else None
print(len(val) if isinstance(val, list) else 0)
EOF
}

get_num_individuals() {
    local yaml_file
    yaml_file=$(get_latest_yaml "$1")
    [ -n "$yaml_file" ] || { echo "ERROR: no yaml"; return; }
    query_yaml_len "$yaml_file" "samples.individual_names"
}

echo ""
echo "══════════════════════════════════════════"
echo " Results summary"
echo "══════════════════════════════════════════"
printf "%-8s  %-35s  %-6s  %s\n" "Pop" "Model (output dir)" "N_ind" "Likelihood"
printf "%-8s  %-35s  %-6s  %s\n" "--------" "-----------------------------------" "------" "----------"
for pop in ACB ASW CLM MXL PEL PUR; do
    for out_dir in "$SCRIPT_DIR/$pop"/output_*/; do
        [ -d "$out_dir" ] || continue
        model_name=$(basename "$out_dir")
        latest_run=$(get_latest_run "$out_dir")
        if [ -z "$latest_run" ]; then
            printf "%-8s  %-35s  %-6s  %s\n" "$pop" "$model_name" "N/A" "ERROR: no timestamped run dirs"
            continue
        fi
        n_ind=$(get_num_individuals "$latest_run")
        lik=$(get_latest_likelihood "$latest_run")
        printf "%-8s  %-35s  %-6s  %s\n" "$pop" "$model_name" "$n_ind" "$lik"
        best_info=$(get_best_run_info "$out_dir")
        if [ -n "$best_info" ]; then
            best_lik=$(echo "$best_info" | awk '{print $1}')
            best_ts=$(echo "$best_info" | awk '{print $2}')
            latest_ts=$(basename "${latest_run%/}")
            if [ "$best_ts" != "$latest_ts" ]; then
                latest_yaml=$(get_latest_yaml "$latest_run")
                reps=""
                [ -n "$latest_yaml" ] && reps=$(query_yaml "$latest_yaml" "optim.repetitions")
                printf "  *** WARNING: better likelihood %s found at earlier run %s (latest run: %s repetitions) ***\n" \
                    "$best_lik" "$best_ts" "${reps:-N/A}"
            fi
        fi
        if [ "$VERBOSITY" -ge 2 ]; then
            anc_file=$(find "$latest_run" -maxdepth 1 -name "*ancestry_proportions.txt" | head -1)
            [ -n "$anc_file" ] && cat "$anc_file" || echo "  (no ancestry_proportions.txt)"
        fi
    done
done
