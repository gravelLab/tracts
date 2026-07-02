#!/usr/bin/env bash
# Run tracts inference for all populations and print a likelihood summary.
# Set RUN_<POP>=1 to rerun a population, 0 to skip.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Per-population run flags ───────────────────────────────────────────────────
RUN_ACB=0
RUN_ASW=0
RUN_CLM=1
RUN_MXL=1
RUN_PEL=1
RUN_PUR=1

# ── Models per population (files must be <POP>/<POP>_<model>.py) ──────────────
ACB_MODELS=(one_pulse ppx_xxp_pxx)
ASW_MODELS=(one_pulse ppx_xxp_ppx ppx_xxp_pxx)
CLM_MODELS=(one_pulse ccc ccp)
MXL_MODELS=(one_pulse ccp)
PEL_MODELS=(one_pulse ccp)
PUR_MODELS=(one_pulse cpc cpp)

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

get_latest_likelihood() {
    local out_dir="$1"
    local latest_run
    latest_run=$(ls -d "$out_dir"/[0-9]*/ 2>/dev/null | sort | tail -1)
    [ -n "$latest_run" ] || { echo "N/A"; return; }
    local params_file
    params_file=$(find "$latest_run" -maxdepth 1 -name "*optimal_parameters.txt" | head -1)
    [ -n "$params_file" ] || { echo "N/A"; return; }
    local lik
    lik=$(grep "^likelihood" "$params_file" | awk '{print $2}')
    echo "${lik:-N/A}"
}

if [ ${#RERUN_POPS[@]} -gt 0 ]; then
    echo ""
    echo "══════════════════════════════════════════"
    echo " Results summary"
    echo "══════════════════════════════════════════"
    printf "%-8s  %-35s  %s\n" "Pop" "Model (output dir)" "Likelihood"
    printf "%-8s  %-35s  %s\n" "--------" "-----------------------------------" "----------"
    for pop in "${RERUN_POPS[@]}"; do
        for out_dir in "$SCRIPT_DIR/$pop"/output_*/; do
            [ -d "$out_dir" ] || continue
            model_name=$(basename "$out_dir")
            lik=$(get_latest_likelihood "$out_dir")
            printf "%-8s  %-35s  %s\n" "$pop" "$model_name" "$lik"
        done
    done
fi
