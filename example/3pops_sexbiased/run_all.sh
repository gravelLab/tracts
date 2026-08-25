#!/usr/bin/env bash
# Run tracts inference for all populations.
# This driver file was mostly AI-generated.
# Set RUN_<POP>=1 to rerun a population, 0 to skip.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Per-population run flags ───────────────────────────────────────────────────
RUN_ACB=1
RUN_ASW=1
RUN_CLM=1
RUN_MXL=1
RUN_PEL=1
RUN_PUR=1

# ── Models per population (files must be <POP>/<POP>_<model>.py) ──────────────
ACB_MODELS=(ppp ppx_xxp_pxx)
ASW_MODELS=(ppp ppx_xxp_ppx ppx_xxp_pxx)
CLM_MODELS=(ppp ccc ccp)
MXL_MODELS=(ppp ccp)
PEL_MODELS=(ppp ccc)
PUR_MODELS=(ppp cpc cpp)

MAX_WORKERS=6

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
            (
                cd "$SCRIPT_DIR/$pop" || exit 1
                python "$script" > /dev/null 2>&1
                status=$?
                if [ "$status" -eq 0 ]; then
                    echo "  ✓ done: ${pop}_${model}"
                else
                    echo "  ✗ failed: ${pop}_${model} (exit $status)"
                fi
            ) &
            _PIDS+=($!)
        else
            echo "  ✗ not found: ${pop}_${model}.py"
        fi
    done
}

[ "$RUN_ACB" -eq 1 ] && run_pop ACB "${ACB_MODELS[@]}"
[ "$RUN_ASW" -eq 1 ] && run_pop ASW "${ASW_MODELS[@]}"
[ "$RUN_CLM" -eq 1 ] && run_pop CLM "${CLM_MODELS[@]}"
[ "$RUN_MXL" -eq 1 ] && run_pop MXL "${MXL_MODELS[@]}"
[ "$RUN_PEL" -eq 1 ] && run_pop PEL "${PEL_MODELS[@]}"
[ "$RUN_PUR" -eq 1 ] && run_pop PUR "${PUR_MODELS[@]}"

wait
echo "All runs complete."
