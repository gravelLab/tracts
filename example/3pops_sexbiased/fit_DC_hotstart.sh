#!/usr/bin/env bash
# Run tracts DC-model inference, hot-started from each population/model's latest
# completed M-model optimization run (see create_hotstart_file.py).
# This driver file was mostly AI-generated.
# Set RUN_<POP>=1 to rerun a population, 0 to skip.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep macOS from idle-sleeping while this script runs (no-op where caffeinate doesn't exist).
command -v caffeinate >/dev/null 2>&1 && ( caffeinate -i -w $$ & )  # subshell: detached, so the final `wait` does not wait for it

SUFFIX="_DC_hotstart"

# Regenerate the hotstart driver files from each population's latest completed
# M-model optimization run before launching anything, so start params are fresh.
echo "Regenerating ${SUFFIX} driver files..."
if ! python "$SCRIPT_DIR/create_hotstart_file.py"; then
    echo "create_hotstart_file.py failed; aborting." >&2
    exit 1
fi

# Per-run timing: one directory per batch, holding runs.csv (start/end Unix time and exit
# status per run) and one /usr/bin/time -l report per run (CPU time, peak memory).
TIMING_DIR="$SCRIPT_DIR/timing/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TIMING_DIR"
echo "pop,model,start,end,status" > "$TIMING_DIR/runs.csv"

# ── Per-population run flags ───────────────────────────────────────────────────
RUN_ACB=1
RUN_ASW=1
RUN_CLM=1
RUN_MXL=1
RUN_PEL=1
RUN_PUR=1

# ── Models per population (files must be <POP>/<POP>_<model>${SUFFIX}.py) ─────
ACB_MODELS=(ppp ppx_xxp_pxx)
ASW_MODELS=(ppp ppx_xxp_pxx)
CLM_MODELS=(ppp ccc)
MXL_MODELS=(ppp ccp)
PEL_MODELS=(ppp pcc)
PUR_MODELS=(ppp cpc)

MAX_WORKERS=5

# Failed or missing runs are recorded here, so that the script can exit with a nonzero status.
FAILED_RUNS_FILE="$(mktemp)"
trap 'rm -f "$FAILED_RUNS_FILE"' EXIT

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
        local script="$SCRIPT_DIR/$pop/${pop}_${model}${SUFFIX}.py"
        if [ -f "$script" ]; then
            echo "  → ${model}${SUFFIX}"
            _wait_for_slot
            (
                cd "$SCRIPT_DIR/$pop" || exit 1
                t0=$(date +%s)
                /usr/bin/time -l -o "$TIMING_DIR/${pop}_${model}${SUFFIX}.time" \
                    python "$script" > /dev/null 2>&1
                status=$?
                echo "${pop},${model}${SUFFIX},${t0},$(date +%s),${status}" >> "$TIMING_DIR/runs.csv"
                if [ "$status" -eq 0 ]; then
                    echo "  ✓ done: ${pop}_${model}${SUFFIX}"
                else
                    echo "  ✗ failed: ${pop}_${model}${SUFFIX} (exit $status)"
                    echo "${pop}_${model}${SUFFIX} (exit $status)" >> "$FAILED_RUNS_FILE"
                fi
                exit "$status"
            ) &
            _PIDS+=($!)
        else
            echo "  ✗ not found: ${pop}_${model}${SUFFIX}.py"
            echo "${pop}_${model}${SUFFIX} (driver script not found; run create_hotstart_file.py first)" >> "$FAILED_RUNS_FILE"
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
if [ -s "$FAILED_RUNS_FILE" ]; then
    echo "Some runs failed ($(wc -l < "$FAILED_RUNS_FILE" | tr -d ' ')):"
    sed 's/^/  - /' "$FAILED_RUNS_FILE"
    exit 1
fi
echo "All runs complete."
