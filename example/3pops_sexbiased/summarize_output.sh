#!/usr/bin/env bash
# Summarize the latest tracts output for all populations.
# This driver file was mostly AI-generated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Output flags ──────────────────────────────────────────────────────────────
PRINT_WARNINGS=1              # print log warnings and likelihood regression warnings
PRINT_ANCESTRY_PROPORTIONS=1  # print ancestry proportions table for each model
PRINT_PARAMETERS=1


# ── Likelihood tolerance: warn only if latest run is worse than best by at least this amount ──
LIKELIHOOD_TOLERANCE=0.1

# ── Date filter: only show results whose latest run is more recent than this date ──
# Pass via -s/--since YYYY-MM-DD (or YYYYMMDD); omit to be prompted, blank input shows all.
SINCE_DATE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--since)
            SINCE_DATE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [-s|--since YYYY-MM-DD]" >&2
            exit 1
            ;;
    esac
done

if [ -z "$SINCE_DATE" ]; then
    read -r -p "Only show results more recent than (YYYY-MM-DD, blank for all): " SINCE_DATE
fi

# Normalize to a run-timestamp-comparable string (YYYYMMDD_HHMMSS, start of day).
SINCE_TS=""
if [ -n "$SINCE_DATE" ]; then
    SINCE_TS="$(echo "$SINCE_DATE" | tr -d '-')_000000"
fi

# ── Summary functions ─────────────────────────────────────────────────────────

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

# ── Summary ───────────────────────────────────────────────────────────────────

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
        run_ts=$(basename "${latest_run%/}")
        if [ -n "$SINCE_TS" ] && [[ ! "$run_ts" > "$SINCE_TS" ]]; then
            continue
        fi
        n_ind=$(get_num_individuals "$latest_run")
        lik=$(get_latest_likelihood "$latest_run")
        printf "%-8s  %-35s  %-6s  %s\n" "$pop" "$model_name" "$n_ind" "$lik"
        if [ "$PRINT_WARNINGS" -eq 1 ] && [[ "$lik" =~ ^-?[0-9] ]]; then
            best_info=$(get_best_run_info "$out_dir")
            if [ -n "$best_info" ]; then
                best_lik=$(echo "$best_info" | awk '{print $1}')
                best_ts=$(echo "$best_info" | awk '{print $2}')
                if awk "BEGIN { exit !(($best_lik) - ($lik) >= $LIKELIHOOD_TOLERANCE) }"; then
                    latest_yaml=$(get_latest_yaml "$latest_run")
                    reps=""
                    [ -n "$latest_yaml" ] && reps=$(query_yaml "$latest_yaml" "optim.repetitions")
                    printf "  *** WARNING: better likelihood %s found at earlier run %s (latest run: %s repetitions) ***\n" \
                        "$best_lik" "$best_ts" "${reps:-N/A}"
                fi
            fi
            log_file=$(find "$latest_run" -maxdepth 1 -name "*.log" | head -1)
            if [ -n "$log_file" ]; then
                grep "tracts.driver_utils - WARNING" "$log_file" | while IFS= read -r line; do
                    printf "  LOG WARNING: %s\n" "$line"
                done
            fi
        fi
        if [ "$PRINT_ANCESTRY_PROPORTIONS" -eq 1 ]; then
            anc_file=$(find "$latest_run" -maxdepth 1 -name "*ancestry_proportions.txt" | head -1)
            [ -n "$anc_file" ] && cat "$anc_file" || echo "  (no ancestry_proportions.txt)"
        fi
        
        if [ "$PRINT_PARAMETERS" -eq 1 ]; then
            param_file=$(find "$latest_run" -maxdepth 1 -name "*optimal_parameters.txt" | head -1)
            [ -n "$param_file" ] && cat "$param_file" || echo "  (no parameters.txt)"
        fi


    done
done
