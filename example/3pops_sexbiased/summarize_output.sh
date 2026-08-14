#!/usr/bin/env bash
# Summarize the latest tracts output for all populations.
# This driver file was mostly AI-generated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Output flags ──────────────────────────────────────────────────────────────
PRINT_WARNINGS=1              # print log warnings and likelihood regression warnings

LATEX_FORMAT=1
LATEX_SIG_FIGS=3              # significant figures for parameter values in the LaTeX table
PRINT_ANCESTRY_PROPORTIONS=1  # 0=off, 1=print ancestry proportions table as-is
PRINT_PARAMETERS=1


# ── Likelihood tolerance: warn only if latest run is worse than best by at least this amount ──
LIKELIHOOD_TOLERANCE=0.1

# ── Date filter: only show results whose latest run is more recent than this date ──
# Pass via -s/--since YYYY-MM-DD (or YYYYMMDD); omit to be prompted.
# Default (blank input) is today; type 'all' to show everything.
SINCE_DATE=""
TODAY="$(date +%Y-%m-%d)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--since)
            SINCE_DATE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [-s|--since YYYY-MM-DD|all]" >&2
            exit 1
            ;;
    esac
done

if [ -z "$SINCE_DATE" ]; then
    read -r -p "Only show results more recent than (YYYY-MM-DD, default ${TODAY}, 'all' for all): " SINCE_DATE
fi
[ -z "$SINCE_DATE" ] && SINCE_DATE="$TODAY"

# Normalize to a run-timestamp-comparable string (YYYYMMDD_HHMMSS, start of day).
SINCE_TS=""
if [ "$SINCE_DATE" != "all" ]; then
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

# print_ancestry_table_latex <ancestry_proportions.txt> <corner_label> — renders the
# fixed-width ancestry proportions table (see _save_ancestry_proportions_table in
# driver_utils.py) as a basic LaTeX tabular, with <corner_label> in the top-left cell.
print_ancestry_table_latex() {
    python3 - "$1" "$2" <<'EOF'
import sys

with open(sys.argv[1]) as f:
    lines = [line.rstrip("\n") for line in f if line.strip()]
corner_label = sys.argv[2]

# Drop the "----" separator lines.
content = [line for line in lines if set(line.strip()) - {"-"}]
if not content:
    sys.exit(0)

header = content[0].split()
n_cols = len(header)

rows = []
for line in content[1:]:
    tokens = line.split()
    label, values = " ".join(tokens[:-n_cols]), tokens[-n_cols:]
    rows.append((label, values))

print(r"\begin{tabular}{l" + "r" * n_cols + "}")
print(r"\hline")
print(" & ".join([corner_label] + header) + r" \\")
print(r"\hline")
for label, values in rows:
    print(" & ".join([label] + values) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
EOF
}

# print_params_table_latex <optimal_parameters.txt> <corner_label> <sig_figs> — renders the
# "parameter\tvalue" file written in _save_optimization_results (driver_utils.py) as a
# basic LaTeX tabular, with one column per parameter (plus likelihood) and <corner_label>
# in the top-left cell. Values are rounded to <sig_figs> significant figures.
print_params_table_latex() {
    python3 - "$1" "$2" "$3" <<'EOF'
import sys

with open(sys.argv[1]) as f:
    lines = [line.rstrip("\n") for line in f if line.strip()]
corner_label = sys.argv[2]
sig_figs = int(sys.argv[3])

# First line is the "parameter value" header; skip it.
rows = []
likelihood = None
for line in lines[1:]:
    tokens = line.split()
    if not tokens:
        continue
    name, value = tokens[0], tokens[-1]
    if name == "likelihood":
        likelihood = value
    else:
        rows.append((name, value))

if not rows:
    sys.exit(0)

def round_sig(value):
    return f"{float(value):.{sig_figs}g}"

names = [name for name, _ in rows]
values = [round_sig(value) for _, value in rows]
if likelihood is not None:
    names.append("likelihood")
    values.append(round_sig(likelihood))

print(r"\begin{tabular}{l" + "r" * len(names) + "}")
print(r"\hline")
print(" & ".join([corner_label] + names) + r" \\")
print(r"\hline")
print(" & ".join([""] + values) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
EOF
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
            if [ -z "$anc_file" ]; then
                echo "  (no ancestry_proportions.txt)"
            elif [ "$LATEX_FORMAT" -eq 1 ]; then
                print_ancestry_table_latex "$anc_file" "${pop}_${model_name}"
            else
                cat "$anc_file"
            fi
        fi
        
        if [ "$PRINT_PARAMETERS" -eq 1 ]; then
            param_file=$(find "$latest_run" -maxdepth 1 -name "*optimal_parameters.txt" | head -1)
            if [ -z "$param_file" ]; then
                echo "  (no parameters.txt)"
            elif [ "$LATEX_FORMAT" -eq 1 ]; then
                print_params_table_latex "$param_file" "${pop}_${model_name}" "$LATEX_SIG_FIGS"
            else
                cat "$param_file"
            fi
        fi


    done
done
