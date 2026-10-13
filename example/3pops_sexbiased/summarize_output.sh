#!/usr/bin/env bash
# Summarize the latest tracts output for all populations.
# This driver file was mostly AI-generated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Output flags ──────────────────────────────────────────────────────────────
PRINT_WARNINGS=1              # print log warnings and likelihood regression warnings

LATEX_FORMAT=1                # 1=print the paper's LaTeX tables per population, 0=print raw files per run
LATEX_SIG_FIGS=3              # significant figures for parameter values in the LaTeX tables
PRINT_ANCESTRY_PROPORTIONS=1  # ancestry proportions table (single-pulse and second model)
PRINT_PARAMETERS=1            # best-fit parameters table of the second model
PRINT_SINGLE_PULSE_PARAMETERS=1  # best-fit parameters table of the single-pulse model (LaTeX only)

# ── Models per population for the LaTeX tables: single-pulse model first, then the second
# model (same convention as run_all.sh; files must be <POP>/<POP>_<model>.yaml) ──
ACB_MODELS=(ppp ppx_xxp_pxx)
ASW_MODELS=(ppp ppx_xxp_pxx)
CLM_MODELS=(ppp ccc)
MXL_MODELS=(ppp ccp)
PEL_MODELS=(ppp ccc)
PUR_MODELS=(ppp cpc)

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

# get_latest_run <out_dir> — latest run folder, by its YYYYMMDD_HHMMSS timestamp name.
get_latest_run() {
    ls -d "$1"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]/ 2>/dev/null | sort | tail -1
}

# Prints "<best_likelihood> <timestamp>" across all runs in out_dir.
# Returns empty if no runs have a likelihood line.
get_best_run_info() {
    local out_dir="$1"
    local best_lik="" best_ts=""
    for run_dir in $(ls -d "$out_dir"/[0-9]*/ 2>/dev/null | sort); do
        local params_file ts lik
        params_file=$(find "$run_dir" -maxdepth 2 -name "*optimal_parameters.txt" | head -1)
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
    params_file=$(find "$latest_run" -maxdepth 2 -name "*optimal_parameters.txt" | head -1)
    [ -n "$params_file" ] || { echo "ERROR: optimal_parameters.txt missing in $(basename "$latest_run")"; return; }
    local lik
    lik=$(grep "^likelihood" "$params_file" | awk '{print $2}')
    [ -n "$lik" ] || { echo "ERROR: no likelihood line in $(basename "$params_file")"; return; }
    echo "$lik"
}

get_latest_yaml() {
    local yaml_file
    yaml_file=$(find "$1" -maxdepth 2 -name "*.yaml" | head -1)
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

# is_complete_run <run_dir> — true if the run contains an optimal_parameters.txt file
# (i.e. it is neither still running nor crashed).
is_complete_run() {
    [ -n "$(find "$1" -maxdepth 2 -name "*optimal_parameters.txt" | head -1)" ]
}

# get_model_output_dir <pop> <model> — output directory of <pop>_<model>.yaml, with the
# "{date}" part removed (e.g. "$SCRIPT_DIR/ACB/output_ppp_popphased").
get_model_output_dir() {
    local yaml_file="$SCRIPT_DIR/$1/$1_$2.yaml"
    [ -f "$yaml_file" ] || return
    local out
    out=$(query_yaml "$yaml_file" "output.output_directory")
    out="${out%%\{date\}*}"
    out="${out#./}"
    out="${out%/}"
    echo "$SCRIPT_DIR/$1/$out"
}

# print_paper_tables <pop> <single_run_dir> <model> <model_run_dir> <sig_figs>
#                    <print_proportions> <print_params> <print_single_params>
# Prints the LaTeX tables of the Results section of the tracts paper for one population:
# ancestry proportions (single-pulse and second model), best-fit parameters of the second
# model and, optionally, best-fit parameters of the single-pulse model. The tables
# reproduce the paper's layout and captions, so they can be pasted as is.
print_paper_tables() {
    python3 - "$@" <<'EOF'
import sys
import re
from decimal import Decimal
from pathlib import Path

pop, single_run, model, model_run = sys.argv[1:5]
sig_figs = int(sys.argv[5])
print_props, print_params, print_single = (a == "1" for a in sys.argv[6:9])

ANCESTRIES = ["EUR", "AMR", "AFR"]
ALIASES = {"NAT": "AMR"}

# ── Per-population settings (labels and captions as in the paper) ────────────
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

# ── Reading tracts output ─────────────────────────────────────────────────────
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

def read_proportions(run_dir):
    path = find_file(run_dir, "ancestry_proportions.txt")
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

def fmt(value):
    """Round to sig_figs significant figures, never in scientific notation."""
    return format(Decimal(f"{value:.{sig_figs}g}"), "f")

def int_digits(value):
    return len(str(int(abs(float(fmt(value))))))

# ── Table writers ─────────────────────────────────────────────────────────────
def print_proportions_table(single, other):
    observed = single["Observed (autosomes)"], single["Observed (X)"]
    lines = [("Observed", observed[0], observed[1]),
             ("Predicted (single pulse)", single["Predicted (autosomes)"], single["Predicted (X)"]),
             (PROP_ROW_LABEL[model], other["Predicted (autosomes)"], other["Predicted (X)"])]
    width = max(len(label) for label, _, _ in lines)
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\sisetup{round-mode=places, round-precision=4}")
    print(r"\begin{tabular}{l")
    print(r"                S[table-format=1.4] S[table-format=1.4] S[table-format=1.4]")
    print(r"                S[table-format=1.4] S[table-format=1.4] S[table-format=1.4]}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Autosomes} & \multicolumn{3}{c}{X chromosome} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r" & {EUR} & {AMR} & {AFR} & {EUR} & {AMR} & {AFR} \\")
    print(r"\midrule")
    for label, aut, x in lines:
        values = [aut[a] for a in ANCESTRIES] + [x[a] for a in ANCESTRIES]
        print(f"{label:<{width}} & " + " & ".join(values) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\\caption{{Observed and predicted ancestry proportions for the {pop} population, on the autosomes and the X chromosome.}}")
    print(f"\\label{{tab:{pop}prop}}")
    print(r"\end{table}")

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

def print_params_table(colspec, group_rows, header, values, caption, label):
    colspec = fit_colspec(colspec, values)
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{" + colspec + "}")
    print(r"\toprule")
    for row in group_rows:
        print(row)
    print(header)
    print(r"\midrule")
    print(" " + " & ".join(values) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\\caption{{Best-fit parameters for the {pop} population, {caption}}}")
    print(f"\\label{{{label}}}")
    print(r"\end{table}")

def print_single_pulse_table(p):
    lik = p["likelihood"]
    colspec = ("S[table-format=1.4] S[table-format=-1.3]\n"
               "                S[table-format=1.4] S[table-format=-1.3]\n"
               "                S[table-format=1.4] S[table-format=-1.3]\n"
               f"                S[table-format={int_digits(p['t'])}.2]\n"
               f"                S[table-format=-{int_digits(lik)}.0]")
    groups = [r" \multicolumn{2}{c}{EUR} & \multicolumn{2}{c}{AMR} & \multicolumn{2}{c}{AFR} & & \\",
              r"\cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6}"]
    header = (" " + rate_header("EUR") + "\n & " + rate_header("AMR") + "\n & " + rate_header("AFR")
              + "\n & {$t$}\n & {$\\mathcal{L}_{\\mathcal{D}}(\\bm{\\theta})$} \\\\")
    values = []
    for anc in ANCESTRIES:
        values += [fmt(rate(p, anc)), fmt(sex_bias(p, anc))]
    values += [fmt(p["t"]), fmt(lik)]
    caption = ("inferred using a single-pulse model in which the three source populations "
               "contribute at a single admixture event $t$ generations ago.")
    print_params_table(colspec, groups, header, values, caption, SINGLE_LABEL.get(pop, f"tab:{pop}params_single"))

def print_model_table(p):
    lik = fmt(p["likelihood"])
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
        values = [fmt(rate(p, "EUR")), fmt(sex_bias(p, "EUR")), fmt(rate(p, "AMR")), fmt(sex_bias(p, "AMR")), fmt(p["t1"]),
                  fmt(rate(p, "AFR")), fmt(sex_bias(p, "AFR")), fmt(p["t2"]),
                  fmt(p["REUR2"]), fmt(p["REUR2_sex_bias"]), fmt(p["t3"]), lik]
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
            values += [fmt(rate(p, anc)), fmt(sex_bias(p, anc))]
        values += [fmt(p["t1"]), fmt(p["t2"]), lik]
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
            values += [fmt(rate(p, anc)), fmt(sex_bias(p, anc))]
        values += [fmt(p["t1"]), fmt(p["t2"]), lik]
    else:
        print(f"% No paper table template for model {model}.")
        return
    print_params_table(colspec, groups, header, values, PARAMS_CAPTION[pop], PARAMS_LABEL.get(pop, f"tab:{pop}params"))

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"% ===== {pop}: ppp from {single_run}, {model} from {model_run} =====")
single_props, model_props = read_proportions(single_run), read_proportions(model_run)
if print_props:
    if single_props is None or model_props is None:
        print("% (missing ancestry_proportions.txt: cannot build the ancestry proportions table)")
    else:
        if any(
            [float(single_props[row][anc]) for anc in ANCESTRIES] !=
            [float(model_props[row][anc]) for anc in ANCESTRIES]
            for row in ("Observed (autosomes)", "Observed (X)")
        ):
            warning = (f"WARNING: {pop}: observed proportions differ between the ppp and {model} runs "
                       "(different samples?); ancestry proportions table skipped.")
            print(f"% {warning}")
            print(f"*** {warning}", file=sys.stderr)
        else:
            print_proportions_table(single_props, model_props)
            print()
if print_single:
    print_single_pulse_table(read_params(single_run))
    print()
if print_params:
    print_model_table(read_params(model_run))
    print()
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
        if [ "$LATEX_FORMAT" -eq 0 ]; then
            if [ "$PRINT_ANCESTRY_PROPORTIONS" -eq 1 ]; then
                anc_file=$(find "$latest_run" -maxdepth 2 -name "*ancestry_proportions.txt" | head -1)
                if [ -z "$anc_file" ]; then echo "  (no ancestry_proportions.txt)"; else cat "$anc_file"; fi
            fi
            if [ "$PRINT_PARAMETERS" -eq 1 ]; then
                param_file=$(find "$latest_run" -maxdepth 2 -name "*optimal_parameters.txt" | head -1)
                if [ -z "$param_file" ]; then echo "  (no optimal_parameters.txt)"; else cat "$param_file"; fi
            fi
        fi
    done
done

# ── LaTeX tables for the paper ────────────────────────────────────────────────
# One block per population, built from the latest run of each model in <POP>_MODELS
# (independent of the date filter above); skipped if a latest run is not complete. Paste each block as is.
if [ "$LATEX_FORMAT" -eq 1 ]; then
    echo ""
    echo "══════════════════════════════════════════"
    echo " LaTeX tables (paper format)"
    echo "══════════════════════════════════════════"
    for pop in ACB ASW CLM MXL PEL PUR; do
        models_var="${pop}_MODELS[@]"
        models=("${!models_var}")
        single_model="${models[0]}"
        model="${models[1]}"
        single_run=$(get_latest_run "$(get_model_output_dir "$pop" "$single_model")")
        model_run=$(get_latest_run "$(get_model_output_dir "$pop" "$model")")
        echo ""
        skip=""
        for m in "$single_model" "$model"; do
            run=$single_run; [ "$m" = "$model" ] && run=$model_run
            if [ -z "$run" ]; then
                skip="$skip ${m}: no run found;"
            elif ! is_complete_run "$run"; then
                skip="$skip ${m}: latest run $(basename "${run%/}") is not complete (still running or crashed);"
            fi
        done
        if [ -n "$skip" ]; then
            echo "% ===== $pop: skipped --$skip ====="
            echo "*** $pop LaTeX tables skipped --$skip" >&2
            continue
        fi
        print_paper_tables "$pop" "${single_run%/}" "$model" "${model_run%/}" "$LATEX_SIG_FIGS" \
            "$PRINT_ANCESTRY_PROPORTIONS" "$PRINT_PARAMETERS" "$PRINT_SINGLE_PULSE_PARAMETERS"
    done
fi
