#!/usr/bin/env python3
"""
Compares the latest fit_M.sh (M-model) run against the latest
fit_DC_hotstart.sh (DC, hot-started) run, for every population/model pair
listed in fit_M.sh.

For each pair, prints the M run's final parameters side by side with the
hotstart run's starting and final parameters, and validates that the
hotstart's starting point (as currently recorded in the <pop>_<model>_DC_hotstart.yaml
driver file, written by create_hotstart_file.py) exactly matches the M run's
final (optimal) parameters -- i.e. that the hotstart file has not gone stale
relative to the latest completed M run.

Reuses create_hotstart_file.py's own parsing/lookup logic (parse_run_all_models,
extract_driver_filename, parse_optimal_parameters, find_latest_optimal_parameters,
hotstart_output_directory) so the comparison is checked against the exact same
rules used to generate the hotstart files in the first place.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import ruamel.yaml
from create_hotstart_file import (
    SUFFIX,
    RUN_ALL_SH,
    parse_run_all_models,
    extract_driver_filename,
    parse_optimal_parameters,
    find_latest_optimal_parameters,
    hotstart_output_directory,
)

yaml = ruamel.yaml.YAML(typ="rt")

FLOAT_TOLERANCE = 1e-9


def read_likelihood(optimal_parameters_path: Path) -> float | None:
    for line in optimal_parameters_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "likelihood":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def resolve_base_output_dir(yaml_path: Path, spec) -> Path:
    output_directory = str(spec["output"]["output_directory"])
    base_output_dir_str = output_directory[: output_directory.index("{date}")].rstrip("/")
    return (yaml_path.parent / base_output_dir_str).resolve()


def compare_pop_model(pop_dir: Path, pop: str, model: str) -> None:
    header = f"=== {pop} / {model} ==="
    py_path = pop_dir / f"{pop}_{model}.py"
    if not py_path.is_file():
        print(f"{header}\n  ! driver script not found: {py_path}\n")
        return

    yaml_filename = extract_driver_filename(py_path)
    yaml_path = pop_dir / yaml_filename
    if not yaml_path.is_file():
        print(f"{header}\n  ! yaml not found: {yaml_path}\n")
        return

    with open(yaml_path) as f:
        spec = yaml.load(f)
    required_keys = set(spec["start_params"].keys())
    base_output_dir = resolve_base_output_dir(yaml_path, spec)

    # ----- Latest M run -----
    m_optimal_path = find_latest_optimal_parameters(base_output_dir, required_keys)
    if m_optimal_path is None:
        print(f"{header}\n  ! no completed+compatible M-model run found (looked in {base_output_dir})\n")
        return
    m_params = parse_optimal_parameters(m_optimal_path)
    m_likelihood = read_likelihood(m_optimal_path)
    m_run_dir = m_optimal_path.parent.parent

    # ----- Hotstart driver file's currently-recorded starting point -----
    hotstart_yaml_path = yaml_path.parent / f"{yaml_path.stem}_{SUFFIX}.yaml"
    if not hotstart_yaml_path.is_file():
        print(f"{header}\n  ! hotstart yaml not found: {hotstart_yaml_path} (run create_hotstart_file.py first)\n")
        return
    with open(hotstart_yaml_path) as f:
        hotstart_spec = yaml.load(f)
    hotstart_start_params = {k: float(v) for k, v in hotstart_spec["start_params"].items()}

    # ----- Latest hotstart (DC) run's own result -----
    hotstart_output_dir_template = hotstart_output_directory(str(spec["output"]["output_directory"]))
    hotstart_base_output_dir = (
        yaml_path.parent / hotstart_output_dir_template[: hotstart_output_dir_template.index("{date}")].rstrip("/")
    ).resolve()
    hotstart_optimal_path = find_latest_optimal_parameters(hotstart_base_output_dir, required_keys)

    print(f"{header}")
    print(f"  M run:        {m_run_dir.relative_to(SCRIPT_DIR)}  (likelihood={m_likelihood})")
    if hotstart_optimal_path is not None:
        hotstart_run_dir = hotstart_optimal_path.parent.parent
        hotstart_params = parse_optimal_parameters(hotstart_optimal_path)
        hotstart_likelihood = read_likelihood(hotstart_optimal_path)
        print(f"  Hotstart run: {hotstart_run_dir.relative_to(SCRIPT_DIR)}  (likelihood={hotstart_likelihood})")
    else:
        hotstart_params = None
        print(f"  Hotstart run: ! no completed+compatible DC-hotstart run found (looked in {hotstart_base_output_dir})")

    # ----- Validate: hotstart's recorded start point == M run's end point -----
    mismatches = []
    for key in sorted(required_keys):
        m_value = m_params.get(key)
        start_value = hotstart_start_params.get(key)
        if m_value is None or start_value is None:
            mismatches.append((key, m_value, start_value, "missing"))
        elif abs(start_value - m_value) > FLOAT_TOLERANCE * max(1.0, abs(m_value)):
            mismatches.append((key, m_value, start_value, "differs"))

    if not mismatches:
        print(f"  Start-point check: PASS (hotstart's start_params exactly match the M run's optimal parameters)")
    else:
        print(f"  Start-point check: FAIL -- hotstart's start_params do NOT match the M run's optimal parameters"
              f" (hotstart may be stale; rerun create_hotstart_file.py):")
        for key, m_value, start_value, reason in mismatches:
            print(f"    {key}: M(end)={m_value!r}  hotstart(start)={start_value!r}  [{reason}]")

    # ----- Side-by-side parameter table -----
    all_keys = sorted(set(m_params) | set(hotstart_start_params) | (set(hotstart_params) if hotstart_params else set()))
    name_w = max(len(k) for k in all_keys) if all_keys else 10
    print(f"\n  {'parameter':<{name_w}}  {'M (end)':>14}  {'hotstart (start)':>17}  {'hotstart (end, DC)':>19}")
    print(f"  {'-'*name_w}  {'-'*14}  {'-'*17}  {'-'*19}")
    for key in all_keys:
        m_value = m_params.get(key)
        start_value = hotstart_start_params.get(key)
        end_value = hotstart_params.get(key) if hotstart_params else None
        m_str = f"{m_value:.6g}" if m_value is not None else "--"
        start_str = f"{start_value:.6g}" if start_value is not None else "--"
        end_str = f"{end_value:.6g}" if end_value is not None else "--"
        print(f"  {key:<{name_w}}  {m_str:>14}  {start_str:>17}  {end_str:>19}")
    print()


def main() -> int:
    pop_models = parse_run_all_models(RUN_ALL_SH)
    for pop, models in pop_models.items():
        pop_dir = SCRIPT_DIR / pop
        for model in models:
            compare_pop_model(pop_dir, pop, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
