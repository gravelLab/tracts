"""
Generate DC_hotstart driver files
==================================

For every population/model pair listed in ``run_all.sh``, this script creates a
"DC_hotstart" variant of the driver YAML (and matching driver .py script):

* ``ad_model_autosomes`` is set to ``DC`` (instead of ``M``).
* ``start_params`` is replaced with the optimal parameters found by the most
  recently *completed* optimization run of the original (M-model) driver file,
  so the DC optimization starts from a known-good point instead of sampling
  random starts.
* ``optim.repetitions`` (the number of random starts) is set to 1, since a
  single hot-started run is all that's needed.
* ``output.output_directory`` gets a ``_DC_hotstart`` suffix so results don't
  collide with existing runs.

Example: ``ASW_ppp.yaml`` -> ``ASW_ppp_DC_hotstart.yaml`` (and
``ASW_ppp.py`` -> ``ASW_ppp_DC_hotstart.py``).
"""

import re
from pathlib import Path

import ruamel.yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ALL_SH = SCRIPT_DIR / "run_all.sh"
SUFFIX = "DC_hotstart"

yaml = ruamel.yaml.YAML(typ="rt")
yaml.preserve_quotes = True
yaml.width = 10**6  # don't rewrap existing long lines (e.g. individual_names)


def parse_run_all_models(run_all_sh_path: Path) -> dict[str, list[str]]:
    """Reads e.g. `ACB_MODELS=(ppp ppx_xxp_pxx)` lines into {"ACB": ["ppp", "ppx_xxp_pxx"]}."""
    pattern = re.compile(r"^([A-Za-z0-9]+)_MODELS=\(([^)]*)\)\s*$")
    pop_models = {}
    for line in run_all_sh_path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            pop, models_str = match.groups()
            pop_models[pop] = models_str.split()
    return pop_models


def extract_driver_filename(py_path: Path) -> str:
    text = py_path.read_text()
    match = re.search(r'driver_filename\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise ValueError(f"Could not find driver_filename assignment in {py_path}")
    return match.group(1)


def read_likelihood(optimal_parameters_path: Path) -> float | None:
    """Reads the `likelihood` value out of a `*_optimal_parameters.txt` file (skipped by
    `parse_optimal_parameters`, which only returns the model parameters themselves)."""
    for line in optimal_parameters_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "likelihood":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


_REMAINDER_PATTERN = re.compile(r"^\w+_([A-Za-z0-9]+)_(rate|sex_bias)$")


def parse_optimal_parameters(path: Path) -> dict[str, float]:
    """Parses a `*_optimal_parameters.txt` file into {parameter: value}.

    The demographic model's implicit population (the one whose rate/sex_bias is
    derived rather than independently optimized, see `implicit_population` in the
    driver YAML) is reported under a `<destpop>_<ABBR>_rate` / `_sex_bias` key
    rather than as `R<ABBR>` / `R<ABBR>_sex_bias`. Since which population is
    implicit can change between runs (e.g. `implicit_population` gets edited),
    these derived values are additionally exposed under the `R<ABBR>[...]` alias
    they'd have if that population were instead a free parameter, so hot-starting
    still works when the current driver YAML's implicit population differs from
    the one used in the run being read.
    """
    params = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, value = parts
        if key in ("parameter", "likelihood"):
            continue
        try:
            value = float(value)
        except ValueError:
            continue
        params[key] = value
        match = _REMAINDER_PATTERN.match(key)
        if match:
            abbr, kind = match.groups()
            alias = f"R{abbr}" if kind == "rate" else f"R{abbr}_sex_bias"
            params.setdefault(alias, value)
    return params


def find_latest_optimal_parameters(output_base_dir: Path, required_keys: set[str]) -> Path | None:
    """Among timestamped run subdirectories of output_base_dir, returns the
    `*_optimal_parameters.txt` file of the most recent one that both completed
    (i.e. actually produced one) and provides values for every one of
    `required_keys` (directly or via the implicit-population alias described in
    `parse_optimal_parameters`), since older/newer runs may have used a
    different free-parameter set. Returns None if no run qualifies."""
    if not output_base_dir.is_dir():
        return None
    run_dirs = sorted((d for d in output_base_dir.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for run_dir in run_dirs:
        matches = sorted(run_dir.rglob("*optimal_parameters.txt"))
        if not matches:
            continue
        if required_keys <= parse_optimal_parameters(matches[0]).keys():
            return matches[0]
    return None


def hotstart_output_directory(output_directory: str) -> str:
    idx = output_directory.index("{date}")
    before = output_directory[:idx].rstrip("/")
    after = output_directory[idx + len("{date}") :]
    return f"{before}_{SUFFIX}/{{date}}{after}"


def build_hotstart_yaml(pop_dir: Path, yaml_filename: str) -> Path | None:
    yaml_path = pop_dir / yaml_filename
    if not yaml_path.is_file():
        print(f"  ! yaml not found: {yaml_path}")
        return None

    with open(yaml_path) as f:
        spec = yaml.load(f)

    output_directory = str(spec["output"]["output_directory"])
    base_output_dir_str = output_directory[: output_directory.index("{date}")].rstrip("/")
    base_output_dir = (yaml_path.parent / base_output_dir_str).resolve()

    start_params = spec["start_params"]
    required_keys = set(start_params.keys())

    optimal_params_path = find_latest_optimal_parameters(base_output_dir, required_keys)
    if optimal_params_path is None:
        print(f"  ! no completed+compatible optimization run found for {yaml_path.name} (looked in {base_output_dir})")
        return None

    optimal_params = parse_optimal_parameters(optimal_params_path)
    for key in required_keys:
        start_params[key] = optimal_params[key]

    # Purely informational provenance record (see ParentRunConfig in tracts.driver_utils):
    # ignored by the inference pipeline itself, kept for later analysis/tooling -- e.g. to
    # detect a hotstart driver file that has gone stale relative to a newer parent run.
    spec["parent_run"] = {
        "optimal_parameters_file": str(optimal_params_path.relative_to(SCRIPT_DIR)),
        "likelihood": read_likelihood(optimal_params_path),
    }

    spec["models"]["ad_model_autosomes"] = "DC"
    spec["optim"]["repetitions"] = 1
    spec["output"]["output_directory"] = hotstart_output_directory(output_directory)

    hotstart_yaml_path = yaml_path.parent / f"{yaml_path.stem}_{SUFFIX}.yaml"
    with open(hotstart_yaml_path, "w") as f:
        yaml.dump(spec, f)

    print(f"  wrote {hotstart_yaml_path.relative_to(SCRIPT_DIR)}  (start params from {optimal_params_path.relative_to(SCRIPT_DIR)})")
    return hotstart_yaml_path


def build_hotstart_driver_script(py_path: Path, old_yaml_name: str, hotstart_yaml_path: Path) -> None:
    text = py_path.read_text()
    new_text, n = re.subn(
        r'(driver_filename\s*=\s*["\'])' + re.escape(old_yaml_name) + r'(["\'])',
        rf"\g<1>{hotstart_yaml_path.name}\g<2>",
        text,
    )
    if n == 0:
        print(f"  ! could not update driver_filename in {py_path}")
        return

    hotstart_py_path = py_path.parent / f"{py_path.stem}_{SUFFIX}.py"
    hotstart_py_path.write_text(new_text)
    print(f"  wrote {hotstart_py_path.relative_to(SCRIPT_DIR)}")


def main():
    pop_models = parse_run_all_models(RUN_ALL_SH)
    for pop, models in pop_models.items():
        pop_dir = SCRIPT_DIR / pop
        print(f"{pop}:")
        for model in models:
            py_path = pop_dir / f"{pop}_{model}.py"
            if not py_path.is_file():
                print(f"  ! driver script not found: {py_path}")
                continue
            yaml_filename = extract_driver_filename(py_path)
            hotstart_yaml_path = build_hotstart_yaml(pop_dir, yaml_filename)
            if hotstart_yaml_path is not None:
                build_hotstart_driver_script(py_path, yaml_filename, hotstart_yaml_path)


if __name__ == "__main__":
    main()
