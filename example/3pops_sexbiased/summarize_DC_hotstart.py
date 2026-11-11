#!/usr/bin/env python3
"""
Summarizes the latest completed runs produced by run_all_DC_hotstart.sh.

For every <POP>/output_*_DC_hotstart/ directory under this folder, finds the
most recent timestamped run and prints the contents of its
optimal_model/*_optimal_parameters.txt.
"""
import argparse
import re
import sys
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def find_latest_run(output_dir: Path) -> Path | None:
    timestamp_dirs = sorted(
        (d for d in output_dir.iterdir() if d.is_dir() and TIMESTAMP_RE.match(d.name)),
        key=lambda d: d.name,
    )
    return timestamp_dirs[-1] if timestamp_dirs else None


def find_optimal_parameters_file(run_dir: Path) -> Path | None:
    optimal_model_dir = run_dir / "optimal_model"
    if not optimal_model_dir.is_dir():
        return None
    matches = sorted(optimal_model_dir.glob("*_optimal_parameters.txt"))
    return matches[0] if matches else None


def model_label(output_dir: Path) -> str:
    name = output_dir.name
    if name.startswith("output_") and name.endswith("_DC_hotstart"):
        return name[len("output_"):-len("_DC_hotstart")]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir", nargs="?", default=None,
        help="Directory containing the <POP>/output_*_DC_hotstart run directories "
             "(default: this script's own directory).",
    )
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent

    output_dirs = sorted(base_dir.glob("*/output_*_DC_hotstart"))
    if not output_dirs:
        print(f"No output_*_DC_hotstart directories found under {base_dir}", file=sys.stderr)
        return 1

    for output_dir in output_dirs:
        population = output_dir.parent.name
        header = f"=== {population} / {model_label(output_dir)} ==="

        latest_run = find_latest_run(output_dir)
        if latest_run is None:
            print(f"{header}\n  (no timestamped run directories found)\n")
            continue

        params_file = find_optimal_parameters_file(latest_run)
        if params_file is None:
            print(f"{header}  [{latest_run.name}]")
            print("  (no optimal_model/*_optimal_parameters.txt found -- run may have failed or not finished)\n")
            continue

        print(f"{header}  [{latest_run.name}]")
        print(params_file.read_text().rstrip())
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
