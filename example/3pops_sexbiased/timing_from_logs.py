"""
Estimate runtimes of the runs launched by ``run_all.sh`` and ``run_all_DC_hotstart.sh`` from the
timestamps in their existing log files. Nothing is run and, unless ``--csv`` is given, nothing is
written: results are printed to the screen.

For each population/model listed in a run_all script (and enabled with ``RUN_<POP>=1``), the
script reads the driver YAML's ``output.output_directory``, picks the most recent completed run
(a log ending with "Results saved to"), and reports:

* wall time (first to last log line);
* number of optimizations, re-optimization rounds and boundary re-optimizations;
* number of objective evaluations by type -- autosomal (A), allosomal (X), both (A+X), or
  out-of-bounds/infeasible (OOB, cheap: no likelihood is computed) -- and the median time per
  evaluation for each type;
* the share of wall time spent in objective evaluations.

For each script it also reports the summed wall time of all runs (each run is a single process,
so this approximates total CPU time when ``N_cores`` is 1) and the batch wall time (earliest start
to latest end), together with the maximum number of runs that overlapped in time.

Caveats: runs that overlapped in time competed for CPU, so their timings are slower than a
standalone run would be. Evaluation times are the gaps between consecutive ``iter=`` log lines; the
first evaluation of each optimization (whose gap includes setup) is not timed, and when
``verbose_log`` > 1 the gap is divided evenly among the evaluations it spans.

Usage::

    python timing_from_logs.py                      # both scripts, latest completed runs
    python timing_from_logs.py --on 20261109        # only runs from timestamp dirs starting with 20261109
    python timing_from_logs.py --scripts run_all.sh --csv timings.csv
    python timing_from_logs.py --plot timing.pdf    # also plot runtime against the number of generations

The number of generations of each run is read from its fitted migration matrix
(``optimal_model/*_female_migration_matrix``), and the admixture models from the driver YAML.
"""

import argparse
import csv
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ruamel.yaml import YAML

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCRIPTS = ["run_all.sh", "run_all_DC_hotstart.sh"]

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ")
_ITER_RE = re.compile(r"^(\S+ \S+) - tracts\.core - INFO - iter=(\d+)\s*\|.*\|\s*params=array\(\[.*?\]\)\s*(.*)$")
_EVAL_TYPES = ["A", "X", "A+X", "OOB"]


def _parse_ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f")


# ----------------------------------------------------------------------------------------------
# run_all script and driver file parsing
# ----------------------------------------------------------------------------------------------

def parse_run_all_script(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Returns the driver suffix and the enabled (population, model) pairs of a run_all script."""
    text = path.read_text()
    suffix_match = re.search(r'^SUFFIX="([^"]*)"', text, re.MULTILINE)
    suffix = suffix_match.group(1) if suffix_match else ""
    enabled = {pop for pop, flag in re.findall(r"^RUN_(\w+)=(\d)", text, re.MULTILINE) if flag == "1"}
    runs = []
    for pop, models in re.findall(r"^(\w+)_MODELS=\(([^)]*)\)", text, re.MULTILINE):
        if pop in enabled:
            runs.extend((pop, model) for model in models.split())
    return suffix, runs


def output_root(driver_yaml: Path) -> Path:
    """The run's output directory, without the ``{date}`` part, resolved from the population directory."""
    spec = YAML(typ="safe").load(driver_yaml.read_text())
    out = spec["output"]["output_directory"]
    out = out.split("{date}")[0]
    return (driver_yaml.parent / out).resolve()


def admixture_models(driver_yaml: Path) -> tuple[str | None, str | None]:
    """The autosomal and allosomal admixture models (e.g. 'M', 'DC') named in a driver file."""
    models = YAML(typ="safe").load(driver_yaml.read_text()).get("models", {})
    return models.get("ad_model_autosomes"), models.get("ad_model_allosomes")


def n_generations(log: Path) -> int | None:
    """
    Number of generations in the run's fitted model: the number of rows of the optimal migration
    matrix saved next to the log (``optimal_model/*_female_migration_matrix``), or None if absent.
    """
    for matrix in (log.parent / "optimal_model").glob("*female_migration_matrix*"):
        if matrix.is_file():
            return sum(1 for line in matrix.read_text().splitlines() if line.strip())
    return None


def latest_completed_log(root: Path, on: str | None) -> Path | None:
    """The log of the most recent completed run under ``root`` (optionally from a given date prefix)."""
    if not root.is_dir():
        return None
    for run_dir in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        if on and not run_dir.name.startswith(on):
            continue
        for log in run_dir.glob("*.log"):
            lines = log.read_text(errors="replace").rstrip().splitlines()
            if lines and "Results saved to" in lines[-1]:
                return log
    return None


# ----------------------------------------------------------------------------------------------
# Log parsing
# ----------------------------------------------------------------------------------------------

@dataclass
class RunTiming:
    pop: str
    model: str
    log: Path
    start: datetime
    end: datetime
    n_optimizations: int = 0
    n_reoptimizations: int = 0
    n_boundary_reoptimizations: int = 0
    n_evals: dict = field(default_factory=lambda: {t: 0 for t in _EVAL_TYPES})
    eval_records: list = field(default_factory=list)  # (type, previous timestamp, timestamp, span) of timed evaluations
    long_gaps: list = field(default_factory=list)     # (start, end) of gaps between log lines longer than max_gap
    suspensions: list = field(default_factory=list)   # the long gaps shared with another run: machine suspended
    generations: int | None = None                    # rows of the fitted migration matrix
    ad_model_autosomes: str | None = None
    ad_model_allosomes: str | None = None

    @property
    def raw_wall(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def suspended(self) -> float:
        return sum((b - a).total_seconds() for a, b in self.suspensions)

    @property
    def wall(self) -> float:
        """Wall time excluding periods when the machine was suspended."""
        return self.raw_wall - self.suspended

    @property
    def eval_times(self) -> dict:
        times = {t: [] for t in _EVAL_TYPES}
        for eval_type, prev_ts, ts, span in self.eval_records:
            if any(prev_ts < b and ts > a for a, b in self.suspensions):
                continue  # spans a suspension: not a real evaluation time
            times[eval_type].extend([(ts - prev_ts).total_seconds() / span] * span)
        return times

    def median_eval(self, eval_type: str) -> float | None:
        times = self.eval_times[eval_type]
        return statistics.median(times) if times else None

    def estimated_eval_total(self) -> float:
        """Time spent in evaluations, using the type median for evaluations that could not be timed."""
        total = 0.0
        all_times = self.eval_times
        for t in _EVAL_TYPES:
            times = all_times[t]
            untimed = self.n_evals[t] - len(times)
            total += sum(times) + (untimed * statistics.median(times) if times and untimed > 0 else 0.0)
        return total


def _eval_type(labels: set[str]) -> str:
    if any(label.startswith(("OOB", "Singular")) for label in labels):
        return "OOB"
    has_a = "Autosomes" in labels
    has_x = any("allosomes" in label for label in labels)
    if has_a and has_x:
        return "A+X"
    return "A" if has_a else "X"


def parse_log(pop: str, model: str, log: Path, max_gap: float) -> RunTiming:
    lines = log.read_text(errors="replace").splitlines()
    stamps = [_parse_ts(m.group(1)) for line in lines if (m := _TS_RE.match(line))]
    timing = RunTiming(pop=pop, model=model, log=log, start=stamps[0], end=stamps[-1])
    timing.long_gaps = [(a, b) for a, b in zip(stamps, stamps[1:]) if (b - a).total_seconds() > max_gap]

    # Group the iter= lines of one evaluation (one line per likelihood component) together.
    evals = []  # (iter, timestamp, labels)
    for line in lines:
        if "Optimization completed." in line:
            timing.n_optimizations += 1
        elif re.search(r"Re-optimization \d+/\d+:", line):
            timing.n_reoptimizations += 1
        elif "Re-optimizing with sex-bias parameter(s)" in line:
            timing.n_boundary_reoptimizations += 1
        m = _ITER_RE.match(line)
        if not m:
            continue
        it, ts, label = int(m.group(2)), _parse_ts(m.group(1)), m.group(3).strip()
        if evals and evals[-1][0] == it:
            evals[-1][2].add(label)
        else:
            evals.append((it, ts, {label}))

    prev_it, prev_ts = None, None
    for it, ts, labels in evals:
        eval_type = _eval_type(labels)
        new_segment = prev_it is None or it <= prev_it  # iteration counter restarted: new optimization
        span = 1 if new_segment else it - prev_it      # evaluations covered by this log line (verbose_log > 1)
        timing.n_evals[eval_type] += span
        if not new_segment:
            timing.eval_records.append((eval_type, prev_ts, ts, span))
        prev_it, prev_ts = it, ts
    return timing


def mark_suspensions(timings: list[RunTiming]) -> None:
    """
    Marks as suspensions the long gaps that overlap a long gap of another run: several runs falling
    silent at the same time means the machine was asleep, not that an evaluation was slow.
    """
    for i, t in enumerate(timings):
        t.suspensions = [
            (a, b) for a, b in t.long_gaps
            if any(a < d and b > c for j, other in enumerate(timings) if j != i for c, d in other.long_gaps)
        ]


def _union(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def idle_periods(timings: list[RunTiming], min_idle: float = 60) -> list[tuple[datetime, datetime]]:
    """Periods between the first start and the last end during which no run was active."""
    busy = _union([(t.start, t.end) for t in timings])
    return [(b, c) for (_, b), (c, _) in zip(busy, busy[1:]) if (c - b).total_seconds() > min_idle]


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _fmt_sec(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def max_overlap(timings: list[RunTiming]) -> int:
    events = sorted([(t.start, 1) for t in timings] + [(t.end, -1) for t in timings], key=lambda e: (e[0], e[1]))
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


def report(script: str, timings: list[RunTiming], missing: list[str]) -> None:
    print(f"\n=== {script} ===")
    header = (f"{'run':<28} {'run dir':<16} {'wall':>10} {'opt':>4} {'reopt':>5} {'bnd':>4} "
              f"{'#A':>5} {'#X':>5} {'#A+X':>5} {'#OOB':>5} {'s/A':>6} {'s/X':>6} {'s/A+X':>6} {'eval%':>6} {'suspended':>10}")
    print(header)
    print("-" * len(header))
    for t in timings:
        eval_share = 100 * t.estimated_eval_total() / t.wall if t.wall > 0 else 0
        print(f"{t.pop + '_' + t.model:<28} {t.log.parent.name:<16} {_fmt_duration(t.wall):>10} "
              f"{t.n_optimizations:>4} {t.n_reoptimizations:>5} {t.n_boundary_reoptimizations:>4} "
              f"{t.n_evals['A']:>5} {t.n_evals['X']:>5} {t.n_evals['A+X']:>5} {t.n_evals['OOB']:>5} "
              f"{_fmt_sec(t.median_eval('A')):>6} {_fmt_sec(t.median_eval('X')):>6} {_fmt_sec(t.median_eval('A+X')):>6} "
              f"{eval_share:>5.0f}% {(_fmt_duration(t.suspended) if t.suspended else ''):>10}")
    for name in missing:
        print(f"{name:<28} (no completed run found)")

    if not timings:
        return
    walls = [t.wall for t in timings]
    summed = sum(walls)
    batch_start, batch_end = min(t.start for t in timings), max(t.end for t in timings)
    batch_suspended = sum((min(b, batch_end) - max(a, batch_start)).total_seconds()
                          for a, b in _union([s for t in timings for s in t.suspensions]))
    batch = (batch_end - batch_start).total_seconds() - batch_suspended
    idle = idle_periods(timings)
    print(f"\n  runs found:                  {len(timings)}" + (f"  (missing: {len(missing)})" if missing else ""))
    print(f"  per-run wall: median {_fmt_duration(statistics.median(walls))}, "
          f"min {_fmt_duration(min(walls))}, max {_fmt_duration(max(walls))}")
    print(f"  summed wall (~CPU time):     {_fmt_duration(summed)}  ({summed / 3600:.2f} h)")
    print(f"  batch wall (first start to last end): {_fmt_duration(batch)}; max concurrent runs: {max_overlap(timings)}")
    if batch_suspended:
        print(f"  machine suspended for {_fmt_duration(batch_suspended)} during the batch (excluded from all times above)")
    if idle:
        print(f"  WARNING: no run was active for {len(idle)} period(s) totalling "
              f"{_fmt_duration(sum((b - a).total_seconds() for a, b in idle))}: these runs probably come from "
              f"different launches, so the batch wall time is not meaningful (use --on to select one launch).")
    for t in timings:
        for a, b in t.long_gaps:
            if (a, b) not in t.suspensions:
                print(f"  NOTE: {t.pop}_{t.model} logged nothing for {_fmt_duration((b - a).total_seconds())} "
                      f"({a:%Y-%m-%d %H:%M} to {b:%H:%M}), not shared with another run: kept as computation time.")
    for eval_type in _EVAL_TYPES[:3]:
        pooled = [x for t in timings for x in t.eval_times[eval_type]]
        if pooled:
            print(f"  time per {eval_type:<3} evaluation (all runs): median {statistics.median(pooled):.2f} s, "
                  f"range {min(pooled):.2f}-{max(pooled):.2f} s, n={len(pooled)}")


def write_csv(path: Path, rows: list[tuple[str, RunTiming]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["script", "population", "model", "run_dir", "start", "end", "wall_s", "suspended_s",
                         "n_optimizations", "n_reoptimizations", "n_boundary_reoptimizations",
                         *[f"n_eval_{t}" for t in _EVAL_TYPES],
                         *[f"median_s_per_eval_{t}" for t in _EVAL_TYPES[:3]], "estimated_eval_time_s"])
        for script, t in rows:
            writer.writerow([script, t.pop, t.model, t.log.parent.name, t.start.isoformat(), t.end.isoformat(),
                             f"{t.wall:.3f}", f"{t.suspended:.3f}", t.n_optimizations, t.n_reoptimizations, t.n_boundary_reoptimizations,
                             *[t.n_evals[e] for e in _EVAL_TYPES],
                             *[("" if t.median_eval(e) is None else f"{t.median_eval(e):.4f}") for e in _EVAL_TYPES[:3]],
                             f"{t.estimated_eval_total():.3f}"])


# ----------------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------------

# Categorical slots 1-3 of the reference palette (validated as a set for scatter plots), with a
# distinct marker per slot so that series are not distinguished by color alone. Series keep their
# slot whatever subset is plotted: M is always slot 1, DC slot 2, X slot 3.
_SERIES_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
_INK, _INK_MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _style_axis(ax) -> None:
    ax.set_yscale("log")
    ax.grid(True, which="major", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_MUTED)
    ax.tick_params(colors=_INK_MUTED, labelcolor=_INK)
    ax.xaxis.get_major_locator().set_params(integer=True)


def _draw_series(ax, series: dict[str, list[tuple[float, float, float, float]]], order: list[str]) -> None:
    """Draws each series (points (x, median, low, high)) with its fixed style, offset slightly in x."""
    present = [name for name in order if series.get(name)]
    for name in present:
        slot = order.index(name)
        color, marker = _SERIES_STYLE[slot % len(_SERIES_STYLE)]
        offset = (present.index(name) - (len(present) - 1) / 2) * 0.12  # so runs at the same depth don't overlap
        points = sorted(series[name])
        xs = [p[0] + offset for p in points]
        ys = [p[1] for p in points]
        yerr = [[p[1] - p[2] for p in points], [p[3] - p[1] for p in points]]
        ax.errorbar(xs, ys, yerr=yerr, fmt=marker, color=color, markersize=7, elinewidth=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=1.2, label=name, zorder=3)


def plot_timings(per_script: list[tuple[str, list[RunTiming], list[str]]], path: Path) -> None:
    """
    Saves a two-panel figure: (left) median time per likelihood evaluation, with interquartile
    range, against the number of generations of the fitted model, by chromosome type and admixture
    model; (right) wall time per run against the number of generations, by run_all script.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Left panel: time per evaluation, one series per (chromosome type, admixture model) ----
    eval_series: dict[str, list] = {}
    for _, timings, _ in per_script:
        for t in timings:
            if t.generations is None:
                continue
            all_times = t.eval_times
            for eval_type, chrom, model in (("A", "Autosomes", t.ad_model_autosomes),
                                            ("X", "X chromosome", t.ad_model_allosomes)):
                times = all_times[eval_type]
                if times:
                    eval_series.setdefault(f"{chrom}, {model} model", []).append(
                        (t.generations, statistics.median(times), _percentile(times, 0.25), _percentile(times, 0.75)))
    # Fixed slot order: M autosomes, DC autosomes, X; any other combination is appended after.
    eval_order = ["Autosomes, M model", "Autosomes, DC model", "X chromosome, DC model"]
    eval_order += sorted(name for name in eval_series if name not in eval_order)

    # ---- Right panel: wall time per run, one series per script, labelled by its admixture models ----
    wall_series: dict[str, list] = {}
    wall_order = []
    for script, timings, _ in per_script:
        if not timings:
            continue
        models = sorted({f"{t.ad_model_autosomes}/{t.ad_model_allosomes}" for t in timings})
        name = f"{script} (autosomes/X: {', '.join(models)})"
        wall_order.append(name)
        wall_series[name] = [(t.generations, t.wall / 60, t.wall / 60, t.wall / 60)
                             for t in timings if t.generations is not None]

    fig, (ax_eval, ax_wall) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    _draw_series(ax_eval, eval_series, eval_order)
    _style_axis(ax_eval)
    ax_eval.set_xlabel("Generations in the fitted model", color=_INK)
    ax_eval.set_ylabel("Time per likelihood evaluation (s)", color=_INK)
    ax_eval.set_title("Time per evaluation (median, interquartile range)", color=_INK, loc="left", fontsize=11)
    ax_eval.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")

    _draw_series(ax_wall, wall_series, wall_order)
    _style_axis(ax_wall)
    ax_wall.set_xlabel("Generations in the fitted model", color=_INK)
    ax_wall.set_ylabel("Wall time per run (min)", color=_INK)
    ax_wall.set_title("Wall time per run (suspensions excluded)", color=_INK, loc="left", fontsize=11)
    ax_wall.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")

    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts", nargs="+", default=DEFAULT_SCRIPTS, help="run_all scripts to analyze")
    parser.add_argument("--on", default=None, help="only use run directories whose timestamp starts with this (e.g. 20261109)")
    parser.add_argument("--csv", type=Path, default=None, help="also write per-run results to this CSV file")
    parser.add_argument("--max-gap", type=float, default=300,
                        help="gaps between log lines longer than this (seconds) that occur simultaneously in "
                             "several runs are treated as machine suspension and excluded (default: 300)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="also plot time per evaluation and wall time per run against the number of "
                             "generations, and save the figure to this file (.png, .pdf, .svg)")
    args = parser.parse_args(argv)

    per_script = []
    for script in args.scripts:
        suffix, runs = parse_run_all_script(SCRIPT_DIR / script)
        timings, missing = [], []
        for pop, model in runs:
            name = f"{pop}_{model}{suffix}"
            driver_yaml = SCRIPT_DIR / pop / f"{name}.yaml"
            log = latest_completed_log(output_root(driver_yaml), args.on) if driver_yaml.exists() else None
            if log is None:
                missing.append(name)
                continue
            timing = parse_log(pop, model, log, args.max_gap)
            timing.generations = n_generations(log)
            timing.ad_model_autosomes, timing.ad_model_allosomes = admixture_models(driver_yaml)
            timings.append(timing)
        per_script.append((script, timings, missing))

    mark_suspensions([t for _, timings, _ in per_script for t in timings])

    csv_rows = []
    for script, timings, missing in per_script:
        report(script, timings, missing)
        csv_rows.extend((script, t) for t in timings)

    if args.csv:
        write_csv(args.csv, csv_rows)
        print(f"\nPer-run results written to {args.csv}")
    if args.plot:
        plot_timings(per_script, args.plot)
        print(f"\nPlot written to {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
