"""
Migration lineage plots: representative lineages of an admixed population followed back in time into its source
populations, drawn from one migration matrix (:func:`plot_migration_lineages`), from sex-specific matrices for the
autosomes and the X chromosome (:func:`plot_sex_biased_migration_lineages`), or for several admixed populations side
by side (:func:`plot_lineages_by_population`).

The expected lineage flows are computed by :func:`compute_lineage_flows` and rounded to whole lineages by
:func:`lineage_counts`. Migration matrices of a :func:`~tracts.driver.run_tracts` output directory can be read with
:func:`~tracts.plot.read_migration_matrices_from_output`.
"""
from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

from tracts.driver_utils import get_population_colors

LINEAGE_CHROMOSOMES = ("autosome", "X")

UNRESOLVED_COLOR = "#9A9A9A"
BAND_COLOR = "#F3F3F3"
INK = "#222222"
INK_MUTED = "#666666"


# -------------- Flow computation --------------


def compute_lineage_flows(migration_matrix_f: np.ndarray, migration_matrix_m: np.ndarray | None = None,
                          chromosome: str = "autosome",
                          initial_female_fraction: float | None = None) -> dict[str, np.ndarray]:
    r"""
    Expected flows of present-day lineages from the admixed population into its source populations, back in
    time, as drawn by :func:`plot_lineage_flows`.

    ``m_f[g, j]`` (``m_m[g, j]``) is the probability that a female (male) individual of the admixed population
    at generation :math:`g` is a migrant from source :math:`j`. The fractions of lineages carried by females
    (:math:`F_g`) and by males (:math:`M_g`) at generation :math:`g` evolve as follows:

    * a fraction :math:`F_g m_f[g, j] + M_g m_m[g, j]` of all lineages migrates to source :math:`j`;
    * the lineages left in admixed individuals, :math:`F_g' = F_g (1 - \sum_j m_f[g, j])` and
      :math:`M_g' = M_g (1 - \sum_j m_m[g, j])`, move to their parents at generation :math:`g + 1`. For
      autosomes, :math:`F_{g+1} = M_{g+1} = (F_g' + M_g') / 2`. For the X chromosome (a male's X comes from
      his mother), :math:`F_{g+1} = F_g' / 2 + M_g'` and :math:`M_{g+1} = F_g' / 2`.

    The resulting ancestry proportions are :math:`(p_f[1] + p_m[1]) / 2` (autosomes) and
    :math:`(2 p_f[1] + p_m[1]) / 3` (X chromosome), with :math:`p_f, p_m` from
    :func:`~tracts.migration_matrices.ancestry_proportions_by_generation`, when row 0 of the matrices is zero. With a single matrix
    (``migration_matrix_m`` None), ``remaining[:T + 1]`` equals :func:`~tracts.migration_matrices.get_survival_factors`.

    Parameters
    ----------
    migration_matrix_f: np.ndarray
        A :math:`(T + 1) \times P` female migration matrix (row :math:`g` = generation :math:`g` ago), or the
        migration matrix of a model without sex bias.
    migration_matrix_m: np.ndarray | None
        The male migration matrix. If None, ``migration_matrix_f`` is used for both sexes.
    chromosome: str
        ``"autosome"`` or ``"X"``.
    initial_female_fraction: float | None
        Fraction of present-day lineages carried by females. Defaults to 1/2 for autosomes and 2/3 for the X
        chromosome (equal numbers of sampled females and males).

    Returns
    -------
    dict
        ``female``, ``male``: length :math:`T + 2` arrays, fraction of all present-day lineages carried by a
        female (male) individual of generation :math:`g`, before that generation's migrations.
        ``remaining``: their sum, the fraction still in the admixed population (``remaining[0] = 1``;
        ``remaining[T + 1]`` is the fraction never assigned to a source, 0 for a well-formed model).
        ``leaks_female``, ``leaks_male``: :math:`(T + 1) \times P` arrays, fraction of all lineages migrating to
        source :math:`j` at generation :math:`g` through a female (male) migrant. ``leaks`` is their sum.
        ``proportions``: length :math:`P` array, the predicted ancestry proportions (``leaks.sum(axis=0)``).
    """
    if migration_matrix_m is None:
        migration_matrix_m = migration_matrix_f
    mf = np.asarray(migration_matrix_f, dtype=float)
    mm = np.asarray(migration_matrix_m, dtype=float)
    if mf.shape != mm.shape or mf.ndim != 2:
        raise ValueError(f"Female and male matrices must be 2D with the same shape, got {mf.shape}, {mm.shape}.")
    for mat in (mf, mm):
        if np.any(mat < -1e-9) or np.any(mat.sum(axis=1) > 1 + 1e-9):
            raise ValueError("Migration rates must be non-negative and each row must sum to at most 1.")
    mf, mm = np.clip(mf, 0, None), np.clip(mm, 0, None)  # floating-point noise such as -5e-16 in saved matrices
    if chromosome not in LINEAGE_CHROMOSOMES:
        raise ValueError(f"chromosome must be one of {LINEAGE_CHROMOSOMES}, got {chromosome!r}.")
    x_linked = chromosome == "X"
    if initial_female_fraction is None:
        initial_female_fraction = 2 / 3 if x_linked else 1 / 2

    n_gen = mf.shape[0]
    female = np.zeros(n_gen + 1)
    male = np.zeros(n_gen + 1)
    female[0], male[0] = initial_female_fraction, 1 - initial_female_fraction
    leaks_f = np.zeros_like(mf)
    leaks_m = np.zeros_like(mm)
    for g in range(n_gen):
        leaks_f[g] = female[g] * mf[g]
        leaks_m[g] = male[g] * mm[g]
        stay_f = female[g] * (1 - mf[g].sum())
        stay_m = male[g] * (1 - mm[g].sum())
        if x_linked:
            female[g + 1] = stay_f / 2 + stay_m
            male[g + 1] = stay_f / 2
        else:
            female[g + 1] = male[g + 1] = (stay_f + stay_m) / 2

    leaks = leaks_f + leaks_m
    return dict(female=female, male=male, remaining=female + male, leaks_female=leaks_f, leaks_male=leaks_m,
                leaks=leaks, proportions=leaks.sum(axis=0))


def _as_flows(matrix_or_flows) -> dict[str, np.ndarray]:
    """Accepts either a migration matrix or an already computed flows dict (``leaks`` and ``remaining``)."""
    if isinstance(matrix_or_flows, dict):
        return matrix_or_flows
    return compute_lineage_flows(matrix_or_flows)


def lineage_counts(flows, n_lineages: int = 100) -> tuple[np.ndarray, int]:
    """
    Rounds expected lineage flows to whole lineages (largest-remainder rounding, so that the counts add up to
    ``n_lineages``).

    Parameters
    ----------
    flows: dict | np.ndarray
        A flows dict as returned by :func:`compute_lineage_flows` (any dict with ``leaks`` and ``remaining``
        works), or a migration matrix.
    n_lineages: int
        Number of lineages.

    Returns
    -------
    (counts, n_unresolved)
        ``counts[g, j]`` is the number of lineages migrating to source ``j`` at generation ``g``;
        ``n_unresolved`` is the number never assigned to a source.
    """
    flows = _as_flows(flows)
    expected = np.append(flows["leaks"].ravel(), flows["remaining"][-1]) * n_lineages
    counts = np.floor(expected).astype(int)
    shortfall = n_lineages - counts.sum()
    counts[np.argsort(-(expected - counts), kind="stable")[:shortfall]] += 1
    return counts[:-1].reshape(flows["leaks"].shape), int(counts[-1])


def choose_n_lineages(flows, n_min: int = 100, n_max: int = 1000, min_proportion: float = 1e-9) -> int:
    """
    The smallest number of lineages, between ``n_min`` and ``n_max``, for which every source population whose
    ancestry proportion exceeds ``min_proportion`` gets at least one lineage after rounding by
    :func:`lineage_counts`. Returns ``n_max`` if no such number exists in that range. ``flows`` is a flows dict
    or a migration matrix, as in :func:`lineage_counts`.
    """
    flows = _as_flows(flows)
    props = flows["leaks"].sum(axis=0)
    present = props > min_proportion
    if not present.any():
        return n_min
    n = int(min(max(n_min, np.ceil(1 / props[present].min() - 1e-9)), n_max))
    # Rounding is per (generation, source), so the bound above may be a few lineages short.
    while n < n_max and np.any(lineage_counts(flows, n)[0].sum(axis=0)[present] == 0):
        n += 1
    return n


# -------------- Plotting --------------

def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3 - 2 * u)


def plot_migration_lineages(migration_matrix: np.ndarray, pop_labels: list[str], **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Draws representative lineages for a single migration matrix. See :func:`plot_lineage_flows` for the
    construction and the keyword arguments.

    Parameters
    ----------
    migration_matrix: np.ndarray
        A ``(T + 1) x P`` migration matrix, row ``g`` = generation ``g`` ago, column ``j`` = source population.
        For a sex-biased model pass the mean of the female and male matrices (autosomal lineages).
    pop_labels: list[str]
        Names of the ``P`` source populations, in column order.
    """
    return plot_lineage_flows(compute_lineage_flows(migration_matrix), pop_labels, **kwargs)


def plot_lineage_flows(flows: dict[str, np.ndarray], pop_labels: list[str], colors: list | dict | None = None,
                            n_lineages: int | None = None, ax: plt.Axes | None = None, title: str | None = None,
                            admixed_label: str = "Admixed", pop_order: list[str] | None = None,
                            alpha: float = 0.6, linewidth: float | None = None, band_gap: float = 0.08,
                            source_gap: float = 0.25, transition_width: float = 1.0, tail: float = 0.6,
                            show_bands: bool | str = "admixed", orientation: str = "vertical",
                            label_fontsize: float = 10, tick_fontsize: float = 9,
                            resolution: int = 25) -> tuple[plt.Figure, plt.Axes]:
    """
    Draws representative lineages moving from an admixed population into its source populations, back in
    time.

    All lineages start in the admixed population. At each generation, the expected fraction of lineages
    migrating to each source (see :func:`compute_lineage_flows`) is rounded to whole
    lineages (:func:`lineage_counts`); a migrating lineage moves into the band of
    its source and stays there. Each lineage is coloured by the source it eventually reaches, so the colour mix
    of the admixed band at the present shows the predicted ancestry proportions. Lineages are semi-transparent
    and ordered to keep crossings rare: they leave the admixed band from the side nearest the sources,
    earliest leavers first, and each source band fills from that side.

    Parameters
    ----------
    flows: dict[str, np.ndarray]
        Expected lineage flows, with ``leaks`` (a ``(T + 1) x P`` array: fraction of all present-day lineages
        that migrate to source ``j`` at generation ``g``) and ``remaining`` (length ``T + 2``; only its last
        entry, the fraction never assigned to a source, is used). See
        :func:`compute_lineage_flows`.
    pop_labels: list[str]
        Names of the ``P`` source populations, in column order.
    colors: list | dict | None
        Colours of the source populations, as a list in column order or a ``{label: color}`` dict. Defaults to
        :func:`~tracts.driver_utils.get_population_colors`, matching the other tracts output plots.
    n_lineages: int | None
        Number of representative lineages. If None, chosen by
        :func:`choose_n_lineages` (the fewest, between
        100 and 1000, that show every source population).
    ax: plt.Axes | None
        Axes to draw on. A new figure is created if None.
    title: str | None
        Optional axes title.
    admixed_label: str
        Label of the admixed population band.
    pop_order: list[str] | None
        Source bands from top to bottom. Defaults to ordering by mean migration generation, with the band
        reached latest (the oldest migrants) closest to the admixed band, which keeps crossings rare.
    alpha: float
        Lineage opacity. Overlapping and crossing lineages blend.
    linewidth: float | None
        Lineage line width, in points. If None, ``0.6 * sqrt(100 / n_lineages)``, so that individual lines stay
        distinguishable as their number grows.
    band_gap: float
        Gap between source bands, as a fraction of ``n_lineages`` slots.
    source_gap: float
        Gap between the admixed band and the nearest source band, as a fraction of ``n_lineages`` slots (at
        least ``band_gap``). Separates the admixed population from its sources.
    transition_width: float
        Extent (in generations, at most 1) of the curve a lineage follows when it migrates. The curve ends at
        the migrant's generation.
    tail: float
        Extra space after the oldest generation, where the bands show final counts.
    show_bands: bool | str
        Light background behind the bands: ``"admixed"`` (admixed band only), ``"all"`` (or True), or
        ``"none"`` (or False).
    orientation: str
        ``"vertical"``: present at the bottom, past at the top, admixed band on the left and sources to its
        right. ``"horizontal"``: present on the left, past on the right, admixed band at the bottom.
    label_fontsize, tick_fontsize: float
        Font sizes.
    resolution: int
        Number of points used to draw each migration curve.

    Returns
    -------
    (fig, ax)
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {orientation!r}.")
    vertical = orientation == "vertical"
    shade = {True: "all", False: "none"}.get(show_bands, show_bands)
    if shade not in ("all", "admixed", "none"):
        raise ValueError(f"show_bands must be 'admixed', 'all', 'none' or a bool, got {show_bands!r}.")
    pop_labels = list(pop_labels)
    n_gen, n_pop = flows["leaks"].shape
    flows = dict(flows, proportions=flows["leaks"].sum(axis=0))
    if n_pop != len(pop_labels):
        raise ValueError(f"Flows have {n_pop} columns but {len(pop_labels)} labels were given.")
    if colors is None:
        colors = get_population_colors(pop_labels)
    if isinstance(colors, dict):
        colors = [colors[p] for p in pop_labels]

    if n_lineages is None:
        n_lineages = choose_n_lineages(flows)
    if linewidth is None:
        linewidth = 0.6 * np.sqrt(100 / n_lineages)
    counts, n_unresolved = lineage_counts(flows, n_lineages)
    if counts[0].sum() > 0:
        warnings.warn("Row 0 of the migration matrix (present generation) is non-zero; "
                      "those lineages are drawn starting in their source population.")
    if n_unresolved > 0:
        warnings.warn(f"{n_unresolved} of {n_lineages} lineages are never assigned to a source population "
                      f"(the oldest row of the migration matrix does not sum to 1); drawn in grey.")

    # Band order (top to bottom). Default: earliest mean arrival on top, latest closest to the admixed band.
    gens = np.arange(n_gen)
    if pop_order is None:
        mean_gen = np.array([(counts[:, j] * gens).sum() / max(counts[:, j].sum(), 1) for j in range(n_pop)])
        order = list(np.argsort(mean_gen, kind="stable"))
    else:
        order = [pop_labels.index(p) for p in pop_order]
    # Sources with no lineage after rounding get no band; they are listed in a note instead.
    dropped = [j for j in order if counts[:, j].sum() == 0]
    order = [j for j in order if counts[:, j].sum() > 0]
    rank = {j: r for r, j in enumerate(order)}  # 0 = top band

    # Layout in (generation, slot) coordinates, one slot per lineage: the admixed band at the lowest slots,
    # then the sources. Drawn as (x, y) = (generation, slot) when horizontal, (slot, generation) when vertical.
    # Below, "top"/"bottom" refer to slots (right/left in the vertical orientation).
    gap = band_gap * n_lineages
    adm_bottom = 0.0
    adm_top = adm_bottom + n_lineages - 1
    band_bottom, band_top = {}, {}
    y = adm_top + max(source_gap * n_lineages, gap) + 1
    for j in reversed(order):
        n_j = counts[:, j].sum()
        band_bottom[j] = y
        band_top[j] = y + n_j - 1
        y = band_top[j] + gap + 1
    y_top = y - gap - 1

    # Lineages, sorted by migration generation, then by band (higher band first). Lineage k leaves from the
    # admixed slot adm_top - k, so earlier leavers are nearer the top and do not cross the remaining ones.
    events = [(g, j) for g in range(n_gen) for j in order for _ in range(counts[g, j])]
    next_slot = {j: band_top[j] for j in order}  # source bands fill from the top down

    x_end = n_gen - 1
    w = min(max(transition_width, 0.05), 1.0)
    s = _smoothstep(np.linspace(0, 1, resolution))
    # Lineages of rare sources (fewer than rare_threshold lineages) are drawn opaque and at least 0.6 pt wide,
    # so that a lone lineage stays visible among hundreds.
    rare_threshold = 5
    n_per_pop = counts.sum(axis=0)

    def style(j):
        if j is not None and n_per_pop[j] < rare_threshold:
            return to_rgba(colors[j], 1.0), max(linewidth, 0.6)
        return to_rgba(colors[j] if j is not None else UNRESOLVED_COLOR, alpha), linewidth

    segments, seg_colors, seg_widths = [], [], []
    for k, (g, j) in enumerate(events):
        y0 = adm_top - k
        y1 = next_slot[j]
        next_slot[j] -= 1
        if g == 0:
            path = np.array([[0, y1], [x_end + tail, y1]])
        else:
            xc = g - w + w * np.linspace(0, 1, resolution)
            path = np.vstack([[0, y0], np.column_stack([xc, y0 + (y1 - y0) * s]), [x_end + tail, y1]])
        segments.append(path)
        c, lw = style(j)
        seg_colors.append(c)
        seg_widths.append(lw)
    for k in range(len(events), len(events) + n_unresolved):
        segments.append(np.array([[0, adm_top - k], [x_end + tail, adm_top - k]]))
        c, lw = style(None)
        seg_colors.append(c)
        seg_widths.append(lw)

    gen_hi = x_end + tail
    pad = 0.5 + 0.25 * gap
    slot_lo, slot_hi = adm_bottom - pad - 0.5, y_top + pad + 0.5
    if vertical:
        segments = [p[:, ::-1] for p in segments]

    def xy(gen, slot):
        """(generation, slot) coordinates to (x, y) data coordinates."""
        return (slot, gen) if vertical else (gen, slot)

    if ax is None:
        long_side = max(6.5, min(13.0, 0.35 * n_gen + 4.5))
        fig, ax = plt.subplots(figsize=(5.5, long_side) if vertical else (long_side, 5.5))
    else:
        fig = ax.figure

    # Background bands, admixed label, and colour bars at the end of the source bands.
    bands = [(adm_bottom, adm_top, f"{admixed_label}", None)]
    bands += [(band_bottom[j], band_top[j], pop_labels[j], j) for j in order]
    for lo, hi, label, j in bands:
        if shade == "all" or (shade == "admixed" and j is None):
            span = ax.axvspan if vertical else ax.axhspan
            span(lo - pad, hi + pad, color=BAND_COLOR, lw=0, zorder=0)
        if j is None:
            if vertical:
                ax.text((lo + hi) / 2, -0.12, label, ha="center", va="top", fontsize=label_fontsize, color=INK)
            else:
                ax.text(-0.15, (lo + hi) / 2, label, ha="right", va="center", fontsize=label_fontsize, color=INK)
        else:
            (x0, y0), (x1, y1) = xy(gen_hi + 0.05, lo - 0.5), xy(gen_hi + 0.05, hi + 0.5)
            ax.plot([x0, x1], [y0, y1], color=colors[j], lw=3, solid_capstyle="butt", zorder=3, clip_on=False)

    ax.add_collection(LineCollection(segments, colors=seg_colors, linewidths=seg_widths,
                                     capstyle="butt", joinstyle="round", zorder=2))

    for g in range(n_gen):
        (ax.axhline if vertical else ax.axvline)(g, color="white", lw=0.8, zorder=1)
    if dropped:
        note = ", ".join(f"{pop_labels[j]} {flows['proportions'][j]:.1%}" for j in dropped)
        ax.text(1.0, -0.06 if vertical else -0.13, f"Not drawn (< 1 of {n_lineages} lineages): {note}",
                transform=ax.transAxes, ha="right", va="top", fontsize=tick_fontsize, color=INK_MUTED)

    # Generation axis.
    gen_axis, slot_axis = (ax.yaxis, ax.xaxis) if vertical else (ax.xaxis, ax.yaxis)
    gen_spine = "left" if vertical else "bottom"
    (ax.set_ylim if vertical else ax.set_xlim)(0, gen_hi)
    (ax.set_xlim if vertical else ax.set_ylim)(slot_lo, slot_hi)
    slot_axis.set_ticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side == gen_spine)
    ax.spines[gen_spine].set_color(INK_MUTED)
    ax.tick_params(axis="y" if vertical else "x", labelsize=tick_fontsize, colors=INK_MUTED)
    gen_axis.set_major_locator(MaxNLocator(integer=True))
    gen_axis.set_label_text("Generations ago", fontsize=label_fontsize, color=INK)
    if title:
        # In the vertical orientation the two-line source labels sit above the axes; leave room for them.
        ax.set_title(title, fontsize=label_fontsize + 2, color=INK,
                     pad=2.6 * label_fontsize + 6 if vertical else 6)

    # Source labels, centred on their band but pushed apart where bands are too thin.
    if vertical:
        # Above the bands, left to right; label widths measured in data units.
        renderer = fig.canvas.get_renderer()
        data_per_px = (slot_hi - slot_lo) / ax.get_window_extent(renderer).width
        sep = 0.6 * label_fontsize * fig.dpi / 72 * data_per_px
        prev_x = prev_w = None
        for j in reversed(order):
            text = ax.text((band_bottom[j] + band_top[j]) / 2, gen_hi + 0.15,
                           f"{pop_labels[j]}\n{flows['proportions'][j]:.1%}", ha="center", va="bottom",
                           fontsize=label_fontsize, color=INK, linespacing=1.1)
            w = text.get_window_extent(renderer).width * data_per_px
            if prev_x is not None:
                x = max(text.get_position()[0], prev_x + (prev_w + w) / 2 + sep)
                text.set_x(x)
            prev_x, prev_w = text.get_position()[0], w
    else:
        y_lo, y_hi = ax.get_ylim()
        axes_height_pt = ax.get_window_extent().height * 72 / fig.dpi
        min_sep = 1.3 * label_fontsize * (y_hi - y_lo) / axes_height_pt
        label_y = []
        for j in order:  # top to bottom
            y = (band_bottom[j] + band_top[j]) / 2
            if label_y:
                y = min(y, label_y[-1] - min_sep)
            label_y.append(y)
        for j, y in zip(order, label_y):
            ax.text(gen_hi + 0.15, y, f"{pop_labels[j]}  {flows['proportions'][j]:.1%}", ha="left",
                    va="center", fontsize=label_fontsize, color=INK)
    return fig, ax


def plot_sex_biased_migration_lineages(migration_matrix_f: np.ndarray, migration_matrix_m: np.ndarray,
                                       pop_labels: list[str], chromosomes: tuple[str, ...] = LINEAGE_CHROMOSOMES,
                                       allosome_label: str = "X", n_lineages: int | None = None,
                                       title: str | None = None, figsize: tuple[float, float] | None = None,
                                       orientation: str = "vertical", **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """
    Lineage plots (see :func:`plot_lineage_flows`) of sex-specific migration matrices, one panel per chromosome
    type (autosomes and X by default), sharing the generation axis: side by side in the vertical orientation,
    stacked in the horizontal one. The flows are computed by
    :func:`compute_lineage_flows`.

    Parameters
    ----------
    migration_matrix_f, migration_matrix_m: np.ndarray
        ``(T + 1) x P`` female and male migration matrices.
    pop_labels: list[str]
        Names of the ``P`` source populations, in column order.
    chromosomes: tuple[str, ...]
        Which panels to draw, among ``"autosome"`` and ``"X"``.
    allosome_label: str
        Name used for the X chromosome in panel titles.
    n_lineages: int | None
        Number of lineages per panel. If None, chosen separately for each panel by
        :func:`choose_n_lineages` (the fewest, between 100 and 1000, that show every
        source population present on that chromosome).
    title: str | None
        Optional figure title.
    figsize: tuple[float, float] | None
        Figure size. Defaults to a generation-axis length that grows with the number of generations, and
        5 (vertical) or 4.5 (horizontal) inches per panel.
    orientation: str
        ``"vertical"`` (past at the top) or ``"horizontal"`` (past on the right).
    **kwargs
        Passed to :func:`plot_lineage_flows` (colours, ``alpha``, ``pop_order``, ...).

    Returns
    -------
    (fig, axes)
    """
    n_gen = np.shape(migration_matrix_f)[0]
    n_panels = len(chromosomes)
    vertical = orientation == "vertical"
    gen_length = max(6.5, min(13.0, 0.35 * n_gen + 4.5))
    if figsize is None:
        figsize = (5.0 * n_panels, gen_length) if vertical else (gen_length, 4.5 * n_panels)
    if vertical:
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True, squeeze=False)
        axes = axes[0, :]
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, squeeze=False)
        axes = axes[:, 0]
    for ax, chrom in zip(axes, chromosomes):
        flows = compute_lineage_flows(migration_matrix_f, migration_matrix_m, chrom)
        n = n_lineages if n_lineages is not None else choose_n_lineages(flows)
        name = "Autosomes" if chrom == "autosome" else f"{allosome_label} chromosome"
        plot_lineage_flows(flows, pop_labels, n_lineages=n, ax=ax, title=f"{name} ({n} lineages)",
                           orientation=orientation, **kwargs)
    if vertical:
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
    else:
        for ax in axes[:-1]:
            ax.set_xlabel("")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_lineages_by_population(models: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
                                chromosomes: str | tuple[str, ...] = "autosome", n_lineages: int | None = None,
                                title: str | None = None, figsize: tuple[float, float] | None = None,
                                panel_width: float = 3.2, allosome_label: str = "X",
                                panel_titles: dict[str, str] | None = None,
                                **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """
    Vertical lineage panels, one column per admixed population and one row per chromosome type, all on a
    common generation axis.

    Parameters
    ----------
    models: dict
        ``{admixed population name: (migration_matrix_f, migration_matrix_m, source labels)}``, in column order.
    chromosomes: str | tuple[str, ...]
        ``"autosome"``, ``"X"``, or several of them (one row each, top to bottom).
    n_lineages: int | None
        Number of lineages per panel; if None, chosen separately for each panel by
        :func:`choose_n_lineages`.
    title: str | None
        Optional figure title.
    figsize: tuple[float, float] | None
        Figure size. Defaults to ``panel_width`` inches per column, and a row height growing with the number
        of generations.
    allosome_label: str
        Name used for the X chromosome in row labels.
    panel_titles: dict[str, str] | None
        ``{admixed population name: title}`` of each column's panels (e.g. the model name and its likelihood).
        Populations not in it get the default title, their name and number of lineages.
    **kwargs
        Passed to :func:`plot_lineage_flows`. Source colours are shared across panels,
        assigned in order of first appearance, unless ``colors`` is given as a ``{label: color}`` dict.

    Returns
    -------
    (fig, axes)
        ``axes`` has shape ``(len(chromosomes), len(models))``.
    """
    if isinstance(chromosomes, str):
        chromosomes = (chromosomes,)
    # Common generation axis: pad shorter matrices with zero rows (no migration in those generations).
    n_gen = max(np.shape(f)[0] for f, _, _ in models.values())

    def pad(mat):
        mat = np.asarray(mat, dtype=float)
        return np.vstack([mat, np.zeros((n_gen - mat.shape[0], mat.shape[1]))])

    if "colors" not in kwargs:
        all_labels = list(dict.fromkeys(label for _, _, labels in models.values() for label in labels))
        kwargs["colors"] = get_population_colors(all_labels)
    n_rows, n_cols = len(chromosomes), len(models)
    row_height = max(6.5, min(13.0, 0.35 * n_gen + 4.5)) * (1 if n_rows == 1 else 0.8)
    if figsize is None:
        figsize = (panel_width * n_cols, row_height * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, squeeze=False)
    for row, chrom in enumerate(chromosomes):
        chrom_name = "Autosomes" if chrom == "autosome" else f"{allosome_label} chromosome"
        for ax, (pop, (mf, mm, labels)) in zip(axes[row], models.items()):
            flows = compute_lineage_flows(pad(mf), pad(mm), chrom)
            n = n_lineages if n_lineages is not None else choose_n_lineages(flows)
            plot_lineage_flows(flows, labels, n_lineages=n, ax=ax, title=(panel_titles or {}).get(pop, f"{pop} ({n} lineages)"),
                               admixed_label=pop, orientation="vertical", **kwargs)
        for ax in axes[row, 1:]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if n_rows > 1:
            axes[row, 0].set_ylabel(f"{chrom_name}\nGenerations ago")
    if title:
        fig.suptitle(title)
        # Keep ~0.4 inch free at the top for the title: tight_layout ignores the source labels above the axes.
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    else:
        fig.tight_layout()
    return fig, axes
