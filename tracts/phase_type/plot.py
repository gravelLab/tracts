"""
Plotting helpers for Phase-Type tract length densities and histograms.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#: Default color, legend label and (for hybrid-pedigree models) "shadow" base-model key for each
#: recognized model key. A hybrid-pedigree curve is drawn twice: once in its own color, and once
#: dashed in its base model's color, to make it easy to see how closely it tracks that model.
DEFAULT_MODEL_STYLES = {
    "M": dict(color="gray", label="Monoecious (M)"),
    "DF": dict(color="blue", label="Dioecious Fine (DF)"),
    "DC": dict(color="green", label="Dioecious Coarse (DC)"),
    "H_DF": dict(color="orange", label="H-DF (TP = 2)", shadow="DF"),
    "H_DC": dict(color="orange", label="H-DC (TP = 2)", shadow="DC"),
}
_SHADOW_LINESTYLE = (2, (2, 2))

def _resolve_styles(model_styles):
    if not model_styles:
        return DEFAULT_MODEL_STYLES
    return {**DEFAULT_MODEL_STYLES, **model_styles}

def _new_axes(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def _finalize(fig, ax, xlabel, ylabel, xlim, handles, labels, legend):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="both", which="major", labelsize=10)

    if legend:
        fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.55, -0.2), ncol=2, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()



def plot_tractlength_density(bins: npt.ArrayLike, curves: dict[str, npt.ArrayLike], L: float,
                            xlabel: str="Tract length", ylabel: str="Density", ax: plt.Axes=None,
                            legend: bool=True, model_styles: dict | None=None, figsize: tuple[float, float]=(5, 4)):
    """
    Plots one or more Phase-Type tract length densities on the same axes.

    Reproduces the formatting used throughout the ``phase_type_models`` tutorial: a
    hybrid-pedigree model (key ``"H_DF"`` or ``"H_DC"``) is drawn twice -- once in its own
    color, once dashed in its base model's color -- and grouped into a single legend entry.
    Since the density can be discontinuous at the chromosome boundary ``L``, the curve is not
    drawn connected to that last bin; instead, the boundary value is shown as a marker (hollow
    for the preceding bin's value, filled for the boundary value itself).

    Parameters
    ----------
    bins: npt.ArrayLike
        Bin edges shared by every curve, as returned by
        ``tractlength_histogram_windowed(..., density=True)``.
    curves: dict[str, npt.ArrayLike]
        Maps a model key to its density array (same length as ``bins``). Recognized keys are
        ``"M"``, ``"DF"``, ``"DC"``, ``"H_DF"``, ``"H_DC"`` (see ``model_styles`` to add
        others); only the keys relevant to the plot need to be included, e.g. omit ``"M"`` for
        X-chromosome admixture, or pass a single key for a single-model plot.
    L: float
        The chromosome length, i.e. the bin edge to render as a boundary marker instead of a
        connected line segment, and to set the x-axis limit from.
    xlabel: str
        The x-axis label, e.g. ``"Tract length on the second chromosome"``.
    ylabel: str
        The y-axis label. Defaults to ``"Density"``.
    ax: matplotlib.axes.Axes | None
        Axes to draw on. A new figure and axes (of size ``figsize``) are created if not given.
    legend: bool
        Whether to add a legend below the axes.
    model_styles: dict | None
        Additional or overriding entries merged onto ``DEFAULT_MODEL_STYLES``, for model keys
        beyond the default five, or to customize colors/labels.
    figsize: tuple[float, float]
        Figure size used when ``ax`` is not given. Defaults to ``(5, 4)``.

    Returns
    -------
    tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]
    """
    styles = _resolve_styles(model_styles)
    bins = np.asarray(bins)
    is_boundary = bins == L
    boundary_idx = int(np.where(is_boundary)[0][0]) if is_boundary.any() else None

    fig, ax = _new_axes(ax, figsize)

    handles, labels = [], []
    for key, values in curves.items():
        if key not in styles:
            raise KeyError(f"Unknown model key {key!r}; expected one of {sorted(DEFAULT_MODEL_STYLES)}, or add it to model_styles.")
        style = styles[key]
        values = np.asarray(values)
        color, label, shadow_key = style["color"], style["label"], style.get("shadow")

        if boundary_idx is None:
            x, y = bins, values
        else:
            x, y = np.delete(bins, boundary_idx), np.delete(values, boundary_idx)

        line, = ax.plot(x, y, color=color, linestyle="-", linewidth=2, label=label, zorder=1)
        handle = line

        if shadow_key is not None:
            shadow_color = styles[shadow_key]["color"]
            shadow_line, = ax.plot(x, y, color=shadow_color, linestyle=_SHADOW_LINESTYLE, linewidth=2, zorder=1)
            handle = (line, shadow_line)

        if boundary_idx is not None:
            before, at = values[boundary_idx - 1], values[boundary_idx]
            ax.scatter(L, before, edgecolor=color, s=30, marker="o", facecolor="white", alpha=1, zorder=3)
            ax.scatter(L, at, edgecolor=color, s=30, marker="o", facecolor=color, zorder=4)
            if shadow_key is not None:
                ax.scatter(L, at, edgecolor=color, s=15, marker="o", facecolor=shadow_color, zorder=4)

        handles.append(handle)
        labels.append(label)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _finalize(fig, ax, xlabel, ylabel, (0, L + 0.1), handles, labels, legend)
    return fig, ax



def plot_tractlength_histogram(bins: npt.ArrayLike, curves: dict[str, npt.ArrayLike], xlabel: str="Tract length",
                                ylabel: str="Expected number of tracts per interval", ax: plt.Axes=None,
                                legend: bool=True, model_styles: dict | None=None, figsize: tuple[float, float]=(5, 4),
                                L: float | None=None):
    """
    Plots one or more Phase-Type tract length histograms (as step curves) on the same axes.

    Reproduces the formatting used throughout the ``phase_type_models`` tutorial: a
    hybrid-pedigree model (key ``"H_DF"`` or ``"H_DC"``) is drawn twice -- once in its own
    color, once dashed in its base model's color -- and grouped into a single legend entry.

    Parameters
    ----------
    bins: npt.ArrayLike
        Bin edges shared by every curve, as returned by
        ``tractlength_histogram_windowed(..., density=False)``; ``bins[:-1]`` is used as the
        step x-positions, matching the array of counts in ``curves``.
    curves: dict[str, npt.ArrayLike]
        Maps a model key to its histogram array (length ``len(bins) - 1``). See
        ``plot_tractlength_density`` for the recognized keys.
    xlabel: str
        The x-axis label, e.g. ``"Tract length on the second chromosome"``.
    ylabel: str
        The y-axis label. Defaults to ``"Expected number of tracts per interval"``.
    ax: matplotlib.axes.Axes | None
        Axes to draw on. A new figure and axes (of size ``figsize``) are created if not given.
    legend: bool
        Whether to add a legend below the axes.
    model_styles: dict | None
        Additional or overriding entries merged onto ``DEFAULT_MODEL_STYLES``.
    figsize: tuple[float, float]
        Figure size used when ``ax`` is not given. Defaults to ``(5, 4)``.
    L: float | None
        If given, the chromosome length, used only to set the x-axis limit to ``(0, L + 0.1)``
        (matching ``plot_tractlength_density``). Left to matplotlib's default if not given.

    Returns
    -------
    tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]
    """
    styles = _resolve_styles(model_styles)
    bins = np.asarray(bins)

    fig, ax = _new_axes(ax, figsize)

    handles, labels = [], []
    for key, values in curves.items():
        if key not in styles:
            raise KeyError(f"Unknown model key {key!r}; expected one of {sorted(DEFAULT_MODEL_STYLES)}, or add it to model_styles.")
        style = styles[key]
        color, label, shadow_key = style["color"], style["label"], style.get("shadow")

        line, = ax.step(bins[:-1], values, color=color, linestyle="-", linewidth=2, label=label)
        handle = line

        if shadow_key is not None:
            shadow_color = styles[shadow_key]["color"]
            shadow_line, = ax.step(bins[:-1], values, color=shadow_color, linestyle=_SHADOW_LINESTYLE, linewidth=2)
            handle = (line, shadow_line)

        handles.append(handle)
        labels.append(label)

    xlim = (0, L + 0.1) if L is not None else None
    _finalize(fig, ax, xlabel, ylabel, xlim, handles, labels, legend)
    return fig, ax
