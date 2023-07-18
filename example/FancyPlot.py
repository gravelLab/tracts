#!/usr/bin/env python

# We use semantic versioning. See http://semver.org/
__version__ = "0.0.0.1"

import numpy as np
import matplotlib.pyplot as plt
import operator as op
import os.path as path
import scipy
from scipy.stats import poisson
from collections import namedtuple

import sys

#################
### Constants ###
#################

# Parameter controlling how 'wide' the distribution should be, used to infer
# the size of the distribution when drawing the plot
alpha = 0.3173105078629141

#########################################################
### Higher-order functions and combinators used later ###
#########################################################


def with_file(fun, path, mode="r"):
    """ Run a function on the handle that results from opening the file
        identified by the given path with the given mode.
        When the function completes, the file handle is automatically closed.
        Resource cleanup is ensured in the case of exceptions.
    """
    with open(path, mode) as handle:
        return fun(handle)


# To parse a TSV file of floats given its path.
def parse_tsv(path):
    def fun(handle):
        return list(map(lambda line: list(map(float, line.strip().split("\t"))), handle))

    return with_file(fun, path,)


#######################################################################
### Functions for finding out the dispersion intervals for the plot ###
#######################################################################


def find_bounds(mean, alpha):
    """ Find both the lower and upper bounds for a given mean value and
        dispersion parameter in a poisson distribution.
    """
    fun = lambda i: poisson.cdf(i, mean)

    upper = None
    lower = None
    i = 0
    while True:
        if upper is None and fun(i) > 1 - alpha / 2.0:
            upper = i
        if lower is None and fun(i) > alpha / 2.0:
            lower = i
        if upper is not None and lower is not None:
            return lower, upper
        i += 1


#####################################################
### Classes that represent the data to be plotted ###
#####################################################


class Theory(object):
    """ Represents an inference as made by tracts for a single population. """

    @staticmethod
    def load(path, bins, model_name, pop_names=[""]):
        """ From a '_pred' file as created by tracts, generate a list of Theory
            objects.
        """
        theories = parse_tsv(path)

        return [
            Theory(bins, t, model_name, name) for t, name in zip(theories, pop_names)
        ]

    def __init__(self, bins, theory, model_name, pop_name):
        self.theory = theory
        self.model_name = model_name
        self.pop_name = pop_name
        self.bins = bins

        self.boundary = [
            find_bounds(expected_value, alpha)
            for expected_value in (np.array(self.theory) + 1e-9)
        ]

        Y1, Y2 = zip(*self.boundary)

        self.X = bins
        self.Y1 = Y1
        self.Y2 = Y2

    def draw(self, ax, color, fill_alpha=0.2, bin_scale=1.0):
        label = self.pop_name + " " + self.model_name.replace("_", "-") + " model"
        ax.fill_between(
            bin_scale * self.X[:-1],
            nonzero(self.Y1)[:-1],
            nonzero(self.Y2)[:-1],
            interpolate=False,
            alpha=fill_alpha,
            color=color,
        )
        ax.plot(bin_scale * self.X[:-1], self.theory[:-1], color=color, label=label)


class Population:
    """ Represents the data used by tracts to perform an inference. """

    @staticmethod
    def load(dat_path, bins, names):
        data = parse_tsv(dat_path)

        return [Population(bins, d, name) for name, d in zip(names, data)]

    def __init__(self, bins, data, name):
        self.bins = bins
        self.data = data
        self.name = name
        self.theories = []

    def add_theory(self, t):
        self.theories.append(t)
        return self

    def draw(self, ax, base_color, theory_colors, bin_scale=1.0):
        for theory, color in zip(self.theories, theory_colors):
            theory.draw(ax, color, bin_scale=bin_scale)

        ax.scatter(
            bin_scale * self.bins[:-1],
            self.data[:-1],
            color=base_color,
            label=(self.name + " data").replace("_", "-"),
        )


class FancyPlot:
    """ A fancy plot is a collection of Population objects, each of which is
        populated with as many Theory objects as there are tracts runs to
        compare.
    """

    @staticmethod
    def load(bins_path, dat_path, pred_paths, pop_names, model_names):
        """ Load a FancyPlot from a set of files, specifically the output from
            tracts.

            Arguments:
                bins_path (string):
                    The path to the '_bins' file.
                dat_path (string):
                    The path to the '_dat' file.
                pred_paths (list of strings):
                    Each element of pred_paths must be a path to a '_pred'
                    file. Each '_pred' file describes an inference made by
                    tracts for each population.
                pop_names (list of strings):
                    The names of the populations over which the inference was
                    made.
                model_names (list of strings):
                    the names of the models used for the inferences. There
                    should be as many model names as there are pred_paths.
        """
        bins = np.array(parse_tsv(bins_path)[0])
        pops = Population.load(dat_path, bins, pop_names)

        # each element of 'theories' is actually a list of Theory objects,
        # which must be ordered as in pop_names.
        theories = [
            Theory.load(p, bins, name, pop_names)
            for p, name in zip(pred_paths, model_names)
        ]

        # hence, we transpose the theories list such that each element of
        # 'theories_t' is a list of all the Theory objects associated with the
        # corresponding population
        theories_t = zip(*theories)

        for p, ts in zip(pops, theories_t):
            for t in ts:
                p.add_theory(t)

        return FancyPlot(pops)

    def __init__(self, pops=[]):
        self.populations = pops

        self.title = "Number of tracts vs tract length (cM)"
        self.xlabel = "tract length (cM)"
        self.ylabel = "number of tracts"
        self.legend = True

        self.top_limit = None
        self.right_limit = None
        self.bottom_limit = 0.92
        self.left_limit = 0.0

        self.bin_scale = 100

    def __getitem__(self, k):
        return self.populations[k]

    def add_population(self, pop):
        self.populations.append(pop)
        return self

    def choose_colors(self):
        """ Based on the data to be shown by this FancyPlot, choose some nice
            colors. A tuple (population colors, theory colors) is returned such
            that its components can be supplied directly to FancyPlot.draw's
            second and third arguments.
        """
        N = len(self.populations)
        M = len(self.populations[0].theories)
        nh_size = 1.0 / N
        S = nh_size / 2
        cmap = plt.get_cmap()

        pop_colors = [
            (k, cmap(k))
            for i, k in enumerate(np.linspace(0.0, 1.0, 2 * N + 1))
            if i % 2 != 0
        ]

        model_colors = [
            [
                (k, cmap(k))
                for i, k in enumerate(np.linspace(j - S, j + S, 2 * M + 1))
                if i % 2 != 0
            ]
            for j, _ in pop_colors
        ]

        fst = lambda t: t[1]

        return (map(fst, pop_colors), map(lambda cs: map(fst, cs), model_colors))

    def draw(self, ax, pop_colors, theory_colors):
        """ Draw this FancyPlot onto an existing set of axes with the given
            colors.
        """
        for pop, color, t_colors in zip(self.populations, pop_colors, theory_colors):
            pop.draw(ax, color, t_colors, bin_scale=self.bin_scale)

    def make_figure(self, pop_colors=None, theory_colors=None, automatic_colors=True):
        """ Create a figure from this FancyPlot.

            Arguments:
                pop_colors (list of RGBA tuples, default: None):
                    The colors with which each population should be drawn,
                    respectively. There should be at least as many colors in
                    pop_colors as there are populations in this FancyPlot.
                theory_colors (list of lists of RGBA tuples, default: None):
                    The colors with which each theory should be drawn for each
                    population. The first level of lists must contain as many
                    elements as there are population in this FancyPlot. The
                    inner lists must contain at least as many elements as the
                    corresponding Population has Theory objects.
                automatic_colors (boolean, default: True):
                    When True, colors will be chosen automatically for the data
                    represented by this FancyPlot.
        """
        if automatic_colors:
            if pop_colors is not None or theory_colors is not None:
                raise ValueError(
                    "When automatic_colors is True, both "
                    "pop_colors and theory_colors must be given as None."
                )
        else:
            if pop_colors is None or theory_colors is None:
                raise ValueError(
                    "When automatic_colors is False, both "
                    "pop_colors and theory_colors must be given non-None "
                    "values."
                )

        fig = plt.figure()
        fig.suptitle(self.title)
        ax = fig.add_subplot(1, 1, 1)

        # Set the axis labels
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.set_yscale("log")

        if automatic_colors:
            pop_colors, theory_colors = self.choose_colors()

        self.draw(ax, pop_colors, theory_colors)

        if self.legend:
            ax.legend()

        ax.set_ylim(bottom=self.bottom_limit, top=self.top_limit)
        ax.set_xlim(left=self.left_limit, right=self.right_limit)

        return fig


#####################
### Miscellaneous ###
#####################


def nonzero(seq, cutoff=1e-9, offset=1e-9):
    """ Any zeroes (values smaller than the cutoff) found in the given sequence
        of numbers are offset by the given offset.
    """
    return [x + offset if x < cutoff else x for x in seq]
