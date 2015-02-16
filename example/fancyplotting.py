#!/usr/bin/env python

""" A Python module to beauty the functionality of the FancyPlotting.nb
    Mathematica code.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import operator as op
import os.path as path

from scipy.stats import poisson
from itertools import imap

import sys

colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0)]

#########################################################
### Higher-order functions and combinators used later ###
#########################################################

def with_file(fun, path, mode='r'):
    """ Run a function on the handle that results from opening the file
        identified by the given path with the given mode.
        When the function completes, the file handle is automatically closed.
        Resource cleanup is ensured in the case of exceptions.
    """
    with open(path, mode) as handle:
        return fun(handle)

def izip_with(fun, *args):
    """ zip two lists by applying a function to each resulting tuple.

        > list(izip_with(op.add, [1, 2, 3], [4, 5, 6]))
        [5, 7, 9]
    """
    return (fun(*t) for t in zip(*args))

# A strict version of zip_with
zip_with = lambda fun, *args: list(izip_with(fun, *args))

# Suffixing function factory: create a function that suffixes the supplied
# string to its argument.
suf = lambda s: lambda name: name + s

# Construct a higher-order function that applies its argument to the given
# constant. This is just a curried, flipped form of the identity function:
# apply x f = flip id
apply_to = lambda c: lambda f: f(c)

# Curry a function by fixing the given constant as the function's first
# parameter.
c2 = lambda f, c: lambda *args: f(c, *args)

# To parse a TSV file of floats given its path.
parse_tsv = \
        lambda path: with_file(
            lambda handle: map(
                lambda line: map(
                    float,
                    line.strip().split('\t')),
                handle),
            path)

# To efforlessly print to stderr in case of errors.
eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

# The multiplicative counterpart of sum.
product = lambda seq: reduce(op.mul, seq, 1)

#######################################################################
### Functions for finding out the dispersion intervals for the plot ###
#######################################################################

def lower_bound(mean, alpha):
    fun = poisson(mu=mean).cdf
    i = 0
    while True:
        if fun(i) > alpha/2.0:
            return max(i - 1, 0)
        i += 1

def upper_bound(mean, alpha):
    fun = poisson(mu=mean).cdf
    i = 0
    while True:
        if fun(i) > 1 - alpha/2.0:
            return i
        i += 1

def find_bounds(mean, alpha):
    """ Find both the lower and upper bounds for a given mean value and
        dispersion parameter in a poisson distribution.
    """
    return (lower_bound(mean, alpha), upper_bound(mean, alpha))

#########################
### Plotting function ###
#########################

def create_figure(plot_theories, boundaries, data, pop_names, colors):
    fig = plt.figure()
    fig.suptitle('Tract length (cM) versus number of tracts')
    ax = fig.add_subplot(1, 1, 1)

    # Set the limits on the axes
    ax.set_ylim(0.4, 1e3) # TODO get rid of the baked-in upper limit
    ax.set_xlim(0, 300)

    # Set the axis labels
    ax.set_ylabel('Tract length (cM)')
    ax.set_xlabel('Number of tracts')

    # Use logarithmic scale on the y-axis
    ax.set_yscale('log')

    # Zip together all the lists indexed by population number so we can
    # aggregate all the information relevant to a single population in each
    # iteration.
    for (theory, bounds, experimental_data, pop_name, color) in \
            zip(plot_theories, boundaries, data, pop_names, colors):
        # Transpose to get the list of lows and list of highs
        X, YS = zip(*bounds)
        # Transpose to get the list of x values and list of pairs (low, high).
        Y1, Y2 = zip(*YS)
        ax.fill_between(X, Y1, Y2,
                interpolate=False, alpha=0.2, color=color)
        ax.plot(*zip(*theory),
                color=color, label=pop_name + " model")
        ax.scatter(bins[:-1], experimental_data[:-1],
                color=color, label=pop_name + " data")

    ax.legend()

    return fig

#####################
### Miscellaneous ###
#####################

class CLIError(Exception):
    """ The class of errors that can arise in parsing the command line
        arguments.
    """
    pass

_usage = [
        "./fancyplotting.py -- create a nice visualization of Tracts output.",
        "usage: ./fancyplotting.py [--input-dir INDIR] [--output-dir OUTDIR]",
        "    --name NAME --population-tags TAGS [--plot-format FMT] [--overwrite]",
        "       ./fancyplotting.py --help",
        "",
        "The output of tracts follows a particular naming convention. Each ",
        "experiment has a name `NAME` onto which is suffixed various endings ",
        "depending on the type of output data. The four that are needed by ",
        "fancyplotting.py are:",
        " * NAME_bins",
        " * NAME_dat",
        " * NAME_mig",
        " * NAME_pred",
        "These files are searched for in the directory INDIR if specified. Else,",
        "they are searched for in the current working directory.",
        "Since these files to not include any labels for the populations",
        "(internally, there are merely numbered), friendly names must be given",
        "as a comma-separated list on the command line after the --population-tags",
        "switch, e.g. `--population-tags AFR,EUR`.",
        "",
        "fancyplotting.py uses Matplotlib to generate its plot, so it is advisable",
        "to use a matplotlibrc file to add additional style to the plot, to make it",
        "look really good. A sample matplotlibrc is bundled with this distribution.",
        "Furthermore, the output format of the plot can thus be any file type that",
        "Matplotlib can output. The default format is a PDF, which can easily be",
        "embedded into LaTeX documents, although you may want to use a PNG for",
        "distribution on the web.",
        "",
        "The generated plot is saved to OUTDIR, if it is provided. Else, the plot",
        "is saved to the current working directory. It's filename has the format",
        "NAME_plot.FMT. If a file with this name already exists and --overwrite",
        "is not used, then a fancyplotting.py will try NAME_plot.N.FMT where N are",
        "the integers starting at 1, tried in order until a free file name is found."
]

def _show_usage():
    for u in _usage:
        eprint(u)

####################
### Script entry ###
####################

if __name__ == "__main__":
    ### Parse command line arguments ###
    ####################################

    name = None
    pop_names = None
    plot_format = "pdf"
    overwrite_plot = False
    input_dir = "."
    output_dir = "."

    try:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            n = lambda: sys.argv[i+1]
            if arg == "--name":
                name = n()
                i += 1
            elif arg == "--input-dir":
                input_dir = n()
                i += 1
            elif arg == "--output-dir":
                output_dir = n()
                i += 1
            elif arg == "--population-tags":
                pop_names = n().split(',')
                i += 1
            elif arg == "--plot-format":
                plot_format = n()
                i += 1
            elif arg == "--overwrite":
                overwrite_plot = True
            elif arg == "--help":
                _show_usage()
                sys.exit(0)
            else:
                raise CLIError("unrecognized command line argument %s" % arg)
            i += 1

        def check_arg(arg_name, arg_value):
            if arg_value is None:
                raise CLIError("missing mandatory argument %s" % arg_name)

        check_arg("--name", name)
        check_arg("--population-tags", pop_names)
        check_arg("--plot-format", plot_format)
        check_arg("--input-dir", input_dir)
        check_arg("--output-dir", output_dir)
    except CLIError as e:
        eprint(e)
        sys.exit(1)
    except IndexError:
        eprint("unexpected end of command line arguments")
        sys.exit(1)


    ### Read in the data ###
    ########################

    # Make suffixing functions for each of the suffixes used by the simulation.
    path_makers = imap(suf, ["_bins", "_dat", "_mig", "_pred"])

    # Parse the data files
    input_prefix = path.join(input_dir, name)
    bins, data, migs, expects = [parse_tsv(p)
            for p in imap(apply_to(input_prefix), path_makers)]

    # since there's only one line in the bins file and units are in hundreds of
    # generations, we pick out element zero of bins and multiply each
    # constituent by 100.
    bins = 100 * np.array(bins[0])

    # The dimensions of the migration data give us the length of the migration
    # and the number of populations.
    migration_length, ancestral_pops_count = np.shape(migs)

    eprint("migration length:", migration_length)
    eprint("ancestral populations:", ancestral_pops_count) # good

    # For each population's expected values, zip on the bin data to form lists
    # of expected ancestry proportion at a given time.
    plot_theories = [zip(bins[:-1], expected[:-1])
            for expected in expects]

    # For the experimental data of each population, zip on the bin data.
    plot_data = [zip(bins[:-1], d[:-1]) for d in data]


    ### Calculate the boundaries for the model prediction ###
    #########################################################

    # parameter controlling how 'wide' the distribution should be
    alpha = 0.3173105078629141

    # For the theoretical prediction of each population, determine the lower
    # and upper bounds on the variability admitted by the theory.
    boundaries = [[(bin, find_bounds(expected_value, alpha))
        for bin, expected_value in (np.array(pt) + 1e-9)] # TODO why this offset?
        for pt in plot_theories]


    ### Make the figure ###
    #######################

    fig = create_figure(plot_theories, boundaries, data, pop_names, colors)


    ### Save the figure ###
    #######################

    p = path.join(output_dir, "%s_plot.%s" % (name, plot_format))
    if not overwrite_plot: # if we care about preserving existing plots
        i = 1
        while path.exists(p):
            p = path.join(output_dir, "%s_plot.%d.%s" % (name, i, plot_format))
            i += 1
    fig.savefig(p)
