#!/usr/bin/env python

"""
fancyplotting.py -- make nice plots of tracts output.

It mimics for the most part the output of fancyplotting.nb, but additionally
provides a command-line interface and is generally more reusable that the
originally-bundled Mathematica code.

fancyplotting.py optionally uses seaborn or brewer2mpl if those packages are
present in order to use their color palettes and otherwise make the plots look
prettier. It is recommended -- although not necessary -- to install both of
them.
"""

from __future__ import print_function

# We use semantic versioning. See http://semver.org/
__version__ = '0.0.0.1'

import os.path as path

from FancyPlot import FancyPlot

import sys

eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

# Suffixing function factory: create a function that suffixes the supplied
# string to its argument.
suf = lambda s: lambda name: name + s

class CLIError(Exception):
    """ The class of errors that can arise in parsing the command line
        arguments.
    """
    pass

_usage = [
        "./fancyplotting.py -- create a nice visualization of Tracts output.",
        "usage: ./fancyplotting.py [--input-dir INDIR] [--output-dir OUTDIR]",
        "[--colors COLORS]"
        "    --name NAME --population-tags TAGS [--plot-format FMT] [--overwrite]",
        "    [--no-legend]",
        "       ./fancyplotting.py --help",
        "",
        "The output of tracts follows a particular naming convention. Each ",
        "experiment has a name `NAME` onto which is suffixed various endings ",
        "depending on the type of output data. The four that are needed by ",
        "fancyplotting.py are:",
        " * NAME_bins",
        " * NAME_dat",
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
        "colors can be specified by the --color flag. colors are comma-separated.",
        "They must be named colors from matplotlib"
        
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

    names = None
    pop_names = None
    plot_format = "pdf"
    overwrite_plot = False
    input_dir = "."
    output_dir = "."
    with_legend = True
    colors=None
    try:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            n = lambda: sys.argv[i+1]
            if arg == "--names" or arg == "--name":
                names = n().split(',')
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
            elif arg == "--colors": #colors matching the previous populations
                colors= n().split(',')
                i += 1
            elif arg == "--plot-format":
                plot_format = n()
                i += 1
            elif arg == "--overwrite":
                overwrite_plot = True
            elif arg == "--no-legend":
                with_legend = False

                
            elif arg == "--help":
                _show_usage()
                sys.exit(0)
            else:
                raise CLIError("unrecognized command line argument %s" % arg)
            i += 1

        def check_arg(arg_name, arg_value):
            if arg_value is None:
                raise CLIError("missing mandatory argument %s" % arg_name)

        check_arg("--names", names)
        check_arg("--population-tags", pop_names)
        check_arg("--plot-format", plot_format)
        check_arg("--input-dir", input_dir)
        check_arg("--output-dir", output_dir)
    except CLIError as e:
        eprint(e, end='\n\n')
        _show_usage()
        sys.exit(1)
    except IndexError:
        eprint("unexpected end of command line arguments", end='\n\n')
        _show_usage()
        sys.exit(1)

    paths = {}

    common_name = names[0]

    for s in ['bins', 'dat']:
        paths[s] = path.join(input_dir, common_name + '_' + s)


    paths['preds'] = [
            path.join(input_dir, name + '_pred')
            for name in names]

    fp = FancyPlot.load(paths['bins'], paths['dat'], paths['preds'],
            pop_names, names)

    fig = fp.make_figure()

    ### Save the figure ###
    #######################

    p = path.join(output_dir, "%s_plot.%s" % (common_name, plot_format))

    if not overwrite_plot: # if we care about preserving existing plots
        i = 1
        while path.exists(p):
            p = path.join(output_dir, "%s_plot.%d.%s" % (common_name, i, plot_format))
            i += 1
    else:
        if path.exists(p):
            print("Notice: overwrote existing plot,", p)

    fig.savefig(p)
