#!/usr/bin/env python

import os
import sys
import scipy
tractspath = "../.."  # the path to tracts if not in your default pythonpath
sys.path.append(tractspath)
import tracts
from tracts.legacy_models import models_3pop
import numpy as np
import pylab

from warnings import warn

# optimization method. Here we use brute force; other options are implemented
# in tracts but are not implemented in this driver script.
method = "brute"

# demographic models to use

# ppx_xxp_fix has an initial pulse of migration from populations 0 and 1,
# followed by a pulse from population 2. "fix" referes to the fact that the
# migration rates in the model are fixed to the observed global ancestry
# proportions--then, we only have to optimize the timing of the migrations.
func = models_3pop.ppx_xxp_fix

# this function keeps track of whether parameters are in a "forbidden" region:
# whether mproportions are between 0 and 1, times positive, etc.
bound = models_3pop.outofbounds_ppx_xxp_fix

# defines the values of parameters to loop over in the brute force
# step. Times are in units of 100 generations: the start time will
# be between 7 and 12 generations, and the timing of the second
# migration between 1 and 12 generations in steps of 1 generation.
# After the initial search there is a "refining" step
slices = (slice(.07, .14, .01), slice(0.01, .12, .01))

# absolute bounds that parameters are not allowed to cross: times
# must be between 0 and 100 generations.
bounds = [(0, 1), (0, 1)]

# choose order of populations: the labels in our ancestry files are
# strings, and we need to tell tracts which string correspond to which
# population in the model. Here the population labels are (somewhat
# confusingly) numbers that do not match the order in the population.
# "labels" will tell tracts that model population 0 has label '0', model
# population 1 has label '2', and model population 2 has label '1' in the
# local ancestry files.
labels = ['0', '2', '1']

# directories in which to read and write
directory = "PUR/"
outdir = "PUR/output/"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# string between individual label and haploid chromosome id in input file
inter = "_anc"
# string at the end of input file. Note that the file should end in ".bed"
end = "_cM.bed"

usage = "python taino_ppx_xxp.py 1 to run without bootstrap \n"\
        "python taino_ppx_xxp.py A [B] if no argument provided, runs "\
        "first 100 bootstrap instances. If A only, A is the number of "\
        "instances. If A and B are provided, it is a range between A "\
        "and B (A must be smaller). seeds ensure that runs with a given "\
        "number are reproducible.\n"\
        "bootstrap instance 0 is the original dataset."

args = sys.argv
print(args)

if len(args) == 1:
    runboots = range(100) # run all instances
elif len(args) == 2:
    runboots = range(int(args[1])) # run just the specified number
elif len(args) == 3:
    if int(args[1]) >= int(args[2]):
        print(usage)
        sys.exit(1)
    else:
        runboots = range(int(args[1]), int(args[2]))

# Get a list of all individuals in directory.
_files = os.listdir(directory)
files = [file
        for file in _files
        if file.split('.')[-1] == "bed"]  # only consider bed files

# Get unique individual labels
names = list(set(file.split('_')[0] for file in files))

if len(_files) != len(files):
    warn("some files in the bed directory were ignored, since they do not "
            "end with `.bed`.")

# Load the population using the population class's constructor. It
# automatically iterates over individuals and haploid copies (labeled _A"
# and "_B" by default
pop = tracts.population(names=names, fname=(directory, inter, end))

# Rather than creating a new population for each bootstrap instance, we
# just replace the list of individuals to iterate over. We need to save a
# copy of the initial list of individuals to do this!
indivs = pop.indivs

def bootsamp(num):
    #generates a list of positions of the samples to pick in a bootstrap 
    return np.random.choice(range(num),replace=True,size=num)

# iterate over bootstrap instances. Iteration 0 is the un-bootstrapped value
for bootnum in runboots:
    # Use a seed for reproducibility.
    np.random.seed(seed=bootnum)

    if bootnum != 0:
        # draw random sample.
        bootorder = bootsamp(len(indivs))

        # to perform the bootstrap, we change the list of individuals in "pop"
        # to match our bootstrapped sample. We will then use that list to
        # generate the histogram of tract lengths
        indivs2 = [indivs[i] for i in bootorder]
        pop.indivs = indivs2
    else:
        bootorder = range(len(indivs))

    # generate the histogram of tract lengths
    (bins, data) = pop.get_global_tractlengths(npts=50)
    print("booted data sample", data['0'][1:10])

    data = [data[poplab] for poplab in labels]

    bypopfrac = [[] for i in range(len(labels))]
    # Calculate ancestry proportions
    for ind in pop.indivs:
        for ii, poplab in enumerate(labels):
            bypopfrac[ii].append(ind.ancestryProps(poplab))

    props = np.mean(bypopfrac, axis=1).flatten()

    Ls = pop.Ls
    nind = pop.nind

    cutoff = 2

    def randomize(arr, scale=2):
        # takes an array and multiplies every element by a factor between 0 and
        # 2, uniformly. caps at 1.
        return map(lambda i: min(i, 1), scale * np.random.random(arr.shape) * arr)

    xopt = tracts.optimize_brute_fracs2(
        bins, Ls, data, nind, func, props, slices, outofbounds_fun=bound, cutoff=cutoff)
    print(xopt)
    optmod = tracts.demographic_model(func(xopt[0], props))
    optpars = xopt[0]
    liks = xopt[1]
    maxlik = optmod.loglik(bins, Ls, data, pop.nind, cutoff=cutoff)

    expects = []
    for popnum in range(len(data)):
        expects.append(optmod.expectperbin(Ls, popnum, bins))

    expects = nind * np.array(expects)

    outf = outdir + "boot%d_%2.2f" % (bootnum, maxlik,)

    fbins = open(outf + "_bins", 'w')
    fbins.write("\t".join(map(str, bins)))
    fbins.close()

    fliks = open(outf + "_liks", 'w')
    fliks.write("\t".join(map(str, liks)))
    fliks.close()

    fdat = open(outf + "_dat", 'w')
    for popnum in range(len(data)):
        fdat.write("\t".join(map(str, data[popnum])) + "\n")

    fdat.close()
    fmig = open(outf + "_mig", 'w')
    for line in optmod.mig:
        fmig.write("\t".join(map(str, line)) + "\n")

    fmig.close()
    fpred = open(outf + "_pred", 'w')
    for popnum in range(len(data)):
        fpred.write(
            "\t".join(map(str, pop.nind * np.array(optmod.expectperbin(Ls, popnum, bins)))) + "\n")
    fpred.close()

    fpars = open(outf + "_pars", 'w')

    fpars.write("\t".join(map(str, optpars)) + "\n")
    fpars.close()

    # bootstrap order in case we need to rerun/check something.
    ford = open(outf + "_ord", 'w')
    ford.write("\t".join(map(lambda i: "%d" % (i,), bootorder)))
    ford.close()
