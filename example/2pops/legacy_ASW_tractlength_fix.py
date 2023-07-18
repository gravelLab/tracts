#!/usr/bin/env python

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
import tracts
import tracts.legacy_models.models_2pop as models_2pop
import numpy

directory = "./G10/"


# number of short tract bins not used in inference.
cutoff = 2

# number of repetitions for each model (to ensure convergence of optimization)
rep_pp = 2
rep_pp_px = 2

# only trio individuals
names = [
    "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
    "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
    "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
    "NA20357", "NA20356"
]

chroms = ['%d' % (i,) for i in range(1, 23)]

# load the population
pop = tracts.population(
    names=names, fname=(directory, "", ".bed"), selectchrom=chroms)
(bins, data) = pop.get_global_tractlengths(npts=50)


# choose order of populations and sort data accordingly
labels = ['EUR', 'AFR']
data = [data[poplab] for poplab in labels]

# we're fixing the global ancestry proportions, so we only need one parameter
startparams = numpy.array([0.0683211])
# (initial admixture time). Times are measured in units of hundred generations
# (i.e., multiply the number by 100 to get the time in generations). The reason
# is that some python optimizers do a poor job when the parameters (time and
# ancestry proportions) are of different magnitudes.  you can also look at the
# "_mig" output file for a generation-by-generation breakdown of the migration
# rates.

Ls = pop.Ls
nind = pop.nind

# calculate the proportion of ancestry in each individual
bypopfrac = [[] for i in range(len(labels))]
for ind in pop.indivs:
    for ii, poplab in enumerate(labels):
        bypopfrac[ii].append(ind.ancestryProps([poplab]))

props = numpy.mean(bypopfrac, axis=1).flatten()

# we compare two models; single pulse versus two European pulses.
func = models_2pop.pp_fix
bound = models_2pop.outofbounds_pp_fix

func2 = models_2pop.pp_px_fix
bound2 = models_2pop.outofbounds_pp_px_fix
# ((tstart,t2,nuEu_prop))
# tstart:     start time,
# t2:         time of second migration,
# nuEu_prop : proportion at second migration (proportion at first migration is
#             fixed by total ancestry proportion)
#
# Note: times are measured in units of 100 generations (see above)

# give two different starting conditions, with one starting near the
# single-pulse model
startparams2 = numpy.array([0.107152,  0.0438957,  0.051725])
startparams2p = numpy.array([0.07152,  0.03,  1e-8])

optmod = tracts.demographic_model(func(startparams, props))


def randomize(arr, scale=2):
    """ Scale each element of an array by some random factor between zero and a
        limit (default: 2), capping the result at 1.
    """
    return [min(i, 1) for i in scale * numpy.random.random(arr.shape) * arr]

liks_orig_pp = []
maxlik = -1e18
startrand = startparams
for i in range(rep_pp):
    xopt = tracts.optimize_cob_fracs2(
        startrand, bins, Ls, data, nind, func, props, outofbounds_fun=bound,
        cutoff=cutoff, epsilon=1e-2)
    # optimize_cob_fracs2 takes one additional parameter: the proportion of
    # each ancestry that will be used to fix the parameters.
    optmodlocal = tracts.demographic_model(func(xopt, props))
    loclik = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
    if loclik > maxlik:
        optmod = optmodlocal
        optpars = xopt
    liks_orig_pp.append(loclik)

    startrand = randomize(startparams)


print("likelihoods found: ", liks_orig_pp)

liks_orig_pp_px = []
startrand2 = startparams2
maxlik2 = -1e18


for i in range(0, rep_pp_px):
    xopt2 = tracts.optimize_cob_fracs2(
        startrand2, bins, Ls, data, nind, func2, props, outofbounds_fun=bound2,
        cutoff=cutoff, epsilon=1e-2)
    try:
        optmod2loc = tracts.demographic_model(func2(xopt2, props))
        loclik = optmod2loc.loglik(bins, Ls, data, nind, cutoff=cutoff)
        if loclik > maxlik2:
            optmod2 = optmod2loc
            optpars2 = xopt2
    except:
        print("convergence error")
        loclik = -1e8
    liks_orig_pp_px.append(loclik)
    startrand2 = randomize(startparams2)

lik1 = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
lik2 = optmod2.loglik(bins, Ls, data, nind, cutoff=cutoff)

#
# Save the data to file for external plotting, model 1

outdir = "./out"

with open(outdir + "_bins", 'w') as fbins:
    fbins.write("\t".join(map(str, bins)))

with open(outdir + "_dat", 'w') as fdat:
    for popnum in range(len(data)):
        fdat.write("\t".join(map(str, data[popnum])) + "\n")

with open(outdir + "_mig", 'w') as fmig:
    for line in optmod.mig:
        fmig.write("\t".join(map(str, line)) + "\n")

with open(outdir + "_pred", 'w') as fpred:
    for popnum in range(len(data)):
        fpred.write(
            "\t".join(map(
                str,
                pop.nind * numpy.array(optmod.expectperbin(Ls, popnum, bins))))
            + "\n")
with open(outdir + "_pars", 'w') as fpars:
    fpars.write("\t".join(map(str, optpars)) + "\n")

#
# The first two files will be identical across models. We save an extra
# copy to facilitate the plotting.

outdir = "./out2"

with open(outdir + "_bins", 'w') as fbins:
    fbins.write("\t".join(map(str, bins)))

with open(outdir + "_dat", 'w') as fdat:
    for popnum in range(len(data)):
        fdat.write("\t".join(map(str, data[popnum])) + "\n")

with open(outdir + "_mig", 'w') as fmig2:
    for line in optmod2.mig:
        fmig2.write("\t".join(map(str, line)) + "\n")

with open(outdir + "_pred", 'w') as fpred2:
    for popnum in range(len(data)):
        fpred2.write("\t".join(map(
            str,
            pop.nind * numpy.array(optmod2.expectperbin(Ls, popnum, bins))))
        + "\n")

with open(outdir + "_pars", 'w') as fpars2:
    fpars2.write("\t".join(map(str, optpars2)) + "\n")

