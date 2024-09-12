#!/usr/bin/env python

import tracts
import tracts.legacy_models.models_2pop as models_2pop
import numpy
from legacy_ASW_data import *


def write_to_files(outdir):
    with open(outdir + "_bins", 'w') as fbins:
        fbins.write("\t".join(map(str, bins)))

    with open(outdir + "_dat", 'w') as fdat:
        for popnum in range(len(data)):
            fdat.write("\t".join(map(str, data[popnum])) + "\n")

    with open(outdir + "_mig", 'w') as fmig2:
        for line in optmod2.migration_matrix:
            fmig2.write("\t".join(map(str, line)) + "\n")

    with open(outdir + "_pred", 'w') as fpred2:
        for popnum in range(len(data)):
            fpred2.write("\t".join(map(
                str,
                pop.nind * numpy.array(optmod2.expectperbin(Ls, popnum, bins))))
                         + "\n")

    with open(outdir + "_pars", 'w') as fpars2:
        fpars2.write("\t".join(map(str, optpars2)) + "\n")


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
startparams2 = numpy.array([0.107152, 0.0438957, 0.051725])
startparams2p = numpy.array([0.07152, 0.03, 1e-8])

optmod = tracts.DemographicModel(func(startparams, props))


def randomize(arr, scale=2):
    """ Scale each element of an array by some random factor between zero and a
        limit (default: 2), capping the result at 1.
    """
    return [min(i, 1) for i in scale * numpy.random.random(arr.shape) * arr]


liks_orig_pp = []
maxlik = -1e18
startrand = startparams

optpars = None
optmod2 = None
optpars2 = None

for i in range(rep_pp):
    xopt = tracts.optimize_cob_fracs2(
        startrand, bins, Ls, data, nind, func, props, outofbounds_fun=bound,
        cutoff=cutoff, epsilon=1e-2)
    # optimize_cob_fracs2 takes one additional parameter: the proportion of
    # each ancestry that will be used to fix the parameters.
    optmodlocal = tracts.DemographicModel(func(xopt, props))
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
        optmod2loc = tracts.DemographicModel(func2(xopt2, props))
        loclik = optmod2loc.loglik(bins, Ls, data, nind, cutoff=cutoff)
        if loclik > maxlik2:
            optmod2 = optmod2loc
            optpars2 = xopt2
    except Exception as ex:
        print(f"Convergence error: {ex}")
        loclik = -1e8
    liks_orig_pp_px.append(loclik)
    startrand2 = randomize(startparams2)

lik1 = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
lik2 = optmod2.loglik(bins, Ls, data, nind, cutoff=cutoff)

# Save the data to file for external plotting, model 1
write_to_files("./out")
# The first two files will be identical across models. We save an extra
# copy to facilitate the plotting.
write_to_files("./out2")
