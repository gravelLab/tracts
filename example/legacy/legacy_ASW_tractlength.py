#!/usr/bin/env python

import numpy

import tracts
import tracts.legacy_models.models_2pop as models_2pop
from legacy_ASW_data import *

# (initial European proportion, and time). Times are
# measured in units of hundred generations (i.e.,
# multiply the number by 100 to get the time in
# generations). The reason is that some python
# optimizers do a poor job when the parameters (time
# and ancestry proportions) are of different
# magnitudes.
startparams = numpy.array([0.173632, 0.0683211])
# you can also look at the "_mig" output file for a
# generation-by-generation breakdown of the migration rates.

Ls = pop.Ls
nind = pop.nind

# we compare two models; single pulse versus two European pulses.
func = models_2pop.pp
bound = models_2pop.outofbounds_pp
func2 = models_2pop.pp_px
bound2 = models_2pop.outofbounds_pp_px
# (init_Eu,tstart,t2,nuEu_prop)
# give two different starting conditions, with one starting near the
# single-pulse model
startparams2 = numpy.array([0.125102, 0.107152, 0.0438957, 0.051725])

optmod = tracts.DemographicModel(func(startparams))


def randomize(arr, scale=2):
    # takes an array and multiplies every element by a factor between 0 and 2,
    # uniformly. caps at 1.
    return list(map(lambda i: min(i, 1), scale * numpy.random.random(arr.shape) * arr))


liks_orig_pp = []
maxlik = -1e18
startrand = startparams
for i in range(rep_pp):
    xopt = tracts.optimize_cob(
        startrand, bins, Ls, data, nind, func, outofbounds_fun=bound, cutoff=cutoff, epsilon=1e-2)
    try:
        optmodlocal = tracts.DemographicModel(func(xopt))
        loclik = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
        if loclik > maxlik:
            optmod = optmodlocal
            optpars = xopt
    except Exception as ex:
        print(f"Convergence error: {ex}")
        loclik = -1e8

    liks_orig_pp.append(loclik)

    startrand = randomize(startparams)

print("likelihoods found: ", liks_orig_pp)

liks_orig_pp_px = []
startrand2 = startparams2
maxlik2 = -1e18
optmod2 = None

for i in range(0, rep_pp_px):
    xopt2 = tracts.optimize_cob(
        startrand2, bins, Ls, data, nind, func2, outofbounds_fun=bound2, cutoff=cutoff, epsilon=1e-2)
    try:
        optmod2loc = tracts.DemographicModel(func2(xopt2))
        loclik = optmod2loc.loglik(bins, Ls, data, nind, cutoff=cutoff)
        if loclik > maxlik2:
            optmod2 = optmod2loc
            optpars = xopt2
    except Exception as ex:
        print(f"Convergence error: {ex}")
        loclik = -1e8

    liks_orig_pp_px.append(loclik)
    startrand2 = randomize(startparams2)

lik1 = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
lik2 = optmod2.loglik(bins, Ls, data, nind, cutoff=cutoff)

print("optimal likelihoods values found for single pulse model:", lik1)
print("optimal likelihoods values found for two pulse model:", lik2)
