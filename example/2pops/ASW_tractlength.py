#!/usr/bin/env python

import sys
sys.path.append("../..")
                #path to tracts, may have to adjust if the file is moved
import tracts
import tracts.legacy_models.models_2pop as models_2pop
import numpy
import pylab

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

# (initial European proportion, and time). Times are
# measured in units of hundred generations (i.e.,
# multiply the number by 100 to get the time in
# generations). The reason is that some python
# optimizers do a poor job when the parameters (time
# and ancestry proportions) are of different
# magnitudes.
startparams = numpy.array([0.173632,  0.0683211])
# you can also look at the "_mig" output file for a
# generation-by-generation breakdown of the migration rates.

Ls = pop.Ls
nind = pop.nind

# we compare two models; single pulse versus two European pulses.
func = models_2pop.pp
bound = models_2pop.outofbounds_pp
func2 = models_2pop.pp_px
bound2 = models_2pop.outofbounds_pp_px
#(init_Eu,tstart,t2,nuEu_prop)
# give two different starting conditions, with one starting near the
# single-pulse model
startparams2 = numpy.array([0.125102, 0.107152, 0.0438957, 0.051725])

optmod = tracts.demographic_model(func(startparams))

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
        optmodlocal = tracts.demographic_model(func(xopt))
        loclik = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
        if loclik > maxlik:
            optmod = optmodlocal
            optpars = xopt
    except:
        print("convergence error")
        loclik = -1e8

    liks_orig_pp.append(loclik)

    startrand = randomize(startparams)


print("likelihoods found: ", liks_orig_pp)

liks_orig_pp_px = []
startrand2 = startparams2
maxlik2 = -1e18

for i in range(0, rep_pp_px):
    xopt2 = tracts.optimize_cob(
        startrand2, bins, Ls, data, nind, func2, outofbounds_fun=bound2, cutoff=cutoff, epsilon=1e-2)
    try:
        optmod2loc = tracts.demographic_model(func2(xopt2))
        loclik = optmod2loc.loglik(bins, Ls, data, nind, cutoff=cutoff)
        if loclik > maxlik2:
            optmod2 = optmod2loc
            optpars = xopt2
    except:
        print("convergence error")
        loclik = -1e8

    liks_orig_pp_px.append(loclik)
    startrand2 = randomize(startparams2)

lik1 = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
lik2 = optmod2.loglik(bins, Ls, data, nind, cutoff=cutoff)

print("optimal likelihoods values found for single pulse model:", lik1)
print("optimal likelihoods values found for two pulse model:", lik2)
