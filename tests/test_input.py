import unittest
import tracts
import time
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../example/2pops')
sys.path.append('../example/3pops')
import pp
import models as threepop


class ManipsTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))


    def test_models_twopops_fix(self):
        time = 0.10
        fracs = [0.5,0.5]
        mig = pp.pp_fix([time,],fracs = fracs )
        model = tracts.demographic_model(mig)
        self.assertTrue(model.proportions[-1,0] == fracs[0])
        self.assertTrue(model.proportions[-1, 1] == fracs[1])

        fracs = [0, 1]
        mig = pp.pp_fix([time, ], fracs=fracs)
        model = tracts.demographic_model(mig)
        self.assertTrue(model.proportions[-1, 0] == fracs[0])
        self.assertTrue(model.proportions[-1, 1] == fracs[1])


    def test_models_twopops(self):
        time = 0.10
        fracs = [0.5,0.5]
        mig = pp.pp([fracs[0],time])
        model = tracts.demographic_model(mig)

        self.assertTrue(model.proportions[0,0] == fracs[0])
        self.assertTrue(model.proportions[0, 1] == fracs[1])

        fracs = [0.1, .9]
        mig = pp.pp([fracs[0], time])
        model = tracts.demographic_model(mig)
        self.assertTrue(model.proportions[0, 0] == fracs[0])
        self.assertTrue(model.proportions[0, 1] == fracs[1])

    def test_models_threepops_fix(self):
        times = (0.10,0.05)
        fracs = [0.5, 0.2,.3]
        mig = threepop.ppx_xxp_fix(times, fracs = fracs )
        model = tracts.demographic_model(mig)

        self.assertTrue(model.proportions[0, 0] == fracs[0])
        self.assertTrue(model.proportions[0, 1] == fracs[1])
        self.assertTrue(model.proportions[0, 2] == fracs[2])



    def test_run_optimization_slow(self):
        # number of short tract bins not used in inference.
        cutoff = 2
        directory = "../example/2pops/G10/"
        names = [
            "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
            "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
            "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
            "NA20357", "NA20356"
            ]
        chroms = ['%d' % (i,) for i in range(1, 23)]

        pop = tracts.population(
            names=names, fname=(directory, "", ".bed"), selectchrom=chroms)
        (bins, data) = pop.get_global_tractlengths(npts=50)

        self.assertTrue(np.sum(data['AFR']) == 2210)
        self.assertTrue(list(data.keys())[0] == 'AFR')
        self.assertTrue(list(data.keys())[1] == 'EUR')
        self.assertTrue(len(bins) == 51)

        rep_pp = 1

        labels = ['EUR', 'AFR']
        data = [data[poplab] for poplab in labels]

        startparams = np.array([0.1683211])

        Ls = pop.Ls
        nind = pop.nind

        def randomize(arr, scale=2):
            """ Scale each element of an array by some random factor between zero and a
                limit (default: 2), capping the result at 1.
            """
            return [min(i, 1) for i in scale * np.random.random(arr.shape) * arr]
        bypopfrac = [[] for i in range(len(labels))]
        for ind in pop.indivs:
            # a list of tracts with labels and names
            tractslst = ind.applychrom(tracts.chrom.tractlengths)
            # a flattened list of tracts with labels and names
            flattracts = [
                np.sum([
                    item[1] for chromo in tractslst
                    for sublist in chromo
                    for item in sublist
                    if item[0] == label])
                for label in labels]
            tracts_sum = np.sum(flattracts)
            for i in range(len(labels)):
                bypopfrac[i].append(flattracts[i] / tracts_sum)

        props = list(map(np.mean, bypopfrac))

        # we compare two models; single pulse versus two European pulses.
        func = pp.pp_fix
        bound = pp.outofbounds_pp_fix

        optmod = tracts.demographic_model(func(startparams, fracs = props))

        liks_orig_pp = []
        maxlik = -1e18
        startrand = startparams
        for i in range(rep_pp):
            xopt = tracts.optimize_cob_fracs2(
                startrand, bins, Ls, data, nind, func, props, outofbounds_fun=bound,
                cutoff=cutoff, epsilon=1e-2)
            # optimize_cob_fracs2 takes one additional parameter: the proportion of
            # each ancestry that will be used to fix the parameters.
            optmodlocal = tracts.demographic_model(func(xopt, fracs = props))
            loclik = optmod.loglik(bins, Ls, data, nind, cutoff=cutoff)
            if loclik > maxlik:
                optmod = optmodlocal
                optpars = xopt
            liks_orig_pp.append(loclik)

            startrand = randomize(startparams)

        print("likelihoods found: ", liks_orig_pp)

suite = unittest.TestLoader().loadTestsFromTestCase(ManipsTestCase)
