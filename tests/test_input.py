import unittest
import tracts
import time
import numpy as np
import sys
sys.path.append('../')


class ManipsTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_readfile(self):

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


suite = unittest.TestLoader().loadTestsFromTestCase(ManipsTestCase)
