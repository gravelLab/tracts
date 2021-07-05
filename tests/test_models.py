import unittest
import tracts
import time
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../example/2pops')
sys.path.append('../example/3pops')
sys.path.append('../example/4pops')
import pp
import models_3pop as threepop
import models_4pop as fourpop

class ManipsTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def ftest_model_params_fix(self, test_function, parameters, fracs, time_parameters):
        self.assertTrue(tracts.test_model_func(test_function, parameters, fracs_list=fracs,
                                               time_params=time_parameters)[0] > 0)
        mig = test_function(parameters, fracs=fracs)
        model = tracts.demographic_model(mig)
        for i in range(len(fracs)):
            self.assertTrue(abs(model.proportions[0, i] - fracs[i]) < 10 ** -8)

    def ftest_model_params(self, test_function, parameters, time_parameters):
        self.assertTrue(tracts.test_model_func(test_function, parameters,
                                               time_params=time_parameters)[0] > 0)


    def test_twopop_model_fix(self):
        """Make sure that model pp_fix outputs a mig matrix with correct end admixture proportions"""

        test_function = pp.pp_fix
        time_parameters = (True,)

        test_time = 0.10
        fracs = [0.5, 0.5]
        test_parameters = [test_time, ]

        self.ftest_model_params_fix(test_function=test_function, parameters=test_parameters, fracs = fracs,
                                time_parameters=time_parameters)



        # Repeat with different parameters
        test_time = 0.11
        fracs = [0, 1]
        test_parameters = [test_time, ]


        self.ftest_model_params_fix(test_function=test_function, parameters=test_parameters, fracs = fracs,
                                time_parameters=time_parameters)



    def test_models_pp(self):
        """Make sure that model pp outputs a mig matrix with correct end admixture proportions"""
        test_function = pp.pp
        time_parameters = [False, True]

        test_time = 0.11
        fracs = [0.5,0.5]
        test_parameters = [fracs[0], test_time]

        self.ftest_model_params(test_function=test_function, parameters=test_parameters,
                                time_parameters=time_parameters)

        test_time = .1
        fracs = [0.1, .9]
        test_parameters = [fracs[0], test_time]

        self.ftest_model_params(test_function=test_function, parameters=test_parameters,
                                time_parameters=time_parameters)

        # Models without fixed ancestry proportions do not have a systematic test for final proportions. Test by hand.
        mig = test_function(test_parameters)
        model = tracts.demographic_model(mig)
        for i in range(len(fracs)):
            self.assertTrue(abs(model.proportions[0, i] - fracs[i]) < 10 ** -8)

    def test_models_outofbounds_pp(self):
        """Make sure that model outofbounds_pp gives a negative value for problematic parameters"""
        test_function = pp.outofbounds_pp
        self.assertTrue(test_function([.5, -1]) < 0)
        self.assertTrue(test_function([2, .1]) < 0)
        self.assertTrue(test_function([-2, .1]) < 0)




    def test_models_ppx_xxp_fix(self):
        """Make sure that model ppx_xxp_fix outputs a mig matrix with correct end admixture proportions"""

        test_function = threepop.ppx_xxp_fix

        times = (0.10,0.05)
        fracs = [0.5, 0.2,.3]
        test_parameters = times
        time_parameters = [True,True]

        self.ftest_model_params_fix(test_function=test_function, parameters=test_parameters, fracs = fracs,
                                time_parameters=time_parameters)


    def test_models_pppp_pxxx_xpxx_xxpx_fix(self):
        """Make sure that model pppp_pxxx_xpxx_xxpx_fix outputs a mig matrix with correct end admixture proportions"""

        test_function = fourpop.pppp_pxxx_xpxx_xxpx_fix
        time_parameters = (True,True,True,True, False, False, False)

        fracs = [0.2, 0.2,.1,.5]
        #(tstart, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)
        test_parameters = (0.10, 0.05, .04, .03,.1,.2,.09)

        self.ftest_model_params_fix(test_function=test_function, parameters=test_parameters, fracs = fracs,
                                time_parameters=time_parameters)



        fracs = [0.5, 0.2,.1,.2]
        #(tstart, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)
        test_parameters = (0.10, 0.04, .04, .04,.1,.2,.1)

        self.ftest_model_params_fix(test_function=test_function, parameters=test_parameters, fracs = fracs,
                                time_parameters=time_parameters)


        fracs = [0.2, 0.2,.3,.3]
        #(tstart, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)
        test_parameters = (0.10, 0.05, .04, .05,.1,.2,.1)

        self.ftest_model_params_fix(test_function=test_function, parameters=test_parameters, fracs = fracs,
                                time_parameters=time_parameters)


    def test_models_outofbounds_pppp_pxxx_xpxx_xxpx_fix(self):
        # (tstart, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)
        """Make sure that model outofbounds_pp gives a negative value for problematic parameters"""
        test_function = fourpop.outofbounds_pppp_pxxx_xpxx_xxpx_fix
        fracs = [0.2, 0.2,.3,.3]
        test_parameters = (-0.10, 0.05, .04, .05, .1, .2, .1)

        self.assertTrue(test_function(test_parameters, fracs=fracs) < 0)

        fracs = [0.2, 0.2, .3, .3]
        test_parameters = (-0.10, 0.05, .04, .05, .1, .2, .1)

        self.assertTrue(test_function(test_parameters, fracs=fracs) < 0)

        fracs = [0.2, 0.2, .3, .3]
        test_parameters = (0.10, 0.15, .04, .05, .1, .2, .1)

        self.assertTrue(test_function(test_parameters, fracs=fracs) < 0)

        fracs = [0.2, 0.2, .3, .3]
        test_parameters = (0.10, 0.05, .04, .05, .1, .9, -.9)

        self.assertTrue(test_function(test_parameters, fracs=fracs) < 0)

suite = unittest.TestLoader().loadTestsFromTestCase(ManipsTestCase)