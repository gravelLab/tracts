import numpy
def pp((init_Eu, tstart)):

        """a simple model in which populations Eu and AFR arrive discretely at first generation. If a time is not integer, the migration
        is divided between neighboring times proportional to the non-integer time fraction.
        """


        tstart *= 100 #the time is scaled by a factor 100 in this model to ease optimization with some routines that expect all parameters to have the same scale


        if  tstart < 0:
                # time shouldn't be negative: that should be caught by
                # constraint. Return empty matrix
                gen = int(numpy.ceil(max(tstart, 0)))+1
                mig = numpy.zeros((gen+1, 2))
                return mig

        gen = int(numpy.ceil(tstart))+1
        frac = gen-tstart-1
        mig = numpy.zeros((gen+1, 2))

        initNat = 1-init_Eu

        # replace a fraction at second generation to ensure a continuous model
        # distribution with generation
        mig[-1,:] = numpy.array([init_Eu, initNat])
        mig[-2,:] = frac*numpy.array([init_Eu, initNat])

        return mig


def outofbounds_pp(params):
        # constraint function evaluating below zero when constraints not
        # satisfied
        ret = 1
        (init_Eu, tstart) = params

        ret = min(1, 1-init_Eu) #migration proportion must be between 0 and 1
        ret = min(ret, init_Eu)



        # generate the migration matrix and test for possible issues
        func = pp #specify the model
        mig = func(params) #get the migration matrix
        totmig = mig.sum(axis=1) #calculate the migration rate per generation

        ret = min(ret, -abs(totmig[-1]-1)+1e-8) #first generation migration must sum up to 1 
        ret = min(ret, -totmig[0], -totmig[1]) #no migrations are allowed in the first two generations

        ret = min(ret, 10*min(1-totmig), 10*min(totmig)) #migration at any given generation cannot be greater than 1



        ret = min(ret, tstart-.02) #start time must be at least two generations ago


        # print some diagnistics (facultative)
        if abs(totmig[-1]-1) > 1e-8:
                        print mig
                        print("founding migration should sum up to 1. Now:")


        if totmig[0] > 1e-10:
                        print("migrants at last generation should be removed from sample!")



        if totmig[1] > 1e-10:
                        print("migrants at penultimate generation should be removed from sample!")



        if ((totmig > 1).any() or (mig < 0).any()):
                        print("migration rates should be between 0 and 1")

        return ret


# now define the same model, but fixing the ancestry proportion using the
# known total ancestry proportions "frac"

def pp_fix((tstart,), fracs):

        """a simple model in which populations Eu and AFR arrive discretely at first generation. If a time is not integer, the migration
        is divided between neighboring times proportional to the non-integer time fraction.
        """



        init_Eu = fracs[0] #init_Eu is specified by the global ancestry proportions and will not be optimized 
        return pp((init_Eu, tstart))


def outofbounds_pp_fix(params, fracs):
        # constraint function evaluating below zero when constraints not
        # satisfied

        init_Eu = fracs[0]

        (tstart,) = params
        return outofbounds_pp((init_Eu, tstart))
