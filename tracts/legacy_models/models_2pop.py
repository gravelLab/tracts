import numpy
import scipy

def pp(*args):
    """ A simple model in which populations Eu and AFR arrive discretely at
    first generation. If a time is not integer, the migration is
    divided between neighboring times proportional to the non-integer
    time fraction.

    args are (init_Eu, tstart)
    """
    # the time is scaled by a factor 100 in this model to ease optimization
    # with some routines that expect all parameters to have the same scale

    (init_Eu, tstart) = args[0]

    tstart *= 100

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

    #migration at any given generation cannot be greater than 1
    ret = min(ret, 10*min(1-totmig), 10*min(totmig))

    #start time must be at least two generations ago
    ret = min(ret, tstart-.02)

    # print some diagnistics (facultative)
    if abs(totmig[-1]-1) > 1e-8:
        print(mig)
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

def pp_fix(args, fracs):
    """ A simple model in which populations Eu and AFR arrive discretely at
        first generation. If a time is not integer, the migration is divided
        between neighboring times proportional to the non-integer time
        fraction.
        """
    (tstart,) = args
    init_Eu = fracs[0] # Init_Eu is specified by the global ancestry proportions and will not be optimized
    return pp((init_Eu, tstart))

def outofbounds_pp_fix(params, fracs):
    # constraint function evaluating below zero when constraints not
    # satisfied

    init_Eu = fracs[0]

    (tstart,) = params
    return outofbounds_pp((init_Eu, tstart))

def propfrommig(mig):
    curr = mig[-1,:]
    for row in mig[-2::-1,:]:
        curr = curr*(1-numpy.sum(row))+row
    return curr

def pp_px(args):
    """ A simple model in which populations EUR and AFR arrive discretely at
        first generation, and a subsequent migration of EUR occurs at time T2.
        If a time is not integer, the migration is divided between neighboring
        times proportional to the non-integer time fraction.
        args = (init_Eu, tstart, t2, nuEu_prop)
        """
    (init_Eu, tstart, t2, nuEu_prop) = args
    tstart *= 100
    t2 *= 100

    # print "times ",tstart,t2

    if t2 > tstart or t2 < 0:
        # that should be caught by constraint. Return empty matrix
        gen = int(numpy.ceil(max(tstart, 0))) + 1
        mig = numpy.zeros( (gen + 1, 2) )
        return mig

    gen = int(numpy.ceil(tstart)) + 1
    frac = gen - tstart - 1
    mig = numpy.zeros((gen + 1, 2))

    init_Af = 1 - init_Eu

    # replace a fraction at second generation to ensure a continuous model
    # distribution with generation
    mig[-1,:] = numpy.array([init_Eu, init_Af])

    interEu = frac * init_Eu
    interAf = frac * init_Af
    mig[-2,:] = numpy.array([interEu, interAf])

    # finally add the second European pulse

    gen = int(numpy.ceil(t2))
    frac = gen - t2

    mig[gen-1, 0] = frac * nuEu_prop
    mig[gen, 0] = (nuEu_prop - frac * nuEu_prop) / (1 - frac * nuEu_prop)

    return mig

def outofbounds_pp_px(params):
    # constraint function evaluating below zero when constraints not satisfied
    ret = 1
    (init_Eu, tstart, t2, nuEu_prop) = params
    ret = min(1 - init_Eu, 1 - nuEu_prop)
    ret = min(ret, init_Eu, nuEu_prop)

    # generate the migration matrix and test for possible issues
    func = pp_px
    mig = func(params)
    totmig = mig.sum(axis=1)

    if init_Eu > 1 or nuEu_prop > 1:
        print("Pulse greater than 1")
    if init_Eu < 0 or nuEu_prop < 0:
        print("Pulse less than 0")

    ret = min(ret, -abs(totmig[-1]-1) + 1e-8)
    ret = min(ret, -totmig[0], -totmig[1])

    ret = min(ret, 10 * min(1 - totmig), 10 * min(totmig))

    ret = min(ret, tstart - t2)

    ret = min(ret, t2)

    if abs(totmig[-1]-1) > 1e-8:
        print(mig)
        print("founding migration should sum up to 1. Now:")

    if totmig[0] > 1e-10:
        print("migrants at last generation should be removed from sample!")

    if totmig[1] > 1e-10:
        print("migrants at penultimate generation should be removed from sample!")

    if ((totmig > 1).any() or (mig < 0).any()):
        print("migration rates should be between 0 and 1")

    return ret

def pp_px_fix(args, fracs):
    (tstart, t2, nuEu_prop) = args
    def fun(init_Eu):
        # If it is pased as an array, can cause problems
        init_Eu = float(init_Eu)
        return propfrommig(pp_px((init_Eu, tstart, t2, nuEu_prop)))[0] \
                - fracs[0]

    (init_Eu,) = scipy.optimize.fsolve(fun, (.2,))
    # print "init_Eu",init_Eu

    return pp_px((init_Eu, tstart, t2, nuEu_prop))

def outofbounds_pp_px_fix(params, fracs):
    # constraint function evaluating below zero when constraints not satisfied
    (tstart, t2, nuEu_prop) = params

    def fun(init_Eu):
        init_Eu = float(init_Eu)
        return propfrommig(pp_px((init_Eu, tstart, t2, nuEu_prop)))[0] \
                - fracs[0]

    # print "example:,",pp_px((.2,tstart,t2,nuEu_prop))
    # print fun(0.2)
    # print "init_Eu," ,scipy.optimize.fsolve(fun,(.2,))
    # print "made it!"

    (init_Eu,) = scipy.optimize.fsolve(fun, (.2,))
    # print "init_Eu,",init_Eu
    return outofbounds_pp_px((init_Eu, tstart, t2, nuEu_prop))
