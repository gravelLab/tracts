import numpy
import scipy

def propfrommig(mig):
    curr = mig[-1,:]
    for row in mig[-2::-1,:]:
        curr = curr*(1-numpy.sum(row))+row
    return curr

def pp_px((init_Eu, tstart, t2, nuEu_prop)):
    """a simple model in which populations Eu and Afr arrive discretely at first generation, and a subsequent migration of Eu occurs at time T2.
    . If a time is not integer, the migration
    is divided between neighboring times proportional to the non-integer time fraction. """


    tstart *= 100
    t2 *= 100

    # print "times ",tstart,t2


    if t2 > tstart or t2 < 0:
        # that should be caught by constraint. Return empty matrix
        gen = int(numpy.ceil(max(tstart, 0)))+1
        mig = numpy.zeros((gen+1, 2))
        return mig

    gen = int(numpy.ceil(tstart))+1
    frac = gen-tstart-1
    mig = numpy.zeros((gen+1, 2))

    init_Af = 1-init_Eu

    # replace a fraction at second generation to ensure a continuous model
    # distribution with generation
    mig[-1,:] = numpy.array([init_Eu, init_Af])

    interEu = frac*init_Eu
    interAf = frac*init_Af
    mig[-2,:] = numpy.array([interEu, interAf])

    # finally add the second European pulse

    gen = int(numpy.ceil(t2))
    frac = gen-t2

    mig[gen-1, 0] = frac*nuEu_prop
    mig[gen, 0] = (nuEu_prop-frac*nuEu_prop)/(1-frac*nuEu_prop)

    return mig

def outofbounds_pp_px(params):
    # constraint function evaluating below zero when constraints not satisfied
    ret = 1
    (init_Eu, tstart, t2, nuEu_prop) = params
    ret = min(1-init_Eu, 1-nuEu_prop)
    ret = min(ret, init_Eu, nuEu_prop)

    # generate the migration matrix and test for possible issues
    func = pp_px
    mig = func(params)
    totmig = mig.sum(axis=1)

    if init_Eu > 1 or nuEu_prop > 1:
        print("Pulse greater than 1")
    if init_Eu < 0 or nuEu_prop < 0:
        print("Pulse less than 0")

    ret = min(ret, -abs(totmig[-1]-1)+1e-8)
    ret = min(ret, -totmig[0], -totmig[1])

    ret = min(ret, 10*min(1-totmig), 10*min(totmig))

    ret = min(ret, tstart-t2)

    ret = min(ret, t2)

    if abs(totmig[-1]-1) > 1e-8:
        print mig
        print("founding migration should sum up to 1. Now:")

    if totmig[0] > 1e-10:
        print("migrants at last generation should be removed from sample!")

    if totmig[1] > 1e-10:
        print(
            "migrants at penultimate generation should be removed from sample!")

    if ((totmig > 1).any() or (mig < 0).any()):
        print("migration rates should be between 0 and 1")

    return ret

def pp_px_fix((tstart, t2, nuEu_prop), fracs):
    def fun(init_Eu):
        init_Eu = float(init_Eu) #if it is pased as an array, can cause problems
        return propfrommig(pp_px((init_Eu, tstart, t2, nuEu_prop)))[0] - fracs[0]

    (init_Eu,) = scipy.optimize.fsolve(fun, (.2,))
    # print "init_Eu",init_Eu

    return pp_px((init_Eu, tstart, t2, nuEu_prop))

def outofbounds_pp_px_fix(params, fracs):
    # constraint function evaluating below zero when constraints not satisfied
    (tstart, t2, nuEu_prop) = params

    def fun(init_Eu):
        init_Eu = float(init_Eu)
        return propfrommig(pp_px((init_Eu, tstart, t2, nuEu_prop)))[0] - fracs[0]

    # print "example:,",pp_px((.2,tstart,t2,nuEu_prop))
    # print fun(0.2)
    # print "init_Eu," ,scipy.optimize.fsolve(fun,(.2,))
    # print "made it!"

    (init_Eu,) = scipy.optimize.fsolve(fun, (.2,))
    # print "init_Eu,",init_Eu
    return outofbounds_pp_px((init_Eu, tstart, t2, nuEu_prop))
