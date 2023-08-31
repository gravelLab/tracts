import numpy
import scipy.optimize


"""
A simple model in which populations 1 and 2 arrive discretely at first
generation, 3 at a subsequent generation. If a time is not integer, the
migration is divided between neighboring times proportional to the
non-integer time fraction.  We'll assume population 3 still replaces
migrants from 1 and 2 after the replacement from population 1 and 2 if they
arrive at same generation.
"""


def ppx_xxp(*params):
    """ A simple model in which populations 1 and 2 arrive discretely at first
        generation, 3 at a subsequent generation. If a time is not integer,
        the migration is divided between neighboring times proportional to the
        non-integer time fraction.  We'll assume population 3 still replaces
        migrants from 1 and 2 after the replacement from population 1 and 2 if
        they arrive at same generation.

        Parameters: (prop1, tstart, prop3, t3)
        In this prop1 is the initial proportion from population 1,
        prop2=1-prop1, tstart is the arrival times of pops (1,2) t3 is the
        arrival time of pop 3

        The two times are measured in units of 100 generations, because some
        python optimizers work better when all parameters have the same scale.
        """
    (prop1, tstart, prop3, t3) = params[0]

    tstart *= 100
    t3 *= 100

    # some sanity checks
    if t3 > tstart or t3 < 0 or tstart < 0:
        # This will be caught by "outofbounds" function. Return empty matrix
        gen = int(numpy.ceil(max(tstart, 0))) + 1
        mig = numpy.zeros((gen + 1, 3))
        return mig

    # How many generations we'll need to accomodate all the migrations. The +1
    # is not really necessary here.
    gen = int(numpy.ceil(tstart)) + 1

    # How far off the continuous time is from its discrete optimization
    timefrac = gen - tstart - 1

    # Build an empty matrix with enough generations to handle all migrations
    mig = numpy.zeros((gen + 1, 3))

    # replace a fraction at first and second generation to ensure a continuous
    # model
    prop2 = 1 - prop1
    mig[-1, :] = numpy.array([prop1, prop2, 0])

    interEu = prop1 * timefrac
    interNat = prop2 * timefrac
    mig[-2, :] = numpy.array([interEu, interNat, 0])

    # Which integer generation to add the migrants from pop 3
    gen3 = int(numpy.ceil(t3)) + 1
    timefrac3 = gen3 - t3 - 1

    # we want the total proportion replaced  by 3 to be prop3. We therefore add
    # a fraction f at generation gen-1, and (prop3-f)/(1-f) at generation gen.

    mig[gen3 - 1, 2] = timefrac3 * prop3
    mig[gen3, 2] = (prop3 - timefrac3 * prop3) / (1 - timefrac3 * prop3)

    return mig

def outofbounds_ppx_xxp(*params):
    """ Constraint function evaluating below zero when constraints are not
        satisfied. """
    ret = 1

    (prop1, tstart, prop3, t3) = params[0]

    ret = min(1-prop1, 1-prop3)
    ret = min(ret, prop1, prop3)

    # Pedestrian way of testing for all possible issues
    func = ppx_xxp
    mig = func(params[0])
    totmig = mig.sum(axis=1)
    # print  "ret1=",ret

    if prop1 > 1 or prop3 > 1:
        print("Pulse greater than 1")
    if prop1 < 0 or prop3 < 0:
        print("Pulse less than 0")
    # print  "ret2 ",ret

    ret = min(ret, -abs(totmig[-1]-1) + 1e-8)
    ret = min(ret, -totmig[0], -totmig[1])
    # print "ret3 " , ret

    ret = min(ret, min(1-totmig), min(totmig))
    # print "ret4 " , ret

    # print "times ",t3,tstart
    ret = min(ret, tstart-t3)

    ret = min(ret, t3)
    ret = min(ret, tstart)
    return ret

# We don't have to calculate all the tract length distributions to have the
# global ancestry proportion right. Here we define a function that
# automatically adjusts the migration rates to have the proper global ancestry
# proportion, saving a lot of optimization time!
# We first define a function that calculates the final proportions of ancestry
# based on the migration matrix
def propfrommig(mig):
    curr = mig[-1,:]
    for row in mig[-2::-1,:]:
        curr = curr*(1-numpy.sum(row))+row
    return curr

def ppx_xxp_fix(times, fracs):
    (tstart, t3) = times
    # An auxiliary function that, given the "missing" parameters, returns the
    # full migration matrix
    def fun(props):
        (prop3, prop1) = props
        return propfrommig(ppx_xxp((prop1, tstart, prop3, t3)))[0:2] - fracs[0:2]

    # Find the best-fitting parameters
    # (.2,.2) is just the starting point for the optimization function, it
    # should not be sensitive to this, but it's better to start with reasonable
    # parameter values.
    (prop3, prop1) = scipy.optimize.fsolve(fun, (.2, .2))
    return ppx_xxp((prop1, tstart, prop3, t3))

def outofbounds_ppx_xxp_fix(params, fracs):
    # constraint function evaluating below zero when constraints not satisfied
    # print "in constraint  outofbounds_211_cont_unif_params"

    ret = 1

    (tstart, t3) = params
    if tstart > 1:
        print("time above 500 generations!")
        return (1 - tstart)

    def fun(props):
        (prop3, prop1) = props
        return propfrommig(ppx_xxp((prop1, tstart, prop3, t3)))[0:2] - fracs[0:2]
    (prop3, prop1) = scipy.optimize.fsolve(fun, (.2, .2)) #(.2,.2) is just the starting point for the optimization function, it should not be sensitive to this, but it's better to start with reasonable parameter values.
    return outofbounds_ppx_xxp((prop1, tstart, prop3, t3))
