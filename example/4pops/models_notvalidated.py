import numpy as np
import scipy.optimize


"""
A simple model in which populations 1 and 2 arrive discretely at first
generation, 3 and 4 at a subsequent generation. If a time is not integer, the
migration is divided between neighboring times proportional to the
non-integer time fraction.  
There is a corner case when the time of the two events are the same: 
should the second wave migrants replace the first wave migrants, or arrive
simultaneously?  

We'll assume populations 3 and 4 do arrive later and replace
migrants from 1 and 2 after the replacement from population 1 and 2 if they
arrive at same generation.
"""


def ppxx_xxpp(*params):
    """ A simple model in which populations 1 and 2 arrive discretely at first
        generation, 3 and 4 at a subsequent generation. If a time is not integer,
        the migration is divided between neighboring times proportional to the
        non-integer time fraction.  We'll assume population 3 and 4 still replaces
        migrants from 1 and 2 after the replacement from population 1 and 2 if
        they arrive at same generation.

        Parameters: (prop1, tstart, prop3, prop4, t3)
        In this prop1 is the initial proportion from population 1,
        prop2=1-prop1, tstart is the arrival times of pops (1,2) t3 is the
        arrival time of pop 3
        prop3 and prop4 are the proportion of migrants from populations 3 and 4.

        The two times are measured in units of 100 generations, because some
        python optimizers work better when all parameters have the same scale.
        """
    (prop1, tstart, prop3, prop4, t3) = params[0]
    n_pops = 4
    tstart *= 100
    t3 *= 100

    # some sanity checks
    if t3 > tstart or t3 < 0 or tstart < 0:
        # This will be caught by "outofbounds" function. Return empty matrix
        gen = int(np.ceil(max(tstart, 0))) + 1
        mig = np.zeros((gen + 1, 3))
        return mig

    # How many generations we'll need to accomodate all the migrations. The +1
    # is not really necessary here.
    gen = int(np.ceil(tstart)) + 1

    # How far off the continuous time is from its discrete optimization
    timefrac = gen - tstart - 1

    # Build an empty matrix with enough generations to handle all migrations
    mig = np.zeros((gen + 1, n_pops))

    # replace a fraction at first and second generation to ensure a continuous
    # model
    prop2 = 1 - prop1
    mig[-1, :] = np.array([prop1, prop2, 0, 0])

    penultimate_prop1 = prop1 * timefrac
    penultimate_prop2 = prop2 * timefrac
    mig[-2, :] = np.array([penultimate_prop1, penultimate_prop2, 0, 0])

    # Which integer generation to add the migrants from pop 3
    gen3 = int(np.ceil(t3)) + 1
    timefrac3 = gen3 - t3 - 1

    # we want the total proportion replaced  by 3 to be prop3.
    # However, because the migration is split over two generations,
    # some of the migrants from the first generation will be replaced
    # by migrants from populations 3 and 4 in the second generation.
    #
    #
    # We therefore add
    # a fraction f at generation gen-1, and (prop3-f)/(1-f) at generation gen.

    prop3_floor = timefrac3 * prop3
    prop4_floor = timefrac3 * prop4
    # The proportion of genomes coming from population 3 will be
    # prop3_floor + prop3_ceiling* ( 1 - prop3_floor + prop4_floor )
    # Set prop3_floor + prop3_ceiling* ( 1 - prop3_floor - prop4_floor ) == prop3
    # Solve for prop3_ceiling
    prop3_ceil = (prop3 - prop3_floor) / (1 - prop3_floor - prop4_floor)
    prop4_ceil = (prop4 - prop4_floor) / (1 - prop3_floor - prop4_floor)

    pop_index_3 = 2  # pop 3 has index two because python is zero-based
    mig[gen3 - 1, pop_index_3] = prop3_floor
    mig[gen3, pop_index_3] = prop3_ceil

    pop_index_4 = 3  # pop 3 has index two because python is zero-based
    mig[gen3 - 1, pop_index_4] = prop4_floor
    mig[gen3, pop_index_4] = prop4_ceil

    return mig


def outofbounds_ppxx_xxpp(*params):
    """ Constraint function evaluating below zero when constraints are not
        satisfied. """

    (prop1, tstart, prop3, prop4, t3) = params[0]

    violation = min(1-prop1, 1-prop3, 1-prop4)
    violation = min(violation, prop1, prop3, prop4)

    # Pedestrian way of testing for all possible issues
    func = ppxx_xxpp
    mig = func(params[0])
    totmig = mig.sum(axis=1)

    violation = min(violation, -abs(totmig[-1]-1) + 1e-8)   # Check that initial migration sums to 1.
    violation = min(violation, -totmig[0], -totmig[1])      # Check that there are no migrations in the last
    # two generations

    violation = min(violation, min(1-totmig), min(totmig))  # Check that total migration rates between 0 and 1

    violation = min(violation, tstart-t3)

    violation = min(violation, t3)
    violation = min(violation, tstart)
    return violation

# We don't have to calculate all the tract length distributions to have the
# global ancestry proportion right. Here we define a function that
# automatically adjusts the migration rates to have the proper global ancestry
# proportion, saving a lot of optimization time!
# We first define a function that calculates the final proportions of ancestry
# based on the migration matrix


def propfrommig(mig):
    """Obtains the propotion of present dat genomes contributed by each population"""

    current_contributions = mig[-1, :]
    for row in mig[-2::-1, :]:
        current_contributions = current_contributions*(1-np.sum(row))+row
    return current_contributions


def ppxx_xxpp_fix(params, fracs):
    # An auxiliary function that, given the "missing" parameters, returns the
    # full migration matrix
    (tstart, t3) = params
    n_pops = 4

    def fun(params_optimize):
        """function taking the missing parameters and compares results to expected ancestry proportions"""
        (prop1_opt, prop3_opt, prop4_opt) = params_optimize
        return propfrommig(ppxx_xxpp((prop1_opt, tstart, prop3_opt, prop4_opt, t3)))[0:n_pops-1] - fracs[0:n_pops-1]

    # Find the best-fitting parameters
    # (.2,.2) is just the starting point for the optimization function, it
    # should not be sensitive to this, but it's better to start with reasonable
    # parameter values.
    (prop1, prop3, prop4) = scipy.optimize.fsolve(fun, np.array((.2, .2, .2)))
    return ppxx_xxpp((prop1, tstart, prop3, prop4, t3))


def outofbounds_ppxx_xxpp_fix(params, fracs):
    """constraint function evaluating below zero when constraints not satisfied"""
    n_pops = 4
    (tstart, t3) = params
    if tstart > 1:
        print("time above 100 generations!")
        return 1 - tstart

    def fun(params_optimize):
        (prop1_opt, prop3_opt, prop4_opt) = params_optimize
        return propfrommig(ppxx_xxpp((prop1_opt, tstart, prop3_opt, prop4_opt, t3)))[0:n_pops-1] - fracs[0:n_pops-1]

    # (.2,.2,.2) blelow is just the starting point for the
    # optimization function. Result should not be sensitive
    # to this, but it's better to start with reasonable parameter values.
    (prop1, prop3, prop4) = scipy.optimize.fsolve(fun, np.array((.2, .2, .2)))
    return outofbounds_ppxx_xxpp((prop1, tstart, prop3, prop4, t3))
