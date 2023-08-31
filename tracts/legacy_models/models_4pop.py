import numpy
import scipy.optimize

def pppp_pxxx_xpxx_xxpx(*params):
    """ A simple model in which populations 1, 2, 3 and 4 arrive discretely at first
        generation, 1 at a subsequent generation, followed 2 thereafter and finally 3. If a time is not integer,
        the migration is divided between neighboring times proportional to the
        non-integer time fraction.  We'll assume population 3 and 4 still replaces
        migrants from 1 and 2 after the replacement from population 1 and 2 if
        they arrive at same generation.
        Need to add another time parameter
        Parameters: (prop1, tstart, prop3, prop4, t3, t4)
        In this prop1 is the initial proportion from population 1,
        prop2=1-prop1, tstart is the arrival times of pops (1,2) t3 is the
        arrival time of pop 3 and t4 is the arrival time of prop4
        prop3 and prop4 are the proportion of migrants from populations 3 and 4.

        The two times are measured in units of 100 generations, because some
        python optimizers work better when all parameters have the same scale.
        """
    (prop1, tstart, prop3, prop4, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary) = params[0]  #Added t4, t5"
    n_pops = 4
    tstart *= 100
    t3 *= 100
    t4 *= 100
    t5 *= 100


    # some sanity checks
    if t3 > tstart or t3 < 0 or tstart < 0 or t4 < 0 or t4 > tstart or t5 < 0 or t5 > tstart:  #Added t4 and t5 sanity checks"
        # This will be caught by "outofbounds" function. Return empty matrix
        gen = int(numpy.ceil(max(tstart, 0))) + 1
        mig = numpy.zeros((gen + 1, n_pops))
        return mig

    # How many generations we'll need to accomodate all the migrations. The +1
    # is not really necessary here.
    gen = int(numpy.ceil(tstart)) + 1

    # How far off the continuous time is from its discrete optimization
    timefrac = gen - tstart - 1

    # Build an empty matrix with enough generations to handle all migrations
    mig = numpy.zeros((gen + 1, n_pops))

    # replace a fraction at first and second generation to ensure a continuous
    # model "This part stays the same as the first part of the equation is still ppxx"
    prop2 = 1 - (prop1 + prop3 + prop4)     #changed this to properly calculate prop2

    mig[-1, :] = numpy.array([prop1, prop2, prop3, prop4])

    penultimate_prop1 = prop1 * timefrac            # Calculate penultimate_prop for all 4 populations
    penultimate_prop2 = prop2 * timefrac
    penultimate_prop3 = prop3 * timefrac
    penultimate_prop4 = prop4 * timefrac

    mig[-2, :] = numpy.array([penultimate_prop1, penultimate_prop2, penultimate_prop3, penultimate_prop4])

    # Which integer generation to add the migrants from pop1
    gen3 = int(numpy.ceil(t3)) + 1
    timefrac3 = gen3 - t3 - 1

    prop1_floor = timefrac3 * prop1_secondary         #At time t3 only migrants from pop1 (San)

    prop1_ceil = (prop1_secondary - prop1_floor) / (  1 - prop1_floor) # Using only prop1 here as migrants from one population (San) only

    pop_index_1 = 0 #pop 1 has index zero because python is zero-based

    mig[gen3 - 1, pop_index_1] += prop1_floor       # Update mig matrix with new pop1 (San) migrants
    mig[gen3, pop_index_1] += prop1_ceil



#Added this for time 4"
    # Which integer generation to add the migrants from pop2 (Bantu)
    gen4 = int(numpy.ceil(t4)) + 1
    timefrac4 = gen4 - t4 - 1

    prop2_floor = timefrac4 * prop2_secondary

    prop2_ceil = (prop2_secondary - prop2_floor) / ( 1 - prop2_floor) # Using only prop2 here as migrants from one population (Bantu) only

    pop_index_2 = 1 #pop 2 has index one because python is zero-based

    mig[gen4 - 1, pop_index_2] += prop2_floor # Update mig matrix with new pop2 (Bantu) migrants
    mig[gen4, pop_index_2] += prop2_ceil


#Added this for time 5"
    # Which integer generation to add the migrants from pop 3 (European)
    gen5 = int(numpy.ceil(t5)) + 1
    timefrac5 = gen5 - t5 - 1

    prop3_floor = timefrac5 * prop3_secondary

    prop3_ceil = (prop3_secondary - prop3_floor) / ( 1 - prop3_floor ) # Using only prop3 here as migrants from one population (European) only

    pop_index_3 = 2 #pop 3 has index two because python is zero-based

    mig[gen5 - 1, pop_index_3] += prop3_floor # Update mig matrix with new pop3 (European) migrants
    mig[gen5, pop_index_3] += prop3_ceil

    return mig

def outofbounds_pppp_pxxx_xpxx_xxpx(*params):       # I did not change much here
    """ Constraint function evaluating below zero when constraints are not
        satisfied. """
    violation = 1

    (prop1, tstart, prop3, prop4, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary) = params[0]       #Added t4 t5"

    violation = min(1-prop1, 1-prop3, 1-prop4, 1-prop1_secondary, 1-prop2_secondary, 1-prop3_secondary)
    violation = min(violation, prop1, prop3, prop4, prop1_secondary, prop2_secondary, prop3_secondary)

    # Pedestrian way of testing for all possible issues
    func = pppp_pxxx_xpxx_xxpx
    mig = func(params[0])
    totmig = mig.sum(axis=1)


    violation = min(violation, -abs(totmig[-1]-1) + 1e-8) # Check that initial migration sums to 1.
    violation = min(violation, -totmig[0], -totmig[1])  # Check that there are no migrations in the last
                                                        # two generations (Sac could have migrants in last two generations?)


    violation = min(violation, min(1-totmig), min(totmig)) # Check that total migration rates between 0 and 1

    violation = min(violation, tstart-t3)
    violation = min(violation, tstart-t4)
    violation = min(violation, tstart-t5) #Added this for t4 and t5"

    violation = min(violation, t3)
    violation = min(violation, t4) #Added this for t4"
    violation = min(violation, t5) #Added this for t5"
    violation = min(violation, tstart)
    return violation

# We don't have to calculate all the tract length distributions to have the
# global ancestry proportion right. Here we define a function that
# automatically adjusts the migration rates to have the proper global ancestry
# proportion, saving a lot of optimization time!
# We first define a function that calculates the final proportions of ancestry
# based on the migration matrix
def propfrommig(mig):
    """Obtains the proportion of present day genomes contributed by each population"""

    current_contributions = mig[-1,:]
    for row in mig[-2::-1,:]:
        current_contributions = current_contributions*(1-numpy.sum(row))+row
    return current_contributions

def pppp_pxxx_xpxx_xxpx_fix(params, fracs):
    # An auxiliary function that, given the "missing" parameters, returns the
    # full migration matrix
    (tstart, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary) = params #Added t4 t5"
    n_pops = 4
    def fun(params ):
        """function taking the missing parameters and compares results to expected ancestry proportions"""
        (prop1, prop3, prop4) = params
        return propfrommig(pppp_pxxx_xpxx_xxpx((prop1, tstart, prop3, prop4, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)))[0:n_pops-1] - fracs[0:n_pops-1] #Added t4, t5"

    # Find the best-fitting parameters
    # (.2,.2) is just the starting point for the optimization function, it
    # should not be sensitive to this, but it's better to start with reasonable
    # parameter values.
    (prop1, prop3, prop4) = scipy.optimize.fsolve(fun, (.3, .2, .2))   # Added .3 here for pop1 (San) as we now have 4-way admixture
    return pppp_pxxx_xpxx_xxpx((prop1, tstart, prop3, prop4, t3, t4, t5,prop1_secondary, prop2_secondary, prop3_secondary)) #Added t4, t5"

def outofbounds_pppp_pxxx_xpxx_xxpx_fix(params, fracs):
    # constraint function evaluating below zero when constraints not satisfied
    n_pops = 4
    (tstart, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary) = params #Added t4 t5"
    if tstart > 1:
        print("time above 100 generations!")
        return (1 - tstart)

    def fun(params):
        (prop1, prop3, prop4) = params
        return propfrommig(pppp_pxxx_xpxx_xxpx((prop1, tstart, prop3, prop4, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)))[0:n_pops-1] - fracs[0:n_pops-1] #Added t4 t5"


    (prop1, prop3, prop4) = scipy.optimize.fsolve(fun, (.3, .2, .2)) #(.3, .2, .2) is just the starting point for the
                                                                    # optimization function, it should not be sensitive
                                                # to this, but it's better to start with reasonable parameter values.



    return outofbounds_pppp_pxxx_xpxx_xxpx((prop1, tstart, prop3, prop4, t3, t4, t5, prop1_secondary, prop2_secondary, prop3_secondary)) #Added t4, t5"
