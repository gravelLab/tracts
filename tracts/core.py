import sys

import numpy as np
import scipy.optimize
from matplotlib import pylab

from tracts.phase_type_distribution import PhTMonoecious, PhTDioecious
from tracts.demography.demographic_model import DemographicModel
from tracts.demography.composite_demographic_model import CompositeDemographicModel
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.population import Population
from tracts.util import eprint

#: Counts calls to object_func
_counter = 0

def plotmig(mig, colordict=None, order=None):
    if colordict is None:
        colordict = {'CEU': 'red', 'NAH': 'orange', 'NAT': 'orange', 'UNKNOWN': 'gray', 'YRI': 'blue'}
    if order is None:
        order = ['CEU', 'NAT', 'YRI']

    pylab.figure()
    axes = pylab.axes()
    shape = mig.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            c = pylab.Circle((j, i), radius=np.sqrt(mig[i, j]) / 1.7, color=colordict[order[j]])
            axes.add_patch(c)
    pylab.axis('scaled')
    pylab.ylabel("generations from present")


def optimize(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None,
             cutoff=0, verbose=0, flush_delay=0.5, epsilon=1e-3, gtol=1e-5,
             maxiter=None, full_output=True, func_args=None, fixed_params=None,
             ll_scale=1):
    """
    Optimizes model parameters using the BFGS method. Best suited for cases where initial values are close to the optimum, converging to a single minimum, and for parameters spanning different scales.
    
    Parameters
    ----------

        p0: 
            Initial parameters.
        data:
            Spectrum with data.
        model_function:
            Function to evaluate model spectrum. Should take arguments (params,
           pts).
        out_of_bounds_fun: default: None
           A funtion evaluating to True if the current parameters are in a
           forbidden region.
        cutoff: default: 0   
           The number of bins to drop at the beginning of the array. This could be
           achieved with masks.
        verbose: default: 0
            If greater than zero, print optimization status every ``verbose`` steps.
        flush_delay: default: 0.5
            Standard output will be flushed once every ``flush_delay`` minutes. This
           is useful to avoid overloading I/O on clusters.
        epsilon: default: 1e-3
            Step-size to use for finite-difference derivatives.
        gtol: default: 1e-5
            Convergence criterion for optimization. For more info, see
            ``help(scipy.optimize.fmin_bfgs)``.
        maxiter: default: None
            Maximum iterations to run for.
        full_output: default: True
          If True, return full outputs as described in ``help(scipy.optimize.fmin_bfgs)``.
        func_args: default: None
            List of additional arguments to ``model_func``. It is assumed that ``model_func``'s
            first argument is an array of parameters to optimize.
        fixed_params: default: None
            (Not yet implemented). If not None, should be a list used to fix model parameters at
            particular values. For example, if the model parameters are
            ``(nu1,nu2,T,m)``, then ``fixed_params = [0.5,None,None,2]`` will hold ``nu1=0.5``
            and ``m=2``. The optimizer will only change ``T`` and ``m``. Note that the bounds
            lists must include all parameters. Optimization will fail if the fixed
            values lie outside their bounds. A full-length ``p0`` should be passed in;
            values corresponding to fixed parameters are ignored.
        ll_scale: default: 1
            The BFGS algorithm may fail if the initial log-likelihood is too large. Using ``ll_scale > 1`` reduces the log-likelihood magnitude, helping the optimizer reach a reasonable region. Afterward, re-optimize with ``ll_scale=1``.
    """
    args = (bins, Ls, data, nsamp, model_func, outofbounds_fun, cutoff, verbose, flush_delay, func_args)
    if func_args is None:
        func_args = []
    if fixed_params is not None:
        raise ValueError("Fixed parameters not implemented in optimize_bfgs.")

    outputs = scipy.optimize.fmin_bfgs(_object_func, p0, epsilon=np.array(epsilon), args=args, gtol=gtol,
                                       full_output=full_output, disp=False, maxiter=maxiter)
    (xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag) = outputs

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


optimize_bfgs = optimize


def optimize_cob(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=1,
                 epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=None, fixed_params=None,
                 ll_scale=1, reset_counter=True, modelling_method=DemographicModel) -> np.ndarray:
    """
    Optimizes model parameters using the COBYLA method. Best suited for cases where initial values are close to the optimum, converging to a single minimum, and for parameters spanning different scales.
    
    Parameters
    ----------

        p0: 
            Initial parameters.
        data:
            Spectrum with data.
        model_function:
            Function to evaluate model spectrum. Should take arguments (params,
           pts).
        out_of_bounds_fun: default: None
           A funtion evaluating to True if the current parameters are in a
           forbidden region.
        cutoff: default: 0   
           The number of bins to drop at the beginning of the array. This could be
           achieved with masks.
        verbose: default: 0
            If greater than zero, print optimization status every ``verbose`` steps.
        flush_delay: default: 0.5
            Standard output will be flushed once every ``flush_delay`` minutes. This
           is useful to avoid overloading I/O on clusters.
        epsilon: default: 1e-3
            Step-size to use for finite-difference derivatives.
        gtol: default: 1e-5
            Convergence criterion for optimization. For more info, see
            ``help(scipy.optimize.fmin_bfgs)``.
        maxiter: default: None
            Maximum iterations to run for.
        full_output: default: True
          If True, return full outputs as described in ``help(scipy.optimize.fmin_bfgs)``.
        func_args: default: None
            List of additional arguments to ``model_func``. It is assumed that ``model_func``'s
            first argument is an array of parameters to optimize.
        fixed_params: default: None
            (Not yet implemented). If not None, should be a list used to fix model parameters at
            particular values. For example, if the model parameters are
            ``(nu1,nu2,T,m)``, then ``fixed_params = [0.5,None,None,2]`` will hold ``nu1=0.5``
            and ``m=2``. The optimizer will only change ``T`` and ``m``. Note that the bounds
            lists must include all parameters. Optimization will fail if the fixed
            values lie outside their bounds. A full-length ``p0`` should be passed in;
            values corresponding to fixed parameters are ignored.
        ll_scale: default: 1
            The BFGS algorithm may fail if the initial log-likelihood is too large. Using ``ll_scale > 1`` reduces the log-likelihood magnitude, helping the optimizer reach a reasonable region. Afterward, re-optimize with ``ll_scale=1``.
        reset_counter: default: True
            Resets the iteration counter to zero. Set to False to
            continue iteration count (e.g., if optimization continues from previous point).
    """
    print(modelling_method)
    if func_args is None:
        func_args = []
    if reset_counter:
        global _counter
        _counter = 0

    fun = lambda x: _object_func(x, bins, Ls, data, nsamp, model_func, outofbounds_fun=outofbounds_fun, cutoff=cutoff,
                                 verbose=verbose, flush_delay=flush_delay, func_args=func_args,
                                 modelling_method=modelling_method)

    outputs = scipy.optimize.fmin_cobyla(
        fun, p0, outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)

    return outputs

def optimize_cob_sex_biased(p0, population: Population, model_func, outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=1,
                 epsilon=1e-3, gtol=1e-5, p_dict=None, exclude_tracts_below_cM=0, maxiter=None, full_output=True, func_args=None, fixed_params=None,
                 ll_scale=1, reset_counter=True, modelling_method=PhTDioecious, D_model='DC', npts=50) -> tuple[np.ndarray, float]:
    if reset_counter:
        global _counter
        _counter = 0
    
    autosome_bins, autosome_data = population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM) 
    n_autosome_bins = len(autosome_bins)
    allosome_bins, allosome_data = population.get_global_allosome_tractlengths('X', npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
    n_allosome_bins = len(allosome_bins)
    allosome_length = population.allosome_lengths['X']
    female_data = allosome_data[SexType.FEMALE]
    male_data = allosome_data[SexType.MALE]
    num_males = population.num_males
    num_females = population.num_females

    autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
    for k, v in autosome_data.items():
        autosome_data_mapped[dict(p_dict)[k]] = v
        
    female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
    for k, v in female_data.items():
        female_data_mapped[dict(p_dict)[k]] = v
        
    male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
    for k, v in male_data.items():
        male_data_mapped[dict(p_dict)[k]] = v
    
    def objective_function(parameters):
        _out_of_bounds_val = -1e32
        global _counter
        _counter += 1

        def flush_result(result, note = str()):
            if (verbose > 0) and (_counter % verbose == 0):
                param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in parameters]))
                eprint('%-8i, %-12g, %s, %s' % (_counter, result, param_str, note))
                # Misc.delayed_flush(delay=flush_delay)

        if outofbounds_fun is not None:
            # outofbounds can return either True or a negative value to signify out-of-boundedness.
            ooa = outofbounds_fun(parameters)
            if ooa < 0:
                flush_result((ooa - 1) * _out_of_bounds_val)
                return (ooa - 1) * _out_of_bounds_val
        else:
            eprint("No bound function defined")

        matrices = model_func(parameters)
        [male_matrix, female_matrix] = [matrix for matrix in matrices.values()]
        
        result_autosomes = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=D_model).loglik(autosome_bins, population.Ls, [mat for mat in autosome_data_mapped], len(population.indivs))
        result_X_females = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=D_model, X_chromosome=True).loglik(allosome_bins, [allosome_length], [mat for mat in female_data_mapped], num_females)
        result_X_males = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=D_model, X_chromosome=True, X_chromosome_male=True).loglik(allosome_bins, [allosome_length], [mat for mat in male_data_mapped], num_males)
        result = (result_autosomes + result_X_females + result_X_males)
        flush_result(result_autosomes, 'Autosomes')
        flush_result(result_X_females, 'Female allosomes')
        flush_result(result_X_males, 'Male allosomes')
        
        return -result
    
    print('\n---------------------------\nOptimizing model likelihood.\n---------------------------\nIter.\t Log-likelihood\t Model parameters \t\t Transmission\n---------------------------------------------------------------------\n')
           
    outputs = scipy.optimize.fmin_cobyla(
        objective_function, p0, outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)
    
    likelihood = objective_function(outputs)

    return outputs, likelihood



def optimize_slsqp(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None, cutoff=0, bounds=None, verbose=0,
                   flush_delay=1, epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=None,
                   fixed_params=None, ll_scale=1, reset_counter=True):
    """
    Optimizes model parameters using the SLSQ method. 
    
    Parameters
    ----------

        p0: 
            Initial parameters.
        data:
            Spectrum with data.
        model_function:
            Function to evaluate model spectrum. Should take arguments (params,
           pts).
        out_of_bounds_fun: default: None
           A funtion evaluating to True if the current parameters are in a
           forbidden region.
        cutoff: default: 0   
           The number of bins to drop at the beginning of the array. This could be
           achieved with masks.
        verbose: default: 0
            If greater than zero, print optimization status every ``verbose`` steps.
        flush_delay: default: 0.5
            Standard output will be flushed once every ``flush_delay`` minutes. This
           is useful to avoid overloading I/O on clusters.
        epsilon: default: 1e-3
            Step-size to use for finite-difference derivatives.
        gtol: default: 1e-5
            Convergence criterion for optimization. For more info, see
            ``help(scipy.optimize.fmin_bfgs)``.
        maxiter: default: None
            Maximum iterations to run for.
        full_output: default: True
          If True, return full outputs as described in ``help(scipy.optimize.fmin_bfgs)``.
        func_args: default: None
            List of additional arguments to ``model_func``. It is assumed that ``model_func``'s
            first argument is an array of parameters to optimize.
        fixed_params: default: None
            (Not yet implemented). If not None, should be a list used to fix model parameters at
            particular values. For example, if the model parameters are
            ``(nu1,nu2,T,m)``, then ``fixed_params = [0.5,None,None,2]`` will hold ``nu1=0.5``
            and ``m=2``. The optimizer will only change ``T`` and ``m``. Note that the bounds
            lists must include all parameters. Optimization will fail if the fixed
            values lie outside their bounds. A full-length ``p0`` should be passed in;
            values corresponding to fixed parameters are ignored.
        ll_scale: default: 1
            The BFGS algorithm may fail if the initial log-likelihood is too large. Using ``ll_scale > 1`` reduces the log-likelihood magnitude, helping the optimizer reach a reasonable region. Afterward, re-optimize with ``ll_scale=1``.
    

    Notes   
    -------
    Best suited for cases where initial values are close to the optimum, converging to a single minimum, and for parameters spanning different scales.
    
    """
    args = (bins, Ls, data, nsamp, model_func, outofbounds_fun, cutoff, verbose, flush_delay, func_args)
    if bounds is None:
        bounds = []
    if func_args is None:
        func_args = []
    if reset_counter:
        global _counter
        _counter = 0

    def onearg(a, *args):
        return outofbounds_fun(a)

    if maxiter is None:
        maxiter = 100

    outputs = scipy.optimize.fmin_slsqp(_object_func, p0, ieqcons=[onearg], bounds=bounds, args=args, iter=maxiter,
                                        acc=1e-4, epsilon=1e-4)

    return outputs
    # xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    # xopt = _project_params_up(np.exp(xopt), fixed_params)
    #
    # if not full_output:
    #    return xopt
    # else:
    #    return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def _project_params_down(pin, fixed_params):
    """ Eliminate fixed parameters from *pin*. Copied from Dadi (Gutenkunst *et al*., PLoS Genetics, 2009). """
    if fixed_params is None:
        return pin

    if len(pin) != len(fixed_params):
        raise ValueError('fixed_params list must have same length as input parameter array.')

    pout = []
    for ii, (curr_val, fixed_val) in enumerate(zip(pin, fixed_params)):
        if fixed_val is None:
            pout.append(curr_val)

    return np.array(pout)


def _project_params_up(pin, fixed_params):
    """ Folds fixed parameters into *pin*. Copied from Dadi (Gutenkunst *et al.*,
        PLoS Genetics, 2009). """
    if fixed_params is None:
        return pin

    pout = np.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout


# #: Counts calls to object_func
# _counter = 0


def _object_func(params, bins, Ls, data, nsamp, model_func, outofbounds_fun=None, cutoff=0, verbose=0,
                 flush_delay=0, func_args=None, modelling_method=DemographicModel):
    """Calculates the log-likelihood value for tract length data."""
    if func_args is None:
        func_args = []
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative value to signify out-of-boundedness.
        ooa = outofbounds_fun(params)
        if ooa < 0:
            result = -(ooa - 1) * _out_of_bounds_val
        else:
            mod = modelling_method(model_func(params))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod = modelling_method(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if True:  # (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in params]))
        eprint('%-8i, %-12g, %s' % (_counter, result, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -result


def optimize_cob_fracs(p0, bins, Ls, data, nsamp, model_func, fracs, outofbounds_fun=None, cutoff=0, verbose=0,
                       flush_delay=1, epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=None,
                       fixed_params=None, ll_scale=1):
    """
    Optimizes parameters to fit the model to data using the COBYLA method. This optimization performs well when the starting point is reasonably close to the optimum and is particularly effective at converging to a single minimum. It also tends to perform better when parameters vary across different scales.
    
    Best suited for cases where initial values are close to the optimum, converging to a single minimum, and for parameters spanning different scales.
    """
    if func_args is None:
        func_args = []

    args = (bins, Ls, data, nsamp, model_func, fracs, outofbounds_fun, cutoff, verbose, flush_delay, func_args)

    outfun = lambda x: outofbounds_fun(x, fracs=fracs)

    outputs = scipy.optimize.fmin_cobyla(_object_func_fracs, p0, outfun, rhobeg=.01, rhoend=.001,
                                         args=args, maxfun=maxiter)

    return outputs


def optimize_cob_fracs2(p0, bins, Ls, data, nsamp, model_func, fracs, outofbounds_fun=None, cutoff=0,
                        verbose=0, flush_delay=1, epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True,
                        func_args=None, fixed_params=None, ll_scale=1, reset_counter=True):
    """
    Optimize parameters to fit the model to data using the COBYLA method. 

    Parameters
    ----------
    p0: 
    	Initial parameters.
    	
    data:
    	Spectrum with data.
    	
    model_function: 
    	Function to evaluate model spectrum. Should take arguments (params, pts).
    	
    out_of_bounds_fun:
    	A function evaluating to True if the current parameters are in a forbidden region.
    	
    cutoff:
    	The number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose:
    	If > 0, print optimization status every ``verbose`` steps.
    	
    flush_delay:
    	Standard output will be flushed once every ``flush_delay`` minutes. This is useful to avoid overloading I/O on clusters.
    	
    epsilon:
    	Step-size to use for finite-difference derivatives.
    	
    gtol:
    	Convergence criterion for optimization. For more info, see ``help(scipy.optimize.fmin_bfgs)``.

    maxiter:
    	Maximum iterations to run for.
    	
    full_output:
    	If True, return full outputs as in described in ``help(scipy.optimize.fmin_bfgs)``.
    	               
    func_args:
        Additional arguments to ``model_func``. It is assumed that ``model_func``'s
        first argument is an array of parameters to optimize.
        
    fixed_params:
        (Not yet implemented) If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters are
        ``(nu1,nu2,T,m)``, then ``fixed_params = [0.5,None,None,2]`` will hold ``nu1=0.5``
        and ``m=2``. The optimizer will only change ``T`` and ``m``. Note that the bounds
        lists must include all parameters. Optimization will fail if the fixed
        values lie outside their bounds. A full-length ``p0`` should be passed in;
        values corresponding to fixed parameters are ignored.
        
    ll_scale:
        The BFGS algorithm may fail if the initial log-likelihood is too large. Using ``ll_scale > 1`` reduces the log-likelihood magnitude, helping the optimizer reach a reasonable region. Afterward, re-optimize with ``ll_scale=1``.
    

    Notes
    -------
    This optimization performs well when the starting point is reasonably close to the optimum and is particularly effective at converging to a single minimum. It also tends to perform better when parameters vary across different scales.
 
    """
    if func_args is None:
        func_args = []

    if reset_counter:
        global _counter
        _counter = 0

    def outfun(p0, verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence, we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs=fracs))

        return outofbounds_fun(x0, fracs=fracs)

    def modstrip(x):
        return model_func(x, fracs=fracs)

    def fun(x):
        return _object_func_fracs2(x, bins, Ls, data, nsamp, modstrip, outofbounds_fun=outfun, cutoff=cutoff,
                                   verbose=verbose, flush_delay=flush_delay, func_args=func_args,
                                   fixed_params=fixed_params)

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_cobyla(fun, p0, outfun, rhobeg=.01, rhoend=.001, maxfun=maxiter)
    xopt = _project_params_up(outputs, fixed_params)

    return xopt


def optimize_cob_multifracs(p0, bins, Ls, data_list, nsamp_list, model_func, fracs_list, outofbounds_fun=None,
                            cutoff=0, verbose=0, flush_delay=1, epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True,
                            func_args=None, fixed_params=None, ll_scale=1):
    """
    Optimizes parameters to fit the model to data using the COBYLA method. 
    
    Notes
    -------

    This optimization performs well when the starting point is reasonably close to the optimum and is particularly effective at converging to a single minimum. It also tends to perform better when parameters vary across different scales.
    """
      
    if func_args is None:
        func_args = []

    # Now we iterate over each set of ancestry proportions in the list, and
    # construct the outofbounds functions and the model functions, storing
    # each into the empty lists defined above.
    # construct the out-of-bounds function.

    def outfun(p0, fracs, verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence, we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs=fracs))

        return outofbounds_fun(x0, fracs=fracs)

    # construct the objective function. The input x is wrapped in the
    # function r constructed above.
    def objfun(x):
        return _object_func_multifracs(x, bins, Ls, data_list, nsamp_list, model_func, fracs_list,
                                       outofbounds_fun=outfun, cutoff=cutoff, verbose=verbose,
                                       flush_delay=flush_delay, func_args=func_args, fixed_params=fixed_params)

    def composite_outfun(x):
        return min(outfun(x, frac) for frac in fracs_list)

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_cobyla(objfun, p0, composite_outfun, rhobeg=.01, rhoend=.001, maxfun=maxiter)
    xopt = _project_params_up(outputs, fixed_params)

    return xopt


def optimize_brute_fracs2(bins, Ls, data, nsamp, model_func, fracs, searchvalues, outofbounds_fun=None, cutoff=0,
                          verbose=0, flush_delay=1, full_output=True, func_args=None, fixed_params=None, ll_scale=1):
    """
    Optimize parameters to fit the model to data using the brute-force method. 

    Parameters
    ----------
    p0: 
    	Initial parameters.
    	
    data:
    	Spectrum with data.
    	
    model_function: 
    	Function to evaluate model spectrum. Should take arguments (params, pts).
    	
    out_of_bounds_fun:
    	A function evaluating to True if the current parameters are in a forbidden region.
    	
    cutoff:
    	The number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose:
    	If > 0, print optimization status every ``verbose`` steps.
    	
    flush_delay:
    	Standard output will be flushed once every ``flush_delay`` minutes. This is useful to avoid overloading I/O on clusters.
    	
    epsilon:
    	Step-size to use for finite-difference derivatives.
    	
    gtol:
    	Convergence criterion for optimization. For more info, see ``help(scipy.optimize.fmin_bfgs)``.

    maxiter:
    	Maximum iterations to run for.
    	
    full_output:
    	If True, return full outputs as in described in ``help(scipy.optimize.fmin_bfgs)``.
    	             
    func_args:
        Additional arguments to ``model_func``. It is assumed that ``model_func``'s
        first argument is an array of parameters to optimize.
        
    fixed_params:
        (Not yet implemented) If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters are
        ``(nu1,nu2,T,m)``, then ``fixed_params = [0.5,None,None,2]`` will hold ``nu1=0.5``
        and ``m=2``. The optimizer will only change ``T`` and ``m``. Note that the bounds
        lists must include all parameters. Optimization will fail if the fixed
        values lie outside their bounds. A full-length ``p0`` should be passed in;
        values corresponding to fixed parameters are ignored.
        
    ll_scale:
        The BFGS algorithm may fail if the initial log-likelihood is too large. Using ``ll_scale > 1`` reduces the log-likelihood magnitude, helping the optimizer reach a reasonable region. Afterward, re-optimize with ``ll_scale=1``.

    Notes
    -------
    This optimization performs well when the starting point is reasonably close to the optimum and is most effective at converging to a single minimum. It also tends to perform better when parameters vary across different scales.
    
    """
    if func_args is None:
        func_args = []

    def outfun(p0, verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence, we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs=fracs))

        return outofbounds_fun(x0, fracs=fracs)

    def modstrip(x):
        return model_func(x, fracs=fracs)

    def fun(x):
        return _object_func_fracs2(x, bins, Ls, data, nsamp, modstrip, outofbounds_fun=outfun, cutoff=cutoff,
                                   verbose=verbose, flush_delay=flush_delay, func_args=func_args,
                                   fixed_params=fixed_params)

    if len(searchvalues) == 1:
        def fun2(x):
            return fun((float(x),))
    else:
        fun2 = fun

    outputs = scipy.optimize.brute(fun2, searchvalues, full_output=full_output)
    xopt = _project_params_up(outputs[0], fixed_params)

    return xopt, outputs[1:]


def optimize_brute_multifracs(bins, Ls, data_list, nsamp_list, model_func, fracs_list, searchvalues,
                              outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=1,
                              full_output=True, func_args=None, fixed_params=None, ll_scale=1):
    """
    Optimizes parameters to fit the model to data using the brute-force method. 

    Parameters
    ----------
    p0: 
    	Initial parameters.
    	
    data:
    	Spectrum with data.
    	
    model_function: 
    	Function to evaluate model spectrum. Should take arguments (params, pts).
    	
    out_of_bounds_fun:
    	A function evaluating to True if the current parameters are in a forbidden region.
    	
    cutoff:
    	The number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose:
    	If > 0, print optimization status every ``verbose`` steps.
    	
    flush_delay:
    	Standard output will be flushed once every ``flush_delay`` minutes. This is useful to avoid overloading I/O on clusters.
    	
    epsilon:
    	Step-size to use for finite-difference derivatives.
    	
    gtol:
    	Convergence criterion for optimization. For more info, see ``help(scipy.optimize.fmin_bfgs)``.

    maxiter:
    	Maximum iterations to run for.
    	
    full_output:
    	If True, return full outputs as in described in ``help(scipy.optimize.fmin_bfgs)``.
    	             
    func_args:
        Additional arguments to ``model_func``. It is assumed that ``model_func``'s
        first argument is an array of parameters to optimize.
        
    fixed_params:
        (Not yet implemented) If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters are
        ``(nu1,nu2,T,m)``, then ``fixed_params = [0.5,None,None,2]`` will hold ``nu1=0.5``
        and ``m=2``. The optimizer will only change ``T`` and ``m``. Note that the bounds
        lists must include all parameters. Optimization will fail if the fixed
        values lie outside their bounds. A full-length ``p0`` should be passed in;
        values corresponding to fixed parameters are ignored.
        
    ll_scale:
        The BFGS algorithm may fail if the initial log-likelihood is too large. Using ``ll_scale > 1`` reduces the log-likelihood magnitude, helping the optimizer reach a reasonable region. Afterward, re-optimize with ``ll_scale=1``.

    Notes
    -------
    This optimization performs well when the starting point is reasonably close to the optimum and is most effective at converging to a single minimum. It also tends to perform better when parameters vary across different scales.
"""
    if func_args is None:
        func_args = []

    # construct the out-of-bounds function.
    def outfun(p0, fracs, verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence, we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs=fracs))

        return outofbounds_fun(x0, fracs=fracs)

    # construct a wrapper function that will tuple up its argument in the case
    # where searchvalues has length 1; in that case, the optimizer expects a
    # tuple (it always wants tuples), but the input will be a single float.
    # Hence, why we need to tuple it up.
    # The wrapper function is called on the x given as input to
    # _object_func_multifracs
    r = (lambda x: x) \
        if len(searchvalues) > 1 else \
        (lambda x: (float(x),))

    # construct the objective function. The input x is wrapped in the
    # function r constructed above.
    def objfun(x):
        return _object_func_multifracs(r(x), bins, Ls, data_list, nsamp_list, model_func, fracs_list,
                                       outofbounds_fun=outfun, cutoff=cutoff, verbose=verbose, flush_delay=flush_delay,
                                       func_args=func_args, fixed_params=fixed_params)

    outputs = scipy.optimize.brute(objfun, searchvalues, full_output=full_output)
    xopt = _project_params_up(outputs[0], fixed_params)

    return xopt, outputs[1:]


def test_model_func(model_func, parameters, fracs_list=None, time_params=True, time_scale=100):
    """Given a demographic model function, run a few debugging tests to ensure
    that it behaves as expected, namely: (i) that migration matrices sum to less than one (exactly one for the last generation)
    and (ii) that it behaves continuously relative to time parameters.
    
    Parameters
    ----------
        model_func: 
    	    A migration model. It takes in parameters and outputs a migration matrix. 
        parameters:
    	    Parameters for which the model will be tested. 
        fracs_list: default: None
    	    Parameters required by some demographic models corresponding to the observed proportion of ancestry from each source population.
        time_params: default: True
    	    If True, test all parameters for continuity as if they were time parameters. If a list of boolean values of the same length of parameters, only test parameters corresponding to True values.
        time_scale: default: 100
    	    The scaling of the time variables: time (in generations) = time_parameter*time_scale. This is used to
            test continuity around integer values. 
    
    Returns
    ----------
    Violation score (negative means that a violation has occurred) and the migration matrix value.
    """

    # First test consistency of migration matrix
    if fracs_list is None:
        mig = model_func(parameters)
    else:
        assert (np.sum(fracs_list) == 1), "fracs_list should sum to 1"
        mig = model_func(parameters, fracs=fracs_list)

    totmig = mig.sum(axis=1)
    violation = 1

    if -abs(totmig[-1] - 1) < - 1e-8:
        violation = min(violation, -abs(totmig[-1] - 1) + 1e-8)  # Check that initial migration sums to 1.
        print("last row of migration matrix should sum to one.")
    if totmig[0] > 0 or totmig[1] > 0:
        print("first two rows of the migration matrix should sum to one")
        violation = min(violation, -totmig[0], -totmig[1])  # Check that there are no migrations in the last
        # two generations
    if max(totmig) > 1 or min(totmig) < 0:
        print("migration rates should be between zero and one")
        violation = min(violation, min(1 - totmig), min(totmig))  # Check that total migration rates between 0 and 1

    # Second, test continuity
    if time_params is True:  # Test continuity on all parameters as time parameters.
        time_params = [True] * len(parameters)

    assert len(time_params) == len(parameters), "time_params should be a boolean list with length len(parameters)"

    perturbation = 10 ** -15
    for i in range(len(parameters)):
        if time_params[i]:
            focal_parameter = parameters[i]
            # Round parameter to integer time
            focal_parameter = round(time_scale * focal_parameter) * 1. / time_scale
            up_param = focal_parameter + perturbation
            down_param = focal_parameter - perturbation

            up_params = list(parameters)
            up_params[i] = up_param
            down_params = list(parameters)
            down_params[i] = down_param
            if fracs_list is None:
                up_mig = model_func(up_params)
                down_mig = model_func(down_params)
            else:
                up_mig = model_func(up_params, fracs_list)
                down_mig = model_func(down_params, fracs_list)

            # mig_down should always be smaller or equal in size to mig_up
            compare_size = down_mig.shape
            trimmed_up_mig = up_mig[:compare_size[0], :]
            max_diff = abs(trimmed_up_mig - down_mig).max()

            if max_diff > 10 * time_scale * perturbation:  # This is fairly arbitrary threshold.
                print("apparent discontinuity in migration matrices in model test at parameters", parameters)
                # print(up_mig)
                # print(down_mig)
                violation = min(violation, 10 * time_scale * perturbation - max_diff)

    return violation, mig


#: Counts calls to object_func
# _counter = 0


def _object_func_fracs(params, bins, Ls, data, nsamp, model_func, fracs, outofbounds_fun=None, cutoff=0, verbose=0,
                       flush_delay=0, func_args=None):
    """Define the objective function for when the ancestry porportions are specified."""
    if func_args is None:
        func_args = []
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative valueto signify out-of-boundedness.
        oob = outofbounds_fun(params, fracs=fracs)
        if oob < 0:
            result = -(oob - 1) * _out_of_bounds_val
        else:
            mod = DemographicModel(model_func(params, fracs=fracs))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod = DemographicModel(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if verbose > 0 and _counter % verbose == 0:
        param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in params]))
        eprint('%-8i, %-12g, %s' % (_counter, result, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -result


def _object_func_fracs2(params, bins, Ls, data, nsamp, model_func, outofbounds_fun=None, cutoff=0, verbose=0,
                        flush_delay=0, func_args=None, fixed_params=None):
    if func_args is None:
        func_args = []
    # this function will be minimized. We first calculate likelihoods (to be
    # maximized), and return minus that.
    eprint("evaluating at params", params)
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    if outofbounds_fun is not None:
        # outofbounds returns  a negative value to signify out-of-boundedness.
        oob = outofbounds_fun(params)

        if oob < 0:
            # we want bad functions to give very low likelihoods, and worse
            # likelihoods when the function is further out of bounds.
            mresult = - (oob - 1) * _out_of_bounds_val
            # challenge: if outofbounds is very close to 0, this can return a
            # reasonable likelihood. When oob is negative, we take away an
            # extra 1 to make sure this cancellation does not happen.
        else:
            mod = DemographicModel(model_func(params_up))

            sys.stdout.flush()
            mresult = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod = DemographicModel(model_func(params_up))
        mresult = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if True:  # (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in params_up]))
        eprint('%-8i, %-12g, %s' % (_counter, mresult, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -mresult


def _object_func_multifracs(params, bins, Ls, data_list, nsamp_list, model_func, fracs_list, outofbounds_fun=None,
                            cutoff=0, verbose=0, flush_delay=0, func_args=None, fixed_params=None):
    """Defines the objective function for when the ancestry porportions are specified."""
    if func_args is None:
        func_args = []
    # this function will be minimized. We first calculate likelihoods (to be
    # maximized), and return minus that.
    eprint("evaluating at params", params)
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    def mkmodel():
        return CompositeDemographicModel(model_func, params, fracs_list)

    if outofbounds_fun is not None:
        # outofbounds returns  a negative value to signify out-of-boundedness.
        # Compute the out-of-bounds function for each fraction and take the
        # minimum as the overall out of bounds value.
        oob = min(outofbounds_fun(params, fracs=fracs) for fracs in fracs_list)

        if oob < 0:
            # we want bad functions to give very low likelihoods, and worse
            # likelihoods when the function is further out of bounds.
            mresult = -(oob - 1) * _out_of_bounds_val
            # challenge: if outofbounds is very close to 0, this can return a
            # reasonable likelihood. When oob is negative, we take away an
            # extra 1 to make sure this cancellation does not happen.
        else:
            comp_model = mkmodel()

            sys.stdout.flush()

            mresult = comp_model.loglik(bins, Ls, data_list, nsamp_list, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        comp_model = mkmodel()

        mresult = comp_model.loglik(bins, Ls, data_list, nsamp_list)

    if True:  # (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in params_up]))
        eprint('%-8i, %-12g, %s' % (_counter, mresult, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -mresult
