import sys

import logging
import numpy as np
import scipy.optimize
from matplotlib import pylab
import copy

import tracts.hybrid_pedigree as HP
from tracts.phase_type_distribution import PhTMonoecious, PhTDioecious
from tracts.demography.demographic_model import DemographicModel
from tracts.demography.composite_demographic_model import CompositeDemographicModel
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.population import Population
from tracts.util import eprint
from tracts.demography.parameter import ParamType

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

#: Counts calls to object_func
_counter = 0
_out_of_bounds_val = -1e32
_min_out_of_bounds_val = -1e-10

def optimize_cob(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=1,
                 maxiter=None, func_args=None, reset_counter=True, modelling_method=PhTMonoecious) -> np.ndarray:
    """
    Optimizes model parameters using the COBYLA method. Best suited for cases where initial values are close to the optimum, converging to a single minimum, and for parameters spanning different scales.
    
    Parameters
    ----------

        p0: 
            Initial parameters.
        data:
            Spectrum with data.
        model_func:
            Function to evaluate model spectrum. Should take arguments (params,
           pts).
        cutoff: default: 0   
           The number of bins to drop at the beginning of the array. This could be
           achieved with masks.
        verbose: default: 0
            If greater than zero, print optimization status every ``verbose`` steps.
        flush_delay: default: 0.5
            Standard output will be flushed once every ``flush_delay`` minutes. This
           is useful to avoid overloading I/O on clusters.
        maxiter: default: None
            Maximum iterations to run for.
        func_args: default: None
            List of additional arguments to ``model_func``. It is assumed that ``model_func``'s
            first argument is an array of parameters to optimize.
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

def optimize_cob_sex_biased(p0, population: Population, model_func, parameter_handler, outofbounds_fun=None, verbose=0, p_dict=None, exclude_tracts_below_cM=0, 
                            maxiter=None, reset_counter=True, ad_model_autosomes='DC', ad_model_allosomes='DC', npts=50) -> tuple[np.ndarray, float]:
    """times in parameters here should be in scaled units (i.e., times are divided by scale), and model_func should take care of rescaling. """
    
    if reset_counter:
        global _counter
        _counter = 0
    
    autosome_bins, autosome_data = population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM) 
    n_autosome_bins = len(autosome_bins)
    
    autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
    for k, v in autosome_data.items():
        autosome_data_mapped[dict(p_dict)[k]] = v
    
    if ad_model_allosomes is not None:
        allosome_bins, allosome_data = population.get_global_allosome_tractlengths('X', npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
        n_allosome_bins = len(allosome_bins)
        allosome_length = population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
        num_males = population.num_males
        num_females = population.num_females

           
        female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in female_data.items():
            female_data_mapped[dict(p_dict)[k]] = v
        
        male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in male_data.items():
            male_data_mapped[dict(p_dict)[k]] = v
    
    def objective_function(parameters):

        global _counter
        _counter += 1

        def flush_result(result, note = str()):
            if (verbose > 0) and (_counter % verbose == 0):
                param_str = 'ocsb: array([%s])' % (', '.join(['%- 12g' % v for v in parameter_handler.convert_to_physical_params(parameters)])) 
                logger.info(
                     "iter=%-6d | obj=%-12g | params=%s %s",
                        _counter,
                        result,
                        param_str,
                        note,
                    )
                
        if outofbounds_fun is not None:
            # outofbounds can return either True or a negative value to signify out-of-boundedness.
            oob = outofbounds_fun(parameters)
            if oob < 0:
                out = oob * _out_of_bounds_val-_min_out_of_bounds_val
                flush_result(out, f'OOB (oob={oob})')
                return out
        else:
            eprint("No bound function defined")

        matrices = model_func(parameters)
        matrix_list = [matrix for matrix in matrices.values()]
        if ad_model_allosomes is not None:
            [male_matrix, female_matrix] = matrix_list
        else:
            male_matrix = matrix_list[0]  # Unbiased migration
            female_matrix = matrix_list[0] 

        # Model for autosomes
        if ad_model_autosomes == 'M':
            model = PhTMonoecious(0.5*(female_matrix+male_matrix), rho=1)
            result_autosomes = model.loglik(autosome_bins, population.Ls, [mat for mat in autosome_data_mapped], len(population.indivs))
        elif ad_model_autosomes == 'H-DC':
            result_autosomes=HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DC', X_chr = False, X_chr_male = False, N_cores = 5, bins=autosome_bins, Ls=population.Ls, data=[mat for mat in autosome_data_mapped], num_samples=len(population.indivs), cutoff=0)
        elif ad_model_autosomes == 'H-DF':
            result_autosomes=HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DF', X_chr = False, X_chr_male = False, N_cores = 5, bins=autosome_bins, Ls=population.Ls, data=[mat for mat in autosome_data_mapped], num_samples=len(population.indivs), cutoff=0)
        else:
            result_autosomes = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_autosomes).loglik(autosome_bins, population.Ls, [mat for mat in autosome_data_mapped], len(population.indivs))
        
        if ad_model_allosomes is not None:
            # Model for allosomes
            if ad_model_allosomes == 'H-DC':
                result_X_females = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DC', X_chr = True, X_chr_male = False, N_cores = 5, bins=allosome_bins, Ls=[allosome_length], data=[mat for mat in female_data_mapped], num_samples=num_females, cutoff=0)
                result_X_males = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DC', X_chr = True, X_chr_male = True, N_cores = 5, bins=allosome_bins, Ls=[allosome_length], data=[mat for mat in male_data_mapped], num_samples=num_males, cutoff=0)
            elif ad_model_allosomes == 'H-DF':
                result_X_females = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DF', X_chr = True, X_chr_male = False, N_cores = 5, bins=allosome_bins, Ls=[allosome_length], data=[mat for mat in female_data_mapped], num_samples=num_females, cutoff=0)
                result_X_males = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DF', X_chr = True, X_chr_male = True, N_cores = 5, bins=allosome_bins, Ls=[allosome_length],data=[mat for mat in male_data_mapped], num_samples=num_males, cutoff=0)   
            else:
                result_X_females = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True).loglik(allosome_bins, [allosome_length], [mat for mat in female_data_mapped], num_females)
                result_X_males = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True, X_chromosome_male=True).loglik(allosome_bins, [allosome_length], [mat for mat in male_data_mapped], num_males)
        else:
            result_X_females = 0
            result_X_males = 0

        result = (result_autosomes + result_X_females + result_X_males)
                
        flush_result(result_autosomes, 'Autosomes')
        if ad_model_allosomes:
            flush_result(result_X_females, 'Female allosomes')
            flush_result(result_X_males, 'Male allosomes')
        
        return -result

    if ad_model_allosomes is not None:
        title_message = f"Admixture is modelled with the {ad_model_autosomes} model for autosomes and with the {ad_model_allosomes} model for allosomes."
    else:
        title_message = f"Admixture is modelled with the {ad_model_autosomes} model for autosomes."
    print('\n--------------------------------------------------------------------------------------------------')
    print(title_message)
    print('--------------------------------------------------------------------------------------------------')
    print('Optimizing model likelihood.\n---------------------------\nIter.\t Log-likelihood\t Model parameters \t\t Transmission\n---------------------------------------------------------------------\n')
           
    outputs = scipy.optimize.fmin_cobyla(
        objective_function, p0, outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)
    
    likelihood = -objective_function(outputs)

    return outputs, likelihood

def optimize_cob_sex_biased_fixed_values(p0, population: Population, model_func, parameter_handler, outofbounds_fun=None, verbose=0,
                 p_dict=None, exclude_tracts_below_cM=0, maxiter=None, reset_counter=True, ad_model_autosomes='DC', ad_model_allosomes='DC', npts=50) -> tuple[np.ndarray, float]:
    """times in parameters here should be in scaled units (i.e., times are divided by scale), and model_func should take care of rescaling. """

    if reset_counter:
        global _counter
        _counter = 0
    
    autosome_bins, autosome_data = population.get_global_tractlengths(npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM) 
    n_autosome_bins = len(autosome_bins)

    autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
    for k, v in autosome_data.items():
        autosome_data_mapped[dict(p_dict)[k]] = v
    
    if ad_model_allosomes is not None:
        allosome_bins, allosome_data = population.get_global_allosome_tractlengths('X', npts=npts, exclude_tracts_below_cM=exclude_tracts_below_cM)
        n_allosome_bins = len(allosome_bins)
        allosome_length = population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
        num_males = population.num_males
        num_females = population.num_females  
        
        female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in female_data.items():
            female_data_mapped[dict(p_dict)[k]] = v
        
        male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in male_data.items():
            male_data_mapped[dict(p_dict)[k]] = v

    local_parameter_handler = copy.deepcopy(parameter_handler)

    free_sex_bias_parameters = {param:0 for param, value in local_parameter_handler.demography.model_base_params.items() if 
                                (value.type == ParamType.SEX_BIAS) and 
                                (param not in local_parameter_handler.user_params_fixed_by_value) and 
                                (param not in local_parameter_handler.params_fixed_by_ancestry)}

    local_parameter_handler.add_fixed_parameters(free_sex_bias_parameters)
    
    best_objective = np.inf
    best_full_params = None

    def objective_function(model_base_parameters, include_allosomes = True):

        nonlocal best_objective, best_full_params

        """parameters are in optimizer space"""

        global _counter
        global _out_of_bounds_val
        global _min_out_of_bounds_val
        _counter += 1

        def flush_result(result, note = str()):
            param_str = 'array([%s])' % (', '.join(['%- 12g' % v for v in local_parameter_handler.convert_to_physical_params(model_base_parameters)]))
            if (verbose > 0) and (_counter % verbose == 0):
                logger.info(
                     "iter=%-6d | obj=%-12g | params=%s %s",
                        _counter,
                        result,
                        param_str,
                        note,
                    )
                #eprint('%-8i, %-12g, %s, %s' % (_counter, result, param_str, note))    

        if outofbounds_fun is not None:
            # outofbounds can return either True or a negative value to signify out-of-boundedness.
            oob = outofbounds_fun(model_base_parameters)
            if oob < 0:
                out = oob * _out_of_bounds_val-_min_out_of_bounds_val
                flush_result(out, f'OOB (oob={oob})')
                return out 
        else:
            eprint("No bound function defined")

        matrices = model_func(model_base_parameters)
        matrix_list = [matrix for matrix in matrices.values()]
        if include_allosomes:
            [male_matrix, female_matrix] = matrix_list
        else:
            male_matrix = matrix_list[0]  # Unbiased migration
            female_matrix = matrix_list[0]  # Unbiased migration


        # Model for autosomes
        if ad_model_autosomes == 'M':
            model = PhTMonoecious(0.5*(female_matrix+male_matrix), rho=1)
            result_autosomes = model.loglik(autosome_bins, population.Ls, [mat for mat in autosome_data_mapped], len(population.indivs))
        elif ad_model_autosomes == 'H-DC':
            result_autosomes=HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DC', X_chr = False, X_chr_male = False, N_cores = 5, bins=autosome_bins, Ls=population.Ls, data=[mat for mat in autosome_data_mapped], num_samples=len(population.indivs), cutoff=0)
        elif ad_model_autosomes == 'H-DF':
            result_autosomes=HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DF', X_chr = False, X_chr_male = False, N_cores = 5, bins=autosome_bins, Ls=population.Ls, data=[mat for mat in autosome_data_mapped], num_samples=len(population.indivs), cutoff=0)
        else:
            assert male_matrix.shape[0] < 20, "PhTDioecious currently only supports less than 20 generations for autosomes."
            result_autosomes = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_autosomes).loglik(autosome_bins, population.Ls, [mat for mat in autosome_data_mapped], len(population.indivs))
        
        if include_allosomes:
            # Model for allosomes
            if ad_model_allosomes == 'H-DC':
                result_X_females = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DC', X_chr = True, X_chr_male = False, N_cores = 5, bins=allosome_bins, Ls=[allosome_length], data=[mat for mat in female_data_mapped], num_samples=num_females, cutoff=0)
                result_X_males = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DC', X_chr = True, X_chr_male = True, N_cores = 5, bins=allosome_bins, Ls=[allosome_length], data=[mat for mat in male_data_mapped], num_samples=num_males, cutoff=0)
            elif ad_model_allosomes == 'H-DF':
                result_X_females = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DF', X_chr = True, X_chr_male = False, N_cores = 5, bins=allosome_bins, Ls=[allosome_length], data=[mat for mat in female_data_mapped], num_samples=num_females, cutoff=0)
                result_X_males = HP.HP_loglik(female_matrix, male_matrix, rr_f=1, rr_m=1, TP = 2, Dioecious_model = 'DF', X_chr = True, X_chr_male = True, N_cores = 5, bins=allosome_bins, Ls=[allosome_length],data=[mat for mat in male_data_mapped], num_samples=num_males, cutoff=0)   
            else:
                result_X_females = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True).loglik(allosome_bins, [allosome_length], [mat for mat in female_data_mapped], num_females)
                result_X_males = PhTDioecious(female_matrix, male_matrix, rho_f=1, rho_m=1, sex_model=ad_model_allosomes, X_chromosome=True, X_chromosome_male=True).loglik(allosome_bins, [allosome_length], [mat for mat in male_data_mapped], num_males)
        
            result = (result_autosomes + result_X_females + result_X_males)
            flush_result(result_X_females, 'Female allosomes')
            flush_result(result_X_males, 'Male allosomes')
        else:
            result = result_autosomes
        
        flush_result(result_autosomes, 'Autosomes')
        
        obj = -result

        if include_allosomes and np.isfinite(obj) and obj < best_objective:
            best_objective = obj
            best_full_params = model_base_parameters.copy()
        return obj
        
    def reduced_objective_function(free_parameters_opt, include_allosomes = True):

        extended_parameters = local_parameter_handler.extend_parameters(free_parameters_opt, units="opt")
        
        return objective_function(extended_parameters, include_allosomes=include_allosomes) #Full parameters in optimizer space
  
    def reduced_outofbounds_fun(free_parameters_opt):

        return outofbounds_fun(local_parameter_handler.extend_parameters(free_parameters_opt, units="opt")) #Full parameters in optimizer space

    reduced_p0 = local_parameter_handler.reduce_parameters(p0)

    if ad_model_allosomes is not None:
        title_message = f"Admixture is modelled with the {ad_model_autosomes} model for autosomes and with the {ad_model_allosomes} model for allosomes."
        subtitle_message = f"Optimization is performed in two steps.\nStep 1 : Optimizing autosomal likelihood over parameters {str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."
    else:
        title_message = f"Admixture is modelled with the {ad_model_autosomes} model for autosomes."
        subtitle_message = f"Optimizing autosomal likelihood over parameters {str(local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices))}."
    
    print('\n--------------------------------------------------------------------------------------------------')
    print(title_message)
    print(subtitle_message)
    
    if verbose>0:
        logger.info(title_message)
        logger.info(subtitle_message)
        logger.info('Iter.\t Log-likelihood\t Model parameters \t\t Transmission')

    reduced_objective_autosomes = lambda x: reduced_objective_function(x, include_allosomes = False)
    
    outputs = scipy.optimize.fmin_cobyla(
        reduced_objective_autosomes, reduced_p0, reduced_outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)
    
    optimized_parameters = local_parameter_handler.extend_parameters(outputs, units="opt")
    step1_full_params_opt = optimized_parameters.copy()

    new_fixed_parameters_names = local_parameter_handler.indices_to_labels(local_parameter_handler.free_parameters_indices)
    new_fixed_values = optimized_parameters[local_parameter_handler.free_parameters_indices]
    new_fixed_parameters = dict(zip(new_fixed_parameters_names, new_fixed_values))

    local_parameter_handler.release_fixed_parameters(free_sex_bias_parameters.keys())

    local_parameter_handler.add_fixed_parameters(new_fixed_parameters)
    reduced_params = local_parameter_handler.reduce_parameters(optimized_parameters)

    if ad_model_allosomes is not None:
        print('Step 1 completed.')
        print('--------------------------------------------------------------------------------------------------')
        step_2_message = f"Step 2 : Optimizing autosomal + allosomal likelihood over parameters : {str(list(free_sex_bias_parameters.keys()))}.\nNon-sex-bias parameters fixed at values from previous optimization step."    
    else:
        step_2_message = f"Optimization completed."   
    print(step_2_message)
        
    if verbose>0:
        logger.info(step_2_message)
    if verbose>0 and ad_model_allosomes is not None:    
        logger.info('Iter.\t Log-likelihood\t Model parameters \t\t Transmission')
   
    best_objective = np.inf
    best_full_params = None

    reduced_objective_autosomes = lambda x: reduced_objective_function(x, include_allosomes = True)
    if len(reduced_params)>0 and ad_model_allosomes is not None:
        outputs = scipy.optimize.fmin_cobyla(
            reduced_objective_autosomes, reduced_params, reduced_outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)
        print('Step 2 completed.')
    else: # No optimization needed
        outputs = reduced_params
    
    print('--------------------------------------------------------------------------------------------------')    
    
    if len(reduced_params) == 0 and ad_model_allosomes is not None:
        full_params_opt = optimized_parameters.copy()
        likelihood = -objective_function(full_params_opt, include_allosomes=True)
        return full_params_opt, likelihood
    if ad_model_allosomes is None:
        full_params_opt = optimized_parameters.copy()
        likelihood = -objective_function(full_params_opt, include_allosomes=False)
        return full_params_opt, likelihood
    
    if best_full_params is None:
        try:
            fallback_likelihood = -objective_function(step1_full_params_opt, include_allosomes=True)
            return step1_full_params_opt, fallback_likelihood
        except Exception:
            return step1_full_params_opt, -1e32

    return best_full_params, -best_objective      


# NOTE: All the functions below are not used. Consider removing them or moving them elsewhere.

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

    global _counter
    global _out_of_bounds_val
    global _min_out_of_bounds_val
    _counter += 1

    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative value to signify out-of-boundedness.
        oob = outofbounds_fun(params)
        if oob < 0:
            out = oob * _out_of_bounds_val-_min_out_of_bounds_val
            result = -out
        else:
            mod = modelling_method(model_func(params))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod = modelling_method(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if True:  # (verbose > 0) and (_counter % verbose == 0):
        param_str = 'of:array([%s])' % (', '.join(['%- 12g' % v for v in params]))
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


def _object_func_fracs(params, bins, Ls, data, nsamp, model_func, fracs, outofbounds_fun=None, cutoff=0, verbose=0,
                       flush_delay=0, func_args=None):
    """Define the objective function for when the ancestry porportions are specified."""
    if func_args is None:
        func_args = []

    global _counter
    global _min_out_of_bounds_val
    global _out_of_bounds_val
    _counter += 1

    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative valueto signify out-of-boundedness.
        oob = outofbounds_fun(params, fracs=fracs)
        if oob < 0:
            out = oob * _out_of_bounds_val-_min_out_of_bounds_val
            result = -out 
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

    global _counter
    _counter += 1
    global _out_of_bounds_val
    global _min_out_of_bounds_val
    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    if outofbounds_fun is not None:
        # outofbounds returns  a negative value to signify out-of-boundedness.
        oob = outofbounds_fun(params)

        if oob < 0:
            # we want bad functions to give very low likelihoods, and worse
            # likelihoods when the function is further out of bounds.
            out = oob * _out_of_bounds_val-_min_out_of_bounds_val
            mresult = - out
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
    global _counter
    _counter += 1
    global _out_of_bounds_val
    global _min_out_of_bounds_val
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
            
            out = oob * _out_of_bounds_val-_min_out_of_bounds_val
            mresult = -out 
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
