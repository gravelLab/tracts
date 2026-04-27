import logging
import numpy as np
import numpy.typing as npt
import scipy.optimize
import copy

from tracts.phase_type import hybrid_pedigree as HP
from tracts.phase_type import PhTMonoecious, PhTDioecious
from tracts.demography.parametrized_demography_sex_biased import SexType
from tracts.population import Population
from tracts.util import eprint
from tracts.demography.parameter import ParamType
logger = logging.getLogger(__name__)

# ----- Counts calls to object_func -----
_counter = 0
_out_of_bounds_val = -1e32
_min_out_of_bounds_val = -1e-10

# ------ Optimizers ------

def optimize_cob(p0:list, bins:npt.ArrayLike, Ls:npt.ArrayLike, data:list[np.ndarray], nsamp:int, 
                 model_func:callable, outofbounds_fun:callable=None, cutoff:int=0, verbose_screen:int=0, 
                 flush_delay:float=1, maxiter:int=None, func_args:list=None, reset_counter:bool=True) -> np.ndarray:
    """
    Optimizes model parameters using the COBYLA method. Valid only for autosomal data. Admixture is modelled with 
    the Monoecious model.

    Parameters
    ----------
    p0: list
        An array of initial parameters to start the optimization.
    bins:npt.ArrayLike
        A point grid on where the tract length distribution has to be evaluated.  
    Ls: npt.ArrayLike
        The lengths of the chromosomes present in data.
    data:list[np.ndarray]
        Spectrum with data.
    model_func:callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    outofbounds_fun: callable, Optional
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    cutoff: int, default:0 
        The number of bins to drop at the beginning of the array. This could be achieved with masks.
    verbose_screen: int, default: 0
        If greater than zero, prints optimization status every ``verbose`` iterations.
    flush_delay: float, default: 1
        Standard output will be flushed once every ``flush_delay`` minutes. This
        is useful to avoid overloading I/O on clusters.
    maxiter: int, default: None
        Maximum iterations to run for.
    func_args: list, default: None
        List of additional arguments to ``model_func``. It is assumed that ``model_func``'s
        first argument is an array of parameters to optimize.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).

    Returns
    -------
    np.ndarray
        An array containing the optimal parameters found by the optimizer.

    """
    print(PhTMonoecious)
    if func_args is None:
        func_args = []
    if reset_counter:
        global _counter
        _counter = 0

    fun = lambda x: _object_func(x, bins, Ls, data, nsamp, model_func, outofbounds_fun=outofbounds_fun, cutoff=cutoff,
                                 verbose=verbose_screen, flush_delay=flush_delay, func_args=func_args,
                                 modelling_method=PhTMonoecious)

    outputs = scipy.optimize.fmin_cobyla(
        fun, p0, outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)

    return outputs

def optimize_cob_sex_biased(p0:list, population: Population, model_func: callable, parameter_handler=None, outofbounds_fun:callable=None, 
                            verbose_log:int=0, verbose_screen:int=10, p_dict:dict=None, exclude_tracts_below_cM:float=0, 
                            maxiter:int=None, reset_counter:bool=True, ad_model_autosomes:str='DC',
                            ad_model_allosomes:str='DC', npts:int=50) -> tuple[np.ndarray, float]:
    """
    Optimizes the log-likelihood over all parameters defined by the demographic model, given a specified pair of admixture models for autosomes and allosomes.
    The optimization is carried out jointly in a single step, estimating all parameters simultaneously using both autosomal and allosomal data.

    Parameters
    ----------    
    p0: list
            An array of initial parameters to start the optimization.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    model_func: callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    parameter_handler: ParameterHandler, optional
        An object that handles parameter transformations and fixed parameters. Default is None.
    outofbounds_fun: callable, Optional
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    verbose_log: int, default: 0
        If greater than zero, logs optimization status every ``verbose`` iterations.
    verbose_screen: int, default: 0
        If greater than zero, prints optimization status every ``verbose`` iterations.
    p_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    maxiter: int, default: None
        Maximum iterations to run for.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).
    ad_model_autosomes: str, optional
        The model to use for autosomal admixture. Must be one of 'DC', 'DF', 'M', 'H-DC' or 'H-DF'. Default is 'DC'.
    ad_model_allosomes: str, optional
        The model to use for allosomal admixture. Must be one of 'DC', 'DF', 'H-DC' or 'H-DF'. Default is 'DC'. If None, allosomal admixture will not be modeled.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.

    Returns
    -------
    tuple [np.ndarray, float]
        A tuple containing the optimal parameters found and the corresponding likelihood.
    """
    
    if reset_counter:
        global _counter
        _counter = 0
    
    # Load and process autosome data
    autosome_bins, autosome_data = population.get_global_tractlengths(npts=npts,
                                                                    exclude_tracts_below_cM=exclude_tracts_below_cM) 
    n_autosome_bins = len(autosome_bins)
    autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _i in dict(p_dict).keys()]
    for k, v in autosome_data.items():
        autosome_data_mapped[dict(p_dict)[k]] = v
    
    if ad_model_allosomes is not None: # Include allosome data for inference
        
        # Load and process allosome data
        allosome_bins, allosome_data = population.get_global_allosome_tractlengths(allosome='X',
                                                                                npts=npts,
                                                                                exclude_tracts_below_cM=exclude_tracts_below_cM)
        n_allosome_bins = len(allosome_bins)
        allosome_length = population.allosome_lengths['X']
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]
        num_males = population.num_males
        num_females = population.num_females
        
        # Allosome female data
        female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in female_data.items():
            female_data_mapped[dict(p_dict)[k]] = v
        
        # Allosome male data
        male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist()  for _i in dict(p_dict).keys()]
        for k, v in male_data.items():
            male_data_mapped[dict(p_dict)[k]] = v
    
    def objective_function(parameters):

        global _counter
        _counter += 1

        def flush_result(result, note = str()): 
            param_str = 'ocsb: array([%s])' % (', '.join(['%- 12g' % v for v in parameter_handler.convert_to_physical_params(parameters)]))
            if (verbose_log > 0) and (_counter % verbose_log == 0): # Adds iterations to log file
                logger.info(
                     "iter=%-6d | obj=%-12g | params=%s %s",
                        _counter,
                        result,
                        param_str,
                        note,
                    )
            if (verbose_screen > 0) and (_counter % verbose_screen == 0): # Prints iterations on screen
                eprint('%-8i, %-12g, %s, %s' % (_counter, result, param_str, note))
                
        if outofbounds_fun is not None: # outofbounds can return either True or a negative value to signify out-of-boundedness.
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
            model = PhTMonoecious(migration_matrix=0.5*(female_matrix+male_matrix),
                                rho=1)
            result_autosomes = model.loglik(bins=autosome_bins,
                                        Ls=population.Ls,
                                        data=[mat for mat in autosome_data_mapped],
                                        num_samples=len(population.indivs),
                                        cutoff=0)
        elif ad_model_autosomes == 'H-DC':
            result_autosomes=HP.HP_loglik(mig_matrix_f=female_matrix,
                                        mig_matrix_m=male_matrix,
                                        rho_f=1,
                                        rho_m=1,
                                        TP = 2,
                                        Dioecious_model = 'DC',
                                        X_chr = False,
                                        X_chr_male = False,
                                        N_cores = 5,
                                        bins=autosome_bins,
                                        Ls=population.Ls,
                                        data=[mat for mat in autosome_data_mapped],
                                        num_samples=len(population.indivs),
                                        cutoff=0)
        elif ad_model_autosomes == 'H-DF':
            result_autosomes=HP.HP_loglik(mig_matrix_f=female_matrix,
                                        mig_matrix_m=male_matrix,
                                        rho_f=1,
                                        rho_m=1,
                                        TP = 2,
                                        Dioecious_model = 'DF',
                                        X_chr = False,
                                        X_chr_male = False,
                                        N_cores = 5,
                                        bins=autosome_bins,
                                        Ls=population.Ls,
                                        data=[mat for mat in autosome_data_mapped],
                                        num_samples=len(population.indivs),
                                        cutoff=0)
        else:
            result_autosomes = PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            sex_model=ad_model_autosomes).loglik(bins=autosome_bins,
                                                                                Ls=population.Ls,
                                                                                data=[mat for mat in autosome_data_mapped],
                                                                                num_samples=len(population.indivs))
        
        if ad_model_allosomes is not None: # Model for allosomes
            
            if ad_model_allosomes == 'H-DC':
                result_X_females = HP.HP_loglik(mig_matrix_f=female_matrix,
                                                mig_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                TP = 2,
                                                Dioecious_model = 'DC',
                                                X_chr = True,
                                                X_chr_male = False,
                                                N_cores = 5,
                                                bins=allosome_bins,
                                                Ls=[allosome_length],
                                                data=[mat for mat in female_data_mapped],
                                                num_samples=num_females, cutoff=0)
                result_X_males = HP.HP_loglik(mig_matrix_f=female_matrix,
                                            mig_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            TP = 2,
                                            Dioecious_model = 'DC',
                                            X_chr = True,
                                            X_chr_male = True,
                                            N_cores = 5,
                                            bins=allosome_bins,
                                            Ls=[allosome_length],
                                            data=[mat for mat in male_data_mapped],
                                            num_samples=num_males, cutoff=0)
                
            elif ad_model_allosomes == 'H-DF':
                result_X_females = HP.HP_loglik(mig_matrix_f=female_matrix,
                                                mig_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                TP = 2,
                                                Dioecious_model = 'DF',
                                                X_chr = True,
                                                X_chr_male = False,
                                                N_cores = 5,
                                                bins=allosome_bins,
                                                Ls=[allosome_length],
                                                data=[mat for mat in female_data_mapped],
                                                num_samples=num_females, cutoff=0)
                result_X_males = HP.HP_loglik(mig_matrix_f=female_matrix,
                                            mig_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            TP = 2,
                                            Dioecious_model = 'DF',
                                            X_chr = True,
                                            X_chr_male = True,
                                            N_cores = 5,
                                            bins=allosome_bins,
                                            Ls=[allosome_length],
                                            data=[mat for mat in male_data_mapped],
                                            num_samples=num_males, cutoff=0)   
            else:
                result_X_females = PhTDioecious(migration_matrix_f=female_matrix,
                                                migration_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                sex_model=ad_model_allosomes,
                                                X_chromosome=True).loglik(bins=allosome_bins,
                                                                        Ls=[allosome_length],
                                                                        data=[mat for mat in female_data_mapped],
                                                                        num_samples=num_females)
                result_X_males = PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            sex_model=ad_model_allosomes,
                                            X_chromosome=True,
                                            X_chromosome_male=True).loglik(bins=allosome_bins,
                                                                        Ls=[allosome_length],
                                                                        data=[mat for mat in male_data_mapped],
                                                                        num_samples=num_males)
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
    subtitle_message = "Optimizing model likelihood.\n---------------------------\nIter.\t Log-likelihood\t Model parameters\t Transmission\n---------------------------------------------------------------------\n"
    
    line = "-" * len(title_message)
    print('\n' + line)
    print(title_message)
    print(line)

    if (verbose_log > 0) and (_counter % verbose_log == 0):
        logger.info(subtitle_message)
    if (verbose_screen > 0) and (_counter % verbose_screen == 0):
        print(subtitle_message)
    
    outputs = scipy.optimize.fmin_cobyla(
        objective_function, p0, outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)
    
    likelihood = -objective_function(outputs)

    return outputs, likelihood


def optimize_cob_sex_biased_fixed_values(p0:list, population: Population, model_func:callable, parameter_handler,
                                    outofbounds_fun:callable=None, verbose_log:int=0, verbose_screen:int=10,
                                    p_dict:dict=None, exclude_tracts_below_cM:float=0, maxiter:int=None, reset_counter:bool=True, 
                                    ad_model_autosomes:str='DC', ad_model_allosomes:str='DC', npts:int=50) -> tuple[np.ndarray, float]:
    """
    Optimizes the log-likelihood over all parameters defined by the demographic model, for a specified admixture model applied to both autosomes and allosomes.
    The procedure is carried out in two steps: first, the non–sex-bias parameters are estimated by maximizing the log-likelihood using autosomal data only.
    Second, the sex-bias parameters are estimated using both autosomal and allosomal data.

    Parameters
    ----------    
    p0: list
            An array of initial parameters to start the optimization.
    population: :class:`tracts.population.Population`
        A Population object containing the data to fit.
    model_func: callable
        A function that takes a parameter array and returns a dictionary of migration matrices for each population.
    parameter_handler: ParameterHandler, optional
        An object that handles parameter transformations and fixed parameters. Default is None.
    outofbounds_fun: callable, Optional
        A function that takes a parameter array and returns a violation score indicating how much the parameters violate the bounds.
    cutoff: int, default:0 
        The number of bins to drop at the beginning of the array. This could be achieved with masks.
    verbose_log: int, default: 0
        If greater than zero, logs optimization status every ``verbose`` iterations.
    verbose_screen: int, default: 0
        If greater than zero, prints optimization status every ``verbose`` iterations.
    p_dict: dict
        A dictionary mapping population labels to their corresponding indices in the model.
    exclude_tracts_below_cM: float, optional
        Minimum tract length in centimorgans to exclude from analysis. Default is 0.
    maxiter: int, default: None
        Maximum iterations to run for.
    reset_counter: bool, default: True
        Resets the iteration counter to zero. Set to False to
        continue iteration count (e.g., if optimization continues from previous point).
    ad_model_autosomes: str, optional
        The model to use for autosomal admixture. Must be one of 'DC', 'DF', 'M', 'H-DC' or 'H-DF'. Default is 'DC'.
    ad_model_allosomes: str, optional
        The model to use for allosomal admixture. Must be one of 'DC', 'DF', 'H-DC' or 'H-DF'. Default is 'DC'. If None, allosomal admixture will not be modeled.
    npts: int, optional
        Number of bins for the tract length histogram. Default is 50.

    Returns
    -------
    tuple [np.ndarray, float]
        A tuple containing the optimal parameters found and the corresponding likelihood.
    """

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
            if (verbose_log > 0) and (_counter % verbose_log == 0): # Add iteration to log file
                logger.info(
                     "iter=%-6d | obj=%-12g | params=%s %s",
                        _counter,
                        result,
                        param_str,
                        note,
                    )
            if (verbose_screen > 0) and (_counter % verbose_screen == 0): # Print iteration on screen
                eprint('%-8i, %-12g, %s, %s' % (_counter, result, param_str, note))    

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
            model = PhTMonoecious(migration_matrix=0.5*(female_matrix+male_matrix),
                                rho=1)
            result_autosomes = model.loglik(bins=autosome_bins,
                                            Ls=population.Ls,
                                            data=[mat for mat in autosome_data_mapped],
                                            num_samples=len(population.indivs))
        elif ad_model_autosomes == 'H-DC':
            result_autosomes=HP.HP_loglik(mig_matrix_f=female_matrix,
                                        mig_matrix_m=male_matrix,
                                        rho_f=1,
                                        rho_m=1,
                                        TP = 2,
                                        Dioecious_model = 'DC',
                                        X_chr = False,
                                        X_chr_male = False,
                                        N_cores = 5,
                                        bins=autosome_bins,
                                        Ls=population.Ls,
                                        data=[mat for mat in autosome_data_mapped],
                                        num_samples=len(population.indivs), cutoff=0)
        elif ad_model_autosomes == 'H-DF':
            result_autosomes=HP.HP_loglik(mig_matrix_f=female_matrix,
                                        mig_matrix_m=male_matrix,
                                        rho_f=1,
                                        rho_m=1,
                                        TP = 2,
                                        Dioecious_model = 'DF',
                                        X_chr = False,
                                        X_chr_male = False,
                                        N_cores = 5,
                                        bins=autosome_bins,
                                        Ls=population.Ls,
                                        data=[mat for mat in autosome_data_mapped],
                                        num_samples=len(population.indivs), cutoff=0)
        else:
            assert male_matrix.shape[0] < 20, "PhTDioecious currently only supports less than 20 generations for autosomes."
            result_autosomes = PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            sex_model=ad_model_autosomes).loglik(bins=autosome_bins,
                                                                                Ls=population.Ls,
                                                                                data=[mat for mat in autosome_data_mapped],
                                                                                num_samples=len(population.indivs))
        
        if include_allosomes: # Model for allosomes
            
            if ad_model_allosomes == 'H-DC':
                result_X_females = HP.HP_loglik(mig_matrix_f=female_matrix,
                                                mig_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                TP = 2,
                                                Dioecious_model = 'DC',
                                                X_chr = True,
                                                X_chr_male = False,
                                                N_cores = 5,
                                                bins=allosome_bins,
                                                Ls=[allosome_length],
                                                data=[mat for mat in female_data_mapped],
                                                num_samples=num_females, cutoff=0)
                
                result_X_males = HP.HP_loglik(mig_matrix_f=female_matrix,
                                              mig_matrix_m=male_matrix,
                                              rho_f=1,
                                              rho_m=1,
                                              TP = 2,
                                              Dioecious_model = 'DC',
                                              X_chr = True,
                                              X_chr_male = True,
                                              N_cores = 5,
                                              bins=allosome_bins,
                                              Ls=[allosome_length],
                                              data=[mat for mat in male_data_mapped],
                                              num_samples=num_males, cutoff=0)
                
            elif ad_model_allosomes == 'H-DF':
                result_X_females = HP.HP_loglik(mig_matrix_f=female_matrix,
                                                mig_matrix_m=male_matrix,
                                                rho_f=1,
                                                rho_m=1,
                                                TP = 2,
                                                Dioecious_model = 'DF',
                                                X_chr = True,
                                                X_chr_male = False,
                                                N_cores = 5,
                                                bins=allosome_bins,
                                                Ls=[allosome_length],
                                                data=[mat for mat in female_data_mapped],
                                                num_samples=num_females,
                                                cutoff=0)
                
                result_X_males = HP.HP_loglik(mig_matrix_f=female_matrix,
                                            mig_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            TP = 2,
                                            Dioecious_model = 'DF',
                                            X_chr = True,
                                            X_chr_male = True,
                                            N_cores = 5,
                                            bins=allosome_bins,
                                            Ls=[allosome_length],
                                            data=[mat for mat in male_data_mapped],
                                            num_samples=num_males, cutoff=0)   
            else:
                result_X_females = PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            sex_model=ad_model_allosomes,
                                            X_chromosome=True).loglik(bins=allosome_bins,
                                                                    Ls=[allosome_length],
                                                                    data=[mat for mat in female_data_mapped],
                                                                    num_samples=num_females)
                result_X_males = PhTDioecious(migration_matrix_f=female_matrix,
                                            migration_matrix_m=male_matrix,
                                            rho_f=1,
                                            rho_m=1,
                                            sex_model=ad_model_allosomes,
                                            X_chromosome=True,
                                            X_chromosome_male=True).loglik(bins=allosome_bins,
                                                                        Ls=[allosome_length],
                                                                        data=[mat for mat in male_data_mapped],
                                                                        num_samples=num_males)

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
    
    line = "-" * len(title_message)
    
    for l in [line, title_message, subtitle_message]:
        logger.info(l)
        print(l)
    
    table_header = "Iter.\t Log-likelihood\t Model parameters\t Transmission"
    for l in [table_header, line]:
        if verbose_log>0:
            logger.info(l)
        if verbose_screen>0:
            print(l)
            

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
        step_2_message_1 = f"Step 2 : Optimizing autosomal + allosomal likelihood over parameters : {str(list(free_sex_bias_parameters.keys()))}."
        step_2_message = f"{step_2_message_1}\nNon-sex-bias parameters fixed at values from previous optimization step."    
        line = "-" * len(step_2_message_1)
    else:
        step_2_message = f"Optimization completed."   
        line = "-" * len(step_2_message)
    
    if len(reduced_params)>0 and verbose_log>0:
        logger.info(line)
        logger.info(step_2_message)
        if ad_model_allosomes is not None:    
            logger.info('Iter.\t Log-likelihood\t Model parameters\t Transmission')
            logger.info(line)
    if len(reduced_params)>0 and verbose_screen>0:
        print(line)
        print(step_2_message)
        if ad_model_allosomes is not None:
            print('Iter.\t Log-likelihood\t Model parameters\t Transmission')
            print(line)

    best_objective = np.inf
    best_full_params = None

    reduced_objective_autosomes = lambda x: reduced_objective_function(x, include_allosomes = True)
    if len(reduced_params)>0 and ad_model_allosomes is not None:
        outputs = scipy.optimize.fmin_cobyla(
            reduced_objective_autosomes, reduced_params, reduced_outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)
        print('Step 2 completed.')
        print(line)

    else: # No optimization needed
        end_message = f"No parameters to optimize in step 2. Optimization completed."
        for l in [end_message, "-" * len(end_message)]:
            print(l)
            logger.info(l)
        outputs = reduced_params
    
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
