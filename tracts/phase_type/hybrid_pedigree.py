import itertools
from functools import partial
import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import gammaln
from .dioecious import PhTDioecious
import logging
logger = logging.getLogger(__name__)

# ------------ Helper functions ------------

def get_pedigree(T: int):
    r"""
    Generates a matrix representing a pedigree structure of a given number of generation. 
    Parameters
    ----------
    T : int
        The number of generations in the pedigree. Should be at least 2.  

    Returns
    -------
    np.ndarray
        A matrix representing the pedigree structure. The matrix has dimensions
        `(n_ind, n_anc)`, where `n_ind` is the number of individuals in the pedigree starting at generation 1
        and `n_anc` is the number of ancestors at generation `T`. The entry :math:`(i, j)` of the matrix is 1 if
        individual :math:`i` is a descendant of ancestor :math:`j`, and 0 otherwise.
    
    Notes
    -----
        The pedigree is structured as a complete binary tree, where each individual has two parents in the previous generation. Since 
        phase-type models are considered for a fixed parent (i.e. fixing the hyper-parameter :math:`\xi` in the manuscript), the pedigree
        starts at generation 1.   
    """
    n_ind = 2 ** T - 1
    n_anc = 2 ** (T - 1)
    the_pedigree = np.zeros([n_ind, n_anc])
    the_pedigree[0, :] = 1
    k = 1
    for i in range(2, T + 1):
        len_i = 2 ** (T - i)
        n_groups_i = int(n_anc / len_i)
        for j in range(n_groups_i):
            vec_ij = np.zeros(n_anc)
            vec_ij[(len_i * j):(len_i * j + len_i)] = 1
            the_pedigree[k, :] = vec_ij
            k += 1
    return the_pedigree


def generate_trees(current_gen:int, max_gen:int, N:int, migrants_at_last_gen:bool=True):
    r"""
    Recursively generates all possible binary trees representing the migration settings in a pedigree of `max_gen` generations.
    For each tree, nodes can take values in :math:`\lbrace 0, 1, ..., N \rbrace`, where 0 means that the node is an admixed individual whose parents are further
    expanded in the tree, and a value :math:`k\in \lbrace 1, ..., N\rbrace` means that the node is an ancestor from population :math:`k` and the branch stops.
    
    Parameters
    ----------
    current_gen : int
        The current generation being processed. Should start at 1 when the function is first called.
    max_gen : int
        The maximum number of generations in the pedigree.
    N : int
        The number of populations.
    migrants_at_last_gen : bool, optional
        If true, all individuals at the last generation are forced to be migrants. Default is True.

    Returns
    -------
    list
        A list of all possible binary trees representing the migration settings. Each tree is represented as a nested list, where each node is a list of the form [`value`, `left_subtree`, `right_subtree`]. 
        The value is an integer in :math:`\lbrace 0, 1, ...,N\rbrace` as described above, and `left_subtree` and `right_subtree` are the left and right subtrees of the node, respectively.
        If a node is a leaf (i.e. it has no children), `left_subtree` and `right_subtree` are set to None.
    """
    if current_gen > max_gen:
        return [[]]
    trees = []
    if current_gen == max_gen:
        # In the last generation, nodes must have values in [1, ..., N]
        start = 1 if migrants_at_last_gen else 0
        for value in range(start, N + 1):
            trees.append([value, None, None])
    else:
        # Node can take value 0 (expand further)
        for left_subtree in generate_trees(current_gen=current_gen + 1,
                                        max_gen=max_gen,
                                        N=N,
                                        migrants_at_last_gen=migrants_at_last_gen):
            for right_subtree in generate_trees(current_gen=current_gen + 1,
                                            max_gen=max_gen,
                                            N=N,
                                            migrants_at_last_gen=migrants_at_last_gen):
                trees.append([0, left_subtree, right_subtree])

        # Node can take values 1, ..., N (branch stops)
        for value in range(1, N + 1):
            trees.append([value, None, None])
    return trees


def tree_to_array(tree:list, max_nodes: int):
    """
    Convert a tree to its array representation. The array is constructed by traversing the tree in level
    order (BFS) and filling the array accordingly. The value of each node is stored in the array at the index
    corresponding to its position in the level order traversal.

    Parameters
    ----------
    tree : list
        A binary tree represented as a nested list, as produced by :func:~`tracts.phase_type.hybrid_pedigree.generate_trees`.
    max_nodes : int
        The maximum number of nodes in the array representation.

    Returns
    -------
    list
        The array representation of the tree.
    """
    array = [np.nan] * max_nodes  # Start with all NaN
    queue = [(tree, 0)]  # (node, index)

    while queue:
        node, index = queue.pop(0)
        if index < max_nodes:
            if isinstance(node, list):
                array[index] = node[0]
                if node[0] == 0:  # Only expand if the node is 0
                    queue.append((node[1], 2 * index + 1))  # Left child
                    queue.append((node[2], 2 * index + 2))  # Right child
            else:
                array[index] = np.nan  # Node doesn't exist

    return array


def all_possible_trees_as_arrays(T:int, N:int, mig_at_last:bool=True):
    """
    Generates all possible trees representing the migration settings in a pedigree of `T` generations with 
    `N` ancestral populations. The function converts them to their array representation and returns them as a list of arrays. 
    See :func:`~tracts.phase_type.hybrid_pedigree.generate_trees` for details on the tree structure and :func:`~tracts.phase_type.hybrid_pedigree.tree_to_array` for details on the array representation.
    
    Parameters
    ----------
    T : int
        The number of generations in the pedigree.
    N : int
        The number of ancestral populations.
    mig_at_last : bool, optional
        Whether to force individuals at the last generation to be migrants. Default is True.

    Returns
    -------
    list
        A list of arrays representing all possible trees. 
    """

    max_nodes = 2 ** T - 1  # Maximum number of nodes in a complete binary tree of T generations
    trees = generate_trees(current_gen=1,
                        max_gen=T,
                        N=N,
                        migrants_at_last_gen=mig_at_last)
    arrays = [tree_to_array(tree=tree, max_nodes=max_nodes) for tree in trees]
    return arrays


def prob_of_pop_setting(ms:int, all_possible_migrations_list:list, migrations_at_T_list:np.ndarray, 
                        f_migrations:np.ndarray, m_migrations:np.ndarray, parent_sex:int,
                        number_ind:int, number_anc:int, pedigree:np.ndarray):
    """
    Computes the probabily of a given configuration of migrant ancestors in the pedigree, as specified by the `ms`-th element of `all_possible_migrations_list`.
    The probability corresponds to Equation (E.1) in the manuscript. For details on this model and the derivation of this probability, see Appendix E in the manuscript.

    Parameters
    ----------
    ms : int
        The index of the migration setting to be considered, corresponding to the `ms`-th element of `all_possible_migrations_list`.
    all_possible_migrations_list : list
        A list of arrays representing all possible migration settings in the pedigree, as produced by :func:`~tracts.phase_type.hybrid_pedigree.all_possible_trees_as_arrays`.
    migrations_at_T_list : np.ndarray
        A 2D array representing all possible migration settings at the last generation.
    f_migrations : np.ndarray
        The female-specific migration matrix.
    m_migrations : np.ndarray
        The male-specific migration matrix.
    parent_sex : int
        The sex of the parent individual, that is, of the individual at the first generation of the pedigree.
    number_ind : int
        The number of individuals in the pedigree.
    number_anc : int
        The number of individuals at the last generation of the pedigree.
    pedigree : np.ndarray
        A 2D array representing the pedigree structure, as given by :func:`~tracts.phase_type.hybrid_pedigree.get_pedigree`.

    Returns
    -------
    int
        The index of the pedigree configuration that is being considered.
    float
        The probability of the migration configuration on the pedigree.
    """

    T = int(np.log(number_ind + 1) / np.log(2))
    sex_and_gen = np.zeros([number_ind, 2])
    sex_and_gen[0, :] = [0, 1]
    k = 1

    for i in range(2, T + 1):
        len_i = 2 ** (T - i)
        n_groups_i = int(number_anc / len_i)
        for j in range(n_groups_i):
            vec_ij = np.zeros(number_anc)
            vec_ij[(len_i * j):(len_i * j + len_i)] = 1
            sex_and_gen[k, 0] = int(1 - sex_and_gen[k - 1, 0])
            sex_and_gen[k, 1] = int(i)
            k += 1
    sex_and_gen[0, :] = [parent_sex, 1]
    
    migrant_individuals = np.asarray(all_possible_migrations_list[ms])
    mig_setting = np.concatenate([sex_and_gen[~np.isnan(migrant_individuals), :],
                                  migrant_individuals[~np.isnan(migrant_individuals), np.newaxis],
                                  np.zeros([np.sum(~np.isnan(migrant_individuals)), 1])], axis=1)    
    
    pfmig = mig_setting[:, 0][:, np.newaxis] * (mig_setting[:,2] > 0)[:, np.newaxis] * f_migrations[mig_setting[:, 1].astype(int), :]
    pmmig = (1 - mig_setting[:, 0][:, np.newaxis]) * (mig_setting[:,2] > 0)[:, np.newaxis] * m_migrations[mig_setting[:, 1].astype(int), :]
    pfad = mig_setting[:, 0] * (mig_setting[:, 2] == 0) * (1 - np.sum(f_migrations[mig_setting[:, 1].astype(int), :], axis=1))
    pmad = (1 - mig_setting[:, 0]) * (mig_setting[:, 2] == 0) * (1 - np.sum(m_migrations[mig_setting[:, 1].astype(int), :], axis=1))

    which_pfmig = (np.array([mig_setting[:,2] > 0])*np.array([mig_setting[:,0] == 1]))[0]                  
    which_pmmig = (np.array([mig_setting[:,2] > 0])*np.array([mig_setting[:,0] == 0]))[0]                  
                 
    pfmig = pfmig[np.where(which_pfmig)[0], mig_setting[which_pfmig, 2].astype(int) - 1]
    pmmig = pmmig[np.where(which_pmmig)[0], mig_setting[which_pmmig ,2].astype(int) - 1]
    pad = (pfad + pmad)
    pad = pad[np.nonzero(pad)]
    p_mig_setting = np.prod(pfmig)*np.prod(pmmig)*np.prod(pad)

    ancestral_states = np.asarray(np.nansum(pedigree * migrant_individuals[:, np.newaxis], axis=0))
    which_ancestral = np.where(np.sum(np.abs(migrations_at_T_list - ancestral_states), axis=1) == 0)[0][0]
    return int(which_ancestral), p_mig_setting


def density_hybrid_pedigree(which_migration: int, migration_list:list, T_PED:int, which_pop:int, D_model:str,
                            which_L: int, bins: np.ndarray, mig_matrix_f: np.ndarray, mig_matrix_m: np.ndarray,
                            rho_f: float, rho_m: float, X_chr: bool, X_chr_male: bool, density: bool):
    r"""
    Computes the tract length density or histogram for a given pedigree configuration, as specified by the `which_migration`-th element of `migration_list`,
    using the Hybrid-Pedigree refinement of the DC or DF model. The Phase-type parameters are computed from a pair of migration matrices, the sex-specific recombination rates
    and the number of pedigree generations, using a :class:~`tracts.phase_type.dioecious.PhTDioecious` object. See Appendix E of the manuscript for details on the model.

    Parameters
    ----------
    which_migration : int
        The index of the pedigree configuration to be considered, corresponding to the `which_migration`-th element of `migration_list`.
    migration_list : list
        A list of arrays representing all possible migration settings at the last generation.
    T_PED : int
        The number of generations in the pedigree.
    which_pop : int
        The population of interest whose tract length distribution has to be computed. An integer from 0 to the number of populations - 1.
    D_model : str
        The Dioecious model to be considered. Takes the value 'DF' for Dioecious-Fine and 'DC' for Dioecious-Coarse.
    which_L : int
        The length of the finite chromosome.
    bins : np.ndarray
        A point grid on :math:`(0, \infty)` where the CDF or density have to be evaluated. If `density` is False, the histogram will be computed on the intervals defined by `bins`.
    mig_matrix_f : np.ndarray
        The female-specific migration matrix.
    mig_matrix_m : np.ndarray
        The male-specific migration matrix.
    rho_f : float
        The female-specific recombination rate.
    rho_m : float
        The male-specific recombination rate. For X chromosome admixture, this value is ignored and set to 0.
    X_chr : bool
        Whether admixture is considered on the X chromosome. If False, the model considers autosomal admixture.
    X_chr_male : bool
        If `X_chr` is True, whether the individual at generation 0 is a male. In that case, only maternally inherited alleles are taken into account. If not `X_chr`, set to False.
    density : bool
        Whether to compute the density (True) or the histogram (False). If True, the function returns the density evaluated on `bins`. If False, the function returns the histogram values on the intervals defined by `bins`.
    
    Returns
    -------
    np.ndarray
        If `density` is True, the Phase-type density evaluated on `bins` for the maternally-inherited tracts (:math:`\xi=1`). If `density` is False, the corresponding histogram values on the intervals defined by `bins`.
    np.ndarray
        If `density` is True, the Phase-type density evaluated on `bins` for the paternally-inherited tracts (:math:`\xi=0`). If `density` is False, the corresponding histogram values on the intervals defined by `bins`.
    int
        The index of the pedigree configuration that is being considered, corresponding to the `which_migration`-th element of `migration_list`.
    float
        The expected tract length for the maternally-inherited tracts (:math:`\xi=1`).
    float
        The expected tract length for the paternally-inherited tracts (:math:`\xi=0`).  
    """
    
    if density:
        bins = np.asarray(bins)
        if ~np.any(np.isin(bins, which_L)): # Add L to the bins vector
            bins = np.append(bins, which_L)
        bins = bins[bins <= which_L]  # Truncate to [0, L], where the distribution is supported
        bins = np.unique(np.sort(bins))
    
    ancestral_setting = np.asarray(migration_list[which_migration])

    ETL_m = None
    if np.all(ancestral_setting == which_pop + 1):
        newbins = bins
        counts_f = np.zeros(len(bins))
        counts_f[np.asarray(bins) >= which_L] = 1
        ETL_f = which_L
        counts_m = np.zeros(len(bins))
        counts_m[np.asarray(bins) >= which_L] = 1
        ETL_m = which_L

    elif np.all(ancestral_setting != which_pop + 1) and np.all(ancestral_setting > 0):
        
        counts_f = np.nan * np.asarray(bins)
        ETL_f = np.nan
        counts_m = np.nan * np.asarray(bins)
        ETL_m = np.nan        
    
    else:
        PhT_ped = PhTDioecious(migration_matrix_f=mig_matrix_f,
                               migration_matrix_m=mig_matrix_m,
                               rho_f=rho_f,
                               rho_m=rho_m,
                               sex_model=D_model,
                               X_chromosome=X_chr,
                               X_chromosome_male=X_chr_male,
                               TPED=T_PED,
                               setting_TP=ancestral_setting)
           
        if np.all(PhT_ped.source_populations_f == which_pop):
            counts_f = np.zeros(len(bins))
            counts_f[np.asarray(bins) >= which_L] = 1
            ETL_f = which_L
        elif np.all(PhT_ped.source_populations_f != which_pop) or np.all(np.isnan(PhT_ped.source_populations_f)):
            counts_f = np.nan * np.asarray(bins)
            ETL_f = np.nan
        else:
            newbins, counts_f, ETL_f = PhT_ped.tractlength_histogram_windowed(population_number=which_pop,
                                                                            bins=bins,
                                                                            L=which_L,
                                                                            density=density,
                                                                            return_only=1,
                                                                            freq=False,
                                                                            hybrid_ped=True)
        
        if np.all(PhT_ped.source_populations_m == which_pop):
            counts_m = np.zeros(len(bins))
            counts_m[np.asarray(bins) >= which_L] = 1
            ETL_m = which_L
        elif np.all(PhT_ped.source_populations_m != which_pop) or np.all(np.isnan(PhT_ped.source_populations_m)):
            counts_m = np.nan * np.asarray(bins)
            ETL_m = np.nan
        else:
            if not X_chr_male:
                newbins, counts_m, ETL_m = PhT_ped.tractlength_histogram_windowed(population_number=which_pop,
                                                                                bins=bins,
                                                                                L=which_L,
                                                                                density=density,
                                                                                return_only=0,
                                                                                freq=False,
                                                                                hybrid_ped=True)
            else:
                counts_m = np.nan*np.ones(len(bins))
                ETL_m = np.nan
    
    return counts_f, counts_m, which_migration, ETL_f, ETL_m


def hybrid_pedigree_distribution(mig_matrix_f: np.ndarray, mig_matrix_m: np.ndarray, L: float, 
                                bingrid: np.ndarray, whichpop: int, TP: int = 2, Dioecious_model: str = 'DC',
                                rho_f: float = 1, rho_m: float = 1, X_chr: bool = False, X_chr_male: bool = False,
                                N_cores: int = 1, density: bool = True, freq: bool = False, print_progress: bool = True):
    r"""
    This function computes the tract length distribution as a Phase-type mixture on a finite 
    chromosome of length L, using the Hybrid-Pedigree refinement of the DC or DF model. The Phase-type parameters are computed from a 
    pair of migration matrices, the sex-specific recombination rates and the number of pedigree generations. See Appendix E of the
    manuscript for details on the model.
    
    Parameters
    ----------
    mig_matrix_f : npt.ArrayLike
        An array containing the female migration proportions from a discrete number of populations over the last generations.
        Each row is a time, each column is a population. Row zero corresponds to the current
        generation. The :math:`(i,j)` element of this matrix specifies the proportion of female individuals from the admixed population that
        are replaced by female individuals from population :math:`j` at generation :math:`i`. 
        The migration rate at the last generation (`migration_matrix_f[-1,:]`) must sum up to 1. 
    mig_matrix_m : npt.ArrayLike
        Counterpart of `mig_matrix_f` for male migration rates.
    L: float
        The length of the finite chromosome.
    bingrid: npt.ArrayLike
        A point grid on :math:`(0, \infty)` where the CDF or density have to be evaluated.  
    whichpop: int
        The population of interest whose tract length distribution has to be computed. 
        An integer from 0 to the number of populations - 1.         
    TP: int, default 2
        The number of generations of the pedigree. Shouldn't be higher than 3 for an acceptable computational efficiency in the current implementation.
    Dioecious_model: default 'DC'
        The Dioecious model to be considered. Takes the value 'DF' for Dioecious-Fine and 'DC' for Dioecious-Coarse.
    rho_f : float, default 1
        The female-specific recombination rate (positive real number).
    rho_m : float, default 1
        The male-specific recombination rate (positive real number). For X chromosome admixture, this value is ignored and set to 0.
    X_chr: bool, default False
        Whether admixture is considered on the X chromosome. If False, the model considers autosomal admixture.
    X_chr_male: bool, default False
        If `X_chr` is True, whether the individual at generation 0 is a male. In that case, only maternally inherited alleles are taken
        into account. If not `X_chr`, set to False.
    N_cores: int, default 1
        The number of threads to be used for parallel computation.
    density : bool, default False
        If True, returns the tract length density. Otherwise, returns the histogram values on the grid.
    freq : bool, default False,
        If density is True, whether to return density on the frequency scale. If density is False, this 
        parameter is ignored.
    print_progress: bool, default True
        Whether to display progress on screen.
        
    Returns
    ----------
    npt.ArrayLike
        If density is True, the corrected bins grid as described in Notes. Else, the user-specified bins.
    npt.ArrayLike
        If density is True, the Phase-type density evaluated on the corrected bins grid. Returned on the frequency scale if `freq` is True. 
        If density is False, the histogram values on the intervals defined by bins.
     
    Notes
    -------    
    If `density` is True, the code truncates bins to the interval :math:`[0,L]` and adds the point :math:`L` if it is not included in bins.
    This is done because the density is defined on the finite chromosome :math:`[0,L]` as a mixture of a continuous density on :math:`[0,L)` and a Dirac measure at :math:`L`.
    Consequently, the function returns as a first argument the transformed grid, that can be used as x-axis to plot the density. 
    If `density` is False, the code produces a histogram supported on the user-specified grid. 
    If the grid has n points, the histogram will be defined on :math:`n-1` intervals. Therefore, the returned 
    array will have length `len(bins)-1`.     
    """
    
    if not X_chr:
        X_chr_male = False
    
    if not density:
        freq = True
    
    if not np.isin(Dioecious_model, ['DF', 'DC']):
        raise Exception('The Dioecious model must be one in DF, DC.')

    T = int(np.shape(mig_matrix_f)[0] - 1) # Number of generations

    mean_mig_matrix = 0.5 * (mig_matrix_f + mig_matrix_m)
    survival_factors = np.ones(len(mean_mig_matrix))
    for generation_number in range(1, len(mean_mig_matrix)):
        survival_factors[generation_number] = survival_factors[generation_number - 1] * (
                1 - sum(mean_mig_matrix[generation_number - 1]))
    
    # Compute ancestry proportions for histograms (see Appendix F.3 in the manuscript)
    if not X_chr:
        t0_proportions_f = t0_proportions_m = np.sum(
            (0.5 * (mig_matrix_f + mig_matrix_m)) * np.transpose(survival_factors)[:, np.newaxis],
            axis=0) # Maternally (t0_proportions_f) and paternally (t0_proportions_m) -inherited ancestry proportions are the same.
    else: # Recursive computation for the X chromosome

        ancestry_proportions_m = mig_matrix_m[T, ]
        ancestry_proportions_f = mig_matrix_f[T, ]
        for generation_number in range(T-1, 0, -1):
                
            ancestry_proportions_f_prev = ancestry_proportions_f.copy()
            ancestry_proportions_m_prev = ancestry_proportions_m.copy()
            ancestry_proportions_m = mig_matrix_m[generation_number, ] + (1 - mig_matrix_m[generation_number, ].sum())*ancestry_proportions_f_prev
            ancestry_proportions_f = mig_matrix_f[generation_number, ] + (1 - mig_matrix_f[generation_number, ].sum())*(ancestry_proportions_m_prev + ancestry_proportions_f_prev)/2           
            
        t0_proportions_f = ancestry_proportions_f
        t0_proportions_m = ancestry_proportions_m
    
    Npops = int(np.shape(mig_matrix_f)[1])
    nind_TP = 2 ** TP - 1
    nanc_TP = 2 ** (TP - 1)
    migrations_at_TP = None
    if TP > T:
        raise Exception('The pedigree must include up to T generations.')

    the_pedigree = get_pedigree(TP)

    if TP < T:
        all_possible_migrations_TP = all_possible_trees_as_arrays(TP, Npops, mig_at_last=False)
        migrations_at_TP = np.array([i for i in itertools.product(np.arange(Npops + 1).tolist(), repeat=nanc_TP)])

    elif TP == T:
        all_possible_migrations_TP = all_possible_trees_as_arrays(TP, Npops, mig_at_last=True)
        migrations_at_TP = np.array([i for i in itertools.product(np.arange(1, Npops + 1).tolist(), repeat=nanc_TP)])
    
    if not X_chr_male:
        
        prob_of_pop_setting_iteration_0 = partial(prob_of_pop_setting,
                                                all_possible_migrations_list=all_possible_migrations_TP,
                                                migrations_at_T_list=migrations_at_TP,
                                                f_migrations=mig_matrix_f,
                                                m_migrations=mig_matrix_m,
                                                parent_sex=0,
                                                number_ind=nind_TP,
                                                number_anc=nanc_TP,
                                                pedigree=the_pedigree)

    prob_of_pop_setting_iteration_1 = partial(prob_of_pop_setting,
                                                all_possible_migrations_list=all_possible_migrations_TP,
                                                migrations_at_T_list=migrations_at_TP,
                                                f_migrations=mig_matrix_f,
                                                m_migrations=mig_matrix_m,
                                                parent_sex=1,
                                                number_ind=nind_TP,
                                                number_anc=nanc_TP,
                                                pedigree=the_pedigree)

    if print_progress:
        message = f"Computing pedigrees probabilities for a Hybrid-Pedigree refinement of the {Dioecious_model} model with T={TP} generations."
        line = "-" * len(message)
        for l in [line, message, line + "\n"]:
            print(l)
        
    if not X_chr_male:
    
        prob_list_m_complete = Parallel(n_jobs=N_cores, verbose=10*print_progress, prefer='processes')(
            delayed(prob_of_pop_setting_iteration_0)(i) for i in range(len(all_possible_migrations_TP)))
        data_prob_m = pd.DataFrame(prob_list_m_complete, columns=['which_ancestral', 'prob'])
    
        if not np.isclose(np.sum(data_prob_m.groupby(['which_ancestral']).sum().reset_index()['prob'].to_numpy()), 1):
            raise Exception('Pedigree probabilities do not sum up to one.')

    prob_list_f_complete = Parallel(n_jobs=N_cores, verbose=10*print_progress, prefer='processes')(
        delayed(prob_of_pop_setting_iteration_1)(i) for i in range(len(all_possible_migrations_TP)))
    data_prob_f = pd.DataFrame(prob_list_f_complete, columns=['which_ancestral', 'prob'])

    if not np.isclose(np.sum(data_prob_f.groupby(['which_ancestral']).sum().reset_index()['prob'].to_numpy()), 1):
        raise Exception('Pedigree probabilities do not sum up to one.')

    if print_progress:
        message_1 = f"Done! Computing phase-type densities conditioned to each pedigree..."
        message_2 = f"{len(migrations_at_TP)} densities to compute using the {Dioecious_model} model."
        line = "-" * max(len(message_1), len(message_2))
        for l in [line, message_1, message_2, line + "\n"]:
             print(l)
       
    density_hybrid_iteration = partial(density_hybrid_pedigree,
                                        migration_list=migrations_at_TP,
                                        which_pop=whichpop,
                                        D_model=Dioecious_model,
                                        T_PED=TP,
                                        which_L=L,
                                        bins=bingrid,
                                        mig_matrix_f=mig_matrix_f,
                                        mig_matrix_m=mig_matrix_m,
                                        rho_f=rho_f,
                                        rho_m=rho_m,
                                        X_chr=X_chr,
                                        X_chr_male=X_chr_male,
                                        density=density)
    
    densities_list = Parallel(n_jobs=N_cores, verbose=10*print_progress, prefer='processes')(
        delayed(density_hybrid_iteration)(i) for i in range(len(migrations_at_TP)))
    
    if print_progress:
        message_done = f"Done! Computing the final mixture density..."
        line = "-" * len(message_done)
        for l in [line, message_done, line + "\n"]:
             print(l)

    bins = None
    if density:
        bins = np.asarray(bingrid)
        if ~np.any(np.isin(bins, L)):  # Add L to the bins vector
            bins = np.append(bins, L)
        bins = bins[bins <= L]  # Truncate to [0, L], where the distribution is supported
        bins = np.unique(np.sort(bins))

    # Remove pedigrees that yield space states with only absorbing states
    mig_out_f = [den[2] for den in densities_list if np.any(np.isnan(den[0]))]
    mig_out_m = [den[2] for den in densities_list if np.any(np.isnan(den[1]))]
    
    if not X_chr_male: # Re-weight pedigree probabilities
        data_prob_m = data_prob_m[~data_prob_m.which_ancestral.isin(mig_out_m)]
        prob_list_m = data_prob_m.groupby(['which_ancestral']).sum().reset_index().to_numpy()
        prob_list_m[:, 1] = prob_list_m[:, 1] / np.sum(prob_list_m[:, 1])
       
    data_prob_f = data_prob_f[~data_prob_f.which_ancestral.isin(mig_out_f)]
    prob_list_f = data_prob_f.groupby(['which_ancestral']).sum().reset_index().to_numpy()
    prob_list_f[:, 1] = prob_list_f[:, 1] / np.sum(prob_list_f[:, 1])
     
    if X_chr and X_chr_male:

        density_f = np.sum(np.array([l[0] * prob_list_f[prob_list_f[:, 0] == l[2], 1]
                                        for l in densities_list if
                                        np.isin(l[2], prob_list_f[:, 0])]), axis=0)
        
        Exp_f = np.sum(np.array([l[3] * prob_list_f[prob_list_f[:, 0] == l[2], 1] 
                                    for l in densities_list if
                                    np.isin(l[2], prob_list_f[:, 0])]), axis=0)
        
        scale_f = t0_proportions_f[whichpop] * L / Exp_f if freq else 1
        final_density = density_f * scale_f

    else:
        density_f = np.sum(np.array([l[0] * prob_list_f[prob_list_f[:, 0] == l[2], 1]
                                        for l in densities_list if
                                        np.isin(l[2], prob_list_f[:, 0])]), axis=0)
        
        density_m = np.sum(np.array([l[1] * prob_list_m[prob_list_m[:, 0] == l[2], 1]
                                        for l in densities_list if
                                        np.isin(l[2], prob_list_m[:, 0])]), axis=0)
        
        Exp_f = np.sum(np.array([l[3] * prob_list_f[prob_list_f[:, 0] == l[2], 1]
                                        for l in densities_list if
                                        np.isin(l[2], prob_list_f[:, 0])]), axis=0)
        
        Exp_m = np.sum(np.array([l[4] * prob_list_m[prob_list_m[:, 0] == l[2], 1]
                                        for l in densities_list if
                                        np.isin(l[2], prob_list_m[:, 0])]), axis=0)     
        
        scale_f = t0_proportions_f[whichpop] * L / Exp_f if freq else 1
        scale_m = t0_proportions_m[whichpop] * L / Exp_m if freq else 1
        final_density = density_f * scale_f + density_m * scale_m

    if density:
        return bins, final_density
    return bingrid, np.real(np.diff(final_density))


def HP_tract_length_histogram_multi_windowed(mig_matrix_f: np.ndarray, mig_matrix_m: np.ndarray, TP: int, D_model: str,
                                            rho_f: float, rho_m: float, X_chr: bool, X_chr_male: bool, N_cores: int,
                                            population_number: int, bins: npt.ArrayLike,
                                            chrom_lengths: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Calculates the tract length histogram for the hybrid-pedigree refinement of the DC or DF model for
        multiple chromosome lengths. The histogram is computed as the sum of the histograms obtained for each chromosome length in `chrom_lengths`.
        
        Parameters
        ----------
        mig_matrix_f : npt.ArrayLike
            The female migration matrix as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
        mig_matrix_m : npt.ArrayLike
            The male migration matrix as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
        TP : int
            The number of generations in the pedigree, as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
        D_model : str
            The Dioecious model to be considered, that is one in 'DC' or 'DF'.
        rho_f : float
            The female-specific recombination rate.
        rho_m : float
            The male-specific recombination rate.
        X_chr : bool
            Whether admixture on the X chromosome is being computed.
        X_chr_male : bool
            If `X_chr` is True, whether the individual at generation 0 is a male. In that case, only maternally inherited alleles are taken into account. If not `X_chr`, set to False.
        N_cores : int
            The number of threads to be used for parallel computation.
        population_number : int
            The population of interest whose tract length distribution has to be computed. An integer from 0 to the number of populations - 1.
        bins : npt.ArrayLike
            A point grid on :math:`(0, \infty)` where the histogram has to be evaluated. The histogram will be defined on the intervals defined by bins, and the returned array will have length `len(bins)-1`.
        chrom_lengths : npt.ArrayLike
            An array of chromosome lengths to be considered. The final histogram will be the sum of the histograms obtained for each chromosome length in this array, computed using the same `bins` grid.

        Returns
        ----------
        npt.ArrayLike
            The tract length histogram on the intervals defined by `bins`, obtained as the sum of the histograms computed for each chromosome length in `chrom_lengths`. The returned array will have length `len(bins)-1`.
        """
        histogram = np.zeros(len(bins) - 1)
        for L in chrom_lengths:
            bins, new_histogram = hybrid_pedigree_distribution(mig_matrix_f=mig_matrix_f,
                                                            mig_matrix_m=mig_matrix_m,
                                                            TP=TP,
                                                            Dioecious_model=D_model,
                                                            L=L,
                                                            bingrid=bins,
                                                            whichpop=population_number,
                                                            rho_f=rho_f,
                                                            rho_m=rho_m,
                                                            X_chr=X_chr,
                                                            X_chr_male=X_chr_male,
                                                            N_cores=N_cores,
                                                            density=False,
                                                            freq=False,
                                                            print_progress=False)
            histogram += new_histogram
        return histogram

def HP_loglik(mig_matrix_f: npt.ArrayLike, mig_matrix_m: npt.ArrayLike, rho_f: float, rho_m: float, 
                TP: int, Dioecious_model: bool, X_chr: bool, X_chr_male: bool, N_cores: int,
                bins: npt.ArrayLike, Ls: npt.ArrayLike, data: list[list[int]], num_samples: int, cutoff: int = 0):
        r""" 
        Calculates the maximum-likelihood in a Poisson Random Field, when admixture is modelled using the hybrid-pedigree refinement
        of the DC or DF models. This is the hybrid-pedigree counterpart of the :func:`~tracts.phase_type.base_phase_type.PhaseTypeDistribution.loglik`
        function. The likelihood is computed as the sum of the log-likelihoods obtained for each chromosome length in `Ls`,
        where the histogram for each chromosome length is computed using :func:`~tracts.phase_type.hybrid_pedigree.HP_tract_length_histogram_multi_windowed`
        with the same `bins` grid. 

        Parameters
        ----------
        mig_matrix_f : npt.ArrayLike
            The female migration matrix as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
        mig_matrix_m : npt.ArrayLike
            The male migration matrix as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
        rho_f : float
            The female-specific recombination rate.
        rho_m : float
            The male-specific recombination rate.
        TP : int
            The number of generations in the pedigree, as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
        Dioecious_model : bool
            The Dioecious model to be considered, that is one in 'DC' or 'DF'.
        X_chr : bool
            Whether admixture on the X chromosome is being computed.
        X_chr_male : bool
            Whether admixture on the male X chromosome is being computed.
        N_cores : int
            The number of CPU cores to use for parallel computation.
        bins : npt.ArrayLike
            The bins for the tract length histogram.
        Ls : npt.ArrayLike
            The chromosome lengths.
        data : list[list[int]]
            A list of length equal to the number of populations, where the entry at index :math:`i` is a list of the observed tract counts in every bin for population :math:`i`.
        num_samples : int
            The number of haploid genomes in the sample. Used to scale the predicted tract counts, as the Phase-type distribution is defined for a single haploid genome. If `num_samples` is zero, the log-likelihood is set to zero, as the data does not contain any information.
        cutoff : int, optional
            The index of the first bin to consider in the likelihood calculation. Used to ignore very short tracts, which are very likely to be false positives in real data. 

        Returns
        -------
        float
            The log-likelihood value for the given model and data, assuming a Poisson random field.
        """
        
        if num_samples == 0:
            return 0
       
        predicted_tractlength_histogram = None
        ll = 0

        for pop in range(np.shape(mig_matrix_f)[1]):

            predicted_tractlength_histogram = HP_tract_length_histogram_multi_windowed(mig_matrix_f=mig_matrix_f,
                                                                                    mig_matrix_m=mig_matrix_m,
                                                                                    TP=TP,
                                                                                    D_model=Dioecious_model,
                                                                                    rho_f=rho_f,
                                                                                    rho_m=rho_m,
                                                                                    X_chr=X_chr,
                                                                                    X_chr_male=X_chr_male,
                                                                                    N_cores=N_cores,
                                                                                    population_number=pop,
                                                                                    bins=bins,
                                                                                    chrom_lengths=Ls)

            # Replace zeros with machine epsilon to avoid -Inf logarithms
            predicted_tractlength_histogram[predicted_tractlength_histogram <= 0] = np.finfo(float).eps

            ll += sum(-num_samples * predicted_tracts + data_tracts * np.log(num_samples * predicted_tracts) - gammaln(
            data_tracts + 1.)
                   for data_tracts, predicted_tracts in itertools.islice(
            zip(data[pop], predicted_tractlength_histogram),
            cutoff, len(predicted_tractlength_histogram))
                   )           
        
        return ll