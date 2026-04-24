import itertools
import warnings
import logging 
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse
from sklearn.preprocessing import normalize
logger = logging.getLogger(__name__)

from .base_phase_type import PhaseTypeDistribution, get_survival_factors

class PhTDioecious(PhaseTypeDistribution):
    r"""
    A subclass of :class:`PhaseTypeDistribution` providing the
    specific Phase-Type tools for the Dioecious Fine (DF)
    and Dioecious Coarse (DC) Markov approximations.

    Attributes
    ----------
    X_chr : bool
        Whether admixture is considered on the X chromosome. Set to the value given as input by the X_chromosome parameter.
    X_chr_male : bool
        If `X_chr` is True, whether the sex of the individual at generation 0 is male. Set to the value given as input by the `X_chromosome_male` parameter.
        If `X_chr` is False, this attribute is ignored.
    rho_f:
        The female-specific recombination rate :math:`\rho_f`, given by the input parameter `rho_f`.
    rho_m:
        The male-specific recombination rate :math:`\rho_m`, given by the input parameter `rho_m`.
    migration_matrix_f : npt.ArrayLike
        A transformed version of the female migration matrix given as input. For internal use only.
    migration_matrix_m : npt.ArrayLike
        A transformed version of the male migration matrix given as input. For internal use only.
    num_populations: int
        The number of populations considered in the demographic model.
    num_generations: int
        The number of generations considered in the demographic model.
    t0_proportions_f : npt.ArrayLike
        The ancestry proportion in the present population from each ancestry among all the haploid copies inherited from a female parent. For autosomes,
        computed using Eq. (F-27) in the manuscript. For the X chromosome, computing using the recursive equations (F-28) and (F-29) in the manuscript.
    t0_proportions_m : npt.ArrayLike
        The ancestry proportion in the present population from each ancestry among all the haploid copies inherited from a male parent. For autosomes,
        computed using Eq. (F-27) in the manuscript. For the X chromosome, computing using the recursive equations (F-28) and (F-29) in the manuscript.
    sex_model
        The Dioecious approximation that is being used. Taken from the input parameter `sex_model`, that is one in 'DF', 'DC'.
    full_transition_matrix_f: npt.ArrayLike
        The intensity matrix :math:`\mathbf{S}_f`  of the Dioecious (Fine or Coarse) Markov odel. Corresponds to Eq. (EQ) and (EQ) in the manuscript for DF and DC, respectively.
        This submodel corresponds to the maternally inherited allele (:math:`\xi=f`).
    full_transition_matrix_m: npt.ArrayLike
        The intensity matrix :math:`\mathbf{S}_m` of the Dioecious (Fine or Coarse) Markov model. Corresponds to Eq. (EQ) and (EQ) in the manuscript for DF and DC, respectively.
        This submodel corresponds to the paternally inherited allele (:math:`\xi=m`).
    alpha_list_f: list
        A list containing, for each ancestral population, the initial state of the Phase-type distribution for maternally inherited tracts (:math:`\xi=f`).
        Corresponds to Eq. (EQ) in the manuscript.
    alpha_list_m: list
        A list containing, for each ancestral population, the initial state of the Phase-type distribution for paternally inherited tracts (:math:`\xi=m`).
        Corresponds to Eq. (EQ) in the manuscript.
    transition_matrices_f: list
        A list containing, for each ancestral population, the submatrix of `full_transition_matrix_f` corresponding to transitions within that population.
        It is used to compute the distribution of tract lengths of maternally (:math:`\xi=f`) inherited alleles.
    transition_matrices_m: list
        A list containing, for each ancestral population, the submatrix of `full_transition_matrix_m` corresponding to transitions within that population.
        It is used to compute the distribution of tract lengths of paternally (:math:`\xi=m`) inherited alleles.
    S0_list_f: list
        A list containing the sum across columns of every transition matrix in `transition_matrices_f`.
    S0_list_m: list
        A list containing the sum across columns of every transition matrix in `transition_matrices_m`.
    inverse_S0_list_f: list
        A list containing the sum across columns of the inverse of every transition matrix in `transition_matrices_f`.
    inverse_S0_list_m: list
        A list containing the sum across columns of the inverse of every transition matrix in `transition_matrices_m`.

    Parameters
    ----------
    migration_matrix_f : npt.ArrayLike
        An array containing the female migration proportions from a discrete number of populations over the last generations.
        Each row is a time, each column is a population. Row zero corresponds to the current
        generation. The :math:`(i,j)` element of this matrix specifies the proportion of female individuals from the admixed population that
        are replaced by female individuals from population :math:`j` at generation :math:`i`. The migration rate at the last generation (`migration_matrix_f[-1,:]`) must sum up to 1.
    migration_matrix_m : npt.ArrayLike
        Counterpart of `migration_matrix_f` for male migration rates.
    rho_f : float, default 1
        The female-specific recombination rate.
    rho_m : float, default 1
        The male-specific recombination rate. For X chromosome admixture, this value is ignored and set to 0.
    X_chromosome: bool, default False
        Whether admixture is considered on the X chromosome. If False, the model considers autosomal admixture.
    X_chromosome_male: bool, default False
        If `X_chromosome` is True, whether the individual at generation 0 is a male. In that case, only maternally inherited alleles are taken
        into account. If `X_chromosome` is False, this parameter is ignored.
    sex_model: default 'DC'
        The Dioecious model to be considered. Takes the value 'DF' for Dioecious Fine and 'DC' for Dioecious Coarse.

    Notes
    -----
    The Dioecious Coarse model (`sex_model` = 'DC') should be preferred over the Dioecious Fine model (`sex_model` = 'DF')
    due to its computational efficiency. Both models produce very similar or identical phase-type densities unless
    there is a strong sex-bias in migration or recombination rates. For autosomal admixture, the Monoecious model
    should be used instead, for the same reasons, unless the sex bias is exceptionally strong.

    Non-listed parameters are for internal use only.
    """

    def __init__(self, migration_matrix_f: np.ndarray, migration_matrix_m: np.ndarray, 
                rho_f: float, rho_m: float,
                X_chromosome: bool=False, X_chromosome_male: bool=False, 
                sex_model: str='DC',
                TPED: int=0, setting_TP=None):
        """
        Initializes the PhTDioecious object by constructing the transition matrix and the initial state of the Phase-Type distribution.
        """
        super().__init__() # State dioecious approximation
        
        # ------ Initial checks and setup------
        
        # Check that migration matrices are well-specified and have the same shape.
        if np.sum(np.abs(np.asarray(np.shape(migration_matrix_f)) - np.asarray(np.shape(migration_matrix_m)))) > 0:
            raise Exception('Migration matrices must have the same shape.')

        if np.any(migration_matrix_m) < 0 or np.any(migration_matrix_f) < 0:
            raise Exception('Contributions from source populations must be non-negative.')

        if any(np.any((s > 1) & ~np.isclose(s, 1, atol = 1e-3)) for s in (np.sum(migration_matrix_m, axis=1), np.sum(migration_matrix_f, axis=1))):
            raise Exception('Migration matrices are not well-specified. Contributions must sum up to a value <= 1 at each generation.')
            
        if np.sum(np.abs(migration_matrix_m[0, :])) > 0 or np.sum(np.abs(migration_matrix_f[0, :])) > 0:
            warnings.warn(
                'Source populations cannot contribute to the admixted population at generation 0. '
                'Contributions at generation 0 will be ignored.')
            migration_matrix_m[0, :] = 0
            migration_matrix_f[0, :] = 0

        # Zap negligible contributions for numerical stability
        migration_matrix_f[migration_matrix_f < 1e-3] = 0
        migration_matrix_m[migration_matrix_m < 1e-3] = 0

        if not np.isclose(np.sum(np.abs(migration_matrix_f[-1, :])), 1, atol=1e-2) or not np.isclose(np.sum(np.abs(migration_matrix_m[-1, :])), 1, atol = 1e-2):
            print('migration_matrix_f : \n', migration_matrix_f, 'with sum ', np.sum(np.abs(migration_matrix_f[-1, :])))
            print('migration_matrix_m : \n', migration_matrix_m, 'with sum ', np.sum(np.abs(migration_matrix_m[-1, :])))
            raise Exception(
                'Contributions from source populations at the last generation in the past must sum up to 1.')

        self.migration_matrix_f_unchanged = migration_matrix_f.copy() # Defined only for testing purposes
        self.migration_matrix_m_unchanged = migration_matrix_m.copy()

        self.num_generations = migration_matrix_f.shape[0]
        self.num_populations = migration_matrix_f.shape[1]
        self.survival_factors = get_survival_factors(0.5 * (migration_matrix_f + migration_matrix_m))
        
        # Set X chromosome parameters and sex-specific recombination rates
        self.X_chr = X_chromosome
        self.X_chr_male = X_chromosome_male
        self.rho_f = rho_f
        self.rho_m = rho_m

        # Compute ancestry proportions for histograms (c.f. Appendix F.3 in the manuscript)
        if not X_chromosome:
            self.t0_proportions_f = self.t0_proportions_m = np.sum(
                (0.5 * (migration_matrix_f + migration_matrix_m)) * np.transpose(self.survival_factors)[:, np.newaxis],
                axis=0) # Maternally (t0_proportions_f) and paternally (t0_proportions_m) -inherited ancestry proportions are the same.
        else:
            # Recursive computation for the X chromosome

            ancestry_proportions_m = migration_matrix_m[self.num_generations - 1, ]
            ancestry_proportions_f = migration_matrix_f[self.num_generations - 1, ]
            for generation_number in range(self.num_generations - 2, 0, -1):
                
                ancestry_proportions_f_prev = ancestry_proportions_f.copy()
                ancestry_proportions_m_prev = ancestry_proportions_m.copy()
                ancestry_proportions_m = migration_matrix_m[generation_number, ] + (1 - migration_matrix_m[generation_number, ].sum())*ancestry_proportions_f_prev
                ancestry_proportions_f = migration_matrix_f[generation_number, ] + (1 - migration_matrix_f[generation_number, ].sum())*(ancestry_proportions_m_prev + ancestry_proportions_f_prev)/2           
            
            self.t0_proportions_f = ancestry_proportions_f
            self.t0_proportions_m = ancestry_proportions_m

        # Used for computing the hybrid-pedrigree refinements of the DF and DC models      
        if TPED > min(self.num_generations - 1, 4):
            raise Exception('The pedigree can include up to min(T,4) generations.')

        self.f_prop_at_1 = migration_matrix_f[1, :].copy() if TPED == 0 else np.zeros(self.num_populations)
        self.f_prop_at_2 = migration_matrix_f[2, :].copy() if TPED == 0 else np.zeros(self.num_populations)
        self.m_prop_at_1 = migration_matrix_m[1, :].copy() if TPED == 0 else np.zeros(self.num_populations)
        self.m_prop_at_2 = migration_matrix_m[2, :].copy() if TPED == 0 else np.zeros(self.num_populations)

        # Format migration matrices. This code ignores migrations at generation 0 and considers that time decreases with row index.
        self.migration_matrix_f = np.flip(migration_matrix_f, axis=0)[0:(np.shape(migration_matrix_f)[0] - 1), :]
        self.migration_matrix_m = np.flip(migration_matrix_m, axis=0)[0:(np.shape(migration_matrix_m)[0] - 1), :]

        if ~np.isin(sex_model, ['DF', 'DC']):
            print('sex_model must be DF or DC. Taking DC as default.')
            self.sex_model = 'DC'
        else:
            self.sex_model = sex_model

        # --------- Computation of sex-specific migration matrices and initial states ---------        

        if self.sex_model == 'DF': # Dioecious-Fine model parameters 
            self.full_transition_matrix_f, self.source_populations_f, self.transition_matrices_f, self.alpha_list_f \
                = self.PhT_parameters_DF(parent_sex=1,
                                        T_pedigree=TPED,
                                        migration_setting_at_TP=setting_TP) # Maternally-inherited sub-model parameters
            self.full_transition_matrix_m, self.source_populations_m, self.transition_matrices_m, self.alpha_list_m \
                = self.PhT_parameters_DF(parent_sex=0,
                                        T_pedigree=TPED,
                                        migration_setting_at_TP=setting_TP) # Paternally-inherited sub-model parameters
        else: # Dioecious-Coarse model parameters
            self.full_transition_matrix_f, self.source_populations_f, self.transition_matrices_f, self.alpha_list_f \
                = self.PhT_parameters_DC(parent_sex=1,
                                        T_pedigree=TPED,
                                        migration_setting_at_TP=setting_TP) # Maternally-inherited sub-model parameters
            self.full_transition_matrix_m, self.source_populations_m, self.transition_matrices_m, self.alpha_list_m \
                = self.PhT_parameters_DC(parent_sex=0, 
                                        T_pedigree=TPED,
                                        migration_setting_at_TP=setting_TP) # Paternally-inherited sub-model parameters
        
        if len(self.source_populations_f) > 1:
            if not np.all([np.all(np.linalg.eig(sub_mat_f)[0] < 0) for sub_mat_f in self.transition_matrices_f]): # Check that eigenvalues are all negative
                raise Exception('At least one submatrix for xi = 1 has a non-negative eigenvalue :', [np.linalg.eig(sub_mat_f)[0] for sub_mat_f in self.transition_matrices_f])
        if len(self.source_populations_m) > 1: # Check that eigenvalues are all negative
            if not np.all([np.all(np.linalg.eig(sub_mat_m)[0] < 0) for sub_mat_m in self.transition_matrices_m]):
                raise Exception('At least one submatrix for xi = 0 has a non-negative eigenvalue :', [np.linalg.eig(sub_mat_m)[0] for sub_mat_m in self.transition_matrices_m])
           
        # Compute row sum of the transition submatrices. This is S*(1^T), which shows up in the probability density function.
        try:
            self.S0_list_f = [-np.sum(transition_matrix, axis=1) for transition_matrix in self.transition_matrices_f]
        except Exception as _:
            self.S0_list_f = []
        try:
            self.S0_list_m = [-np.sum(transition_matrix, axis=1) for transition_matrix in self.transition_matrices_m]
        except Exception as _:
            self.S0_list_m = []

        # Compute row sum of inverse of the transition submatrix. This is equal to (S^-1)*(1^T), which shows up in CDF calculation.
        try:
            self.inverse_S0_list_f = [
                np.sum(np.linalg.inv(transition_matrix), axis=1) if len(transition_matrix) > 0 else np.array([]) for
                transition_matrix in self.transition_matrices_f]
        except Exception as _:
            self.inverse_S0_list_f = []
        try:
            self.inverse_S0_list_m = [
                np.sum(np.linalg.inv(transition_matrix), axis=1) if len(transition_matrix) > 0 else np.array([]) for
                transition_matrix in self.transition_matrices_m]
        except Exception as _:
            self.inverse_S0_list_m = []

    # ------- Density and CDF functions -------

    def PhT_density(self, x: float, population_number: int, s1: int | None=None):
        r"""
        Computes a Phase-type density at a given point :math:`x` in :math:`(0, \infty)`.
        The Phase-type parameters (initial state, transition matrix) are taken from a
        PhTDioecious object together with the specification of a population of interest.

        Parameters
        ----------
        x : float
            A point in :math:`(0, \infty)` where the density function is evaluated.
        population_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1.
        s1 :
            The sex of the individual at generation 1. If `s1 = 0` (resp. 1), only alleles
            paternally (resp. maternally) inherited alleles are considered. If set to None,
            tracts on both copies are combined.

        Returns
        -------
        float
            The density value at :math:`x`.
        """

        f_f = 0.0
        f_m = 0.0
        if s1 == 0 or s1 is None:
            transition_matrix_m = self.transition_matrices_m[population_number]
            alpha_m = self.alpha_list_m[population_number]
            d_val = np.dot(np.dot(alpha_m, scipy.linalg.expm(x * transition_matrix_m)), \
                                       self.S0_list_m[population_number])
            if not np.isreal(d_val) or d_val < -1e-3:
                raise Exception('Density value not a real positive :', d_val)
            f_m = float(np.real(d_val).item())
        if s1 == 1 or s1 is None:
            transition_matrix_f = self.transition_matrices_f[population_number]
            alpha_f = self.alpha_list_f[population_number]
            d_val = np.dot(np.dot(alpha_f, scipy.linalg.expm(x * transition_matrix_f)), \
                                       self.S0_list_f[population_number])
            if not np.isreal(d_val) or d_val < -1e-3:
                raise Exception('Density value is not a real positive :', d_val)
            f_f = float(np.real(d_val).item())
        if self.X_chr_male or s1 == 1:
            return f_f
        if s1 == 0:
            return f_m
        return 0.5 * (f_f + f_m)

    def PhT_CDF(self, x: float, population_number: int, s1: int | None=None) -> npt.ArrayLike:
        r"""
        Computes a Phase-type CDF at a given point :math:`x` in :math:`(0, \infty)`.
        The Phase-type parameters (initial state, transition matrix) are taken from a
        PhTDioecious object togther with the specification of a population of interest.

        Parameters
        ----------
        x : float
            A point in :math:`(0, \infty)` where the density function is evaluted.
        population_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1.
        s1 :
            The sex of the individual at generation 1. If `s1 = 0` (resp. 1), only alleles
            paternally (resp. maternally) inherited alleles are considered. If set to None,
            tracts on both copies are combined.

        Returns
        -------
        float
            The CDF value at :math:`x`.
        """

        F_f = 0.0
        F_m = 0.0
        if s1 == 1 or s1 is None:
            transition_matrix_f = self.transition_matrices_f[population_number]
            alpha_f = self.alpha_list_f[population_number]
            F_val = np.sum(np.dot(alpha_f, scipy.linalg.expm(x * transition_matrix_f)))
            if not np.isreal(F_val) or F_val < -1e-3:
                raise Exception('CDF value is not a real positive :', F_val)
            F_f = 1 - float(np.real(F_val).item())
        if s1 == 0 or s1 is None:
            transition_matrix_m = self.transition_matrices_m[population_number]
            alpha_m = self.alpha_list_m[population_number]
            F_val = np.sum(np.dot(alpha_m, scipy.linalg.expm(x * transition_matrix_m)))
            if not np.isreal(F_val) or F_val < -1e-3:
                raise Exception('CDF value is not a real positive :', F_val)
            F_m = 1 - float(np.real(F_val).item())
        if self.X_chr_male or s1 == 1:
            return F_f
        if s1 == 0:
            return F_m
        return 0.5 * (F_f + F_m)

    def tractlength_histogram_windowed(self, population_number: int, bins: npt.ArrayLike, L: float,
                                       exp_Sx_per_bin_f: npt.ArrayLike = None,
                                       exp_Sx_per_bin_m: npt.ArrayLike = None, density: bool = False, freq: bool = False,
                                       return_only: int | None = None, hybrid_ped: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        r"""
        Calculates the tractlength histogram or density function on a finite chromosome, 
        using the :class:`PhTDioecious` admixture model.

        Parameters
        ----------
        population_number: int  
            The index of the population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        bins: npt.ArrayLike
            A point grid where the CDF or density have to be computed.
        L: float
            The length of the finite chromosome.
        exp_Sx_per_bin_f: npt.ArrayLike, default None
            The precomputed values of :math:`e^{\mathbf{S}x}` for every :math:`x` in bins, for the maternally inherited alleles. Used internally to speed up computation.
        exp_Sx_per_bin_m: npt.ArrayLike, default None
            The precomputed values of :math:`e^{\mathbf{S}x}` for every :math:`x` in bins, for the paternally inherited alleles. Used internally to speed up computation.
        density: bool, default False
            If True, computes the Phase-type density values evaluated on the grid. Else, returns the histogram values on the grid.
        freq: bool, default False
            If density is True, whether to return density on the frequency scale.
            If True, the density values are scaled so that their integral over :math:`(0,L)` is equal to the expected number of tracts on :math:`(0,L)`.
            If False, the density values integrate to 1 over :math:`(0,L)`.
        return_only: int | None, default None
            For internal use only. Manages the combination of maternally and paternally inherited fracts. If set to 0 (resp. 1), only paternally (resp. maternally) inherited tracts are considered.
            If None, tracts from both parents are combined. If the X chromosome is considered and the individual at generation 0 is male (`X_chromosome_male = True`),
            this parameter is ignored and only maternally inherited tracts are computed. 
        hybrid_ped: bool, default False
            For internal use only. Whether the hybrid pedigree model is being used. If True, no scale corrections are performed and densities or CDFs corresponding to connected components are returned, to be combined
            in the `hybrid_pedigree` module.
                    
        Returns
        -------
        npt.ArrayLike
            If density is True, the corrected bins grid as described in Notes. Else, the bins provided as input.
        npt.ArrayLike
            If density is True, the Phase-type density evaluated on the corrected bins grid. Returned on the frequency scale if `freq = True`.
            If density is False, the histogram values on the intervals defined by bins.
        float
            The tract length expectation of the corresponding model.
        
        Notes
        -----   
        When `density` is True, the first returned argument is a transformed version of the input bins grid. This is because the density is defined on the finite chromosome :math:`[0,L]` as a mixture of a density
        with support on :math:`(0,L)` and point masses at 0 and L. The returned bins grid removes the points :math:`0` and :math:`L` if they were included in the input bins grid,
        since the density is not defined at these points. The returned density values correspond to this transformed bins grid.   

        For details on the scale factors and the transformation of the Phase-type densities into histograms, see Appendix F.3 of the manuscript.

        The `return_only` parameter is used to select only maternally or paternally inherited tracts. Besides controlling the case of allosomal admixture on male individuals, it is used to return
        distributions corresponding to connected components in the hybrid pedigree model, that need to be combined a posteriori: connected components corresponding to maternally (resp. paternally) 
        -inherited alleles are first combined into one maternally (resp. paternally) -inherited distribution. Then, the resulting pair of Phase-type mixtures is combined at the end. See `hybrid_pedigree.py` for details.
        """
        #TODO: Make return_only an enum type.
        
        if return_only == 0 and self.X_chr_male:
            raise Exception('X chromosome is not paternally inherited. Set return_only to 1 or None.')
        newbins = None
        density_per_bin_f = 0.0
        density_per_bin_m = 0.0
        ETL_m = 0.0
        ETL_f = 0.0
        normalized_CDF_f = 0.0
        normalized_CDF_m = 0.0
        xi_m = True
        xi_f = True
        
        try:
            S0_inv_m = self.inverse_S0_list_m[population_number]
        except Exception as _:
            xi_m = False
        
        try:
            S0_inv_f = self.inverse_S0_list_f[population_number]
        except Exception as _:
            xi_f = False
        
        if self.X_chr_male and xi_f:
            return_only = 1
            
        elif self.X_chr_male and not xi_f:
            raise Exception('The state space for population', population_number,' is empty. No tracts from population ', population_number, ' allowed.')
        
        elif xi_m and not xi_f:
            if return_only == None:
                return_only = 0
                print('No maternally inherited tracts for population',population_number,'. Computing only paternally inherited tracts.')
            elif return_only == 1:
                raise Exception('No maternally inherited tracts for population',population_number,'.')
    
        elif xi_f and not xi_m:
            if return_only is None:
                return_only = 1
                print('No paternally inherited tracts for population',population_number,'. Computing only maternally inherited tracts.')
            elif return_only == 0:
                raise Exception('No paternally inherited tracts for population',population_number,'.')
                
        elif not xi_f and not xi_m:
            raise Exception('The state space for population',population_number, ' is empty. No tracts from population ', population_number, ' allowed.')
        
        if return_only == 0 or return_only is None:
            S_m = self.transition_matrices_m[population_number]
            alpha_m = self.alpha_list_m[population_number]
            
            if density:
                newbins, density_per_bin_m, ETL_m = self.PhT_density_windowed(population_number=population_number,
                                                                            S=S_m, 
                                                                            alpha=alpha_m,
                                                                            S0_inv=S0_inv_m,
                                                                            bins=bins, 
                                                                            L=L,
                                                                            s1=0,
                                                                            hybrid_pedigree=hybrid_ped)
            else:
                normalized_CDF_m, ET_m, Z_m, ETL_m = self.PhT_CDF_windowed(S=S_m,
                                                                            alpha=alpha_m,
                                                                            S0_inv=S0_inv_m,
                                                                            bins=bins,
                                                                            L=L,
                                                                            exp_Sx_per_bin=exp_Sx_per_bin_m,
                                                                            s1=0,
                                                                            pop_number=population_number,
                                                                            hybrid_pedigree = hybrid_ped)
        if return_only == 1 or return_only is None:
            S_f = self.transition_matrices_f[population_number]
            alpha_f = self.alpha_list_f[population_number]
            
            if density:
                newbins, density_per_bin_f, ETL_f = self.PhT_density_windowed(population_number=population_number,
                                                                            S=S_f,
                                                                            alpha=alpha_f,
                                                                            S0_inv=S0_inv_f,
                                                                            bins=bins,
                                                                            L=L,
                                                                            s1=1,
                                                                            hybrid_pedigree = hybrid_ped)
            else:
                normalized_CDF_f, ET_f, Z_f, ETL_f = self.PhT_CDF_windowed(S=S_f,
                                                                        alpha=alpha_f, 
                                                                        S0_inv=S0_inv_f,
                                                                        bins=bins,
                                                                        L=L,
                                                                        exp_Sx_per_bin=exp_Sx_per_bin_f, s1=1,
                                                                        pop_number=population_number,
                                                                        hybrid_pedigree = hybrid_ped)

        # --------- Return Phase-type distribution as a density function, at the frequency or density scale ---------

        # --------- Phase-type distribution returned as a density function ---------

        if density:
            
            if return_only is None and not self.X_chr_male: # For autosomes or X chromosome in females, both haploids copies are combined.
                scale_m = self.t0_proportions_m[population_number] * L / ETL_m  if freq else 0.5
                scale_f = self.t0_proportions_f[population_number] * L / ETL_f  if freq else 0.5
                return newbins, scale_m * density_per_bin_f + scale_f * density_per_bin_m, 0.5 * (ETL_m + ETL_f)
            
            elif return_only == 0 and not self.X_chr_male: # If return_only == 0, only paternally inherited tracts are considered.
                scale_m = self.t0_proportions_m[population_number] * L / ETL_m if freq else 1
                return newbins, scale_m * density_per_bin_m, ETL_m
            
            else: # If return_only == 1 or X_chr_male is True (i.e. X chromosome in male individuals), only maternally inherited tracts are considered.
                scale_f = self.t0_proportions_f[population_number] * L / ETL_f if freq else 1
                return newbins, scale_f * density_per_bin_f, ETL_f
        
        # --------- Phase-type distribution returned as a histogram ---------
        
        if return_only is None and not self.X_chr_male: # For autosomes or X chromosome in females, both haploids copies are combined.
            normalized_CDF = 0.5 * (normalized_CDF_f + normalized_CDF_m)
            scale_m = self.t0_proportions_m[population_number] * L / ETL_m
            scale_f = self.t0_proportions_f[population_number] * L / ETL_f
            E = (ETL_f + ETL_m) / 2
        
        elif return_only == 0 and not self.X_chr_male: # If return_only == 0, only paternally inherited tracts are considered.
            normalized_CDF = normalized_CDF_m
            normalized_CDF_f = np.zeros(len(normalized_CDF_m))
            scale_f = 0
            scale_m = self.t0_proportions_m[population_number] * L / ETL_m
            E = ETL_m
        
        else: # If return_only == 1 or X_chr_male is True (i.e. X chromosome in male individuals), only maternally inherited tracts are considered.
            normalized_CDF = normalized_CDF_f
            normalized_CDF_m = np.zeros(len(normalized_CDF_f))
            scale_m = 0
            scale_f = self.t0_proportions_f[population_number] * L / ETL_f
            E = ETL_f
        
        # Check that CDF values are real and positive before computing the histogram. This is for numerical stability, since the difference of two close CDF values is computed in the histogram calculation.
        if not np.all(np.isreal(normalized_CDF_m)) or np.any(normalized_CDF_m < -1e-3):
            raise Exception('type-m CDF is not positive and real : ', normalized_CDF_m)
        if not np.all(np.isreal(normalized_CDF_f)) or np.any(normalized_CDF_f < -1e-3):
            raise Exception('type-f CDF is not positive and real : ', normalized_CDF_f)
        
        if not hybrid_ped: # In the Dioecious Model, maternally and paternally -inherited distributions are combined and returned now.
            return bins, np.real(np.diff(normalized_CDF_m) * scale_m + np.diff(normalized_CDF_f) * scale_f), E
        else: # In the hybrid pedigree model, connected components are combined later.
            return bins, normalized_CDF, E

    def tract_length_histogram_multi_windowed(self, population_number: int, bins: npt.ArrayLike,
                                              chrom_lengths: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculates the tract length histogram on multiple chromosomes of different lengths.

        Parameters
        ----------
        population_number: int
            The index of the population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        bins: npt.ArrayLike
            A point grid where the histogram has to be computed. The same grid is used for all chromosomes, and should be defined on the interval `(0, max(chrom_lengths))`.
        chrom_lengths: npt.ArrayLike
            A list of chromosome lengths.
        
        Returns
        -------
        npt.ArrayLike
            The histogram values on the intervals defined by bins, summed across all chromosomes.
        """

        histogram = np.zeros(len(bins) - 1)
        exp_Sx_per_bin = None
        S_f, S_m = self.transition_matrices_f[population_number], self.transition_matrices_m[population_number]
        exp_Sx_per_bin_f, exp_Sx_per_bin_m = [0] * len(bins), [0] * len(bins)
        for bin_number, bin_val in enumerate(bins):
            exp_Sx_per_bin_f[bin_number] = scipy.linalg.expm(bin_val * S_f)
            exp_Sx_per_bin_m[bin_number] = scipy.linalg.expm(bin_val * S_m)
        for L in chrom_lengths:
            bins, new_histogram, E = self.tractlength_histogram_windowed(population_number=population_number,
                                                                        bins=bins, 
                                                                        L=L,
                                                                        exp_Sx_per_bin_f=exp_Sx_per_bin_f,
                                                                        exp_Sx_per_bin_m = exp_Sx_per_bin_m)
            histogram += new_histogram
        return histogram

    def submodel_probabilities(self, population_number: int, s1: int | None): 
        r"""
        Computes the probability for a tract at generation :math:`t=0` to be drawn from the connected Markov model or the single-state isolated models
        that yield tracts of full length. See Appendix G in the manuscript for details.
        
        Parameters
        ----------
        population_number: int
            The index of the population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        s1 : int | None
            The sex of the individual at generation 1. If `s1 = 0` (resp. 1), only alleles
            paternally (resp. maternally) inherited alleles are considered. If set to None,
            tracts on both copies are combined.

        Returns
        -------
        float
            The probability for a tract at generation :math:`t=0` to be drawn from the single-state isolated model that yields tracts of full length.
        float
            The probability for a tract at generation :math:`t=0` to be drawn from the connected Markov model.
        """

        prob_mig_f_1, prob_mig_m_1 = self.f_prop_at_1[population_number], self.m_prop_at_1[population_number]
        prob_mig_f_2, prob_mig_m_2 = self.f_prop_at_2[population_number], self.m_prop_at_2[population_number]
        prob_ad_f_1, prob_ad_m_1 = 1 - np.sum(self.f_prop_at_1), 1 - np.sum(self.m_prop_at_1)
        prob_ad_f_2, prob_ad_m_2 = 1 - np.sum(self.f_prop_at_2), 1 - np.sum(self.m_prop_at_2)
        norm_f_1, norm_m_1 = prob_mig_f_1 + prob_ad_f_1, prob_mig_m_1 + prob_ad_m_1
        norm_f_2, norm_m_2 = prob_mig_f_2 + prob_ad_f_2, prob_mig_m_2 + prob_ad_m_2
        prob_mig_f_1, prob_mig_m_1 = prob_mig_f_1 / norm_f_1, prob_mig_m_1 / norm_m_1
        prob_ad_f_1, prob_ad_m_1 = prob_ad_f_1 / norm_f_1, prob_ad_m_1 / norm_m_1
        prob_mig_f_2, prob_mig_m_2 = prob_mig_f_2 / norm_f_2, prob_mig_m_2 / norm_m_2
        prob_ad_f_2, prob_ad_m_2 = prob_ad_f_2 / norm_f_2, prob_ad_m_2 / norm_m_2
        if s1 == 0:
            if self.X_chr:
                prop_isolated = prob_mig_m_1 + prob_mig_f_2 * prob_ad_m_1
                prop_connected = prob_ad_m_1 * prob_ad_f_2
            else:
                prop_isolated = prob_mig_m_1
                prop_connected = prob_ad_m_1
        else:
            prop_isolated = prob_mig_f_1
            prop_connected = prob_ad_f_1
        return prop_isolated, prop_connected

    def PhT_density_windowed(self, population_number: int, S: npt.ArrayLike, alpha: npt.ArrayLike, S0_inv: npt.ArrayLike,
                            bins: npt.ArrayLike, L: float, s1: int | None = None, exp_Sx_per_bin: npt.ArrayLike = None, hybrid_pedigree = False):
        r"""
        Computes a Phase-type density on a finite chromosome of length :math:`L` and evaluates it on a point grid.
        The Phase-type parameters (initial state, transition matrix) are taken from a :class:`PhTDioecious` object 
        (together with the specification of a population of interest) but also directly
        introduced as an input.

        Parameters
        ----------
        population_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        S : npt.ArrayLike
            The transition submatrix.
        alpha : npt.ArrayLike
            The initial state of the Phase-type distribution.
        S0_inv : npt.ArrayLike
            The sum across columns of the inverse of the transition submatrix.
        bins: npt.ArrayLike
            A point grid on :math:`(0, L)` where the density has to be evaluated.  
        L: float
            The length of the finite chromosome.
        s1 : int, optional
            The sex of the individual at generation :math:`t=1`. If `s1 = 0` (resp. 1), only alleles
            paternally (resp. maternally) inherited alleles are considered. If set to None,
            tracts on both copies are combined.
        hybrid_pedigree : bool, default False
            For internal use only. This parameter indicates whether a hybrid pedigree model is being used.
        exp_Sx_per_bin: npt.ArrayLike, default None
            The precomputed values of :math:`e^{\mathbf{S}x}` for every :math:`x` in bins. Used internally to speed up computation. 
        
        Returns
        -------
        npt.ArrayLike
            The corrected bins grid as described in Notes.
        npt.ArrayLike
            The density evaluated on bins.
        float
            The tract length expectation of the corresponding model.  

        Notes
        -------     
        The code truncates bins to the interval :math:`[0,L]` and adds the point :math:`L` if it is not included in bins.
        This is done because the density is defined on the finite chromosome :math:`[0,L]` as a mixture of a continuous density on :math:`[0,L)` and a Dirac measure at :math:`L`.
        Consequently, the function returns as a first argument the transformed grid, that can be used as x-axis to plot the density.

        **Don't run** this function directly. To get a Phase-type density on a finite chromosome,
        use :func:`~tracts.phase_type.PhTDioecious.tractlength_histogram_windowed` setting `density=True`.
        """
        
        bins, ETL, Z = PhaseTypeDistribution.initialize_density_bins(bins=bins,
                                                                    S0_inv=S0_inv,
                                                                    alpha=alpha,
                                                                    L=L)
        if not hybrid_pedigree:
            prop_isolated, prop_connected = self.submodel_probabilities(population_number=population_number,
                                                                        s1=s1)
        else:
            prop_isolated, prop_connected = 0, 1
            
        return self.populate_density_bins(bins=bins,
                                        population_number=population_number,
                                        L=L,
                                        S=S,
                                        S0_inv=S0_inv,
                                        Z=Z,
                                        s1=s1,
                                        exp_Sx_per_bin=exp_Sx_per_bin,
                                        prop_isolated=prop_isolated,
                                        prop_connected=prop_connected,
                                        ETL=ETL,
                                        alpha=alpha)

    def PhT_CDF_windowed(self, S, alpha, S0_inv, bins: npt.ArrayLike, L: float, s1: float, pop_number: int,
                         exp_Sx_per_bin: npt.ArrayLike = None, hybrid_pedigree = False) -> tuple[np.ndarray, float, float, float]:
        """
        Computes a Phase-Type CDF on a finite chromosome of length L and evaluates it on a point grid.
        The PhT parameters (initial state, transition matrix) are taken from a PhTDioecious object
        (together with the specification of a population of interest) but also directly
        introduced as an input.

        Parameters
        ----------
        S : npt.ArrayLike
            The transition submatrix.
        alpha : npt.ArrayLike
            The initial state of the Phase-Type distribution.
        S0_inv : npt.ArrayLike
            The sum across columns of the inverse of the transition submatrix.
        bins: npt.ArrayLike
            A point grid on (0, L) where the CDF has to be evaluated.  
        L: float
            The length of the finite chromosome.
        s1 : float
            The sex of the individual at generation 1. If s1 = 0 (resp. 1), only alleles
            paternally (resp. maternally) inherited alleles are considered. If set to None,
            tracts on both copies are combined.
        hybrid_pedigree : bool, default False
            For internal use only. This parameter indicates whether a hybrid pedigree model is being used.
        pop_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        exp_Sx_per_bin: npt.ArrayLike, default None
            The precomputed values of e^(S*x) for every x in bins. Used internally to speed up computation.
        
        Returns
        -------
        npt.ArrayLike
            The CDF evaluated on bins.
        float
            The tract length expectation of the corresponding model considering an infinite chromosome.
        float
            The normalization factor Z of the corresponding model.
        float
            The tract length expectation on the finite chromosome of the corresponding model.
        """

        bins, CDF_values, S0_inv, ET, ETL, Z = PhaseTypeDistribution.initialize_CDF_values(bins=bins, S0_inv=S0_inv,
                                                                                           alpha=alpha, L=L)
        
        if not hybrid_pedigree:
            prop_isolated, prop_connected = self.submodel_probabilities(population_number=pop_number,
                                                                                   s1=s1)
        else:
            prop_isolated, prop_connected = 0, 1
            
        return self.populate_CDF_values(bins=bins, CDF_values=CDF_values, S0_inv=S0_inv, Z=Z,
                                        exp_Sx_per_bin=exp_Sx_per_bin, ET=ET, ETL=ETL, alpha=alpha,
                                        L=L, prop_isolated=prop_isolated, prop_connected=prop_connected, S=S)

    # ---------- Computation of model parameters ----------

    def discrete_prob_DF(self, pulses: np.ndarray, state_left: np.ndarray, state_right: np.ndarray, T_ped: int = 0):
        r"""
        Compute the transition probabilities between a pair of states for the embedded
        Dioecious-Fine Markov model, which is a discrete TCMC. This corresponds to
        Equation (EQ) in the manuscript.

        Parameters
        ----------
        pulses : np.ndarray
            Pulse matrix of the model, with shape ``(number of pulses, 4)``.
            Each row is of the form :math:`(p, \delta, t, m_p^{\delta}(t))`.
        state_left : np.ndarray
            Current state of the embedded process; see Notes.
        state_right : np.ndarray
            Next state of the embedded process; see Notes.
        T_ped : int
            Number of generations included in the pedigree when the hybrid-pedigree
            refinement is used. Ignored otherwise.

        Notes
        -----
        States in the DF model are of the form :math:`(\delta, p, \vec{s})`,
        where :math:`\delta` is the sex of the ancestor, :math:`p` is their
        ancestral population, and :math:`\vec{s}` contains the sexes of all
        individuals that carried the haplotype from the present time up to the
        ancestor.

        In the code, states are represented as vectors of length :math:`3 + t`
        of the form :math:`(p, \delta, m_p^{\delta}(t), s_0, s_1, \ldots, s_{t-1})`,
        where :math:`t` is the generation of the ancestor. See Section (SEC) in
        the manuscript for details on the DF model.
        """

        T = np.max(pulses[:, 1])
        t_left = int(np.sum(~np.isnan(state_left[3:])) + 1)
        t_right = int(np.sum(~np.isnan(state_right[3:])) + 1)
        T_ped = int(T_ped)

        if T_ped > min(T, 4):
            raise Exception('The pedigree can include up to min(T,4) generations.')

        # Condition 1 (c1) same value for s1:s_tr; Condition 2 (c2) different value for s_tr+1
        admissible_c1 = np.cumsum(np.abs(state_left[3:] - state_right[3:]))
        s_tr1_left = np.concatenate([state_left[4:][~np.isnan(state_left[4:])], state_left[1][np.newaxis],
                                     state_left[4:][np.isnan(state_left[4:])]])
        s_tr1_right = np.concatenate([state_right[4:][~np.isnan(state_right[4:])], state_right[1][np.newaxis],
                                      state_right[4:][np.isnan(state_right[4:])]])
        admissible_c2 = 1 - np.abs(s_tr1_left - s_tr1_right)
        comp_times = np.arange(T - 1)[admissible_c1 + admissible_c2 == 0]
        comp_times = comp_times[comp_times < np.min([t_left, t_right]) - 1]

        if len(comp_times) == 0:
            return tuple([0, 0, 0])
        else:

            rec_time = int(comp_times[0])
            vec_fm = np.concatenate([np.array([True]), (
                    (state_right[~np.isnan(state_right)][4:] - state_right[~np.isnan(state_right)][3:-1]) == 1)])
            vec_fm = vec_fm + 0.5 * (1 - vec_fm)

            if not self.X_chr:

                vec_fm = 0.5 * np.ones(len(vec_fm))
                vec_fm[0] = 1
                prob_state = 0.5 * (state_right[2] ** (t_right > T_ped))

            elif self.X_chr and state_right[~np.isnan(state_right)][-1] == 0:

                prob_state = state_right[2] ** (t_right > T_ped)

            else:

                prob_state = 0.5 * (state_right[2] ** (t_right > T_ped))

            vec_fm[:(rec_time + 1)] = 1
            sex_prob = 2 * np.nanprod(vec_fm)
            surv_prob = np.prod([1 - np.sum(
                pulses[(pulses[:, 1] == s + 1) * (pulses[:, 2] == state_right[3:][s]), 3]) if s + 1 > T_ped else 1 for s
                                 in range(rec_time + 1, t_right - 1)])
            probs = prob_state * sex_prob * surv_prob
            rec_rate = self.rho_f * state_left[3:][rec_time] + self.rho_m * (1 - state_left[3:][rec_time])

            return probs, rec_time, rec_rate

    def unnormalized_prob_sex_vector(self, pulses: np.ndarray, state_left: np.ndarray, T_ped: int = 0):
        """
        Compute the unnormalized probability of the sex vector. Corresponds to Equation (EQ) in the manuscript.

        Parameters
        ----------
        pulses : np.ndarray
            The pulse matrix of the model, of shape (number of pulses, 4), as specified in :func:`~tracts.phase_type.PhTDioecious.discrete_prob_DF`.
        state_left : np.ndarray
            The current state of the embedded process.
        T_ped : int, default 0
            If the hybrid-pedigree refinement is used, the number of generations included in the pedigree. Ignored otherwise.
        """

        state_left = state_left[~np.isnan(state_left)]

        t_left = np.sum(~np.isnan(state_left[3:])) + 1
        m_time_f = [np.sum(pulses[(pulses[:, 1] == j) & (pulses[:, 2] == 1), 3]) for j in range(1, t_left)]
        m_time_m = [np.sum(pulses[(pulses[:, 1] == j) & (pulses[:, 2] == 0), 3]) for j in range(1, t_left)]

        surv_prob = np.prod(
            [1 - (state_left[3:][u] * m_time_f[u] + (1 - state_left[3:][u]) * m_time_m[u]) if u + 1 > T_ped else 1 for u
             in range(t_left - 1)])

        vec_fm = np.concatenate([np.ones([1]), (state_left[4:] - state_left[3:-1]) == 1])
        sex_prob = np.nanprod(vec_fm + 0.5 * (1 - vec_fm))

        if not self.X_chr:
            vec_fm = np.ones(np.shape(vec_fm)) * 0.5
            vec_fm[0] = 1
            sex_prob = np.nanprod(vec_fm)
            state_probs = 0.5 * (state_left[2] ** (t_left > T_ped))

        elif self.X_chr and state_left[-1] == 0:

            state_probs = state_left[2] ** (t_left > T_ped)

        else:
            state_probs = 0.5 * (state_left[2] ** (t_left > T_ped))

        unnormalized_sex_vec_prob = state_probs * surv_prob * sex_prob
        return unnormalized_sex_vec_prob

    def S_matrix(self, states, pulses, T_ped, D_model='DF'):
        """
        Compute the transition matrix of the TCMC defined by the Dioecious admixture model. Corresponds to Equation (EQ) for the Dioecious-Fine 
        model and to Equation (EQ) for the Dioecious-Coarse model.

        Parameters
        ----------
        states : npt.ArrayLike
            The state space of the process, represented as a matrix of shape (number of states, 3 + T), where T is the maximum generation of migration. 
            Each row corresponds to a state and is of the form [:math:`p`, :math:`{\delta}`, :math:`m_p^{\delta}(t)`, :math:`s_0`, :math:`s_1`, ..., :math:`s_{t-1}`], where
            :math:`p` is the ancestral population, :math:`{\delta}` is the sex of the ancestor.
        pulses: np.ndarray
            The pulse matrix of the model, of shape (number of pulses, 4), as specified in :func:`~tracts.phase_type.PhTDioecious.discrete_prob_DF`.
        T_ped: int
            If the hybrid-pedigree refinement is used, the number of generations included in the pedigree. Ignored otherwise.
        D_model: str
            The type of discrete model for which the transition matrix is computed. Must be either 'DF' for the Dioecious-Fine model or 'DC' for the Dioecious-Coarse model.
        
        Returns
        -------
        sparse.csr_matrix
            The transition matrix of the TCMC defined by the Dioecious admixture model, in compressed sparse row format. The order of the states in the matrix corresponds to the order of the states in the input `states` matrix.
        """

        if not np.isin(D_model, ['DF', 'DC']):
            raise Exception('D_model must be either DF or DC.')

        pulses_copy = pulses.copy()
        T = np.shape(self.migration_matrix_f)[0]
        coarse_states = np.zeros(len(states))
        unnorm_sex_prob = np.zeros(len(states))
        N_vectors = 2 ** T - 2

        # Recombination matrix Rho                    
        Rho_m = sparse.lil_matrix((np.shape(states)[0], N_vectors))
        Rho_f = sparse.lil_matrix((np.shape(states)[0], N_vectors))

        # Migration matrix M
        M = sparse.lil_matrix((N_vectors, np.shape(states)[0]))

        for k in range(len(states)):

            pop_state, delta_state = states[k, 0], states[k, 1]
            t_state = 1 + np.sum(~np.isnan(states[k, 3:]))
            unnorm_sex_prob[k] = self.unnormalized_prob_sex_vector(pulses=pulses,
                                                                state_left=states[k, :],
                                                                T_ped=T_ped)

            # Find or append the coarse state
            mask = (pulses_copy[:, 0] == pop_state) & (pulses_copy[:, 1] == t_state) & (
                    pulses_copy[:, 2] == delta_state)
            coarse_index = np.where(mask)[0]
            if coarse_index.size > 0:
                coarse_states[k] = coarse_index[0]
            else:
                new_row = np.array([[pop_state, t_state, delta_state, 1]])
                pulses_copy = np.vstack([pulses_copy, new_row])
                coarse_states[k] = pulses_copy.shape[0] - 1

            # Process sexes and reconstruct vectors
            sexes_k = np.append(states[k, 3:][~np.isnan(states[k, 3:])], delta_state)

            rec_vectors = [
                int('1' + ''.join(map(str, np.concatenate([sexes_k[1:end], [np.abs(1 - sexes_k[end])]]).astype(int))),
                    2) - 2
                for end in range(1, len(sexes_k))]
            rec_vectors = np.array(rec_vectors, dtype=object)

            # Separate male and female recombinations
            sex_at_tr = sexes_k[:-1]

            ind_m = rec_vectors[np.where(sex_at_tr == 0)[0]]
            ind_f = rec_vectors[np.where(sex_at_tr == 1)[0]]

            if len(ind_m) > 0:
                Rho_m[k, ind_m] = 1
            if len(ind_f) > 0:
                Rho_f[k, ind_f] = 1

            # Process migration vectors
            mig_vectors = [
                int('1' + ''.join(map(str, sexes_k[1:end + 1].astype(int))), 2) - 2
                for end in range(1, len(sexes_k))]

            rec_times = np.arange(0, len(sexes_k) - 1)

            # Calculate sex recombination probabilities
            sexes_k = sexes_k[:-1]

            diff = sexes_k[1:] - sexes_k[:-1]
            vec_fm = np.ones(len(sexes_k))
            vec_fm[1:] = (diff == 1) + 0.5 * (diff != 1)

            if not self.X_chr:
                vec_fm = np.full(len(vec_fm), 0.5)
                vec_fm[0] = 1
                prob_state = 0.5 * (states[k, 2] ** (t_state > T_ped))
            elif self.X_chr and sexes_k[-1] == 0:
                prob_state = states[k, 2] ** (t_state > T_ped)
            else:
                prob_state = 0.5 * (states[k, 2] ** (t_state > T_ped))

            # Compute migration probabilities
            mig_probs = []           
            
            surv_prob_by_generation = [1 - np.sum(pulses[(pulses[:, 1] == s + 1) & (pulses[:, 2]== sexes_k[s]), 3])
                        if s + 1 > T_ped else 1
                        for s in range(1, t_state - 1)
                    ]
            # NOTE: We could precompute the cumulative product to speed up the survival probability calculation. 
            for rtime in rec_times:
                vec_fm_rt = vec_fm.copy()
                vec_fm_rt[:rtime + 1] = 1
                sex_prob = 2 * np.nanprod(vec_fm_rt)
                surv_prob = np.prod(surv_prob_by_generation[rtime:])
                mig_probs.append(prob_state * sex_prob * surv_prob)

            M[mig_vectors, k] = np.array(mig_probs)

        Rho_m = Rho_m.tocsr()
        Rho_f = Rho_f.tocsr()
        Rho = self.rho_m * Rho_m + self.rho_f * Rho_f

        ## Keep admissible recombination vectors if X chromosome
        if self.X_chr:
            Rho = Rho[((Rho.getnnz(axis=1) > 0) | (M.getnnz(axis=0) > 0))]
            M = M[:, ((Rho.getnnz(axis=1) > 0) | (M.getnnz(axis=0) > 0))]

        if D_model == 'DF':

            S_DF = Rho.dot(M).todense()

            if np.any(np.sum(S_DF, axis=1) == 0):
                raise Exception('State space is not connected.')
            np.fill_diagonal(S_DF, -np.sum(S_DF, axis=1))

            return S_DF.astype(float)

        else:  # Build A and P matrices for DC model computation

            A = sparse.csr_matrix((np.ones(len(states)), (np.arange(len(states)), coarse_states.astype(int))),
                                  shape=tuple((len(states), len(pulses_copy))))
            P = sparse.csr_matrix((unnorm_sex_prob, (np.arange(len(states)), coarse_states.astype(int))),
                                  shape=tuple((len(states), len(pulses_copy))))
            Pt = normalize(P, norm='l1', axis=0).transpose()

            S_DC = (((Pt.dot(Rho)).dot(M)).dot(A)).todense()
            np.fill_diagonal(S_DC, 0)
            np.fill_diagonal(S_DC, -np.sum(S_DC, axis=1))

            return pulses_copy, S_DC.astype(float)

    def PhT_parameters_DF(self, parent_sex:int, computing_coarse:bool=False, pulses:np.ndarray=None, 
                        T_pedigree:int=0, migration_setting_at_TP:np.ndarray=None):
        r"""
        Computes the parameters of the Phase-type distribution under the Dioecious-Fine admixture model.

        Parameters
        ----------
        parent_sex : int
            The sex of the parent from which the tract is inherited. Must be either 0 (paternal inheritance) or 1 (maternal inheritance).
        computing_coarse : bool, default False
            Whether the parameters are computed for the coarse model, as this function is called in that case. If False, the parameters are computed for the fine model.
        pulses : np.ndarray, optional
            The pulse matrix of the model, of shape (number of pulses, 4), as specified in :func:`~tracts.phase_type.PhTDioecious.discrete_prob_DF`. If not provided, it is computed from the migration matrices.
        T_pedigree : int, default 0
            If the hybrid-pedigree refinement is used, the number of generations included in the pedigree. Ignored otherwise.
        migration_setting_at_TP : np.ndarray, optional
            If the hybrid-pedigree refinement is used, the migration setting at the time of pedigree truncation. See :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution` for details. Ignored otherwise.
        
        Returns
        -------
        npt.ArrayLike
            The transition matrix of the Phase-type distribution.
        npt.ArrayLike
            The source populations involved in the model.
        npt.ArrayLike
            The transition submatrices for each source population.
        npt.ArrayLike
            The initial state of the Phase-type distribution.       
        """
        T = np.shape(self.migration_matrix_f)[0]
        NP = np.shape(self.migration_matrix_f)[1]
        states_at_TP = None
        states_after_TP = None
        if pulses is None:
            ind_f = np.where(self.migration_matrix_f != 0)
            ind_m = np.where(self.migration_matrix_m != 0)
            pulses = np.zeros([len(ind_m[0]) + len(ind_f[0]), 4])
            pulses[:, 0] = np.concatenate([ind_m[1], ind_f[1]]) # p
            pulses[:, 1] = np.concatenate([T - ind_m[0], T - ind_f[0]]) # t
            pulses[:, 2] = np.concatenate([np.zeros(len(ind_m[0])), np.ones(len(ind_f[0]))]) # delta
            pulses[:, 3] = np.concatenate([self.migration_matrix_m[ind_m], self.migration_matrix_f[ind_f]]) #m_p^{\delta}(t)

        # Remove pulses at generation t=1 if they exist.
        # These pulses are taken into account a posteriori,
        # when phase-type densities are computed on the finite chromosome.
        pulses = pulses[pulses[:, 1] > 1, :]

        sex_comb = np.concatenate(
            [np.array([s if T - int(l) == 0 else np.concatenate([s, np.repeat(np.nan, T - int(l))])
                       for s in itertools.product([0, 1], repeat=int(l) - 1)]) for l in pulses[:, 1]])
        pop_comb = np.concatenate([np.reshape(np.repeat(pulses[i, [0, 2, 3]], 2 ** (pulses[i, 1] - 1)),
                                              [int(2 ** (pulses[i, 1] - 1)), 3], order='F')
                                   for i in range(np.shape(pulses)[0])], axis=0)

        states = np.concatenate([pop_comb, sex_comb], axis=1)

        # Keep states with s1 = parent_sex
        states = states[np.isnan(states[:, 3]) * (states[:, 1] == parent_sex) | (states[:, 3] == parent_sex), :]        
        # Every line has the form (p, delta=s_t, m_p^delta(t), s_1, ..., s_{t-1})

        if self.X_chr:

            self.rho_m = 0

            states = states[~(((states[:, 3:-1] + states[:, 4:]) == 0).any(axis=1)), :]
            last_sex = np.asarray([states[i, :][~np.isnan(states[i, :])][-1] for i in range(np.shape(states)[0])])
            states = states[~((last_sex == 0) * (states[:, 1] == 0)), :]
            states = states[np.nansum(states[:, 3:], axis=1) > 0, :]  # Remove states with s = (m)
            if parent_sex == 0:
                # Pulses at t = 2 if xi = 0 are taken into account in a different sub-model
                pulses = pulses[pulses[:, 1] > 2, :]

        if T_pedigree > 0 and migration_setting_at_TP is not None:

            state_order = np.flip(np.array([i for i in itertools.product([0, 1], repeat=T_pedigree - 1)]), axis=0)
            admixed_at_TP = state_order[migration_setting_at_TP == 0, :]
            if len(admixed_at_TP) > 0 and T_pedigree == T:
                raise Exception('No admixed individuals at TP are allowed.')
            keep_states = list()
            s_admixed = False
            if len(admixed_at_TP) > 0:
                for ad in range(np.shape(admixed_at_TP)[0]):
                    keep_states.append(
                        np.where(np.all(np.equal(states[:, 4:(3 + T_pedigree)], admixed_at_TP[ad, :]), axis=1))[0])
                keep_states = np.concatenate(keep_states)
                states_after_TP = states[keep_states, :]  # States with s >= T_pedigree
                s_admixed = True if len(states_after_TP) > 0 else False

            # Add states with s == T_pedigree
            s_TP = False
            if np.sum(migration_setting_at_TP > 0) > 0:
                
                pops_at_TP = migration_setting_at_TP[migration_setting_at_TP > 0][:, np.newaxis] - 1
                delta_at_TP = state_order[migration_setting_at_TP > 0, -1][:, np.newaxis]
                sex_vectors_at_TP = np.concatenate(
                    [parent_sex * np.ones(np.shape(delta_at_TP)), state_order[migration_setting_at_TP > 0, :-1]],
                    axis=1)
                m_at_TP = np.ones(np.shape(delta_at_TP))
                states_at_TP = np.concatenate([pops_at_TP, delta_at_TP, m_at_TP, sex_vectors_at_TP], axis=1)
                states_at_TP = np.concatenate([states_at_TP, np.ones(
                    [np.shape(states_at_TP)[0], np.shape(states)[1] - 3 - np.shape(sex_vectors_at_TP)[1]]) * np.nan],
                                              axis=1)
                if self.X_chr:
                    states_at_TP = states_at_TP[~(((states_at_TP[:, 3:-1] + states_at_TP[:, 4:]) == 0).any(axis=1)), :]
                    last_sex_TP = np.asarray([states_at_TP[i, :][~np.isnan(states_at_TP[i, :])][-1] for i in
                                              range(np.shape(states_at_TP)[0])])
                    states_at_TP = states_at_TP[~((last_sex_TP == 0) * (states_at_TP[:, 1] == 0)), :]
                
                s_TP = True if len(states_at_TP) > 0 else False

            if not s_TP and not s_admixed:
                return np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
            else:
                states = np.concatenate([states_after_TP,
                                         states_at_TP]) \
                    if s_admixed * s_TP else states_after_TP if s_admixed else states_at_TP
                if len(states) == 1:
                    return np.array([np.nan]), np.sort(np.unique(states[:, 0])), np.array([np.nan]), np.array([np.nan])
        
              
        if computing_coarse:
            return pulses, states

        else:
            S = self.S_matrix(states=states,
                            pulses=pulses,
                            T_ped=T_pedigree,
                            D_model='DF')
            
            # Populations
            source_pops = np.sort(np.unique(states[:, 0]))

            # Sub-transition matrices        
            sub_matrices = [S[states[:, 0] == tract_pop, :][:, states[:, 0] == tract_pop] for tract_pop in range(NP)]

            # Compute equilibrium distribution
            transition_matrix_eigs = np.linalg.eig(S.transpose())
            
            try:
                eq_dist = np.asarray([eigenvector for eigenvalue, eigenvector in
                                      zip(transition_matrix_eigs[0], np.transpose(transition_matrix_eigs[1])) if
                                      np.isclose(eigenvalue, 0)][0]).flatten()
            except IndexError as _:
                raise Exception(
                    'Equilibrium distribution could not be calculated. '
                    'The transition matrix does not have a 0 eigenvalue.')
                 
            # Compute initial state alpha
            alpha_list = [np.asarray(np.dot(eq_dist[states[:, 0] != tract_pop], S[states[:, 0] != tract_pop, :][:,
                                                                                states[:,
                                                                                0] == tract_pop])).flatten() if np.isin(
                tract_pop, source_pops) and len(eq_dist[states[:, 0] != tract_pop]) > 0 else np.array([]) for tract_pop in range(NP)]
                                                                                    
            alpha_list = [alpha / np.sum(alpha) if len(alpha) > 0 else alpha for alpha in alpha_list]
            
            return S, source_pops, sub_matrices, alpha_list

    def PhT_parameters_DC(self, parent_sex: int, T_pedigree:int=0, migration_setting_at_TP:np.ndarray=None):
        r"""
        Computes the parameters of the Dioecious-Coarse model, given the sex of the parent at generation :math:`t=1`, that is,
        the value of :math:`\xi`.

        Parameters
        ----------
        parent_sex: int
            The sex of the individual at generation :math:`t=1`. If `parent_sex=0` (resp. `parent_sex=1`), paternally- (resp. maternally-) inherited tracts are considered.
        T_pedigree: int, default 0
            The number of generations in the pedigree when computing the hybrid-pedigree refinement of this model. If the hybrid-pedigree refinement is not being computed,
            this parameter is ignored.
        migration_setting_at_TP: np.ndarray, default None
            A binary matrix of shape (`T_pedigree`, number of populations) describing the migration setting at generation `T_pedigree` for the hybrid-pedigree refinement of this model. 
            The entry at row `t` and column `p` is 0 if the ancestor from population `p` is admixed at generation `t`, and 1 otherwise. If the hybrid-pedigree refinement is not being computed, 
            this parameter is ignored.
        
        Returns
        -------
        S : npt.ArrayLike
            The transition matrix of the Dioecious-Coarse model.
        source_pops : npt.ArrayLike
            The populations from which tracts can be drawn in the model.
        sub_matrices : list of npt.ArrayLike
            The list of transition sub-matrices corresponding to tracts drawn from each source population.
        alpha_list : list of npt.ArrayLike
            The list of initial states corresponding to tracts drawn from each source population.
        
        Notes
        -----
        Transition probabilities for the Dioecious-Coarse model are computed by appropriately averaging 
        transition probabilities under the Dioecious-Fine model, as the DC model is built as a quotient 
        Markov chain of the DF model. 
        """

        NP = np.shape(self.migration_matrix_f)[1] # Number of populations
        results_DF = self.PhT_parameters_DF(parent_sex, True, None, T_pedigree, migration_setting_at_TP) # Compute transition probabilities under the DF model
        if len(results_DF) > 3:
            return results_DF
        else:
            pulses, states = results_DF

        pulses_copy, S = self.S_matrix(states=states,
                                       pulses=pulses,
                                       T_ped=T_pedigree,
                                       D_model='DC')

        keep = np.asarray(~np.isclose(S, 0).all(axis=1)).T[0]
        S = S[keep, :][:, keep]
        pulses_copy = pulses_copy[keep, :]

        source_pops = np.sort(np.unique(pulses_copy[:, 0])) # Populations

        sub_matrices = [S[pulses_copy[:, 0] == tract_pop, :][:, pulses_copy[:, 0] == tract_pop] for tract_pop in
                        range(NP)] # Sub-transition matrices

        transition_matrix_eigs = np.linalg.eig(S.transpose()) # Compute equilibrium distribution

        try: # Compute equilibrium distribution
            eq_dist = np.asarray([eigenvector for eigenvalue, eigenvector in
                                  zip(transition_matrix_eigs[0], np.transpose(transition_matrix_eigs[1])) if
                                  np.isclose(eigenvalue, 0)][0]).flatten()
        except IndexError as e:
            raise Exception(
                'Equilibrium distribution could not be calculated. The transition matrix does not have a 0 eigenvalue.')

        # Compute initial state
        alpha_list = [np.asarray(np.dot(eq_dist[pulses_copy[:, 0] != tract_pop], S[pulses_copy[:, 0] != tract_pop, :][:,
                                                                                 pulses_copy[:,
                                                                                 0] == tract_pop])).flatten() if np.isin(
            tract_pop, source_pops) and len(eq_dist[pulses_copy[:, 0] != tract_pop]) > 0 else np.array([]) for tract_pop in range(NP)]
        
        alpha_list = [alpha / np.sum(alpha) if len(alpha) > 0 else alpha for alpha in alpha_list]
        
        return S, source_pops, sub_matrices, alpha_list
