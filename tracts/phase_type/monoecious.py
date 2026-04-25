import warnings
import logging 
import numpy as np
import numpy.typing as npt
import scipy
from tracts.util import all_same_sign
from .base_phase_type import PhaseTypeDistribution, get_survival_factors
logger = logging.getLogger(__name__)


class PhTMonoecious(PhaseTypeDistribution):
    r"""
    A subclass of :class:`PhaseTypeDistribution` providing the
    specific Phase-Type tools for the Monoecious Markov approximation.

    Attributes
    ----------
    migration_matrix : npt.ArrayLike
        The migration matrix given as input without contributions at generations 0 and 1.
    num_populations: int
        The number of populations considered in the demographic model.
    num_generations: int
        The number of generations considered in the demographic model.
    t0_proportions : npt.ArrayLike
        The total contribution from each ancestral population.
    full_transition_matrix: npt.ArrayLike
        The intensity matrix :math:`\mathbf{S}^M` of the Monoecious Markov Model.
    equilibrium_distribution: npt.ArrayLike
        The equilibrium distribution of the Monoecious Markov Model.
    alpha_list: list
        A list containing, for each ancestral population, the initial state of the population-specific Phase-Type distribution.
    transition_matrices: list
        A list containing, for each ancestral population, the submatrix of full_transition_matrix corresponding to transitions within the population.
        It is used to compute the population-specificdistribution of tract lengths.
    S0_list: list
        A list containing the sum across columns of every transition matrix in transition_matrices.
    inverse_S0_list: list
        A list containing the sum across columns of the inverse of every transition matrix in transition_matrices.


    Parameters
    ----------
    migration_matrix : npt.ArrayLike
        An array containing the migration proportions from a discrete number of populations over the last generations.
        Each row is a time, each column is a population. Row zero corresponds to the current
        generation. The migration rate at the last generation (`migration_matrix[-1,:]`) is
        the founding generation and should sum up to 1.
    rho : float, default 1
        The recombination rate.

    Notes
    ----------
    Non-listed attributes are for internal use only.
    """

    def __init__(self, migration_matrix: npt.ArrayLike, rho: float=1):
        """
        Initializes the PhTMonoecious object by constructing the transition matrix and the initial state of the Phase-Type distribution.
        """
        super().__init__() # State monoecious approximation

        # ------ Initial checks ------
        
        self.migration_matrix = np.array(migration_matrix, copy = True)
        self.migration_matrix[self.migration_matrix < 1e-3] = 0 # Zap negligible contributions for numerical stability.
        
        # Check that contributions at the last generation in the past sum up to 1, as they correspond to the founding generation.
        if not np.isclose(np.sum(np.abs(self.migration_matrix[-1, :])), 1, atol=1e-2):
            print('migration_matrix : \n', self.migration_matrix, 'with sum ', np.sum(np.abs(self.migration_matrix[-1, :])))
            raise Exception('Contributions from source populations at the last generation in the past must sum up to 1.')

        # Check that no migration is considered at generation 0.
        if np.sum(np.abs(self.migration_matrix[0, :])) > 0:
            warnings.warn(
                'Source populations cannot contribute to the admixted population at generation 0. '
                'Contributions at generation 0 will be ignored.')
            self.migration_matrix[0, :] = 0

        # Check for migration contributions in (0,1)
        mig_per_row = np.sum(self.migration_matrix, axis=1)    
        if np.any(self.migration_matrix < 0) or np.any(mig_per_row > 1 + 1e-4):
            print("Offending migration matrix", self.migration_matrix)
            raise Exception('Contributions from source populations must be non-negative and sum up to a value in [0,1].')


        # ------ Compute transition matrix and initial state of the Phase-Type distribution ------
             
        self.num_populations = self.migration_matrix.shape[1] # Number of populations is given by the number of columns of the migration matrix.
        self.num_generations = len(self.migration_matrix) # Number of generations is given by the number of rows of the migration matrix.
        
        self.survival_factors = get_survival_factors(migration_matrix) 
        self.prop_at_1 = migration_matrix[1, :].copy()
        self.t0_proportions = np.sum(self.migration_matrix * np.transpose([self.survival_factors]), axis=0)
        self.transition_matrices = [0] * self.num_populations

        # Contributions at generation t=1 in the past (if they exist) are treated separately.
        # They will be taken into account when the phase-type density on the finite chromosome is computed.
        # Now, they have to be removed from the model.
        self.migration_matrix[1, :] = 0

        # List of initial probabilities for each state.
        self.alpha_list = [0] * self.num_populations
        self.all_states = self.migration_matrix.nonzero()
        # all_states[0] contains the migration times and all_states[1] contains the migration populations
        # In other words, (all_states[0][i], all_states[1][i]) gives the i-th (time, pop) migration.
        
        self.full_transition_matrix = self.get_transition_matrix()
        if not np.all(np.isreal(self.full_transition_matrix)):
            print(f'Transition matrix is complex.\n{self.full_transition_matrix}')
            print(f'Migration matrix:\n{self.migration_matrix}')

        # Check for non-connected states
        if np.any(np.sum(self.full_transition_matrix, axis=0) == 0):
            print(self.full_transition_matrix)
            raise Exception('State space is not connected.')

        self.full_transition_matrix -= np.diag(self.full_transition_matrix.sum(axis=1)) # In a continuous-time markov chain, diagonal entries are normalized to be equal to 0.
        self.maxLen = None
        self.equilibrium_distribution = self.get_equilibrium_distribution()
        
        # The following lines use NumPy broadcasting to compute the alpha values efficiently.
        # The operation [:, None] reshapes the population vector (e.g., [0, 1, 2]) into a column vector [[0], [1], [2]].
        # When compared with the row vector of states, this produces an S × P boolean matrix,
        # where S is the number of states and P is the number of populations.
        # Entry (i, j) is True if state i belongs to population j, and False otherwise.
        state_filters = self.all_states[1] == np.arange(self.num_populations)[:, None]

        # Multiplying equilibrium_distribution by (1 - state_filters) zeros out the focal states in each row.
        # The dot product with self.full_transition_matrix then redistributes mass across states.
        # The entries that reappear in the focal states correspond to the alpha values.
        # The list comprehension extracts these values row-wise into the final array.
        # A final normalization ensures each row sums to 1, though this may be redundant if done later.
        def _normalize_sum_to_1(array):
            return array / array.sum()

        self.alpha_list = [_normalize_sum_to_1(alpha[state_filter]) for alpha, state_filter in
                           zip(np.dot(self.equilibrium_distribution * (1 - state_filters),
                                      self.full_transition_matrix),
                               state_filters)]

        self.transition_matrices = [rho * self.full_transition_matrix[state_filter][:, state_filter] for state_filter in
                                    state_filters]

        # Row sum of the transition submatrices. This is S*(1^T), which shows up in the probability density function.
        self.S0_list = [-np.sum(transition_matrix, axis=1) for transition_matrix in self.transition_matrices]

        # Row sum of inverse of the transition submatrix. This is equal to (S^-1)*(1^T), which shows up frequently in CDF calculation.
        self.inverse_S0_list = [np.sum(np.linalg.inv(transition_matrix), axis=1) for transition_matrix in
                                self.transition_matrices]

        self.scaling_factor = [self.distribution_scaling_factor(population_number=pop_number) for pop_number in
                               range(len(self.transition_matrices))]

    def PhT_density(self, x: float, population_number: int, s1=None):
        r"""
        Computes a Phase-type density at a given point :math:`x` in :math:`(0, \infty)`.
        The Phase-type parameters (initial state, transition matrix) are taken from a :class:`PhTMonoecious`
        object together with the specification of a population of interest.

        Parameters
        ----------
        x : float
            A point in :math:`(0, \infty)` where the density function is evaluated.
        population_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1.
        s1 :
            Not used in the Monoecious model.

        Returns
        -------
        float
            The density value at :math:`x`.
        """
        transition_matrix = self.transition_matrices[population_number]
        alpha = self.alpha_list[population_number]
        den_val = np.dot(np.dot(alpha, scipy.linalg.expm(x * transition_matrix)), \
                   self.S0_list[population_number])
        if not np.isreal(den_val) or den_val < -1e-3:
            raise Exception('Density value is not a real positive : ', den_val)
        return float(np.real(den_val).item())

    def PhT_CDF(self, x: float, population_number: int, s1=None) -> npt.ArrayLike:
        r"""
        Computes a Phase-type CDF at a given point :math:`x` in :math:`(0, \infty)`.
        The Phase-type parameters (initial state, transition matrix) are taken from a :class:`PhTMonoecious`
        object togther with the specification of a population of interest.

        Parameters
        ----------
        x : float
            A point in :math:`(0, \infty)` where the density function is evaluted.
        population_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1.
        s1 :
            Not used in the Monoecious model.

        Returns
        -------
        float
            The CDF value at :math:`x`.
        """

        transition_matrix = self.transition_matrices[population_number]
        alpha = self.alpha_list[population_number]
        CDF_val = np.sum(np.dot(alpha, scipy.linalg.expm(x * transition_matrix)))
        if not np.isreal(CDF_val) or CDF_val < -1e-3:
            raise Exception('CDF is not a positive real : ', CDF_val)
        return 1 - float(np.real(CDF_val).item())

    def tractlength_histogram_windowed(self, population_number: int, bins: npt.ArrayLike, L: float,
                                       exp_Sx_per_bin: npt.ArrayLike = None, density=False, freq=False) -> tuple[npt.ArrayLike, npt.ArrayLike, float]:
        r"""
        Calculates the tractlength histogram or density function on a finite chromosome, using the Monoecious (M) admixture model.

        Parameters
        ----------
        population_number: int  
            The index of the population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        bins: npt.ArrayLike
            A point grid where the CDF or density have to be computed.
        L: float
            The length of the finite chromosome.
        exp_Sx_per_bin: npt.ArrayLike, default None
            The precomputed values of :math:`e^{Sx}` for every :math:`x` in bins. Used internally to speed up computation.
        density: bool, default False
            If `density` is True, computes the PhT density values evaluated on the grid. Else, returns the histogram values on the grid.
        freq: bool, default False
            If `density` is True, whether to return density on the frequency scale.

        Returns
        -------
        npt.ArrayLike
            If density is True, the corrected bins grid as described in Notes. Else, the bins introduced as input.
        npt.ArrayLike
            If density is True, the Phase-type density evaluated on the corrected bins grid. Returned on the frequency scale if freq = True.
            If density is False, the histogram values on the intervals defined by bins.
        float
            The tract length expectation of the corresponding model.
        
        """
        
        S = self.transition_matrices[population_number]
        alpha = self.alpha_list[population_number]
        S0_inv = self.inverse_S0_list[population_number]
        if density:
            newbins, density_per_bin, ETL = self.PhT_density_windowed(population_number=population_number,
                                                                      S=S,
                                                                      alpha=alpha,
                                                                      S0_inv=S0_inv,
                                                                      bins=bins,
                                                                      L=L)
            
            scale = 2 * self.t0_proportions[population_number] * L / ETL if freq else 1
            if not np.all(np.isreal(density_per_bin)):
                print(f'Density is complex.\n{density_per_bin}')
            return newbins, scale * density_per_bin, ETL
        
        normalized_CDF, ET, Z, ETL = self.PhT_CDF_windowed(S=S,
                                                           alpha=alpha, 
                                                           S0_inv=S0_inv,
                                                           bins=bins, 
                                                           L=L,
                                                           exp_Sx_per_bin=exp_Sx_per_bin,
                                                           s1=0xDEADBEEF,
                                                           pop_number=population_number)
        
        if not np.all(np.isreal(normalized_CDF)) or np.any(normalized_CDF < -1e-3):
            raise Exception('CDF not positive and real : ', normalized_CDF)
        scale = 2 * self.t0_proportions[population_number] * L / ETL
        return bins, np.real(np.diff(normalized_CDF) * scale), ETL

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
        S = self.transition_matrices[population_number]
        exp_Sx_per_bin = [0] * len(bins)
        for bin_number, bin_val in enumerate(bins):
            exp_Sx_per_bin[bin_number] = scipy.linalg.expm(bin_val * S)
        for L in chrom_lengths:
            bins, new_histogram, ETL = self.tractlength_histogram_windowed(population_number=population_number,
                                                                        bins=bins,
                                                                        L=L,
                                                                        exp_Sx_per_bin=exp_Sx_per_bin)
            histogram += new_histogram
        return histogram

    def PhT_density_windowed(self, population_number: int, S: npt.ArrayLike, alpha: npt.ArrayLike, 
                            S0_inv: npt.ArrayLike, bins: npt.ArrayLike, L: float, s1=None,
                            exp_Sx_per_bin: npt.ArrayLike = None):
        r"""
        Computes a Phase-type density on a finite chromosome of length :math:`L` and evaluates it on a point grid.
        The Phase-type parameters (initial state, transition matrix) are taken from a :class:`PhTMonoecious` object 
        (together with the specification of a population of interest) but also directly
        introduced as an input.

        Parameters
        ----------
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
        s1 : float, default None
            Not used in the Monoecious model.
        exp_Sx_per_bin: npt.ArrayLike, default None
            The precomputed values of :math:`e^{Sx}` for every :math:`x` in bins. Used internally to speed up computation. 
        
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
        The code truncates bins to the interval :math:`[0, L]` and adds the point :math:`L` if it is not included in bins.
        This is done because the density is defined on the finite chromosome :math:`[0, L]` as a mixture of a continuous density on :math:`[0, L)` and a Dirac measure at :math:`L`.
        Consequently, the function returns as a first argument the transformed grid, that can be used as x-axis to plot the density.

        **Don't run** this function directly. To get a Phase-type density on a finite chromosome,
        use :func:`~tracts.phase_type.PhTMonoecious.tractlength_histogram_windowed` setting `density=True`.
        """
        
        bins, ETL, Z = PhaseTypeDistribution.initialize_density_bins(bins=bins,
                                                                    S0_inv=S0_inv,
                                                                    alpha=alpha,
                                                                    L=L)
        prob_mig_1 = self.prop_at_1[population_number]
        prob_ad_1 = 1 - np.sum(self.prop_at_1)
        norm_1 = prob_mig_1 + prob_ad_1
        prob_mig_1 = prob_mig_1 / norm_1
        prob_ad_1 = prob_ad_1 / norm_1
        prop_isolated = prob_mig_1
        prop_connected = prob_ad_1
        
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

    def PhT_CDF_windowed(self, S: npt.ArrayLike, alpha: npt.ArrayLike, S0_inv: npt.ArrayLike, bins: npt.ArrayLike, L: float, pop_number: int,
                         s1: float | None = None,  exp_Sx_per_bin: npt.ArrayLike = None) -> tuple[np.ndarray, float, float, float]:
        r"""
        Computes a Phase-type CDF on a finite chromosome of length :math:`L` and evaluates it on a point grid.
        The Phase-type parameters (initial state, transition matrix) are taken from a :class:`PhTMonoecious` object
        (together with the specification of a population of interest) but also directly
        introduced as an input.

        Parameters
        ----------
        S : npt.ArrayLike
            The transition submatrix.
        alpha : npt.ArrayLike
            The initial state of the Phase-type distribution.
        S0_inv : npt.ArrayLike
            The sum across columns of the inverse of the transition submatrix.
        bins: npt.ArrayLike
            A point grid on :math:`(0, L)` where the CDF has to be evaluated.  
        L: float
            The length of the finite chromosome.
        s1 : float | None, default None
            Not used in the Monoecious model.
        pop_number : int
            The population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        exp_Sx_per_bin: npt.ArrayLike, default None
            The precomputed values of :math:`e^{Sx}` for every :math:`x` in bins. Used internally to speed up computation.
        
        Returns
        -------
        npt.ArrayLike
            The CDF evaluated on bins.
        float
            The tract length expectation of the corresponding model considering an infinite chromosome.
        float
            The normalization factor :math:`Z` of the corresponding model.
        float
            The tract length expectation on the finite chromosome of the corresponding model.
        """
     
        bins, CDF_values, S0_inv, ET, ETL, Z = PhaseTypeDistribution.initialize_CDF_values(bins=bins, 
                                                                                           S0_inv=S0_inv,
                                                                                           alpha=alpha,
                                                                                           L=L)
        prob_mig_1 = self.prop_at_1[pop_number]
        prob_ad_1 = 1 - np.sum(self.prop_at_1)
        norm_1 = prob_mig_1 + prob_ad_1
        prob_mig_1 = prob_mig_1 / norm_1
        prob_ad_1 = prob_ad_1 / norm_1
        prop_isolated = prob_mig_1
        prop_connected = prob_ad_1
        return self.populate_CDF_values(bins=bins, 
                                        CDF_values=CDF_values,
                                        S0_inv=S0_inv,
                                        Z=Z,
                                        exp_Sx_per_bin=exp_Sx_per_bin, 
                                        ET=ET, 
                                        ETL=ETL, 
                                        alpha=alpha,
                                        L=L, 
                                        prop_isolated=prop_isolated,
                                        prop_connected=prop_connected,
                                        S=S)

    def _get_time_transition_factor(self, initial_time, final_time):
        return sum([self.survival_factors[final_time] / self.survival_factors[T + 1] for T in
                    range(1, min(initial_time, final_time))])

    def get_transition_matrix(self):
        r"""
        Computes the transition matrix of the Monoecious Phase-type model.

        Returns
        -------
        npt.ArrayLike
            The transition matrix of the Monoecious Phase-type model. Each entry :math:`(i,j)` corresponds to the transition rate from state :math:`i` to state :math:`j`. 
        """    
        return np.array([[self.migration_matrix[dest_time, dest_pop]*self._get_time_transition_factor(initial_time=initial_time,
                                                                                                    final_time=dest_time) for
                          dest_time, dest_pop in zip(self.all_states[0], self.all_states[1])] for initial_time in
                         self.all_states[0]])

    def get_equilibrium_distribution(self):
        """
        Computes the equilibrium distribution of the Monoecious Phase-type model.

        Returns
        -------
        npt.ArrayLike
            The equilibrium distribution of the Monoecious Phase-type model.
        """
        transposed_transition_matrix = self.full_transition_matrix.transpose()
        transition_matrix_eigs = np.linalg.eig(transposed_transition_matrix)

        try:
            result_vector = [eigenvector for eigenvalue, eigenvector in
                             zip(transition_matrix_eigs[0], np.transpose(transition_matrix_eigs[1])) if
                             np.isclose(eigenvalue, 0)][0]
            result_vector = result_vector / np.linalg.norm(result_vector, ord=1)

            assert all_same_sign(result_vector) # Verify that all the entries have the same sign.
            result_vector *= np.sign(result_vector[0])
            return result_vector
        except IndexError as _:
            raise Exception('Equilibrium distribution could not be calculated. The transition matrix does not have a 0 eigenvalue.')

    def distribution_scaling_factor(self, population_number: int):
        """
        Computes the scaling factor to transform the CDF values into counts. 

        Parameters        
        ----------
        population_number: int
            The index of the population of interest whose tract length distribution has to be computed.
            An integer from 0 to the number of populations - 1, corresponding to the column of the migration matrix.
        
        Returns
        -------
        float
            The scaling factor to transform the CDF values into counts.
        """
        return -2 * self.t0_proportions[population_number] / np.dot(self.alpha_list[population_number],
                                                                    self.inverse_S0_list[population_number])

    def _get_TpopTau(self, t, pop, Tau):
        """
        Confirmation method for calculating equivalent of `TpopTau` from `demographic_model`. Used only for testing
        purposed and not documented.
        """
        return self.survival_factors[t] / self.survival_factors[Tau + 1] * self.migration_matrix[t, pop]
