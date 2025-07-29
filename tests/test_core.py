import matplotlib.pyplot as plt
import numpy
import pytest
import scipy
from tracts.hybrid_pedigree import hybrid_pedigree_distribution

from test_data import bins, Ls
from tracts import DemographicModel
from tracts.phase_type_distribution import PhTDioecious, PhTMonoecious, PhaseTypeDistribution

"""
Tests for component methods of tracts core
"""


@pytest.fixture
def migration_matrix_A():
    return numpy.array([[0, 0], [0, 0], [0.1, 0], [0, 0], [0, 0], [0, 0], [0.2, 0.8]])


@pytest.fixture
def migration_matrix_B():
    return numpy.array([[0, 0], [0, 0], [0, 0.5], [0.6, 0.4]])


@pytest.fixture
def migration_matrix_C():
    return numpy.array([[0, 0], [0, 0], [0, 0.5], [0.2, 0.2], [0.6, 0.4]])


@pytest.fixture
def migration_matrix_D():
    return numpy.array([[0, 0], [0, 0], [0, 0.3], [1, 0]])


@pytest.fixture
def migration_matrix_E():
    return numpy.array([[0, 0], [0, 0], [0.7, 0.3]])


# TODO: The test fails with migration_matrix_D
@pytest.mark.parametrize("data", ["migration_matrix_A", "migration_matrix_B",
                                  "migration_matrix_C", "migration_matrix_D"])
def test_PDT_Monoecious(data, request):
    migration_matrix = request.getfixturevalue(data)
    PTD = PhTMonoecious(migration_matrix)
    # Verify that the equilibrium distribution is a valid probability vector
    assert min(PTD.equilibrium_distribution) >= 0
    assert numpy.isclose(numpy.linalg.norm(PTD.equilibrium_distribution, ord=1), 1)
    # Verify that the determinant of the transition matrix is 0
    assert numpy.isclose(numpy.linalg.det(PTD.full_transition_matrix), 0)
    assert numpy.allclose(numpy.dot(PTD.equilibrium_distribution, PTD.full_transition_matrix), 0)


def verify_similar_ptd_models(ptd_first: PhaseTypeDistribution, ptd_second: PhaseTypeDistribution,
                              atol: float = 0.02):
    newbins, counts_first, E = ptd_first.tractlength_histogram_windowed(population_number=0, bins=bins, L=Ls[1],
                                                                        density=True, freq=False)
    newbins, counts_second, E = ptd_second.tractlength_histogram_windowed(population_number=0, bins=bins, L=Ls[1],
                                                                          density=True, freq=False)
    newbins, counts_first_freq, E = ptd_first.tractlength_histogram_windowed(population_number=0, bins=bins, L=Ls[1],
                                                                             density=True, freq=True)
    newbins, counts_second_freq, E = ptd_second.tractlength_histogram_windowed(population_number=0, bins=bins, L=Ls[1],
                                                                               density=True, freq=True)
    counts_first_hist, E = ptd_first.tractlength_histogram_windowed(population_number=0, bins=bins, L=Ls[1],
                                                                    density=False)
    counts_second_hist, E = ptd_second.tractlength_histogram_windowed(population_number=0, bins=bins, L=Ls[1],
                                                                      density=False)

    # Densities must be close
    assert numpy.all(numpy.isclose(counts_first, counts_second, atol=atol))
    # And the frequencies
    assert numpy.all(numpy.isclose(counts_first_freq, counts_second_freq, atol=atol))
    # And the histograms
    assert numpy.all(numpy.isclose(counts_first_hist, counts_second_hist, atol=atol))


@pytest.mark.parametrize("data", ["migration_matrix_A", "migration_matrix_B",
                                  "migration_matrix_C", "migration_matrix_D"])
def test_PDT_Dioecious(data, request):
    migration_matrix = request.getfixturevalue(data)
    PTD_Monoecious = PhTMonoecious(migration_matrix)
    PTD_Dioecious_F = PhTDioecious(migration_matrix, migration_matrix, rho_f=1,
                                   rho_m=1, sex_model='DF')
    PTD_Dioecious_C = PhTDioecious(migration_matrix, migration_matrix, rho_f=1,
                                   rho_m=1, sex_model='DC')
    verify_similar_ptd_models(ptd_first=PTD_Monoecious, ptd_second=PTD_Dioecious_F)
    verify_similar_ptd_models(ptd_first=PTD_Monoecious, ptd_second=PTD_Dioecious_C)
    verify_ptd_dioecious_matrices(PTD_Dioecious_F)
    verify_ptd_dioecious_matrices(PTD_Dioecious_C)


def verify_ptd_dioecious_matrices(pht: PhTDioecious):
    assert numpy.all(numpy.isclose(numpy.sum(pht.full_transition_matrix_f, axis=1), 0))
    assert numpy.all(numpy.isclose(numpy.sum(pht.full_transition_matrix_m, axis=1), 0))


# TODO: The test fails with migration_matrix_D
@pytest.mark.parametrize("data", ["migration_matrix_A", "migration_matrix_B",
                                  "migration_matrix_C", "migration_matrix_D"])
def test_PDT_X(data, request):
    migration_matrix = request.getfixturevalue(data)
    for Xcmale in [True, False]:
        PTD_Dioecious_F = PhTDioecious(migration_matrix, migration_matrix, rho_f=1,
                                       rho_m=1, sex_model='DF', X_chromosome=True,
                                       X_chromosome_male=Xcmale)
        PTD_Dioecious_C = PhTDioecious(migration_matrix, migration_matrix, rho_f=1,
                                       rho_m=1, sex_model='DC', X_chromosome=True,
                                       X_chromosome_male=Xcmale)
        verify_similar_ptd_models(ptd_first=PTD_Dioecious_F, ptd_second=PTD_Dioecious_C, atol=0.15)

        # Basic checks for X models
        verify_ptd_dioecious_matrices(PTD_Dioecious_F)
        verify_ptd_dioecious_matrices(PTD_Dioecious_C)


@pytest.mark.parametrize("data", ["migration_matrix_A", "migration_matrix_B",
                                  "migration_matrix_C", "migration_matrix_D"])
def test_pedigree(data, request):
    # Hybrid pedigree model with TP = 2
    migration_matrix = request.getfixturevalue(data)
    # Densities
    result_bins, counts_HP_aut = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                              mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                              bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                              X_chr=False, X_chr_male=False, N_cores=5,
                                                              density=True, freq=False)
    result_bins, counts_HP_X = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                            mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                            bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                            X_chr=True,
                                                            X_chr_male=False, N_cores=5, density=True,
                                                            freq=False)
    result_bins, counts_HP_Xm = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                             mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                             bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                             X_chr=True,
                                                             X_chr_male=True, N_cores=5, density=True,
                                                             freq=False)

    assert numpy.all(counts_HP_aut >= 0)
    assert numpy.all(counts_HP_X >= 0)
    assert numpy.all(counts_HP_Xm >= 0)

    # Frequencies
    result_bins, counts_HP_aut = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                              mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                              bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                              X_chr=False, X_chr_male=False, N_cores=5,
                                                              density=True, freq=True)
    result_bins, counts_HP_X = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                            mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                            bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                            X_chr=True,
                                                            X_chr_male=False, N_cores=5, density=True,
                                                            freq=True)
    result_bins, counts_HP_Xm = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                             mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                             bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                             X_chr=True,
                                                             X_chr_male=True, N_cores=5, density=True,
                                                             freq=True)

    assert numpy.all(counts_HP_aut >= 0)
    assert numpy.all(counts_HP_X >= 0)
    assert numpy.all(counts_HP_Xm >= 0)

    # Histograms
    result_bins, counts_HP_aut = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                              mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                              bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                              X_chr=False, X_chr_male=False, N_cores=5,
                                                              density=False, freq=False)
    result_bins, counts_HP_X = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                            mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                            bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                            X_chr=True,
                                                            X_chr_male=False, N_cores=5, density=False,
                                                            freq=False)
    result_bins, counts_HP_Xm = hybrid_pedigree_distribution(mig_matrix_f=migration_matrix,
                                                             mig_matrix_m=migration_matrix, TP=2, L=Ls[1],
                                                             bingrid=bins, whichpop=0, rr_f=1, rr_m=1,
                                                             X_chr=True,
                                                             X_chr_male=True, N_cores=5, density=False,
                                                             freq=False)

    assert numpy.all(counts_HP_aut >= 0)
    assert numpy.all(counts_HP_X >= 0)
    assert numpy.all(counts_HP_Xm >= 0)


def verify_PDT(migration_matrix):
    print(f'Migration Matrix:\n{migration_matrix}')
    models = ModelComparison(migration_matrix)
    models.compare_models(bins, Ls[0], 0)
    models.compare_proportions()
    models.compare_TpopTau()
    models.compare_models_4(bins, Ls[0], 0)
    compare_models_2(migration_matrix, [1], 1)


@pytest.mark.parametrize("data", ["migration_matrix_A", "migration_matrix_B",
                                  "migration_matrix_C", "migration_matrix_D"])
def test_verify_PDT(data, request):
    """
    Test that phase-type distributions gives the same result as demographic_model.expectperbin()
    """
    migration_matrix = request.getfixturevalue(data)
    verify_PDT(migration_matrix)


def plot_histogram_model_comparison(PTD_hist, demo_hist, L):
    # TODO: Consider moving this to examples
    print(sum(PTD_hist))
    print(sum(demo_hist))
    fig, ax = plt.subplots(1, 2)
    L = round(L, 3)
    ax[0].loglog(bins, demo_hist, label='demographic_model')
    ax[0].loglog(bins, PTD_hist, label='Phase-Type Distribution')
    ax[0].set_title(f"Log-Log Plot of Expected Number of Tracts Per Bin\n On a chromosome of length {L}")
    ax[0].set_xlabel("Bin Value")
    ax[0].set_ylabel("Expected number of tracts")
    ax[1].plot(demo_hist, PTD_hist)
    ax[1].set_title(
        f"Ratio of Phase-Type Distribution tracts to demographic_model tracts\n On a chromosome of length {L}")
    ax[1].set_xlabel("demographic_model")
    ax[1].set_ylabel("Phase-Type Distribution")
    ratio = round(sum(demo_hist) / sum(PTD_hist), 3)
    Rstat = scipy.stats.pearsonr(PTD_hist, demo_hist)
    ax[1].text(0.05, 0.95,
               f'Slope: {ratio} \nR-value: {round(Rstat.statistic, 3)} \nP-Value: {round(Rstat.pvalue, 3)}',
               transform=ax[1].transAxes, fontsize=14,
               verticalalignment='top')
    plt.show()


class ModelComparison:
    """
    A class for comparing PhaseType and demographic_model values
    """

    def __init__(self, migration_matrix):
        self.migration_matrix = migration_matrix
        self.PTD = PhTMonoecious(migration_matrix)
        self.demo = DemographicModel(migration_matrix)

    def compare_proportions(self):
        print(f'Tracts proportions at t0: {self.demo.proportions[0]}')
        print(f'PTD proportions at t0: {self.PTD.t0_proportions}')
        assert numpy.allclose(self.demo.proportions[0], self.PTD.t0_proportions, atol=0.01)

    def compare_TpopTau(self):
        for tpopTau, result in self.demo.dicTpopTau.items():
            assert self.PTD.get_TpopTau(tpopTau[0], tpopTau[1], tpopTau[2]) == result
        print('TpopTau is equal for PTD and demographic_model. All assertions passed.')

    def compare_models(self, bins, L, population_number):
        PTD_hist_tuple = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        PTD_hist = PTD_hist_tuple[0]
        demo_hist = self.demo.expectperbin([L], population_number, bins)[:-1]
        print(f'\nTractlength histogram from PTD: \n{PTD_hist}')
        print(f'\nTractlength histogram from tracts: \n{numpy.array(demo_hist)}')
        assert numpy.allclose(PTD_hist, numpy.array(demo_hist), atol=0.01)

    def compare_models_2(self, input_bins, L, population_number):
        # TODO: Decide if this should be moved to examples
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, input_bins, L)
        demo_hist = self.demo.expectperbin([L], population_number, input_bins)
        plot_histogram_model_comparison(PTD_hist, demo_hist, L)

    def compare_models_3(self, bins, L, population_number):
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        # TODO: Check that the correct function name has been guessed correctly
        # PTD_CDF = PTD.tractlength_CDF_windowed(population_number, bins, L)
        PTD_CDF = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        print(numpy.array([numpy.sum(PTD_hist[:i]) for i in range(len(PTD_hist) + 1)]))
        print(PTD_CDF - PTD_CDF[0])

    def compare_models_4(self, bins, L, population_number):
        # TODO: Move to examples
        PTD_hist_tuple = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        PTD_hist = PTD_hist_tuple[0]
        demo_hist = per_bin_noscale(self.demo, bins, L, population_number)[:-1]
        # demo_hist2 = per_bin_noscale(self.demo, bins, L, population_number)
        # TODO: demo_hist and demo_hist2 can be compared with numpy.allclose(demo_hist, demo_hist2)
        assert len(PTD_hist) == len(demo_hist)  # == len(demo_hist2)
        print(f'Chromosome has length {L}.')
        print(f'Normalized tractlength distribution from tracts PDF using bin midpoints:\n{demo_hist}')
        print(f'Normalized tractlength distribution from PhaseType CDF:\n{PTD_hist}')
        # print(f'Normalized tractlength distribution from tracts PDF using scipy integration:\n {demo_hist2}')

    def compare_models_5(self, bins, Ls, population_number):
        # TODO: Discuss if this can be automatically verified
        print(f'Chromosome lengths:\n {Ls}')
        Z = numpy.array([self.PTD.normalization_factor([L], self.PTD.transition_matrices[population_number],
                                                       self.PTD.inverse_S0_list[population_number],
                                                       self.PTD.alpha_list[population_number]) for L in Ls])

        ratios = numpy.array([numpy.sum(self.demo.expectperbin([L], population_number, bins)) / numpy.sum(
            self.PTD.tractlength_histogram_windowed(population_number, bins, L)) for L in Ls])
        print(f'\nRatios of expectperbin (tracts) to histogram (PhaseType): \n{ratios}')
        print(f'\nRatios divided by Z: \n{numpy.divide(ratios, Z)}')
        weird_factors = self.weird_factors(Ls, population_number)
        print(f'\nScaling factor from tracts: \n{weird_factors} ')
        print(f'\nScaling factor over Z: \n{numpy.divide(weird_factors, Z)}')

    def compare_models_6(self, population_number, Ls):
        # TODO: Discuss if this can be automatically verified
        # TODO: This function wasn't even in the comments
        # print(f'Alpha * S0_inverse:\n {self.alpha_s0_inv(population_number)} ')
        print(f'Chromosome lengths: \n{numpy.array(Ls)}')
        print(f'\nScaling factor from tracts: \n{self.weird_factors(Ls, population_number)} ')
        Z_to_tracts_factor = 2 * self.PTD.t0_proportions[population_number] / self.alpha_s0_inv(population_number)
        Z = numpy.array([self.PTD.normalization_factor([L], self.PTD.transition_matrices[population_number],
                                                       self.PTD.inverse_S0_list[population_number],
                                                       self.PTD.alpha_list[population_number]) for L in Ls])
        print(f'\n  Z * 2 * (proportions at t0) / (alpha*s0_inverse) * -1: \n{-Z * Z_to_tracts_factor}')
        print(f'\nTracts self.totSwitchDensity:\n {self.demo.totSwitchDens[population_number]}')
        print(f'\n2 * (proportions at t0) / (alpha*s0_inverse) * -1: \n{-Z_to_tracts_factor}')
        return

    def alpha_s0_inv(self, population_number):
        """
        The product of alpha and S0_inv for the given population
        """
        # TODO: The function is only present in the comments, discuss if it should be removed
        return numpy.dot(self.PTD.alpha_list[population_number], self.PTD.inverse_S0_list[population_number])

    def weird_factors(self, Ls, population_number):
        return numpy.array(
            [L * self.demo.totSwitchDens[population_number] + 2. * self.demo.proportions[0, population_number] for L in
             Ls])


def compare_models_2(migration_matrix, Ls, population_number):
    # TODO: This compare_models_2 is performing a different verification frpm ModelComparison.compare_models_2.
    # Come up with better names for all the functions
    PTD = PhTMonoecious(migration_matrix)
    demo = DemographicModel(migration_matrix)
    tracts_z = demo.Z(Ls, population_number)
    phase_type_z = numpy.array([PTD.normalization_factor([L], PTD.transition_matrices[population_number],
                                                         PTD.inverse_S0_list[population_number],
                                                         PTD.alpha_list[population_number]) for L in Ls])
    print(f'Z from tracts: {tracts_z}')
    print(f'Z from PhaseType Calculation: {phase_type_z}')
    assert numpy.allclose(tracts_z, phase_type_z)
    # print(scipy.stats.pearsonr(PTD_hist,demo_hist))


def per_bin_noscale(demographic_model: DemographicModel, input_bins, L, population_number):
    PDF = lambda x: (demographic_model.inners(L, x, population_number) +
                     demographic_model.outers(L, x, population_number)) / demographic_model.Z(L, population_number)
    binval = lambda binNum: scipy.integrate.quad(PDF, input_bins[binNum], input_bins[binNum + 1])
    lsval = [binval(binNum)[0] for binNum in range(len(input_bins) - 1)]
    lsval.append(demographic_model.full(L, population_number) / demographic_model.Z(L, population_number))
    return numpy.array(lsval)


def per_bin_noscale_integral(demographic_model: DemographicModel, bins, L, population_number):
    # TODO: This function is never used, consider removing it
    mid = lambda binNum: (bins[binNum + 1] + bins[binNum]) / 2
    diff = lambda binNum: bins[binNum + 1] - bins[binNum]
    PDF = lambda x: (demographic_model.inners(L, x, population_number) +
                     demographic_model.outers(L, x, population_number)) / demographic_model.Z(L, population_number)
    lsval = [PDF(mid(binNum)) * diff(binNum) for binNum in range(len(bins) - 1)]
    lsval.append(demographic_model.full(L, population_number) / demographic_model.Z(L, population_number))
    return numpy.array(lsval)


def expectperbin_ratios(migration_matrix, input_bins, Ls, population_number):
    # TODO: This function is never used, consider removing it
    demo = DemographicModel(migration_matrix)
    PTD = PhTMonoecious(migration_matrix)
    demo_hist = lambda L: demo.expectperbin([L], population_number, input_bins)
    PTD_hist = lambda L: PTD.tractlength_histogram_windowed(population_number, input_bins, L)
    return numpy.array([numpy.sum(demo_hist(L)) / numpy.sum(PTD_hist(L)) for L in Ls])
