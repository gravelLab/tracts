import matplotlib.pyplot as plt
import numpy
import pytest
import scipy
from tracts.phase_type_distribution import normalization_factor_2

import tracts
from test_data import bins, Ls

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


def test_PDT_general(migration_matrix_A):
    PTD = tracts.PhaseTypeDistribution(migration_matrix_A)
    # Verify that the equilibrium distribution is a valid probability vector
    assert min(PTD.equilibrium_distribution) >= 0
    assert numpy.isclose(numpy.linalg.norm(PTD.equilibrium_distribution, ord=1), 1)
    # Verify that the determinant of the transition matrix is 0
    assert numpy.isclose(numpy.linalg.det(PTD.full_transition_matrix), 0)
    assert numpy.allclose(numpy.dot(PTD.equilibrium_distribution, PTD.full_transition_matrix), 0)


def vefify_PDT(migration_matrix):
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
    vefify_PDT(migration_matrix)


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
        self.PTD = tracts.PhaseTypeDistribution(migration_matrix)
        self.demo = tracts.DemographicModel(migration_matrix)

    def compare_proportions(self):
        print(f'Tracts proportions at t0: {self.demo.proportions[0]}')
        print(f'PTD proportions at t0: {self.PTD.t0_proportions}')
        assert numpy.allclose(self.demo.proportions[0], self.PTD.t0_proportions, atol=0.01)

    def compare_TpopTau(self):
        for tpopTau, result in self.demo.dicTpopTau.items():
            assert self.PTD.get_TpopTau(tpopTau[0], tpopTau[1], tpopTau[2]) == result
        print('TpopTau is equal for PTD and demographic_model. All assertions passed.')

    def compare_models(self, bins, L, population_number):
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        demo_hist = self.demo.expectperbin([L], population_number, bins)
        print(f'\nTractlength histogram from PTD: \n{PTD_hist}')
        print(f'\nTractlength histogram from tracts: \n{numpy.array(demo_hist)}')
        assert numpy.allclose(PTD_hist, numpy.array(demo_hist), atol=0.01)

    def compare_models_2(self, bins, L, population_number):
        # TODO: Decide if this should be moved to examples
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        demo_hist = self.demo.expectperbin([L], population_number, bins)
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
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        demo_hist = per_bin_noscale(self.demo, bins, L, population_number)
        # demo_hist2 = per_bin_noscale(self.demo, bins, L, population_number)
        # TODO: demo_hist and demo_hist2 can be compared with numpy.allclose(demo_hist, demo_hist2)
        assert len(PTD_hist) == len(demo_hist) # == len(demo_hist2)
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
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    demo = tracts.DemographicModel(migration_matrix)
    tracts_z = demo.Z(Ls, population_number)
    phase_type_z = numpy.array([PTD.normalization_factor([L], PTD.transition_matrices[population_number],
                                                         PTD.inverse_S0_list[population_number],
                                                         PTD.alpha_list[population_number]) for L in Ls])
    phase_type_cdf_z = numpy.array([normalization_factor_2([L], PTD.transition_matrices[population_number],
                                                           PTD.inverse_S0_list[population_number],
                                                           PTD.alpha_list[population_number])[0] for L in Ls])
    print(f'Z from tracts: {tracts_z}')
    print(f'Z from PhaseType Calculation: {phase_type_z}')
    print(f'Z from PhaseType CDF: {phase_type_cdf_z}')
    assert numpy.allclose(tracts_z, phase_type_z)
    assert numpy.allclose(tracts_z, phase_type_cdf_z)
    assert numpy.allclose(phase_type_z, phase_type_cdf_z)
    # print(scipy.stats.pearsonr(PTD_hist,demo_hist))


def per_bin_noscale(demographic_model: tracts.DemographicModel, bins, L, population_number):
    PDF = lambda x: (demographic_model.inners(L, x, population_number) +
                     demographic_model.outers(L, x, population_number)) / demographic_model.Z(L, population_number)
    binval = lambda binNum: scipy.integrate.quad(PDF, bins[binNum], bins[binNum + 1])
    lsval = [binval(binNum)[0] for binNum in range(len(bins) - 1)]
    lsval.append(demographic_model.full(L, population_number) / demographic_model.Z(L, population_number))
    return numpy.array(lsval)


def per_bin_noscale_integral(demographic_model: tracts.DemographicModel, bins, L, population_number):
    # TODO: This function is never used, consider removing it
    mid = lambda binNum: (bins[binNum + 1] + bins[binNum]) / 2
    diff = lambda binNum: bins[binNum + 1] - bins[binNum]
    PDF = lambda x: (demographic_model.inners(L, x, population_number) +
                     demographic_model.outers(L, x, population_number)) / demographic_model.Z(L, population_number)
    lsval = [PDF(mid(binNum)) * diff(binNum) for binNum in range(len(bins) - 1)]
    lsval.append(demographic_model.full(L, population_number) / demographic_model.Z(L, population_number))
    return numpy.array(lsval)


def expectperbin_ratios(migration_matrix, bins, Ls, population_number):
    # TODO: This function is never used, consider removing it
    demo = tracts.DemographicModel(migration_matrix)
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    demo_hist = lambda L: demo.expectperbin([L], population_number, bins)
    PTD_hist = lambda L: PTD.tractlength_histogram_windowed(population_number, bins, L)
    return numpy.array([numpy.sum(demo_hist(L)) / numpy.sum(PTD_hist(L)) for L in Ls])
